from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from flux_untwist.spectrum_h3 import (
    VISUAL_PATCH_KIND,
    VISUAL_PATCH_PROFILES_KEY,
    VISUAL_PATCH_RUNTIME_KEY,
    VISUAL_PATCH_SCHEMA_VERSION,
)
from nodes import MiniMaxH3UntwistRoPE


class FakePatcher:
    def __init__(self, diffusion_model, model_options=None):
        self.model = SimpleNamespace(diffusion_model=diffusion_model)
        self.model_options = model_options or {"transformer_options": {}}

    def clone(self):
        return FakePatcher(self.model.diffusion_model, dict(self.model_options))

    def set_model_unet_function_wrapper(self, wrapper):
        self.model_options["model_function_wrapper"] = wrapper


class FakeH3:
    def __init__(self):
        self.blocks = [SimpleNamespace(attn=SimpleNamespace(head_dim=128)) for _ in range(50)]
        self.rope = SimpleNamespace(inv_freq=torch.ones(16))
        self.rope_freqs = lambda *args, **kwargs: None
        self.audio_patch_proj = object()
        self.video_patch_proj = object()
        self.final_layer = object()


def _patch(
    model,
    *,
    high_scale_start=0.95,
    high_scale_end=1.0,
    low_scale_start=1.0,
    low_scale_end=1.05,
    beta=2.0,
    start_percent=0.0,
    end_percent=0.90,
    reference_scope="image_and_video",
    scale_temporal_axis=False,
):
    return MiniMaxH3UntwistRoPE().patch(
        model,
        high_scale_start=high_scale_start,
        high_scale_end=high_scale_end,
        low_scale_start=low_scale_start,
        low_scale_end=low_scale_end,
        beta=beta,
        start_percent=start_percent,
        end_percent=end_percent,
        verbose=False,
        reference_scope=reference_scope,
        scale_temporal_axis=scale_temporal_axis,
    )[0]


def _single_image_payload():
    return {
        "layout": SimpleNamespace(
            segments=[
                (0, 10, "text"),
                (10, 18, "ref_img"),
                (18, 28, "audio"),
                (28, 80, "video"),
            ]
        ),
        "refs": [{"kind": "image"}],
    }


def test_h3_node_ui_defaults_match_tested_runtime_defaults():
    inputs = MiniMaxH3UntwistRoPE.INPUT_TYPES()
    required = inputs["required"]
    optional = inputs["optional"]

    assert required["high_scale_start"][1]["default"] == 0.95
    assert required["high_scale_end"][1]["default"] == 1.00
    assert required["low_scale_start"][1]["default"] == 1.00
    assert required["low_scale_end"][1]["default"] == 1.05
    assert required["beta"][1]["default"] == 2.0
    assert required["start_percent"][1]["default"] == 0.0
    assert required["end_percent"][1]["default"] == 0.90
    assert optional["reference_scope"][1]["default"] == "image_and_video"
    assert optional["scale_temporal_axis"][1]["default"] is False


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda h3: setattr(h3, "blocks", []), "at least one transformer block"),
        (lambda h3: setattr(h3.blocks[0], "attn", SimpleNamespace()), "attention head dimension"),
        (lambda h3: setattr(h3.blocks[0].attn, "head_dim", 64), "rotated_dim=96 exceeds head_dim=64"),
    ],
)
def test_h3_node_rejects_invalid_attention_geometry(mutate, message):
    h3 = FakeH3()
    mutate(h3)
    with pytest.raises(RuntimeError, match=message):
        _patch(FakePatcher(h3))


def test_h3_node_pairs_native_refs_and_protects_continuum_context():
    patched = _patch(FakePatcher(FakeH3()))
    wrapper = patched.model_options["model_function_wrapper"]
    layout = SimpleNamespace(
        segments=[
            (0, 10, "text"),
            (10, 18, "ref_img"),
            (18, 22, "ref_audio"),
            (22, 30, "ref_img"),
            (30, 40, "audio"),
            (40, 100, "video"),
        ]
    )
    payload = {
        "layout": layout,
        "refs": [
            {"kind": "image"},
            {
                "kind": "video_audio",
                "_h3_continuum": {
                    "api": 1,
                    "role": "video_context",
                    "preserve_rope": True,
                },
            },
        ],
    }
    seen = {}

    def apply_model(input_x, timestep, **c):
        seen.update(c)
        return "ok"

    result = wrapper(
        apply_model,
        {
            "input": torch.zeros(1),
            "timestep": torch.tensor([0.5]),
            "c": {"minimax_payload": payload, "transformer_options": {}},
            "cond_or_uncond": [0],
        },
    )

    assert result == "ok"
    cfg = seen["transformer_options"]["minimax_h3_untwist_rope"]
    assert cfg["enabled"] is True
    assert cfg["reference_ranges"] == [[10, 18]]
    assert cfg["reference_scope"] == "image_and_video"
    assert cfg["rope_axis_count"] == 3
    assert cfg["rope_freqs_per_axis"] == 16
    assert cfg["scale_temporal_axis"] is False
    assert "optimized_attention_override" in seen["transformer_options"]


def test_h3_node_composes_call_time_attention_override():
    patch_time_calls = []
    call_time_calls = []

    def patch_time_override(original, q, k, v, heads, *args, **kwargs):
        patch_time_calls.append(True)
        return original(q, k, v, heads, *args, **kwargs)

    def call_time_override(original, q, k, v, heads, *args, **kwargs):
        call_time_calls.append(True)
        return original(q, k, v, heads, *args, **kwargs)

    patched = _patch(
        FakePatcher(
            FakeH3(),
            {"transformer_options": {"optimized_attention_override": patch_time_override}},
        )
    )
    wrapper = patched.model_options["model_function_wrapper"]
    seen = {}

    wrapper(
        lambda input_x, timestep, **c: seen.update(c),
        {
            "input": torch.zeros(1),
            "timestep": torch.tensor([0.5]),
            "c": {
                "minimax_payload": _single_image_payload(),
                "transformer_options": {"optimized_attention_override": call_time_override},
            },
        },
    )

    to = seen["transformer_options"]
    override = to["optimized_attention_override"]
    q = torch.ones((1, 1, 80, 128), dtype=torch.float32)
    k = torch.ones_like(q)
    v = torch.ones_like(q)

    override(
        lambda q_in, k_in, v_in, heads, *args, **kwargs: k_in,
        q,
        k,
        v,
        1,
        transformer_options=to,
    )

    assert call_time_calls == [True]
    assert patch_time_calls == []


def test_h3_node_default_selects_pure_video_but_excludes_video_audio():
    patched = _patch(FakePatcher(FakeH3()))
    wrapper = patched.model_options["model_function_wrapper"]
    layout = SimpleNamespace(
        segments=[
            (0, 10, "text"),
            (10, 18, "ref_img"),
            (18, 26, "ref_img"),
            (26, 34, "ref_img"),
            (34, 40, "audio"),
            (40, 80, "video"),
        ]
    )
    payload = {
        "layout": layout,
        "refs": [
            {"kind": "image"},
            {"kind": "video"},
            {"kind": "video_audio"},
        ],
    }
    seen = {}

    wrapper(
        lambda input_x, timestep, **c: seen.update(c),
        {
            "input": torch.zeros(1),
            "timestep": torch.tensor([0.5]),
            "c": {"minimax_payload": payload},
        },
    )

    cfg = seen["transformer_options"]["minimax_h3_untwist_rope"]
    assert cfg["reference_ranges"] == [[10, 18], [18, 26]]


def test_h3_node_image_only_scope_remains_available():
    patched = _patch(FakePatcher(FakeH3()), reference_scope="image_only")
    wrapper = patched.model_options["model_function_wrapper"]
    layout = SimpleNamespace(
        segments=[
            (0, 10, "text"),
            (10, 18, "ref_img"),
            (18, 26, "ref_img"),
            (26, 40, "audio"),
            (40, 80, "video"),
        ]
    )
    payload = {"layout": layout, "refs": [{"kind": "image"}, {"kind": "video"}]}
    seen = {}

    wrapper(
        lambda input_x, timestep, **c: seen.update(c),
        {
            "input": torch.zeros(1),
            "timestep": torch.tensor([0.5]),
            "c": {"minimax_payload": payload},
        },
    )

    cfg = seen["transformer_options"]["minimax_h3_untwist_rope"]
    assert cfg["reference_ranges"] == [[10, 18]]


def test_h3_progress_window_uses_comfy_sample_sigmas_and_deactivates_final_calls():
    patched = _patch(FakePatcher(FakeH3()), end_percent=0.90)
    wrapper = patched.model_options["model_function_wrapper"]
    denoiser_sigmas = torch.linspace(1.0, 0.10, 19)
    sample_sigmas = torch.cat((denoiser_sigmas, torch.zeros(1)))
    calls = []

    def apply_model(input_x, timestep, **c):
        calls.append(c["transformer_options"])
        return "ok"

    for index in (16, 17, 18):
        result = wrapper(
            apply_model,
            {
                "input": torch.zeros(1),
                "timestep": denoiser_sigmas[index],
                "c": {
                    "minimax_payload": _single_image_payload(),
                    "transformer_options": {"sample_sigmas": sample_sigmas},
                },
            },
        )
        assert result == "ok"

    cfg16 = calls[0]["minimax_h3_untwist_rope"]
    cfg17 = calls[1]["minimax_h3_untwist_rope"]
    cfg18 = calls[2]["minimax_h3_untwist_rope"]
    assert cfg16["progress"] == pytest.approx(16 / 18)
    assert cfg16["enabled"] is True
    assert "optimized_attention_override" in calls[0]
    assert cfg17["progress"] == pytest.approx(17 / 18)
    assert cfg17["enabled"] is False
    assert "optimized_attention_override" not in calls[1]
    assert cfg18["progress"] == 1.0
    assert cfg18["enabled"] is False
    assert "optimized_attention_override" not in calls[2]


def test_h3_node_fails_closed_when_layout_ref_mapping_is_ambiguous():
    patched = _patch(FakePatcher(FakeH3()))
    wrapper = patched.model_options["model_function_wrapper"]
    layout = SimpleNamespace(
        segments=[
            (0, 10, "text"),
            (10, 18, "ref_img"),
            (18, 26, "ref_img"),
            (26, 40, "audio"),
            (40, 80, "video"),
        ]
    )
    seen = {}

    wrapper(
        lambda input_x, timestep, **c: seen.update(c),
        {
            "input": torch.zeros(1),
            "timestep": torch.tensor([0.5]),
            "c": {"minimax_payload": {"layout": layout, "refs": [{"kind": "image"}]}},
        },
    )

    to = seen["transformer_options"]
    assert to["minimax_h3_untwist_rope"]["enabled"] is False
    assert "optimized_attention_override" not in to


def test_h3_node_is_noop_without_native_visual_references():
    patched = _patch(FakePatcher(FakeH3()))
    wrapper = patched.model_options["model_function_wrapper"]
    layout = SimpleNamespace(segments=[(0, 10, "text"), (10, 20, "audio"), (20, 80, "video")])
    seen = {}

    def apply_model(input_x, timestep, **c):
        seen.update(c)
        return None

    wrapper(
        apply_model,
        {
            "input": torch.zeros(1),
            "timestep": torch.tensor([0.5]),
            "c": {"minimax_payload": {"layout": layout, "refs": []}},
            "custom_field": "preserved",
        },
    )
    to = seen["transformer_options"]
    assert to["minimax_h3_untwist_rope"]["enabled"] is False
    assert "optimized_attention_override" not in to


def test_h3_spectrum_profile_contains_window_scope_and_strength_metadata():
    patched = _patch(FakePatcher(FakeH3()))
    profiles = patched.model_options[VISUAL_PATCH_PROFILES_KEY]
    assert len(profiles) == 1
    profile = profiles[0]
    assert profile["schema_version"] == VISUAL_PATCH_SCHEMA_VERSION == 2
    assert profile["kind"] == VISUAL_PATCH_KIND
    assert profile["progress_start"] == 0.0
    assert profile["progress_end"] == 0.90
    assert profile["hard_start"] is False
    assert profile["hard_end"] is True
    assert profile["scope"] == "image_and_video"
    assert profile["high_scale_start"] == 0.95
    assert profile["high_scale_end"] == 1.0
    assert profile["low_scale_start"] == 1.0
    assert profile["low_scale_end"] == 1.05
    assert profile["beta"] == 2.0
    assert profile["strength"] == pytest.approx(0.05)
    assert profile["terminal_pece_exact_corrector_safe"] is True

    seen = {}
    patched.model_options["model_function_wrapper"](
        lambda input_x, timestep, **c: seen.update(c),
        {
            "input": torch.zeros(1),
            "timestep": torch.tensor([0.5]),
            "c": {"minimax_payload": _single_image_payload(), "transformer_options": {}},
        },
    )
    runtime = seen["transformer_options"][VISUAL_PATCH_RUNTIME_KEY]
    assert len(runtime) == 1
    assert runtime[0]["instance_id"] == profile["instance_id"]
    assert runtime[0]["schema_version"] == profile["schema_version"]
    assert runtime[0]["active"] is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"start_percent": 0.10},
        {"end_percent": 0.89},
        {"end_percent": 1.0},
        {"reference_scope": "all_visual_including_continuum"},
        {"scale_temporal_axis": True},
        {"high_scale_start": 0.90},
    ],
)
def test_h3_spectrum_terminal_pece_capability_is_narrow(overrides):
    profile = _patch(FakePatcher(FakeH3()), **overrides).model_options[
        VISUAL_PATCH_PROFILES_KEY
    ][0]
    assert profile["terminal_pece_exact_corrector_safe"] is False


def test_h3_patch_clones_model_options_without_mutating_source_state():
    original_options = {
        "transformer_options": {"sentinel": {"value": 7}},
        "foreign_option": {"nested": [1, 2, 3]},
    }
    snapshot = deepcopy(original_options)
    source = FakePatcher(FakeH3(), original_options)
    patched = _patch(source)

    assert source.model_options == snapshot
    assert VISUAL_PATCH_PROFILES_KEY not in source.model_options
    assert VISUAL_PATCH_PROFILES_KEY in patched.model_options
    assert patched.model_options is not source.model_options
    assert patched.model_options["transformer_options"] is not source.model_options["transformer_options"]


def test_h3_all_ones_patch_is_exact_model_wrapper_noop_and_emits_no_profile():
    old_wrapper = object()
    patcher = FakePatcher(
        FakeH3(),
        {
            "transformer_options": {},
            "model_function_wrapper": old_wrapper,
        },
    )
    patched = _patch(
        patcher,
        high_scale_start=1.0,
        high_scale_end=1.0,
        low_scale_start=1.0,
        low_scale_end=1.0,
    )
    assert patched.model_options["model_function_wrapper"] is old_wrapper
    assert "optimized_attention_override" not in patched.model_options["transformer_options"]
    assert VISUAL_PATCH_PROFILES_KEY not in patched.model_options


def test_h3_node_preserves_existing_model_wrapper_argument_shape():
    received = {}

    def old_wrapper(apply_model, args):
        received.update(args)
        return apply_model(args["input"], args["timestep"], **args["c"])

    patcher = FakePatcher(
        FakeH3(),
        {
            "transformer_options": {},
            "model_function_wrapper": old_wrapper,
        },
    )
    patched = _patch(patcher)
    wrapper = patched.model_options["model_function_wrapper"]

    wrapper(
        lambda input_x, timestep, **c: None,
        {
            "input": torch.zeros(1),
            "timestep": torch.tensor([0.5]),
            "c": {"minimax_payload": {"layout": SimpleNamespace(segments=[]), "refs": []}},
            "cond_or_uncond": [0],
            "custom_field": "keep-me",
        },
    )

    assert received["custom_field"] == "keep-me"
