from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from flux_untwist.keyless_h3 import (
    KEYLESS_CONTRACT_KEY,
    KEYLESS_ROUTING_PREPROCESSORS_KEY,
    KeylessH3CompatibilityError,
)
from nodes import MiniMaxH3UntwistRoPE


class Contract:
    api = 1
    architecture = "h3_keyless_core50_v1"
    core_blocks = 50
    token_refiner = "native_qkv"
    token_refiner_blocks = 2
    heads = 56
    head_dim = 128
    inner_dim = 7168
    hidden_size = 5376
    routing_source = "value"
    retrieval_source = "raw_projected_value"
    routing_norm = "rmsnorm"
    routing_norm_epsilon = 1e-5
    rope_policy = "h3_split_half_96_v1"
    qv_order = "q_effective;v"
    projection_attr = "qv_proj"
    checkpoint_format_version = 1
    provenance_identity = "node-test"

    def identity(self):
        return (
            self.api,
            self.architecture,
            self.checkpoint_format_version,
            self.qv_order,
            self.heads,
            self.head_dim,
            self.inner_dim,
            self.routing_source,
            self.retrieval_source,
            self.rope_policy,
            self.provenance_identity,
        )


class KeylessAttention:
    def __init__(self):
        self.head_dim = 128
        self.qv_proj = SimpleNamespace(weight=SimpleNamespace(shape=(14336, 5376)))
        self.q_norm = object()
        self.route_norm = object()


class FakeKeylessH3:
    def __init__(self):
        self.blocks = [SimpleNamespace(attn=KeylessAttention()) for _ in range(50)]
        self.token_refiner = SimpleNamespace(blocks=[object(), object()])
        self.rope = SimpleNamespace(inv_freq=torch.ones(16))
        self.rope_freqs = lambda *args, **kwargs: None
        self.audio_patch_proj = object()
        self.video_patch_proj = object()
        self.final_layer = object()
        setattr(self, KEYLESS_CONTRACT_KEY, Contract())


class FakePatcher:
    def __init__(self, diffusion_model, model_options=None):
        self.model = SimpleNamespace(diffusion_model=diffusion_model)
        self.model_options = model_options or {"transformer_options": {}}

    def clone(self):
        options = dict(self.model_options)
        options["transformer_options"] = dict(
            self.model_options.get("transformer_options", {})
        )
        return FakePatcher(self.model.diffusion_model, options)

    def set_model_unet_function_wrapper(self, wrapper):
        self.model_options["model_function_wrapper"] = wrapper


def _patch(model):
    return MiniMaxH3UntwistRoPE().patch(
        model,
        high_scale_start=0.5,
        high_scale_end=0.5,
        low_scale_start=1.5,
        low_scale_end=1.5,
        beta=1.0,
        start_percent=0.0,
        end_percent=1.0,
        verbose=False,
        reference_scope="image_and_video",
        scale_temporal_axis=False,
    )[0]


def _payload():
    return {
        "layout": SimpleNamespace(
            seq_len=12,
            segments=[
                (0, 2, "text"),
                (2, 4, "ref_img"),
                (4, 6, "audio"),
                (6, 12, "video"),
            ],
        ),
        "refs": [{"kind": "image"}],
    }


def _invoke(patched, *, transformer_options=None):
    wrapper = patched.model_options["model_function_wrapper"]
    seen = {}
    result = wrapper(
        lambda input_x, timestep, **c: seen.update(c) or "ok",
        {
            "input": torch.zeros(1),
            "timestep": torch.tensor([0.5]),
            "c": {
                "minimax_payload": _payload(),
                "transformer_options": transformer_options or {},
            },
        },
    )
    assert result == "ok"
    return seen["transformer_options"]


def test_keyless_node_appends_routing_preprocessor_without_replacing_attention_owner() -> None:
    patch_time_owner = object()
    inherited_route = SimpleNamespace(identity="existing-route")
    source = FakePatcher(
        FakeKeylessH3(),
        {
            "transformer_options": {
                "optimized_attention_override": patch_time_owner,
                KEYLESS_ROUTING_PREPROCESSORS_KEY: (inherited_route,),
            }
        },
    )
    patched = _patch(source)
    options = _invoke(patched)

    assert options["optimized_attention_override"] is patch_time_owner
    preprocessors = options[KEYLESS_ROUTING_PREPROCESSORS_KEY]
    assert preprocessors[0] is inherited_route
    assert len(preprocessors) == 2
    untwist = preprocessors[1]
    assert callable(untwist)
    assert isinstance(untwist.identity, str)
    assert untwist.identity.startswith("minimax_h3_untwist_keyless_route_v1:")

    route = torch.ones((12, 2, 128))
    raw_v = torch.randn_like(route)
    raw_before = raw_v.clone()
    transformed = untwist(route)
    assert torch.equal(transformed[:2], route[:2])
    assert not torch.equal(transformed[2:4], route[2:4])
    assert torch.equal(transformed[4:], route[4:])
    assert torch.equal(route, torch.ones_like(route))
    assert torch.equal(raw_v, raw_before)


def test_keyless_node_preserves_call_time_attention_owner_over_patch_time_owner() -> None:
    patch_time_owner = object()
    call_time_owner = object()
    patched = _patch(
        FakePatcher(
            FakeKeylessH3(),
            {"transformer_options": {"optimized_attention_override": patch_time_owner}},
        )
    )
    options = _invoke(
        patched,
        transformer_options={"optimized_attention_override": call_time_owner},
    )
    assert options["optimized_attention_override"] is call_time_owner
    assert KEYLESS_ROUTING_PREPROCESSORS_KEY in options


def test_keyless_node_rejects_explicit_value_domain_before_model_execution() -> None:
    patched = _patch(FakePatcher(FakeKeylessH3()))
    with pytest.raises(RuntimeError, match="does not yet support explicit row domains"):
        _invoke(
            patched,
            transformer_options={"minimax_h3_keyless_value_domain_v1": slice(0, 8)},
        )


def test_keyless_node_fails_closed_on_malformed_advertised_contract() -> None:
    model = FakeKeylessH3()
    getattr(model, KEYLESS_CONTRACT_KEY).retrieval_source = "route"
    with pytest.raises(KeylessH3CompatibilityError, match="retrieval_source"):
        _patch(FakePatcher(model))


def test_keyless_inactive_call_installs_no_route_preprocessor() -> None:
    patched = MiniMaxH3UntwistRoPE().patch(
        FakePatcher(FakeKeylessH3()),
        high_scale_start=0.5,
        high_scale_end=0.5,
        low_scale_start=1.5,
        low_scale_end=1.5,
        beta=1.0,
        start_percent=0.0,
        end_percent=0.1,
        verbose=False,
        reference_scope="image_and_video",
        scale_temporal_axis=False,
    )[0]
    denoiser_sigmas = torch.linspace(1.0, 0.1, 11)
    sample_sigmas = torch.cat((denoiser_sigmas, torch.zeros(1)))
    wrapper = patched.model_options["model_function_wrapper"]
    seen = {}
    wrapper(
        lambda input_x, timestep, **c: seen.update(c),
        {
            "input": torch.zeros(1),
            "timestep": denoiser_sigmas[-1],
            "c": {
                "minimax_payload": _payload(),
                "transformer_options": {"sample_sigmas": sample_sigmas},
            },
        },
    )
    options = seen["transformer_options"]
    assert options["minimax_h3_untwist_rope"]["enabled"] is False
    assert KEYLESS_ROUTING_PREPROCESSORS_KEY not in options
