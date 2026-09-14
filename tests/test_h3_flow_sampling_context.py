from types import SimpleNamespace

import pytest
import torch

from flux_untwist.spectrum_h3 import VISUAL_PATCH_RUNTIME_KEY
from flux_untwist.utils import progress_from_h3_flow_sampling_context
from nodes import MiniMaxH3UntwistRoPE


GLOBAL_NONZERO = [
    1.0,
    0.9882352948188782,
    0.9729729890823364,
    0.9523809552192688,
    0.9230769276618958,
    0.8780487775802612,
    0.800000011920929,
    0.6315789222717285,
]


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


def _payload():
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


def _context(stage="high", start_index=5):
    return {
        "api": 1,
        "source": "h3_flow_execution_contract_untwist_clock_trial",
        "units": "comfy_sigma",
        "stage": stage,
        "original_nonzero_sigmas": list(GLOBAL_NONZERO),
        "original_schedule_digest": "4e0fb5ab2d9e5f6a",
        "original_stage_start_index": start_index,
        "invocation_generation": 1789371316199037927,
    }


def _patched():
    return MiniMaxH3UntwistRoPE().patch(
        FakePatcher(FakeH3()),
        high_scale_start=0.95,
        high_scale_end=1.0,
        low_scale_start=1.0,
        low_scale_end=1.05,
        beta=2.0,
        start_percent=0.0,
        end_percent=0.90,
        verbose=False,
        reference_scope="image_and_video",
        scale_temporal_axis=False,
    )[0]


def test_flow_context_maps_first_high_to_original_progress_without_replacing_sample_sigmas():
    patched = _patched()
    wrapper = patched.model_options["model_function_wrapper"]
    local_sigmas = torch.tensor([GLOBAL_NONZERO[5], GLOBAL_NONZERO[6], GLOBAL_NONZERO[7], 0.0])
    seen = {}

    result = wrapper(
        lambda input_x, timestep, **c: seen.update(c) or "ok",
        {
            "input": torch.zeros(1),
            "timestep": torch.tensor([GLOBAL_NONZERO[5]]),
            "c": {
                "minimax_payload": _payload(),
                "transformer_options": {
                    "sample_sigmas": local_sigmas,
                    "h3_flow_sampling_context": _context(),
                },
            },
        },
    )

    assert result == "ok"
    to = seen["transformer_options"]
    cfg = to["minimax_h3_untwist_rope"]
    assert cfg["progress"] == pytest.approx(5 / 7)
    assert cfg["progress_source"] == "h3_flow_sampling_context_v1"
    assert cfg["flow_stage"] == "high"
    assert cfg["flow_stage_start_index"] == 5
    assert torch.equal(to["sample_sigmas"], local_sigmas)
    runtime_entry = to[VISUAL_PATCH_RUNTIME_KEY][-1]
    assert runtime_entry["schedule_progress"] == pytest.approx(5 / 7)
    assert runtime_entry["active"] is True


def test_flow_context_maps_probe_same_coordinate_to_same_original_progress():
    assert progress_from_h3_flow_sampling_context(
        torch.tensor([GLOBAL_NONZERO[5]]),
        _context(stage="probe", start_index=5),
    ) == pytest.approx(5 / 7)


def test_absent_flow_context_retains_legacy_child_local_progress():
    patched = _patched()
    wrapper = patched.model_options["model_function_wrapper"]
    local_sigmas = torch.tensor([GLOBAL_NONZERO[5], GLOBAL_NONZERO[6], GLOBAL_NONZERO[7], 0.0])
    seen = {}

    wrapper(
        lambda input_x, timestep, **c: seen.update(c),
        {
            "input": torch.zeros(1),
            "timestep": torch.tensor([GLOBAL_NONZERO[5]]),
            "c": {
                "minimax_payload": _payload(),
                "transformer_options": {"sample_sigmas": local_sigmas},
            },
        },
    )

    cfg = seen["transformer_options"]["minimax_h3_untwist_rope"]
    assert cfg["progress"] == 0.0
    assert cfg["progress_source"] == "local_sample_sigmas"


def test_malformed_recognized_flow_context_fails_closed():
    context = _context()
    context["units"] = "unknown"
    with pytest.raises(RuntimeError, match="comfy_sigma"):
        progress_from_h3_flow_sampling_context(torch.tensor([GLOBAL_NONZERO[5]]), context)
