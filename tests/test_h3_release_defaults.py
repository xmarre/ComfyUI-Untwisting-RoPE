import pytest
import torch

from flux_untwist.patches import make_minimax_h3_attention_override
from nodes import MiniMaxH3UntwistRoPE


def test_h3_release_default_contract_is_exact():
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
    assert required["verbose"][1]["default"] is False
    assert optional["reference_scope"][1]["default"] == "image_and_video"
    assert optional["scale_temporal_axis"][1]["default"] is False


@pytest.mark.parametrize(
    ("progress", "expected"),
    [
        (0.0, torch.tensor([0.95, 1.00, 0.95, 1.00])),
        (0.90, torch.tensor([1.00, 1.05, 1.00, 1.05])),
    ],
)
def test_h3_attention_fallbacks_match_release_schedule(progress, expected):
    q = torch.ones((1, 1, 2, 4), dtype=torch.float32)
    k = torch.ones_like(q)
    v = torch.ones_like(q)

    override = make_minimax_h3_attention_override()
    out = override(
        lambda q_in, k_in, v_in, heads, *args, **kwargs: k_in,
        q,
        k,
        v,
        1,
        transformer_options={
            "minimax_h3_untwist_rope": {
                "enabled": True,
                "reference_ranges": [[0, 1]],
                "rope_axis_count": 1,
                "rope_freqs_per_axis": 2,
                "progress": progress,
            }
        },
    )

    assert torch.allclose(out[0, 0, 0], expected)
    assert torch.allclose(out[0, 0, 1], torch.ones(4))
    assert torch.allclose(k, torch.ones_like(k))
