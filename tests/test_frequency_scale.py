import pytest
import torch

from flux_untwist.patches import (
    _reference_ranges_from_options,
    build_frequency_scale_vector,
    build_h3_frequency_scale_vector,
    make_minimax_h3_attention_override,
    minimax_h3_visual_reference_selection,
)
from flux_untwist.utils import progress_from_schedule_index


def test_frequency_scale_vector_endpoints_for_two_spatial_axes():
    v = build_frequency_scale_vector(
        head_dim=8,
        axes_dim=[4, 4],
        high_scale=0.25,
        low_scale=1.5,
        beta=2.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert v.shape == (8,)
    assert torch.allclose(v[0:2], torch.tensor([0.25, 0.25]))
    assert torch.allclose(v[2:4], torch.tensor([1.5, 1.5]))
    assert torch.allclose(v[4:6], torch.tensor([0.25, 0.25]))
    assert torch.allclose(v[6:8], torch.tensor([1.5, 1.5]))


def test_three_axis_first_axis_uses_low_scale():
    v = build_frequency_scale_vector(12, [4, 4, 4], 0.2, 1.4, 2.0, torch.device("cpu"), torch.float32)
    assert torch.allclose(v[0:4], torch.full((4,), 1.4))


def test_reference_ranges_are_tail_of_image_slice():
    target_range, refs = _reference_ranges_from_options(
        180,
        {
            "img_slice": [64, 180],
            "reference_image_num_tokens": [16, 20],
        },
    )
    assert target_range == (64, 144)
    assert refs == [(144, 160), (160, 180)]


def test_h3_split_half_scale_keeps_temporal_axis_native_by_default():
    v = build_h3_frequency_scale_vector(
        head_dim=128,
        rope_axis_count=3,
        rope_freqs_per_axis=16,
        high_scale=0.25,
        low_scale=1.5,
        beta=2.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert v.shape == (128,)
    assert torch.allclose(v[:48], v[48:96])
    assert torch.allclose(v[0:16], torch.ones(16))
    assert torch.allclose(v[48:64], torch.ones(16))
    for axis_start in (16, 32, 64, 80):
        assert torch.isclose(v[axis_start], torch.tensor(0.25))
        assert torch.isclose(v[axis_start + 15], torch.tensor(1.5))
    assert torch.allclose(v[96:], torch.ones(32))


def test_h3_temporal_axis_scaling_is_explicit_opt_in():
    v = build_h3_frequency_scale_vector(
        head_dim=128,
        rope_axis_count=3,
        rope_freqs_per_axis=16,
        high_scale=0.25,
        low_scale=1.5,
        beta=2.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
        scale_temporal_axis=True,
    )
    for axis_start in (0, 16, 32, 48, 64, 80):
        assert torch.isclose(v[axis_start], torch.tensor(0.25))
        assert torch.isclose(v[axis_start + 15], torch.tensor(1.5))


def test_h3_schedule_progress_uses_real_19_call_schedule_position():
    denoiser_sigmas = torch.linspace(1.0, 0.10, 19)
    sample_sigmas = torch.cat((denoiser_sigmas, torch.zeros(1)))

    assert progress_from_schedule_index(denoiser_sigmas[0], sigmas=sample_sigmas) == 0.0
    assert progress_from_schedule_index(denoiser_sigmas[-1], sigmas=sample_sigmas) == 1.0
    assert progress_from_schedule_index(denoiser_sigmas[16], sigmas=sample_sigmas) == pytest.approx(16 / 18)


def _mixed_h3_payload():
    class Layout:
        segments = [
            (0, 10, "text"),
            (10, 18, "ref_img"),
            (18, 26, "ref_img"),
            (26, 34, "ref_img"),
            (34, 38, "ref_audio"),
            (38, 54, "ref_img"),
            (54, 64, "audio"),
            (64, 120, "video"),
        ]

    return {
        "layout": Layout(),
        "refs": [
            {"kind": "image"},
            {"kind": "video", "latent_t": 1},
            {"kind": "video_audio", "latent_t": 1, "ref_audio_t": 1},
            {
                "kind": "video_audio",
                "latent_t": 2,
                "ref_audio_t": 2,
                "_h3_continuum": {
                    "api": 1,
                    "role": "video_context",
                    "preserve_rope": True,
                },
            },
        ],
    }


def test_h3_scope_image_only_selects_only_images():
    selected = minimax_h3_visual_reference_selection(_mixed_h3_payload(), scope="image_only")
    assert selected.mapping_valid is True
    assert selected.ranges == ((10, 18),)
    assert selected.selected_kinds == ("image",)
    assert selected.total_visual_refs == 4
    assert selected.skipped_video_refs == 2
    assert selected.skipped_continuum_refs == 1


def test_h3_scope_image_and_video_selects_pure_video_and_excludes_video_audio():
    selected = minimax_h3_visual_reference_selection(_mixed_h3_payload(), scope="image_and_video")
    assert selected.ranges == ((10, 18), (18, 26))
    assert selected.selected_kinds == ("image", "video")
    assert selected.skipped_video_refs == 1
    assert selected.skipped_continuum_refs == 1


def test_h3_scope_all_visual_includes_video_audio_and_continuum():
    selected = minimax_h3_visual_reference_selection(
        _mixed_h3_payload(),
        scope="all_visual_including_continuum",
    )
    assert selected.ranges == ((10, 18), (18, 26), (26, 34), (38, 54))
    assert selected.selected_kinds == ("image", "video", "video_audio", "video_audio")
    assert selected.skipped_video_refs == 0
    assert selected.skipped_continuum_refs == 0


def test_h3_reference_selection_fails_closed_on_native_mapping_mismatch():
    payload = _mixed_h3_payload()
    payload["refs"] = payload["refs"][:2]
    selected = minimax_h3_visual_reference_selection(payload, scope="image_and_video")
    assert selected.mapping_valid is False
    assert selected.ranges == tuple()
    assert "does not match" in selected.reason


def test_h3_attention_override_modulates_only_reference_keys():
    q = torch.ones((1, 2, 8, 8), dtype=torch.float32)
    k = torch.ones_like(q)
    v = torch.ones_like(q)
    seen = {}

    def original(q_in, k_in, v_in, heads, *args, **kwargs):
        seen["q"] = q_in
        seen["k"] = k_in
        seen["v"] = v_in
        return k_in

    override = make_minimax_h3_attention_override()
    out = override(
        original,
        q,
        k,
        v,
        2,
        transformer_options={
            "minimax_h3_untwist_rope": {
                "enabled": True,
                "reference_ranges": [[2, 4]],
                "reference_scope": "image_and_video",
                "rope_axis_count": 1,
                "rope_freqs_per_axis": 2,
                "high_scale_start": 0.5,
                "high_scale_end": 0.5,
                "low_scale_start": 1.5,
                "low_scale_end": 1.5,
                "beta": 1.0,
                "start_percent": 0.0,
                "end_percent": 1.0,
                "progress": 0.5,
                "scale_temporal_axis": False,
            }
        },
    )

    expected = torch.tensor([0.5, 1.5, 0.5, 1.5, 1.0, 1.0, 1.0, 1.0])
    assert seen["q"] is q
    assert seen["v"] is v
    assert torch.allclose(out[:, :, :2, :], torch.ones((1, 2, 2, 8)))
    assert torch.allclose(out[:, :, 2:4, :], expected.view(1, 1, 1, 8).expand(1, 2, 2, 8))
    assert torch.allclose(out[:, :, 4:, :], torch.ones((1, 2, 4, 8)))
    assert torch.allclose(k, torch.ones_like(k))


def test_h3_attention_override_all_ones_is_exact_key_noop():
    q = torch.randn((1, 1, 2, 4))
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    seen = {}

    def original(q_in, k_in, v_in, heads, *args, **kwargs):
        seen["k"] = k_in
        return k_in

    override = make_minimax_h3_attention_override()
    override(
        original,
        q,
        k,
        v,
        1,
        transformer_options={
            "minimax_h3_untwist_rope": {
                "enabled": True,
                "reference_ranges": [[0, 1]],
                "reference_scope": "image_and_video",
                "rope_axis_count": 1,
                "rope_freqs_per_axis": 1,
                "high_scale_start": 1.0,
                "high_scale_end": 1.0,
                "low_scale_start": 1.0,
                "low_scale_end": 1.0,
                "beta": 2.0,
                "start_percent": 0.0,
                "end_percent": 1.0,
                "progress": 0.5,
            }
        },
    )
    assert seen["k"] is k


def test_h3_attention_override_chains_existing_override_after_scaling():
    q = torch.ones((1, 1, 2, 4))
    k = torch.ones_like(q)
    v = torch.ones_like(q)
    seen = {}

    def previous(original, q_in, k_in, v_in, heads, *args, **kwargs):
        seen["k"] = k_in.clone()
        return original(q_in, k_in, v_in, heads, *args, **kwargs)

    def original(q_in, k_in, v_in, heads, *args, **kwargs):
        return k_in

    override = make_minimax_h3_attention_override(previous)
    override(
        original,
        q,
        k,
        v,
        1,
        transformer_options={
            "minimax_h3_untwist_rope": {
                "enabled": True,
                "reference_ranges": [[0, 1]],
                "reference_scope": "image_and_video",
                "rope_axis_count": 1,
                "rope_freqs_per_axis": 1,
                "high_scale_start": 0.5,
                "high_scale_end": 0.5,
                "low_scale_start": 1.5,
                "low_scale_end": 1.5,
                "beta": 2.0,
                "start_percent": 0.0,
                "end_percent": 1.0,
                "progress": 0.5,
            }
        },
    )
    assert torch.allclose(seen["k"][0, 0, 0], torch.tensor([0.5, 0.5, 1.0, 1.0]))
    assert torch.allclose(seen["k"][0, 0, 1], torch.ones(4))
