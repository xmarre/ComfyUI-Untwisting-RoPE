from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from .config import clamp_float, schedule_fraction


_H3_VISUAL_REFERENCE_KINDS = frozenset({"image", "video", "video_audio"})
_H3_REFERENCE_SCOPES = frozenset({"image_only", "image_and_video", "all_visual_including_continuum"})
_H3_CONTINUUM_METADATA_KEY = "_h3_continuum"
_H3_CONTINUUM_PRESERVE_ROPE_KEY = "preserve_rope"
_H3_CONTINUUM_VIDEO_CONTEXT_ROLE = "video_context"
_H3_CONTINUUM_LEGACY_VIDEO_MARKER = "_h3cj_video_context"


@dataclass(frozen=True)
class MiniMaxH3ReferenceSelection:
    """Resolved native H3 visual-reference rows and conservative filtering diagnostics."""

    ranges: Tuple[Tuple[int, int], ...]
    selected_kinds: Tuple[str, ...]
    total_visual_refs: int
    skipped_video_refs: int
    skipped_continuum_refs: int
    mapping_valid: bool
    reason: str = ""


def lerp(a: float, b: float, t: float) -> float:
    return float(a) + (float(b) - float(a)) * float(t)


def build_frequency_scale_vector(
    head_dim: int,
    axes_dim: Sequence[int],
    high_scale: float,
    low_scale: float,
    beta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Build a per-channel scale vector for RoPE pair chunks.

    The vector is applied before ComfyUI applies RoPE. Since each 2D RoPE pair receives
    a scalar shared by both entries, that scaling commutes with the 2D rotation and is
    equivalent to scaling the already-rotated reference keys.
    """
    head_dim = int(head_dim)
    axes = [int(x) for x in axes_dim if int(x) > 0]
    if not axes or sum(axes) != head_dim:
        axes = [head_dim]

    beta = max(0.01, float(beta))
    pieces: List[torch.Tensor] = []
    has_three_axes = len(axes) == 3

    for axis_index, axis_dim in enumerate(axes):
        n_pairs = axis_dim // 2
        if n_pairs <= 0:
            pieces.append(torch.ones(axis_dim, device=device, dtype=dtype))
            continue

        # FLUX-style axes are normally [image/ref index, y, x]. The paper treats
        # the non-spatial partition as not responsible for spatial copying, so keep
        # it on the low-frequency/reference-preserving side instead of suppressing it.
        if has_three_axes and axis_index == 0:
            pair_scales = torch.full((n_pairs,), float(low_scale), device=device, dtype=torch.float32)
        else:
            if n_pairs == 1:
                d = torch.zeros((1,), device=device, dtype=torch.float32)
            else:
                d = torch.linspace(0.0, 1.0, n_pairs, device=device, dtype=torch.float32)
            pair_scales = float(high_scale) + (float(low_scale) - float(high_scale)) * d.pow(beta)

        pieces.append(pair_scales.to(dtype=dtype).repeat_interleave(2))
        if axis_dim % 2:
            pieces.append(torch.ones(1, device=device, dtype=dtype))

    out = torch.cat(pieces, dim=0) if pieces else torch.ones(head_dim, device=device, dtype=dtype)
    if out.numel() < head_dim:
        out = torch.nn.functional.pad(out, (0, head_dim - out.numel()), value=1.0)
    return out[:head_dim]


def build_h3_frequency_scale_vector(
    head_dim: int,
    rope_axis_count: int,
    rope_freqs_per_axis: int,
    high_scale: float,
    low_scale: float,
    beta: float,
    device: torch.device,
    dtype: torch.dtype,
    scale_temporal_axis: bool = False,
) -> torch.Tensor:
    """Build MiniMax H3's split-half RoPE scale vector.

    Native H3 rotates the first ``2 * axes * freqs`` head channels with split-half
    RoPE. For the current three-axis layout one half is
    ``[t freqs | h freqs | w freqs]`` and the second half contains the paired
    channels in the same order.

    Untwisting RoPE was validated by the paper for spatial reference behavior. H3's
    temporal rotary bank is therefore left exactly native by default; callers may
    explicitly enable temporal-axis scaling for experiments. The unrotated tail is
    always left at scale 1.
    """
    head_dim = int(head_dim)
    axis_count = int(rope_axis_count)
    freq_count = int(rope_freqs_per_axis)
    if head_dim <= 0 or axis_count <= 0 or freq_count <= 0:
        return torch.ones(max(0, head_dim), device=device, dtype=dtype)

    half_dim = axis_count * freq_count
    rot_dim = half_dim * 2
    if rot_dim > head_dim:
        return torch.ones(head_dim, device=device, dtype=dtype)

    beta = max(0.01, float(beta))
    if freq_count == 1:
        d = torch.zeros((1,), device=device, dtype=torch.float32)
    else:
        d = torch.linspace(0.0, 1.0, freq_count, device=device, dtype=torch.float32)
    scheduled = float(high_scale) + (float(low_scale) - float(high_scale)) * d.pow(beta)

    axis_scales: List[torch.Tensor] = []
    for axis_index in range(axis_count):
        if axis_count == 3 and axis_index == 0 and not bool(scale_temporal_axis):
            axis_scales.append(torch.ones(freq_count, device=device, dtype=torch.float32))
        else:
            axis_scales.append(scheduled)

    half = torch.cat(axis_scales, dim=0).to(dtype=dtype)
    rotated = torch.cat((half, half), dim=0)
    if rot_dim == head_dim:
        return rotated
    return torch.cat((rotated, torch.ones(head_dim - rot_dim, device=device, dtype=dtype)), dim=0)


def _reference_ranges_from_options(seq_len: int, extra_options: Dict[str, Any]) -> Tuple[Optional[Tuple[int, int]], List[Tuple[int, int]]]:
    img_slice = extra_options.get("img_slice", None)
    ref_counts = extra_options.get("reference_image_num_tokens", None)
    if not isinstance(img_slice, (list, tuple)) or len(img_slice) != 2:
        return None, []

    try:
        img_start = max(0, min(int(img_slice[0]), int(seq_len)))
        img_end = max(img_start, min(int(img_slice[1]), int(seq_len)))
    except Exception:
        return None, []

    if ref_counts is None:
        return (img_start, img_end), []
    if isinstance(ref_counts, int):
        counts = [int(ref_counts)]
    elif isinstance(ref_counts, (list, tuple)):
        counts = []
        for value in ref_counts:
            try:
                count = int(value)
            except Exception:
                continue
            if count > 0:
                counts.append(count)
    else:
        return (img_start, img_end), []

    total_ref = sum(counts)
    total_img = img_end - img_start
    if total_ref <= 0 or total_ref > total_img:
        return (img_start, img_end), []

    ref_start = img_end - total_ref
    ranges: List[Tuple[int, int]] = []
    cur = ref_start
    for count in counts:
        end = min(img_end, cur + int(count))
        if end > cur:
            ranges.append((cur, end))
        cur = end
    return (img_start, ref_start), ranges


def _h3_layout_ref_img_ranges(layout: Any) -> Tuple[Tuple[int, int], ...]:
    segments = getattr(layout, "segments", None)
    if not isinstance(segments, (list, tuple)):
        return tuple()

    ranges: List[Tuple[int, int]] = []
    for segment in segments:
        if not isinstance(segment, (list, tuple)) or len(segment) != 3:
            continue
        start, end, kind = segment
        if str(kind) != "ref_img":
            continue
        try:
            start_i = int(start)
            end_i = int(end)
        except Exception:
            continue
        if start_i >= 0 and end_i > start_i:
            ranges.append((start_i, end_i))
    return tuple(ranges)


def _h3_reference_requires_native_rope(ref: Dict[str, Any]) -> bool:
    """Return True for Continuum/foreign refs that explicitly request native RoPE."""
    if bool(ref.get(_H3_CONTINUUM_LEGACY_VIDEO_MARKER, False)):
        return True
    metadata = ref.get(_H3_CONTINUUM_METADATA_KEY)
    if not isinstance(metadata, dict):
        return False
    if metadata.get(_H3_CONTINUUM_PRESERVE_ROPE_KEY) is True:
        return True
    return str(metadata.get("role", "")) == _H3_CONTINUUM_VIDEO_CONTEXT_ROLE


def _kind_selected(kind: str, scope: str) -> bool:
    """Return whether a native H3 reference kind belongs to the requested scope."""
    if scope == "image_only":
        return kind == "image"
    if scope == "image_and_video":
        return kind in {"image", "video"}
    if scope == "all_visual_including_continuum":
        return kind in _H3_VISUAL_REFERENCE_KINDS
    return kind == "image"


def minimax_h3_visual_reference_selection(
    payload: Any,
    *,
    scope: str = "image_only",
) -> MiniMaxH3ReferenceSelection:
    """Map native H3 refs to ``ref_img`` rows and conservatively choose targets.

    ``PackedLayout.segments`` alone is insufficient because image, video and
    video+audio refs all expose their visual rows as ``ref_img``. Native ref order
    in ``minimax_payload['refs']`` is therefore treated as authoritative and paired
    one-for-one with visual ``ref_img`` segments. Any mismatch fails closed.

    ``image_and_video`` deliberately means image plus *pure* video. Mixed
    ``video_audio`` references remain outside the safe default, and Continuum
    context marked for native RoPE is protected unless the explicit all-visual
    scope is selected.
    """
    scope = str(scope or "image_only")
    if scope not in _H3_REFERENCE_SCOPES:
        scope = "image_only"

    if not isinstance(payload, dict):
        return MiniMaxH3ReferenceSelection(tuple(), tuple(), 0, 0, 0, False, "minimax_payload is missing")

    layout = payload.get("layout")
    packed_ranges = _h3_layout_ref_img_ranges(layout)
    raw_refs = payload.get("refs")

    if raw_refs is None:
        if packed_ranges:
            return MiniMaxH3ReferenceSelection(
                tuple(), tuple(), len(packed_ranges), 0, 0, False,
                "layout contains ref_img rows but minimax_payload.refs is missing",
            )
        return MiniMaxH3ReferenceSelection(tuple(), tuple(), 0, 0, 0, True)

    if not isinstance(raw_refs, (list, tuple)):
        return MiniMaxH3ReferenceSelection(
            tuple(), tuple(), len(packed_ranges), 0, 0, False,
            "minimax_payload.refs is not a sequence",
        )

    visual_refs: List[Dict[str, Any]] = []
    for ref in raw_refs:
        if not isinstance(ref, dict):
            continue
        if str(ref.get("kind", "")) in _H3_VISUAL_REFERENCE_KINDS:
            visual_refs.append(ref)

    if len(visual_refs) != len(packed_ranges):
        return MiniMaxH3ReferenceSelection(
            tuple(),
            tuple(),
            len(visual_refs),
            0,
            0,
            False,
            f"native visual-ref count {len(visual_refs)} does not match ref_img range count {len(packed_ranges)}",
        )

    selected_ranges: List[Tuple[int, int]] = []
    selected_kinds: List[str] = []
    skipped_scope = 0
    skipped_continuum = 0
    include_continuum_context = scope == "all_visual_including_continuum"

    for ref, row_range in zip(visual_refs, packed_ranges, strict=True):
        kind = str(ref.get("kind", ""))
        preserve_native_rope = _h3_reference_requires_native_rope(ref)
        if preserve_native_rope and not include_continuum_context:
            skipped_continuum += 1
            continue
        if not _kind_selected(kind, scope):
            skipped_scope += 1
            continue
        selected_ranges.append(row_range)
        selected_kinds.append(kind)

    return MiniMaxH3ReferenceSelection(
        tuple(selected_ranges),
        tuple(selected_kinds),
        len(visual_refs),
        skipped_scope,
        skipped_continuum,
        True,
    )


def minimax_h3_visual_reference_ranges(
    layout_or_payload: Any,
    refs: Any = None,
    *,
    scope: str = "image_only",
) -> List[Tuple[int, int]]:
    """Backward-compatible list form of the conservative H3 reference selection."""
    if isinstance(layout_or_payload, dict) and "layout" in layout_or_payload:
        payload = layout_or_payload
    else:
        payload = {"layout": layout_or_payload, "refs": refs}
    selection = minimax_h3_visual_reference_selection(payload, scope=scope)
    return list(selection.ranges)


def _adain_tokens(target: torch.Tensor, style: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """AdaIN over the token dimension for tensors shaped [B, H, L, D]."""
    if target.numel() == 0 or style.numel() == 0:
        return target
    t_float = target.float()
    s_float = style.float()
    t_mean = t_float.mean(dim=2, keepdim=True)
    s_mean = s_float.mean(dim=2, keepdim=True)
    t_std = t_float.var(dim=2, keepdim=True, unbiased=False).add(eps).sqrt()
    s_std = s_float.var(dim=2, keepdim=True, unbiased=False).add(eps).sqrt()
    return ((t_float - t_mean) / t_std * s_std + s_mean).to(dtype=target.dtype)


def _concat_ranges(x: torch.Tensor, ranges: Sequence[Tuple[int, int]]) -> torch.Tensor:
    parts = []
    length = int(x.shape[2])
    for start, end in ranges:
        s = max(0, min(int(start), length))
        e = max(s, min(int(end), length))
        if e > s:
            parts.append(x[:, :, s:e, :])
    if not parts:
        return x[:, :, 0:0, :]
    return torch.cat(parts, dim=2)


def _scaled_h3_reference_keys(k: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    if k.ndim != 4:
        return k

    active, t = schedule_fraction(
        float(cfg.get("progress", 0.0)),
        float(cfg.get("start_percent", 0.0)),
        float(cfg.get("end_percent", 0.9)),
    )
    if not active:
        return k

    high_scale = lerp(cfg.get("high_scale_start", 0.95), cfg.get("high_scale_end", 1.0), t)
    low_scale = lerp(cfg.get("low_scale_start", 1.0), cfg.get("low_scale_end", 1.05), t)
    beta = clamp_float(cfg.get("beta", 2.0), 0.01, 32.0, 2.0)

    # An all-ones schedule must remain an exact tensor no-op. Apart from avoiding
    # unnecessary clone/multiply work, this makes fixed-seed neutral controls
    # genuinely isolate the attention-override plumbing from the intervention.
    if high_scale == 1.0 and low_scale == 1.0:
        return k

    scale_vec = build_h3_frequency_scale_vector(
        head_dim=int(k.shape[-1]),
        rope_axis_count=int(cfg.get("rope_axis_count", 3)),
        rope_freqs_per_axis=int(cfg.get("rope_freqs_per_axis", 0)),
        high_scale=high_scale,
        low_scale=low_scale,
        beta=beta,
        device=k.device,
        dtype=k.dtype,
        scale_temporal_axis=bool(cfg.get("scale_temporal_axis", False)),
    ).view(1, 1, 1, int(k.shape[-1]))

    ranges = cfg.get("reference_ranges", [])
    seq_len = int(k.shape[2])
    valid_ranges: List[Tuple[int, int]] = []
    for item in ranges if isinstance(ranges, (list, tuple)) else ():
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        try:
            start = max(0, min(int(item[0]), seq_len))
            end = max(start, min(int(item[1]), seq_len))
        except Exception:
            continue
        if end > start:
            valid_ranges.append((start, end))
    if not valid_ranges:
        return k

    k_out = k.clone()
    for start, end in valid_ranges:
        k_out[:, :, start:end, :] = k_out[:, :, start:end, :] * scale_vec
    return k_out


def make_minimax_h3_attention_override(previous_override: Optional[Callable[..., Any]] = None) -> Callable[..., Any]:
    """Compose H3 post-RoPE key modulation with an existing optimized-attention override."""

    def override(original: Callable[..., Any], q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, *args: Any, **kwargs: Any):
        transformer_options = kwargs.get("transformer_options", None)
        cfg = transformer_options.get("minimax_h3_untwist_rope", None) if isinstance(transformer_options, dict) else None
        if isinstance(cfg, dict) and cfg.get("enabled", False):
            k = _scaled_h3_reference_keys(k, cfg)

        if previous_override is not None:
            return previous_override(original, q, k, v, heads, *args, **kwargs)
        return original(q, k, v, heads, *args, **kwargs)

    return override


def flux_untwist_attn1_patch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pe: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    extra_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """ComfyUI `attn1_patch` hook for FLUX single-stream blocks."""
    if extra_options is None:
        return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": attn_mask}

    cfg = extra_options.get("flux_untwist_rope", None)
    if not isinstance(cfg, dict) or not cfg.get("enabled", False):
        return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": attn_mask}

    # The paper applies reference sharing only in FLUX single-stream blocks.
    if str(extra_options.get("block_type", "")) != "single":
        return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": attn_mask}

    block_index = int(extra_options.get("block_index", -1))
    if block_index < int(cfg.get("start_single_block", 0)) or block_index > int(cfg.get("end_single_block", 999)):
        return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": attn_mask}

    seq_len = int(k.shape[2])
    target_range, ref_ranges = _reference_ranges_from_options(seq_len, extra_options)
    if not ref_ranges:
        return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": attn_mask}

    active, t = schedule_fraction(
        float(cfg.get("progress", 0.0)),
        float(cfg.get("start_percent", 0.0)),
        float(cfg.get("end_percent", 1.0)),
    )
    if not active:
        return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": attn_mask}

    high_scale = lerp(cfg.get("high_scale_start", 0.25), cfg.get("high_scale_end", 0.75), t)
    low_scale = lerp(cfg.get("low_scale_start", 1.0), cfg.get("low_scale_end", 1.4), t)
    beta = clamp_float(cfg.get("beta", 2.0), 0.01, 32.0, 2.0)

    scale_vec = build_frequency_scale_vector(
        head_dim=int(k.shape[-1]),
        axes_dim=cfg.get("axes_dim", []) or [],
        high_scale=high_scale,
        low_scale=low_scale,
        beta=beta,
        device=k.device,
        dtype=k.dtype,
    ).view(1, 1, 1, int(k.shape[-1]))

    q_out = q
    k_out = k.clone()

    for start, end in ref_ranges:
        k_out[:, :, start:end, :] = k_out[:, :, start:end, :] * scale_vec

    adain_strength = clamp_float(cfg.get("qk_adain_strength", 0.0), 0.0, 1.0, 0.0)
    if adain_strength > 0.0 and target_range is not None:
        target_start, target_end = target_range
        if target_end > target_start:
            ref_q = _concat_ranges(q, ref_ranges)
            ref_k = _concat_ranges(k, ref_ranges)
            if ref_q.shape[2] > 0 and ref_k.shape[2] > 0:
                q_out = q.clone()
                q_adain = _adain_tokens(q_out[:, :, target_start:target_end, :], ref_q)
                k_adain = _adain_tokens(k_out[:, :, target_start:target_end, :], ref_k)
                q_out[:, :, target_start:target_end, :] = (
                    q_out[:, :, target_start:target_end, :] * (1.0 - adain_strength)
                    + q_adain * adain_strength
                )
                k_out[:, :, target_start:target_end, :] = (
                    k_out[:, :, target_start:target_end, :] * (1.0 - adain_strength)
                    + k_adain * adain_strength
                )

    return {"q": q_out, "k": k_out, "v": v, "pe": pe, "attn_mask": attn_mask}
