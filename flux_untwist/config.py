from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


_PREFIX = "[Flux2UntwistRoPE]"
_H3_PREFIX = "[MiniMaxH3UntwistRoPE]"


@dataclass(frozen=True)
class UntwistConfig:
    """Runtime configuration copied into ComfyUI transformer_options per model call."""

    enabled: bool
    axes_dim: Tuple[int, ...]
    high_scale_start: float
    high_scale_end: float
    low_scale_start: float
    low_scale_end: float
    beta: float
    start_percent: float
    end_percent: float
    start_single_block: int
    end_single_block: int
    qk_adain_strength: float
    reference_latents_method: str
    progress: float
    verbose: bool

    def as_transformer_options(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "axes_dim": list(self.axes_dim),
            "high_scale_start": float(self.high_scale_start),
            "high_scale_end": float(self.high_scale_end),
            "low_scale_start": float(self.low_scale_start),
            "low_scale_end": float(self.low_scale_end),
            "beta": float(self.beta),
            "start_percent": float(self.start_percent),
            "end_percent": float(self.end_percent),
            "start_single_block": int(self.start_single_block),
            "end_single_block": int(self.end_single_block),
            "qk_adain_strength": float(self.qk_adain_strength),
            "reference_latents_method": str(self.reference_latents_method),
            "progress": float(self.progress),
            "verbose": bool(self.verbose),
        }


@dataclass(frozen=True)
class MiniMaxH3UntwistConfig:
    """MiniMax H3 runtime configuration for post-RoPE reference-key modulation."""

    enabled: bool
    reference_ranges: Tuple[Tuple[int, int], ...]
    reference_scope: str
    rope_axis_count: int
    rope_freqs_per_axis: int
    high_scale_start: float
    high_scale_end: float
    low_scale_start: float
    low_scale_end: float
    beta: float
    start_percent: float
    end_percent: float
    scale_temporal_axis: bool
    progress: float
    verbose: bool

    def as_transformer_options(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "reference_ranges": [list(r) for r in self.reference_ranges],
            "reference_scope": str(self.reference_scope),
            "rope_axis_count": int(self.rope_axis_count),
            "rope_freqs_per_axis": int(self.rope_freqs_per_axis),
            "high_scale_start": float(self.high_scale_start),
            "high_scale_end": float(self.high_scale_end),
            "low_scale_start": float(self.low_scale_start),
            "low_scale_end": float(self.low_scale_end),
            "beta": float(self.beta),
            "start_percent": float(self.start_percent),
            "end_percent": float(self.end_percent),
            "scale_temporal_axis": bool(self.scale_temporal_axis),
            "progress": float(self.progress),
            "verbose": bool(self.verbose),
        }


def clamp_float(value: Any, lo: float, hi: float, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if out != out or out in (float("inf"), float("-inf")):
        out = float(default)
    return max(float(lo), min(float(hi), out))


def clamp_int(value: Any, lo: int, hi: int, default: int) -> int:
    try:
        out = int(value)
    except Exception:
        out = int(default)
    return max(int(lo), min(int(hi), out))


def coerce_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y", "t"}
    return bool(value)


def normalize_percent_window(start: Any, end: Any) -> Tuple[float, float]:
    s = clamp_float(start, 0.0, 1.0, 0.0)
    e = clamp_float(end, 0.0, 1.0, 1.0)
    if e < s:
        s, e = e, s
    if abs(e - s) < 1e-6:
        e = min(1.0, s + 1e-6)
    return s, e


def schedule_fraction(progress: float, start_percent: float, end_percent: float) -> Tuple[bool, float]:
    """Return (active, normalized_t) for a denoising progress in [0, 1]."""
    p = clamp_float(progress, 0.0, 1.0, 0.0)
    s, e = normalize_percent_window(start_percent, end_percent)
    if p < s or p > e:
        return False, 0.0 if p < s else 1.0
    return True, max(0.0, min(1.0, (p - s) / max(e - s, 1e-6)))


def normalize_reference_method(method: Any) -> str:
    text = str(method or "index").strip().lower()
    if text in {"uxo/uno", "uno", "uso"}:
        return "uxo"
    if text in {"index", "offset", "uxo", "index_timestep_zero"}:
        return text
    return "index"


def safe_axes_dim(params_or_model: Any) -> Tuple[int, ...]:
    """Extract FLUX axes_dim from a diffusion model or its params object."""
    obj = getattr(params_or_model, "params", params_or_model)
    axes = getattr(obj, "axes_dim", None)
    if axes is None:
        return tuple()
    try:
        return tuple(int(x) for x in axes)
    except Exception:
        return tuple()
