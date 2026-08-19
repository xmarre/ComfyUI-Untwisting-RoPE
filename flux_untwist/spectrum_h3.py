from __future__ import annotations

import math
from typing import Any, Dict, Tuple


VISUAL_PATCH_PROFILES_KEY = "spectrum_h3_visual_reference_patch_profiles"
VISUAL_PATCH_RUNTIME_KEY = "spectrum_h3_visual_reference_patch_runtime"
VISUAL_PATCH_SCHEMA_VERSION = 1
VISUAL_PATCH_PROVIDER = "comfyui-flux2-untwisting-rope"
VISUAL_PATCH_KIND = "visual_reference_attention_modulation"
VISUAL_PATCH_ARCHITECTURE = "minimax_h3"


def _existing_entries(options: Dict[str, Any], key: str) -> list[Any]:
    value = options.get(key)
    if value is None:
        return []
    if isinstance(value, (tuple, list)):
        return list(value)
    # Preserve malformed/foreign data. A Spectrum consumer can then fail safe
    # instead of this producer silently replacing another extension's metadata.
    return [value]


def _next_instance_id(entries: list[Any]) -> str:
    used = {
        str(value.get("instance_id"))
        for value in entries
        if isinstance(value, dict) and value.get("instance_id") is not None
    }
    ordinal = 1
    while f"untwist-h3-{ordinal}" in used:
        ordinal += 1
    return f"untwist-h3-{ordinal}"


def _strength_summary(
    high_scale_start: float,
    high_scale_end: float,
    low_scale_start: float,
    low_scale_end: float,
) -> float:
    return max(
        abs(float(high_scale_start) - 1.0),
        abs(float(high_scale_end) - 1.0),
        abs(float(low_scale_start) - 1.0),
        abs(float(low_scale_end) - 1.0),
    )


def register_spectrum_h3_profile(
    model_options: Dict[str, Any],
    *,
    block_count: int,
    high_scale_start: float,
    high_scale_end: float,
    low_scale_start: float,
    low_scale_end: float,
    beta: float,
    start_percent: float,
    end_percent: float,
    scope: str,
    scale_temporal_axis: bool,
) -> Tuple[Dict[str, Any], str]:
    """Return copied model options carrying Untwist's static Spectrum profile."""
    out = dict(model_options or {})
    entries = _existing_entries(out, VISUAL_PATCH_PROFILES_KEY)
    instance_id = _next_instance_id(entries)

    start = max(0.0, min(1.0, float(start_percent)))
    end = max(start, min(1.0, float(end_percent)))
    high_start = float(high_scale_start)
    high_end = float(high_scale_end)
    low_start = float(low_scale_start)
    low_end = float(low_scale_end)
    beta_value = float(beta)
    if not all(math.isfinite(value) for value in (high_start, high_end, low_start, low_end, beta_value)):
        raise ValueError("Untwist Spectrum profile requires finite scale metadata")

    start_strength = max(abs(high_start - 1.0), abs(low_start - 1.0))
    end_strength = max(abs(high_end - 1.0), abs(low_end - 1.0))
    entries.append(
        {
            "schema_version": VISUAL_PATCH_SCHEMA_VERSION,
            "provider": VISUAL_PATCH_PROVIDER,
            "kind": VISUAL_PATCH_KIND,
            "architecture": VISUAL_PATCH_ARCHITECTURE,
            "instance_id": instance_id,
            "block_indices_0based": list(range(int(block_count))),
            "model_block_count": int(block_count),
            "strength": _strength_summary(high_start, high_end, low_start, low_end),
            "progress_start": start,
            "progress_end": end,
            "hard_start": bool(start > 0.0 and start_strength > 0.0),
            "hard_end": bool(end < 1.0 and end_strength > 0.0),
            "scope": str(scope),
            "high_scale_start": high_start,
            "high_scale_end": high_end,
            "low_scale_start": low_start,
            "low_scale_end": low_end,
            "beta": beta_value,
            "scale_temporal_axis": bool(scale_temporal_axis),
        }
    )
    out[VISUAL_PATCH_PROFILES_KEY] = tuple(entries)
    return out, instance_id


def append_spectrum_h3_runtime(
    transformer_options: Dict[str, Any],
    *,
    instance_id: str,
    progress: float,
    active: bool,
) -> Dict[str, Any]:
    """Return copied call-local transformer options with Untwist runtime state."""
    out = dict(transformer_options or {})
    entries = _existing_entries(out, VISUAL_PATCH_RUNTIME_KEY)
    entries.append(
        {
            "schema_version": VISUAL_PATCH_SCHEMA_VERSION,
            "provider": VISUAL_PATCH_PROVIDER,
            "instance_id": str(instance_id),
            "schedule_progress": max(0.0, min(1.0, float(progress))),
            "active": bool(active),
        }
    )
    out[VISUAL_PATCH_RUNTIME_KEY] = tuple(entries)
    return out


__all__ = [
    "VISUAL_PATCH_ARCHITECTURE",
    "VISUAL_PATCH_KIND",
    "VISUAL_PATCH_PROFILES_KEY",
    "VISUAL_PATCH_PROVIDER",
    "VISUAL_PATCH_RUNTIME_KEY",
    "VISUAL_PATCH_SCHEMA_VERSION",
    "append_spectrum_h3_runtime",
    "register_spectrum_h3_profile",
]
