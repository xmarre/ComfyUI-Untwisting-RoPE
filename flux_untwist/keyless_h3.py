from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

import torch

from .config import schedule_fraction
from .patches import build_h3_frequency_scale_vector, lerp


KEYLESS_CONTRACT_KEY = "minimax_h3_keyless_contract_v1"
KEYLESS_ARCHITECTURE = "h3_keyless_core50_v1"
KEYLESS_ROUTING_PREPROCESSORS_KEY = "minimax_h3_keyless_routing_preprocessors_v1"
KEYLESS_VALUE_DOMAIN_KEY = "minimax_h3_keyless_value_domain_v1"
KEYLESS_ROUTING_POSITION_DOMAIN_KEY = "minimax_h3_keyless_routing_position_domain_v1"

_KEYLESS_CORE_BLOCKS = 50
_KEYLESS_TOKEN_REFINER_BLOCKS = 2
_KEYLESS_HEADS = 56
_KEYLESS_HEAD_DIM = 128
_KEYLESS_INNER_DIM = 7_168
_KEYLESS_HIDDEN_SIZE = 5_376
_KEYLESS_QV_ROWS = 2 * _KEYLESS_INNER_DIM
_KEYLESS_ROPE_POLICY = "h3_split_half_96_v1"
_KEYLESS_QV_ORDER = "q_effective;v"
_KEYLESS_NORM_EPS = 1e-5
_KEYLESS_ROUTE_PREPROCESSOR_VERSION = 1
_REFERENCE_SCOPES = frozenset(
    {"image_only", "image_and_video", "all_visual_including_continuum"}
)


class KeylessH3CompatibilityError(TypeError):
    """An advertised Keyless H3 model does not satisfy Untwist's v1 boundary."""


def _field(contract: Any, name: str) -> Any:
    if not hasattr(contract, name):
        raise KeylessH3CompatibilityError(
            f"{KEYLESS_CONTRACT_KEY} is missing required field {name!r}"
        )
    return getattr(contract, name)


def validate_keyless_h3_contract(model: Any) -> Any | None:
    """Validate the public Keyless v1 contract without importing its package.

    Absence means the ordinary native/mixed-QKV Untwist path remains authoritative.
    A present but malformed contract fails closed rather than falling back to logical
    K interception, because canonical Keyless retrieval must never acquire a fake K.
    """

    if model is None or not hasattr(model, KEYLESS_CONTRACT_KEY):
        return None
    contract = getattr(model, KEYLESS_CONTRACT_KEY)
    expected = {
        "api": 1,
        "architecture": KEYLESS_ARCHITECTURE,
        "core_blocks": _KEYLESS_CORE_BLOCKS,
        "token_refiner": "native_qkv",
        "token_refiner_blocks": _KEYLESS_TOKEN_REFINER_BLOCKS,
        "heads": _KEYLESS_HEADS,
        "head_dim": _KEYLESS_HEAD_DIM,
        "inner_dim": _KEYLESS_INNER_DIM,
        "hidden_size": _KEYLESS_HIDDEN_SIZE,
        "routing_source": "value",
        "retrieval_source": "raw_projected_value",
        "routing_norm": "rmsnorm",
        "routing_norm_epsilon": _KEYLESS_NORM_EPS,
        "rope_policy": _KEYLESS_ROPE_POLICY,
        "qv_order": _KEYLESS_QV_ORDER,
        "projection_attr": "qv_proj",
        "checkpoint_format_version": 1,
    }
    mismatches = []
    for name, wanted in expected.items():
        actual = _field(contract, name)
        if actual != wanted:
            mismatches.append(f"{name}={actual!r} (expected {wanted!r})")
    if mismatches:
        raise KeylessH3CompatibilityError(
            f"unsupported {KEYLESS_CONTRACT_KEY}: " + ", ".join(mismatches)
        )

    blocks = getattr(model, "blocks", None)
    refiners = getattr(getattr(model, "token_refiner", None), "blocks", None)
    try:
        block_count = len(blocks)
        refiner_count = len(refiners)
    except TypeError as exc:
        raise KeylessH3CompatibilityError("Keyless H3 block topology is not sized") from exc
    if block_count != _KEYLESS_CORE_BLOCKS or refiner_count != _KEYLESS_TOKEN_REFINER_BLOCKS:
        raise KeylessH3CompatibilityError(
            "Keyless H3 must expose 50 core blocks and two native-QKV token-refiner blocks"
        )

    for index, block in enumerate(blocks):
        attention = getattr(block, "attn", None)
        if attention is None:
            raise KeylessH3CompatibilityError(f"Keyless block {index} has no attention module")
        if hasattr(attention, "qkv_proj"):
            raise KeylessH3CompatibilityError(
                f"Keyless block {index} exposes qkv_proj; Untwist will not accept a fake/dead K path"
            )
        projection = getattr(attention, "qv_proj", None)
        weight = getattr(projection, "weight", None)
        shape = tuple(int(value) for value in getattr(weight, "shape", ()))
        if shape != (_KEYLESS_QV_ROWS, _KEYLESS_HIDDEN_SIZE):
            raise KeylessH3CompatibilityError(
                f"Keyless block {index} does not expose canonical qv_proj.weight geometry"
            )
        if not hasattr(attention, "q_norm") or not hasattr(attention, "route_norm"):
            raise KeylessH3CompatibilityError(
                f"Keyless block {index} is missing q_norm/route_norm routing semantics"
            )
        if int(getattr(attention, "head_dim", _KEYLESS_HEAD_DIM)) != _KEYLESS_HEAD_DIM:
            raise KeylessH3CompatibilityError(
                f"Keyless block {index} attention head_dim disagrees with the v1 contract"
            )

    identity_fn = getattr(contract, "identity", None)
    if not callable(identity_fn):
        raise KeylessH3CompatibilityError(f"{KEYLESS_CONTRACT_KEY} must expose identity()")
    try:
        identity = tuple(identity_fn())
        hash(identity)
    except (TypeError, ValueError) as exc:
        raise KeylessH3CompatibilityError(
            f"{KEYLESS_CONTRACT_KEY}.identity() must return a hashable tuple-like value"
        ) from exc
    if not identity:
        raise KeylessH3CompatibilityError(f"{KEYLESS_CONTRACT_KEY}.identity() may not be empty")
    return contract


def minimax_h3_payload_row_count(payload: Any) -> int:
    """Resolve the full packed sequence length used by absolute reference ranges."""

    if not isinstance(payload, dict):
        raise RuntimeError("Keyless Untwist requires minimax_payload to bind packed row coordinates")
    layout = payload.get("layout")
    if layout is None:
        raise RuntimeError("Keyless Untwist requires minimax_payload.layout")

    declared = getattr(layout, "seq_len", None)
    try:
        declared_rows = int(declared) if declared is not None else 0
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Keyless Untwist layout.seq_len is invalid") from exc

    segment_rows = 0
    segments = getattr(layout, "segments", None)
    if isinstance(segments, (list, tuple)):
        for segment in segments:
            if not isinstance(segment, (list, tuple)) or len(segment) != 3:
                continue
            try:
                end = int(segment[1])
            except (TypeError, ValueError):
                continue
            segment_rows = max(segment_rows, end)

    if declared_rows <= 0:
        declared_rows = segment_rows
    elif segment_rows > declared_rows:
        raise RuntimeError(
            "Keyless Untwist layout segments extend beyond declared packed sequence length"
        )
    if declared_rows <= 0:
        raise RuntimeError("Keyless Untwist could not resolve the packed sequence length")
    return declared_rows


def _finite_float(cfg: Mapping[str, Any], name: str, default: float) -> float:
    try:
        value = float(cfg.get(name, default))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Keyless Untwist {name} is not numeric") from exc
    if not math.isfinite(value):
        raise RuntimeError(f"Keyless Untwist {name} must be finite")
    return value


def _normalized_ranges(value: Any, expected_rows: int) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise RuntimeError("Keyless Untwist requires at least one selected reference row range")
    out = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise RuntimeError("Keyless Untwist reference range must contain exactly start/end")
        try:
            start, end = int(item[0]), int(item[1])
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Keyless Untwist reference range bounds must be integers") from exc
        if start < 0 or end <= start or end > expected_rows:
            raise RuntimeError(
                f"Keyless Untwist reference range {(start, end)!r} is outside [0,{expected_rows})"
            )
        out.append((start, end))
    return tuple(out)


@dataclass(frozen=True)
class _KeylessUntwistSnapshot:
    instance_id: str
    expected_rows: int
    reference_ranges: tuple[tuple[int, int], ...]
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
    progress: float
    scale_temporal_axis: bool

    @classmethod
    def from_options(
        cls,
        cfg: Mapping[str, Any],
        *,
        instance_id: str,
        expected_rows: int,
    ) -> "_KeylessUntwistSnapshot":
        if not isinstance(cfg, Mapping) or cfg.get("enabled") is not True:
            raise RuntimeError("Keyless Untwist routing preprocessor requires an enabled config")
        expected_rows = int(expected_rows)
        if expected_rows <= 0:
            raise RuntimeError("Keyless Untwist expected row count must be positive")
        scope = str(cfg.get("reference_scope", "image_and_video"))
        if scope not in _REFERENCE_SCOPES:
            raise RuntimeError(f"Keyless Untwist has unsupported reference_scope {scope!r}")
        axis_count = int(cfg.get("rope_axis_count", 0))
        freqs = int(cfg.get("rope_freqs_per_axis", 0))
        if axis_count <= 0 or freqs <= 0:
            raise RuntimeError("Keyless Untwist requires positive RoPE axis/frequency geometry")
        return cls(
            instance_id=str(instance_id),
            expected_rows=expected_rows,
            reference_ranges=_normalized_ranges(cfg.get("reference_ranges"), expected_rows),
            reference_scope=scope,
            rope_axis_count=axis_count,
            rope_freqs_per_axis=freqs,
            high_scale_start=_finite_float(cfg, "high_scale_start", 0.95),
            high_scale_end=_finite_float(cfg, "high_scale_end", 1.0),
            low_scale_start=_finite_float(cfg, "low_scale_start", 1.0),
            low_scale_end=_finite_float(cfg, "low_scale_end", 1.05),
            beta=_finite_float(cfg, "beta", 2.0),
            start_percent=_finite_float(cfg, "start_percent", 0.0),
            end_percent=_finite_float(cfg, "end_percent", 0.9),
            progress=_finite_float(cfg, "progress", 0.0),
            scale_temporal_axis=bool(cfg.get("scale_temporal_axis", False)),
        )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": "minimax_h3_untwist_keyless_routing_preprocessor_v1",
            "version": _KEYLESS_ROUTE_PREPROCESSOR_VERSION,
            "instance_id": self.instance_id,
            "expected_rows": self.expected_rows,
            "reference_ranges": [list(item) for item in self.reference_ranges],
            "reference_scope": self.reference_scope,
            "rope_axis_count": self.rope_axis_count,
            "rope_freqs_per_axis": self.rope_freqs_per_axis,
            "high_scale_start": self.high_scale_start,
            "high_scale_end": self.high_scale_end,
            "low_scale_start": self.low_scale_start,
            "low_scale_end": self.low_scale_end,
            "beta": self.beta,
            "start_percent": self.start_percent,
            "end_percent": self.end_percent,
            "progress": self.progress,
            "scale_temporal_axis": self.scale_temporal_axis,
        }


def _snapshot_identity(snapshot: _KeylessUntwistSnapshot) -> str:
    encoded = json.dumps(
        snapshot.identity_payload(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return f"minimax_h3_untwist_keyless_route_v1:{digest}"


def _apply_keyless_untwist_route(
    route: torch.Tensor,
    snapshot: _KeylessUntwistSnapshot,
) -> torch.Tensor:
    """Scale the already normalized/positioned logical route, never retrieval V."""

    if not torch.is_tensor(route) or route.ndim != 3:
        raise RuntimeError(
            "Keyless Untwist routing preprocessor requires route [rows,heads,head_dim]"
        )
    if int(route.shape[0]) != snapshot.expected_rows:
        raise RuntimeError(
            "Keyless Untwist absolute reference ranges no longer match the current value domain; "
            f"expected {snapshot.expected_rows} rows, got {int(route.shape[0])}. "
            "Selected/reordered Keyless value domains require a domain-aware Untwist contract."
        )
    head_dim = int(route.shape[-1])
    rotated_dim = 2 * snapshot.rope_axis_count * snapshot.rope_freqs_per_axis
    if rotated_dim > head_dim:
        raise RuntimeError(
            f"Keyless Untwist rotated_dim={rotated_dim} exceeds route head_dim={head_dim}"
        )

    active, t = schedule_fraction(
        snapshot.progress,
        snapshot.start_percent,
        snapshot.end_percent,
    )
    if not active:
        return route
    high_scale = lerp(snapshot.high_scale_start, snapshot.high_scale_end, t)
    low_scale = lerp(snapshot.low_scale_start, snapshot.low_scale_end, t)
    if high_scale == 1.0 and low_scale == 1.0:
        return route

    scale = build_h3_frequency_scale_vector(
        head_dim=head_dim,
        rope_axis_count=snapshot.rope_axis_count,
        rope_freqs_per_axis=snapshot.rope_freqs_per_axis,
        high_scale=high_scale,
        low_scale=low_scale,
        beta=snapshot.beta,
        device=route.device,
        dtype=route.dtype,
        scale_temporal_axis=snapshot.scale_temporal_axis,
    ).view(1, 1, head_dim)

    out = route.clone()
    for start, end in snapshot.reference_ranges:
        out[start:end, :, :] = out[start:end, :, :] * scale
    return out


class KeylessUntwistRoutingPreprocessor:
    """Duck-typed Keyless routing preprocessor with stable, semantic identity."""

    def __init__(self, snapshot: _KeylessUntwistSnapshot) -> None:
        self._snapshot = snapshot
        self.identity = _snapshot_identity(snapshot)

    def fn(self, route: torch.Tensor) -> torch.Tensor:
        return _apply_keyless_untwist_route(route, self._snapshot)

    __call__ = fn


def make_keyless_untwist_routing_preprocessor(
    cfg: Mapping[str, Any],
    *,
    instance_id: str,
    expected_rows: int,
) -> KeylessUntwistRoutingPreprocessor:
    snapshot = _KeylessUntwistSnapshot.from_options(
        cfg,
        instance_id=instance_id,
        expected_rows=expected_rows,
    )
    return KeylessUntwistRoutingPreprocessor(snapshot)


def append_keyless_routing_preprocessor(
    transformer_options: Mapping[str, Any],
    preprocessor: KeylessUntwistRoutingPreprocessor,
) -> dict[str, Any]:
    """Append Untwist to the public routing-only chain without stealing attention ownership.

    The current Keyless preprocessor ABI receives only the materialized route tensor.
    Absolute H3 reference ranges therefore cannot be remapped after an explicit value
    or routing-position selection. Such domains fail closed instead of applying scales
    to the wrong physical rows. A provider-created subdomain is additionally caught by
    the preprocessor's exact full-row-count check at execution.
    """

    out = dict(transformer_options)
    explicit_domains = [
        name
        for name in (KEYLESS_VALUE_DOMAIN_KEY, KEYLESS_ROUTING_POSITION_DOMAIN_KEY)
        if out.get(name) is not None
    ]
    if explicit_domains:
        raise RuntimeError(
            "Keyless Untwist routing-only preprocessing does not yet support explicit row domains: "
            + ", ".join(explicit_domains)
        )

    existing = out.get(KEYLESS_ROUTING_PREPROCESSORS_KEY, ())
    if existing is None:
        existing = ()
    if not isinstance(existing, (list, tuple)):
        raise RuntimeError(
            f"{KEYLESS_ROUTING_PREPROCESSORS_KEY} must be a list/tuple for composable routing transforms"
        )
    out[KEYLESS_ROUTING_PREPROCESSORS_KEY] = (*tuple(existing), preprocessor)
    return out


__all__ = [
    "KEYLESS_CONTRACT_KEY",
    "KEYLESS_ROUTING_POSITION_DOMAIN_KEY",
    "KEYLESS_ROUTING_PREPROCESSORS_KEY",
    "KEYLESS_VALUE_DOMAIN_KEY",
    "KeylessH3CompatibilityError",
    "KeylessUntwistRoutingPreprocessor",
    "append_keyless_routing_preprocessor",
    "make_keyless_untwist_routing_preprocessor",
    "minimax_h3_payload_row_count",
    "validate_keyless_h3_contract",
]
