from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from flux_untwist.keyless_h3 import (
    KEYLESS_CONTRACT_KEY,
    KEYLESS_ROUTING_POSITION_DOMAIN_KEY,
    KEYLESS_ROUTING_PREPROCESSORS_KEY,
    KEYLESS_VALUE_DOMAIN_KEY,
    KeylessH3CompatibilityError,
    append_keyless_routing_preprocessor,
    make_keyless_untwist_routing_preprocessor,
    minimax_h3_payload_row_count,
    validate_keyless_h3_contract,
)


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
    provenance_identity = "keyless-test"

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


def _attention(*, fake_qkv: bool = False):
    attention = SimpleNamespace(
        head_dim=128,
        qv_proj=SimpleNamespace(weight=SimpleNamespace(shape=(14336, 5376))),
        q_norm=object(),
        route_norm=object(),
    )
    if fake_qkv:
        attention.qkv_proj = object()
    return attention


def _keyless_model(*, contract=None, fake_qkv: bool = False):
    model = SimpleNamespace(
        blocks=[SimpleNamespace(attn=_attention(fake_qkv=fake_qkv)) for _ in range(50)],
        token_refiner=SimpleNamespace(blocks=[object(), object()]),
    )
    setattr(model, KEYLESS_CONTRACT_KEY, Contract() if contract is None else contract)
    return model


def _cfg(**overrides):
    value = {
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
    value.update(overrides)
    return value


def test_keyless_contract_accepts_real_qv_and_rejects_fake_qkv() -> None:
    model = _keyless_model()
    assert validate_keyless_h3_contract(model) is getattr(model, KEYLESS_CONTRACT_KEY)

    with pytest.raises(KeylessH3CompatibilityError, match="fake/dead K"):
        validate_keyless_h3_contract(_keyless_model(fake_qkv=True))


def test_present_malformed_contract_fails_closed_instead_of_native_fallback() -> None:
    contract = Contract()
    contract.routing_source = "key"
    with pytest.raises(KeylessH3CompatibilityError, match="routing_source"):
        validate_keyless_h3_contract(_keyless_model(contract=contract))


def test_payload_row_count_uses_full_packed_layout_and_rejects_inconsistent_bounds() -> None:
    payload = {
        "layout": SimpleNamespace(
            seq_len=12,
            segments=[(0, 2, "text"), (2, 4, "ref_img"), (4, 12, "video")],
        )
    }
    assert minimax_h3_payload_row_count(payload) == 12
    payload["layout"] = SimpleNamespace(
        seq_len=10,
        segments=[(0, 12, "video")],
    )
    with pytest.raises(RuntimeError, match="extend beyond"):
        minimax_h3_payload_row_count(payload)


def test_keyless_preprocessor_scales_only_logical_route_reference_rows() -> None:
    preprocessor = make_keyless_untwist_routing_preprocessor(
        _cfg(),
        instance_id="untwist-keyless-test",
        expected_rows=6,
    )
    route = torch.ones((6, 2, 8), dtype=torch.float32)
    retrieval_v = torch.arange(route.numel(), dtype=route.dtype).reshape_as(route)
    retrieval_before = retrieval_v.clone()

    out = preprocessor(route)

    expected = torch.tensor([0.5, 1.5, 0.5, 1.5, 1.0, 1.0, 1.0, 1.0])
    torch.testing.assert_close(out[:2], route[:2])
    torch.testing.assert_close(
        out[2:4],
        expected.view(1, 1, 8).expand(2, 2, 8),
    )
    torch.testing.assert_close(out[4:], route[4:])
    torch.testing.assert_close(route, torch.ones_like(route))
    torch.testing.assert_close(retrieval_v, retrieval_before)


def test_keyless_preprocessor_maps_selected_reordered_rows_by_routing_positions() -> None:
    preprocessor = make_keyless_untwist_routing_preprocessor(
        _cfg(),
        instance_id="untwist-keyless-test",
        expected_rows=6,
    )
    route = torch.ones((4, 2, 8), dtype=torch.float32)
    value_domain = SimpleNamespace(indices=(5, 3, 1, 2), start=None, stop=None)
    routing_domain = SimpleNamespace(indices=(5, 3, 1, 2), start=None, stop=None)

    out = preprocessor.apply_domain(route, value_domain, routing_domain)

    expected_scale = torch.tensor([0.5, 1.5, 0.5, 1.5, 1.0, 1.0, 1.0, 1.0])
    torch.testing.assert_close(out[0], route[0])
    torch.testing.assert_close(
        out[1],
        expected_scale.view(1, 8).expand(2, 8),
    )
    torch.testing.assert_close(out[2], route[2])
    torch.testing.assert_close(
        out[3],
        expected_scale.view(1, 8).expand(2, 8),
    )
    torch.testing.assert_close(route, torch.ones_like(route))


def test_keyless_preprocessor_maps_slice_domain_to_original_reference_rows() -> None:
    preprocessor = make_keyless_untwist_routing_preprocessor(
        _cfg(reference_ranges=[[3, 5]]),
        instance_id="untwist-keyless-test",
        expected_rows=6,
    )
    route = torch.ones((3, 2, 8), dtype=torch.float32)
    routing_domain = SimpleNamespace(indices=None, start=2, stop=5)

    out = preprocessor.apply_domain(route, None, routing_domain)

    expected_scale = torch.tensor([0.5, 1.5, 0.5, 1.5, 1.0, 1.0, 1.0, 1.0])
    torch.testing.assert_close(out[0], route[0])
    torch.testing.assert_close(
        out[1:],
        expected_scale.view(1, 1, 8).expand(2, 2, 8),
    )


@pytest.mark.parametrize(
    "routing_domain,match",
    [
        (SimpleNamespace(indices=(0, 1), start=None, stop=None), "2 coordinates"),
        (SimpleNamespace(indices=(0, 6, 1), start=None, stop=None), "outside"),
        (SimpleNamespace(indices=None, start=1, stop=5), "does not match"),
        (SimpleNamespace(indices=None, start=None, stop=None), "does not expose row coordinates"),
    ],
)
def test_keyless_preprocessor_rejects_malformed_selected_domains(
    routing_domain,
    match: str,
) -> None:
    preprocessor = make_keyless_untwist_routing_preprocessor(
        _cfg(),
        instance_id="untwist-keyless-test",
        expected_rows=6,
    )
    with pytest.raises(RuntimeError, match=match):
        preprocessor.apply_domain(
            torch.ones((3, 2, 8)),
            None,
            routing_domain,
        )


def test_keyless_preprocessor_requires_position_domain_for_reduced_rows() -> None:
    preprocessor = make_keyless_untwist_routing_preprocessor(
        _cfg(),
        instance_id="untwist-keyless-test",
        expected_rows=6,
    )
    with pytest.raises(RuntimeError, match="domain is missing after row selection"):
        preprocessor.apply_domain(
            torch.ones((4, 2, 8)),
            SimpleNamespace(indices=(0, 1, 2, 3), start=None, stop=None),
            None,
        )


def test_keyless_preprocessor_identity_binds_dynamic_route_semantics() -> None:
    first = make_keyless_untwist_routing_preprocessor(
        _cfg(progress=0.25), instance_id="instance-a", expected_rows=6
    )
    same = make_keyless_untwist_routing_preprocessor(
        _cfg(progress=0.25), instance_id="instance-a", expected_rows=6
    )
    changed_progress = make_keyless_untwist_routing_preprocessor(
        _cfg(progress=0.5), instance_id="instance-a", expected_rows=6
    )
    changed_ranges = make_keyless_untwist_routing_preprocessor(
        _cfg(reference_ranges=[[1, 3]]), instance_id="instance-a", expected_rows=6
    )

    assert first.identity == same.identity
    assert first.identity.startswith("minimax_h3_untwist_keyless_route_v2:")
    assert first.identity != changed_progress.identity
    assert first.identity != changed_ranges.identity
    assert first.fn.__func__ is same.fn.__func__
    assert callable(first.apply_domain)


def test_append_preserves_preprocessor_order_and_does_not_claim_attention_override() -> None:
    inherited = SimpleNamespace(identity="existing-route")
    preprocessor = make_keyless_untwist_routing_preprocessor(
        _cfg(), instance_id="instance-a", expected_rows=6
    )
    owner = object()
    original = {
        KEYLESS_ROUTING_PREPROCESSORS_KEY: (inherited,),
        "optimized_attention_override": owner,
        "unrelated": "keep",
    }

    out = append_keyless_routing_preprocessor(original, preprocessor)

    assert out is not original
    assert out[KEYLESS_ROUTING_PREPROCESSORS_KEY] == (inherited, preprocessor)
    assert out["optimized_attention_override"] is owner
    assert out["unrelated"] == "keep"
    assert original[KEYLESS_ROUTING_PREPROCESSORS_KEY] == (inherited,)


@pytest.mark.parametrize(
    "domain_key",
    [KEYLESS_VALUE_DOMAIN_KEY, KEYLESS_ROUTING_POSITION_DOMAIN_KEY],
)
def test_explicit_keyless_row_domain_is_preserved_when_untwist_is_appended(
    domain_key: str,
) -> None:
    preprocessor = make_keyless_untwist_routing_preprocessor(
        _cfg(), instance_id="instance-a", expected_rows=6
    )
    domain = object()

    out = append_keyless_routing_preprocessor({domain_key: domain}, preprocessor)

    assert out[domain_key] is domain
    assert out[KEYLESS_ROUTING_PREPROCESSORS_KEY] == (preprocessor,)
