"""Tests for B3 6-D plan token contract."""
from __future__ import annotations

import numpy as np
import pytest

from vllatent.plan_tokens import (
    PLAN_TOKEN_DIM,
    PLAN_TOKEN_FIELDS,
    plan_tokens_from_deltas,
)
from vllatent.schemas import DELTA_DTYPE


def _deltas() -> np.ndarray:
    return np.array([
        [1.0, 0.0, 0.0, 10.0],
        [0.0, 2.0, 0.0, -20.0],
        [0.0, 0.0, 0.5, 200.0],
    ], dtype=DELTA_DTYPE)


def test_contract_fields() -> None:
    assert PLAN_TOKEN_DIM == 6
    assert PLAN_TOKEN_FIELDS == (
        "unit_dir_x",
        "unit_dir_y",
        "unit_dir_z",
        "log_speed_ratio",
        "yaw_rate_norm",
        "valid",
    )


def test_plan_token_shape_and_valid_mask() -> None:
    result = plan_tokens_from_deltas(_deltas(), dt_seconds=0.2, reference_speed=5.0)
    assert result.tokens.shape == (3, PLAN_TOKEN_DIM)
    assert result.tokens.dtype == DELTA_DTYPE
    np.testing.assert_array_equal(result.tokens[:, 5].astype(bool), result.valid_mask)
    assert np.all(result.valid_mask)


def test_scale_free_translation_invariance() -> None:
    base = _deltas()
    scaled = base.copy()
    scaled[:, :3] *= 9.0
    ref = 5.0
    a = plan_tokens_from_deltas(base, dt_seconds=0.2, reference_speed=ref)
    b = plan_tokens_from_deltas(scaled, dt_seconds=0.2, reference_speed=ref * 9.0)
    np.testing.assert_allclose(a.tokens[:, :4], b.tokens[:, :4], atol=1e-6)
    np.testing.assert_allclose(a.tokens[:, 4:], b.tokens[:, 4:], atol=1e-6)


def test_yaw_rate_norm_clips() -> None:
    deltas = np.array([[0.1, 0.0, 0.0, 1000.0]], dtype=DELTA_DTYPE)
    result = plan_tokens_from_deltas(
        deltas,
        dt_seconds=1.0,
        reference_speed=0.1,
        yaw_rate_cap_deg_s=180.0,
    )
    assert result.tokens[0, 4] == pytest.approx(1.0)


def test_valid_combines_moving_speed_and_vo_confidence() -> None:
    deltas = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [100000.0, 0.0, 0.0, 0.0],
    ], dtype=DELTA_DTYPE)
    result = plan_tokens_from_deltas(
        deltas,
        dt_seconds=1.0,
        reference_speed=1.0,
        vo_confidence=np.array([1.0, 0.1, 1.0], dtype=DELTA_DTYPE),
        vo_conf_threshold=0.3,
    )
    assert result.moving_mask.tolist() == [False, True, True]
    assert result.vo_valid_mask.tolist() == [True, False, True]
    assert result.speed_valid_mask.tolist() == [False, True, False]
    assert result.valid_mask.tolist() == [False, False, False]
    np.testing.assert_allclose(result.tokens[:, 5], 0.0)


def test_rejects_bad_dt() -> None:
    with pytest.raises(ValueError, match="dt_seconds"):
        plan_tokens_from_deltas(_deltas(), dt_seconds=0.0)
