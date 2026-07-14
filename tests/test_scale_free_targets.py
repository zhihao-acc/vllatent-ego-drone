"""B2.1 tests for pure scale-free future-action targets."""
from __future__ import annotations

import ast
import dataclasses
import inspect
from pathlib import Path

import numpy as np
import pytest

from vllatent.scale_free_targets import (
    CONTROLLER_MAX_SPEED_MPS,
    SCALE_FREE_ACTION_DIM,
    SCALE_FREE_ACTION_FIELDS,
    SCALE_FREE_LOG_SPEED_CLAMP,
    SCALE_FREE_SPEED_EPS,
    ScaleFreeActionTargets,
    future_deltas_to_scale_free_targets,
    metric_speed_command_from_log_ratio,
    reference_speed_from_deltas,
    scale_free_action_diagnostics,
)
from vllatent.schemas import DELTA_DTYPE, HORIZON


def _future_deltas() -> np.ndarray:
    base = np.array(
        [
            [1.0, 0.0, 0.0, 3.0],
            [0.0, 2.0, 0.0, -7.0],
            [0.0, 0.0, -4.0, 0.0],
            [2.0, 2.0, 1.0, 12.0],
        ],
        dtype=DELTA_DTYPE,
    )
    repeats = (HORIZON + len(base) - 1) // len(base)
    return np.tile(base, (repeats, 1))[:HORIZON]


def test_locked_four_field_contract() -> None:
    assert SCALE_FREE_ACTION_DIM == 4
    assert SCALE_FREE_ACTION_FIELDS == (
        "unit_dir_x",
        "unit_dir_y",
        "unit_dir_z",
        "log_speed_ratio",
    )
    assert "yaw" not in " ".join(SCALE_FREE_ACTION_FIELDS)


def test_target_shape_dtype_and_unit_norm() -> None:
    targets = future_deltas_to_scale_free_targets(_future_deltas(), dt_seconds=0.2)
    assert isinstance(targets, ScaleFreeActionTargets)
    assert targets.actions.shape == (HORIZON, SCALE_FREE_ACTION_DIM)
    assert targets.actions.dtype == DELTA_DTYPE
    assert targets.moving_mask.shape == (HORIZON,)
    assert targets.moving_mask.dtype == np.bool_
    assert targets.speed_valid_mask.shape == (HORIZON,)
    assert targets.speed_valid_mask.dtype == np.bool_
    assert np.all(np.isfinite(targets.actions))

    unit_norms = np.linalg.norm(targets.actions[targets.moving_mask, :3], axis=1)
    np.testing.assert_allclose(unit_norms, np.ones_like(unit_norms), atol=1e-6)


def test_single_delta_returns_vector_action_and_scalar_mask() -> None:
    targets = future_deltas_to_scale_free_targets(np.array([1.0, 0.0, 0.0, 0.0], dtype=DELTA_DTYPE))
    assert targets.actions.shape == (SCALE_FREE_ACTION_DIM,)
    assert targets.moving_mask.shape == ()
    assert targets.speed_valid_mask.shape == ()
    assert bool(targets.moving_mask)
    assert bool(targets.speed_valid_mask)
    np.testing.assert_allclose(targets.actions, [1.0, 0.0, 0.0, 0.0], atol=1e-6)


@pytest.mark.parametrize("scale", [0.01, 0.5, 7.0, 123.0])
def test_uniform_translation_scale_invariance_with_internal_reference(scale: float) -> None:
    base = _future_deltas()
    scaled = base.copy()
    scaled[:, :3] *= scale

    a = future_deltas_to_scale_free_targets(base, dt_seconds=np.full(HORIZON, 0.2, dtype=DELTA_DTYPE))
    b = future_deltas_to_scale_free_targets(scaled, dt_seconds=np.full(HORIZON, 0.2, dtype=DELTA_DTYPE))

    np.testing.assert_array_equal(a.moving_mask, b.moving_mask)
    np.testing.assert_allclose(a.actions, b.actions, atol=1e-6)


def test_uniform_translation_scale_invariance_with_observed_reference() -> None:
    past = np.array([[0.5, 0.0, 0.0, 1.0], [1.5, 0.0, 0.0, -1.0]], dtype=DELTA_DTYPE)
    future = _future_deltas()
    scale = 42.0

    ref = reference_speed_from_deltas(past, dt_seconds=0.2)
    scaled_ref = reference_speed_from_deltas(past * np.array([scale, scale, scale, 1.0]), dt_seconds=0.2)
    scaled_future = future.copy()
    scaled_future[:, :3] *= scale

    a = future_deltas_to_scale_free_targets(future, dt_seconds=0.2, reference_speed=ref)
    b = future_deltas_to_scale_free_targets(scaled_future, dt_seconds=0.2, reference_speed=scaled_ref)

    np.testing.assert_allclose(a.actions, b.actions, atol=1e-6)
    np.testing.assert_array_equal(a.moving_mask, b.moving_mask)


def test_yaw_column_is_not_part_of_b2_1_contract() -> None:
    base = _future_deltas()
    yaw_changed = base.copy()
    yaw_pattern = np.array([999.0, -999.0, 45.0, -45.0], dtype=DELTA_DTYPE)
    yaw_changed[:, 3] = np.resize(yaw_pattern, HORIZON)

    a = future_deltas_to_scale_free_targets(base, dt_seconds=0.2)
    b = future_deltas_to_scale_free_targets(yaw_changed, dt_seconds=0.2)

    np.testing.assert_allclose(a.actions, b.actions, atol=1e-6)
    np.testing.assert_array_equal(a.moving_mask, b.moving_mask)


def test_zero_motion_is_finite_and_masked() -> None:
    targets = future_deltas_to_scale_free_targets(np.zeros((HORIZON, 4), dtype=DELTA_DTYPE), dt_seconds=0.2)
    assert not np.any(targets.moving_mask)
    assert not np.any(targets.speed_valid_mask)
    assert np.all(np.isfinite(targets.actions))
    np.testing.assert_allclose(targets.actions[:, :3], np.tile([1.0, 0.0, 0.0], (HORIZON, 1)))
    np.testing.assert_allclose(targets.actions[:, 3], np.zeros(HORIZON), atol=1e-6)


def test_near_zero_motion_is_finite_and_masked_with_external_reference() -> None:
    deltas = np.array(
        [
            [SCALE_FREE_SPEED_EPS * 0.1, 0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0, 0.0],
        ],
        dtype=DELTA_DTYPE,
    )
    targets = future_deltas_to_scale_free_targets(deltas, dt_seconds=1.0, reference_speed=5.0)
    assert targets.moving_mask.tolist() == [False, True]
    assert targets.speed_valid_mask.tolist() == [False, True]
    assert np.all(np.isfinite(targets.actions))
    np.testing.assert_allclose(targets.actions[0, :3], [1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(targets.actions[1, :3], [0.6, 0.8, 0.0], atol=1e-6)
    assert targets.actions[0, 3] < 0.0


def test_future_target_record_has_no_model_input_fields() -> None:
    targets = future_deltas_to_scale_free_targets(_future_deltas())
    assert {f.name for f in dataclasses.fields(targets)} == {
        "actions",
        "moving_mask",
        "speed_valid_mask",
    }
    forbidden = {"input", "last", "past", "previous", "odom_reference_speed", "reference_speed"}
    assert forbidden.isdisjoint({f.name for f in dataclasses.fields(targets)})


def test_tiny_reference_speed_clips_and_masks_speed_outliers() -> None:
    targets = future_deltas_to_scale_free_targets(
        np.array([[1.0, 0.0, 0.0, 0.0]], dtype=DELTA_DTYPE),
        reference_speed=np.exp(-20.0),
    )
    assert bool(targets.moving_mask[0])
    assert not bool(targets.speed_valid_mask[0])
    assert targets.actions[0, 3] == pytest.approx(SCALE_FREE_LOG_SPEED_CLAMP)

    diag = scale_free_action_diagnostics(
        targets.actions,
        targets.moving_mask,
        targets.speed_valid_mask,
    )
    assert diag.unmasked_log_speed_outliers == 0
    assert diag.speed_valid_count == 0


def test_target_diagnostics_report_percentiles_and_unmasked_outliers() -> None:
    actions = np.zeros((4, SCALE_FREE_ACTION_DIM), dtype=DELTA_DTYPE)
    actions[:, 0] = 1.0
    actions[:, 3] = np.array([-1.0, 0.0, 1.0, SCALE_FREE_LOG_SPEED_CLAMP], dtype=DELTA_DTYPE)
    mask = np.ones(4, dtype=np.bool_)
    speed_mask = np.array([True, True, True, False], dtype=np.bool_)

    diag = scale_free_action_diagnostics(actions, mask, speed_mask)

    assert diag.count == 4
    assert diag.moving_count == 4
    assert diag.speed_valid_count == 3
    assert diag.log_speed_p50 == pytest.approx(0.0)
    assert diag.max_abs_log_speed == pytest.approx(1.0)
    assert diag.unmasked_log_speed_outliers == 0


def test_target_generation_signature_keeps_metric_controller_scale_out_of_labels() -> None:
    sig = inspect.signature(future_deltas_to_scale_free_targets)
    assert "odom_reference_speed_mps" not in sig.parameters
    assert "max_speed_mps" not in sig.parameters
    assert "margin_mps" not in sig.parameters


def test_metric_speed_conversion_is_inference_only_and_clamps_strictly_below_cap() -> None:
    command = metric_speed_command_from_log_ratio(
        np.array([0.0, 10.0], dtype=DELTA_DTYPE),
        odom_reference_speed_mps=np.array([3.0, 3.0], dtype=DELTA_DTYPE),
    )
    assert command.dtype == DELTA_DTYPE
    assert command[0] == pytest.approx(3.0)
    assert command[1] < CONTROLLER_MAX_SPEED_MPS
    assert command[1] == pytest.approx(CONTROLLER_MAX_SPEED_MPS - 1e-3, abs=1e-6)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((HORIZON, 3), dtype=DELTA_DTYPE),
        np.zeros((HORIZON, 5), dtype=DELTA_DTYPE),
        np.array([[np.nan, 0.0, 0.0, 0.0]], dtype=DELTA_DTYPE),
    ],
)
def test_rejects_bad_delta_inputs(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        future_deltas_to_scale_free_targets(bad)


def test_rejects_bad_dt_and_reference() -> None:
    with pytest.raises(ValueError, match="dt_seconds"):
        future_deltas_to_scale_free_targets(_future_deltas(), dt_seconds=0.0)
    with pytest.raises(ValueError, match="reference_speed"):
        future_deltas_to_scale_free_targets(_future_deltas(), reference_speed=-1.0)


def test_module_has_no_heavy_imports() -> None:
    path = Path("vllatent/scale_free_targets.py")
    tree = ast.parse(path.read_text())
    forbidden = {"torch", "transformers", "timm", "airsim", "ultralytics"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in forbidden
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            assert node.module.split(".")[0] not in forbidden
