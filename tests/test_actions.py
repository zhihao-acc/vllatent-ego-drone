"""Step-4 tests: discrete -> continuous 4-DoF action mapping (PURE tier).

Asserts each ``action_to_delta`` against the AirVLN enum + step constants, checks
``apply_delta`` reproduces ``env_utils.getPoseAfterMakeAction`` at known starts, and
round-trips ``apply_delta`` -> ``pose_pair_to_body_delta`` == ``action_to_delta`` for
several yaws (this also pre-validates the step-5 audit derived-Δ check).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from vllatent.actions import (
    FORWARD_STEP_SIZE,
    LEFT_RIGHT_STEP_SIZE,
    TURN_ANGLE,
    UP_DOWN_STEP_SIZE,
    Action,
    action_to_delta,
    apply_delta,
    pose_pair_to_body_delta,
)

POS0 = np.array([10.0, 20.0, -30.0])


def _quat_yaw(deg: float) -> np.ndarray:
    """xyzw quaternion for a yaw-only rotation of ``deg`` degrees."""
    y = math.radians(deg)
    return np.array([0.0, 0.0, math.sin(y / 2.0), math.cos(y / 2.0)], dtype=float)


# --- enum + constants transcription ---

def test_enum_values_match_airvln() -> None:
    assert (Action.STOP, Action.MOVE_FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT) == (0, 1, 2, 3)
    assert (Action.GO_UP, Action.GO_DOWN, Action.MOVE_LEFT, Action.MOVE_RIGHT) == (4, 5, 6, 7)
    assert (FORWARD_STEP_SIZE, LEFT_RIGHT_STEP_SIZE, UP_DOWN_STEP_SIZE, TURN_ANGLE) == (5, 5, 2, 15)


@pytest.mark.parametrize(
    "action, expected",
    [
        (Action.STOP, (0, 0, 0, 0)),
        (Action.MOVE_FORWARD, (5, 0, 0, 0)),
        (Action.TURN_LEFT, (0, 0, 0, -15)),
        (Action.TURN_RIGHT, (0, 0, 0, 15)),
        (Action.GO_UP, (0, 0, -2, 0)),      # NED up = -z
        (Action.GO_DOWN, (0, 0, 2, 0)),     # NED down = +z
        (Action.MOVE_LEFT, (0, -5, 0, 0)),  # body-right is +y
        (Action.MOVE_RIGHT, (0, 5, 0, 0)),
    ],
)
def test_action_to_delta(action: Action, expected: tuple[int, int, int, int]) -> None:
    d = action_to_delta(action)
    assert d.dtype == np.float32
    assert d.shape == (4,)
    np.testing.assert_allclose(d, np.array(expected, dtype=np.float32))


def test_action_to_delta_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        action_to_delta(8)


# --- apply_delta reproduces env_utils at known starts ---

def test_forward_is_planar_and_along_body_x_at_yaw0() -> None:
    pos, _ = apply_delta((POS0, _quat_yaw(0)), Action.MOVE_FORWARD)
    # env_utils asserts unit_z == 0: forward never changes z.
    assert pos[2] == POS0[2]
    np.testing.assert_allclose(pos, POS0 + np.array([FORWARD_STEP_SIZE, 0, 0]), atol=1e-9)


def test_forward_follows_yaw() -> None:
    # Facing +y (yaw=90): forward moves along +y.
    pos, _ = apply_delta((POS0, _quat_yaw(90)), Action.MOVE_FORWARD)
    np.testing.assert_allclose(pos, POS0 + np.array([0, FORWARD_STEP_SIZE, 0]), atol=1e-9)


def test_lateral_sign_at_yaw0() -> None:
    left, _ = apply_delta((POS0, _quat_yaw(0)), Action.MOVE_LEFT)
    right, _ = apply_delta((POS0, _quat_yaw(0)), Action.MOVE_RIGHT)
    np.testing.assert_allclose(left, POS0 + np.array([0, -LEFT_RIGHT_STEP_SIZE, 0]), atol=1e-9)
    np.testing.assert_allclose(right, POS0 + np.array([0, LEFT_RIGHT_STEP_SIZE, 0]), atol=1e-9)


def test_z_sign_up_down() -> None:
    up, _ = apply_delta((POS0, _quat_yaw(0)), Action.GO_UP)
    down, _ = apply_delta((POS0, _quat_yaw(0)), Action.GO_DOWN)
    assert up[2] == POS0[2] - UP_DOWN_STEP_SIZE     # up = more negative z (NED)
    assert down[2] == POS0[2] + UP_DOWN_STEP_SIZE


def test_turn_yaw_sign() -> None:
    before = (POS0, _quat_yaw(0))
    left = apply_delta(before, Action.TURN_LEFT)
    right = apply_delta(before, Action.TURN_RIGHT)
    # Position unchanged; yaw moves -15 / +15 deg.
    np.testing.assert_allclose(left[0], POS0, atol=1e-9)
    np.testing.assert_allclose(pose_pair_to_body_delta(before, left), [0, 0, 0, -TURN_ANGLE], atol=1e-3)
    np.testing.assert_allclose(pose_pair_to_body_delta(before, right), [0, 0, 0, TURN_ANGLE], atol=1e-3)


def test_stop_is_identity() -> None:
    before = (POS0, _quat_yaw(37))
    pos, rot = apply_delta(before, Action.STOP)
    np.testing.assert_allclose(pos, POS0)
    np.testing.assert_allclose(rot, _quat_yaw(37))


# --- round-trip: apply_delta -> derive == quantized delta (pre-validates step-5 audit) ---

@pytest.mark.parametrize("yaw_deg", [0, 30, 90, 150, -45, -120])
@pytest.mark.parametrize("action", list(Action))
def test_derived_delta_matches_quantized(yaw_deg: float, action: Action) -> None:
    before = (POS0, _quat_yaw(yaw_deg))
    after = apply_delta(before, action)
    derived = pose_pair_to_body_delta(before, after)
    np.testing.assert_allclose(derived, action_to_delta(action), atol=1e-3)
