"""Frame remap unit tests (PURE tier) — Phase-A step A5.2 (review M2).

The #1 foot-gun is re-deriving the NED->FLU->ENU remap by hand and silently flipping an
axis. This hard CI gate pins the **no-flip basis** against fly0's ``geometry/frames.py``
SEMANTICS (fly0 is NEVER imported — A-C are standalone):

    up -> up,  down -> down,  forward -> forward,  right -> right-of-forward.

It also checks the rotation matrices are PROPER rotations (det = +1, orthonormal) and that
the 4-DoF waypoint body remap = (dx, -dy, -dz, -dyaw) preserves forward/magnitude, is its
own inverse, and keeps the AerialVLN action semantics (GO_UP stays up, MOVE_FORWARD stays
forward). The live closed-loop world-ENU handoff (needs odom) is Phase D.
"""
from __future__ import annotations

import numpy as np

from vllatent.actions import Action, action_to_delta
from vllatent.frames import (
    R_ENU_FROM_NED,
    R_FLU_FROM_FRD,
    ned_frd_to_flu,
    ned_to_enu,
    remap_waypoint_ned_body_to_flu,
)

# Body-frame (FRD) semantic unit directions and their FLU images.
FRD_FORWARD = np.array([1.0, 0.0, 0.0])
FRD_RIGHT = np.array([0.0, 1.0, 0.0])
FRD_DOWN = np.array([0.0, 0.0, 1.0])
FRD_UP = -FRD_DOWN


def test_rotation_matrices_are_proper_rotations() -> None:
    # Arrange / Act / Assert: det = +1 (no reflection) and orthonormal (R @ R.T == I).
    for r in (R_FLU_FROM_FRD, R_ENU_FROM_NED):
        assert np.isclose(np.linalg.det(r), 1.0)
        assert np.allclose(r @ r.T, np.eye(3))


def test_body_no_flip_basis_frd_to_flu() -> None:
    # forward -> forward (+x); up -> up (+z); down -> down (-z); right -> right-of-forward (-y in FLU).
    assert np.allclose(ned_frd_to_flu(FRD_FORWARD), [1.0, 0.0, 0.0])  # forward -> forward
    assert np.allclose(ned_frd_to_flu(FRD_UP), [0.0, 0.0, 1.0])       # up -> up (FLU +z)
    assert np.allclose(ned_frd_to_flu(FRD_DOWN), [0.0, 0.0, -1.0])    # down -> down (FLU -z)
    assert np.allclose(ned_frd_to_flu(FRD_RIGHT), [0.0, -1.0, 0.0])   # right -> right-of-forward (-FLU left)


def test_world_no_flip_basis_ned_to_enu() -> None:
    # North/East stay their compass direction; up -> up, down -> down (no vertical flip).
    assert np.allclose(ned_to_enu([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0])   # NED north -> ENU +y (north)
    assert np.allclose(ned_to_enu([0.0, 1.0, 0.0]), [1.0, 0.0, 0.0])   # NED east  -> ENU +x (east)
    assert np.allclose(ned_to_enu([0.0, 0.0, -1.0]), [0.0, 0.0, 1.0])  # NED up    -> ENU up
    assert np.allclose(ned_to_enu([0.0, 0.0, 1.0]), [0.0, 0.0, -1.0])  # NED down  -> ENU down


def test_waypoint_remap_flips_y_z_yaw_keeps_forward() -> None:
    # Arrange
    wp_ned = np.array([5.0, 3.0, -2.0, 15.0])  # forward, right, up(-z NED), turn-right(+deg NED)
    # Act
    wp_flu = remap_waypoint_ned_body_to_flu(wp_ned)
    # Assert: dx kept; dy, dz, dyaw negated.
    assert np.allclose(wp_flu, [5.0, -3.0, 2.0, -15.0])


def test_waypoint_remap_is_its_own_inverse() -> None:
    wp = np.array([1.0, -2.0, 3.0, -7.0])
    assert np.allclose(remap_waypoint_ned_body_to_flu(remap_waypoint_ned_body_to_flu(wp)), wp)


def test_waypoint_remap_preserves_translation_magnitude() -> None:
    wp = np.array([5.0, 3.0, -2.0, 15.0])
    out = remap_waypoint_ned_body_to_flu(wp)
    assert np.isclose(np.linalg.norm(out[0:3]), np.linalg.norm(wp[0:3]))


def test_action_semantics_survive_remap_no_flip() -> None:
    # GO_UP (NED -z) must remain UP (FLU +z); MOVE_FORWARD must remain forward (+x); a turn negates.
    up_flu = remap_waypoint_ned_body_to_flu(action_to_delta(Action.GO_UP))
    assert up_flu[2] > 0.0  # up stays up
    fwd_flu = remap_waypoint_ned_body_to_flu(action_to_delta(Action.MOVE_FORWARD))
    assert fwd_flu[0] > 0.0 and np.isclose(fwd_flu[1], 0.0) and np.isclose(fwd_flu[2], 0.0)
    down_flu = remap_waypoint_ned_body_to_flu(action_to_delta(Action.GO_DOWN))
    assert down_flu[2] < 0.0  # down stays down
    right_flu = remap_waypoint_ned_body_to_flu(action_to_delta(Action.MOVE_RIGHT))
    assert right_flu[1] < 0.0  # body-right -> FLU -y (right-of-forward)
