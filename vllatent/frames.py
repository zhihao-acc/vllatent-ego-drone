"""Frame / axis conventions + quaternion primitives (PURE tier) — the #1 foot-gun.

The waypoint head output is AirSim-NED body, yaw-only. For the Phase-D EGO-Planner
seam it is remapped NED-body -> fly0-FLU -> world ENU. In Phases A-C this remap is
re-derived here and unit-tested against fly0's ``geometry/frames.py`` SEMANTICS
(R_FLU_FROM_FRD, R_ENU_FROM_NED) but fly0 is NEVER imported (A-C are standalone).

Pins the orientation conventions (the audit + render foot-gun). CONFIRMED against the
real AerialVLN dataset (step 5b) — the two orientation fields use DIFFERENT formats:
    AerialVLN start_rotation : [w, x, y, z]  QUATERNION, w-FIRST
    airsim.Quaternionr(x, y, z, w)           : xyzw
    Canonical internal quaternion order      : xyzw
    AerialVLN reference_path row             : [x, y, z, pitch, roll, yaw]  EULER radians
        -- 6-wide (NOT a 7-wide quaternion); pitch == roll == 0 (4-DoF); yaw = row[5].

This module OWNS the project's quaternion/yaw primitives (``yaw_from_xyzw``,
``xyzw_from_yaw``, ``wrap_pi``, ``reorder_wxyz_to_xyzw``); ``actions.py`` and ``audit.py``
import them from here so the frame/quaternion concern lives in one place — no private
cross-module imports (A5.1 / review M1). The no-flip unit basis (up->up, down->down,
right->right-of-forward, forward->forward) and the NED->FLU->ENU remap math are
implemented here and pinned by ``tests/test_frames.py`` (A5.2); the live closed-loop
world-ENU handoff (needs odom) is Phase D. See docs/io-contract.md + plans/phase-a5-replan-postpivot.md.
"""
from __future__ import annotations

import math

import numpy as np

QUAT_ORDER_START_ROTATION = "wxyz"  # AerialVLN episode start_rotation quaternion (w-FIRST)
QUAT_ORDER_CANONICAL = "xyzw"       # vllatent internal canonical quaternion order / airsim.Quaternionr

# AerialVLN reference_path stores EULER orientation, not a quaternion (confirmed step 5b).
REFERENCE_PATH_ORIENTATION = "euler_pitch_roll_yaw_rad"  # row = [x, y, z, pitch, roll, yaw]
REFERENCE_PATH_ROW_WIDTH = 6        # 3 position + 3 euler
REFERENCE_PATH_YAW_INDEX = 5        # yaw = row[5] (pitch=roll==0 in AerialVLN)


def yaw_from_xyzw(q: np.ndarray) -> float:
    """Yaw (radians) from an xyzw quaternion — reproduces airsim.to_eularian_angles (z-axis)."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    ysqr = y * y
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (ysqr + z * z)
    return math.atan2(t3, t4)


def xyzw_from_yaw(yaw: float) -> np.ndarray:
    """xyzw quaternion for a yaw-only rotation — reproduces airsim.to_quaternion(0, 0, yaw)."""
    return np.array([0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5)], dtype=float)


def wrap_pi(angle: float) -> float:
    """Wrap a radian angle to (-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def reorder_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """Reorder a w-FIRST ``[w, x, y, z]`` quaternion to canonical ``xyzw`` (foot-gun #1).

    AerialVLN ``start_rotation`` is w-FIRST; airsim.Quaternionr / our canonical order is xyzw —
    AirVLN's ``env.py`` builds ``Quaternionr(x=sr[1], y=sr[2], z=sr[3], w=sr[0])``.
    """
    q = np.asarray(q, dtype=float)
    return np.array([q[1], q[2], q[3], q[0]], dtype=float)


# --- NED -> FLU -> world-ENU remap (the #1 foot-gun) -----------------------------------
# Re-derived against fly0's geometry/frames.py SEMANTICS; fly0 is NEVER imported (A-C standalone).
# Body frames:  AirSim NED body = FRD (Forward +x, Right +y, Down +z).
#               fly0 body       = FLU (Forward +x, Left  +y, Up   +z).
# World frames: AirSim world = NED (North +x, East +y, Down +z).
#               fly0  world  = ENU (East  +x, North +y, Up  +z).
# Both matrices are PROPER rotations (det = +1): no improper flip / handedness change.

R_FLU_FROM_FRD = np.array(
    [[1.0, 0.0, 0.0],    # FLU forward = FRD forward
     [0.0, -1.0, 0.0],   # FLU left    = -FRD right   (a body-right command stays right-of-forward)
     [0.0, 0.0, -1.0]],  # FLU up      = -FRD down
    dtype=float,
)

R_ENU_FROM_NED = np.array(
    [[0.0, 1.0, 0.0],    # ENU east  = NED east  (y)
     [1.0, 0.0, 0.0],    # ENU north = NED north (x)
     [0.0, 0.0, -1.0]],  # ENU up    = -NED down
    dtype=float,
)


def ned_frd_to_flu(vec_frd: np.ndarray) -> np.ndarray:
    """Rotate a body-frame 3-vector from AirSim NED-body (FRD) to fly0-body (FLU)."""
    return R_FLU_FROM_FRD @ np.asarray(vec_frd, dtype=float)


def ned_to_enu(vec_ned: np.ndarray) -> np.ndarray:
    """Rotate a world-frame 3-vector from AirSim world NED to fly0 world ENU."""
    return R_ENU_FROM_NED @ np.asarray(vec_ned, dtype=float)


def remap_waypoint_ned_body_to_flu(waypoint_4dof: np.ndarray) -> np.ndarray:
    """Remap a 4-DoF waypoint delta ``(dx, dy, dz, dyaw)`` from NED-body (FRD) to FLU-body.

    Stage 1 of the io-contract seam (d): the body-frame axis remap = ``(dx, -dy, -dz, -dyaw)``.
    Forward and magnitude are preserved; right/down/yaw-sense flip with the FRD->FLU axes
    (yaw negates because the rotation axis flips from +down to +up). The ABSOLUTE world-ENU
    handoff (``WaypointHandoff.G_world``, which needs the live odom pose) is **Phase D**.
    """
    w = np.asarray(waypoint_4dof, dtype=float)
    dxyz_flu = ned_frd_to_flu(w[0:3])      # in: NED-FRD body delta -> out: FLU body delta
    dyaw_flu = -float(w[3])                # +down-axis yaw -> +up-axis yaw negates
    return np.array([dxyz_flu[0], dxyz_flu[1], dxyz_flu[2], dyaw_flu], dtype=float)


__all__ = [
    "QUAT_ORDER_START_ROTATION",
    "QUAT_ORDER_CANONICAL",
    "REFERENCE_PATH_ORIENTATION",
    "REFERENCE_PATH_ROW_WIDTH",
    "REFERENCE_PATH_YAW_INDEX",
    "yaw_from_xyzw",
    "xyzw_from_yaw",
    "wrap_pi",
    "reorder_wxyz_to_xyzw",
    "R_FLU_FROM_FRD",
    "R_ENU_FROM_NED",
    "ned_frd_to_flu",
    "ned_to_enu",
    "remap_waypoint_ned_body_to_flu",
]
