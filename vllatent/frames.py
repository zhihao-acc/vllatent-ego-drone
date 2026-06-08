"""Frame / axis convention helpers (PURE tier) — the #1 foot-gun.

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

STUB at scaffold time; the no-flip unit basis (up->up, down->down,
right->right-of-forward, forward->forward) and the NED->FLU->ENU remap land with
``tests/test_frames.py``. See plans/phase-a-data-and-io-contract.md + docs/io-contract.md.
"""
from __future__ import annotations

QUAT_ORDER_START_ROTATION = "wxyz"  # AerialVLN episode start_rotation quaternion (w-FIRST)
QUAT_ORDER_CANONICAL = "xyzw"       # vllatent internal canonical quaternion order / airsim.Quaternionr

# AerialVLN reference_path stores EULER orientation, not a quaternion (confirmed step 5b).
REFERENCE_PATH_ORIENTATION = "euler_pitch_roll_yaw_rad"  # row = [x, y, z, pitch, roll, yaw]
REFERENCE_PATH_ROW_WIDTH = 6        # 3 position + 3 euler
REFERENCE_PATH_YAW_INDEX = 5        # yaw = row[5] (pitch=roll==0 in AerialVLN)

__all__ = [
    "QUAT_ORDER_START_ROTATION",
    "QUAT_ORDER_CANONICAL",
    "REFERENCE_PATH_ORIENTATION",
    "REFERENCE_PATH_ROW_WIDTH",
    "REFERENCE_PATH_YAW_INDEX",
]
