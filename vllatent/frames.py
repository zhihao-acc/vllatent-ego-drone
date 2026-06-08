"""Frame / axis convention helpers (PURE tier) — the #1 foot-gun.

The waypoint head output is AirSim-NED body, yaw-only. For the Phase-D EGO-Planner
seam it is remapped NED-body -> fly0-FLU -> world ENU. In Phases A-C this remap is
re-derived here and unit-tested against fly0's ``geometry/frames.py`` SEMANTICS
(R_FLU_FROM_FRD, R_ENU_FROM_NED) but fly0 is NEVER imported (A-C are standalone).

Also pins the quaternion-order constants (the audit + render foot-gun):
    AerialVLN start_rotation : [w, x, y, z]  (w-FIRST)
    AerialVLN reference_path : [x, y, z, qx, qy, qz, qw]  (xyzw, w-LAST)
    airsim.Quaternionr(x, y, z, w) : xyzw
Canonical internal order: xyzw.

STUB at scaffold time; the no-flip unit basis (up->up, down->down,
right->right-of-forward, forward->forward) and the NED->FLU->ENU remap land with
``tests/test_frames.py``. See plans/phase-a-data-and-io-contract.md + docs/io-contract.md.
"""
from __future__ import annotations

QUAT_ORDER_START_ROTATION = "wxyz"  # AerialVLN episode start_rotation
QUAT_ORDER_REFERENCE_PATH = "xyzw"  # AerialVLN reference_path poses / airsim.Quaternionr
QUAT_ORDER_CANONICAL = "xyzw"       # vllatent internal canonical order

__all__ = [
    "QUAT_ORDER_START_ROTATION",
    "QUAT_ORDER_REFERENCE_PATH",
    "QUAT_ORDER_CANONICAL",
]
