"""Discrete -> continuous 4-DoF action mapping (PURE tier) — Phase-A step 4.

Transcribes the AerialVLN action set + step constants VERBATIM from
``third_party/AirVLN/airsim_plugin/airsim_settings.py`` and reproduces the
ground-truth pose-step arithmetic of ``utils/env_utils.py::getPoseAfterMakeAction``
in pure numpy (NO airsim import). STUB at scaffold time; implemented in step 4.

Action enum (AirsimActions):
    STOP=0, MOVE_FORWARD=1, TURN_LEFT=2, TURN_RIGHT=3,
    GO_UP=4, GO_DOWN=5, MOVE_LEFT=6, MOVE_RIGHT=7
Step constants: FORWARD/LEFT/RIGHT = 5 m, UP_DOWN = 2 m, TURN = 15 deg.
Frame = AirSim NED, z-DOWN (GO_UP = -z, GO_DOWN = +z); yaw-only (pitch=roll=0).

4-DoF delta = (dx, dy, dz, dyaw) in AirSim-NED body frame:
    MOVE_FORWARD = (+5, 0, 0, 0)
    MOVE_LEFT / MOVE_RIGHT = +/-5 body-lateral (yaw +/- 90 deg)
    GO_UP = (0, 0, -2, 0)   GO_DOWN = (0, 0, +2, 0)
    TURN_LEFT = (0, 0, 0, -15 deg)   TURN_RIGHT = (0, 0, 0, +15 deg)
    STOP = identity

See plans/phase-a-data-and-io-contract.md step 4.
"""
from __future__ import annotations

# Implemented in Phase-A step 4 (action_to_delta, apply_delta, the enum + constants).
__all__: list[str] = []
