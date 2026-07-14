"""Discrete -> continuous 4-DoF action mapping (PURE tier).

Transcribes the AerialVLN action set + step constants VERBATIM from
``third_party/AirVLN/airsim_plugin/airsim_settings.py`` and reproduces the
ground-truth pose-step arithmetic of ``utils/env_utils.py::getPoseAfterMakeAction``
in **pure numpy** (NO airsim import — the AirSim quaternion<->euler formulas are
reproduced here).

Frame: **AirSim NED, z-DOWN** (``GO_UP`` = -z, ``GO_DOWN`` = +z); pitch/roll forced
to 0 ⇒ effective **4-DoF (x, y, z, yaw)**; lateral moves are **body-relative**
(yaw±90°). The continuous **body-frame** delta is ``(dx, dy, dz, dyaw)`` with
``dyaw`` in **degrees** (matches ``TURN_ANGLE``); distances in metres.

Public API:
  * ``action_to_delta(id) -> (4,) float32`` — the canonical body-frame quantized delta.
  * ``apply_delta(pose, id) -> pose``       — reproduces ``getPoseAfterMakeAction`` (world frame).
  * ``pose_pair_to_body_delta(a, b) -> (4,)`` — inverse: derive the body delta between two
    NED poses (the AerialVLN audit uses this to verify dataset poses vs the quantized deltas).

See ``docs/io-contract.md``. This module originated in the historical Phase-A
data-contract work.
"""
from __future__ import annotations

import math
from enum import IntEnum

import numpy as np

from vllatent.frames import wrap_pi, xyzw_from_yaw, yaw_from_xyzw

# A NED pose: (position (3,) [x,y,z], rotation (4,) quaternion xyzw). See vllatent.frames.
Pose = tuple[np.ndarray, np.ndarray]


class Action(IntEnum):
    """AerialVLN discrete action set — verbatim from airsim_settings._DefaultAirsimActions."""

    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    GO_UP = 4
    GO_DOWN = 5
    MOVE_LEFT = 6
    MOVE_RIGHT = 7


# Step constants — verbatim from airsim_settings._DefaultAirsimActionSettings (metres / degrees).
FORWARD_STEP_SIZE = 5
LEFT_RIGHT_STEP_SIZE = 5
UP_DOWN_STEP_SIZE = 2
TURN_ANGLE = 15  # degrees

# Canonical body-frame delta per action: (dx, dy, dz, dyaw_deg). NED z-down; +y = body-right.
_DELTA_TABLE: dict[Action, tuple[float, float, float, float]] = {
    Action.STOP: (0.0, 0.0, 0.0, 0.0),
    Action.MOVE_FORWARD: (float(FORWARD_STEP_SIZE), 0.0, 0.0, 0.0),
    Action.TURN_LEFT: (0.0, 0.0, 0.0, float(-TURN_ANGLE)),
    Action.TURN_RIGHT: (0.0, 0.0, 0.0, float(+TURN_ANGLE)),
    Action.GO_UP: (0.0, 0.0, float(-UP_DOWN_STEP_SIZE), 0.0),       # NED up = -z
    Action.GO_DOWN: (0.0, 0.0, float(+UP_DOWN_STEP_SIZE), 0.0),     # NED down = +z
    Action.MOVE_LEFT: (0.0, float(-LEFT_RIGHT_STEP_SIZE), 0.0, 0.0),
    Action.MOVE_RIGHT: (0.0, float(+LEFT_RIGHT_STEP_SIZE), 0.0, 0.0),
}


def action_to_delta(action_id: int) -> np.ndarray:
    """Canonical body-frame quantized delta ``(dx, dy, dz, dyaw_deg)`` for a discrete action id."""
    return np.array(_DELTA_TABLE[Action(int(action_id))], dtype=np.float32)


def apply_delta(pose: Pose, action_id: int) -> Pose:
    """Reproduce ``env_utils.getPoseAfterMakeAction(pose, action)`` in pure numpy.

    ``pose`` = (position (3,) NED, rotation (4,) xyzw). Returns the new pose. Pitch/roll
    are forced to 0 (as in env_utils); only yaw is read from the input rotation.
    """
    position = np.asarray(pose[0], dtype=float).copy()
    rotation = np.asarray(pose[1], dtype=float).copy()
    action = Action(int(action_id))
    yaw = yaw_from_xyzw(rotation)  # pitch = roll = 0 (env_utils forces them)

    new_position = position.copy()
    new_rotation = rotation.copy()

    if action == Action.MOVE_FORWARD:
        unit = np.array([math.cos(yaw), math.sin(yaw), 0.0])  # pitch=0 -> unit_z = 0
        new_position = position + unit * FORWARD_STEP_SIZE
    elif action == Action.TURN_LEFT:
        new_yaw = yaw - math.radians(TURN_ANGLE)
        if math.degrees(new_yaw) < -180:
            new_yaw = math.radians(360) + new_yaw
        new_rotation = xyzw_from_yaw(new_yaw)
    elif action == Action.TURN_RIGHT:
        new_yaw = yaw + math.radians(TURN_ANGLE)
        if math.degrees(new_yaw) > 180:
            new_yaw = math.radians(-360) + new_yaw
        new_rotation = xyzw_from_yaw(new_yaw)
    elif action == Action.GO_UP:
        new_position = position + np.array([0.0, 0.0, -1.0]) * UP_DOWN_STEP_SIZE
    elif action == Action.GO_DOWN:
        new_position = position + np.array([0.0, 0.0, -1.0]) * UP_DOWN_STEP_SIZE * (-1)
    elif action in (Action.MOVE_LEFT, Action.MOVE_RIGHT):
        # Body-lateral: env_utils builds a unit vector at (yaw + 90 deg); LEFT scales by -1.
        # radians(degrees(yaw) + 90) == yaw + pi/2 (kept verbatim to match env_utils arithmetic).
        unit_x = math.cos(math.radians(math.degrees(yaw) + 90))
        unit_y = math.sin(math.radians(math.degrees(yaw) + 90))
        unit = np.array([unit_x, unit_y, 0.0])
        sign = -1 if action == Action.MOVE_LEFT else 1
        new_position = position + unit * LEFT_RIGHT_STEP_SIZE * sign
    # STOP / unknown: identity (new_position / new_rotation already copies).

    return new_position, new_rotation


def pose_pair_to_body_delta(pose_before: Pose, pose_after: Pose) -> np.ndarray:
    """Derive the body-frame delta ``(dx, dy, dz, dyaw_deg)`` between two NED poses.

    Inverse of :func:`apply_delta`: rotate the world position difference into the BEFORE
    pose's body frame (by -yaw) and wrap the yaw difference into (-180, 180]. The AerialVLN
    audit compares this against :func:`action_to_delta` to confirm the dataset poses
    reproduce the quantized action deltas.
    """
    pos_before = np.asarray(pose_before[0], dtype=float)
    pos_after = np.asarray(pose_after[0], dtype=float)
    yaw_before = yaw_from_xyzw(np.asarray(pose_before[1], dtype=float))
    yaw_after = yaw_from_xyzw(np.asarray(pose_after[1], dtype=float))

    dpos_world = pos_after - pos_before
    cos_y, sin_y = math.cos(yaw_before), math.sin(yaw_before)
    body_x = cos_y * dpos_world[0] + sin_y * dpos_world[1]
    body_y = -sin_y * dpos_world[0] + cos_y * dpos_world[1]
    body_z = dpos_world[2]
    dyaw_deg = math.degrees(wrap_pi(yaw_after - yaw_before))
    return np.array([body_x, body_y, body_z, dyaw_deg], dtype=np.float32)


__all__ = [
    "Action",
    "FORWARD_STEP_SIZE",
    "LEFT_RIGHT_STEP_SIZE",
    "UP_DOWN_STEP_SIZE",
    "TURN_ANGLE",
    "Pose",
    "action_to_delta",
    "apply_delta",
    "pose_pair_to_body_delta",
]
