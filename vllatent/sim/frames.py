"""Float64 named-frame and analytic camera SE(3) contracts for B3-CS.

Transforms are named ``T_target_from_source`` and map source-frame coordinates
into the target frame.  The camera pilot is kinematic: requested body-FRD twists
are integrated analytically, while requested and achieved transforms remain
separate immutable records.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from vllatent.sim.contracts import (
    FIXED_DT_SECONDS,
    HORIZON_STEPS,
    BranchProgram,
    CameraContract,
    default_camera_contract,
)

_ROTATION_ATOL = 1.0e-12
_HOMOGENEOUS_ATOL = 1.0e-12


def _nonempty_string(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name}: expected str, got {type(value).__name__}")
    if not value:
        raise ValueError(f"{name}: expected non-empty string")
    return value


def _f64(name: str, value: object, shape: tuple[int, ...]) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name}: expected np.ndarray, got {type(value).__name__}")
    if value.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {value.shape}")
    if value.dtype != np.dtype("<f8"):
        raise ValueError(f"{name}: expected dtype float64, got {value.dtype}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name}: expected all finite values")
    contiguous = np.array(value, dtype=np.dtype("<f8"), order="C", copy=True)
    contiguous[contiguous == 0.0] = 0.0
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=np.dtype("<f8")).reshape(shape)


def _bool(name: str, value: object, shape: tuple[int, ...]) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name}: expected np.ndarray, got {type(value).__name__}")
    if value.shape != shape or value.dtype != np.bool_:
        raise ValueError(f"{name}: expected shape {shape} and dtype bool")
    contiguous = np.ascontiguousarray(value, dtype=np.bool_)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=np.bool_).reshape(shape)


def _validate_rotation(name: str, value: object) -> np.ndarray:
    rotation = _f64(name, value, (3, 3))
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3, dtype=np.float64),
        rtol=0.0,
        atol=_ROTATION_ATOL,
    ):
        raise ValueError(f"{name}: expected an orthonormal rotation")
    determinant = float(np.linalg.det(rotation))
    if not math.isclose(determinant, 1.0, rel_tol=0.0, abs_tol=_ROTATION_ATOL):
        raise ValueError(f"{name}: expected determinant +1, got {determinant}")
    return rotation


def _validate_transform_array(name: str, value: object, shape: tuple[int, ...]) -> np.ndarray:
    array = _f64(name, value, shape)
    matrices = array.reshape((-1, 4, 4))
    for index, matrix in enumerate(matrices):
        if not np.allclose(
            matrix[3],
            np.array([0.0, 0.0, 0.0, 1.0]),
            rtol=0.0,
            atol=_HOMOGENEOUS_ATOL,
        ):
            raise ValueError(f"{name}[{index}]: expected homogeneous bottom row")
        _validate_rotation(f"{name}[{index}] rotation", matrix[:3, :3].copy())
    return array


@dataclass(frozen=True, eq=False)
class NamedSE3:
    """One immutable row-major float64 ``T_target_from_source``."""

    matrix: np.ndarray
    target_frame: str
    source_frame: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "matrix",
            _validate_transform_array("matrix", self.matrix, (4, 4)),
        )
        object.__setattr__(self, "target_frame", _nonempty_string("target_frame", self.target_frame))
        object.__setattr__(self, "source_frame", _nonempty_string("source_frame", self.source_frame))


def identity_se3(*, target_frame: str, source_frame: str) -> NamedSE3:
    """Return an explicitly named identity transform."""
    return NamedSE3(
        np.eye(4, dtype=np.float64),
        target_frame=target_frame,
        source_frame=source_frame,
    )


def invert_se3(transform: NamedSE3) -> NamedSE3:
    """Invert a named SE(3) transform and swap its frame labels."""
    if not isinstance(transform, NamedSE3):
        raise TypeError(f"transform: expected NamedSE3, got {type(transform).__name__}")
    rotation = transform.matrix[:3, :3]
    translation = transform.matrix[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return NamedSE3(
        inverse,
        target_frame=transform.source_frame,
        source_frame=transform.target_frame,
    )


def compose_se3(outer: NamedSE3, inner: NamedSE3) -> NamedSE3:
    """Return ``outer @ inner`` after checking the shared named frame."""
    if not isinstance(outer, NamedSE3) or not isinstance(inner, NamedSE3):
        raise TypeError("outer and inner: expected NamedSE3")
    if inner.target_frame != outer.source_frame:
        raise ValueError(
            f"frame mismatch: inner.target_frame {inner.target_frame!r} != outer.source_frame {outer.source_frame!r}"
        )
    return NamedSE3(
        outer.matrix @ inner.matrix,
        target_frame=outer.target_frame,
        source_frame=inner.source_frame,
    )


def body_frd_twist_increment(command: np.ndarray, dt_seconds: float) -> np.ndarray:
    """Analytic SE(3) increment for constant body-FRD velocity and yaw rate.

    Positive yaw is rotation about semantic ``+down`` and therefore turns body
    forward toward body right.  Translation follows the exact constant-twist
    exponential, not component-wise Euler integration.
    """
    values = _f64("command", command, (4,))
    if isinstance(dt_seconds, (bool, np.bool_)) or not isinstance(dt_seconds, (int, float, np.integer, np.floating)):
        raise TypeError(f"dt_seconds: expected real scalar, got {type(dt_seconds).__name__}")
    dt = float(dt_seconds)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt_seconds: expected finite > 0, got {dt_seconds}")

    forward, right, down, yaw_rate = (float(item) for item in values)
    theta = yaw_rate * dt
    cosine = math.cos(theta)
    sine = math.sin(theta)
    increment = np.eye(4, dtype=np.float64)
    increment[:3, :3] = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    if abs(yaw_rate) <= 1.0e-15:
        increment[:3, 3] = np.array([forward * dt, right * dt, down * dt], dtype=np.float64)
    else:
        a = sine / yaw_rate
        b = (1.0 - cosine) / yaw_rate
        increment[:3, 3] = np.array(
            [a * forward - b * right, b * forward + a * right, down * dt],
            dtype=np.float64,
        )
    increment.setflags(write=False)
    return increment


def integrate_requested_relative(program: BranchProgram) -> np.ndarray:
    """Integrate one program into requested ``T_rig0_from_rig_t[1..8]``."""
    if not isinstance(program, BranchProgram):
        raise TypeError(f"program: expected BranchProgram, got {type(program).__name__}")
    current = np.eye(4, dtype=np.float64)
    trajectory = np.empty((HORIZON_STEPS, 4, 4), dtype=np.float64)
    for index in range(HORIZON_STEPS):
        increment = body_frd_twist_increment(program.requested_command[index], float(program.dt_seconds[index]))
        current = current @ increment
        trajectory[index] = current
    trajectory.setflags(write=False)
    return trajectory


@dataclass(frozen=True, eq=False)
class CameraTrajectory:
    """Distinct requested and achieved eight-step camera pose records."""

    requested_command: np.ndarray
    dt_seconds: np.ndarray
    record_valid: np.ndarray
    T_world_from_rig: np.ndarray
    requested_T_rig0_from_rig_t: np.ndarray
    achieved_T_rig0_from_rig_t: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "requested_command",
            _f64("requested_command", self.requested_command, (HORIZON_STEPS, 4)),
        )
        object.__setattr__(
            self,
            "dt_seconds",
            _f64("dt_seconds", self.dt_seconds, (HORIZON_STEPS,)),
        )
        if not np.array_equal(
            self.dt_seconds,
            np.full(HORIZON_STEPS, FIXED_DT_SECONDS, dtype=np.float64),
        ):
            raise ValueError("dt_seconds: expected eight fixed 0.2-second rows")
        object.__setattr__(
            self,
            "record_valid",
            _bool("record_valid", self.record_valid, (HORIZON_STEPS,)),
        )
        object.__setattr__(
            self,
            "T_world_from_rig",
            _validate_transform_array("T_world_from_rig", self.T_world_from_rig, (HORIZON_STEPS, 4, 4)),
        )
        object.__setattr__(
            self,
            "requested_T_rig0_from_rig_t",
            _validate_transform_array(
                "requested_T_rig0_from_rig_t",
                self.requested_T_rig0_from_rig_t,
                (HORIZON_STEPS, 4, 4),
            ),
        )
        object.__setattr__(
            self,
            "achieved_T_rig0_from_rig_t",
            _validate_transform_array(
                "achieved_T_rig0_from_rig_t",
                self.achieved_T_rig0_from_rig_t,
                (HORIZON_STEPS, 4, 4),
            ),
        )
        if np.shares_memory(self.requested_T_rig0_from_rig_t, self.achieved_T_rig0_from_rig_t):
            raise ValueError("requested and achieved transforms must not alias")

    def T_world_from_cam(self, camera: CameraContract | None = None) -> np.ndarray:
        """Derive ``T_world_from_cam = T_world_from_rig @ T_rig_from_cam``."""
        contract = default_camera_contract() if camera is None else camera
        if not isinstance(contract, CameraContract):
            raise TypeError(f"camera: expected CameraContract, got {type(contract).__name__}")
        T_rig_from_cam = np.eye(4, dtype=np.float64)
        T_rig_from_cam[:3, :3] = contract.R_rig_from_cam
        T_rig_from_cam[:3, 3] = contract.t_rig_from_cam_m
        result = self.T_world_from_rig @ T_rig_from_cam
        result.setflags(write=False)
        return result


def kinematic_camera_trajectory(
    program: BranchProgram,
    *,
    initial_T_world_from_rig: np.ndarray | None = None,
) -> CameraTrajectory:
    """Build the analytic pilot trajectory with separate achieved/requested arrays."""
    if not isinstance(program, BranchProgram):
        raise TypeError(f"program: expected BranchProgram, got {type(program).__name__}")
    initial = (
        np.eye(4, dtype=np.float64)
        if initial_T_world_from_rig is None
        else _validate_transform_array("initial_T_world_from_rig", initial_T_world_from_rig, (4, 4))
    )
    requested = integrate_requested_relative(program)
    world = initial @ requested
    return CameraTrajectory(
        requested_command=program.requested_command,
        dt_seconds=program.dt_seconds,
        record_valid=program.record_valid,
        T_world_from_rig=world,
        requested_T_rig0_from_rig_t=np.array(requested, copy=True),
        achieved_T_rig0_from_rig_t=np.array(requested, copy=True),
    )


def rotation_geodesic_angle(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Return the SO(3) geodesic angle between two float64 rotations."""
    first = _validate_rotation("R_a", R_a)
    second = _validate_rotation("R_b", R_b)
    if np.array_equal(first, second):
        return 0.0
    relative = first.T @ second
    skew_vector = np.array(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ],
        dtype=np.float64,
    )
    sine = 0.5 * float(np.linalg.norm(skew_vector))
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.atan2(sine, cosine)


def requested_achieved_se3_error(trajectory: CameraTrajectory) -> tuple[np.ndarray, np.ndarray]:
    """Return per-step translation norms and rotation geodesic errors."""
    if not isinstance(trajectory, CameraTrajectory):
        raise TypeError(f"trajectory: expected CameraTrajectory, got {type(trajectory).__name__}")
    translation = np.linalg.norm(
        trajectory.requested_T_rig0_from_rig_t[:, :3, 3] - trajectory.achieved_T_rig0_from_rig_t[:, :3, 3],
        axis=1,
    ).astype(np.float64)
    rotation = np.array(
        [
            rotation_geodesic_angle(
                trajectory.requested_T_rig0_from_rig_t[index, :3, :3],
                trajectory.achieved_T_rig0_from_rig_t[index, :3, :3],
            )
            for index in range(HORIZON_STEPS)
        ],
        dtype=np.float64,
    )
    translation.setflags(write=False)
    rotation.setflags(write=False)
    return translation, rotation


def project_blender_camera(points_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project Blender-camera points (+X right,+Y up,-Z forward) to x-right/y-down pixels."""
    if not isinstance(points_cam, np.ndarray):
        raise TypeError(f"points_cam: expected np.ndarray, got {type(points_cam).__name__}")
    if points_cam.ndim < 1 or points_cam.shape[-1] != 3:
        raise ValueError(f"points_cam: expected shape (...,3), got {points_cam.shape}")
    if points_cam.dtype != np.dtype("<f8") or not np.all(np.isfinite(points_cam)):
        raise ValueError("points_cam: expected finite float64 values")
    intrinsic = _f64("K", K, (3, 3))
    depth = -points_cam[..., 2]
    if np.any(depth <= 0.0):
        raise ValueError("points_cam: expected positive optical depth (-Z)")
    result = np.empty(points_cam.shape[:-1] + (2,), dtype=np.float64)
    result[..., 0] = intrinsic[0, 0] * points_cam[..., 0] / depth + intrinsic[0, 2]
    result[..., 1] = intrinsic[1, 2] - intrinsic[1, 1] * points_cam[..., 1] / depth
    result.setflags(write=False)
    return result


__all__ = [
    "CameraTrajectory",
    "NamedSE3",
    "body_frd_twist_increment",
    "compose_se3",
    "identity_se3",
    "integrate_requested_relative",
    "invert_se3",
    "kinematic_camera_trajectory",
    "project_blender_camera",
    "requested_achieved_se3_error",
    "rotation_geodesic_angle",
]
