"""SE(3) to body-frame delta conversion (PURE tier) — Phase B1 step 3.

Converts MegaSaM camera trajectory (SE(3) 4x4 matrices, OpenCV convention) to
body-frame deltas ``(dx, dy, dz, dyaw)`` matching the ``delta_4dof`` format.

MegaSaM camera convention: X-right, Y-down, Z-forward.
Drone body NED convention: X-forward, Y-right, Z-down.
The rotation between them is a fixed permutation.

Scale is ambiguous from monocular VO — ``normalize_scale`` provides per-clip
normalization. GPS/IMU alignment is stubbed for future implementation.
"""
from __future__ import annotations

import numpy as np

from vllatent.frames import wrap_pi
from vllatent.schemas import DELTA_DTYPE

# OpenCV camera to drone body (NED/FRD) rotation.
# Camera: X-right, Y-down, Z-forward → Body: X-forward, Y-right, Z-down.
# body_x = cam_z, body_y = cam_x, body_z = cam_y.
R_BODY_FROM_CAM = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
], dtype=np.float64)


def rotation_to_yaw(R: np.ndarray) -> float:
    """Extract yaw angle (radians) from a 3x3 rotation matrix.

    Assumes the rotation is predominantly yaw (pitch/roll ≈ 0), consistent
    with the 4-DoF model. Uses atan2 on the body-frame XY projection.
    """
    return float(np.arctan2(R[1, 0], R[0, 0]))


def camera_to_drone_body(
    R_cam: np.ndarray,
    t_cam: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert camera-frame rotation + translation to drone body frame (NED).

    Parameters
    ----------
    R_cam : (3,3) rotation matrix in camera convention
    t_cam : (3,) translation in camera convention

    Returns
    -------
    R_body, t_body : rotation and translation in body (NED) convention
    """
    R_body = R_BODY_FROM_CAM @ R_cam @ R_BODY_FROM_CAM.T
    t_body = R_BODY_FROM_CAM @ t_cam
    return R_body, t_body


def se3_to_body_delta(
    T_prev: np.ndarray,
    T_curr: np.ndarray,
) -> np.ndarray:
    """Compute body-frame delta from two consecutive SE(3) camera poses.

    Parameters
    ----------
    T_prev, T_curr : (4,4) SE(3) camera-to-world transforms

    Returns
    -------
    delta : (4,) f32 — (dx, dy, dz, dyaw_deg) in body frame
    """
    T_rel = np.linalg.inv(T_prev) @ T_curr
    R_rel_cam = T_rel[:3, :3]
    t_rel_cam = T_rel[:3, 3]

    R_rel_body, t_rel_body = camera_to_drone_body(R_rel_cam, t_rel_cam)
    dyaw_rad = rotation_to_yaw(R_rel_body)
    dyaw_deg = float(np.degrees(dyaw_rad))

    return np.array(
        [t_rel_body[0], t_rel_body[1], t_rel_body[2], dyaw_deg],
        dtype=DELTA_DTYPE,
    )


def se3_sequence_to_deltas(
    poses: np.ndarray,
    fps: float | None = None,
) -> np.ndarray:
    """Convert a sequence of SE(3) poses to body-frame deltas.

    Parameters
    ----------
    poses : (N, 4, 4) SE(3) camera-to-world transforms
    fps : optional, not used in computation but validated

    Returns
    -------
    deltas : (N-1, 4) f32 — per-transition body-frame deltas
    """
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"poses: expected (N, 4, 4), got shape {poses.shape}")
    n = poses.shape[0]
    if n < 2:
        raise ValueError(f"poses: need >= 2 poses for deltas, got {n}")

    deltas = np.empty((n - 1, 4), dtype=DELTA_DTYPE)
    for i in range(n - 1):
        deltas[i] = se3_to_body_delta(poses[i], poses[i + 1])
    return deltas


def normalize_scale(
    deltas: np.ndarray,
    mode: str = "median_speed",
) -> np.ndarray:
    """Normalize the scale of body-frame deltas.

    Parameters
    ----------
    deltas : (N, 4) f32 — body-frame deltas
    mode : "median_speed" (default) — normalize so median displacement = 1.0
           "unit_max" — normalize so max displacement = 1.0

    Returns
    -------
    normalized : (N, 4) f32 — new array, input not mutated
    """
    if deltas.ndim != 2 or deltas.shape[1] != 4:
        raise ValueError(f"deltas: expected (N, 4), got shape {deltas.shape}")

    xyz = deltas[:, :3]
    magnitudes = np.linalg.norm(xyz, axis=1)

    if mode == "median_speed":
        scale = float(np.median(magnitudes))
    elif mode == "unit_max":
        scale = float(np.max(magnitudes))
    else:
        raise ValueError(f"normalize_scale: unknown mode {mode!r}, expected 'median_speed' or 'unit_max'")

    if scale < 1e-10:
        return deltas.copy()

    result = deltas.copy()
    result[:, :3] = xyz / scale
    return result


def validate_scale_consistency(deltas: np.ndarray) -> dict[str, float]:
    """Compute statistics on displacement magnitudes for quality assessment."""
    if deltas.ndim != 2 or deltas.shape[1] != 4:
        raise ValueError(f"deltas: expected (N, 4), got shape {deltas.shape}")

    magnitudes = np.linalg.norm(deltas[:, :3], axis=1).astype(float)
    q1, q3 = float(np.percentile(magnitudes, 25)), float(np.percentile(magnitudes, 75))
    iqr = q3 - q1
    outlier_lo = q1 - 1.5 * iqr
    outlier_hi = q3 + 1.5 * iqr
    n_outliers = int(np.sum((magnitudes < outlier_lo) | (magnitudes > outlier_hi)))

    return {
        "mean": float(np.mean(magnitudes)),
        "std": float(np.std(magnitudes)),
        "median": float(np.median(magnitudes)),
        "min": float(np.min(magnitudes)),
        "max": float(np.max(magnitudes)),
        "n_outliers": float(n_outliers),
        "outlier_fraction": float(n_outliers / len(magnitudes)) if len(magnitudes) > 0 else 0.0,
    }


def sim3_align(
    poses_vo: np.ndarray,
    trajectory_metric: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Sim(3) alignment of VO trajectory to metric trajectory (stub).

    Parameters
    ----------
    poses_vo : (N, 4, 4) VO camera poses (up-to-scale)
    trajectory_metric : (N, 3) metric positions (e.g., from GPS)

    Returns
    -------
    scale, R, t : Sim(3) parameters (scale, rotation, translation)

    Raises
    ------
    NotImplementedError
        Always — GPS/IMU alignment is a future capability.
    """
    raise NotImplementedError(
        "Sim(3) alignment requires GPS or IMU metric trajectory. "
        "For the seed dataset, use normalize_scale(mode='median_speed') instead."
    )


__all__ = [
    "R_BODY_FROM_CAM",
    "rotation_to_yaw",
    "camera_to_drone_body",
    "se3_to_body_delta",
    "se3_sequence_to_deltas",
    "normalize_scale",
    "validate_scale_consistency",
    "sim3_align",
]
