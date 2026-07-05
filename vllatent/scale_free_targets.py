"""Scale-free future-action target contract (PURE tier) -- Phase B2.1.

Youtube/MegaSaM translation magnitude is not metric truth, so B2 labels encode
only direction/shape and relative speed intent.  The locked per-horizon vector is:

    [unit_dir_x, unit_dir_y, unit_dir_z, log_speed_ratio]

``unit_dir_*`` is a body-frame translation direction. ``log_speed_ratio`` is
relative to a reference speed in the same arbitrary VO scale, so uniformly
rescaling all translations leaves the target unchanged.  Metric conversion is an
inference/controller concern: multiply the predicted speed ratio by onboard odom
reference speed, then clamp strictly below the vehicle speed cap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from vllatent.schemas import DELTA_DTYPE

SCALE_FREE_ACTION_DIM: Final[int] = 4
SCALE_FREE_ACTION_FIELDS: Final[tuple[str, ...]] = (
    "unit_dir_x",
    "unit_dir_y",
    "unit_dir_z",
    "log_speed_ratio",
)
SCALE_FREE_SPEED_EPS: Final[float] = 1e-6
SCALE_FREE_LOG_SPEED_CLAMP: Final[float] = 8.0
DEFAULT_FALLBACK_UNIT_XYZ: Final[tuple[float, float, float]] = (1.0, 0.0, 0.0)

CONTROLLER_MAX_SPEED_MPS: Final[float] = 7.5
CONTROLLER_SPEED_MARGIN_MPS: Final[float] = 1e-3


@dataclass(frozen=True, eq=False)
class ScaleFreeActionTargets:
    """Future action labels plus validity mask.

    This record intentionally contains no model-input fields.  B2 loaders may
    use ``actions`` and ``moving_mask`` as supervision targets, while previous
    observed motion must be computed separately from past deltas.
    """

    actions: np.ndarray       # (..., 4) f32, ordered by SCALE_FREE_ACTION_FIELDS
    moving_mask: np.ndarray   # (...) bool, True where translation direction is defined
    speed_valid_mask: np.ndarray   # (...) bool, True where speed-ratio supervision is not clipped

    def __post_init__(self) -> None:
        if not isinstance(self.actions, np.ndarray):
            raise TypeError(f"actions: expected np.ndarray, got {type(self.actions).__name__}")
        if self.actions.shape[-1:] != (SCALE_FREE_ACTION_DIM,):
            raise ValueError(
                f"actions: expected last axis {SCALE_FREE_ACTION_DIM}, got shape {self.actions.shape}"
            )
        if self.actions.dtype != DELTA_DTYPE:
            raise ValueError(f"actions: expected dtype {DELTA_DTYPE}, got {self.actions.dtype}")
        if not np.all(np.isfinite(self.actions)):
            raise ValueError("actions: expected all finite values")
        if not isinstance(self.moving_mask, np.ndarray):
            raise TypeError(f"moving_mask: expected np.ndarray, got {type(self.moving_mask).__name__}")
        if self.moving_mask.dtype != np.dtype(np.bool_):
            raise ValueError(f"moving_mask: expected dtype bool, got {self.moving_mask.dtype}")
        if self.moving_mask.shape != self.actions.shape[:-1]:
            raise ValueError(
                f"moving_mask: expected shape {self.actions.shape[:-1]}, got {self.moving_mask.shape}"
            )
        if not isinstance(self.speed_valid_mask, np.ndarray):
            raise TypeError(
                f"speed_valid_mask: expected np.ndarray, got {type(self.speed_valid_mask).__name__}"
            )
        if self.speed_valid_mask.dtype != np.dtype(np.bool_):
            raise ValueError(f"speed_valid_mask: expected dtype bool, got {self.speed_valid_mask.dtype}")
        if self.speed_valid_mask.shape != self.actions.shape[:-1]:
            raise ValueError(
                f"speed_valid_mask: expected shape {self.actions.shape[:-1]}, "
                f"got {self.speed_valid_mask.shape}"
            )


@dataclass(frozen=True)
class ScaleFreeActionDiagnostics:
    """Small pure summary for B2 target-health checks."""

    count: int
    moving_count: int
    speed_valid_count: int
    moving_fraction: float
    speed_valid_fraction: float
    log_speed_min: float
    log_speed_p50: float
    log_speed_p95: float
    log_speed_p99: float
    log_speed_max: float
    max_abs_log_speed: float
    unmasked_log_speed_outliers: int


def _as_deltas(deltas: np.ndarray) -> np.ndarray:
    arr = np.asarray(deltas, dtype=DELTA_DTYPE)
    if arr.ndim < 1 or arr.shape[-1] != 4:
        raise ValueError(f"deltas: expected shape (..., 4), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("deltas: expected all finite values")
    return arr


def _broadcast_dt(dt_seconds: np.ndarray | float | None, shape: tuple[int, ...]) -> np.ndarray:
    if dt_seconds is None:
        return np.ones(shape, dtype=DELTA_DTYPE)
    dt = np.asarray(dt_seconds, dtype=DELTA_DTYPE)
    if not np.all(np.isfinite(dt)):
        raise ValueError("dt_seconds: expected finite values")
    if np.any(dt <= 0.0):
        raise ValueError("dt_seconds: expected values > 0")
    try:
        return np.broadcast_to(dt, shape).astype(DELTA_DTYPE, copy=False)
    except ValueError as exc:
        raise ValueError(f"dt_seconds: expected broadcastable to {shape}, got {dt.shape}") from exc


def _translation_speeds(deltas: np.ndarray, dt_seconds: np.ndarray | float | None) -> tuple[np.ndarray, np.ndarray]:
    arr = _as_deltas(deltas)
    dt = _broadcast_dt(dt_seconds, arr.shape[:-1])
    xyz = arr[..., :3].astype(np.float64, copy=False)
    speeds = (np.linalg.norm(xyz, axis=-1) / dt.astype(np.float64, copy=False)).astype(DELTA_DTYPE)
    return arr, speeds


def _validate_reference_speed(reference_speed: float, *, speed_epsilon: float) -> float:
    if isinstance(reference_speed, bool):
        raise TypeError("reference_speed: expected a positive finite float, got bool")
    ref = float(reference_speed)
    if not np.isfinite(ref) or ref < 0.0:
        raise ValueError(f"reference_speed: expected finite value >= 0, got {reference_speed}")
    return max(ref, speed_epsilon)


def _validate_fallback_unit(fallback_unit_xyz: tuple[float, float, float] | np.ndarray) -> np.ndarray:
    unit = np.asarray(fallback_unit_xyz, dtype=DELTA_DTYPE)
    if unit.shape != (3,):
        raise ValueError(f"fallback_unit_xyz: expected shape (3,), got {unit.shape}")
    if not np.all(np.isfinite(unit)):
        raise ValueError("fallback_unit_xyz: expected finite values")
    norm = float(np.linalg.norm(unit.astype(np.float64)))
    if norm <= 0.0:
        raise ValueError("fallback_unit_xyz: expected non-zero vector")
    return (unit / norm).astype(DELTA_DTYPE)


def reference_speed_from_deltas(
    deltas: np.ndarray,
    dt_seconds: np.ndarray | float | None = None,
    *,
    speed_epsilon: float = SCALE_FREE_SPEED_EPS,
) -> float:
    """Return a robust speed reference from observed deltas in arbitrary VO scale.

    Use this on past/observed motion for model inputs or diagnostics.  Calling it
    on future labels would create a target-derived quantity, so target helpers do
    not expose their internally derived reference.
    """
    if speed_epsilon <= 0.0:
        raise ValueError(f"speed_epsilon: expected > 0, got {speed_epsilon}")
    _, speeds = _translation_speeds(deltas, dt_seconds)
    valid = speeds > speed_epsilon
    if not np.any(valid):
        return float(speed_epsilon)
    return float(np.median(speeds[valid]))


def scale_free_actions_from_deltas(
    deltas: np.ndarray,
    dt_seconds: np.ndarray | float | None = None,
    *,
    reference_speed: float | None = None,
    fallback_unit_xyz: tuple[float, float, float] | np.ndarray = DEFAULT_FALLBACK_UNIT_XYZ,
    speed_epsilon: float = SCALE_FREE_SPEED_EPS,
    log_speed_clip: float = SCALE_FREE_LOG_SPEED_CLAMP,
) -> ScaleFreeActionTargets:
    """Convert body-frame 4-DoF deltas to scale-free action labels.

    ``deltas`` may be raw frame-to-frame deltas or velocity-like rows.  If
    ``dt_seconds`` is provided, translation magnitude is converted to speed before
    forming the ratio.  The yaw column is accepted for compatibility with the
    existing ``(dx, dy, dz, dyaw)`` cache format but is not part of this B2.1
    target contract.
    """
    if speed_epsilon <= 0.0:
        raise ValueError(f"speed_epsilon: expected > 0, got {speed_epsilon}")
    if log_speed_clip <= 0.0:
        raise ValueError(f"log_speed_clip: expected > 0, got {log_speed_clip}")

    arr, speeds = _translation_speeds(deltas, dt_seconds)
    fallback = _validate_fallback_unit(fallback_unit_xyz)
    ref = (
        reference_speed_from_deltas(arr, dt_seconds, speed_epsilon=speed_epsilon)
        if reference_speed is None
        else _validate_reference_speed(reference_speed, speed_epsilon=speed_epsilon)
    )

    moving = speeds > speed_epsilon
    unit = np.broadcast_to(fallback, arr.shape[:-1] + (3,)).copy()
    if np.any(moving):
        xyz = arr[..., :3].astype(np.float64, copy=False)
        moving_norms = np.linalg.norm(xyz[moving], axis=-1)
        unit[moving] = (xyz[moving] / moving_norms[:, None]).astype(DELTA_DTYPE)

    safe_speeds = np.maximum(speeds, speed_epsilon)
    raw_log_ratio = np.log(safe_speeds.astype(np.float64) / ref)
    speed_valid = moving & (np.abs(raw_log_ratio) <= log_speed_clip)
    log_ratio = np.clip(raw_log_ratio, -log_speed_clip, log_speed_clip).astype(DELTA_DTYPE)

    actions = np.empty(arr.shape[:-1] + (SCALE_FREE_ACTION_DIM,), dtype=DELTA_DTYPE)
    actions[..., :3] = unit
    actions[..., 3] = log_ratio
    return ScaleFreeActionTargets(
        actions=actions,
        moving_mask=np.asarray(moving, dtype=np.bool_),
        speed_valid_mask=np.asarray(speed_valid, dtype=np.bool_),
    )


def future_deltas_to_scale_free_targets(
    future_deltas: np.ndarray,
    dt_seconds: np.ndarray | float | None = None,
    *,
    reference_speed: float | None = None,
    fallback_unit_xyz: tuple[float, float, float] | np.ndarray = DEFAULT_FALLBACK_UNIT_XYZ,
    speed_epsilon: float = SCALE_FREE_SPEED_EPS,
    log_speed_clip: float = SCALE_FREE_LOG_SPEED_CLAMP,
) -> ScaleFreeActionTargets:
    """Target-only wrapper for future action supervision.

    The returned object deliberately contains only labels and a target validity
    mask.  It does not return previous actions, reference speed, odom speed, or
    any other model input.
    """
    return scale_free_actions_from_deltas(
        future_deltas,
        dt_seconds,
        reference_speed=reference_speed,
        fallback_unit_xyz=fallback_unit_xyz,
        speed_epsilon=speed_epsilon,
        log_speed_clip=log_speed_clip,
    )


def scale_free_action_diagnostics(
    actions: np.ndarray,
    moving_mask: np.ndarray,
    speed_valid_mask: np.ndarray | None = None,
    *,
    log_speed_outlier_threshold: float = SCALE_FREE_LOG_SPEED_CLAMP,
) -> ScaleFreeActionDiagnostics:
    """Summarize target-label health without importing torch.

    ``unmasked_log_speed_outliers`` uses ``speed_valid_mask`` when supplied, so
    clipped or masked speed spikes are not counted as active training labels.
    """
    arr = np.asarray(actions, dtype=DELTA_DTYPE)
    if arr.ndim < 1 or arr.shape[-1] != SCALE_FREE_ACTION_DIM:
        raise ValueError(f"actions: expected shape (..., {SCALE_FREE_ACTION_DIM}), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("actions: expected all finite values")
    moving = np.asarray(moving_mask, dtype=np.bool_)
    if moving.shape != arr.shape[:-1]:
        raise ValueError(f"moving_mask: expected shape {arr.shape[:-1]}, got {moving.shape}")
    if speed_valid_mask is None:
        speed_valid = moving
    else:
        speed_valid = np.asarray(speed_valid_mask, dtype=np.bool_)
        if speed_valid.shape != arr.shape[:-1]:
            raise ValueError(f"speed_valid_mask: expected shape {arr.shape[:-1]}, got {speed_valid.shape}")
    if log_speed_outlier_threshold <= 0.0:
        raise ValueError(
            f"log_speed_outlier_threshold: expected > 0, got {log_speed_outlier_threshold}"
        )

    logs = arr[..., 3].reshape(-1).astype(np.float64, copy=False)
    active_logs = logs[speed_valid.reshape(-1)]
    percentile_source = active_logs if active_logs.size else logs
    if percentile_source.size:
        p50, p95, p99 = np.percentile(percentile_source, [50, 95, 99])
        log_min = float(np.min(percentile_source))
        log_max = float(np.max(percentile_source))
        max_abs = float(np.max(np.abs(percentile_source)))
    else:
        p50 = p95 = p99 = log_min = log_max = max_abs = 0.0
    unmasked = np.abs(logs) > log_speed_outlier_threshold
    unmasked &= speed_valid.reshape(-1)

    count = int(np.prod(arr.shape[:-1], dtype=np.int64))
    moving_count = int(moving.sum())
    speed_valid_count = int(speed_valid.sum())
    denom = max(count, 1)
    return ScaleFreeActionDiagnostics(
        count=count,
        moving_count=moving_count,
        speed_valid_count=speed_valid_count,
        moving_fraction=float(moving_count / denom),
        speed_valid_fraction=float(speed_valid_count / denom),
        log_speed_min=log_min,
        log_speed_p50=float(p50),
        log_speed_p95=float(p95),
        log_speed_p99=float(p99),
        log_speed_max=log_max,
        max_abs_log_speed=max_abs,
        unmasked_log_speed_outliers=int(unmasked.sum()),
    )


def metric_speed_command_from_log_ratio(
    log_speed_ratio: np.ndarray | float,
    odom_reference_speed_mps: np.ndarray | float,
    *,
    max_speed_mps: float = CONTROLLER_MAX_SPEED_MPS,
    margin_mps: float = CONTROLLER_SPEED_MARGIN_MPS,
) -> np.ndarray:
    """Convert relative speed intent to metric speed at inference time.

    This helper is intentionally separate from target generation.  It uses real
    onboard odometry scale and clamps to ``max_speed_mps - margin_mps`` so the
    command is strictly below the vehicle cap.
    """
    if max_speed_mps <= 0.0:
        raise ValueError(f"max_speed_mps: expected > 0, got {max_speed_mps}")
    if not (0.0 < margin_mps < max_speed_mps):
        raise ValueError(f"margin_mps: expected in (0, max_speed_mps), got {margin_mps}")

    log_ratio = np.asarray(log_speed_ratio, dtype=DELTA_DTYPE)
    ref = np.asarray(odom_reference_speed_mps, dtype=DELTA_DTYPE)
    if not np.all(np.isfinite(log_ratio)):
        raise ValueError("log_speed_ratio: expected finite values")
    if not np.all(np.isfinite(ref)) or np.any(ref < 0.0):
        raise ValueError("odom_reference_speed_mps: expected finite values >= 0")

    speed = ref * np.exp(log_ratio.astype(np.float64))
    cap = np.asarray(max_speed_mps - margin_mps, dtype=DELTA_DTYPE)
    return np.minimum(speed, cap).astype(DELTA_DTYPE)


__all__ = [
    "CONTROLLER_MAX_SPEED_MPS",
    "CONTROLLER_SPEED_MARGIN_MPS",
    "DEFAULT_FALLBACK_UNIT_XYZ",
    "SCALE_FREE_ACTION_DIM",
    "SCALE_FREE_ACTION_FIELDS",
    "SCALE_FREE_LOG_SPEED_CLAMP",
    "SCALE_FREE_SPEED_EPS",
    "ScaleFreeActionDiagnostics",
    "ScaleFreeActionTargets",
    "future_deltas_to_scale_free_targets",
    "metric_speed_command_from_log_ratio",
    "reference_speed_from_deltas",
    "scale_free_action_diagnostics",
    "scale_free_actions_from_deltas",
]
