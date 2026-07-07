"""B3 6-D candidate camera/drone plan tokens (PURE tier)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from vllatent.scale_free_targets import (
    DEFAULT_FALLBACK_UNIT_XYZ,
    SCALE_FREE_LOG_SPEED_CLAMP,
    SCALE_FREE_SPEED_EPS,
    reference_speed_from_deltas,
    scale_free_actions_from_deltas,
)
from vllatent.schemas import DELTA_DTYPE

PLAN_TOKEN_DIM: Final[int] = 6
PLAN_TOKEN_FIELDS: Final[tuple[str, ...]] = (
    "unit_dir_x",
    "unit_dir_y",
    "unit_dir_z",
    "log_speed_ratio",
    "yaw_rate_norm",
    "valid",
)
DEFAULT_YAW_RATE_CAP_DEG_S: Final[float] = 180.0
DEFAULT_PLAN_VO_CONF_THRESHOLD: Final[float] = 0.3


@dataclass(frozen=True, eq=False)
class PlanTokenResult:
    """6-D plan tokens plus constituent validity masks."""

    tokens: np.ndarray
    valid_mask: np.ndarray
    moving_mask: np.ndarray
    speed_valid_mask: np.ndarray
    vo_valid_mask: np.ndarray

    def __post_init__(self) -> None:
        if self.tokens.shape[-1:] != (PLAN_TOKEN_DIM,):
            raise ValueError(f"tokens: expected last axis {PLAN_TOKEN_DIM}, got {self.tokens.shape}")
        if self.tokens.dtype != DELTA_DTYPE:
            raise ValueError(f"tokens: expected dtype {DELTA_DTYPE}, got {self.tokens.dtype}")
        if not np.all(np.isfinite(self.tokens)):
            raise ValueError("tokens: expected all finite values")
        expected = self.tokens.shape[:-1]
        for name in ("valid_mask", "moving_mask", "speed_valid_mask", "vo_valid_mask"):
            mask = getattr(self, name)
            if mask.shape != expected:
                raise ValueError(f"{name}: expected shape {expected}, got {mask.shape}")
            if mask.dtype != np.dtype(np.bool_):
                raise ValueError(f"{name}: expected dtype bool, got {mask.dtype}")


def _broadcast_optional(values: np.ndarray | float | None, shape: tuple[int, ...], default: float) -> np.ndarray:
    if values is None:
        return np.full(shape, default, dtype=DELTA_DTYPE)
    arr = np.asarray(values, dtype=DELTA_DTYPE)
    if not np.all(np.isfinite(arr)):
        raise ValueError("broadcast value: expected finite values")
    try:
        return np.broadcast_to(arr, shape).astype(DELTA_DTYPE, copy=False)
    except ValueError as exc:
        raise ValueError(f"broadcast value: expected broadcastable to {shape}, got {arr.shape}") from exc


def _yaw_rate_from_deltas(deltas: np.ndarray, dt_seconds: np.ndarray | float | None) -> np.ndarray:
    yaw = deltas[..., 3].astype(np.float64, copy=False)
    if dt_seconds is None:
        return yaw.astype(DELTA_DTYPE)
    dt = _broadcast_optional(dt_seconds, deltas.shape[:-1], 1.0).astype(np.float64, copy=False)
    if np.any(dt <= 0.0):
        raise ValueError("dt_seconds: expected values > 0")
    return (yaw / dt).astype(DELTA_DTYPE)


def plan_tokens_from_deltas(
    deltas: np.ndarray,
    dt_seconds: np.ndarray | float | None = None,
    *,
    reference_speed: float | None = None,
    vo_confidence: np.ndarray | float | None = None,
    vo_conf_threshold: float = DEFAULT_PLAN_VO_CONF_THRESHOLD,
    yaw_rate_cap_deg_s: float = DEFAULT_YAW_RATE_CAP_DEG_S,
    fallback_unit_xyz: tuple[float, float, float] | np.ndarray = DEFAULT_FALLBACK_UNIT_XYZ,
    speed_epsilon: float = SCALE_FREE_SPEED_EPS,
    log_speed_clip: float = SCALE_FREE_LOG_SPEED_CLAMP,
) -> PlanTokenResult:
    """Convert body-frame future deltas to B3 6-D plan tokens.

    ``dt_seconds=None`` means the translation and yaw columns are already
    velocity/rate-like. Passing ``dt_seconds`` treats rows as frame-to-frame raw
    deltas and normalizes translation speed and yaw rate by dt.
    """
    arr = np.asarray(deltas, dtype=DELTA_DTYPE)
    if arr.ndim < 1 or arr.shape[-1] != 4:
        raise ValueError(f"deltas: expected shape (..., 4), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("deltas: expected all finite values")
    if yaw_rate_cap_deg_s <= 0.0:
        raise ValueError(f"yaw_rate_cap_deg_s: expected > 0, got {yaw_rate_cap_deg_s}")
    if not (0.0 <= vo_conf_threshold <= 1.0):
        raise ValueError(f"vo_conf_threshold: expected [0,1], got {vo_conf_threshold}")

    ref = (
        reference_speed_from_deltas(arr, dt_seconds, speed_epsilon=speed_epsilon)
        if reference_speed is None
        else reference_speed
    )
    sf = scale_free_actions_from_deltas(
        arr,
        dt_seconds,
        reference_speed=ref,
        fallback_unit_xyz=fallback_unit_xyz,
        speed_epsilon=speed_epsilon,
        log_speed_clip=log_speed_clip,
    )

    yaw_rate = _yaw_rate_from_deltas(arr, dt_seconds)
    yaw_rate_norm = np.clip(
        yaw_rate.astype(np.float64, copy=False) / float(yaw_rate_cap_deg_s),
        -1.0,
        1.0,
    ).astype(DELTA_DTYPE)

    vo_conf = _broadcast_optional(vo_confidence, arr.shape[:-1], 1.0)
    vo_valid = vo_conf >= vo_conf_threshold
    valid = sf.moving_mask & sf.speed_valid_mask & vo_valid

    tokens = np.zeros(arr.shape[:-1] + (PLAN_TOKEN_DIM,), dtype=DELTA_DTYPE)
    tokens[..., :4] = sf.actions
    tokens[..., 4] = yaw_rate_norm
    tokens[..., 5] = valid.astype(DELTA_DTYPE)
    return PlanTokenResult(
        tokens=tokens,
        valid_mask=np.asarray(valid, dtype=np.bool_),
        moving_mask=sf.moving_mask,
        speed_valid_mask=sf.speed_valid_mask,
        vo_valid_mask=np.asarray(vo_valid, dtype=np.bool_),
    )


__all__ = [
    "DEFAULT_PLAN_VO_CONF_THRESHOLD",
    "DEFAULT_YAW_RATE_CAP_DEG_S",
    "PLAN_TOKEN_DIM",
    "PLAN_TOKEN_FIELDS",
    "PlanTokenResult",
    "plan_tokens_from_deltas",
]
