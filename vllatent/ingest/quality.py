"""Frame quality scoring and filtering (PURE tier).

All scoring uses numpy only (Laplacian via ``np.gradient``, histogram via
``np.histogram``, pixel statistics). No cv2 dependency at runtime.

Scores are in [0, 1] where higher = better quality.
"""
from __future__ import annotations

import numpy as np

_BLUR_NORM = 500.0
_EXPOSURE_BINS = 64


def motion_blur_score(frame: np.ndarray) -> float:
    """Score frame sharpness via Laplacian variance (higher = sharper)."""
    if frame.ndim == 3:
        gray = np.mean(frame, axis=2)
    else:
        gray = frame.astype(np.float64)

    gy, gx = np.gradient(gray.astype(np.float64))
    laplacian_var = float(np.var(gx) + np.var(gy))
    return min(laplacian_var / _BLUR_NORM, 1.0)


def exposure_score(frame: np.ndarray) -> float:
    """Score exposure quality via histogram spread."""
    if frame.ndim == 3:
        gray = np.mean(frame, axis=2)
    else:
        gray = frame.astype(np.float64)

    hist, _ = np.histogram(gray.ravel(), bins=_EXPOSURE_BINS, range=(0, 255))
    occupied = int(np.sum(hist > 0))
    return occupied / _EXPOSURE_BINS


def snow_whiteout_score(frame: np.ndarray) -> float:
    """Score snow whiteout as fraction of near-white pixels (higher = worse)."""
    if frame.ndim == 3:
        brightness = np.mean(frame, axis=2)
    else:
        brightness = frame.astype(np.float64)

    near_white = np.sum(brightness > 230)
    return float(near_white / brightness.size) if brightness.size > 0 else 0.0


def composite_quality(
    frame: np.ndarray,
    *,
    w_blur: float = 0.5,
    w_exposure: float = 0.3,
    w_whiteout: float = 0.2,
) -> float:
    """Weighted composite quality score in [0, 1]."""
    blur = motion_blur_score(frame)
    exposure = exposure_score(frame)
    whiteout_frac = snow_whiteout_score(frame)
    whiteout_quality = 1.0 - whiteout_frac
    return float(w_blur * blur + w_exposure * exposure + w_whiteout * whiteout_quality)


def score_frames(frames: list[np.ndarray] | np.ndarray) -> np.ndarray:
    """Score a batch of frames, returning per-frame composite quality (N,) f32."""
    if isinstance(frames, np.ndarray) and frames.ndim == 4:
        return np.array(
            [composite_quality(frames[i]) for i in range(frames.shape[0])],
            dtype=np.float32,
        )
    return np.array([composite_quality(f) for f in frames], dtype=np.float32)


def filter_frames(
    qualities: np.ndarray,
    threshold: float = 0.3,
) -> np.ndarray:
    """Per-frame accept/reject mask. True = passes quality gate."""
    return qualities >= threshold


def clip_quality_summary(qualities: np.ndarray, threshold: float = 0.3) -> dict[str, float]:
    """Summary statistics for a clip's frame quality distribution."""
    mask = filter_frames(qualities, threshold)
    n_total = len(qualities)
    n_rejected = int(np.sum(~mask))
    return {
        "min": float(np.min(qualities)) if n_total > 0 else 0.0,
        "max": float(np.max(qualities)) if n_total > 0 else 0.0,
        "mean": float(np.mean(qualities)) if n_total > 0 else 0.0,
        "median": float(np.median(qualities)) if n_total > 0 else 0.0,
        "n_total": float(n_total),
        "n_rejected": float(n_rejected),
        "rejection_rate": float(n_rejected / n_total) if n_total > 0 else 0.0,
    }


__all__ = [
    "motion_blur_score",
    "exposure_score",
    "snow_whiteout_score",
    "composite_quality",
    "score_frames",
    "filter_frames",
    "clip_quality_summary",
]
