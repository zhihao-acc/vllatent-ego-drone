"""Sports sliding-window Dataset (TORCH tier) — Phase B step B1.13.

Map-style Dataset over ingest ``.npz`` cache files (from ``vllatent.ingest.pipeline``).
Produces sliding windows of ``(H + T)`` frames within each fixed-length clip.

**Cache on-disk format** (written by ``vllatent.ingest.pipeline._build_clip_npz``):

  ``latents``         (N, 196, 768) fp16  — DINOv3 patch tokens per frame
  ``deltas``          (N-1, 4)      f32   — 4-DoF body-frame delta (dx,dy,dz,dyaw)
  ``vo_confidence``   (N,)          f32   — MegaSaM VO confidence per frame
  ``frame_quality``   (N,)          f32   — composite quality per frame
  ``timestamps``      (N,)          f64   — absolute frame timestamps (seconds)

History latents are **GT from cache** (not predicted). Block-causal mask handles
the first H-1 frames where history is zero-padded.

Delta preprocessing pipeline (applied once at dataset construction):
  1. Physics hard clip (max displacement, max dyaw — scaled by dt)
  2. Median filter k=3
  3. Velocity normalization (delta / dt)
  4. Per-dimension z-score normalization (store mean/std for inference)

numpy-only emission — torch enters at DataLoader collation (B1.14).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vllatent.schemas import (
    DOF,
    EMBED_DIM,
    HISTORY,
    HORIZON,
    LATENT_DTYPE,
    MASK_DTYPE,
    PATCH_TOKENS,
)

MAX_DISPLACEMENT_5HZ = 4.0
MAX_DYAW_DEG_5HZ = 24.0
MEDIAN_FILTER_K = 3
NOISE_SCALE = 0.05


@dataclass(frozen=True)
class NormStats:
    """Per-dimension mean/std for z-score normalization of velocity-deltas."""

    mean: np.ndarray   # (4,) f32
    std: np.ndarray    # (4,) f32

    def normalize(self, v: np.ndarray) -> np.ndarray:
        safe_std = np.where(self.std > 1e-8, self.std, np.ones_like(self.std))
        return (v - self.mean) / safe_std

    def denormalize(self, v: np.ndarray) -> np.ndarray:
        safe_std = np.where(self.std > 1e-8, self.std, np.ones_like(self.std))
        return v * safe_std + self.mean


@dataclass(frozen=True)
class SportsSample:
    """One training sample emitted by the sports loader."""

    z_t: np.ndarray              # (P, D) fp16 — current observation latent
    history_latents: np.ndarray  # (H, P, D) fp16 — GT latents for previous H frames
    history_mask: np.ndarray     # (H,) bool — True=real, False=padding
    target_latents: np.ndarray   # (T, P, D) fp16 — GT future latents (L_latent targets)
    target_deltas: np.ndarray    # (T, 4) f32 — preprocessed future deltas (L_wp targets)
    vo_confidence: np.ndarray    # (T,) f32 — per-step VO confidence
    frame_quality: float         # composite quality of z_t frame
    dt_seconds: np.ndarray       # (T,) f32 — inter-frame time deltas


def physics_clip(deltas: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """Hard-clip deltas by physics limits scaled by dt."""
    result = deltas.copy()
    scale = dt / 0.2  # 0.2s = 1/5Hz baseline
    max_disp = MAX_DISPLACEMENT_5HZ * scale
    max_dyaw = MAX_DYAW_DEG_5HZ * scale
    for i in range(3):
        result[:, i] = np.clip(result[:, i], -max_disp, max_disp)
    result[:, 3] = np.clip(result[:, 3], -max_dyaw, max_dyaw)
    return result


def median_filter_deltas(deltas: np.ndarray, k: int = MEDIAN_FILTER_K) -> np.ndarray:
    """Apply median filter along time axis per dimension."""
    if deltas.shape[0] < k:
        return deltas.copy()
    result = np.empty_like(deltas)
    half = k // 2
    for i in range(deltas.shape[0]):
        lo = max(0, i - half)
        hi = min(deltas.shape[0], i + half + 1)
        result[i] = np.median(deltas[lo:hi], axis=0)
    return result


def velocity_normalize(deltas: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """Convert positional deltas to velocities: delta / dt."""
    safe_dt = np.where(dt > 1e-8, dt, np.ones_like(dt) * 0.2)
    return deltas / safe_dt[:, None]


def compute_norm_stats(all_velocities: list[np.ndarray]) -> NormStats:
    """Compute per-dimension mean/std across all clips."""
    if not all_velocities:
        return NormStats(mean=np.zeros(DOF, dtype=np.float32),
                         std=np.ones(DOF, dtype=np.float32))
    concat = np.concatenate(all_velocities, axis=0)
    return NormStats(
        mean=concat.mean(axis=0).astype(np.float32),
        std=concat.std(axis=0).astype(np.float32),
    )


def _preprocess_deltas(
    deltas: np.ndarray,
    dt: np.ndarray,
) -> np.ndarray:
    """Full preprocessing pipeline: clip → median → velocity."""
    clipped = physics_clip(deltas, dt)
    filtered = median_filter_deltas(clipped)
    return velocity_normalize(filtered, dt)


def _load_clip(path: Path) -> dict[str, np.ndarray]:
    """Load a sports ingest .npz clip."""
    with np.load(str(path)) as data:
        return {k: data[k] for k in data.files}


class SportsTrainingDataset:
    """Map-style Dataset over sports ingest .npz cache files.

    Sliding window of (H + T) frames within each clip.  History latents are GT
    from cache (not predicted).  Block-causal mask handles the first H-1 frames
    where full history isn't available.

    Parameters
    ----------
    cache_dir : str | Path
        Directory containing .npz clip files.
    clip_ids : list[str] | None
        Specific clip IDs to load.  If None, loads all .npz files.
    augment : bool
        Enable training augmentation (temporal jitter, delta noise).
    norm_stats : NormStats | None
        Pre-computed normalization stats.  If None, computed from loaded clips.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        clip_ids: list[str] | None = None,
        augment: bool = False,
        norm_stats: NormStats | None = None,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._augment = augment
        self._rng = np.random.default_rng(42)

        npz_paths = self._discover_clips(clip_ids)
        if not npz_paths:
            raise ValueError(f"No .npz clips found in {cache_dir}")

        self._clips: list[dict[str, np.ndarray]] = []
        self._clip_velocities: list[np.ndarray] = []
        self._clip_dt: list[np.ndarray] = []

        for p in npz_paths:
            clip = _load_clip(p)
            n_frames = clip["latents"].shape[0]
            if n_frames < HISTORY + HORIZON + 1:
                continue

            ts = clip["timestamps"]
            dt = np.diff(ts).astype(np.float32)
            velocities = _preprocess_deltas(clip["deltas"], dt)

            self._clips.append(clip)
            self._clip_velocities.append(velocities)
            self._clip_dt.append(dt)

        if not self._clips:
            raise ValueError("No clips with enough frames (need >= H+T+1)")

        if norm_stats is not None:
            self._norm_stats = norm_stats
        else:
            self._norm_stats = compute_norm_stats(self._clip_velocities)

        self._samples: list[tuple[int, int]] = []
        for clip_idx, clip in enumerate(self._clips):
            n = clip["latents"].shape[0]
            for t in range(n - HORIZON):
                self._samples.append((clip_idx, t))

    def _discover_clips(self, clip_ids: list[str] | None) -> list[Path]:
        if clip_ids is not None:
            paths = []
            for cid in clip_ids:
                p = self._cache_dir / f"{cid}.npz"
                if p.exists():
                    paths.append(p)
            return sorted(paths)
        return sorted(self._cache_dir.glob("*.npz"))

    @property
    def norm_stats(self) -> NormStats:
        return self._norm_stats

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> SportsSample:
        clip_idx, t = self._samples[idx]
        clip = self._clips[clip_idx]
        velocities = self._clip_velocities[clip_idx]
        dt_all = self._clip_dt[clip_idx]

        if self._augment:
            n = clip["latents"].shape[0]
            jitter = self._rng.integers(-1, 2)
            t = max(0, min(t + jitter, n - HORIZON - 1))

        latents = clip["latents"]
        z_t = latents[t]

        history = np.zeros((HISTORY, PATCH_TOKENS, EMBED_DIM), dtype=LATENT_DTYPE)
        mask = np.zeros(HISTORY, dtype=MASK_DTYPE)
        for h in range(HISTORY):
            src_idx = t - HISTORY + 1 + h
            if src_idx >= 0:
                history[h] = latents[src_idx]
                mask[h] = True

        target_lat = latents[t + 1: t + 1 + HORIZON]
        actual_t = target_lat.shape[0]
        if actual_t < HORIZON:
            pad = np.zeros((HORIZON - actual_t, PATCH_TOKENS, EMBED_DIM), dtype=LATENT_DTYPE)
            target_lat = np.concatenate([target_lat, pad], axis=0)

        target_v = np.zeros((HORIZON, DOF), dtype=np.float32)
        dt_sec = np.full(HORIZON, 0.2, dtype=np.float32)
        vo_conf = np.ones(HORIZON, dtype=np.float32)

        for k in range(HORIZON):
            delta_idx = t + k
            if delta_idx < velocities.shape[0]:
                target_v[k] = self._norm_stats.normalize(velocities[delta_idx])
                dt_sec[k] = dt_all[delta_idx]
            if t + k < clip["vo_confidence"].shape[0]:
                vo_conf[k] = clip["vo_confidence"][t + k]

        if self._augment and self._norm_stats.std is not None:
            noise = self._rng.normal(
                0, NOISE_SCALE, size=target_v.shape
            ).astype(np.float32)
            target_v = target_v + noise

        fq = float(clip["frame_quality"][t])

        return SportsSample(
            z_t=z_t,
            history_latents=history,
            history_mask=mask,
            target_latents=target_lat,
            target_deltas=target_v,
            vo_confidence=vo_conf,
            frame_quality=fq,
            dt_seconds=dt_sec,
        )

    def save_norm_stats(self, path: str | Path) -> None:
        np.savez(
            str(path),
            mean=self._norm_stats.mean,
            std=self._norm_stats.std,
        )

    @staticmethod
    def load_norm_stats(path: str | Path) -> NormStats:
        with np.load(str(path)) as data:
            return NormStats(mean=data["mean"], std=data["std"])
