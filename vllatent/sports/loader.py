"""Sports cached-latent map-style Dataset (PURE tier) — Phase B1 step 9.

Reads the per-clip ``.npz`` cache produced by ``vllatent.sports.cache`` and emits
``SportsSample`` tuples. History window is LEFT zero-padded at clip start.

Quality-mask filtering: only frames where ``quality_mask[t] == True`` are emittable.
The mask still allows them as history context — a blurry frame is bad as a prediction
target but acceptable as context.

Cache on-disk format (per clip, N frames):

    latents       (N, 196, 768) fp16
    deltas        (N-1, 4)      f32   — body-frame (dx,dy,dz,dyaw)
    vo_confidence (N,)          f32
    frame_quality (N,)          f32
    timestamps    (N,)          f64
    quality_mask  (N,)          bool

A sample exists for each transition ``t in [0, N-2]`` where ``quality_mask[t]``
AND ``quality_mask[t+1]`` are both True (both source and target must be quality-approved).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vllatent.schemas import (
    DELTA_DTYPE,
    EMBED_DIM,
    HISTORY,
    LATENT_DTYPE,
    MASK_DTYPE,
    PATCH_TOKENS,
)
from vllatent.sports.schemas import SportsSample


class SportsDataset:
    """Map-style Dataset over a sports latent cache; emits ``SportsSample``."""

    def __init__(
        self,
        cache_dir: str | Path,
        history: int = HISTORY,
        require_quality: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.history = history
        self.require_quality = require_quality

        manifest_path = self.cache_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        self._entries: list[dict] = list(manifest.get("entries", []))

        self._index: list[tuple[int, int]] = []
        for clip_i, entry in enumerate(self._entries):
            n_frames = int(entry["n_frames"])
            npz_path = self.cache_dir / entry["latent_path"]

            if self.require_quality and npz_path.exists():
                with np.load(str(npz_path)) as data:
                    qm = data["quality_mask"]
                for t in range(n_frames - 1):
                    if qm[t] and qm[t + 1]:
                        self._index.append((clip_i, t))
            else:
                for t in range(n_frames - 1):
                    self._index.append((clip_i, t))

    def __len__(self) -> int:
        return len(self._index)

    @property
    def n_clips(self) -> int:
        return len(self._entries)

    def _load_clip(self, clip_i: int) -> dict[str, np.ndarray]:
        entry = self._entries[clip_i]
        with np.load(self.cache_dir / entry["latent_path"]) as data:
            return {k: data[k] for k in data.files}

    def __getitem__(self, i: int) -> SportsSample:
        clip_i, t = self._index[i]
        clip = self._load_clip(clip_i)
        latents = clip["latents"]

        h = self.history
        history = np.zeros((h, PATCH_TOKENS, EMBED_DIM), dtype=LATENT_DTYPE)
        history_mask = np.zeros((h,), dtype=MASK_DTYPE)
        for j in range(h):
            src = t - (h - 1 - j)
            if src >= 0:
                history[j] = latents[src]
                history_mask[j] = True

        dt = 0.0
        if "timestamps" in clip:
            dt = float(clip["timestamps"][t + 1] - clip["timestamps"][t])

        return SportsSample(
            z_t=latents[t].astype(LATENT_DTYPE),
            history_latents=history,
            history_mask=history_mask,
            z_next=latents[t + 1].astype(LATENT_DTYPE),
            delta_4dof=clip["deltas"][t].astype(DELTA_DTYPE),
            vo_confidence=float(clip["vo_confidence"][t]),
            frame_quality=float(clip["frame_quality"][t]),
            dt_seconds=dt,
        )


def inspect_sports_cache(cache_dir: str | Path, n: int = 5) -> int:
    """Print the first ``n`` samples of a sports cache."""
    ds = SportsDataset(cache_dir, require_quality=False)
    print(f"cache {cache_dir}: {ds.n_clips} clips, {len(ds)} transitions (H={ds.history})")
    for i in range(min(n, len(ds))):
        s = ds[i]
        print(
            f"[{i}] z_t {tuple(s.z_t.shape)} {s.z_t.dtype} "
            f"delta={np.round(s.delta_4dof, 3).tolist()} "
            f"confidence={s.vo_confidence:.3f} quality={s.frame_quality:.3f} "
            f"dt={s.dt_seconds:.3f}s hist_mask={s.history_mask.tolist()}"
        )
    return 0


__all__ = ["SportsDataset", "inspect_sports_cache"]
