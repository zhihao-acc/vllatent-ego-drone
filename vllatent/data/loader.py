"""Cached-latent / distillation map-style Dataset (TORCH tier) — Phase-A step A5.15 (was step 10).

Emits the per-step distillation pair ``(StepSample student-inputs, OracleTarget teacher-targets)``
(the A5.9 seam) over a render-once latent cache. Inherits manifest reading, lazy loading, and
history windowing from ``vllatent.data.base_loader.LatentDatasetBase``.

Training is sim-free — Phases B+ read these cached fp16 latents, never the sim.
``H`` (history) / ``T`` (horizon) DEFAULT from the typed ``Config``
(no local re-declaration of the swept knobs — review L2); per-experiment overrides via
``Config.from_yaml``.

**Tier note.** The Dataset itself is numpy-only (it emits the typed *numpy* contract objects, validated
per sample), so it imports on a torch-free box. ``torch`` enters only when a ``DataLoader`` collates
these into batched GPU tensors — that is Phase B; this module never imports torch. It is a *map-style*
dataset (``__len__`` + ``__getitem__``), which is all ``torch.utils.data.DataLoader`` requires.

**Cache on-disk format (the read-contract that A5.14 writes to).** ``<cache_dir>/manifest.json`` is the
``vllatent.manifest`` manifest; each ``entries[i]`` has ``episode_id``, ``n_frames`` (= N poses), and
``latent_path`` (relative). ``<cache_dir>/<latent_path>`` is an ``.npz`` with, per episode of N poses:

  ``latents``         (N, 196, 768) fp16  — DINOv3 latent at each rendered reference_path pose
  ``actions``         (N,)          int   — AerialVLN discrete id driving pose t -> t+1 (t=N-1 is STOP)
  ``deltas``          (N, 4)        f32   — GT 4-DoF body delta (dx,dy,dz,dyaw) for transition t
  ``lang_tokens``     (M, 768)      fp16  — frozen text-tower tokens for the instruction (per episode)
  ``waypoint_4dof``   (N, 4)        f32   — OracleTarget: 6->4-projected teacher waypoint
  ``teacher_pose6``   (N, 6)        f32   — OracleTarget: raw teacher pose [roll,yaw,pitch,x,y,z]
  ``rollpitch_resid`` (N,)          f32   — OracleTarget: |roll|+|pitch| audit (>= 0)
  ``disagreement``    (N,)          f32   — OracleTarget: K-rollout spread scalar (>= 0)
  ``disagreement``    (N,)          f32   — OracleTarget: K-rollout spread scalar (>= 0)

A training sample exists for each transition ``t in [0, N-2]`` (the terminal STOP at ``t=N-1`` has no
stored next pose, so it yields no ``z_next`` and is excluded). The per-transition arrays are stored
pose-aligned at length N; the terminal slot is unused.

CLI:  ``python -m vllatent.data inspect --cache <dir> --n 4``   (A5.16 inspects the real dump)

See plans/phase-a5-replan-postpivot.md steps A5.15/A5.16.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vllatent.config import Config
from vllatent.data.base_loader import LatentDatasetBase
from vllatent.schemas import (
    DELTA_DTYPE,
    DOF,
    EMBED_DIM,
    HISTORY,
    LATENT_DTYPE,
    MASK_DTYPE,
    PATCH_TOKENS,
    TEACHER_DOF,
    OracleTarget,
    StepSample,
)

_LATENTS = "latents"
_ACTIONS = "actions"
_DELTAS = "deltas"
_LANG = "lang_tokens"

SOURCE_AERIALVLN = "aerialvln"
SOURCE_WILD_VIDEO = "wild_video"


def _detect_source_type(cache_dir: Path) -> str:
    """Read manifest.json and return the source_type (default: aerialvln)."""
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return SOURCE_AERIALVLN
    manifest = json.loads(manifest_path.read_text())
    ds = manifest.get("dataset", {})
    return ds.get("source_type", SOURCE_AERIALVLN)


class CachedLatentDataset(LatentDatasetBase):
    """Map-style Dataset over a latent cache; emits ``(StepSample, OracleTarget)``.

    Handles both AerialVLN episode caches (with actions/lang_tokens/teacher arrays)
    and wild-video ingest caches (with quality/VO metadata). Source type is
    auto-detected from ``manifest.json``'s ``dataset.source_type`` field.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        history: int | None = None,
        horizon: int | None = None,
        config: Config | None = None,
        require_quality: bool = False,
    ) -> None:
        cfg = config if config is not None else Config()
        h = cfg.predictor.history if history is None else history
        self.horizon = cfg.predictor.horizon if horizon is None else horizon
        self.require_quality = require_quality
        self._source_type = _detect_source_type(Path(cache_dir))
        if h != HISTORY:
            raise ValueError(
                f"history={h} but StepSample locks the cached history window to "
                f"HISTORY={HISTORY} (arch-locked); a different H needs a schema change, not a loader flag."
            )
        super().__init__(cache_dir, h)

    @property
    def source_type(self) -> str:
        return self._source_type

    @property
    def n_episodes(self) -> int:
        return self.n_entries

    def _build_index(self) -> list[tuple[int, int]]:
        index: list[tuple[int, int]] = []
        for ep_i, entry in enumerate(self._entries):
            n_frames = int(entry["n_frames"])

            if self._source_type == SOURCE_WILD_VIDEO and self.require_quality:
                npz_path = self.cache_dir / entry["latent_path"]
                if npz_path.exists():
                    with np.load(str(npz_path)) as data:
                        qm = data["quality_mask"]
                    for t in range(n_frames - 1):
                        if qm[t] and qm[t + 1]:
                            index.append((ep_i, t))
                    continue

            for t in range(n_frames - 1):
                index.append((ep_i, t))
        return index

    def __getitem__(self, i: int) -> tuple[StepSample, OracleTarget]:
        ep_i, t = self._index[i]
        ep = self._load_entry_arrays(ep_i)
        latents = ep[_LATENTS]

        history, history_mask = self._pad_history(latents, t)

        if self._source_type == SOURCE_WILD_VIDEO:
            return self._getitem_wild_video(ep, latents, t, history, history_mask)
        return self._getitem_aerialvln(ep, latents, t, history, history_mask)

    def _getitem_aerialvln(
        self,
        ep: dict[str, np.ndarray],
        latents: np.ndarray,
        t: int,
        history: np.ndarray,
        history_mask: np.ndarray,
    ) -> tuple[StepSample, OracleTarget]:
        lang = ep[_LANG].astype(LATENT_DTYPE)
        lang_mask = np.ones((lang.shape[0],), dtype=MASK_DTYPE)

        step = StepSample(
            z_t=latents[t].astype(LATENT_DTYPE),
            history_latents=history,
            history_mask=history_mask,
            lang_tokens=lang,
            lang_mask=lang_mask,
            action_id=int(ep[_ACTIONS][t]),
            z_next=latents[t + 1].astype(LATENT_DTYPE),
            delta_4dof=ep[_DELTAS][t].astype(DELTA_DTYPE),
        )
        oracle = OracleTarget(
            waypoint_4dof=ep["waypoint_4dof"][t].astype(DELTA_DTYPE),
            teacher_pose6=ep["teacher_pose6"][t].astype(np.float32),
            rollpitch_resid=float(ep["rollpitch_resid"][t]),
            disagreement=float(ep["disagreement"][t]),
        )
        return step, oracle

    def _getitem_wild_video(
        self,
        ep: dict[str, np.ndarray],
        latents: np.ndarray,
        t: int,
        history: np.ndarray,
        history_mask: np.ndarray,
    ) -> tuple[StepSample, OracleTarget]:
        lang = np.zeros((1, EMBED_DIM), dtype=LATENT_DTYPE)
        lang_mask = np.zeros((1,), dtype=MASK_DTYPE)

        vo_conf = float(ep["vo_confidence"][t]) if "vo_confidence" in ep else None
        fq = float(ep["frame_quality"][t]) if "frame_quality" in ep else None
        dt = None
        if "timestamps" in ep:
            dt = float(ep["timestamps"][t + 1] - ep["timestamps"][t])

        delta = ep[_DELTAS][t].astype(DELTA_DTYPE)

        step = StepSample(
            z_t=latents[t].astype(LATENT_DTYPE),
            history_latents=history,
            history_mask=history_mask,
            lang_tokens=lang,
            lang_mask=lang_mask,
            action_id=0,
            z_next=latents[t + 1].astype(LATENT_DTYPE),
            delta_4dof=delta,
            vo_confidence=vo_conf,
            frame_quality=fq,
            dt_seconds=dt,
        )
        # Wild-video has no teacher — zeros are placeholders, not real oracle data.
        oracle = OracleTarget(
            waypoint_4dof=delta.copy(),
            teacher_pose6=np.zeros(TEACHER_DOF, dtype=np.float32),
            rollpitch_resid=0.0,
            disagreement=0.0,
        )
        return step, oracle


def inspect_cache(cache_dir: str | Path, n: int = 4) -> int:
    """Print the first ``n`` distillation samples of a cache (A5.16's real-dump inspection)."""
    ds = CachedLatentDataset(cache_dir)
    print(f"cache {cache_dir}: {ds.n_episodes} episodes, {len(ds)} transitions (H={ds.history})")
    for i in range(min(n, len(ds))):
        step, oracle = ds[i]
        print(
            f"[{i}] z_t {tuple(step.z_t.shape)} {step.z_t.dtype} action={step.action_id} "
            f"hist_mask={step.history_mask.tolist()} lang={tuple(step.lang_tokens.shape)} "
            f"waypoint={np.round(oracle.waypoint_4dof, 3).tolist()} "
            f"disagree={oracle.disagreement:.4f}"
        )
    return 0


__all__ = ["CachedLatentDataset", "inspect_cache"]
