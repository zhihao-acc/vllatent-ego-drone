"""Cached-latent / distillation map-style Dataset (TORCH tier) — Phase-A step A5.15 (was step 10).

Emits the per-step distillation pair ``(StepSample student-inputs, OracleTarget teacher-targets)``
(the A5.9 seam) over a render-once latent cache. Training is sim-free — Phases B+ read these cached
fp16 latents, never the sim. ``H`` (history) / ``T`` (horizon) DEFAULT from the typed ``Config``
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
  ``vjepa_surprise``  (N,)          f32   — OracleTarget: V-JEPA-2 surprise (>= 0)

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
from vllatent.schemas import (
    DELTA_DTYPE,
    EMBED_DIM,
    HISTORY,
    LATENT_DTYPE,
    MASK_DTYPE,
    PATCH_TOKENS,
    OracleTarget,
    StepSample,
)

# The per-episode .npz array keys (the contract A5.14 writes / this loader reads).
_LATENTS = "latents"
_ACTIONS = "actions"
_DELTAS = "deltas"
_LANG = "lang_tokens"
_ORACLE_KEYS = ("waypoint_4dof", "teacher_pose6", "rollpitch_resid", "disagreement", "vjepa_surprise")


class CachedLatentDataset:
    """Map-style Dataset over a render-once latent cache; emits ``(StepSample, OracleTarget)``.

    Lazy per-episode load (an episode's arrays are read from disk on access, not all up front), so a
    large cache is not pulled into RAM at construction. ``history`` / ``horizon`` default from
    ``Config`` (the single source of truth) unless explicitly overridden.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        history: int | None = None,
        horizon: int | None = None,
        config: Config | None = None,
    ) -> None:
        cfg = config if config is not None else Config()
        self.cache_dir = Path(cache_dir)
        self.history = cfg.predictor.history if history is None else history
        self.horizon = cfg.predictor.horizon if horizon is None else horizon
        # StepSample fixes its history window at the arch-locked HISTORY constant, so the cached
        # window MUST match. Fail fast here with a clear message rather than a deep StepSample error
        # in __getitem__ (a Config that sweeps predictor.history != HISTORY needs a schema change).
        if self.history != HISTORY:
            raise ValueError(
                f"history={self.history} but StepSample locks the cached history window to "
                f"HISTORY={HISTORY} (arch-locked); a different H needs a schema change, not a loader flag."
            )

        manifest_path = self.cache_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        self._entries: list[dict] = list(manifest.get("entries", []))
        # Flat index of (episode_idx, t) for every valid transition t in [0, N-2].
        self._index: list[tuple[int, int]] = []
        for ep_i, entry in enumerate(self._entries):
            n_frames = int(entry["n_frames"])
            for t in range(n_frames - 1):  # terminal STOP (t = N-1) has no z_next -> excluded
                self._index.append((ep_i, t))

    def __len__(self) -> int:
        return len(self._index)

    @property
    def n_episodes(self) -> int:
        return len(self._entries)

    def _load_episode(self, ep_i: int) -> dict[str, np.ndarray]:
        entry = self._entries[ep_i]
        with np.load(self.cache_dir / entry["latent_path"]) as data:
            return {k: data[k] for k in data.files}

    def __getitem__(self, i: int) -> tuple[StepSample, OracleTarget]:
        ep_i, t = self._index[i]
        ep = self._load_episode(ep_i)
        latents = ep[_LATENTS]  # (N, 196, 768) fp16

        # History window ending at t (inclusive), LEFT zero-padded at the block-causal episode start.
        h = self.history
        history = np.zeros((h, PATCH_TOKENS, EMBED_DIM), dtype=LATENT_DTYPE)
        history_mask = np.zeros((h,), dtype=MASK_DTYPE)
        for j in range(h):
            src = t - (h - 1 - j)  # j = h-1 -> src = t (current frame); earlier j -> older frames
            if src >= 0:
                history[j] = latents[src]
                history_mask[j] = True

        lang = ep[_LANG].astype(LATENT_DTYPE)
        lang_mask = np.ones((lang.shape[0],), dtype=MASK_DTYPE)  # cached per-episode tokens: all real

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
            vjepa_surprise=float(ep["vjepa_surprise"][t]),
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
            f"disagree={oracle.disagreement:.4f} surprise={oracle.vjepa_surprise:.4f}"
        )
    return 0


__all__ = ["CachedLatentDataset", "inspect_cache"]
