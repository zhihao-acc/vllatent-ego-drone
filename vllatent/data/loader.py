"""Cached-latent / distillation torch Dataset (TORCH tier) — Phase-A step A5.15 (was step 10).

A torch Dataset over a cached-latent directory that emits the StepSample student-inputs +
(post-pivot) the OracleTarget teacher-targets. Builds history windows (H) + horizon targets
(T), masks episode boundaries, loads fp16 latents. Training is sim-free (latents are cached).

H (history) and T (horizon) DEFAULT from the typed Config (vllatent.config) — no local
re-declaration of the swept knobs (review L2); per-experiment overrides come via
``Config.from_yaml``. torch import is LAZY. STUB at scaffold time; implemented in A5.15.

CLI:  python -m vllatent.data inspect --cache <dir> --n 4

See plans/phase-a5-replan-postpivot.md steps A5.15/A5.16.
"""
from __future__ import annotations

from vllatent.config import Config


class CachedLatentDataset:  # pragma: no cover - implemented in A5.15
    """torch Dataset over a cached-latent dir. Lazy-imports torch; H/T come from Config."""

    def __init__(
        self,
        cache_dir: str,
        history: int | None = None,
        horizon: int | None = None,
        config: Config | None = None,
    ) -> None:
        cfg = config if config is not None else Config()
        self.cache_dir = cache_dir
        self.history = cfg.predictor.history if history is None else history
        self.horizon = cfg.predictor.horizon if horizon is None else horizon
        raise NotImplementedError("CachedLatentDataset lands in Phase-A step A5.15")


__all__ = ["CachedLatentDataset"]
