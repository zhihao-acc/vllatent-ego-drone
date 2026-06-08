"""Cached-latent torch Dataset / loader (TORCH tier) — Phase-A step 10 (DoD item 3).

A torch Dataset over a cached-latent directory that emits the full StepSample tuple
(vllatent.schemas): (z_t, history H=3, lang_tokens, action_id, z_next, delta_4dof,
future_frame_rgb). Builds history windows (H=3) + horizon targets (T=4), masks
episode boundaries, loads fp16 latents. Training is sim-free (latents are cached).

torch import is LAZY. STUB at scaffold time; implemented in step 10.

CLI:  python -m vllatent.data inspect --cache <dir> --n 4

See plans/phase-a-data-and-io-contract.md step 10/11.
"""
from __future__ import annotations

HISTORY = 3   # H
HORIZON = 4   # T


class CachedLatentDataset:  # pragma: no cover - implemented in step 10
    """torch Dataset over a cached-latent dir. Lazy-imports torch."""

    def __init__(self, cache_dir: str, history: int = HISTORY, horizon: int = HORIZON) -> None:
        raise NotImplementedError("CachedLatentDataset lands in Phase-A step 10")


__all__ = ["CachedLatentDataset", "HISTORY", "HORIZON"]
