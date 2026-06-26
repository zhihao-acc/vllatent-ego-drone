"""Collate function for batched sports training (TORCH tier) — B1.14.

Converts numpy ``SportsSample`` instances to batched GPU tensors via a
``TrainingBatch`` NamedTuple. Used as ``collate_fn`` for
``torch.utils.data.DataLoader``.

torch is imported LAZILY (inside functions) so this module imports on a
torch-free box.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    import torch

    from vllatent.data.sports_loader import SportsSample


class TrainingBatch(NamedTuple):
    """Batched GPU tensors for one training step."""

    z_t: torch.Tensor              # (B, P, D) fp16
    history_latents: torch.Tensor  # (B, H, P, D) fp16
    history_mask: torch.Tensor     # (B, H) bool
    target_latents: torch.Tensor   # (B, T, P, D) fp16
    target_deltas: torch.Tensor    # (B, T, 4) f32
    vo_confidence: torch.Tensor    # (B, T) f32
    frame_quality: torch.Tensor    # (B,) f32
    dt_seconds: torch.Tensor       # (B, T) f32
    sample_weight: torch.Tensor    # (B,) f32


def collate_sports_batch(samples: list[SportsSample]) -> TrainingBatch:
    """Collate a list of SportsSample into a TrainingBatch of tensors."""
    import torch

    z_t = torch.from_numpy(np.stack([s.z_t for s in samples]))
    history_latents = torch.from_numpy(np.stack([s.history_latents for s in samples]))
    history_mask = torch.from_numpy(np.stack([s.history_mask for s in samples]))
    target_latents = torch.from_numpy(np.stack([s.target_latents for s in samples]))
    target_deltas = torch.from_numpy(np.stack([s.target_deltas for s in samples]))
    vo_conf = torch.from_numpy(np.stack([s.vo_confidence for s in samples]))
    dt_sec = torch.from_numpy(np.stack([s.dt_seconds for s in samples]))

    fq = torch.tensor([s.frame_quality for s in samples], dtype=torch.float32)

    weight = fq.clamp(min=0.1) * vo_conf.mean(dim=1).clamp(min=0.05)

    return TrainingBatch(
        z_t=z_t,
        history_latents=history_latents,
        history_mask=history_mask,
        target_latents=target_latents,
        target_deltas=target_deltas,
        vo_confidence=vo_conf,
        frame_quality=fq,
        dt_seconds=dt_sec,
        sample_weight=weight,
    )
