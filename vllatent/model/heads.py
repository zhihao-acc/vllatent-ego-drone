"""Waypoint head (TORCH tier) — B1.16.

Simple MLP: D → 256 → 128 → 4.  Takes the predictor output (B, T, D) and
produces predicted 4-DoF deltas (B, T, 4).

Trust head was removed (commit 125576f). No stub needed.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from vllatent.schemas import DOF, EMBED_DIM


class WaypointHead(nn.Module):
    """MLP waypoint decoder: (B, T, D) → (B, T, 4)."""

    def __init__(self, dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, DOF),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
