"""Waypoint head (TORCH tier) — B1.16.

Simple MLP: D → 256 → 128 → 4.  Takes the predictor output (B, T, D) and
produces predicted 4-DoF deltas (B, T, 4).

Trust head was removed (commit 125576f). No stub needed.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from vllatent.scale_free_targets import SCALE_FREE_ACTION_DIM
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


class ScaleFreeActionHead(nn.Module):
    """MLP decoder for B2 scale-free action residuals: (B, T, D) -> (B, T, 4)."""

    def __init__(
        self,
        dim: int = EMBED_DIM,
        hidden_dim: int = 256,
        action_dim: int = SCALE_FREE_ACTION_DIM,
        final_init_std: float = 1e-2,
    ) -> None:
        super().__init__()
        if action_dim != SCALE_FREE_ACTION_DIM:
            raise ValueError(f"action_dim is locked to {SCALE_FREE_ACTION_DIM}, got {action_dim}")
        if hidden_dim < action_dim:
            raise ValueError(f"hidden_dim must be >= {action_dim}, got {hidden_dim}")
        if final_init_std < 0:
            raise ValueError(f"final_init_std must be >= 0, got {final_init_std}")

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )
        final = self.net[-1]
        if isinstance(final, nn.Linear):
            if final_init_std == 0.0:
                nn.init.zeros_(final.weight)
            else:
                nn.init.normal_(final.weight, std=final_init_std)
            nn.init.zeros_(final.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
