"""Direct scale-free future-action policy (TORCH tier) -- Phase B2.3.

The policy consumes frozen cached DINO history/current latents plus previous
observed scale-free motion and predicts the future scale-free action sequence.
It does not call the B1 latent predictor and does not accept future action
labels as inputs.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from vllatent.scale_free_targets import SCALE_FREE_ACTION_DIM
from vllatent.schemas import EMBED_DIM, HISTORY, HORIZON


class ScaleFreeActionPolicy(nn.Module):
    """Small direct action policy over mean-pooled DINO frame tokens."""

    def __init__(
        self,
        dim: int = EMBED_DIM,
        hidden_dim: int = 256,
        depth: int = 2,
        heads: int = 4,
        mlp_ratio: int = 2,
        dropout: float = 0.1,
        history: int = HISTORY,
        horizon: int = HORIZON,
        action_dim: int = SCALE_FREE_ACTION_DIM,
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim must be divisible by heads, got {hidden_dim=} {heads=}")
        if history < 1:
            raise ValueError(f"history must be >= 1, got {history}")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if action_dim != SCALE_FREE_ACTION_DIM:
            raise ValueError(f"action_dim is locked to {SCALE_FREE_ACTION_DIM}, got {action_dim}")

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.history = history
        self.horizon = horizon
        self.action_dim = action_dim

        self.frame_proj = nn.Linear(dim, hidden_dim)
        self.temporal_embed = nn.Parameter(torch.zeros(1, history + 1, hidden_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)

        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dt_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.horizon_embed = nn.Parameter(torch.zeros(1, horizon, hidden_dim))
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
        )

        nn.init.trunc_normal_(self.temporal_embed, std=0.02)
        nn.init.trunc_normal_(self.horizon_embed, std=0.02)

    def forward(
        self,
        history_latents: torch.Tensor,
        z_t: torch.Tensor,
        history_mask: torch.Tensor,
        last_action_scale_free: torch.Tensor,
        dt_seconds: torch.Tensor,
    ) -> torch.Tensor:
        """Predict future scale-free actions.

        Parameters
        ----------
        history_latents : (B, H, P, D)
        z_t : (B, P, D)
        history_mask : (B, H) bool, True=real history frame
        last_action_scale_free : (B, 4)
        dt_seconds : (B, T)

        Returns
        -------
        actions : (B, T, 4)
        """
        if history_latents.ndim != 4:
            raise ValueError(f"history_latents: expected (B,H,P,D), got {history_latents.shape}")
        if z_t.ndim != 3:
            raise ValueError(f"z_t: expected (B,P,D), got {z_t.shape}")
        if history_mask.shape != history_latents.shape[:2]:
            raise ValueError(f"history_mask: expected {history_latents.shape[:2]}, got {history_mask.shape}")
        if last_action_scale_free.shape != (z_t.shape[0], self.action_dim):
            raise ValueError(
                f"last_action_scale_free: expected {(z_t.shape[0], self.action_dim)}, "
                f"got {last_action_scale_free.shape}"
            )
        if dt_seconds.shape != (z_t.shape[0], self.horizon):
            raise ValueError(f"dt_seconds: expected {(z_t.shape[0], self.horizon)}, got {dt_seconds.shape}")

        batch_size = z_t.shape[0]
        device = z_t.device
        dtype = self.temporal_embed.dtype

        history = history_latents.float().mean(dim=2)
        current = z_t.float().mean(dim=1, keepdim=True)
        frames = torch.cat([history, current], dim=1)
        frames = self.frame_proj(frames.to(dtype=dtype))

        mask = history_mask.to(device=device, dtype=torch.bool)
        current_valid = torch.ones(batch_size, 1, device=device, dtype=torch.bool)
        valid_frames = torch.cat([mask, current_valid], dim=1)
        frames = frames * valid_frames.unsqueeze(-1).to(dtype=frames.dtype)

        tokens = frames + self.temporal_embed[:, : self.history + 1]
        encoded = self.context_encoder(tokens, src_key_padding_mask=~valid_frames)
        valid_count = valid_frames.sum(dim=1, keepdim=True).clamp(min=1).to(dtype=encoded.dtype)
        context = (encoded * valid_frames.unsqueeze(-1).to(dtype=encoded.dtype)).sum(dim=1) / valid_count

        action_context = self.action_proj(last_action_scale_free.to(device=device, dtype=dtype)).unsqueeze(1)
        dt_context = self.dt_proj(dt_seconds.to(device=device, dtype=dtype).unsqueeze(-1))
        query = context.unsqueeze(1) + action_context + dt_context + self.horizon_embed[:, : self.horizon]

        return self.head(self.output_norm(query))
