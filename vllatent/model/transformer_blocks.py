"""Shared transformer primitives for the active B3 latent world model."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMProjection(nn.Module):
    """Project a conditioning signal to scale and shift vectors."""

    def __init__(self, cond_dim: int, model_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 2 * model_dim),
        )
        final = self.net[-1]
        if not isinstance(final, nn.Linear):  # pragma: no cover - construction invariant
            raise TypeError("FiLMProjection must end in a linear layer")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale, shift = self.net(cond).chunk(2, dim=-1)
        return scale, shift


class PredictorBlock(nn.Module):
    """Transformer block with FiLM-modulated normalization and SDPA attention."""

    def __init__(self, dim: int, heads: int, mlp_ratio: int, dropout: float) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim must be divisible by heads, got {dim=} {heads=}")
        self.heads = heads
        self.head_dim = dim // heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_dropout = dropout
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        scale1: torch.Tensor,
        shift1: torch.Tensor,
        scale2: torch.Tensor,
        shift2: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, dim = x.shape
        hidden = self.norm1(x)
        hidden = hidden * (1 + scale1) + shift1

        qkv = self.qkv(hidden).reshape(
            batch_size,
            sequence_length,
            3,
            self.heads,
            self.head_dim,
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        hidden = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        hidden = hidden.transpose(1, 2).reshape(batch_size, sequence_length, dim)
        x = x + self.out_proj(hidden)

        hidden = self.norm2(x)
        hidden = hidden * (1 + scale2) + shift2
        return x + self.mlp(hidden)


__all__ = ["FiLMProjection", "PredictorBlock"]
