"""Block-causal ViT latent predictor + FiLM conditioning (TORCH tier) — B1.15.

Input:  history_latents (B, H, P, D) + z_t (B, P, D) + action (B, 4) + dt (B, T) f32
Output: predicted_latents (B, T, P, D) — future DINOv3 patch tokens

Block-causal mask: each horizon step t+k attends to [history..z_t, t+1..t+k-1]
but NOT to t+k+1..t+T. History and z_t are always visible.

Two FiLM conditioning sources:
  - Action FiLM: projects action (4,) → (scale, shift) per block
  - dt FiLM: projects dt_seconds scalar → (scale, shift) per block
Both applied after LayerNorm via adaptive modulation.

torch imported LAZILY (entire module behind function-level imports in the
public API). The module itself uses torch at top level for nn.Module — this
is fine because it's TORCH tier (only imported where torch is available).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllatent.schemas import DOF, EMBED_DIM, HISTORY, HORIZON, PATCH_TOKENS

PREDICTION_MODES = ("absolute", "residual")


class FiLMProjection(nn.Module):
    """Project a conditioning signal to (scale, shift) for FiLM modulation."""

    def __init__(self, cond_dim: int, model_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 2 * model_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(cond)
        scale, shift = out.chunk(2, dim=-1)
        return scale, shift


class PredictorBlock(nn.Module):
    """One transformer block with FiLM-modulated LayerNorm.

    Uses F.scaled_dot_product_attention (flash/memory-efficient kernels)
    instead of nn.MultiheadAttention to avoid materializing the full
    S×S attention matrix (1568×1568 at H=3,T=4,P=196).
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: int, dropout: float) -> None:
        super().__init__()
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
        B, S, D = x.shape
        h = self.norm1(x)
        h = h * (1 + scale1) + shift1

        qkv = self.qkv(h).reshape(B, S, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, S, head_dim)
        q, k, v = qkv.unbind(0)

        h = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        h = h.transpose(1, 2).reshape(B, S, D)
        h = self.out_proj(h)
        x = x + h

        h = self.norm2(x)
        h = h * (1 + scale2) + shift2
        h = self.mlp(h)
        x = x + h
        return x


class LatentPredictor(nn.Module):
    """Block-causal ViT predictor for future DINOv3 latents.

    Parameters
    ----------
    dim : int
        Model dimension (= EMBED_DIM, typically 768).
    depth : int
        Number of transformer blocks.
    heads : int
        Number of attention heads.
    mlp_ratio : int
        FFN expansion ratio.
    dropout : float
        Dropout rate.
    history : int
        Number of history frames (H).
    horizon : int
        Number of future prediction steps (T).
    use_action_film : bool
        If False the action-FiLM path is skipped (dt-FiLM only) — the
        action-free predictor ablation (B-1 sub-decision). Default True.
    """

    def __init__(
        self,
        dim: int = EMBED_DIM,
        depth: int = 6,
        heads: int = 12,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        history: int = HISTORY,
        horizon: int = HORIZON,
        use_action_film: bool = True,
        prediction_mode: str = "absolute",
    ) -> None:
        super().__init__()
        if prediction_mode not in PREDICTION_MODES:
            raise ValueError(f"prediction_mode must be one of {PREDICTION_MODES}, got {prediction_mode!r}")
        self.dim = dim
        self.depth = depth
        self.history = history
        self.horizon = horizon
        self.use_action_film = use_action_film
        self.prediction_mode = prediction_mode
        self.n_patches = PATCH_TOKENS

        self.blocks = nn.ModuleList([
            PredictorBlock(dim, heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.action_film = nn.ModuleList([
            FiLMProjection(DOF, dim) for _ in range(depth)
        ])
        self.dt_film = nn.ModuleList([
            FiLMProjection(1, dim) for _ in range(depth)
        ])

        n_temporal = history + 1 + horizon
        self.temporal_embed = nn.Parameter(
            torch.zeros(1, n_temporal, 1, dim)
        )
        nn.init.trunc_normal_(self.temporal_embed, std=0.02)

        self.output_norm = nn.LayerNorm(dim)
        if self.prediction_mode == "residual":
            self.residual_out = nn.Linear(dim, dim)
            nn.init.zeros_(self.residual_out.weight)
            nn.init.zeros_(self.residual_out.bias)

    def _build_block_causal_mask(
        self, n_frames: int, device: torch.device
    ) -> torch.Tensor:
        """Build block-causal attention mask for SDPA.

        Returns a boolean mask where True = CAN attend (SDPA convention).
        Each frame's patches attend to all patches in the same or earlier
        frames. History + z_t are always visible. Horizon frames are
        causally masked (can't see future horizon steps).
        """
        n_visible = self.history + 1
        frame_mask = torch.zeros(n_frames, n_frames, device=device, dtype=torch.bool)
        for i in range(n_frames):
            for j in range(n_frames):
                if j < n_visible or j <= i:
                    frame_mask[i, j] = True
        return frame_mask.repeat_interleave(self.n_patches, dim=0) \
                         .repeat_interleave(self.n_patches, dim=1)

    def forward(
        self,
        history_latents: torch.Tensor,
        z_t: torch.Tensor,
        action_4dof: torch.Tensor,
        dt_seconds: torch.Tensor,
        history_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        history_latents : (B, H, P, D)
        z_t : (B, P, D)
        action_4dof : (B, 4) f32
        dt_seconds : (B, T) f32
        history_mask : (B, H) bool, optional — True=real, False=padding

        Returns
        -------
        predicted_latents : (B, T, P, D)
        """
        B = z_t.shape[0]
        device = z_t.device

        horizon_tokens = torch.zeros(
            B, self.horizon, self.n_patches, self.dim,
            device=device, dtype=z_t.dtype,
        )

        all_frames = torch.cat([
            history_latents,
            z_t.unsqueeze(1),
            horizon_tokens,
        ], dim=1)

        n_frames = all_frames.shape[1]
        te = self.temporal_embed[:, :n_frames]

        if history_mask is not None:
            mask_4d = history_mask[:, :, None, None].to(dtype=te.dtype, device=device)
            te_hist = te[:, :self.history] * mask_4d
            te_rest = te[:, self.history:].expand(B, -1, -1, -1)
            te = torch.cat([te_hist, te_rest], dim=1)

        all_frames = all_frames + te

        x = all_frames.reshape(B, n_frames * self.n_patches, self.dim)

        attn_mask = self._build_block_causal_mask(n_frames, device)

        dt_mean = dt_seconds.mean(dim=1, keepdim=True)

        for i, block in enumerate(self.blocks):
            d_scale, d_shift = self.dt_film[i](dt_mean)
            if self.use_action_film:
                a_scale, a_shift = self.action_film[i](action_4dof)
                scale1 = (a_scale + d_scale).unsqueeze(1)
                shift1 = (a_shift + d_shift).unsqueeze(1)
            else:
                # action-free ablation: dt-FiLM only (action_film params unused, no grad)
                scale1 = d_scale.unsqueeze(1)
                shift1 = d_shift.unsqueeze(1)
            scale2 = scale1
            shift2 = shift1

            x = block(x, attn_mask, scale1, shift1, scale2, shift2)

        x = self.output_norm(x)

        x = x.reshape(B, n_frames, self.n_patches, self.dim)
        predicted = x[:, self.history + 1:]

        if self.prediction_mode == "residual":
            delta = self.residual_out(predicted)
            base = z_t.unsqueeze(1).expand(-1, self.horizon, -1, -1)
            return base + delta

        return predicted


__all__ = ["LatentPredictor", "PREDICTION_MODES"]
