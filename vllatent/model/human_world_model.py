"""B3 human-conditioned latent world model (TORCH tier).

The B3 path consumes observed DINO history/current latents plus candidate
future 6-D plan tokens. Future latents and person labels are training targets
only and are deliberately absent from ``forward``.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

from vllatent.model.transformer_blocks import FiLMProjection, PredictorBlock
from vllatent.plan_tokens import PLAN_TOKEN_DIM
from vllatent.schemas import EMBED_DIM, HISTORY, PATCH_TOKENS


class HumanWorldModelOutput(NamedTuple):
    """B3 rollout outputs; cycle verification requires an explicit transition."""

    predicted_latents: torch.Tensor       # (B, T, 196, D)
    predicted_person_state: torch.Tensor  # (B, T, 4): cx, cy, log_h, visibility_logit


def apply_whole_plan_dropout(
    planned_actions: torch.Tensor,
    *,
    dropout_p: float,
    training: bool,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop complete plan sequences and return the exact retained-step mask.

    Whole-window dropout makes the all-zero null-plan branch part of training and
    lets the loss exclude actions that were unavailable to the predictor.
    """
    if planned_actions.ndim != 3 or planned_actions.shape[-1] != PLAN_TOKEN_DIM:
        raise ValueError(
            f"planned_actions: expected (B,T,{PLAN_TOKEN_DIM}), got {planned_actions.shape}"
        )
    if not 0.0 <= dropout_p <= 1.0:
        raise ValueError(f"dropout_p must be in [0, 1], got {dropout_p}")

    batch_size, horizon, _ = planned_actions.shape
    if not training or dropout_p == 0.0:
        keep_mask = torch.ones(
            batch_size,
            horizon,
            device=planned_actions.device,
            dtype=torch.bool,
        )
        return planned_actions, keep_mask
    if dropout_p == 1.0:
        keep_rows = torch.zeros(
            batch_size,
            1,
            device=planned_actions.device,
            dtype=torch.bool,
        )
    else:
        keep_rows = torch.rand(
            batch_size,
            1,
            device=planned_actions.device,
            generator=generator,
        ) >= dropout_p
    keep_mask = keep_rows.expand(-1, horizon)
    conditioned = planned_actions * keep_mask.unsqueeze(-1).to(dtype=planned_actions.dtype)
    return conditioned, keep_mask


class TransitionPlanVerifier(nn.Module):
    """Recover physical plan fields from an action-blind latent transition."""

    def __init__(
        self,
        dim: int = EMBED_DIM,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if hidden_dim < PLAN_TOKEN_DIM - 1:
            raise ValueError(
                f"hidden_dim must be >= {PLAN_TOKEN_DIM - 1}, got {hidden_dim}"
            )
        self.dim = dim
        self.net = nn.Sequential(
            nn.LayerNorm(2 * dim),
            nn.Linear(2 * dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, PLAN_TOKEN_DIM - 1),
        )

    def forward(
        self,
        previous_latents: torch.Tensor,
        next_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Predict five physical fields from consecutive `(B,T,P,D)` states."""
        if previous_latents.ndim != 4:
            raise ValueError(
                "previous_latents: expected (B,T,P,D), "
                f"got {previous_latents.shape}"
            )
        if next_latents.ndim != 4:
            raise ValueError(
                f"next_latents: expected (B,T,P,D), got {next_latents.shape}"
            )
        if previous_latents.shape != next_latents.shape:
            raise ValueError(
                "previous/next latent shape mismatch: "
                f"{previous_latents.shape} vs {next_latents.shape}"
            )
        if previous_latents.shape[-1] != self.dim:
            raise ValueError(
                f"latent last dimension: expected {self.dim}, got {previous_latents.shape[-1]}"
            )

        previous = previous_latents.float().mean(dim=2)
        transition = (next_latents.float() - previous_latents.float()).mean(dim=2)
        return self.net(torch.cat([previous, transition], dim=-1))


class PlanConditionedLatentPredictor(nn.Module):
    """Depth-configurable block-causal latent predictor with per-step plan conditioning."""

    def __init__(
        self,
        dim: int = EMBED_DIM,
        depth: int = 6,
        heads: int = 12,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        history: int = HISTORY,
        horizon: int = 8,
        plan_dim: int = PLAN_TOKEN_DIM,
        patch_tokens: int = PATCH_TOKENS,
    ) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim must be divisible by heads, got {dim=} {heads=}")
        if history < 1:
            raise ValueError(f"history must be >= 1, got {history}")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if plan_dim != PLAN_TOKEN_DIM:
            raise ValueError(f"plan_dim is locked to {PLAN_TOKEN_DIM}, got {plan_dim}")
        if patch_tokens < 1:
            raise ValueError(f"patch_tokens must be >= 1, got {patch_tokens}")

        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.history = history
        self.horizon = horizon
        self.plan_dim = plan_dim
        self.n_patches = patch_tokens

        self.blocks = nn.ModuleList(
            [PredictorBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.plan_step_embed = nn.Sequential(
            nn.Linear(plan_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.dt_step_embed = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.plan_film = nn.ModuleList([FiLMProjection(plan_dim, dim) for _ in range(depth)])
        self.dt_film = nn.ModuleList([FiLMProjection(1, dim) for _ in range(depth)])

        n_temporal = history + 1 + horizon
        self.temporal_embed = nn.Parameter(torch.zeros(1, n_temporal, 1, dim))
        nn.init.trunc_normal_(self.temporal_embed, std=0.02)
        self.patch_query_embed = nn.Parameter(torch.zeros(1, 1, patch_tokens, dim))
        nn.init.trunc_normal_(self.patch_query_embed, std=0.02)

        self.output_norm = nn.LayerNorm(dim)
        self.residual_out = nn.Linear(dim, dim)
        nn.init.normal_(self.residual_out.weight, std=1e-3)
        nn.init.zeros_(self.residual_out.bias)

    def _build_block_causal_mask(self, n_frames: int, device: torch.device) -> torch.Tensor:
        """Return SDPA boolean attention mask where True means "can attend"."""
        n_visible = self.history + 1
        frame_mask = torch.zeros(n_frames, n_frames, device=device, dtype=torch.bool)
        for i in range(n_frames):
            for j in range(n_frames):
                if j < n_visible or j <= i:
                    frame_mask[i, j] = True
        return frame_mask.repeat_interleave(self.n_patches, dim=0).repeat_interleave(self.n_patches, dim=1)

    def _validate_inputs(
        self,
        history_latents: torch.Tensor,
        z_t: torch.Tensor,
        history_mask: torch.Tensor,
        planned_actions: torch.Tensor,
        dt_seconds: torch.Tensor,
    ) -> None:
        batch_size = z_t.shape[0] if z_t.ndim >= 1 else -1
        if history_latents.shape != (batch_size, self.history, self.n_patches, self.dim):
            raise ValueError(
                "history_latents: expected "
                f"{(batch_size, self.history, self.n_patches, self.dim)}, got {history_latents.shape}"
            )
        if z_t.shape != (batch_size, self.n_patches, self.dim):
            raise ValueError(f"z_t: expected {(batch_size, self.n_patches, self.dim)}, got {z_t.shape}")
        if history_mask.shape != (batch_size, self.history):
            raise ValueError(f"history_mask: expected {(batch_size, self.history)}, got {history_mask.shape}")
        if planned_actions.shape != (batch_size, self.horizon, self.plan_dim):
            raise ValueError(
                f"planned_actions: expected {(batch_size, self.horizon, self.plan_dim)}, got {planned_actions.shape}"
            )
        if dt_seconds.shape != (batch_size, self.horizon):
            raise ValueError(f"dt_seconds: expected {(batch_size, self.horizon)}, got {dt_seconds.shape}")

    def forward(
        self,
        history_latents: torch.Tensor,
        z_t: torch.Tensor,
        history_mask: torch.Tensor,
        planned_actions: torch.Tensor,
        dt_seconds: torch.Tensor,
    ) -> torch.Tensor:
        """Predict future DINO latents from observed history/current state and candidate plans."""
        self._validate_inputs(history_latents, z_t, history_mask, planned_actions, dt_seconds)

        batch_size = z_t.shape[0]
        device = z_t.device
        dtype = self.temporal_embed.dtype
        history = history_latents.to(device=device, dtype=dtype)
        current = z_t.to(device=device, dtype=dtype)
        plan = planned_actions.to(device=device, dtype=dtype)
        dt = dt_seconds.to(device=device, dtype=dtype).unsqueeze(-1)

        history_valid = history_mask.to(device=device, dtype=torch.bool)
        history = history * history_valid[:, :, None, None].to(dtype=history.dtype)

        step_embed = self.plan_step_embed(plan) + self.dt_step_embed(dt)
        horizon_tokens = (
            current[:, None, :, :]
            + self.patch_query_embed
            + step_embed[:, :, None, :]
        )

        all_frames = torch.cat([history, current.unsqueeze(1), horizon_tokens], dim=1)
        n_frames = all_frames.shape[1]
        all_frames = all_frames + self.temporal_embed[:, :n_frames]

        plan_condition = torch.zeros(
            batch_size,
            n_frames,
            self.plan_dim,
            device=device,
            dtype=dtype,
        )
        dt_condition = torch.zeros(
            batch_size,
            n_frames,
            1,
            device=device,
            dtype=dtype,
        )
        plan_condition[:, self.history + 1 :] = plan
        dt_condition[:, self.history + 1 :] = dt

        x = all_frames.reshape(batch_size, n_frames * self.n_patches, self.dim)
        attn_mask = self._build_block_causal_mask(n_frames, device)

        for plan_film, dt_film, block in zip(self.plan_film, self.dt_film, self.blocks, strict=True):
            p_scale, p_shift = plan_film(plan_condition)
            d_scale, d_shift = dt_film(dt_condition)
            frame_scale = p_scale + d_scale
            frame_shift = p_shift + d_shift
            scale = frame_scale[:, :, None, :].expand(
                -1,
                -1,
                self.n_patches,
                -1,
            ).reshape(batch_size, n_frames * self.n_patches, self.dim)
            shift = frame_shift[:, :, None, :].expand(
                -1,
                -1,
                self.n_patches,
                -1,
            ).reshape(batch_size, n_frames * self.n_patches, self.dim)
            x = block(x, attn_mask, scale, shift, scale, shift)

        x = self.output_norm(x)
        x = x.reshape(batch_size, n_frames, self.n_patches, self.dim)
        predicted = x[:, self.history + 1 :]
        delta = self.residual_out(predicted)
        base = current.unsqueeze(1).expand(-1, self.horizon, -1, -1)
        return base + delta


class HumanWorldModel(nn.Module):
    """B3 wrapper: plan-conditioned latent rollout plus an explicit transition verifier."""

    def __init__(
        self,
        dim: int = EMBED_DIM,
        depth: int = 6,
        heads: int = 12,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        history: int = HISTORY,
        horizon: int = 8,
        hidden_dim: int = 256,
        patch_tokens: int = PATCH_TOKENS,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.history = history
        self.horizon = horizon
        self.predictor = PlanConditionedLatentPredictor(
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            history=history,
            horizon=horizon,
            patch_tokens=patch_tokens,
        )
        self.person_state_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )
        self.transition_plan_verifier = TransitionPlanVerifier(
            dim=dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        history_latents: torch.Tensor,
        z_t: torch.Tensor,
        history_mask: torch.Tensor,
        planned_actions: torch.Tensor,
        dt_seconds: torch.Tensor,
    ) -> HumanWorldModelOutput:
        """Run the B3 model without accepting future targets or person labels."""
        predicted_latents = self.predictor(
            history_latents=history_latents,
            z_t=z_t,
            history_mask=history_mask,
            planned_actions=planned_actions,
            dt_seconds=dt_seconds,
        )
        pooled = predicted_latents.float().mean(dim=2)
        return HumanWorldModelOutput(
            predicted_latents=predicted_latents,
            predicted_person_state=self.person_state_head(pooled),
        )

    def recover_plan(
        self,
        previous_latents: torch.Tensor,
        next_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Recover physical actions from an explicitly supplied transition pair."""
        return self.transition_plan_verifier(previous_latents, next_latents)


def count_parameters(module: nn.Module) -> int:
    """Count all parameters for exact B3.5 logging."""
    return sum(p.numel() for p in module.parameters())


__all__ = [
    "HumanWorldModel",
    "HumanWorldModelOutput",
    "PlanConditionedLatentPredictor",
    "TransitionPlanVerifier",
    "apply_whole_plan_dropout",
    "count_parameters",
]
