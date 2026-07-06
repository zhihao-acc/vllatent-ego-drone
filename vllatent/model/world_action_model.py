"""Control-relevant world-action model (TORCH tier) -- Phase B2.10.

This is the B2 revision of the B1 model shape: a B1-style latent/world
predictor is kept as the bottleneck, and a scale-free action head decodes the
predicted world tokens into future actions.  Future actions and future target
latents are labels only; the forward path accepts observed latents and observed
motion history.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vllatent.model.action_policy import ScaleFreeActionPolicy
from vllatent.model.heads import ScaleFreeActionHead
from vllatent.model.predictor import LatentPredictor
from vllatent.scale_free_targets import SCALE_FREE_ACTION_DIM
from vllatent.schemas import EMBED_DIM

if TYPE_CHECKING:
    from vllatent.config import PredictorConfig


class WorldActionOutput:
    """Diagnostic rollout output for the B2 world-action model."""

    __slots__ = ("predicted_latents", "predicted_actions")

    def __init__(self, predicted_latents: torch.Tensor, predicted_actions: torch.Tensor) -> None:
        self.predicted_latents = predicted_latents
        self.predicted_actions = predicted_actions


class WorldActionModel(nn.Module):
    """B1-style latent predictor plus B2 scale-free action head.

    The primary ``forward`` returns action predictions so B2 action metrics can
    score this model like the direct-policy diagnostic.  ``rollout`` exposes the
    intermediate predicted latents for diagnostics or optional auxiliary losses.
    """

    def __init__(
        self,
        cfg: PredictorConfig,
        dim: int = EMBED_DIM,
        action_hidden_dim: int = 256,
        use_action_film: bool = True,
        prediction_mode: str = "residual",
        latent_residual_init_std: float = 1e-3,
        action_head_final_init_std: float = 0.0,
        use_direct_anchor: bool = False,
    ) -> None:
        super().__init__()
        if latent_residual_init_std < 0:
            raise ValueError(f"latent_residual_init_std must be >= 0, got {latent_residual_init_std}")
        if action_head_final_init_std < 0:
            raise ValueError(f"action_head_final_init_std must be >= 0, got {action_head_final_init_std}")

        self.dim = dim
        self.history = cfg.history
        self.horizon = cfg.horizon
        self.action_dim = SCALE_FREE_ACTION_DIM
        self.use_direct_anchor = use_direct_anchor
        self.direct_anchor_frozen = False

        self.history_action_proj = nn.Linear(SCALE_FREE_ACTION_DIM, dim)
        self.history_path_proj = nn.Linear(3, dim)
        self.current_action_proj = nn.Linear(SCALE_FREE_ACTION_DIM, dim)

        self.predictor = LatentPredictor(
            dim=dim,
            depth=cfg.depth,
            heads=cfg.heads,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.dropout,
            history=cfg.history,
            horizon=cfg.horizon,
            use_action_film=use_action_film,
            prediction_mode=prediction_mode,
        )
        if prediction_mode == "residual" and latent_residual_init_std > 0.0:
            nn.init.normal_(self.predictor.residual_out.weight, std=latent_residual_init_std)
            nn.init.zeros_(self.predictor.residual_out.bias)

        self.direct_anchor = (
            ScaleFreeActionPolicy(
                dim=dim,
                hidden_dim=action_hidden_dim,
                depth=max(1, min(2, cfg.depth)),
                heads=cfg.heads,
                mlp_ratio=max(1, cfg.mlp_ratio),
                dropout=cfg.dropout,
                history=cfg.history,
                horizon=cfg.horizon,
            )
            if use_direct_anchor
            else None
        )
        self.world_pool_norm = nn.LayerNorm(dim)
        self.world_pool_score = nn.Linear(dim, 1)
        nn.init.zeros_(self.world_pool_score.weight)
        nn.init.zeros_(self.world_pool_score.bias)

        self.head_last_action_proj = nn.Linear(SCALE_FREE_ACTION_DIM, dim)
        self.head_history_action_proj = nn.Linear(SCALE_FREE_ACTION_DIM, dim)
        self.head_history_path_proj = nn.Linear(3, dim)
        self.head_dt_proj = nn.Linear(1, dim)
        self.head_horizon_embed = nn.Parameter(torch.zeros(1, self.horizon, dim))
        self.action_head = ScaleFreeActionHead(
            dim=dim,
            hidden_dim=action_hidden_dim,
            final_init_std=action_head_final_init_std,
        )

    def _direct_anchor_actions(
        self,
        history_latents: torch.Tensor,
        z_t: torch.Tensor,
        history_mask: torch.Tensor,
        last_action_scale_free: torch.Tensor,
        dt_seconds: torch.Tensor,
        action_history_scale_free: torch.Tensor | None,
        action_history_mask: torch.Tensor | None,
        camera_history_path_scale_free: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.direct_anchor is None:
            return last_action_scale_free.to(device=z_t.device, dtype=torch.float32).unsqueeze(1).expand(
                -1,
                self.horizon,
                -1,
            )
        if self.direct_anchor_frozen:
            with torch.no_grad():
                return self.direct_anchor(
                    history_latents=history_latents,
                    z_t=z_t,
                    history_mask=history_mask,
                    last_action_scale_free=last_action_scale_free,
                    dt_seconds=dt_seconds,
                    action_history_scale_free=action_history_scale_free,
                    action_history_mask=action_history_mask,
                    camera_history_path_scale_free=camera_history_path_scale_free,
                )
        return self.direct_anchor(
            history_latents=history_latents,
            z_t=z_t,
            history_mask=history_mask,
            last_action_scale_free=last_action_scale_free,
            dt_seconds=dt_seconds,
            action_history_scale_free=action_history_scale_free,
            action_history_mask=action_history_mask,
            camera_history_path_scale_free=camera_history_path_scale_free,
        )

    def freeze_direct_anchor(self) -> None:
        """Freeze a loaded direct policy so WAM learns only the world residual."""
        if self.direct_anchor is None:
            raise ValueError("Cannot freeze a missing direct anchor")
        for param in self.direct_anchor.parameters():
            param.requires_grad_(False)
        self.direct_anchor_frozen = True
        self.direct_anchor.eval()

    def train(self, mode: bool = True) -> WorldActionModel:
        super().train(mode)
        if self.direct_anchor is not None and self.direct_anchor_frozen:
            self.direct_anchor.eval()
        return self

    def _pool_world_tokens(self, predicted_latents: torch.Tensor) -> torch.Tensor:
        scores = self.world_pool_score(self.world_pool_norm(predicted_latents.float())).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        return (predicted_latents.float() * weights.unsqueeze(-1)).sum(dim=2)

    def _head_context(
        self,
        pooled_world: torch.Tensor,
        last_action_scale_free: torch.Tensor,
        dt_seconds: torch.Tensor,
        action_history_scale_free: torch.Tensor | None,
        action_history_mask: torch.Tensor | None,
        camera_history_path_scale_free: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size = pooled_world.shape[0]
        device = pooled_world.device
        dtype = pooled_world.dtype
        last_action = last_action_scale_free.to(device=device, dtype=dtype)
        dt = dt_seconds.to(device=device, dtype=dtype)

        action_history = (
            torch.zeros(batch_size, self.history, self.action_dim, device=device, dtype=dtype)
            if action_history_scale_free is None
            else action_history_scale_free.to(device=device, dtype=dtype)
        )
        camera_path = (
            torch.zeros(batch_size, self.history, 3, device=device, dtype=dtype)
            if camera_history_path_scale_free is None
            else camera_history_path_scale_free.to(device=device, dtype=dtype)
        )
        history_valid = (
            torch.ones(batch_size, self.history, device=device, dtype=torch.bool)
            if action_history_mask is None
            else action_history_mask.to(device=device, dtype=torch.bool)
        )
        history_features = self.head_history_action_proj(action_history) + self.head_history_path_proj(camera_path)
        history_weights = history_valid.to(dtype=dtype)
        history_denom = history_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        history_context = (history_features * history_weights.unsqueeze(-1)).sum(dim=1) / history_denom

        return (
            pooled_world
            + self.head_last_action_proj(last_action).unsqueeze(1)
            + history_context.unsqueeze(1)
            + self.head_dt_proj(dt.unsqueeze(-1))
            + self.head_horizon_embed[:, : self.horizon]
        )

    def _validate_inputs(
        self,
        history_latents: torch.Tensor,
        z_t: torch.Tensor,
        history_mask: torch.Tensor,
        last_action_scale_free: torch.Tensor,
        dt_seconds: torch.Tensor,
        action_history_scale_free: torch.Tensor | None,
        action_history_mask: torch.Tensor | None,
        camera_history_path_scale_free: torch.Tensor | None,
    ) -> None:
        batch_size = z_t.shape[0]
        if history_latents.ndim != 4:
            raise ValueError(f"history_latents: expected (B,H,P,D), got {history_latents.shape}")
        if z_t.ndim != 3:
            raise ValueError(f"z_t: expected (B,P,D), got {z_t.shape}")
        if history_latents.shape[0] != batch_size:
            raise ValueError("history_latents and z_t batch sizes differ")
        if history_latents.shape[1] != self.history:
            raise ValueError(f"history_latents: expected H={self.history}, got {history_latents.shape[1]}")
        if history_latents.shape[-1] != self.dim or z_t.shape[-1] != self.dim:
            raise ValueError(f"latent dim must be {self.dim}, got {history_latents.shape[-1]} and {z_t.shape[-1]}")
        if history_mask.shape != history_latents.shape[:2]:
            raise ValueError(f"history_mask: expected {history_latents.shape[:2]}, got {history_mask.shape}")
        if last_action_scale_free.shape != (batch_size, self.action_dim):
            raise ValueError(
                f"last_action_scale_free: expected {(batch_size, self.action_dim)}, "
                f"got {last_action_scale_free.shape}"
            )
        if dt_seconds.shape != (batch_size, self.horizon):
            raise ValueError(f"dt_seconds: expected {(batch_size, self.horizon)}, got {dt_seconds.shape}")
        if action_history_scale_free is not None and action_history_scale_free.shape != (
            batch_size,
            self.history,
            self.action_dim,
        ):
            raise ValueError(
                f"action_history_scale_free: expected {(batch_size, self.history, self.action_dim)}, "
                f"got {action_history_scale_free.shape}"
            )
        if action_history_mask is not None and action_history_mask.shape != (batch_size, self.history):
            raise ValueError(f"action_history_mask: expected {(batch_size, self.history)}, got {action_history_mask.shape}")
        if camera_history_path_scale_free is not None and camera_history_path_scale_free.shape != (
            batch_size,
            self.history,
            3,
        ):
            raise ValueError(
                f"camera_history_path_scale_free: expected {(batch_size, self.history, 3)}, "
                f"got {camera_history_path_scale_free.shape}"
            )

    def _condition_observed_latents(
        self,
        history_latents: torch.Tensor,
        z_t: torch.Tensor,
        history_mask: torch.Tensor,
        last_action_scale_free: torch.Tensor,
        action_history_scale_free: torch.Tensor | None,
        action_history_mask: torch.Tensor | None,
        camera_history_path_scale_free: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = z_t.shape[0]
        device = z_t.device
        history = history_latents.float()
        current = z_t.float()
        base_history_valid = history_mask.to(device=device, dtype=torch.bool)
        history = history * base_history_valid[:, :, None, None].to(dtype=history.dtype)

        action_history = (
            torch.zeros(batch_size, self.history, self.action_dim, device=device, dtype=torch.float32)
            if action_history_scale_free is None
            else action_history_scale_free.to(device=device, dtype=torch.float32)
        )
        camera_path = (
            torch.zeros(batch_size, self.history, 3, device=device, dtype=torch.float32)
            if camera_history_path_scale_free is None
            else camera_history_path_scale_free.to(device=device, dtype=torch.float32)
        )
        history_valid = base_history_valid
        if action_history_mask is not None:
            history_valid = history_valid & action_history_mask.to(device=device, dtype=torch.bool)

        history_context = self.history_action_proj(action_history) + self.history_path_proj(camera_path)
        history_context = history_context * history_valid.unsqueeze(-1).to(dtype=history_context.dtype)
        history = history + history_context.unsqueeze(2)

        current_context = self.current_action_proj(
            last_action_scale_free.to(device=device, dtype=torch.float32)
        ).unsqueeze(1)
        current = current + current_context
        return history, current

    def rollout(
        self,
        history_latents: torch.Tensor,
        z_t: torch.Tensor,
        history_mask: torch.Tensor,
        last_action_scale_free: torch.Tensor,
        dt_seconds: torch.Tensor,
        action_history_scale_free: torch.Tensor | None = None,
        action_history_mask: torch.Tensor | None = None,
        camera_history_path_scale_free: torch.Tensor | None = None,
    ) -> WorldActionOutput:
        """Predict latent rollout and future scale-free actions from observations only."""
        self._validate_inputs(
            history_latents,
            z_t,
            history_mask,
            last_action_scale_free,
            dt_seconds,
            action_history_scale_free,
            action_history_mask,
            camera_history_path_scale_free,
        )
        conditioned_history, conditioned_z_t = self._condition_observed_latents(
            history_latents=history_latents,
            z_t=z_t,
            history_mask=history_mask,
            last_action_scale_free=last_action_scale_free,
            action_history_scale_free=action_history_scale_free,
            action_history_mask=action_history_mask,
            camera_history_path_scale_free=camera_history_path_scale_free,
        )
        predicted_latents = self.predictor(
            history_latents=conditioned_history,
            z_t=conditioned_z_t,
            action_4dof=last_action_scale_free.to(device=z_t.device, dtype=torch.float32),
            dt_seconds=dt_seconds.to(device=z_t.device, dtype=torch.float32),
            history_mask=history_mask.to(device=z_t.device, dtype=torch.bool),
        )
        pooled_world = self._pool_world_tokens(predicted_latents)
        head_input = self._head_context(
            pooled_world=pooled_world,
            last_action_scale_free=last_action_scale_free,
            dt_seconds=dt_seconds,
            action_history_scale_free=action_history_scale_free,
            action_history_mask=action_history_mask,
            camera_history_path_scale_free=camera_history_path_scale_free,
        )
        residual_actions = self.action_head(head_input)
        base_actions = self._direct_anchor_actions(
            history_latents=history_latents,
            z_t=z_t,
            history_mask=history_mask,
            last_action_scale_free=last_action_scale_free,
            dt_seconds=dt_seconds,
            action_history_scale_free=action_history_scale_free,
            action_history_mask=action_history_mask,
            camera_history_path_scale_free=camera_history_path_scale_free,
        ).to(device=residual_actions.device, dtype=residual_actions.dtype)
        predicted_actions = base_actions + residual_actions
        return WorldActionOutput(predicted_latents=predicted_latents, predicted_actions=predicted_actions)

    def forward(
        self,
        history_latents: torch.Tensor,
        z_t: torch.Tensor,
        history_mask: torch.Tensor,
        last_action_scale_free: torch.Tensor,
        dt_seconds: torch.Tensor,
        action_history_scale_free: torch.Tensor | None = None,
        action_history_mask: torch.Tensor | None = None,
        camera_history_path_scale_free: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return future scale-free actions shaped like the direct-policy diagnostic."""
        return self.rollout(
            history_latents=history_latents,
            z_t=z_t,
            history_mask=history_mask,
            last_action_scale_free=last_action_scale_free,
            dt_seconds=dt_seconds,
            action_history_scale_free=action_history_scale_free,
            action_history_mask=action_history_mask,
            camera_history_path_scale_free=camera_history_path_scale_free,
        ).predicted_actions

    @classmethod
    def from_config(
        cls,
        cfg: PredictorConfig,
        dim: int = EMBED_DIM,
        action_hidden_dim: int = 256,
        use_action_film: bool = True,
        prediction_mode: str = "residual",
        action_head_final_init_std: float = 0.0,
        use_direct_anchor: bool = False,
    ) -> WorldActionModel:
        return cls(
            cfg=cfg,
            dim=dim,
            action_hidden_dim=action_hidden_dim,
            use_action_film=use_action_film,
            prediction_mode=prediction_mode,
            action_head_final_init_std=action_head_final_init_std,
            use_direct_anchor=use_direct_anchor,
        )


__all__ = ["WorldActionModel", "WorldActionOutput"]
