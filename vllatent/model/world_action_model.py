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
    ) -> None:
        super().__init__()
        if latent_residual_init_std < 0:
            raise ValueError(f"latent_residual_init_std must be >= 0, got {latent_residual_init_std}")

        self.dim = dim
        self.history = cfg.history
        self.horizon = cfg.horizon
        self.action_dim = SCALE_FREE_ACTION_DIM

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

        self.action_head = ScaleFreeActionHead(dim=dim, hidden_dim=action_hidden_dim)

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
        pooled_world = predicted_latents.float().mean(dim=2)
        residual_actions = self.action_head(pooled_world)
        base_actions = last_action_scale_free.to(
            device=residual_actions.device,
            dtype=residual_actions.dtype,
        ).unsqueeze(1).expand(-1, self.horizon, -1)
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
    ) -> WorldActionModel:
        return cls(
            cfg=cfg,
            dim=dim,
            action_hidden_dim=action_hidden_dim,
            use_action_film=use_action_film,
            prediction_mode=prediction_mode,
        )


__all__ = ["WorldActionModel", "WorldActionOutput"]
