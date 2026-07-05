"""Full model assembly (TORCH tier) — B1.17.

SportsFollowingModel = LatentPredictor + WaypointHead.
Forward takes TrainingBatch fields → (predicted_latents, predicted_deltas).
Encoder is NOT part of forward (latents cached). Config-driven from PredictorConfig.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vllatent.model.heads import WaypointHead
from vllatent.model.predictor import LatentPredictor
from vllatent.schemas import EMBED_DIM

if TYPE_CHECKING:
    from vllatent.config import PredictorConfig
    from vllatent.data.collate import TrainingBatch


class ModelOutput:
    """Training-time output: batched tensors (not numpy PredictorOutput)."""

    __slots__ = ("predicted_latents", "predicted_deltas")

    def __init__(
        self, predicted_latents: torch.Tensor, predicted_deltas: torch.Tensor
    ) -> None:
        self.predicted_latents = predicted_latents
        self.predicted_deltas = predicted_deltas


class SportsFollowingModel(nn.Module):
    """Predictor + waypoint head, config-driven."""

    def __init__(
        self,
        cfg: PredictorConfig,
        dim: int = EMBED_DIM,
        use_action_film: bool = True,
        prediction_mode: str = "absolute",
    ) -> None:
        super().__init__()
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
        self.waypoint_head = WaypointHead(dim=dim)

    def forward(self, batch: TrainingBatch) -> ModelOutput:
        predicted_latents = self.predictor(
            history_latents=batch.history_latents,
            z_t=batch.z_t,
            action_4dof=batch.last_action,
            dt_seconds=batch.dt_seconds,
            history_mask=batch.history_mask,
        )

        pooled = predicted_latents.mean(dim=2)
        predicted_deltas = self.waypoint_head(pooled)

        return ModelOutput(
            predicted_latents=predicted_latents,
            predicted_deltas=predicted_deltas,
        )

    @classmethod
    def from_config(
        cls,
        cfg: PredictorConfig,
        dim: int = EMBED_DIM,
        use_action_film: bool = True,
        prediction_mode: str = "absolute",
    ) -> SportsFollowingModel:
        return cls(
            cfg=cfg,
            dim=dim,
            use_action_film=use_action_film,
            prediction_mode=prediction_mode,
        )
