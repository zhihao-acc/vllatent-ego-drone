"""B3 Stage-1 world-model metrics (TORCH tier)."""
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from vllatent.plan_tokens import PLAN_TOKEN_DIM
from vllatent.schemas import PATCH_TOKENS
from vllatent.train.world_model_losses import person_patch_weights

if TYPE_CHECKING:
    import torch


class Stage1BatchMetrics(NamedTuple):
    """Per-batch B3 gate metrics."""

    model_loss: float
    persistence_loss: float
    null_plan_loss: float
    shuffled_plan_loss: float
    flipped_plan_loss: float
    improvement_vs_persistence: float
    improvement_vs_null: float
    per_step_model_loss: list[float]
    per_step_persistence_loss: list[float]
    rollout_beats_persistence: list[bool]
    true_beats_shuffled_rate: float
    true_beats_flipped_rate: float


def persistence_rollout(z_t: torch.Tensor, horizon: int) -> torch.Tensor:
    """Repeat the current latent across the future horizon."""
    if z_t.ndim != 3:
        raise ValueError(f"z_t: expected (B,P,D), got {z_t.shape}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    return z_t.unsqueeze(1).expand(-1, horizon, -1, -1)


def null_plan(planned_actions: torch.Tensor) -> torch.Tensor:
    """Construct a zero candidate-plan ablation."""
    if planned_actions.ndim != 3 or planned_actions.shape[-1] != PLAN_TOKEN_DIM:
        raise ValueError(f"planned_actions: expected (B,T,{PLAN_TOKEN_DIM}), got {planned_actions.shape}")
    return planned_actions * 0.0


def flipped_plan(planned_actions: torch.Tensor) -> torch.Tensor:
    """Flip translation direction and yaw-rate while preserving speed and validity fields."""
    if planned_actions.ndim != 3 or planned_actions.shape[-1] != PLAN_TOKEN_DIM:
        raise ValueError(f"planned_actions: expected (B,T,{PLAN_TOKEN_DIM}), got {planned_actions.shape}")
    flipped = planned_actions.clone()
    flipped[..., :3] = -flipped[..., :3]
    flipped[..., 4] = -flipped[..., 4]
    return flipped


def shuffled_plan(planned_actions: torch.Tensor) -> torch.Tensor:
    """Shuffle candidate plans across the batch, falling back to horizon roll for B=1."""
    import torch

    if planned_actions.ndim != 3 or planned_actions.shape[-1] != PLAN_TOKEN_DIM:
        raise ValueError(f"planned_actions: expected (B,T,{PLAN_TOKEN_DIM}), got {planned_actions.shape}")
    if planned_actions.shape[0] > 1:
        return planned_actions[torch.randperm(planned_actions.shape[0], device=planned_actions.device)]
    return planned_actions.roll(shifts=1, dims=1)


def per_window_latent_loss(
    predicted_latents: torch.Tensor,
    target_latents: torch.Tensor,
    person_state_target: torch.Tensor,
    person_state_valid: torch.Tensor,
    person_conf: torch.Tensor | None = None,
    *,
    beta: float = 0.1,
    person_weight: float = 2.0,
    background_weight: float = 0.25,
) -> torch.Tensor:
    """Return person-weighted latent loss per sample and horizon step."""
    import torch.nn.functional as F

    if predicted_latents.shape != target_latents.shape:
        raise ValueError(f"predicted/target shape mismatch: {predicted_latents.shape} vs {target_latents.shape}")
    if predicted_latents.ndim != 4 or predicted_latents.shape[2] != PATCH_TOKENS:
        raise ValueError(f"predicted_latents: expected (B,T,{PATCH_TOKENS},D), got {predicted_latents.shape}")
    weights = background_weight + person_weight * person_patch_weights(
        person_state_target,
        person_state_valid,
        person_conf,
    )
    per_patch = F.smooth_l1_loss(
        predicted_latents.float(),
        target_latents.float(),
        beta=beta,
        reduction="none",
    ).mean(dim=-1)
    denom = weights.sum(dim=-1).clamp(min=1e-8)
    return (per_patch * weights).sum(dim=-1) / denom


def summarize_stage1_batch(
    *,
    predicted_latents: torch.Tensor,
    persistence_latents: torch.Tensor,
    null_plan_latents: torch.Tensor,
    shuffled_plan_latents: torch.Tensor,
    flipped_plan_latents: torch.Tensor,
    target_latents: torch.Tensor,
    person_state_target: torch.Tensor,
    person_state_valid: torch.Tensor,
    person_conf: torch.Tensor | None = None,
) -> Stage1BatchMetrics:
    """Compute G1a/G1b/G1d-style metrics for one evaluation batch."""
    model = per_window_latent_loss(
        predicted_latents,
        target_latents,
        person_state_target,
        person_state_valid,
        person_conf,
    )
    persistence = per_window_latent_loss(
        persistence_latents,
        target_latents,
        person_state_target,
        person_state_valid,
        person_conf,
    )
    null = per_window_latent_loss(
        null_plan_latents,
        target_latents,
        person_state_target,
        person_state_valid,
        person_conf,
    )
    shuffled = per_window_latent_loss(
        shuffled_plan_latents,
        target_latents,
        person_state_target,
        person_state_valid,
        person_conf,
    )
    flipped = per_window_latent_loss(
        flipped_plan_latents,
        target_latents,
        person_state_target,
        person_state_valid,
        person_conf,
    )

    model_step = model.mean(dim=0)
    persistence_step = persistence.mean(dim=0)
    model_loss = float(model.mean().detach().cpu())
    persistence_loss = float(persistence.mean().detach().cpu())
    null_loss = float(null.mean().detach().cpu())
    shuffled_loss = float(shuffled.mean().detach().cpu())
    flipped_loss = float(flipped.mean().detach().cpu())
    return Stage1BatchMetrics(
        model_loss=model_loss,
        persistence_loss=persistence_loss,
        null_plan_loss=null_loss,
        shuffled_plan_loss=shuffled_loss,
        flipped_plan_loss=flipped_loss,
        improvement_vs_persistence=(persistence_loss - model_loss) / max(persistence_loss, 1e-8),
        improvement_vs_null=(null_loss - model_loss) / max(null_loss, 1e-8),
        per_step_model_loss=[float(x) for x in model_step.detach().cpu()],
        per_step_persistence_loss=[float(x) for x in persistence_step.detach().cpu()],
        rollout_beats_persistence=[bool(x) for x in (model_step < persistence_step).detach().cpu()],
        true_beats_shuffled_rate=float((model.mean(dim=1) < shuffled.mean(dim=1)).float().mean().detach().cpu()),
        true_beats_flipped_rate=float((model.mean(dim=1) < flipped.mean(dim=1)).float().mean().detach().cpu()),
    )


def aggregate_stage1_metrics(metrics: list[Stage1BatchMetrics]) -> dict[str, object]:
    """Aggregate per-batch Stage-1 metrics into JSON-friendly gate readouts."""
    import numpy as np

    if not metrics:
        raise ValueError("metrics must not be empty")
    horizon = len(metrics[0].per_step_model_loss)
    for item in metrics:
        if len(item.per_step_model_loss) != horizon:
            raise ValueError("all metrics must have the same horizon")

    def mean_attr(name: str) -> float:
        return float(np.mean([getattr(item, name) for item in metrics]))

    per_step_model = np.asarray([item.per_step_model_loss for item in metrics], dtype=np.float64).mean(axis=0)
    per_step_persistence = np.asarray(
        [item.per_step_persistence_loss for item in metrics],
        dtype=np.float64,
    ).mean(axis=0)
    rollout = per_step_model < per_step_persistence
    g1a_persistence = mean_attr("improvement_vs_persistence")
    g1a_null = mean_attr("improvement_vs_null")
    true_beats_shuffled = mean_attr("true_beats_shuffled_rate")
    true_beats_flipped = mean_attr("true_beats_flipped_rate")
    return {
        "model_loss": mean_attr("model_loss"),
        "persistence_loss": mean_attr("persistence_loss"),
        "null_plan_loss": mean_attr("null_plan_loss"),
        "shuffled_plan_loss": mean_attr("shuffled_plan_loss"),
        "flipped_plan_loss": mean_attr("flipped_plan_loss"),
        "improvement_vs_persistence": g1a_persistence,
        "improvement_vs_null": g1a_null,
        "per_step_model_loss": [float(x) for x in per_step_model],
        "per_step_persistence_loss": [float(x) for x in per_step_persistence],
        "rollout_beats_persistence": [bool(x) for x in rollout],
        "true_beats_shuffled_rate": true_beats_shuffled,
        "true_beats_flipped_rate": true_beats_flipped,
        "g1a_pass": bool(g1a_persistence >= 0.10 and g1a_null >= 0.05),
        "g1b_pass": bool(np.all(rollout)),
        "g1d_pass": bool(true_beats_shuffled >= 0.70 and true_beats_flipped >= 0.70),
    }


__all__ = [
    "Stage1BatchMetrics",
    "aggregate_stage1_metrics",
    "flipped_plan",
    "null_plan",
    "per_window_latent_loss",
    "persistence_rollout",
    "shuffled_plan",
    "summarize_stage1_batch",
]
