"""Loss functions for sports-following training (TORCH tier) — B1.18.

L_latent: smooth L1 (beta=0.1) between predicted and GT future latents,
    quality-weighted per sample.
L_wp: smooth L1 between predicted and GT future deltas,
    confidence-weighted per sample (NOT quality-weighted).
combined_loss: L_total = w_quality * L_latent + lambda_wp * w_vo * L_wp.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    import torch


class LossOutput(NamedTuple):
    """Decomposed loss output for logging."""

    total: torch.Tensor
    latent: torch.Tensor
    waypoint: torch.Tensor
    cosine_sim: torch.Tensor


def latent_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    quality_weight: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """Quality-weighted smooth L1 on latent predictions."""
    import torch.nn.functional as F

    per_sample = F.smooth_l1_loss(
        predicted, target, beta=beta, reduction="none"
    ).mean(dim=(1, 2, 3))

    return (per_sample * quality_weight).mean()


def waypoint_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    confidence_weight: torch.Tensor,
) -> torch.Tensor:
    """Confidence-weighted smooth L1 on waypoint deltas."""
    import torch.nn.functional as F

    per_sample = F.smooth_l1_loss(
        predicted, target, reduction="none"
    ).mean(dim=(1, 2))

    return (per_sample * confidence_weight).mean()


def action_policy_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    moving_mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    *,
    direction_weight: float = 1.0,
    speed_weight: float = 1.0,
    path_weight: float = 1.0,
) -> torch.Tensor:
    """Masked differentiable loss for B2 scale-free action prediction."""
    import torch
    import torch.nn.functional as F

    from vllatent.train.action_metrics import LOG_SPEED_CLAMP, action_step_vectors

    if predicted.shape != target.shape:
        raise ValueError(f"predicted/target shape mismatch: {predicted.shape} vs {target.shape}")
    if moving_mask.shape != predicted.shape[:2]:
        raise ValueError(f"moving_mask: expected {predicted.shape[:2]}, got {moving_mask.shape}")

    mask = moving_mask.to(device=predicted.device, dtype=torch.float32)
    weights = mask
    if sample_weight is not None:
        if sample_weight.shape != (predicted.shape[0],):
            raise ValueError(f"sample_weight: expected {(predicted.shape[0],)}, got {sample_weight.shape}")
        weights = weights * sample_weight.to(device=predicted.device, dtype=torch.float32)[:, None]
    denom = weights.sum()
    if float(denom) <= 0.0:
        return predicted.sum() * 0.0

    pred_dir = F.normalize(predicted[..., :3].float(), dim=-1, eps=1e-6)
    target_dir = F.normalize(target[..., :3].float(), dim=-1, eps=1e-6)
    direction = 1.0 - (pred_dir * target_dir).sum(dim=-1).clamp(-1.0, 1.0)
    speed = F.smooth_l1_loss(
        predicted[..., 3].float().clamp(-LOG_SPEED_CLAMP, LOG_SPEED_CLAMP),
        target[..., 3].float().clamp(-LOG_SPEED_CLAMP, LOG_SPEED_CLAMP),
        reduction="none",
    )
    pred_path = (action_step_vectors(predicted) * mask.unsqueeze(-1)).cumsum(dim=1)
    target_path = (action_step_vectors(target) * mask.unsqueeze(-1)).cumsum(dim=1)
    path = F.smooth_l1_loss(pred_path, target_path, reduction="none").mean(dim=-1)

    per_step = direction_weight * direction + speed_weight * speed + path_weight * path
    return (per_step * weights).sum() / denom


def cosine_similarity_diagnostic(
    predicted: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Mean cosine similarity across batch (diagnostic, not gradient source)."""
    import torch.nn.functional as F

    pred_flat = predicted.reshape(predicted.shape[0], -1)
    tgt_flat = target.reshape(target.shape[0], -1)
    return F.cosine_similarity(pred_flat, tgt_flat, dim=1).mean()


def combined_loss(
    predicted_latents: torch.Tensor,
    target_latents: torch.Tensor,
    predicted_deltas: torch.Tensor,
    target_deltas: torch.Tensor,
    frame_quality: torch.Tensor,
    vo_confidence: torch.Tensor,
    lambda_latent: float = 1.0,
    lambda_waypoint: float = 1.0,
    beta: float = 0.1,
) -> LossOutput:
    """Combined training loss with per-sample weighting."""
    import torch

    w_quality = frame_quality.clamp(min=0.1)
    w_vo = vo_confidence.mean(dim=1).clamp(min=0.05)

    l_lat = latent_loss(predicted_latents, target_latents, w_quality, beta=beta)
    l_wp = waypoint_loss(predicted_deltas, target_deltas, w_vo)

    total = lambda_latent * l_lat + lambda_waypoint * l_wp

    with torch.no_grad():
        cos_sim = cosine_similarity_diagnostic(predicted_latents, target_latents)

    return LossOutput(total=total, latent=l_lat, waypoint=l_wp, cosine_sim=cos_sim)
