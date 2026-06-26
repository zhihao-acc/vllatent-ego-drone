"""Loss functions for sports-following training (TORCH tier) — B1.18.

L_latent: smooth L1 (beta=0.1) between predicted and GT future latents,
    quality-weighted per sample.
L_wp: smooth L1 between predicted and GT future deltas,
    confidence-weighted per sample (NOT quality-weighted).
combined_loss: L_total = w_quality * L_latent + lambda_wp * w_vo * L_wp.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F


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
    """Quality-weighted smooth L1 on latent predictions.

    Parameters
    ----------
    predicted : (B, T, P, D)
    target : (B, T, P, D)
    quality_weight : (B,) — frame_quality.clamp(min=0.1)
    beta : smooth L1 beta (DINO-world precedent: 0.1, NOT default 1.0)
    """
    per_sample = F.smooth_l1_loss(
        predicted, target, beta=beta, reduction="none"
    ).mean(dim=(1, 2, 3))

    return (per_sample * quality_weight).mean()


def waypoint_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    confidence_weight: torch.Tensor,
) -> torch.Tensor:
    """Confidence-weighted smooth L1 on waypoint deltas.

    Parameters
    ----------
    predicted : (B, T, 4)
    target : (B, T, 4)
    confidence_weight : (B,) — vo_confidence.mean(1).clamp(min=0.05)
    """
    per_sample = F.smooth_l1_loss(
        predicted, target, reduction="none"
    ).mean(dim=(1, 2))

    return (per_sample * confidence_weight).mean()


def cosine_similarity_diagnostic(
    predicted: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Mean cosine similarity across batch (diagnostic, not gradient source)."""
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
    """Combined training loss with per-sample weighting.

    Parameters
    ----------
    predicted_latents : (B, T, P, D)
    target_latents : (B, T, P, D)
    predicted_deltas : (B, T, 4)
    target_deltas : (B, T, 4)
    frame_quality : (B,)
    vo_confidence : (B, T)
    lambda_latent : weight for L_latent
    lambda_waypoint : weight for L_wp
    beta : smooth L1 beta for L_latent
    """
    w_quality = frame_quality.clamp(min=0.1)
    w_vo = vo_confidence.mean(dim=1).clamp(min=0.05)

    l_lat = latent_loss(predicted_latents, target_latents, w_quality, beta=beta)
    l_wp = waypoint_loss(predicted_deltas, target_deltas, w_vo)

    total = lambda_latent * l_lat + lambda_waypoint * l_wp

    with torch.no_grad():
        cos_sim = cosine_similarity_diagnostic(predicted_latents, target_latents)

    return LossOutput(total=total, latent=l_lat, waypoint=l_wp, cosine_sim=cos_sim)
