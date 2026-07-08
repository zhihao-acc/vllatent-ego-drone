"""B3 human world-model losses (TORCH tier).

YOLO/ByteTrack person labels become bounded soft weights over the DINO 14x14
patch grid. They are loss weights and targets, not extra DINO class tokens.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from vllatent.plan_tokens import PLAN_TOKEN_DIM
from vllatent.schemas import PATCH_TOKENS

if TYPE_CHECKING:
    import torch


class WorldModelLossOutput(NamedTuple):
    """Decomposed B3 loss output for logging."""

    total: torch.Tensor
    latent: torch.Tensor
    person_state: torch.Tensor
    inverse_plan: torch.Tensor
    latent_cosine: torch.Tensor


def person_patch_weights(
    person_state_target: torch.Tensor,
    person_state_valid: torch.Tensor,
    person_conf: torch.Tensor | None = None,
    *,
    grid_size: int = 14,
    min_sigma: float | None = None,
    max_sigma: float = 0.5,
) -> torch.Tensor:
    """Convert ``cx,cy,log_h,visibility`` labels into bounded patch weights."""
    import torch

    if person_state_target.ndim != 3 or person_state_target.shape[-1] != 4:
        raise ValueError(f"person_state_target: expected (B,T,4), got {person_state_target.shape}")
    if person_state_valid.shape != person_state_target.shape[:2]:
        raise ValueError(f"person_state_valid: expected {person_state_target.shape[:2]}, got {person_state_valid.shape}")
    if person_conf is not None and person_conf.shape != person_state_target.shape[:2]:
        raise ValueError(f"person_conf: expected {person_state_target.shape[:2]}, got {person_conf.shape}")
    if grid_size < 1:
        raise ValueError(f"grid_size must be >= 1, got {grid_size}")
    if grid_size * grid_size != PATCH_TOKENS:
        raise ValueError(f"grid_size {grid_size} does not match PATCH_TOKENS={PATCH_TOKENS}")

    device = person_state_target.device
    dtype = torch.float32
    state = person_state_target.to(dtype=dtype)
    centers = (torch.arange(grid_size, device=device, dtype=dtype) + 0.5) / float(grid_size)
    yy, xx = torch.meshgrid(centers, centers, indexing="ij")
    patch_xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)

    cxcy = state[..., :2].clamp(0.0, 1.0).unsqueeze(-2)
    log_h = state[..., 2:3].clamp(min=-12.0, max=0.0)
    visibility = state[..., 3].clamp(0.0, 1.0)
    height = torch.exp(log_h).clamp(min=1.0 / float(grid_size), max=1.0)
    sigma_floor = 0.5 / float(grid_size) if min_sigma is None else float(min_sigma)
    sigma = (height * 0.35).clamp(min=sigma_floor, max=max_sigma).unsqueeze(-2)

    dist2 = (patch_xy.view(1, 1, PATCH_TOKENS, 2) - cxcy).square().sum(dim=-1)
    weights = torch.exp(-0.5 * dist2 / sigma.squeeze(-1).square().clamp(min=1e-8))
    valid = person_state_valid.to(device=device, dtype=dtype) * visibility
    if person_conf is not None:
        valid = valid * person_conf.to(device=device, dtype=dtype).clamp(0.0, 1.0)
    return (weights * valid.unsqueeze(-1)).clamp(0.0, 1.0)


def person_weighted_latent_loss(
    predicted_latents: torch.Tensor,
    target_latents: torch.Tensor,
    person_state_target: torch.Tensor,
    person_state_valid: torch.Tensor,
    person_conf: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
    *,
    beta: float = 0.1,
    person_weight: float = 2.0,
    background_weight: float = 0.25,
) -> torch.Tensor:
    """Smooth-L1 latent loss with background coverage and bounded foreground emphasis."""
    import torch
    import torch.nn.functional as F

    if predicted_latents.shape != target_latents.shape:
        raise ValueError(f"predicted/target shape mismatch: {predicted_latents.shape} vs {target_latents.shape}")
    if predicted_latents.ndim != 4 or predicted_latents.shape[2] != PATCH_TOKENS:
        raise ValueError(f"predicted_latents: expected (B,T,{PATCH_TOKENS},D), got {predicted_latents.shape}")
    if person_weight < 0.0:
        raise ValueError(f"person_weight must be >= 0, got {person_weight}")
    if background_weight <= 0.0:
        raise ValueError(f"background_weight must be > 0, got {background_weight}")

    patch_weights = background_weight + person_weight * person_patch_weights(
        person_state_target,
        person_state_valid,
        person_conf,
    )
    if sample_weight is not None:
        if sample_weight.shape != (predicted_latents.shape[0],):
            raise ValueError(f"sample_weight: expected {(predicted_latents.shape[0],)}, got {sample_weight.shape}")
        patch_weights = patch_weights * sample_weight.to(
            device=predicted_latents.device,
            dtype=torch.float32,
        )[:, None, None]

    per_patch = F.smooth_l1_loss(
        predicted_latents.float(),
        target_latents.float(),
        beta=beta,
        reduction="none",
    ).mean(dim=-1)
    denom = patch_weights.sum().clamp(min=1e-8)
    return (per_patch * patch_weights).sum() / denom


def person_state_loss(
    predicted_person_state: torch.Tensor,
    person_state_target: torch.Tensor,
    person_state_valid: torch.Tensor,
    person_conf: torch.Tensor | None = None,
    *,
    beta: float = 0.05,
    visibility_weight: float = 1.0,
) -> torch.Tensor:
    """Masked center/log-height regression plus visibility-logit BCE."""
    import torch
    import torch.nn.functional as F

    if predicted_person_state.shape != person_state_target.shape:
        raise ValueError(
            f"predicted/target person-state shape mismatch: "
            f"{predicted_person_state.shape} vs {person_state_target.shape}"
        )
    if predicted_person_state.ndim != 3 or predicted_person_state.shape[-1] != 4:
        raise ValueError(f"predicted_person_state: expected (B,T,4), got {predicted_person_state.shape}")
    if person_state_valid.shape != predicted_person_state.shape[:2]:
        raise ValueError(f"person_state_valid: expected {predicted_person_state.shape[:2]}, got {person_state_valid.shape}")

    valid = person_state_valid.to(device=predicted_person_state.device, dtype=torch.float32)
    if person_conf is not None:
        if person_conf.shape != predicted_person_state.shape[:2]:
            raise ValueError(f"person_conf: expected {predicted_person_state.shape[:2]}, got {person_conf.shape}")
        valid = valid * person_conf.to(device=predicted_person_state.device, dtype=torch.float32).clamp(0.0, 1.0)

    state_err = F.smooth_l1_loss(
        predicted_person_state[..., :3].float(),
        person_state_target[..., :3].float(),
        beta=beta,
        reduction="none",
    ).mean(dim=-1)
    state_denom = valid.sum()
    state_loss = predicted_person_state.sum() * 0.0 if float(state_denom) <= 0.0 else (state_err * valid).sum() / state_denom

    target_vis = person_state_target[..., 3].to(device=predicted_person_state.device, dtype=torch.float32).clamp(0.0, 1.0)
    visibility = F.binary_cross_entropy_with_logits(
        predicted_person_state[..., 3].float(),
        target_vis,
        reduction="mean",
    )
    return state_loss + visibility_weight * visibility


def inverse_plan_loss(
    predicted_plan: torch.Tensor,
    target_plan: torch.Tensor,
    plan_valid_mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    *,
    beta: float = 0.05,
) -> torch.Tensor:
    """Masked inverse-dynamics auxiliary loss over B3 6-D plan tokens."""
    import torch
    import torch.nn.functional as F

    if predicted_plan.shape != target_plan.shape:
        raise ValueError(f"predicted/target plan shape mismatch: {predicted_plan.shape} vs {target_plan.shape}")
    if predicted_plan.ndim != 3 or predicted_plan.shape[-1] != PLAN_TOKEN_DIM:
        raise ValueError(f"predicted_plan: expected (B,T,{PLAN_TOKEN_DIM}), got {predicted_plan.shape}")
    if plan_valid_mask.shape != predicted_plan.shape[:2]:
        raise ValueError(f"plan_valid_mask: expected {predicted_plan.shape[:2]}, got {plan_valid_mask.shape}")

    mask = plan_valid_mask.to(device=predicted_plan.device, dtype=torch.float32)
    if sample_weight is not None:
        if sample_weight.shape != (predicted_plan.shape[0],):
            raise ValueError(f"sample_weight: expected {(predicted_plan.shape[0],)}, got {sample_weight.shape}")
        mask = mask * sample_weight.to(device=predicted_plan.device, dtype=torch.float32)[:, None]
    denom = mask.sum()
    if float(denom) <= 0.0:
        return predicted_plan.sum() * 0.0

    per_step = F.smooth_l1_loss(
        predicted_plan.float(),
        target_plan.float(),
        beta=beta,
        reduction="none",
    ).mean(dim=-1)
    return (per_step * mask).sum() / denom


def latent_cosine_similarity(predicted_latents: torch.Tensor, target_latents: torch.Tensor) -> torch.Tensor:
    """Mean cosine similarity diagnostic over latent rollout tensors."""
    import torch.nn.functional as F

    pred_flat = predicted_latents.float().reshape(predicted_latents.shape[0], -1)
    target_flat = target_latents.float().reshape(target_latents.shape[0], -1)
    return F.cosine_similarity(pred_flat, target_flat, dim=1).mean()


def human_world_model_loss(
    *,
    predicted_latents: torch.Tensor,
    target_latents: torch.Tensor,
    predicted_person_state: torch.Tensor,
    person_state_target: torch.Tensor,
    person_state_valid: torch.Tensor,
    predicted_plan: torch.Tensor,
    planned_actions: torch.Tensor,
    planned_actions_valid_mask: torch.Tensor,
    person_conf: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
    lambda_latent: float = 1.0,
    lambda_person_state: float = 1.0,
    lambda_inverse_plan: float = 0.1,
) -> WorldModelLossOutput:
    """Combined B3 loss. Targets enter here, not the model ``forward``."""
    latent = person_weighted_latent_loss(
        predicted_latents=predicted_latents,
        target_latents=target_latents,
        person_state_target=person_state_target,
        person_state_valid=person_state_valid,
        person_conf=person_conf,
        sample_weight=sample_weight,
    )
    state = person_state_loss(
        predicted_person_state=predicted_person_state,
        person_state_target=person_state_target,
        person_state_valid=person_state_valid,
        person_conf=person_conf,
    )
    inverse = inverse_plan_loss(
        predicted_plan=predicted_plan,
        target_plan=planned_actions,
        plan_valid_mask=planned_actions_valid_mask,
        sample_weight=sample_weight,
    )
    total = lambda_latent * latent + lambda_person_state * state + lambda_inverse_plan * inverse
    return WorldModelLossOutput(
        total=total,
        latent=latent,
        person_state=state,
        inverse_plan=inverse,
        latent_cosine=latent_cosine_similarity(predicted_latents, target_latents),
    )


__all__ = [
    "WorldModelLossOutput",
    "human_world_model_loss",
    "inverse_plan_loss",
    "latent_cosine_similarity",
    "person_patch_weights",
    "person_state_loss",
    "person_weighted_latent_loss",
]
