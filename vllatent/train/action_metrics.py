"""B2 action-policy metrics and baselines (TORCH tier).

Metrics operate on the locked scale-free action vector:
``[unit_dir_x, unit_dir_y, unit_dir_z, log_speed_ratio]``.
Lower aggregate score is better.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    import torch

from vllatent.scale_free_targets import SCALE_FREE_ACTION_DIM
from vllatent.schemas import HORIZON

LOG_SPEED_CLAMP = 8.0
EPS = 1e-6


class ActionMetricResult(NamedTuple):
    """Scalar action metrics for a batch or aggregate."""

    direction_cosine: float
    angular_error_deg: float
    path_ade: float
    path_fde: float
    speed_ratio_mae: float
    aggregate_score: float
    n_samples: int
    n_valid: int
    n_speed_valid: int


class ActionScorecard(NamedTuple):
    """Model metrics plus deterministic baseline margins."""

    model: ActionMetricResult
    baselines: dict[str, ActionMetricResult]
    best_baseline: str
    best_baseline_score: float
    margin: float


def _as_mask(mask: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    import torch

    if mask.shape != shape:
        raise ValueError(f"moving_mask: expected {shape}, got {mask.shape}")
    return mask.to(dtype=torch.bool)


def _check_actions(name: str, actions: torch.Tensor) -> None:
    if actions.ndim != 3 or actions.shape[-1] != SCALE_FREE_ACTION_DIM:
        raise ValueError(f"{name}: expected (B,T,{SCALE_FREE_ACTION_DIM}), got {actions.shape}")


def _normalize_dir(actions: torch.Tensor) -> torch.Tensor:
    import torch.nn.functional as F

    return F.normalize(actions[..., :3].float(), dim=-1, eps=EPS)


def action_step_vectors(actions: torch.Tensor) -> torch.Tensor:
    """Convert scale-free actions to relative path-step vectors."""
    import torch

    dirs = _normalize_dir(actions)
    speed_ratio = torch.exp(actions[..., 3].float().clamp(-LOG_SPEED_CLAMP, LOG_SPEED_CLAMP))
    return dirs * speed_ratio.unsqueeze(-1)


def normalized_paths(actions: torch.Tensor, target_actions: torch.Tensor, moving_mask: torch.Tensor) -> torch.Tensor:
    """Cumulative path normalized by target path length."""
    _check_actions("actions", actions)
    _check_actions("target_actions", target_actions)
    mask = _as_mask(moving_mask, actions.shape[:2]).float()

    steps = action_step_vectors(actions) * mask.unsqueeze(-1)
    target_steps = action_step_vectors(target_actions) * mask.unsqueeze(-1)
    target_length = target_steps.norm(dim=-1).sum(dim=1).clamp_min(EPS)
    return steps.cumsum(dim=1) / target_length[:, None, None]


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    denom = weights.sum()
    if float(denom) <= 0.0:
        return values.sum() * 0.0
    return (values * weights).sum() / denom


def compute_action_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    moving_mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    speed_mask: torch.Tensor | None = None,
) -> ActionMetricResult:
    """Compute scale-free action metrics for one batch.

    Direction and speed metrics are valid-step weighted. Path ADE/FDE are
    sample-weighted after each sample's path is normalized by target path length.
    """
    import torch

    _check_actions("predicted", predicted)
    _check_actions("target", target)
    if predicted.shape != target.shape:
        raise ValueError(f"predicted/target shape mismatch: {predicted.shape} vs {target.shape}")

    batch_size, horizon = predicted.shape[:2]
    mask = _as_mask(moving_mask, (batch_size, horizon))
    if speed_mask is not None:
        speed_valid = _as_mask(speed_mask, (batch_size, horizon)) & mask
    else:
        speed_valid = mask
    weights_b = (
        torch.ones(batch_size, device=predicted.device, dtype=torch.float32)
        if sample_weight is None
        else sample_weight.to(device=predicted.device, dtype=torch.float32)
    )
    if weights_b.shape != (batch_size,):
        raise ValueError(f"sample_weight: expected {(batch_size,)}, got {weights_b.shape}")

    with torch.no_grad():
        pred_dir = _normalize_dir(predicted)
        target_dir = _normalize_dir(target)
        cos = (pred_dir * target_dir).sum(dim=-1).clamp(-1.0, 1.0)
        angle = torch.rad2deg(torch.acos(cos))
        speed_mae = (predicted[..., 3].float() - target[..., 3].float()).abs()

        step_weights = mask.float() * weights_b[:, None]
        direction_cosine = _weighted_mean(cos, step_weights)
        angular_error_deg = _weighted_mean(angle, step_weights)
        speed_weights = speed_valid.float() * weights_b[:, None]
        speed_ratio_mae = _weighted_mean(speed_mae, speed_weights)

        pred_path = normalized_paths(predicted, target, mask)
        target_path = normalized_paths(target, target, mask)
        path_error = (pred_path - target_path).norm(dim=-1)
        valid_counts = mask.float().sum(dim=1)
        sample_valid = valid_counts > 0
        ade_per_sample = (path_error * mask.float()).sum(dim=1) / valid_counts.clamp_min(1.0)
        fde_idx = valid_counts.long().clamp_min(1) - 1
        fde_per_sample = path_error.gather(1, fde_idx[:, None]).squeeze(1)
        sample_weights = weights_b * sample_valid.float()
        path_ade = _weighted_mean(ade_per_sample, sample_weights)
        path_fde = _weighted_mean(fde_per_sample, sample_weights)

        aggregate = angular_error_deg / 180.0 + speed_ratio_mae + path_ade + path_fde

    return ActionMetricResult(
        direction_cosine=float(direction_cosine),
        angular_error_deg=float(angular_error_deg),
        path_ade=float(path_ade),
        path_fde=float(path_fde),
        speed_ratio_mae=float(speed_ratio_mae),
        aggregate_score=float(aggregate),
        n_samples=int(batch_size),
        n_valid=int(mask.sum()),
        n_speed_valid=int(speed_valid.sum()),
    )


def repeat_last_baseline(last_action_scale_free: torch.Tensor, horizon: int = HORIZON) -> torch.Tensor:
    """Repeat the previous observed scale-free action for every future horizon."""
    return last_action_scale_free.unsqueeze(1).expand(-1, horizon, -1).clone()


def no_turn_baseline(last_action_scale_free: torch.Tensor, horizon: int = HORIZON) -> torch.Tensor:
    """Keep the previous direction but command neutral relative speed."""
    out = repeat_last_baseline(last_action_scale_free, horizon=horizon)
    out[..., 3] = 0.0
    return out


def zero_action_baseline(
    batch_size: int,
    horizon: int = HORIZON,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Neutral action baseline: forward direction, log-speed ratio 0."""
    import torch

    out = torch.zeros(batch_size, horizon, SCALE_FREE_ACTION_DIM, device=device, dtype=dtype or torch.float32)
    out[..., 0] = 1.0
    return out


def mean_action_baseline(
    mean_action: torch.Tensor | None,
    batch_size: int,
    horizon: int = HORIZON,
) -> torch.Tensor:
    """Repeat a train-set mean action; fall back to the neutral baseline if absent."""
    if mean_action is None:
        return zero_action_baseline(batch_size, horizon=horizon)
    if mean_action.shape != (SCALE_FREE_ACTION_DIM,):
        raise ValueError(f"mean_action: expected {(SCALE_FREE_ACTION_DIM,)}, got {mean_action.shape}")
    return mean_action.unsqueeze(0).unsqueeze(1).expand(batch_size, horizon, -1).clone()


def linear_extrapolation_baseline(
    last_action_scale_free: torch.Tensor,
    previous_action_scale_free: torch.Tensor | None = None,
    horizon: int = HORIZON,
) -> torch.Tensor:
    """Linear action extrapolation from two past observed actions.

    If only one past action is available, this degrades deterministically to
    repeat-last.
    """
    import torch

    if previous_action_scale_free is None:
        return repeat_last_baseline(last_action_scale_free, horizon=horizon)
    delta = last_action_scale_free - previous_action_scale_free
    steps = torch.arange(1, horizon + 1, device=last_action_scale_free.device, dtype=last_action_scale_free.dtype)
    return last_action_scale_free[:, None, :] + steps[None, :, None] * delta[:, None, :]


def baseline_action_predictions(
    last_action_scale_free: torch.Tensor,
    *,
    horizon: int = HORIZON,
    mean_action: torch.Tensor | None = None,
    previous_action_scale_free: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return deterministic B2 baseline predictions."""
    batch_size = last_action_scale_free.shape[0]
    return {
        "repeat_last": repeat_last_baseline(last_action_scale_free, horizon=horizon),
        "no_turn": no_turn_baseline(last_action_scale_free, horizon=horizon),
        "zero": zero_action_baseline(
            batch_size,
            horizon=horizon,
            device=last_action_scale_free.device,
            dtype=last_action_scale_free.dtype,
        ),
        "mean": mean_action_baseline(mean_action, batch_size, horizon=horizon).to(
            device=last_action_scale_free.device,
            dtype=last_action_scale_free.dtype,
        ),
        "linear": linear_extrapolation_baseline(
            last_action_scale_free,
            previous_action_scale_free=previous_action_scale_free,
            horizon=horizon,
        ),
    }


def score_action_predictions(
    predicted: torch.Tensor,
    target: torch.Tensor,
    moving_mask: torch.Tensor,
    last_action_scale_free: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    speed_mask: torch.Tensor | None = None,
    *,
    mean_action: torch.Tensor | None = None,
    previous_action_scale_free: torch.Tensor | None = None,
) -> ActionScorecard:
    """Score model predictions and compute margin over the best deterministic baseline."""
    horizon = target.shape[1]
    model = compute_action_metrics(
        predicted,
        target,
        moving_mask,
        sample_weight=sample_weight,
        speed_mask=speed_mask,
    )
    baseline_metrics = {
        name: compute_action_metrics(
            base,
            target,
            moving_mask,
            sample_weight=sample_weight,
            speed_mask=speed_mask,
        )
        for name, base in baseline_action_predictions(
            last_action_scale_free,
            horizon=horizon,
            mean_action=mean_action,
            previous_action_scale_free=previous_action_scale_free,
        ).items()
    }
    best_name, best_metrics = min(baseline_metrics.items(), key=lambda item: item[1].aggregate_score)
    margin = best_metrics.aggregate_score - model.aggregate_score
    return ActionScorecard(
        model=model,
        baselines=baseline_metrics,
        best_baseline=best_name,
        best_baseline_score=best_metrics.aggregate_score,
        margin=margin,
    )


__all__ = [
    "ActionMetricResult",
    "ActionScorecard",
    "baseline_action_predictions",
    "compute_action_metrics",
    "linear_extrapolation_baseline",
    "mean_action_baseline",
    "no_turn_baseline",
    "normalized_paths",
    "repeat_last_baseline",
    "score_action_predictions",
    "zero_action_baseline",
]
