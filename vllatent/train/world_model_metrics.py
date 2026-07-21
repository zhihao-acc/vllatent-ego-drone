"""Historical passive-video Stage-1 world-model metrics (TORCH tier).

These six-field metrics remain only until their CS6/CS7 replacements land.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from vllatent.plan_tokens import PLAN_TOKEN_DIM
from vllatent.schemas import PATCH_TOKENS
from vllatent.train.world_model_losses import person_patch_weights

MIN_G1D_CONFIDENCE_SOURCES = 5

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
    true_beats_shuffled_windows: int
    true_beats_flipped_windows: int
    model_window_losses: list[float]
    shuffled_window_losses: list[float]
    flipped_window_losses: list[float]
    shuffled_window_wins: list[bool]
    flipped_window_wins: list[bool]
    n_windows: int


class YawGeometryCounts(NamedTuple):
    """Eligible/correct window counts for the yaw-only counterfactual."""

    correct: int
    eligible: int
    rate: float
    correct_by_window: list[bool]
    eligible_by_window: list[bool]


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


def yaw_flipped_plan(planned_actions: torch.Tensor) -> torch.Tensor:
    """Flip yaw-rate only, preserving translation, speed, and validity."""
    if planned_actions.ndim != 3 or planned_actions.shape[-1] != PLAN_TOKEN_DIM:
        raise ValueError(f"planned_actions: expected (B,T,{PLAN_TOKEN_DIM}), got {planned_actions.shape}")
    flipped = planned_actions.clone()
    flipped[..., 4] = -flipped[..., 4]
    return flipped


def build_plan_derangements(
    planned_actions: torch.Tensor,
    clip_ids: list[str],
    *,
    num_permutations: int,
    seed: int,
    min_plan_distance: float,
) -> list[torch.Tensor]:
    """Build deterministic, different-clip plan permutations for G1d.

    Each returned tensor maps evaluation row ``i`` to one donor-plan row. Every
    donor is used exactly once per permutation, fixed points and same-clip pairs
    are forbidden, and the RMS distance between paired plan tensors must clear
    ``min_plan_distance``. The function fails closed when those constraints do
    not admit the requested number of unique perfect matchings.
    """
    import math
    import random

    import torch

    if planned_actions.ndim != 3 or planned_actions.shape[-1] != PLAN_TOKEN_DIM:
        raise ValueError(f"planned_actions: expected (N,T,{PLAN_TOKEN_DIM}), got {planned_actions.shape}")
    n_windows = int(planned_actions.shape[0])
    if n_windows < 2:
        raise ValueError("plan derangement requires at least two windows")
    if len(clip_ids) != n_windows:
        raise ValueError(f"clip_ids: expected {n_windows} entries, got {len(clip_ids)}")
    if num_permutations < 1:
        raise ValueError(f"num_permutations must be >= 1, got {num_permutations}")
    if not math.isfinite(min_plan_distance) or min_plan_distance < 0.0:
        raise ValueError(f"min_plan_distance must be finite and >= 0, got {min_plan_distance}")
    if len(set(clip_ids)) < 2:
        raise ValueError("different-clip plan derangement is impossible with one clip")

    # The sixth field is a validity mask, not a physical action dimension.
    flattened = planned_actions[..., : PLAN_TOKEN_DIM - 1].detach().float().reshape(n_windows, -1).cpu()
    distances = torch.cdist(flattened, flattened) / math.sqrt(flattened.shape[1])
    identity = torch.eye(n_windows, dtype=torch.bool)
    different_clip = torch.tensor(
        [[clip_ids[row] != clip_ids[donor] for donor in range(n_windows)] for row in range(n_windows)],
        dtype=torch.bool,
    )
    allowed = ~identity & different_clip & (distances >= float(min_plan_distance))
    candidates = [torch.nonzero(allowed[row], as_tuple=False).flatten().tolist() for row in range(n_windows)]
    if any(not row_candidates for row_candidates in candidates):
        raise ValueError("no plan derangement satisfies the different-clip and distance constraints")

    rng = random.Random(seed)

    def find_matching() -> tuple[int, ...] | None:
        candidate_order = [list(row) for row in candidates]
        for row in candidate_order:
            rng.shuffle(row)
        row_order = list(range(n_windows))
        rng.shuffle(row_order)
        donor_to_row = [-1] * n_windows

        def assign(row: int, seen_donors: list[bool]) -> bool:
            for donor in candidate_order[row]:
                if seen_donors[donor]:
                    continue
                seen_donors[donor] = True
                previous_row = donor_to_row[donor]
                if previous_row == -1 or assign(previous_row, seen_donors):
                    donor_to_row[donor] = row
                    return True
            return False

        for row in row_order:
            if not assign(row, [False] * n_windows):
                return None

        row_to_donor = [-1] * n_windows
        for donor, row in enumerate(donor_to_row):
            row_to_donor[row] = donor
        return tuple(row_to_donor)

    permutations: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    max_attempts = max(16, num_permutations * 8)
    for _ in range(max_attempts):
        matching = find_matching()
        if matching is None:
            raise ValueError("no perfect plan derangement satisfies the different-clip and distance constraints")
        if matching in seen:
            continue
        seen.add(matching)
        permutations.append(matching)
        if len(permutations) == num_permutations:
            break
    if len(permutations) != num_permutations:
        raise ValueError(
            "could not construct the requested number of unique plan "
            "derangements under the different-clip and distance constraints"
        )
    return [
        torch.tensor(
            permutation,
            dtype=torch.long,
            device=planned_actions.device,
        )
        for permutation in permutations
    ]


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


def yaw_geometry_counts(
    predicted_person_state: torch.Tensor,
    yaw_flipped_person_state: torch.Tensor,
    planned_actions: torch.Tensor,
    dt_seconds: torch.Tensor,
    plan_valid_mask: torch.Tensor,
    *,
    min_cumulative_yaw: float = 0.05,
) -> YawGeometryCounts:
    """Count yaw flips whose final person-center response has the expected sign.

    Positive NED yaw turns the camera right, so a static image subject moves left.
    The comparison isolates yaw by keeping all other plan fields unchanged.
    """
    import math

    import torch

    if predicted_person_state.ndim != 3 or predicted_person_state.shape[-1] != 4:
        raise ValueError(
            "predicted_person_state: expected (B,T,4), "
            f"got {predicted_person_state.shape}"
        )
    if yaw_flipped_person_state.shape != predicted_person_state.shape:
        raise ValueError(
            "predicted/yaw-flipped person-state shape mismatch: "
            f"{predicted_person_state.shape} vs {yaw_flipped_person_state.shape}"
        )
    if planned_actions.shape != predicted_person_state.shape[:2] + (PLAN_TOKEN_DIM,):
        raise ValueError(
            "planned_actions: expected "
            f"{predicted_person_state.shape[:2] + (PLAN_TOKEN_DIM,)}, got {planned_actions.shape}"
        )
    if dt_seconds.shape != predicted_person_state.shape[:2]:
        raise ValueError(f"dt_seconds: expected {predicted_person_state.shape[:2]}, got {dt_seconds.shape}")
    if plan_valid_mask.shape != predicted_person_state.shape[:2]:
        raise ValueError(
            f"plan_valid_mask: expected {predicted_person_state.shape[:2]}, got {plan_valid_mask.shape}"
        )
    if not math.isfinite(min_cumulative_yaw) or min_cumulative_yaw <= 0.0:
        raise ValueError(
            "min_cumulative_yaw must be finite and > 0, "
            f"got {min_cumulative_yaw}"
        )

    valid = plan_valid_mask.to(device=planned_actions.device, dtype=torch.bool)
    cumulative_yaw = (
        planned_actions[..., 4].float()
        * dt_seconds.to(device=planned_actions.device, dtype=torch.float32)
        * valid.to(dtype=torch.float32)
    ).sum(dim=1)
    eligible = cumulative_yaw.abs() >= float(min_cumulative_yaw)
    center_response = (
        predicted_person_state[:, -1, 0].float()
        - yaw_flipped_person_state[:, -1, 0].float()
    )
    correct = eligible & ((center_response * cumulative_yaw) < 0.0)
    n_eligible = int(eligible.sum().detach().cpu())
    n_correct = int(correct.sum().detach().cpu())
    return YawGeometryCounts(
        correct=n_correct,
        eligible=n_eligible,
        rate=float(n_correct / n_eligible) if n_eligible else 0.0,
        correct_by_window=[bool(value) for value in correct.detach().cpu()],
        eligible_by_window=[bool(value) for value in eligible.detach().cpu()],
    )


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
    import torch

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
    if shuffled_plan_latents.ndim == 4:
        shuffled_per_negative = per_window_latent_loss(
            shuffled_plan_latents,
            target_latents,
            person_state_target,
            person_state_valid,
            person_conf,
        ).unsqueeze(0)
    elif shuffled_plan_latents.ndim == 5:
        shuffled_per_negative = torch.stack(
            [
                per_window_latent_loss(
                    negative,
                    target_latents,
                    person_state_target,
                    person_state_valid,
                    person_conf,
                )
                for negative in shuffled_plan_latents
            ],
            dim=0,
        )
    else:
        raise ValueError(
            "shuffled_plan_latents: expected (B,T,P,D) or (N,B,T,P,D), "
            f"got {shuffled_plan_latents.shape}"
        )
    shuffled = shuffled_per_negative.mean(dim=0)
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
    n_windows = int(model.shape[0])
    if n_windows < 1:
        raise ValueError("evaluation batch must contain at least one window")
    model_window = model.mean(dim=1)
    shuffled_negative_windows = shuffled_per_negative.mean(dim=2)
    shuffled_window = shuffled.mean(dim=1)
    flipped_window = flipped.mean(dim=1)
    shuffled_majority_wins = (
        (model_window.unsqueeze(0) < shuffled_negative_windows)
        .float()
        .mean(dim=0)
        > 0.5
    )
    flipped_wins = model_window < flipped_window
    shuffled_win_count = int(shuffled_majority_wins.sum().detach().cpu())
    flipped_win_count = int(flipped_wins.sum().detach().cpu())
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
        true_beats_shuffled_rate=shuffled_win_count / n_windows,
        true_beats_flipped_rate=flipped_win_count / n_windows,
        true_beats_shuffled_windows=shuffled_win_count,
        true_beats_flipped_windows=flipped_win_count,
        model_window_losses=[float(value) for value in model_window.detach().cpu()],
        shuffled_window_losses=[float(value) for value in shuffled_window.detach().cpu()],
        flipped_window_losses=[float(value) for value in flipped_window.detach().cpu()],
        shuffled_window_wins=[bool(value) for value in shuffled_majority_wins.detach().cpu()],
        flipped_window_wins=[bool(value) for value in flipped_wins.detach().cpu()],
        n_windows=n_windows,
    )


def _cluster_index_groups(cluster_ids: list[str], *, n_values: int) -> list[list[int]]:
    """Return stable index groups for a source-cluster bootstrap."""
    if len(cluster_ids) != n_values:
        raise ValueError(
            f"cluster_ids: expected {n_values} entries, got {len(cluster_ids)}"
        )
    groups: dict[str, list[int]] = {}
    for index, cluster_id in enumerate(cluster_ids):
        if not cluster_id:
            raise ValueError("cluster_ids must be non-empty strings")
        groups.setdefault(cluster_id, []).append(index)
    return list(groups.values())


def wilson_interval(
    successes: int,
    total: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Return a boundary-aware two-sided Wilson interval for a binomial rate."""
    import math

    if total < 1:
        raise ValueError(f"total must be >= 1, got {total}")
    if successes < 0 or successes > total:
        raise ValueError(f"successes must be in [0, {total}], got {successes}")
    if not math.isfinite(z) or z <= 0.0:
        raise ValueError(f"z must be finite and > 0, got {z}")
    rate = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (rate + z2 / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            rate * (1.0 - rate) / total
            + z2 / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def source_majority_counts(
    values: list[bool],
    cluster_ids: list[str],
) -> tuple[int, int]:
    """Count sources whose within-source binary rate is a strict majority."""
    groups = _cluster_index_groups(cluster_ids, n_values=len(values))
    successes = sum(
        sum(bool(values[index]) for index in group) / len(group) > 0.5
        for group in groups
    )
    return successes, len(groups)


def clustered_paired_margin_ci(
    model_losses: list[float],
    counterfactual_losses: list[float],
    cluster_ids: list[str],
    *,
    seed: int = 0,
    n_resamples: int = 2000,
) -> tuple[float | None, float | None]:
    """Bootstrap a paired relative-loss margin by resampling source clusters."""
    import numpy as np

    model = np.asarray(model_losses, dtype=np.float64)
    counterfactual = np.asarray(counterfactual_losses, dtype=np.float64)
    if model.ndim != 1 or counterfactual.shape != model.shape:
        raise ValueError(
            "model/counterfactual losses must be equal-length one-dimensional arrays"
        )
    if not np.all(np.isfinite(model)) or not np.all(np.isfinite(counterfactual)):
        raise ValueError("paired losses must be finite")
    groups = _cluster_index_groups(cluster_ids, n_values=int(model.size))
    if len(groups) < 2:
        return None, None
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    rng = np.random.default_rng(seed)
    margins = np.empty(n_resamples, dtype=np.float64)
    for sample_index in range(n_resamples):
        selected_groups = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate(
            [np.asarray(groups[int(group)], dtype=np.int64) for group in selected_groups]
        )
        sampled_model = model[indices].mean()
        sampled_counterfactual = counterfactual[indices].mean()
        margins[sample_index] = (
            sampled_counterfactual - sampled_model
        ) / max(sampled_counterfactual, 1e-8)
    lower, upper = np.percentile(margins, [2.5, 97.5])
    return float(lower), float(upper)


def aggregate_stage1_metrics(
    metrics: list[Stage1BatchMetrics],
    *,
    cluster_ids: list[str],
    yaw_geometry_correct_by_window: list[bool],
    yaw_geometry_eligible_by_window: list[bool],
) -> dict[str, object]:
    """Aggregate per-batch Stage-1 metrics into JSON-friendly gate readouts."""
    import numpy as np

    if not metrics:
        raise ValueError("metrics must not be empty")
    horizon = len(metrics[0].per_step_model_loss)
    for item in metrics:
        if len(item.per_step_model_loss) != horizon:
            raise ValueError("all metrics must have the same horizon")
        if item.n_windows < 1:
            raise ValueError("each batch metric must represent at least one window")
        if not (
            len(item.model_window_losses)
            == len(item.shuffled_window_losses)
            == len(item.flipped_window_losses)
            == len(item.shuffled_window_wins)
            == len(item.flipped_window_wins)
            == item.n_windows
        ):
            raise ValueError("per-window metric arrays must match n_windows")

    batch_weights = np.asarray([item.n_windows for item in metrics], dtype=np.float64)
    n_windows = int(batch_weights.sum())
    if len(cluster_ids) != n_windows:
        raise ValueError(
            f"cluster_ids: expected {n_windows} entries, got {len(cluster_ids)}"
        )
    if len(yaw_geometry_correct_by_window) != n_windows:
        raise ValueError(
            "yaw_geometry_correct_by_window: expected "
            f"{n_windows} entries, got {len(yaw_geometry_correct_by_window)}"
        )
    if len(yaw_geometry_eligible_by_window) != n_windows:
        raise ValueError(
            "yaw_geometry_eligible_by_window: expected "
            f"{n_windows} entries, got {len(yaw_geometry_eligible_by_window)}"
        )
    if any(
        correct and not eligible
        for correct, eligible in zip(
            yaw_geometry_correct_by_window,
            yaw_geometry_eligible_by_window,
            strict=True,
        )
    ):
        raise ValueError("a yaw response cannot be correct when it is not eligible")

    def weighted_attr(name: str) -> float:
        values = np.asarray([getattr(item, name) for item in metrics], dtype=np.float64)
        return float(np.average(values, weights=batch_weights))

    per_step_model = np.average(
        np.asarray([item.per_step_model_loss for item in metrics], dtype=np.float64),
        axis=0,
        weights=batch_weights,
    )
    per_step_persistence = np.average(
        np.asarray(
            [item.per_step_persistence_loss for item in metrics],
            dtype=np.float64,
        ),
        axis=0,
        weights=batch_weights,
    )
    rollout = per_step_model < per_step_persistence
    model_loss = weighted_attr("model_loss")
    persistence_loss = weighted_attr("persistence_loss")
    null_loss = weighted_attr("null_plan_loss")
    shuffled_loss = weighted_attr("shuffled_plan_loss")
    flipped_loss = weighted_attr("flipped_plan_loss")
    g1a_persistence = (persistence_loss - model_loss) / max(persistence_loss, 1e-8)
    g1a_null = (null_loss - model_loss) / max(null_loss, 1e-8)
    improvement_vs_shuffled = (shuffled_loss - model_loss) / max(shuffled_loss, 1e-8)
    improvement_vs_flipped = (flipped_loss - model_loss) / max(flipped_loss, 1e-8)
    shuffled_window_wins = [
        value for item in metrics for value in item.shuffled_window_wins
    ]
    flipped_window_wins = [
        value for item in metrics for value in item.flipped_window_wins
    ]
    shuffled_wins = sum(shuffled_window_wins)
    flipped_wins = sum(flipped_window_wins)
    true_beats_shuffled = shuffled_wins / n_windows
    true_beats_flipped = flipped_wins / n_windows
    shuffled_source_wins, confidence_source_count = source_majority_counts(
        shuffled_window_wins,
        cluster_ids,
    )
    flipped_source_wins, flipped_source_count = source_majority_counts(
        flipped_window_wins,
        cluster_ids,
    )
    if flipped_source_count != confidence_source_count:
        raise ValueError("shuffled/flipped confidence source counts differ")
    shuffled_rate_ci = wilson_interval(
        shuffled_source_wins,
        confidence_source_count,
    )
    flipped_rate_ci = wilson_interval(
        flipped_source_wins,
        confidence_source_count,
    )
    yaw_geometry_correct = sum(yaw_geometry_correct_by_window)
    yaw_geometry_eligible = sum(yaw_geometry_eligible_by_window)
    if yaw_geometry_eligible:
        yaw_geometry_rate = yaw_geometry_correct / yaw_geometry_eligible
        eligible_yaw_values = [
            correct
            for correct, eligible in zip(
                yaw_geometry_correct_by_window,
                yaw_geometry_eligible_by_window,
                strict=True,
            )
            if eligible
        ]
        eligible_yaw_clusters = [
            cluster_id
            for cluster_id, eligible in zip(
                cluster_ids,
                yaw_geometry_eligible_by_window,
                strict=True,
            )
            if eligible
        ]
        yaw_source_wins, yaw_eligible_source_count = source_majority_counts(
            eligible_yaw_values,
            eligible_yaw_clusters,
        )
        yaw_rate_ci = wilson_interval(yaw_source_wins, yaw_eligible_source_count)
    else:
        yaw_geometry_rate = 0.0
        yaw_rate_ci = (None, None)
        yaw_source_wins = 0
        yaw_eligible_source_count = 0
    model_window_losses = [
        value for item in metrics for value in item.model_window_losses
    ]
    shuffled_window_losses = [
        value for item in metrics for value in item.shuffled_window_losses
    ]
    flipped_window_losses = [
        value for item in metrics for value in item.flipped_window_losses
    ]
    shuffled_margin_ci = clustered_paired_margin_ci(
        model_window_losses,
        shuffled_window_losses,
        cluster_ids,
        seed=0,
    )
    flipped_margin_ci = clustered_paired_margin_ci(
        model_window_losses,
        flipped_window_losses,
        cluster_ids,
        seed=1,
    )
    shuffled_margin_confident = (
        shuffled_margin_ci[0] is not None and shuffled_margin_ci[0] > 0.0
    )
    flipped_margin_confident = (
        flipped_margin_ci[0] is not None and flipped_margin_ci[0] > 0.0
    )
    return {
        "n_windows": n_windows,
        "model_loss": model_loss,
        "persistence_loss": persistence_loss,
        "null_plan_loss": null_loss,
        "shuffled_plan_loss": shuffled_loss,
        "flipped_plan_loss": flipped_loss,
        "improvement_vs_persistence": g1a_persistence,
        "improvement_vs_null": g1a_null,
        "improvement_vs_shuffled": improvement_vs_shuffled,
        "improvement_vs_flipped": improvement_vs_flipped,
        "per_step_model_loss": [float(x) for x in per_step_model],
        "per_step_persistence_loss": [float(x) for x in per_step_persistence],
        "rollout_beats_persistence": [bool(x) for x in rollout],
        "true_beats_shuffled_rate": true_beats_shuffled,
        "true_beats_flipped_rate": true_beats_flipped,
        "true_beats_shuffled_windows": shuffled_wins,
        "true_beats_flipped_windows": flipped_wins,
        "true_beats_shuffled_source_majorities": shuffled_source_wins,
        "true_beats_flipped_source_majorities": flipped_source_wins,
        "true_beats_shuffled_ci95": list(shuffled_rate_ci),
        "true_beats_flipped_ci95": list(flipped_rate_ci),
        "improvement_vs_shuffled_ci95": list(shuffled_margin_ci),
        "improvement_vs_flipped_ci95": list(flipped_margin_ci),
        "confidence_cluster_unit": "source",
        "confidence_cluster_count": confidence_source_count,
        "confidence_min_source_count": MIN_G1D_CONFIDENCE_SOURCES,
        "yaw_geometry_rate": float(yaw_geometry_rate),
        "yaw_geometry_correct_windows": yaw_geometry_correct,
        "yaw_geometry_eligible_windows": yaw_geometry_eligible,
        "yaw_geometry_correct_source_majorities": yaw_source_wins,
        "yaw_geometry_eligible_source_count": yaw_eligible_source_count,
        "yaw_geometry_ci95": list(yaw_rate_ci),
        "yaw_geometry_ci95_lower": yaw_rate_ci[0],
        "g1a_pass": bool(g1a_persistence >= 0.10 and g1a_null >= 0.05),
        "g1b_pass": bool(np.all(rollout)),
        "g1d_pass": bool(
            true_beats_shuffled >= 0.70
            and true_beats_flipped >= 0.70
            and confidence_source_count >= MIN_G1D_CONFIDENCE_SOURCES
            and shuffled_rate_ci[0] is not None
            and shuffled_rate_ci[0] > 0.50
            and flipped_rate_ci[0] is not None
            and flipped_rate_ci[0] > 0.50
            and improvement_vs_shuffled >= 0.05
            and improvement_vs_flipped >= 0.05
            and shuffled_margin_confident
            and flipped_margin_confident
            and yaw_rate_ci[0] is not None
            and yaw_rate_ci[0] > 0.50
            and yaw_eligible_source_count >= MIN_G1D_CONFIDENCE_SOURCES
        ),
    }


__all__ = [
    "MIN_G1D_CONFIDENCE_SOURCES",
    "Stage1BatchMetrics",
    "YawGeometryCounts",
    "aggregate_stage1_metrics",
    "build_plan_derangements",
    "clustered_paired_margin_ci",
    "flipped_plan",
    "null_plan",
    "per_window_latent_loss",
    "persistence_rollout",
    "source_majority_counts",
    "summarize_stage1_batch",
    "wilson_interval",
    "yaw_flipped_plan",
    "yaw_geometry_counts",
]
