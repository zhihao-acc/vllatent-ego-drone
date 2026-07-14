#!/usr/bin/env python
"""B3 Stage-1 local training/gate harness.

This is intentionally narrow: it trains the depth-6 B3 model on the existing
latent cache and reports G1a/G1b/G1d-style local metrics. It does not download
data, run H20, or operate external systems.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import numpy as np

from vllatent.data.collate import TrainingBatch, collate_sports_batch
from vllatent.data.sports_loader import SportsTrainingDataset
from vllatent.model.human_world_model import (
    HumanWorldModel,
    apply_whole_plan_dropout,
    count_parameters,
)
from vllatent.plan_tokens import PLAN_TOKEN_FIELDS
from vllatent.schemas import EMBED_DIM, HISTORY
from vllatent.train.world_model_losses import (
    human_world_model_loss,
    physical_inverse_plan_loss,
)
from vllatent.train.world_model_metrics import (
    aggregate_stage1_metrics,
    build_plan_derangements,
    flipped_plan,
    null_plan,
    persistence_rollout,
    summarize_stage1_batch,
    yaw_flipped_plan,
    yaw_geometry_counts,
)

PHYSICAL_PLAN_FIELDS = PLAN_TOKEN_FIELDS[:-1]
G1D_PROTOCOL_VERSION = "global-different-clip-multineg-source-wilson-v3"
B3_PROTOCOL_VERSION = "b3.6-transition-verifier-g1d-source-ci-v4"


class SplitIndices(NamedTuple):
    train: list[int]
    val: list[int]
    train_sources: list[str]
    val_sources: list[str]


class TrainingRunResult(NamedTuple):
    """Main predictor training trace and objective diagnostics."""

    losses: list[float]
    steps_per_second: float
    components: dict[str, object]
    gradient_interaction: dict[str, float | None]


def real_transition_previous_latents(z_t, target_latents):
    """Teacher-forced previous state for each real/predicted next transition."""
    import torch

    if z_t.ndim != 3:
        raise ValueError(f"z_t: expected (B,P,D), got {z_t.shape}")
    if target_latents.ndim != 4:
        raise ValueError(f"target_latents: expected (B,T,P,D), got {target_latents.shape}")
    if target_latents.shape[0] != z_t.shape[0] or target_latents.shape[2:] != z_t.shape[1:]:
        raise ValueError(
            "z_t/target latent shape mismatch: "
            f"{z_t.shape} vs {target_latents.shape}"
        )
    return torch.cat([z_t.unsqueeze(1), target_latents[:, :-1]], dim=1)


def _index_sha256(indices: list[int]) -> str:
    values = np.asarray(indices, dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _per_source_counts(sample_sources: list[str], indices: list[int]) -> dict[str, int]:
    return dict(sorted(Counter(sample_sources[index] for index in indices).items()))


def _series_summary(values: list[float], *, window: int = 5) -> dict[str, float | None]:
    initial, final, improvement = loss_window_improvement(values, window=window)
    return {
        "first": values[0] if values else None,
        "last": values[-1] if values else None,
        "mean": float(np.mean(values)) if values else None,
        "initial_window_mean": initial,
        "final_window_mean": final,
        "window_improvement": improvement,
    }


def source_split_indices(
    sample_sources: list[str],
    *,
    val_frac: float = 0.25,
    seed: int = 0,
) -> SplitIndices:
    """Split sample indices by source video, not subclip/window."""
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be in (0,1), got {val_frac}")
    sources = sorted(set(sample_sources))
    if len(sources) < 2:
        raise ValueError("need at least two sources for source split")
    rng = random.Random(seed)
    rng.shuffle(sources)
    n_val = min(len(sources) - 1, max(1, round(len(sources) * val_frac)))
    val_sources = sorted(sources[:n_val])
    train_sources = sorted(sources[n_val:])
    val_set = set(val_sources)
    train = [idx for idx, src in enumerate(sample_sources) if src not in val_set]
    val = [idx for idx, src in enumerate(sample_sources) if src in val_set]
    if not train or not val:
        raise ValueError("source split produced an empty train or val set")
    return SplitIndices(train=train, val=val, train_sources=train_sources, val_sources=val_sources)


def limit_indices(indices: list[int], max_samples: int | None, *, seed: int) -> list[int]:
    """Deterministically subsample indices for local gate runs."""
    if max_samples is None or len(indices) <= max_samples:
        return list(indices)
    rng = random.Random(seed)
    limited = list(indices)
    rng.shuffle(limited)
    return sorted(limited[:max_samples])


def source_balanced_limit_indices(
    indices: list[int],
    sample_sources: list[str],
    *,
    max_samples: int | None,
    seed: int,
) -> list[int]:
    """Deterministically cap windows by round-robin sampling across sources."""
    if len(set(indices)) != len(indices):
        raise ValueError("indices must be unique")
    if any(index < 0 or index >= len(sample_sources) for index in indices):
        raise ValueError("indices contain an out-of-range sample index")
    if max_samples is not None and max_samples < 0:
        raise ValueError(f"max_samples must be >= 0 or None, got {max_samples}")
    if max_samples is None or len(indices) <= max_samples:
        return list(indices)
    if max_samples == 0:
        return []

    by_source: dict[str, list[int]] = {}
    for index in indices:
        by_source.setdefault(sample_sources[index], []).append(index)
    rng = random.Random(seed)
    source_order = sorted(by_source)
    rng.shuffle(source_order)
    for source_indices in by_source.values():
        rng.shuffle(source_indices)

    selected: list[int] = []
    offsets = {source: 0 for source in source_order}
    while len(selected) < max_samples:
        made_progress = False
        for source in source_order:
            offset = offsets[source]
            source_indices = by_source[source]
            if offset >= len(source_indices):
                continue
            selected.append(source_indices[offset])
            offsets[source] += 1
            made_progress = True
            if len(selected) == max_samples:
                break
        if not made_progress:
            break
    return sorted(selected)


def represented_sources(
    sample_sources: list[str],
    indices: list[int],
) -> list[str]:
    """Return sorted source ids actually represented by selected indices."""
    if any(index < 0 or index >= len(sample_sources) for index in indices):
        raise ValueError("indices contain an out-of-range sample index")
    return sorted({sample_sources[index] for index in indices})


def select_train_val_indices(
    split: SplitIndices,
    *,
    sample_sources: list[str] | None = None,
    train_max_samples: int | None,
    val_max_samples: int | None,
    overfit_tiny: bool,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Select train/val windows, with overfit-tiny evaluating the exact train slice."""
    if sample_sources is None:
        train_indices = limit_indices(split.train, train_max_samples, seed=seed)
    else:
        train_indices = source_balanced_limit_indices(
            split.train,
            sample_sources,
            max_samples=train_max_samples,
            seed=seed,
        )
    if overfit_tiny:
        return train_indices, list(train_indices)
    if sample_sources is None:
        val_indices = limit_indices(split.val, val_max_samples, seed=seed + 1)
    else:
        val_indices = source_balanced_limit_indices(
            split.val,
            sample_sources,
            max_samples=val_max_samples,
            seed=seed + 1,
        )
    return train_indices, val_indices


def loss_window_improvement(losses: list[float], *, window: int = 5) -> tuple[float | None, float | None, float | None]:
    """Compare mean early and late training losses for tiny-overfit reporting."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if len(losses) < 2:
        return None, None, None
    n = min(window, len(losses))
    initial = float(np.mean(losses[:n]))
    final = float(np.mean(losses[-n:]))
    return initial, final, (initial - final) / max(initial, 1e-8)


def make_loader(
    dataset: SportsTrainingDataset,
    indices: list[int],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
):
    import torch
    from torch.utils.data import DataLoader, Subset

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=num_workers,
        collate_fn=collate_sports_batch,
        pin_memory=False,
    )


def _move_batch(batch: TrainingBatch, device):
    return {
        "history_latents": batch.history_latents.to(device),
        "z_t": batch.z_t.to(device),
        "history_mask": batch.history_mask.to(device),
        "planned_actions": batch.planned_actions.to(device),
        "dt_seconds": batch.dt_seconds.to(device),
        "target_latents": batch.target_latents.to(device),
        "person_state_target": batch.person_state_target.to(device),
        "person_state_valid": batch.target_person_state_valid.to(device),
        "person_conf": batch.target_person_conf.to(device),
        "planned_actions_valid_mask": batch.planned_actions_valid_mask.to(device),
        "sample_weight": batch.sample_weight.to(device),
    }


def mean_physical_plan(loader, *, device):
    """Compute the valid-step training mean used by the inverse baseline."""
    import torch

    total = torch.zeros(len(PHYSICAL_PLAN_FIELDS), device=device, dtype=torch.float64)
    count = 0
    for batch in loader:
        plan = batch.planned_actions.to(device=device, dtype=torch.float64)
        valid = batch.planned_actions_valid_mask.to(device=device, dtype=torch.bool)
        total += (plan[..., : len(PHYSICAL_PLAN_FIELDS)] * valid.unsqueeze(-1)).sum(dim=(0, 1))
        count += int(valid.sum().detach().cpu())
    if count == 0:
        raise ValueError("training split has no valid physical plan transitions")
    return (total / count).to(dtype=torch.float32)


def pretrain_transition_verifier(
    model,
    loader,
    *,
    device,
    max_steps: int,
    lr: float,
    use_amp: bool,
) -> dict[str, object]:
    """Fit the action-blind verifier on real consecutive DINO transitions."""
    import torch

    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")
    verifier = model.transition_plan_verifier
    verifier.requires_grad_(True)
    verifier.train()
    optimizer = torch.optim.AdamW(verifier.parameters(), lr=lr, weight_decay=0.01)
    totals: list[float] = []
    per_field: list[list[float]] = [[] for _ in PHYSICAL_PLAN_FIELDS]
    start = time.perf_counter()
    step = 0
    while step < max_steps:
        for batch in loader:
            tensors = _move_batch(batch, device)
            previous = real_transition_previous_latents(
                tensors["z_t"],
                tensors["target_latents"],
            ).detach()
            target = tensors["target_latents"].detach()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                recovered = verifier(previous, target)
                inverse = physical_inverse_plan_loss(
                    recovered,
                    tensors["planned_actions"],
                    tensors["planned_actions_valid_mask"],
                )
            inverse.total.backward()
            optimizer.step()
            totals.append(float(inverse.total.detach().cpu()))
            for field_idx, value in enumerate(inverse.per_field.detach().cpu()):
                per_field[field_idx].append(float(value))
            step += 1
            if step >= max_steps:
                break
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "steps": len(totals),
        "lr": lr,
        "steps_per_second": len(totals) / max(elapsed, 1e-8),
        "loss": _series_summary(totals),
        "per_field_loss_mean": {
            field: float(np.mean(values)) if values else None
            for field, values in zip(PHYSICAL_PLAN_FIELDS, per_field, strict=True)
        },
    }


def evaluate_transition_verifier(
    model,
    loader,
    *,
    device,
    mean_plan,
    max_batches: int,
    use_amp: bool,
) -> dict[str, object]:
    """Evaluate real-transition action recovery against a train-mean baseline."""
    import torch

    verifier = model.transition_plan_verifier
    verifier.eval()
    loss_sum = 0.0
    baseline_sum = 0.0
    field_sum = np.zeros(len(PHYSICAL_PLAN_FIELDS), dtype=np.float64)
    baseline_field_sum = np.zeros(len(PHYSICAL_PLAN_FIELDS), dtype=np.float64)
    transition_count = 0
    window_count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            tensors = _move_batch(batch, device)
            previous = real_transition_previous_latents(
                tensors["z_t"],
                tensors["target_latents"],
            )
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                recovered = verifier(previous, tensors["target_latents"])
                inverse = physical_inverse_plan_loss(
                    recovered,
                    tensors["planned_actions"],
                    tensors["planned_actions_valid_mask"],
                )
                baseline_prediction = mean_plan.to(
                    device=device,
                    dtype=recovered.dtype,
                ).view(1, 1, -1).expand_as(recovered)
                baseline = physical_inverse_plan_loss(
                    baseline_prediction,
                    tensors["planned_actions"],
                    tensors["planned_actions_valid_mask"],
                )
            count = int(tensors["planned_actions_valid_mask"].sum().detach().cpu())
            if count == 0:
                continue
            loss_sum += float(inverse.total.detach().cpu()) * count
            baseline_sum += float(baseline.total.detach().cpu()) * count
            field_sum += inverse.per_field.detach().cpu().numpy().astype(np.float64) * count
            baseline_field_sum += baseline.per_field.detach().cpu().numpy().astype(np.float64) * count
            transition_count += count
            window_count += int(tensors["planned_actions"].shape[0])
    if transition_count == 0:
        raise ValueError("verifier evaluation found no valid plan transitions")
    loss = loss_sum / transition_count
    baseline_loss = baseline_sum / transition_count
    improvement = (baseline_loss - loss) / max(baseline_loss, 1e-8)
    return {
        "windows": window_count,
        "valid_transitions": transition_count,
        "loss": loss,
        "mean_plan_baseline_loss": baseline_loss,
        "improvement_vs_mean_plan": improvement,
        "passed": bool(improvement > 0.0),
        "per_field_loss": {
            field: float(value / transition_count)
            for field, value in zip(PHYSICAL_PLAN_FIELDS, field_sum, strict=True)
        },
        "per_field_mean_plan_baseline_loss": {
            field: float(value / transition_count)
            for field, value in zip(
                PHYSICAL_PLAN_FIELDS,
                baseline_field_sum,
                strict=True,
            )
        },
    }


def _predictor_gradient_interaction(primary_loss, inverse_loss, parameters) -> dict[str, float | None]:
    """Measure unweighted shared-predictor gradient norms and cosine once."""
    import math

    import torch

    primary_grads = torch.autograd.grad(
        primary_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    inverse_grads = torch.autograd.grad(
        inverse_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    primary_sq = torch.zeros((), device=primary_loss.device, dtype=torch.float64)
    inverse_sq = torch.zeros_like(primary_sq)
    dot = torch.zeros_like(primary_sq)
    for primary_grad, inverse_grad in zip(primary_grads, inverse_grads, strict=True):
        if primary_grad is not None:
            primary_sq += primary_grad.detach().double().square().sum()
        if inverse_grad is not None:
            inverse_sq += inverse_grad.detach().double().square().sum()
        if primary_grad is not None and inverse_grad is not None:
            dot += (primary_grad.detach().double() * inverse_grad.detach().double()).sum()
    primary_norm = math.sqrt(float(primary_sq.detach().cpu()))
    inverse_norm = math.sqrt(float(inverse_sq.detach().cpu()))
    cosine = None
    if primary_norm > 0.0 and inverse_norm > 0.0:
        cosine = float(dot.detach().cpu()) / (primary_norm * inverse_norm)
    return {
        "primary_gradient_norm": primary_norm,
        "inverse_gradient_norm_unweighted": inverse_norm,
        "cosine": cosine,
    }


def train_steps(
    model,
    loader,
    *,
    device,
    max_steps: int,
    lr: float,
    use_amp: bool,
    lambda_latent: float,
    lambda_person_state: float,
    lambda_inverse_plan: float,
    action_dropout_p: float,
    dropout_seed: int,
) -> TrainingRunResult:
    import torch

    if any(parameter.requires_grad for parameter in model.transition_plan_verifier.parameters()):
        raise ValueError("transition verifier must be frozen before predictor training")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=lr,
        weight_decay=0.05,
    )
    losses: list[float] = []
    component_values: dict[str, list[float]] = {
        "latent": [],
        "person_state": [],
        "inverse_plan": [],
        "latent_cosine": [],
    }
    per_field_values: dict[str, list[float]] = {
        field: [] for field in PHYSICAL_PLAN_FIELDS
    }
    gradient_interaction: dict[str, float | None] | None = None
    plan_dropout_generator = torch.Generator(device=device)
    plan_dropout_generator.manual_seed(dropout_seed)
    start = time.perf_counter()
    model.train()
    model.transition_plan_verifier.eval()
    step = 0
    while step < max_steps:
        for batch in loader:
            tensors = _move_batch(batch, device)
            conditioned_plans, plan_keep_mask = apply_whole_plan_dropout(
                tensors["planned_actions"],
                dropout_p=action_dropout_p,
                training=True,
                generator=plan_dropout_generator,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = model(
                    tensors["history_latents"],
                    tensors["z_t"],
                    tensors["history_mask"],
                    conditioned_plans,
                    tensors["dt_seconds"],
                )
                real_previous = real_transition_previous_latents(
                    tensors["z_t"],
                    tensors["target_latents"],
                ).detach()
                recovered_plan = model.recover_plan(
                    real_previous,
                    out.predicted_latents,
                )
                loss_output = human_world_model_loss(
                    predicted_latents=out.predicted_latents,
                    target_latents=tensors["target_latents"],
                    predicted_person_state=out.predicted_person_state,
                    person_state_target=tensors["person_state_target"],
                    person_state_valid=tensors["person_state_valid"],
                    predicted_plan=recovered_plan,
                    planned_actions=tensors["planned_actions"],
                    planned_actions_valid_mask=tensors["planned_actions_valid_mask"],
                    plan_keep_mask=plan_keep_mask,
                    person_conf=tensors["person_conf"],
                    sample_weight=tensors["sample_weight"],
                    lambda_latent=lambda_latent,
                    lambda_person_state=lambda_person_state,
                    lambda_inverse_plan=lambda_inverse_plan,
                )
            if gradient_interaction is None and float(loss_output.inverse_plan.detach().cpu()) > 0.0:
                predictor_parameters = [
                    parameter
                    for parameter in model.predictor.parameters()
                    if parameter.requires_grad
                ]
                primary = (
                    lambda_latent * loss_output.latent
                    + lambda_person_state * loss_output.person_state
                )
                gradient_interaction = _predictor_gradient_interaction(
                    primary,
                    loss_output.inverse_plan,
                    predictor_parameters,
                )
                inverse_norm = gradient_interaction["inverse_gradient_norm_unweighted"]
                primary_norm = gradient_interaction["primary_gradient_norm"]
                weighted_inverse_norm = (
                    None
                    if inverse_norm is None
                    else lambda_inverse_plan * inverse_norm
                )
                gradient_interaction["inverse_gradient_norm_weighted"] = weighted_inverse_norm
                gradient_interaction["weighted_inverse_to_primary_norm_ratio"] = (
                    None
                    if weighted_inverse_norm is None
                    or primary_norm is None
                    or primary_norm <= 0.0
                    else weighted_inverse_norm / primary_norm
                )
            loss_output.total.backward()
            optimizer.step()
            losses.append(float(loss_output.total.detach().cpu()))
            component_values["latent"].append(float(loss_output.latent.detach().cpu()))
            component_values["person_state"].append(
                float(loss_output.person_state.detach().cpu())
            )
            component_values["inverse_plan"].append(
                float(loss_output.inverse_plan.detach().cpu())
            )
            component_values["latent_cosine"].append(
                float(loss_output.latent_cosine.detach().cpu())
            )
            for field, value in zip(
                PHYSICAL_PLAN_FIELDS,
                loss_output.inverse_plan_per_field.detach().cpu(),
                strict=True,
            ):
                per_field_values[field].append(float(value))
            step += 1
            if step >= max_steps:
                break
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return TrainingRunResult(
        losses=losses,
        steps_per_second=float(len(losses) / max(elapsed, 1e-8)),
        components={
            **{
                name: _series_summary(values)
                for name, values in component_values.items()
            },
            "inverse_plan_per_field": {
                field: _series_summary(values)
                for field, values in per_field_values.items()
            },
        },
        gradient_interaction=gradient_interaction
        or {
            "primary_gradient_norm": None,
            "inverse_gradient_norm_unweighted": None,
            "inverse_gradient_norm_weighted": None,
            "weighted_inverse_to_primary_norm_ratio": None,
            "cosine": None,
        },
    )


def evaluate_stage1(
    model,
    loader,
    *,
    device,
    max_batches: int,
    use_amp: bool,
    shuffled_plan_banks,
    evaluation_cluster_ids: list[str],
    min_cumulative_yaw: float,
) -> dict[str, object]:
    import torch

    if not shuffled_plan_banks:
        raise ValueError("shuffled_plan_banks must not be empty")
    model.eval()
    metrics = []
    offset = 0
    yaw_correct_by_window: list[bool] = []
    yaw_eligible_by_window: list[bool] = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            tensors = _move_batch(batch, device)
            batch_size = int(tensors["planned_actions"].shape[0])
            negative_plans = [
                bank[offset : offset + batch_size].to(device=device)
                for bank in shuffled_plan_banks
            ]
            if any(plan.shape != tensors["planned_actions"].shape for plan in negative_plans):
                raise ValueError("shuffled plan bank is shorter than the evaluated validation slice")
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = model(
                    tensors["history_latents"],
                    tensors["z_t"],
                    tensors["history_mask"],
                    tensors["planned_actions"],
                    tensors["dt_seconds"],
                )
                null_out = model(
                    tensors["history_latents"],
                    tensors["z_t"],
                    tensors["history_mask"],
                    null_plan(tensors["planned_actions"]),
                    tensors["dt_seconds"],
                )
                shuffled_out = [
                    model(
                        tensors["history_latents"],
                        tensors["z_t"],
                        tensors["history_mask"],
                        negative_plan,
                        tensors["dt_seconds"],
                    )
                    for negative_plan in negative_plans
                ]
                flipped_out = model(
                    tensors["history_latents"],
                    tensors["z_t"],
                    tensors["history_mask"],
                    flipped_plan(tensors["planned_actions"]),
                    tensors["dt_seconds"],
                )
                yaw_flipped_out = model(
                    tensors["history_latents"],
                    tensors["z_t"],
                    tensors["history_mask"],
                    yaw_flipped_plan(tensors["planned_actions"]),
                    tensors["dt_seconds"],
                )
            metrics.append(
                summarize_stage1_batch(
                    predicted_latents=out.predicted_latents,
                    persistence_latents=persistence_rollout(tensors["z_t"], tensors["target_latents"].shape[1]),
                    null_plan_latents=null_out.predicted_latents,
                    shuffled_plan_latents=torch.stack(
                        [negative.predicted_latents for negative in shuffled_out],
                        dim=0,
                    ),
                    flipped_plan_latents=flipped_out.predicted_latents,
                    target_latents=tensors["target_latents"],
                    person_state_target=tensors["person_state_target"],
                    person_state_valid=tensors["person_state_valid"],
                    person_conf=tensors["person_conf"],
                )
            )
            yaw_counts = yaw_geometry_counts(
                out.predicted_person_state,
                yaw_flipped_out.predicted_person_state,
                tensors["planned_actions"],
                tensors["dt_seconds"],
                tensors["planned_actions_valid_mask"],
                min_cumulative_yaw=min_cumulative_yaw,
            )
            yaw_correct_by_window.extend(yaw_counts.correct_by_window)
            yaw_eligible_by_window.extend(yaw_counts.eligible_by_window)
            offset += batch_size
    cluster_ids = evaluation_cluster_ids[:offset]
    aggregate = aggregate_stage1_metrics(
        metrics,
        cluster_ids=cluster_ids,
        yaw_geometry_correct_by_window=yaw_correct_by_window,
        yaw_geometry_eligible_by_window=yaw_eligible_by_window,
    )
    aggregate["evaluated_windows"] = offset
    return aggregate


def prepare_plan_derangements(
    dataset: SportsTrainingDataset,
    indices: list[int],
    *,
    num_permutations: int,
    seed: int,
    min_plan_distance: float,
):
    """Materialize fixed global G1d negative-plan banks and protocol metadata."""
    import torch

    plans = torch.stack(
        [torch.from_numpy(dataset[index].planned_actions) for index in indices],
        dim=0,
    )
    clip_ids = [dataset.sample_clip_ids[index] for index in indices]
    sources = [dataset.sample_sources[index] for index in indices]
    permutations = build_plan_derangements(
        plans,
        clip_ids,
        num_permutations=num_permutations,
        seed=seed,
        min_plan_distance=min_plan_distance,
    )
    banks = [plans[permutation].contiguous() for permutation in permutations]
    all_distances: list[float] = []
    same_source_pairs = 0
    total_pairs = 0
    permutation_hashes: list[str] = []
    for permutation, bank in zip(permutations, banks, strict=True):
        distances = torch.sqrt(
            torch.mean(
                (plans[..., : len(PHYSICAL_PLAN_FIELDS)] - bank[..., : len(PHYSICAL_PLAN_FIELDS)]).square(),
                dim=(1, 2),
            )
        )
        all_distances.extend(float(value) for value in distances)
        donor_indices = permutation.detach().cpu().numpy().astype(np.int64)
        permutation_hashes.append(hashlib.sha256(donor_indices.tobytes()).hexdigest())
        same_source_pairs += sum(
            sources[row] == sources[int(donor)]
            for row, donor in enumerate(donor_indices)
        )
        total_pairs += len(donor_indices)
    return banks, {
        "protocol": G1D_PROTOCOL_VERSION,
        "seed": seed,
        "num_permutations": len(permutations),
        "min_plan_distance_threshold": min_plan_distance,
        "observed_min_plan_distance": min(all_distances),
        "observed_median_plan_distance": float(np.median(all_distances)),
        "same_source_pair_rate": same_source_pairs / max(total_pairs, 1),
        "different_source_pair_rate": 1.0 - same_source_pairs / max(total_pairs, 1),
        "permutation_sha256": permutation_hashes,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate the B3 Stage-1 human world model gate")
    parser.add_argument("--cache-dir", default="ingest_data/latent_cache")
    parser.add_argument("--run-dir", default="reports/b3_stage1_local")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--train-max-samples", type=int, default=128)
    parser.add_argument("--val-max-samples", type=int, default=128)
    parser.add_argument("--val-frac", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-latent", type=float, default=1.0)
    parser.add_argument("--lambda-person-state", type=float, default=0.1)
    parser.add_argument("--lambda-inverse-plan", type=float, default=0.01)
    parser.add_argument("--action-dropout-p", type=float, default=0.2)
    parser.add_argument("--verifier-steps", type=int, default=400)
    parser.add_argument("--verifier-lr", type=float, default=1e-3)
    parser.add_argument("--g1d-negatives", type=int, default=3)
    parser.add_argument("--g1d-min-plan-distance", type=float, default=0.10)
    parser.add_argument("--g1d-min-cumulative-yaw", type=float, default=0.05)
    parser.add_argument("--history", type=int, default=HISTORY)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument(
        "--strict-person-windows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require full person_state_valid history and future supervision",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--overfit-tiny", action="store_true", help="Use the train subset for validation")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import torch

    args = parse_args(argv)
    if not 0.0 <= args.action_dropout_p <= 1.0:
        raise ValueError("--action-dropout-p must be in [0,1]")
    if args.verifier_steps < 1:
        raise ValueError("--verifier-steps must be >= 1")
    if args.verifier_lr <= 0.0:
        raise ValueError("--verifier-lr must be > 0")
    if args.eval_batches < 1:
        raise ValueError("--eval-batches must be >= 1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    dataset = SportsTrainingDataset(
        args.cache_dir,
        history=args.history,
        horizon=args.horizon,
        augment=False,
        strict_person_windows=args.strict_person_windows,
    )
    split = source_split_indices(dataset.sample_sources, val_frac=args.val_frac, seed=args.seed)
    selection_sources = None if args.overfit_tiny else dataset.sample_sources
    selection_strategy = (
        "uniform_random_tiny"
        if args.overfit_tiny
        else "source_balanced_round_robin"
    )
    train_indices, val_indices = select_train_val_indices(
        split,
        sample_sources=selection_sources,
        train_max_samples=args.train_max_samples,
        val_max_samples=args.val_max_samples,
        overfit_tiny=args.overfit_tiny,
        seed=args.seed,
    )

    train_loader = make_loader(
        dataset,
        train_indices,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    verifier_loader = make_loader(
        dataset,
        train_indices,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed + 101,
        num_workers=args.num_workers,
    )
    train_mean_loader = make_loader(
        dataset,
        train_indices,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed + 102,
        num_workers=args.num_workers,
    )
    val_loader = make_loader(
        dataset,
        val_indices,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed + 2,
        num_workers=args.num_workers,
    )

    model = HumanWorldModel(
        dim=EMBED_DIM,
        depth=6,
        heads=12,
        horizon=args.horizon,
    ).to(device)
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    mean_plan = mean_physical_plan(train_mean_loader, device=device)
    verifier_pretraining = pretrain_transition_verifier(
        model,
        verifier_loader,
        device=device,
        max_steps=args.verifier_steps,
        lr=args.verifier_lr,
        use_amp=use_amp,
    )
    verifier_validation = evaluate_transition_verifier(
        model,
        val_loader,
        device=device,
        mean_plan=mean_plan,
        max_batches=args.eval_batches,
        use_amp=use_amp,
    )
    model.transition_plan_verifier.requires_grad_(False)
    model.transition_plan_verifier.eval()

    train_result: TrainingRunResult | None = None
    negative_protocol: dict[str, object] | None = None
    if bool(verifier_validation["passed"]):
        shuffled_plan_banks, negative_protocol = prepare_plan_derangements(
            dataset,
            val_indices,
            num_permutations=args.g1d_negatives,
            seed=args.seed + 10_000,
            min_plan_distance=args.g1d_min_plan_distance,
        )
        # Verifier pretraining has an independent data-loader seed. Reset model
        # stochastic layers before the predictor comparison.
        torch.manual_seed(args.seed + 20_000)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + 20_000)
        train_result = train_steps(
            model,
            train_loader,
            device=device,
            max_steps=args.max_steps,
            lr=args.lr,
            use_amp=use_amp,
            lambda_latent=args.lambda_latent,
            lambda_person_state=args.lambda_person_state,
            lambda_inverse_plan=args.lambda_inverse_plan,
            action_dropout_p=args.action_dropout_p,
            dropout_seed=args.seed + 30_000,
        )
        metrics = evaluate_stage1(
            model,
            val_loader,
            device=device,
            max_batches=args.eval_batches,
            use_amp=use_amp,
            shuffled_plan_banks=shuffled_plan_banks,
            evaluation_cluster_ids=[dataset.sample_sources[index] for index in val_indices],
            min_cumulative_yaw=args.g1d_min_cumulative_yaw,
        )
        metrics["evaluated"] = True
    else:
        metrics = {
            "evaluated": False,
            "reason": "real-transition verifier did not beat train-mean action baseline",
            "g1a_pass": False,
            "g1b_pass": False,
            "g1d_pass": False,
        }
    peak_gib = None
    if device.type == "cuda":
        peak_gib = float(torch.cuda.max_memory_allocated() / 1024**3)
    train_losses = train_result.losses if train_result is not None else []
    initial_loss_mean, final_loss_mean, loss_improvement = loss_window_improvement(train_losses)
    tiny_health_pass = bool(
        args.overfit_tiny
        and verifier_validation["passed"]
        and metrics["g1a_pass"]
        and metrics["g1b_pass"]
        and metrics["g1d_pass"]
    )

    report = {
        "protocol_version": B3_PROTOCOL_VERSION,
        "mode": "overfit_tiny" if args.overfit_tiny else "source_split_gate",
        "cache_dir": args.cache_dir,
        "dataset": {
            "clips": len(dataset._clip_ids),
            "sources": len(set(dataset.sample_sources)),
            "windows": len(dataset),
            "strict_person_windows": dataset.strict_person_windows,
            "train_windows": len(train_indices),
            "val_windows": len(val_indices),
            "train_index_sha256": _index_sha256(train_indices),
            "val_index_sha256": _index_sha256(val_indices),
            "train_sources": represented_sources(dataset.sample_sources, train_indices),
            "val_sources": represented_sources(dataset.sample_sources, val_indices),
            "train_source_window_counts": _per_source_counts(
                dataset.sample_sources,
                train_indices,
            ),
            "val_source_window_counts": _per_source_counts(
                dataset.sample_sources,
                val_indices,
            ),
            "split_train_windows": len(split.train),
            "split_val_windows": len(split.val),
            "split_train_sources": split.train_sources,
            "split_val_sources": split.val_sources,
            "selection_strategy": selection_strategy,
            "selection_seed": args.seed,
        },
        "model": {
            "params": count_parameters(model),
            "depth": 6,
            "dim": EMBED_DIM,
            "history": args.history,
            "horizon": args.horizon,
            "transition_verifier_params": count_parameters(
                model.transition_plan_verifier
            ),
        },
        "training": {
            "batch_size": args.batch_size,
            "max_steps": args.max_steps,
            "lr": args.lr,
            "lambda_latent": args.lambda_latent,
            "lambda_person_state": args.lambda_person_state,
            "lambda_inverse_plan": args.lambda_inverse_plan,
            "action_dropout_p": args.action_dropout_p,
            "action_dropout_scope": "whole_plan_window",
            "amp": use_amp,
            "steps_per_second": (
                train_result.steps_per_second if train_result is not None else None
            ),
            "initial_loss": train_losses[0] if train_losses else None,
            "final_loss": train_losses[-1] if train_losses else None,
            "initial_loss_mean": initial_loss_mean,
            "final_loss_mean": final_loss_mean,
            "loss_window_improvement": loss_improvement,
            "tiny_overfit_loss_improvement": loss_improvement if args.overfit_tiny else None,
            "overfit_eval_same_indices": bool(args.overfit_tiny and val_indices == train_indices),
            "effective_sample_epochs": (
                args.max_steps * args.batch_size / len(train_indices)
                if train_result is not None
                else 0.0
            ),
            "world_model_training_skipped": train_result is None,
            "loss_components": (
                train_result.components if train_result is not None else None
            ),
            "shared_predictor_gradient_interaction": (
                train_result.gradient_interaction
                if train_result is not None
                else None
            ),
            "transition_verifier_pretraining": verifier_pretraining,
            "transition_verifier_validation": verifier_validation,
            "transition_verifier_train_mean_plan": {
                field: float(value)
                for field, value in zip(
                    PHYSICAL_PLAN_FIELDS,
                    mean_plan.detach().cpu(),
                    strict=True,
                )
            },
            "peak_cuda_allocated_gib": peak_gib,
        },
        "g1d_protocol": negative_protocol,
        "tiny_health_pass": tiny_health_pass if args.overfit_tiny else None,
        "stage1": metrics,
    }

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "metrics.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"[b3-stage1] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
