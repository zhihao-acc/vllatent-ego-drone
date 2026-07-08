#!/usr/bin/env python
"""B3 Stage-1 local training/gate harness.

This is intentionally narrow: it trains the depth-6 B3 model on the existing
latent cache and reports G1a/G1b/G1d-style local metrics. It does not download
data, run H20, or operate external systems.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

from vllatent.data.collate import TrainingBatch, collate_sports_batch
from vllatent.data.sports_loader import SportsTrainingDataset
from vllatent.model.human_world_model import HumanWorldModel, count_parameters
from vllatent.schemas import EMBED_DIM, HISTORY
from vllatent.train.world_model_losses import human_world_model_loss
from vllatent.train.world_model_metrics import (
    aggregate_stage1_metrics,
    flipped_plan,
    null_plan,
    persistence_rollout,
    shuffled_plan,
    summarize_stage1_batch,
)


class SplitIndices(NamedTuple):
    train: list[int]
    val: list[int]
    train_sources: list[str]
    val_sources: list[str]


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
) -> tuple[list[float], float]:
    import torch

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    losses: list[float] = []
    start = time.perf_counter()
    model.train()
    step = 0
    while step < max_steps:
        for batch in loader:
            tensors = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = model(
                    tensors["history_latents"],
                    tensors["z_t"],
                    tensors["history_mask"],
                    tensors["planned_actions"],
                    tensors["dt_seconds"],
                )
                loss = human_world_model_loss(
                    predicted_latents=out.predicted_latents,
                    target_latents=tensors["target_latents"],
                    predicted_person_state=out.predicted_person_state,
                    person_state_target=tensors["person_state_target"],
                    person_state_valid=tensors["person_state_valid"],
                    predicted_plan=out.predicted_plan,
                    planned_actions=tensors["planned_actions"],
                    planned_actions_valid_mask=tensors["planned_actions_valid_mask"],
                    person_conf=tensors["person_conf"],
                    sample_weight=tensors["sample_weight"],
                    lambda_latent=lambda_latent,
                    lambda_person_state=lambda_person_state,
                    lambda_inverse_plan=lambda_inverse_plan,
                ).total
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            step += 1
            if step >= max_steps:
                break
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return losses, float(len(losses) / max(elapsed, 1e-8))


def evaluate_stage1(model, loader, *, device, max_batches: int, use_amp: bool) -> dict[str, object]:
    import torch

    model.eval()
    metrics = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            tensors = _move_batch(batch, device)
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
                shuffled_out = model(
                    tensors["history_latents"],
                    tensors["z_t"],
                    tensors["history_mask"],
                    shuffled_plan(tensors["planned_actions"]),
                    tensors["dt_seconds"],
                )
                flipped_out = model(
                    tensors["history_latents"],
                    tensors["z_t"],
                    tensors["history_mask"],
                    flipped_plan(tensors["planned_actions"]),
                    tensors["dt_seconds"],
                )
            metrics.append(
                summarize_stage1_batch(
                    predicted_latents=out.predicted_latents,
                    persistence_latents=persistence_rollout(tensors["z_t"], tensors["target_latents"].shape[1]),
                    null_plan_latents=null_out.predicted_latents,
                    shuffled_plan_latents=shuffled_out.predicted_latents,
                    flipped_plan_latents=flipped_out.predicted_latents,
                    target_latents=tensors["target_latents"],
                    person_state_target=tensors["person_state_target"],
                    person_state_valid=tensors["person_state_valid"],
                    person_conf=tensors["person_conf"],
                )
            )
    return aggregate_stage1_metrics(metrics)


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
    parser.add_argument("--history", type=int, default=HISTORY)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--overfit-tiny", action="store_true", help="Use the train subset for validation")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import torch

    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    dataset = SportsTrainingDataset(args.cache_dir, history=args.history, horizon=args.horizon, augment=False)
    split = source_split_indices(dataset.sample_sources, val_frac=args.val_frac, seed=args.seed)
    train_indices = limit_indices(split.train, args.train_max_samples, seed=args.seed)
    val_base = split.train if args.overfit_tiny else split.val
    val_indices = limit_indices(val_base, args.val_max_samples, seed=args.seed + 1)

    train_loader = make_loader(
        dataset,
        train_indices,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
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

    model = HumanWorldModel(dim=EMBED_DIM, depth=6, heads=12, horizon=args.horizon).to(device)
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    train_losses, steps_per_second = train_steps(
        model,
        train_loader,
        device=device,
        max_steps=args.max_steps,
        lr=args.lr,
        use_amp=use_amp,
        lambda_latent=args.lambda_latent,
        lambda_person_state=args.lambda_person_state,
        lambda_inverse_plan=args.lambda_inverse_plan,
    )
    metrics = evaluate_stage1(model, val_loader, device=device, max_batches=args.eval_batches, use_amp=use_amp)
    peak_gib = None
    if device.type == "cuda":
        peak_gib = float(torch.cuda.max_memory_allocated() / 1024**3)

    report = {
        "mode": "overfit_tiny" if args.overfit_tiny else "source_split_gate",
        "cache_dir": args.cache_dir,
        "dataset": {
            "clips": len(dataset._clip_ids),
            "sources": len(set(dataset.sample_sources)),
            "windows": len(dataset),
            "train_windows": len(train_indices),
            "val_windows": len(val_indices),
            "train_sources": split.train_sources,
            "val_sources": split.val_sources,
        },
        "model": {
            "params": count_parameters(model),
            "depth": 6,
            "dim": EMBED_DIM,
            "history": args.history,
            "horizon": args.horizon,
        },
        "training": {
            "batch_size": args.batch_size,
            "max_steps": args.max_steps,
            "lr": args.lr,
            "lambda_latent": args.lambda_latent,
            "lambda_person_state": args.lambda_person_state,
            "lambda_inverse_plan": args.lambda_inverse_plan,
            "amp": use_amp,
            "steps_per_second": steps_per_second,
            "initial_loss": train_losses[0] if train_losses else None,
            "final_loss": train_losses[-1] if train_losses else None,
            "tiny_overfit_loss_improvement": (
                (train_losses[0] - train_losses[-1]) / max(train_losses[0], 1e-8)
                if len(train_losses) >= 2
                else None
            ),
            "peak_cuda_allocated_gib": peak_gib,
        },
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
