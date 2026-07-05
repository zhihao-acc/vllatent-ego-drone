#!/usr/bin/env python
"""Train the B2 direct scale-free action policy.

This script is separate from ``train_sports.py`` so the B1 latent-world trainer
remains reproducible.  B2 trains only ``ScaleFreeActionPolicy`` on cached DINO
latents and scale-free future-action labels.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllatent.config import Config, PredictorConfig
from vllatent.data.collate import ActionPolicyBatch, collate_action_policy_batch
from vllatent.data.sports_loader import SportsTrainingDataset, clip_source, split_clips_by_source
from vllatent.model.action_policy import ScaleFreeActionPolicy
from vllatent.schemas import EMBED_DIM
from vllatent.train.action_metrics import ActionScorecard, score_action_predictions
from vllatent.train.checkpoint import save_checkpoint, seed_everything, snapshot_config
from vllatent.train.losses import action_policy_loss


@dataclass(frozen=True)
class ActionTrainConfig:
    """B2 action-policy training knobs."""

    lr: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    epochs: int = 10
    hidden_dim: int = 256
    depth: int = 2
    heads: int = 4
    dropout: float = 0.1
    val_frac: float = 0.2
    eval_every_epochs: int = 1
    early_stop_patience: int = 5
    grad_clip: float = 1.0
    amp_dtype: str = "fp32"
    seed: int = 42
    num_workers: int = 0
    direction_weight: float = 1.0
    speed_weight: float = 1.0
    path_weight: float = 1.0


def _write_jsonl(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _autocast_ctx(device: str, amp_dtype: str):  # noqa: ANN202
    if device == "cuda" and amp_dtype != "fp32":
        dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
        return torch.autocast("cuda", dtype=dtype)
    return nullcontext()


def _to_device(batch: ActionPolicyBatch, device: str) -> ActionPolicyBatch:
    return batch._replace(
        z_t=batch.z_t.to(device),
        history_latents=batch.history_latents.to(device),
        history_mask=batch.history_mask.to(device),
        target_actions_scale_free=batch.target_actions_scale_free.to(device),
        target_actions_moving_mask=batch.target_actions_moving_mask.to(device),
        target_actions_speed_mask=batch.target_actions_speed_mask.to(device),
        last_action_scale_free=batch.last_action_scale_free.to(device),
        action_history_scale_free=batch.action_history_scale_free.to(device),
        action_history_mask=batch.action_history_mask.to(device),
        camera_history_path_scale_free=batch.camera_history_path_scale_free.to(device),
        dt_seconds=batch.dt_seconds.to(device),
        odom_reference_speed=batch.odom_reference_speed.to(device),
        vo_confidence=batch.vo_confidence.to(device),
        frame_quality=batch.frame_quality.to(device),
        sample_weight=batch.sample_weight.to(device),
    )


def _make_loader(dataset: SportsTrainingDataset, cfg: ActionTrainConfig, *, shuffle: bool) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        collate_fn=collate_action_policy_batch,
        drop_last=False,
        num_workers=cfg.num_workers,
    )


def _make_model(cfg: ActionTrainConfig) -> ScaleFreeActionPolicy:
    return ScaleFreeActionPolicy(
        dim=EMBED_DIM,
        hidden_dim=cfg.hidden_dim,
        depth=cfg.depth,
        heads=cfg.heads,
        dropout=cfg.dropout,
    )


def _flatten_scorecard(scorecard: ActionScorecard) -> dict[str, Any]:
    out = {
        "action_score": scorecard.model.aggregate_score,
        "direction_cosine": scorecard.model.direction_cosine,
        "angular_error_deg": scorecard.model.angular_error_deg,
        "path_ade": scorecard.model.path_ade,
        "path_fde": scorecard.model.path_fde,
        "speed_ratio_mae": scorecard.model.speed_ratio_mae,
        "best_baseline": scorecard.best_baseline,
        "best_baseline_score": scorecard.best_baseline_score,
        "action_margin": scorecard.margin,
        "action_margin_frac": scorecard.margin / max(scorecard.best_baseline_score, 1e-6),
        "n_samples": scorecard.model.n_samples,
        "n_valid": scorecard.model.n_valid,
        "n_speed_valid": scorecard.model.n_speed_valid,
    }
    for name, metrics in scorecard.baselines.items():
        out[f"baseline_{name}_score"] = metrics.aggregate_score
    return out


def evaluate_action_policy(
    model: ScaleFreeActionPolicy,
    loader: Any,
    device: str,
    *,
    amp_dtype: str = "fp32",
) -> dict[str, Any]:
    """Evaluate a B2 action policy over a loader and return flattened action metrics."""
    preds = []
    targets = []
    masks = []
    speed_masks = []
    last_actions = []
    weights = []

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                batch = _to_device(batch, device)
                with _autocast_ctx(device, amp_dtype):
                    pred = model(
                        history_latents=batch.history_latents,
                        z_t=batch.z_t,
                        history_mask=batch.history_mask,
                        last_action_scale_free=batch.last_action_scale_free,
                        dt_seconds=batch.dt_seconds,
                        action_history_scale_free=batch.action_history_scale_free,
                        action_history_mask=batch.action_history_mask,
                        camera_history_path_scale_free=batch.camera_history_path_scale_free,
                    )
                preds.append(pred.float().cpu())
                targets.append(batch.target_actions_scale_free.float().cpu())
                masks.append(batch.target_actions_moving_mask.bool().cpu())
                speed_masks.append(batch.target_actions_speed_mask.bool().cpu())
                last_actions.append(batch.last_action_scale_free.float().cpu())
                weights.append(batch.sample_weight.float().cpu())
    finally:
        if was_training:
            model.train()

    if not preds:
        raise ValueError("evaluate_action_policy(): loader yielded no batches")

    scorecard = score_action_predictions(
        predicted=torch.cat(preds, dim=0),
        target=torch.cat(targets, dim=0),
        moving_mask=torch.cat(masks, dim=0),
        last_action_scale_free=torch.cat(last_actions, dim=0),
        sample_weight=torch.cat(weights, dim=0),
        speed_mask=torch.cat(speed_masks, dim=0),
    )
    return _flatten_scorecard(scorecard)


def _train_batch(
    model: ScaleFreeActionPolicy,
    batch: ActionPolicyBatch,
    optimizer: torch.optim.Optimizer,
    cfg: ActionTrainConfig,
    device: str,
) -> float:
    batch = _to_device(batch, device)
    optimizer.zero_grad()
    pred = model(
        history_latents=batch.history_latents,
        z_t=batch.z_t,
        history_mask=batch.history_mask,
        last_action_scale_free=batch.last_action_scale_free,
        dt_seconds=batch.dt_seconds,
        action_history_scale_free=batch.action_history_scale_free,
        action_history_mask=batch.action_history_mask,
        camera_history_path_scale_free=batch.camera_history_path_scale_free,
    )
    loss = action_policy_loss(
        pred.float(),
        batch.target_actions_scale_free.float(),
        batch.target_actions_moving_mask,
        speed_mask=batch.target_actions_speed_mask,
        sample_weight=batch.sample_weight,
        direction_weight=cfg.direction_weight,
        speed_weight=cfg.speed_weight,
        path_weight=cfg.path_weight,
    )
    if not torch.isfinite(loss.detach()).all():
        raise FloatingPointError(f"non-finite action loss: {loss.detach().item()}")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach().cpu())


def _write_config(run_dir: Path, cfg: ActionTrainConfig) -> Config:
    run_dir.mkdir(parents=True, exist_ok=True)
    config = Config(predictor=PredictorConfig(depth=cfg.depth, heads=cfg.heads, dropout=cfg.dropout))
    snapshot_config(config, run_dir)
    (run_dir / "train_b2_config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
    return config


def train_overfit_tiny(args: argparse.Namespace, cfg: ActionTrainConfig) -> dict[str, Any]:
    """Overfit a tiny subset and write train-action metrics plus final checkpoint."""
    dataset = SportsTrainingDataset(cache_dir=args.cache_dir, augment=False)
    n_samples = min(args.overfit_samples, len(dataset))
    subset = torch.utils.data.Subset(dataset, list(range(n_samples)))
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=min(cfg.batch_size, n_samples),
        shuffle=True,
        collate_fn=collate_action_policy_batch,
        drop_last=False,
        num_workers=cfg.num_workers,
    )
    eval_loader = torch.utils.data.DataLoader(
        subset,
        batch_size=min(cfg.batch_size, n_samples),
        shuffle=False,
        collate_fn=collate_action_policy_batch,
        drop_last=False,
        num_workers=cfg.num_workers,
    )

    run_dir = Path(args.run_dir)
    config = _write_config(run_dir, cfg)
    model = _make_model(cfg).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_margin = float("-inf")
    best_metrics: dict[str, Any] = {}
    step = 0
    while step < args.max_steps:
        for batch in loader:
            if step >= args.max_steps:
                break
            loss = _train_batch(model, batch, optimizer, cfg, args.device)
            if step % args.log_every == 0 or step == args.max_steps - 1:
                metrics = evaluate_action_policy(model, eval_loader, args.device, amp_dtype=cfg.amp_dtype)
                metrics.update({"step": step, "loss": loss})
                _write_jsonl(run_dir / "train_action_metrics.jsonl", metrics)
                if metrics["action_margin"] > best_margin:
                    best_margin = metrics["action_margin"]
                    best_metrics = metrics
                    save_checkpoint(
                        model,
                        optimizer,
                        0,
                        step,
                        config,
                        {"action_margin": best_margin, "loss": loss},
                        run_dir / "ckpt_best.pt",
                        val_metrics=metrics,
                    )
            step += 1

    save_checkpoint(
        model,
        optimizer,
        0,
        max(0, step - 1),
        config,
        {"action_margin": best_margin},
        run_dir / "ckpt_final.pt",
        val_metrics=best_metrics,
    )
    return best_metrics


def _source_groups(stems: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for stem in stems:
        groups.setdefault(clip_source(stem), []).append(stem)
    return {src: sorted(ids) for src, ids in sorted(groups.items())}


def _select_clip_stems(cache_dir: str | Path, *, max_clips: int = 0, max_clips_per_source: int = 0) -> list[str]:
    stems = sorted(p.stem for p in Path(cache_dir).glob("*.npz"))
    if max_clips_per_source:
        capped: list[str] = []
        for ids in _source_groups(stems).values():
            capped.extend(ids[:max_clips_per_source])
        stems = sorted(capped)
    if max_clips:
        stems = stems[:max_clips]
    return stems


def train_full(args: argparse.Namespace, cfg: ActionTrainConfig) -> dict[str, Any]:
    """Train with a source split and checkpoint best by action margin."""
    stems = _select_clip_stems(
        args.cache_dir,
        max_clips=args.max_clips,
        max_clips_per_source=args.max_clips_per_source,
    )
    if not stems:
        raise ValueError(f"No .npz clips in {args.cache_dir}")
    train_stems, val_stems = split_clips_by_source(stems, cfg.val_frac, seed=cfg.seed)
    if not val_stems:
        raise ValueError("B2 full training requires at least one held-out source")

    train_ds = SportsTrainingDataset(cache_dir=args.cache_dir, clip_ids=train_stems, augment=True)
    val_ds = SportsTrainingDataset(cache_dir=args.cache_dir, clip_ids=val_stems, augment=False, norm_stats=train_ds.norm_stats)
    train_loader = _make_loader(train_ds, cfg, shuffle=True)
    val_loader = _make_loader(val_ds, cfg, shuffle=False)

    run_dir = Path(args.run_dir)
    config = _write_config(run_dir, cfg)
    train_ds.save_norm_stats(run_dir / "norm_stats.npz")
    model = _make_model(cfg).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_margin = float("-inf")
    best_metrics: dict[str, Any] = {}
    patience_left = cfg.early_stop_patience
    global_step = 0
    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        for batch in train_loader:
            loss = _train_batch(model, batch, optimizer, cfg, args.device)
            global_step += 1
        if (epoch + 1) % cfg.eval_every_epochs == 0:
            metrics = evaluate_action_policy(model, val_loader, args.device, amp_dtype=cfg.amp_dtype)
            metrics.update({"epoch": epoch, "step": global_step, "last_loss": loss})
            _write_jsonl(run_dir / "val_action_metrics.jsonl", metrics)
            if args.eval_by_source:
                for source, ids in _source_groups(val_stems).items():
                    src_ds = SportsTrainingDataset(
                        cache_dir=args.cache_dir,
                        clip_ids=ids,
                        augment=False,
                        norm_stats=train_ds.norm_stats,
                    )
                    src_metrics = evaluate_action_policy(
                        model,
                        _make_loader(src_ds, cfg, shuffle=False),
                        args.device,
                        amp_dtype=cfg.amp_dtype,
                    )
                    _write_jsonl(
                        run_dir / "source_action_metrics.jsonl",
                        {"epoch": epoch, "step": global_step, "source": source, **src_metrics},
                    )
            if metrics["action_margin"] > best_margin:
                best_margin = metrics["action_margin"]
                best_metrics = metrics
                patience_left = cfg.early_stop_patience
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    global_step,
                    config,
                    {"action_margin": best_margin, "loss": float(loss)},
                    run_dir / "ckpt_best.pt",
                    val_metrics=metrics,
                )
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

    (run_dir / "train_summary.json").write_text(json.dumps({
        "steps": global_step,
        "seconds": time.time() - t0,
        "best_action_margin": best_margin,
        "best_metrics": best_metrics,
    }, indent=2))
    return best_metrics


def train(args: argparse.Namespace) -> dict[str, Any]:
    cfg = ActionTrainConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        heads=args.heads,
        dropout=args.dropout,
        val_frac=args.val_frac,
        eval_every_epochs=args.eval_every_epochs,
        early_stop_patience=args.early_stop_patience,
        grad_clip=args.grad_clip,
        amp_dtype=args.amp_dtype,
        seed=args.seed,
        num_workers=args.num_workers,
        direction_weight=args.direction_weight,
        speed_weight=args.speed_weight,
        path_weight=args.path_weight,
    )
    seed_everything(cfg.seed)
    if args.overfit_tiny:
        return train_overfit_tiny(args, cfg)
    return train_full(args, cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train B2 direct scale-free action policy")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--run-dir", default="runs/b2_action")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp-dtype", default="fp32", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--eval-every-epochs", type=int, default=1)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--eval-by-source", action="store_true")
    parser.add_argument("--max-clips", type=int, default=0, help="Optional local smoke limit")
    parser.add_argument("--max-clips-per-source", type=int, default=0, help="Optional source-balanced smoke limit")
    parser.add_argument("--direction-weight", type=float, default=1.0)
    parser.add_argument("--speed-weight", type=float, default=1.0)
    parser.add_argument("--path-weight", type=float, default=1.0)
    parser.add_argument("--overfit-tiny", action="store_true")
    parser.add_argument("--overfit-samples", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    metrics = train(args)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
