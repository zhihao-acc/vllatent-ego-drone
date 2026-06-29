#!/usr/bin/env python
"""Training script for sports-following model (B1.20).

Usage:
    # Overfit-tiny-batch (first-ever training test)
    python scripts/train_sports.py --overfit-tiny --cache-dir ingest_data/latent_cache \
        --run-dir runs/overfit_tiny --device cuda

    # Resume from checkpoint
    python scripts/train_sports.py --overfit-tiny --cache-dir ingest_data/latent_cache \
        --run-dir runs/overfit_tiny --resume runs/overfit_tiny/ckpt_step200.pt --device cuda
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllatent.config import Config, PredictorConfig
from vllatent.data.collate import collate_sports_batch
from vllatent.data.sports_loader import SportsTrainingDataset
from vllatent.model.sports_model import SportsFollowingModel
from vllatent.schemas import EMBED_DIM
from vllatent.train.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    snapshot_config,
)
from vllatent.train.losses import combined_loss, latent_loss, waypoint_loss
from vllatent.train.sanity import run_sanity_check
from vllatent.train.viz import TrainingLogger


def compute_baseline_loss(
    dataset: SportsTrainingDataset,
    n_samples: int = 16,
    device: str = "cpu",
) -> tuple[float, float]:
    """Dumb baseline: predict zeros for both latents and deltas."""
    from torch.utils.data import DataLoader, Subset

    indices = list(range(min(n_samples, len(dataset))))
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=len(indices), collate_fn=collate_sports_batch)
    batch = next(iter(loader))

    z = torch.zeros_like(batch.target_latents).to(device)
    tgt_lat = batch.target_latents.to(device)
    fq = batch.frame_quality.to(device)
    w_quality = fq.clamp(min=0.1)
    l_lat = latent_loss(z, tgt_lat, w_quality, beta=0.1).item()

    z_wp = torch.zeros_like(batch.target_deltas).to(device)
    tgt_wp = batch.target_deltas.to(device)
    vo = batch.vo_confidence.to(device)
    w_vo = vo.mean(dim=1).clamp(min=0.05)
    l_wp = waypoint_loss(z_wp, tgt_wp, w_vo).item()

    return l_lat, l_wp


def train(args: argparse.Namespace) -> None:
    device = args.device
    seed_everything(args.seed)

    pred_cfg = PredictorConfig(
        depth=args.depth,
        heads=args.heads,
        dropout=args.dropout,
    )
    config = Config(predictor=pred_cfg)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_config(config, run_dir)

    print(f"[train] Loading dataset from {args.cache_dir} ...")
    dataset = SportsTrainingDataset(
        cache_dir=args.cache_dir,
        augment=not args.overfit_tiny,
    )
    print(f"[train] Dataset: {len(dataset)} samples")

    print("[train] Running sanity check ...")
    run_sanity_check(dataset, n_samples=min(5, len(dataset)))
    print("[train] Sanity check PASSED")

    dataset.save_norm_stats(run_dir / "norm_stats.npz")
    print(f"[train] Saved norm_stats to {run_dir / 'norm_stats.npz'}")

    model = SportsFollowingModel(pred_cfg, dim=EMBED_DIM).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] Model: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_steps, eta_min=args.lr * 0.01
    )

    start_step = 0
    start_epoch = 0
    if args.resume:
        print(f"[train] Resuming from {args.resume} ...")
        ckpt = load_checkpoint(args.resume, model, optimizer, map_location=device)
        start_step = ckpt["global_step"] + 1
        start_epoch = ckpt["epoch"]
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        else:
            scheduler.last_epoch = start_step
        print(f"[train] Resumed at step {start_step}, epoch {start_epoch}")

    if args.overfit_tiny:
        n_samples = min(args.overfit_samples, len(dataset))
        from torch.utils.data import Subset
        subset = Subset(dataset, list(range(n_samples)))
        loader = torch.utils.data.DataLoader(
            subset,
            batch_size=min(args.batch_size, n_samples),
            shuffle=True,
            collate_fn=collate_sports_batch,
            drop_last=False,
        )
        print(f"[train] Overfit-tiny mode: {n_samples} samples, {args.max_steps} steps")
    else:
        if len(dataset) < args.batch_size:
            raise ValueError(
                f"Dataset ({len(dataset)} samples) smaller than batch_size ({args.batch_size}). "
                f"Use --overfit-tiny or reduce --batch-size."
            )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_sports_batch,
            drop_last=True,
        )

    baseline_lat, baseline_wp = compute_baseline_loss(dataset, device=device)
    print(f"[train] Baseline (zeros): L_latent={baseline_lat:.4f}, L_wp={baseline_wp:.4f}")

    logger = TrainingLogger(log_dir=run_dir, log_every=args.log_every)

    model.train()
    step = start_step
    epoch = start_epoch
    loss_out = None
    t0 = time.time()

    while step < args.max_steps:
        for batch in loader:
            if step >= args.max_steps:
                break

            batch = batch._replace(
                z_t=batch.z_t.to(device),
                history_latents=batch.history_latents.to(device),
                history_mask=batch.history_mask.to(device),
                target_latents=batch.target_latents.to(device),
                target_deltas=batch.target_deltas.to(device),
                last_action=batch.last_action.to(device),
                vo_confidence=batch.vo_confidence.to(device),
                frame_quality=batch.frame_quality.to(device),
                dt_seconds=batch.dt_seconds.to(device),
                sample_weight=batch.sample_weight.to(device),
            )

            out = model(batch)
            loss_out = combined_loss(
                out.predicted_latents, batch.target_latents,
                out.predicted_deltas, batch.target_deltas,
                batch.frame_quality, batch.vo_confidence,
                lambda_latent=args.lambda_latent,
                lambda_waypoint=args.lambda_waypoint,
                beta=0.1,
            )

            optimizer.zero_grad()
            loss_out.total.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            if logger.should_log(step):
                lr = scheduler.get_last_lr()[0]
                logger.log_step(
                    step=step, epoch=epoch, loss_output=loss_out, lr=lr,
                    predicted_latents=out.predicted_latents.detach(),
                    target_latents=batch.target_latents,
                    predicted_deltas=out.predicted_deltas.detach(),
                    target_deltas=batch.target_deltas,
                )
                elapsed = time.time() - t0
                steps_per_sec = (step - start_step + 1) / max(elapsed, 1e-6)
                beat_lat = "YES" if loss_out.latent.item() < baseline_lat else "no"
                beat_wp = "YES" if loss_out.waypoint.item() < baseline_wp else "no"
                print(
                    f"  step {step:4d} | "
                    f"L_total={loss_out.total.item():.4f} "
                    f"L_lat={loss_out.latent.item():.4f} ({beat_lat}) "
                    f"L_wp={loss_out.waypoint.item():.4f} ({beat_wp}) "
                    f"cos={loss_out.cosine_sim.item():.3f} "
                    f"lr={lr:.2e} "
                    f"{steps_per_sec:.1f} steps/s"
                )

            if step > 0 and step % args.save_every == 0:
                ckpt_path = run_dir / f"ckpt_step{step}.pt"
                save_checkpoint(
                    model, optimizer, epoch, step, config,
                    {"loss": loss_out.total.item()}, ckpt_path,
                    scheduler=scheduler,
                )
                print(f"[train] Saved checkpoint: {ckpt_path}")

            step += 1

        if step >= args.max_steps:
            break
        epoch += 1

    if loss_out is not None:
        final_path = run_dir / "ckpt_final.pt"
        save_checkpoint(
            model, optimizer, epoch, step - 1, config,
            {"loss": loss_out.total.item()}, final_path,
            scheduler=scheduler,
        )
        print(f"[train] Final checkpoint: {final_path}")
    print(f"[train] Done. {step} steps in {time.time() - t0:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sports-following model")
    parser.add_argument("--cache-dir", required=True, help="Path to .npz cache directory")
    parser.add_argument("--run-dir", default="runs/default", help="Output directory for checkpoints and logs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--overfit-tiny", action="store_true", help="Overfit on small batch")
    parser.add_argument("--overfit-samples", type=int, default=16, help="Number of samples for overfit-tiny")
    parser.add_argument("--max-steps", type=int, default=500, help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-latent", type=float, default=1.0)
    parser.add_argument("--lambda-waypoint", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)

    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
