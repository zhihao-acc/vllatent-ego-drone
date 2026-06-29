#!/usr/bin/env python
"""Training script for the sports-following model (B1.20 + B1.22a).

Two modes:

    # B-1 latent world-model run (predictor only, scene-split val, early-stop)
    python scripts/train_sports.py --cache-dir ingest_data/latent_cache \
        --run-dir runs/b1_latent --latent-only --amp-dtype bf16 --depth 6 \
        --batch-size 64 --lr 2e-4 --warmup-frac 0.05 --weight-decay 0.05 \
        --val-frac 0.2 --eval-every-epochs 1 --early-stop-patience 8 \
        --early-stop-metric val_cos --device cuda

    # Overfit-tiny smoke (first-ever training test — must beat zeros baseline <200 steps)
    python scripts/train_sports.py --overfit-tiny --cache-dir ingest_data/latent_cache \
        --run-dir runs/overfit_tiny --device cuda --max-steps 500

B-1 trains the LATENT PREDICTOR only (``--latent-only``); the waypoint head + L_wp +
staged freeze/joint control are Phase B-2 and are NOT in this script.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllatent.config import Config, PredictorConfig, TrainConfig
from vllatent.data.collate import collate_sports_batch
from vllatent.data.sports_loader import SportsTrainingDataset, split_clips_by_source
from vllatent.model.sports_model import SportsFollowingModel
from vllatent.schemas import EMBED_DIM
from vllatent.train.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    snapshot_config,
)
from vllatent.train.evaluate import evaluate
from vllatent.train.losses import (
    LossOutput,
    combined_loss,
    cosine_similarity_diagnostic,
    latent_loss,
    waypoint_loss,
)
from vllatent.train.optim import build_param_groups
from vllatent.train.sanity import run_sanity_check
from vllatent.train.viz import TrainingLogger


def _resolve_amp(device: str, amp_dtype: str) -> tuple[bool, torch.dtype, bool]:
    """(autocast_enabled, autocast_dtype, use_grad_scaler) for the requested precision.

    bf16 needs NO GradScaler (wide exponent); fp16 does; fp32/CPU disables autocast.
    """
    if device != "cuda" or amp_dtype == "fp32":
        return False, torch.float32, False
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return True, dtype, amp_dtype == "fp16"


def _worker_init_fn(worker_id: int) -> None:
    """Re-seed each DataLoader worker's numpy RNG so augmentation streams differ."""
    import numpy as np

    info = torch.utils.data.get_worker_info()
    if info is not None and hasattr(info.dataset, "_rng"):
        info.dataset._rng = np.random.default_rng(torch.initial_seed() % (2**32))


def _build_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_frac: float, base_lr: float
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warmup → cosine annealing (SequentialLR); cosine-only if warmup is empty."""
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    warmup_steps = int(warmup_frac * total_steps)
    eta_min = base_lr * 0.01
    if 1 <= warmup_steps < total_steps:
        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=eta_min)
        return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])
    return CosineAnnealingLR(optimizer, T_max=max(1, total_steps), eta_min=eta_min)


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
    w_quality = batch.frame_quality.to(device).clamp(min=0.1)
    l_lat = latent_loss(z, tgt_lat, w_quality, beta=0.1).item()

    z_wp = torch.zeros_like(batch.target_deltas).to(device)
    tgt_wp = batch.target_deltas.to(device)
    w_vo = batch.vo_confidence.to(device).mean(dim=1).clamp(min=0.05)
    l_wp = waypoint_loss(z_wp, tgt_wp, w_vo).item()

    return l_lat, l_wp


def _to_device(batch, device: str):  # noqa: ANN001 — TrainingBatch
    return batch._replace(
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


def _forward_loss(
    model: SportsFollowingModel, batch, tcfg: TrainConfig, device: str,  # noqa: ANN001
    lambda_latent: float = 1.0, lambda_waypoint: float = 1.0,
):
    """Forward + loss. ``latent-only`` runs the predictor alone on L_latent."""
    if tcfg.latent_only:
        predicted = model.predictor(
            history_latents=batch.history_latents,
            z_t=batch.z_t,
            action_4dof=batch.last_action,
            dt_seconds=batch.dt_seconds,
            history_mask=batch.history_mask,
        )
        # Loss in fp32 (cast all loss inputs out of autocast); domain weighting is via the sampler.
        w_quality = batch.frame_quality.float().clamp(min=0.1)
        l_lat = latent_loss(predicted.float(), batch.target_latents.float(), w_quality, beta=0.1)
        with torch.no_grad():
            cos = cosine_similarity_diagnostic(predicted.float(), batch.target_latents.float())
        zero = torch.zeros((), device=device)
        return LossOutput(total=l_lat, latent=l_lat, waypoint=zero, cosine_sim=cos), predicted, None

    out = model(batch)
    loss_out = combined_loss(
        out.predicted_latents.float(), batch.target_latents.float(),
        out.predicted_deltas.float(), batch.target_deltas.float(),
        batch.frame_quality, batch.vo_confidence,
        lambda_latent=lambda_latent, lambda_waypoint=lambda_waypoint, beta=0.1,
    )
    return loss_out, out.predicted_latents, out.predicted_deltas


def _make_train_loader(dataset: SportsTrainingDataset, tcfg: TrainConfig):  # noqa: ANN201
    """DataLoader with optional WeightedRandomSampler down-weighting domain=game samples."""
    from torch.utils.data import DataLoader, WeightedRandomSampler

    domains = dataset.sample_domains
    has_game = any(d == "game" for d in domains)
    common = dict(
        batch_size=tcfg.batch_size,
        collate_fn=collate_sports_batch,
        drop_last=True,
        num_workers=tcfg.num_workers,
        worker_init_fn=_worker_init_fn,
    )
    if tcfg.domain_weight != 1.0 and has_game:
        weights = [tcfg.domain_weight if d == "game" else 1.0 for d in domains]
        sampler = WeightedRandomSampler(weights, num_samples=len(domains), replacement=True)
        return DataLoader(dataset, sampler=sampler, **common)
    return DataLoader(dataset, shuffle=True, **common)


def train(args: argparse.Namespace) -> None:
    device = args.device
    tcfg = TrainConfig(
        latent_only=args.latent_only,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_frac=args.warmup_frac,
        batch_size=args.batch_size,
        epochs=args.epochs,
        amp_dtype=args.amp_dtype,
        val_frac=args.val_frac,
        eval_every_epochs=args.eval_every_epochs,
        early_stop_patience=args.early_stop_patience,
        early_stop_metric=args.early_stop_metric,
        domain_weight=args.domain_weight,
        use_action_film=not args.no_action_film,
        grad_clip=args.grad_clip,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    seed_everything(tcfg.seed)
    pred_cfg = PredictorConfig(depth=args.depth, heads=args.heads, dropout=args.dropout)
    config = Config(predictor=pred_cfg)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_config(config, run_dir)
    (run_dir / "train_config.json").write_text(json.dumps(dataclasses.asdict(tcfg), indent=2))

    use_amp, amp_dtype, use_scaler = _resolve_amp(device, tcfg.amp_dtype)
    autocast = (
        torch.autocast("cuda", dtype=amp_dtype) if use_amp else _nullcontext()
    )
    model = SportsFollowingModel(pred_cfg, dim=EMBED_DIM, use_action_film=tcfg.use_action_film).to(device)
    opt_target = model.predictor if tcfg.latent_only else model
    n_params = sum(p.numel() for p in opt_target.parameters() if p.requires_grad)
    print(f"[train] Model: {n_params:,} optimized params (latent_only={tcfg.latent_only}, "
          f"action_film={tcfg.use_action_film}, amp={tcfg.amp_dtype})")
    optimizer = torch.optim.AdamW(build_param_groups(opt_target, tcfg.weight_decay), lr=tcfg.lr)

    if args.overfit_tiny:
        _train_overfit(args, tcfg, model, optimizer, opt_target, config, run_dir,
                       device, use_scaler, autocast)
    else:
        _train_full(args, tcfg, model, optimizer, opt_target, config, run_dir,
                    device, use_scaler, autocast)


def _train_overfit(args, tcfg, model, optimizer, opt_target, config, run_dir,  # noqa: ANN001
                   device, use_scaler, autocast) -> None:
    """Step-based overfit-tiny smoke: must beat the zeros baseline within ~200 steps."""
    from torch.utils.data import DataLoader, Subset

    dataset = SportsTrainingDataset(cache_dir=args.cache_dir, augment=False)
    print(f"[train] Dataset: {len(dataset)} samples")
    run_sanity_check(dataset, n_samples=min(5, len(dataset)))
    dataset.save_norm_stats(run_dir / "norm_stats.npz")

    n_samples = min(args.overfit_samples, len(dataset))
    subset = Subset(dataset, list(range(n_samples)))
    loader = DataLoader(subset, batch_size=min(tcfg.batch_size, n_samples), shuffle=True,
                        collate_fn=collate_sports_batch, drop_last=False)
    print(f"[train] Overfit-tiny: {n_samples} samples, {args.max_steps} steps")

    scheduler = _build_scheduler(optimizer, args.max_steps, tcfg.warmup_frac, tcfg.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    logger = TrainingLogger(log_dir=run_dir, log_every=args.log_every)
    start_step, _ = _maybe_resume(args, model, optimizer, scheduler, device)

    baseline_lat, baseline_wp = compute_baseline_loss(dataset, device=device)
    print(f"[train] Baseline (zeros): L_latent={baseline_lat:.4f}, L_wp={baseline_wp:.4f}")

    model.train()
    step, t0, loss_out = start_step, time.time(), None
    while step < args.max_steps:
        for batch in loader:
            if step >= args.max_steps:
                break
            batch = _to_device(batch, device)
            with autocast:
                loss_out, pred_lat, pred_delta = _forward_loss(
                    model, batch, tcfg, device, args.lambda_latent, args.lambda_waypoint)
            _step_optimizer(loss_out.total, optimizer, opt_target, scaler, use_scaler, tcfg.grad_clip)
            scheduler.step()
            if logger.should_log(step):
                lr = scheduler.get_last_lr()[0]
                logger.log_step(step=step, epoch=0, loss_output=loss_out, lr=lr,
                                predicted_latents=pred_lat.detach(), target_latents=batch.target_latents,
                                predicted_deltas=None if pred_delta is None else pred_delta.detach(),
                                target_deltas=None if pred_delta is None else batch.target_deltas)
                beat = "YES" if loss_out.latent.item() < baseline_lat else "no"
                print(f"  step {step:4d} | L_lat={loss_out.latent.item():.4f} ({beat}) "
                      f"cos={loss_out.cosine_sim.item():.3f} lr={lr:.2e}")
            step += 1
    if loss_out is not None:
        save_checkpoint(model, optimizer, 0, step - 1, config,
                        {"loss": loss_out.total.item()}, run_dir / "ckpt_final.pt", scheduler=scheduler)
    print(f"[train] Done. {step} steps in {time.time() - t0:.1f}s")


def _train_full(args, tcfg, model, optimizer, opt_target, config, run_dir,  # noqa: ANN001
                device, use_scaler, autocast) -> None:
    """Epoch-based run with scene-split val, eval-every-N-epochs, best-ckpt + early-stop."""
    stems = sorted(p.stem for p in Path(args.cache_dir).glob("*.npz"))
    if not stems:
        raise ValueError(f"No .npz clips in {args.cache_dir}")
    train_stems, val_stems = split_clips_by_source(stems, tcfg.val_frac, seed=tcfg.seed)
    print(f"[train] Scene split: {len(train_stems)} train / {len(val_stems)} val clips "
          f"(by source video, val_frac={tcfg.val_frac})")

    train_ds = SportsTrainingDataset(cache_dir=args.cache_dir, clip_ids=train_stems, augment=True)
    run_sanity_check(train_ds, n_samples=min(5, len(train_ds)))
    train_ds.save_norm_stats(run_dir / "norm_stats.npz")  # TRAIN-only stats

    val_loader = None
    if val_stems:
        # val uses TRAIN norm-stats (no leakage) and no augmentation
        val_ds = SportsTrainingDataset(cache_dir=args.cache_dir, clip_ids=val_stems,
                                       augment=False, norm_stats=train_ds.norm_stats)
        from torch.utils.data import DataLoader
        val_loader = DataLoader(val_ds, batch_size=tcfg.batch_size, shuffle=False,
                                collate_fn=collate_sports_batch, drop_last=False,
                                num_workers=tcfg.num_workers)
        print(f"[train] Val: {len(val_ds)} windows")
    else:
        print("[train] WARNING: no val sources (need >= 2 source videos for scene-split) — "
              "training without held-out val / early-stop")

    if len(train_ds) < tcfg.batch_size:
        raise ValueError(f"Train set ({len(train_ds)}) < batch_size ({tcfg.batch_size}). "
                         f"Use --overfit-tiny or reduce --batch-size.")

    train_loader = _make_train_loader(train_ds, tcfg)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = tcfg.epochs * steps_per_epoch
    scheduler = _build_scheduler(optimizer, total_steps, tcfg.warmup_frac, tcfg.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    logger = TrainingLogger(log_dir=run_dir, log_every=args.log_every)
    val_log = run_dir / "val_metrics.jsonl"
    start_step, start_epoch = _maybe_resume(args, model, optimizer, scheduler, device)

    best_metric, patience_left, t0 = float("-inf"), tcfg.early_stop_patience, time.time()
    global_step = start_step
    for epoch in range(start_epoch, tcfg.epochs):
        model.train()
        for batch in train_loader:
            batch = _to_device(batch, device)
            with autocast:
                loss_out, pred_lat, pred_delta = _forward_loss(
                    model, batch, tcfg, device, args.lambda_latent, args.lambda_waypoint)
            _step_optimizer(loss_out.total, optimizer, opt_target, scaler, use_scaler, tcfg.grad_clip)
            scheduler.step()
            if logger.should_log(global_step):
                lr = scheduler.get_last_lr()[0]
                logger.log_step(step=global_step, epoch=epoch, loss_output=loss_out, lr=lr,
                                predicted_latents=pred_lat.detach(), target_latents=batch.target_latents,
                                predicted_deltas=None if pred_delta is None else pred_delta.detach(),
                                target_deltas=None if pred_delta is None else batch.target_deltas)
                print(f"  e{epoch} s{global_step} | L_lat={loss_out.latent.item():.4f} "
                      f"cos={loss_out.cosine_sim.item():.3f} lr={lr:.2e} "
                      f"{(global_step + 1) / max(time.time() - t0, 1e-6):.1f} it/s")
            global_step += 1

        if val_loader is not None and (epoch + 1) % tcfg.eval_every_epochs == 0:
            val = evaluate(model, val_loader, device, amp_dtype=tcfg.amp_dtype)
            with open(val_log, "a") as f:
                f.write(json.dumps({"epoch": epoch, "step": global_step, **val}) + "\n")
            metric = val["val_cos"] if tcfg.early_stop_metric == "val_cos" else val["val_margin"]
            print(f"[eval] e{epoch} val_cos={val['val_cos']:.4f} "
                  f"persistence={val['val_persistence']:.4f} margin={val['val_margin']:+.4f}")
            if metric > best_metric:
                best_metric, patience_left = metric, tcfg.early_stop_patience
                save_checkpoint(model, optimizer, epoch, global_step, config,
                                {"loss": loss_out.total.item(), tcfg.early_stop_metric: metric},
                                run_dir / "ckpt_best.pt", scheduler=scheduler, val_metrics=val)
                print(f"[eval] new best {tcfg.early_stop_metric}={metric:.4f} → ckpt_best.pt")
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"[train] Early stop at epoch {epoch} "
                          f"(no {tcfg.early_stop_metric} improvement in {tcfg.early_stop_patience} evals)")
                    break

    print(f"[train] Done. {global_step} steps, best {tcfg.early_stop_metric}={best_metric:.4f} "
          f"in {time.time() - t0:.1f}s")


def _maybe_resume(args, model, optimizer, scheduler, device) -> tuple[int, int]:  # noqa: ANN001
    """Restore model/optimizer/scheduler from ``--resume``; return (start_step, start_epoch)."""
    if not args.resume:
        return 0, 0
    ckpt = load_checkpoint(args.resume, model, optimizer, map_location=device)
    if "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_step, start_epoch = ckpt["global_step"] + 1, ckpt["epoch"] + 1
    print(f"[train] Resumed from {args.resume}: epoch {start_epoch}, step {start_step}")
    return start_step, start_epoch


def _step_optimizer(loss, optimizer, opt_target, scaler, use_scaler, grad_clip):  # noqa: ANN001
    """One optimizer step with optional fp16 GradScaler + grad clipping."""
    optimizer.zero_grad()
    if use_scaler:
        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(opt_target.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(opt_target.parameters(), grad_clip)
        optimizer.step()


def _nullcontext():  # noqa: ANN202
    from contextlib import nullcontext
    return nullcontext()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sports-following latent world model (B-1)")
    parser.add_argument("--cache-dir", required=True, help="Path to .npz cache directory")
    parser.add_argument("--run-dir", default="runs/default", help="Checkpoints + logs output dir")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)

    # B-1 latent-only run
    parser.add_argument("--latent-only", action="store_true",
                        help="Train the predictor only on L_latent (B-1; skip head/L_wp)")
    parser.add_argument("--no-action-film", action="store_true",
                        help="Action-free predictor ablation (dt-FiLM only)")
    parser.add_argument("--amp-dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-frac", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)

    # scene-split val + eval / early-stop
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--eval-every-epochs", type=int, default=1)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--early-stop-metric", default="val_cos", choices=("val_cos", "val_margin"))

    # game-domain mix (B1.22d)
    parser.add_argument("--domain-weight", type=float, default=1.0,
                        help="Sampling weight for domain=game clips (1.0 = no game mix)")

    # loss weights (combined / non latent-only path)
    parser.add_argument("--lambda-latent", type=float, default=1.0)
    parser.add_argument("--lambda-waypoint", type=float, default=1.0)

    # overfit-tiny smoke
    parser.add_argument("--overfit-tiny", action="store_true", help="Overfit a small subset (smoke)")
    parser.add_argument("--overfit-samples", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=500, help="Step cap (overfit-tiny mode)")

    # model structure
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.1)

    # logging
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None, help="(overfit/full) checkpoint to resume")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
