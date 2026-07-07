#!/usr/bin/env python3
"""Run B3.4 Stage-0 G0/K1/K2 gates over cached sports latents."""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from vllatent.data.sports_loader import SportsTrainingDataset
from vllatent.schemas import HISTORY
from vllatent.train.person_probes import (
    collect_frame_probe_examples,
    collect_window_probe_examples,
    evaluate_stage0_gates,
    fit_stage0_probes,
    run_k1_plan_only_causality,
    run_k2_conditioned_predictor,
)


def _clip_ids(cache_dir: Path, limit: int | None) -> list[str] | None:
    if limit is None:
        return None
    return [p.stem for p in sorted(cache_dir.glob("*.npz"))[:limit]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, help="Directory containing .npz latent caches")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--history", type=int, default=HISTORY)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ridge-l2", type=float, default=1e-3)
    parser.add_argument("--limit-clips", type=int, default=None)
    parser.add_argument("--max-frames-per-clip", type=int, default=None)
    parser.add_argument("--limit-windows", type=int, default=None)
    parser.add_argument("--spatial-projections", type=int, default=8)
    parser.add_argument("--out", default=None, help="Optional JSON report path")
    args = parser.parse_args(argv)

    cache_dir = Path(args.cache_dir)
    clip_ids = _clip_ids(cache_dir, args.limit_clips)

    frame_examples = collect_frame_probe_examples(
        cache_dir,
        clip_ids=clip_ids,
        max_frames_per_clip=args.max_frames_per_clip,
        n_spatial_projections=args.spatial_projections,
    )
    stage0 = fit_stage0_probes(
        frame_examples,
        val_frac=args.val_frac,
        seed=args.seed,
        l2=args.ridge_l2,
    )

    dataset = SportsTrainingDataset(
        cache_dir,
        clip_ids=clip_ids,
        history=args.history,
        horizon=args.horizon,
    )
    window_examples = collect_window_probe_examples(
        dataset,
        limit_samples=args.limit_windows,
        n_spatial_projections=args.spatial_projections,
    )
    k1 = run_k1_plan_only_causality(
        window_examples,
        val_frac=args.val_frac,
        seed=args.seed,
        l2=args.ridge_l2,
    )
    k2 = run_k2_conditioned_predictor(
        window_examples,
        val_frac=args.val_frac,
        seed=args.seed,
        l2=args.ridge_l2,
    )
    decision = evaluate_stage0_gates(stage0, k1, k2)

    report = {
        "cache_dir": str(cache_dir),
        "horizon": args.horizon,
        "history": args.history,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "limit_clips": args.limit_clips,
        "max_frames_per_clip": args.max_frames_per_clip,
        "limit_windows": args.limit_windows,
        "spatial_projections": args.spatial_projections,
        "stage0": dataclasses.asdict(stage0),
        "k1": dataclasses.asdict(k1),
        "k2": dataclasses.asdict(k2),
        "decision": dataclasses.asdict(decision),
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
    print(text)
    return 0 if decision.passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
