#!/usr/bin/env python3
"""Verify content filter: split frames into accepted/ vs rejected/ for visual inspection.

Usage:
    python scripts/verify_filter.py --frames data/frames/ski01 --device cuda
    # Then open accepted/ and rejected/ to see which frames the filter kept vs dropped.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify content filter visually")
    parser.add_argument("--frames", required=True, help="Directory of extracted JPEGs")
    parser.add_argument("--device", default="cuda", help="Torch device")
    parser.add_argument("--out", default=None, help="Output dir (default: <frames>/../filter_verify)")
    args = parser.parse_args(argv)

    frames_dir = Path(args.frames)
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        print(f"No JPEGs in {frames_dir}", file=sys.stderr)
        return 1

    print(f"[verify] {len(frame_paths)} frames in {frames_dir}")

    from vllatent.ingest.content_filter import (
        compute_motion_scores,
        detect_rejected_objects_from_paths,
        extract_fpv_ranges,
        filter_video_from_paths,
    )

    print("[verify] Computing motion scores...")
    motion_scores = compute_motion_scores(frame_paths)

    print("[verify] Running YOLO-World object detection...")
    rejected = detect_rejected_objects_from_paths(frame_paths, device=args.device)

    print(f"\n[verify] Per-frame diagnostics (first 30 frames):")
    print(f"  {'frame':<14} {'motion':>8} {'objects':>8} {'decision'}")
    print(f"  {'─' * 14} {'─' * 8} {'─' * 8} {'─' * 10}")
    for i in range(min(30, len(frame_paths))):
        motion = motion_scores[i]
        has_obj = rejected[i]
        ok = motion >= 8.0 and not has_obj
        if ok:
            tag = "  FPV"
        elif motion < 8.0:
            tag = "  STATIC"
        else:
            tag = "  OBJECT"
        obj_str = "YES" if has_obj else "---"
        print(f"  {frame_paths[i].name:<14} {motion:8.1f} {obj_str:>8} {tag}")

    print(f"\n[verify] Running full filter pipeline...")
    result = filter_video_from_paths(frame_paths, device=args.device)

    print(f"[verify] Verdict: {result.verdict.value}")
    print(f"[verify] FPV frames: {result.n_fpv_frames}/{result.n_frames}")
    print(f"[verify] Shots: {len(result.shots)} total, {sum(1 for s in result.shots if s.is_fpv)} FPV")

    n_objects = int(rejected.sum())
    n_static = int((motion_scores < 8.0).sum())
    print(f"[verify] Rejected: {n_objects} object-detected, {n_static} static")

    out_dir = Path(args.out) if args.out else frames_dir.parent / "filter_verify"
    accepted_dir = out_dir / "accepted"
    rejected_dir = out_dir / "rejected"
    for d in [accepted_dir, rejected_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    for i, path in enumerate(frame_paths):
        dst = accepted_dir if result.fpv_mask[i] else rejected_dir
        shutil.copy2(path, dst / path.name)

    fpv_ranges = extract_fpv_ranges(result.shots, result.fpv_mask)

    print(f"\n[verify] Results written to {out_dir}/")
    print(f"  accepted/  — {result.n_fpv_frames} frames")
    print(f"  rejected/  — {result.n_frames - result.n_fpv_frames} frames")
    print(f"\n[verify] FPV ranges (frame indices):")
    for start, end in fpv_ranges:
        print(f"  [{start}–{end}) = {end - start} frames")

    print(f"\n[verify] Shot breakdown:")
    for s in result.shots:
        label = "FPV" if s.is_fpv else "---"
        print(f"  [{s.start}–{s.end})  {label}  score={s.mean_score:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
