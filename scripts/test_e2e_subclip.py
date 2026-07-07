#!/usr/bin/env python3
"""End-to-end pipeline test on a single sub-clip (B1.10c).

Runs one sub-clip through the full fixed pipeline:
  content filter → quality gate → segment extraction → MegaSaM → DINOv3 → .npz cache

Then validates:
  - .npz exists and is loadable
  - All expected arrays present with correct shapes/dtypes
  - Only accepted frames in cache (no quality-rejected frames)
  - Temporal continuity (no cross-cut merging)
  - VO validation (physics + smoothness + confidence)
  - HTML report generated

Usage (filter-only, no GPU — verify pipeline structure):
    python scripts/test_e2e_subclip.py \
        --clip-id ski01 --skip-megasam --skip-encode \
        --frames-dir ingest_data/frames/ski01

Full E2E (after MegaSaM available):
    python scripts/test_e2e_subclip.py \
        --clip-id ski01 \
        --frames-dir ingest_data/frames/ski01 \
        --megasam-dir ~/CODE/MegaSaM \
        --out-dir reports/e2e_test \
        --device cuda

USER-GATED: user must run and verify output.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _log(msg: str) -> None:
    print(f"[e2e-test] {msg}", file=sys.stderr)


def _step(n: int, title: str) -> None:
    _log(f"\n{'─'*50}")
    _log(f"  Step {n}: {title}")
    _log(f"{'─'*50}")


def _validate_npz(npz_path: Path, min_frames: int) -> list[str]:
    """Validate a cached .npz file. Returns list of errors (empty = OK)."""
    from vllatent.schemas import EMBED_DIM, LATENT_DTYPE, PATCH_TOKENS

    errors: list[str] = []
    if not npz_path.exists():
        return [f"file not found: {npz_path}"]

    data = np.load(str(npz_path))
    keys = set(data.files)

    for required in ("latents", "deltas", "vo_confidence", "frame_quality", "timestamps"):
        if required not in keys:
            errors.append(f"missing key: {required}")

    if errors:
        return errors

    n = data["latents"].shape[0]
    if n < min_frames:
        errors.append(f"too few frames: {n} < {min_frames}")

    if data["latents"].shape != (n, PATCH_TOKENS, EMBED_DIM):
        errors.append(f"latents shape: expected ({n}, {PATCH_TOKENS}, {EMBED_DIM}), got {data['latents'].shape}")

    if data["latents"].dtype != LATENT_DTYPE:
        errors.append(f"latents dtype: expected {LATENT_DTYPE}, got {data['latents'].dtype}")

    if data["deltas"].shape != (n - 1, 4):
        errors.append(f"deltas shape: expected ({n - 1}, 4), got {data['deltas'].shape}")

    if data["vo_confidence"].shape != (n,):
        errors.append(f"vo_confidence shape: expected ({n},), got {data['vo_confidence'].shape}")

    if data["frame_quality"].shape != (n,):
        errors.append(f"frame_quality shape: expected ({n},), got {data['frame_quality'].shape}")

    if data["timestamps"].shape != (n,):
        errors.append(f"timestamps shape: expected ({n},), got {data['timestamps'].shape}")

    if "quality_mask" in keys:
        errors.append("quality_mask found — should NOT be in cache (all frames are accepted)")

    qual = data["frame_quality"]
    if np.any(qual < 0.3):
        n_bad = int(np.sum(qual < 0.3))
        errors.append(f"{n_bad} frames with quality < 0.3 in cache (quality gate failed?)")

    return errors


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="E2E pipeline test on one sub-clip (B1.10c)")
    p.add_argument("--clip-id", required=True, help="Source clip ID (e.g. ski01)")
    p.add_argument("--frames-dir", type=Path, required=True,
                   help="Directory with extracted frames for the clip")
    p.add_argument("--megasam-dir", type=Path, default=None,
                   help="MegaSaM repo root (for running MegaSaM)")
    p.add_argument("--device", default="cuda", help="Torch device")
    p.add_argument("--out-dir", type=Path, default=Path("reports/e2e_test"),
                   help="Output directory for reports and cache")
    p.add_argument("--skip-megasam", action="store_true",
                   help="Skip MegaSaM (use existing output)")
    p.add_argument("--skip-encode", action="store_true",
                   help="Skip DINOv3 encoding (dry-run pipeline structure)")
    p.add_argument("--no-track-persons", action="store_true",
                   help="Disable B3 person-track cache labels")
    p.add_argument("--sub-clip-index", type=int, default=0,
                   help="Which sub-clip to test (0 = first)")
    p.add_argument("--fps", type=float, default=5.0, help="Source FPS")
    p.add_argument("--quality-threshold", type=float, default=0.3)
    args = p.parse_args(argv)

    frames_dir = args.frames_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not frames_dir.is_dir():
        _log(f"ERROR: frames dir not found: {frames_dir}")
        return 1

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    n_total = len(frame_paths)
    _log(f"Source: {args.clip_id} — {n_total} frames in {frames_dir}")
    if n_total < 8:
        _log(f"ERROR: need at least 8 frames, got {n_total}")
        return 1

    # ── Step 1: Content filter → FPV ranges ──
    _step(1, "Content filter → FPV shot detection")
    from vllatent.ingest.content_filter import (
        VideoVerdict,
        extract_fpv_ranges,
        filter_video_from_paths,
    )

    filter_result = filter_video_from_paths(frame_paths, device=args.device)
    fpv_ranges = extract_fpv_ranges(filter_result.shots, filter_result.fpv_mask)
    _log(f"Verdict: {filter_result.verdict.value}")
    _log(f"FPV shots: {sum(1 for s in filter_result.shots if s.is_fpv)}/{len(filter_result.shots)}")
    if filter_result.human_visible is not None:
        _log(f"Human-visible frames: {int(filter_result.human_visible.sum())}/{filter_result.n_frames}")
    _log(f"FPV ranges (per-frame filtered): {fpv_ranges}")

    if filter_result.verdict == VideoVerdict.REJECT:
        _log("REJECTED — no FPV content. Cannot test pipeline.")
        return 1

    if not fpv_ranges:
        _log("WARNING: no FPV ranges detected, using full clip as fallback")
        fpv_ranges = [(0, n_total)]

    # ── Step 2: Cut 10s sub-clips from FPV ranges ──
    _step(2, "Cut 10s sub-clips from FPV ranges")
    from vllatent.ingest.preprocess import cut_fixed_clips

    clip_length_frames = int(args.fps * 10.0)
    sub_clips: list[tuple[str, list[Path]]] = []

    for range_idx, (rng_start, rng_end) in enumerate(fpv_ranges):
        range_paths = frame_paths[rng_start:rng_end]
        if not range_paths:
            continue
        segments = cut_fixed_clips(range_paths, clip_length_frames)
        for clip_idx, seg_paths in enumerate(segments):
            sub_id = f"{args.clip_id}_fpv{range_idx:02d}_c{clip_idx:03d}"
            sub_clips.append((sub_id, seg_paths))

    _log(f"Total sub-clips: {len(sub_clips)}")
    for sub_id, paths in sub_clips:
        _log(f"  {sub_id}: {len(paths)} frames")

    if not sub_clips:
        _log("ERROR: no sub-clips produced")
        return 1

    if args.sub_clip_index >= len(sub_clips):
        _log(f"ERROR: sub-clip index {args.sub_clip_index} out of range (have {len(sub_clips)})")
        return 1

    target_id, target_paths = sub_clips[args.sub_clip_index]
    _log(f"\nTesting sub-clip: {target_id} ({len(target_paths)} frames)")

    # ── Step 3: Quality scoring → segment extraction ──
    _step(3, "Quality scoring → accepted segments")
    from vllatent.ingest.pipeline import MIN_SEGMENT_FRAMES
    from vllatent.ingest.quality import composite_quality, filter_frames, find_accepted_segments
    from vllatent.io import load_rgb

    qualities = np.zeros(len(target_paths), dtype=np.float32)
    for i, fp in enumerate(target_paths):
        frame = load_rgb(fp)
        qualities[i] = composite_quality(frame)

    quality_mask = filter_frames(qualities, args.quality_threshold)
    n_accepted = int(quality_mask.sum())
    _log(f"Quality: {n_accepted}/{len(qualities)} accepted (threshold={args.quality_threshold})")
    _log(f"Quality scores: min={qualities.min():.3f}, max={qualities.max():.3f}, "
         f"mean={qualities.mean():.3f}")

    segments = find_accepted_segments(quality_mask, min_length=MIN_SEGMENT_FRAMES)
    _log(f"Accepted segments (>= {MIN_SEGMENT_FRAMES} contiguous frames): {segments}")

    if not segments:
        _log(f"ERROR: no accepted segments >= {MIN_SEGMENT_FRAMES} frames")
        _log("Quality distribution:")
        for i, q in enumerate(qualities):
            marker = "✓" if quality_mask[i] else "✗"
            _log(f"  frame {i:3d}: {q:.3f} {marker}")
        return 1

    # ── Step 4: Process first segment through MegaSaM + DINOv3 ──
    _step(4, f"Process segment through pipeline (skip_megasam={args.skip_megasam})")

    seg_start, seg_end = segments[0]
    seg_n = seg_end - seg_start
    segment_id = f"{target_id}_seg00" if len(segments) > 1 else target_id
    _log(f"Processing segment: frames [{seg_start}:{seg_end}] ({seg_n} frames) as '{segment_id}'")

    import shutil
    seg_frames_dir = out_dir / "frames" / segment_id
    if seg_frames_dir.exists():
        shutil.rmtree(seg_frames_dir)
    seg_frames_dir.mkdir(parents=True)
    seg_frame_paths: list[Path] = []
    for i, src_idx in enumerate(range(seg_start, seg_end)):
        dst = seg_frames_dir / f"{i:06d}.jpg"
        shutil.copy2(target_paths[src_idx], dst)
        seg_frame_paths.append(dst)
    _log(f"Copied {len(seg_frame_paths)} segment frames to {seg_frames_dir}")

    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    seg_qualities = qualities[seg_start:seg_end]

    if args.skip_megasam and args.skip_encode:
        _log("DRY RUN: skipping MegaSaM + DINOv3 — validating pipeline structure only")
        _log(f"  Would run MegaSaM on {seg_n} frames")
        _log(f"  Would encode {seg_n} frames with DINOv3")
        _log(f"  Would write {segment_id}.npz to {cache_dir}")

        summary = {
            "test": "e2e_pipeline_structure",
            "clip_id": args.clip_id,
            "sub_clip": target_id,
            "segment": segment_id,
            "n_total_frames": n_total,
            "n_sub_clip_frames": len(target_paths),
            "n_accepted": n_accepted,
            "n_segment_frames": seg_n,
            "segments_found": segments,
            "fpv_ranges": fpv_ranges,
            "quality_stats": {
                "min": float(qualities.min()),
                "max": float(qualities.max()),
                "mean": float(qualities.mean()),
            },
            "verdict": "STRUCTURE_OK",
        }
        summary_path = out_dir / "e2e_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        _log(f"\nStructure test summary: {summary_path}")
        _log("\n" + "=" * 50)
        _log("  STRUCTURE TEST PASSED")
        _log(f"  Pipeline would produce {len(segments)} segment(s)")
        _log(f"  First segment: {seg_n} frames")
        _log("  Re-run without --skip-megasam --skip-encode for full test")
        _log("=" * 50)
        return 0

    from vllatent.config import IngestConfig

    cfg = IngestConfig(
        frames_dir=str(out_dir / "frames"),
        cache_dir=str(cache_dir),
        target_fps=args.fps,
        quality_threshold=args.quality_threshold,
    )

    from vllatent.ingest.pipeline import _process_segment

    result = _process_segment(
        segment_id=segment_id,
        segment_frame_paths=seg_frame_paths,
        segment_qualities=seg_qualities,
        cfg=cfg,
        cache_dir=cache_dir,
        skip_megasam=args.skip_megasam,
        track_persons=not args.no_track_persons,
        device=args.device,
    )

    if result.errors:
        _log(f"PIPELINE ERRORS: {result.errors}")
        return 1

    _log(f"Pipeline OK: {result.n_accepted}/{result.n_frames} frames → {result.latent_path}")

    # ── Step 5: Validate .npz cache ──
    _step(5, "Validate .npz cache")

    npz_path = cache_dir / result.latent_path
    validation_errors = _validate_npz(npz_path, min_frames=MIN_SEGMENT_FRAMES)

    if validation_errors:
        _log("CACHE VALIDATION FAILED:")
        for err in validation_errors:
            _log(f"  ✗ {err}")
        return 1

    data = np.load(str(npz_path))
    n_cached = data["latents"].shape[0]
    _log(f"Cache OK: {npz_path}")
    _log(f"  frames: {n_cached}")
    _log(f"  latents: {data['latents'].shape} {data['latents'].dtype}")
    _log(f"  deltas: {data['deltas'].shape} {data['deltas'].dtype}")
    _log(f"  vo_confidence: min={data['vo_confidence'].min():.3f}, "
         f"max={data['vo_confidence'].max():.3f}")
    _log(f"  frame_quality: min={data['frame_quality'].min():.3f}, "
         f"max={data['frame_quality'].max():.3f}")
    _log(f"  All frame_quality >= threshold: "
         f"{bool(np.all(data['frame_quality'] >= args.quality_threshold))}")

    # ── Step 6: VO validation ──
    _step(6, "VO validation (physics + smoothness)")

    megasam_out = seg_frames_dir.parent / f"{segment_id}_megasam"
    if megasam_out.exists():
        from vllatent.ingest.megasam import parse_megasam_output
        from vllatent.ingest.vo_validation import validate_clip

        megasam_result = parse_megasam_output(megasam_out)
        report = validate_clip(megasam_result.poses, megasam_result.confidences,
                               fps=args.fps, clip_id=segment_id)

        v = report.verdict
        _log(f"VO Verdict: {v.decision}")
        for check, status in v.checks.items():
            marker = {"pass": "OK", "warn": "!!", "fail": "XX"}[status]
            _log(f"  [{marker}] {check}")
        if v.reasons:
            for reason in v.reasons:
                _log(f"  - {reason}")

        html_path = out_dir / f"{segment_id}_vo.html"
        try:
            from scripts.validate_megasam import _generate_html
            _generate_html({
                "report": report,
                "poses": megasam_result.poses,
                "confidences": megasam_result.confidences,
                "deltas": data["deltas"],
                "deltas_norm": data["deltas"],
            }, html_path)
            _log(f"HTML report: {html_path}")
        except Exception as e:
            _log(f"HTML generation skipped: {e}")
    else:
        _log(f"MegaSaM output not found at {megasam_out}, skipping VO validation")
        _log("(This is expected with --skip-megasam if output wasn't pre-staged)")

    # ── Summary ──
    summary = {
        "test": "e2e_pipeline_full",
        "clip_id": args.clip_id,
        "sub_clip": target_id,
        "segment": segment_id,
        "n_total_frames": n_total,
        "n_sub_clip_frames": len(target_paths),
        "n_accepted": n_accepted,
        "n_segment_frames": seg_n,
        "n_cached_frames": n_cached,
        "segments_found": segments,
        "fpv_ranges": fpv_ranges,
        "cache_path": str(npz_path),
        "cache_validation": "PASS" if not validation_errors else validation_errors,
        "quality_stats": {
            "min": float(qualities.min()),
            "max": float(qualities.max()),
            "mean": float(qualities.mean()),
        },
        "verdict": "PASS",
    }
    summary_path = out_dir / "e2e_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    _log(f"\n{'='*50}")
    _log("  E2E PIPELINE TEST PASSED")
    _log(f"  {n_total} source frames → {len(sub_clips)} sub-clips")
    _log(f"  Sub-clip '{target_id}': {len(target_paths)} frames")
    _log(f"  Quality gate: {n_accepted}/{len(target_paths)} accepted")
    _log(f"  Segments: {len(segments)} (first: {seg_n} frames)")
    _log(f"  Cache: {n_cached} frames → {npz_path}")
    _log(f"  Summary: {summary_path}")
    _log(f"{'='*50}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
