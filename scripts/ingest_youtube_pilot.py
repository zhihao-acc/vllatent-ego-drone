#!/usr/bin/env python3
"""YouTube pilot ingest: download + content filter + full pipeline (B1.7).

Orchestrates:
  1. Download clips from configs/sports_clips.yaml via yt-dlp (SponsorBlock pre-strip)
  2. Run content filter (motion + YOLO object-negative + YOLO human-positive)
  3. Cut accepted human-visible FPV ranges into fixed clips
  4. Run full pipeline on accepted clips (quality → person gate → MegaSaM → DINOv3 → cache)
  5. Update manifest

Usage:
    python scripts/ingest_youtube_pilot.py [--limit N] [--device cuda] [--skip-download]

USER-GATED: the user must run this script and verify output.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _log(msg: str) -> None:
    print(f"[youtube-pilot] {msg}", file=sys.stderr)


def _find_existing_video(raw_dir: Path, clip_id: str) -> Path | None:
    """Return the best existing video file for a clip, ignoring yt-dlp audio streams."""
    candidates = list(raw_dir.glob(f"{clip_id}.*"))
    exact_video_exts = {".mp4", ".mkv", ".avi", ".mov"}
    webm_audio_formats = {".f249.webm", ".f250.webm", ".f251.webm"}

    exact = sorted(
        p for p in candidates
        if p.stem == clip_id and p.suffix in {*exact_video_exts, ".webm"}
    )
    if exact:
        return exact[0]

    split_mp4 = sorted(
        p for p in candidates
        if p.suffix in exact_video_exts and ".f140." not in p.name
    )
    if split_mp4:
        return split_mp4[0]

    split_webm = sorted(
        p for p in candidates
        if p.suffix == ".webm" and not any(p.name.endswith(audio_suffix) for audio_suffix in webm_audio_formats)
    )
    if split_webm:
        return split_webm[0]

    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="YouTube pilot ingest (B1.7)")
    parser.add_argument("--clips", default="configs/sports_clips.yaml", help="Clips YAML")
    parser.add_argument("--config", default="configs/sports.yaml", help="Sports config")
    parser.add_argument("--limit", type=int, default=0, help="Max clips to process (0=all)")
    parser.add_argument("--device", default="cuda", help="Torch device for CLIP/DINOv3")
    parser.add_argument("--skip-download", action="store_true", help="Skip yt-dlp download")
    parser.add_argument("--skip-megasam", action="store_true", help="Skip MegaSaM VO")
    parser.add_argument("--no-track-persons", action="store_true", help="Disable B3 person-track segment gate")
    parser.add_argument("--filter-only", action="store_true", help="Download + filter only, no pipeline")
    args = parser.parse_args(argv)

    clips_data = yaml.safe_load(Path(args.clips).read_text()) or {}
    clips = clips_data.get("clips", [])
    if not clips:
        _log(f"ERROR: no clips in {args.clips}")
        return 1

    if args.limit > 0:
        clips = clips[:args.limit]

    from vllatent.config import Config
    config = Config.from_yaml(args.config)
    if config.ingest is None:
        _log(f"ERROR: {args.config} has no 'ingest' section")
        return 1
    cfg = config.ingest

    raw_dir = Path(cfg.raw_dir)
    frames_dir = Path(cfg.frames_dir)
    cache_dir = Path(cfg.cache_dir)

    raw_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    _log(f"Processing {len(clips)} clips from {args.clips}")
    _log(f"  raw_dir={raw_dir}, frames_dir={frames_dir}, cache_dir={cache_dir}")
    _log(f"  device={args.device}, skip_download={args.skip_download}")
    _log(f"  track_persons={not args.no_track_persons}")

    results_summary: list[dict] = []

    for i, clip in enumerate(clips):
        clip_id = clip.get("clip_id", f"clip_{i:03d}")
        url = clip.get("url", "")
        if not url:
            _log(f"SKIP {clip_id}: no URL")
            continue

        _log(f"\n{'='*60}")
        _log(f"[{i+1}/{len(clips)}] {clip_id}: {url}")

        # --- Step 1: Download with SponsorBlock ---
        if not args.skip_download:
            existing_video = _find_existing_video(raw_dir, clip_id)
            if existing_video is not None:
                _log(f"  already downloaded, skipping ({existing_video.name})")
            else:
                _log("  downloading (SponsorBlock enabled)...")
                try:
                    from vllatent.ingest.acquire import download_clip
                    download_clip(
                        url, str(raw_dir),
                        clip_id=clip_id,
                        max_height=cfg.resolution_h,
                        sponsorblock=True,
                    )
                    _log("  download OK")
                except Exception as e:
                    _log(f"  DOWNLOAD FAILED: {e}")
                    results_summary.append({"clip_id": clip_id, "status": "download_failed", "error": str(e)})
                    continue

        # --- Step 2: Quick content filter (sample frames from video) ---
        clip_frames_dir = frames_dir / clip_id
        if not clip_frames_dir.exists() or not list(clip_frames_dir.glob("*.jpg")):
            video_path = _find_existing_video(raw_dir, clip_id)
            if video_path is None:
                _log(f"  SKIP: no video file for {clip_id}")
                results_summary.append({"clip_id": clip_id, "status": "no_video"})
                continue
            _log(f"  extracting frames @ {cfg.target_fps} fps...")
            try:
                from vllatent.ingest.preprocess import extract_frames
                extract_frames(
                    video_path, str(clip_frames_dir),
                    target_fps=cfg.target_fps,
                    resolution_hw=(cfg.resolution_h, cfg.resolution_w),
                )
            except Exception as e:
                _log(f"  EXTRACT FAILED: {e}")
                results_summary.append({"clip_id": clip_id, "status": "extract_failed", "error": str(e)})
                continue

        frame_paths = sorted(clip_frames_dir.glob("*.jpg"))
        n_frames = len(frame_paths)
        _log(f"  {n_frames} frames on disk")

        if n_frames < 7:
            _log(f"  SKIP: too few frames ({n_frames})")
            results_summary.append({"clip_id": clip_id, "status": "too_few_frames", "n_frames": n_frames})
            continue

        _log("  running content filter (motion + object-negative + human-positive, every frame)...")
        try:
            from vllatent.ingest.content_filter import (
                VideoVerdict,
                extract_fpv_ranges,
                filter_video_from_paths,
                save_filter_result,
            )

            filter_result = filter_video_from_paths(frame_paths, device=args.device)

            # Persist per-frame decisions for QC (read by scripts/qc_report.py).
            save_filter_result(clip_frames_dir, filter_result)

            fpv_count = sum(1 for s in filter_result.shots if s.is_fpv)
            n_object_frames = (
                int(filter_result.rejected_objects.sum())
                if filter_result.rejected_objects is not None
                else 0
            )
            n_human_frames = (
                int(filter_result.human_visible.sum())
                if filter_result.human_visible is not None
                else 0
            )
            _log(f"  verdict: {filter_result.verdict.value}, FPV shots: {fpv_count}/{len(filter_result.shots)}")
            _log(f"  FPV frames: {filter_result.n_fpv_frames}/{filter_result.n_frames}")
            _log(f"  YOLO frames: {n_object_frames} object-rejected, {n_human_frames} human-visible")

            if filter_result.verdict == VideoVerdict.REJECT:
                _log("  REJECTED — skipping")
                results_summary.append({
                    "clip_id": clip_id,
                    "status": "content_rejected",
                    "fpv_shots": fpv_count,
                    "total_shots": len(filter_result.shots),
                    "object_rejected_frames": n_object_frames,
                    "human_visible_frames": n_human_frames,
                })
                continue

        except Exception as e:
            _log(f"  CONTENT FILTER WARNING: {e} — proceeding anyway")
            filter_result = None

        # --- Step 3: Extract FPV ranges and cut into 10s clips ---
        from vllatent.ingest.content_filter import extract_fpv_ranges
        from vllatent.ingest.preprocess import cut_fixed_clips

        fpv_ranges: list[tuple[int, int]] = []
        if filter_result is not None:
            fpv_ranges = extract_fpv_ranges(filter_result.shots, filter_result.fpv_mask)

        if not fpv_ranges:
            fpv_ranges = [(0, n_frames)]

        clip_length_frames = int(cfg.clip_length_seconds * cfg.target_fps)
        sub_clips: list[tuple[str, list[Path]]] = []

        for range_idx, (rng_start, rng_end) in enumerate(fpv_ranges):
            range_frame_paths = frame_paths[rng_start:rng_end]
            if not range_frame_paths:
                continue
            segments = cut_fixed_clips(range_frame_paths, clip_length_frames)
            for clip_idx, seg_paths in enumerate(segments):
                sub_id = f"{clip_id}_fpv{range_idx:02d}_c{clip_idx:03d}"
                sub_clips.append((sub_id, seg_paths))

        _log(f"  FPV ranges: {len(fpv_ranges)}, 10s clips: {len(sub_clips)}")

        if args.filter_only:
            _log("  ACCEPTED (filter-only mode, skipping pipeline)")
            results_summary.append({
                "clip_id": clip_id,
                "status": "filter_accepted",
                "fpv_ranges": len(fpv_ranges),
                "n_sub_clips": len(sub_clips),
                "human_visible_frames": (
                    int(filter_result.human_visible.sum())
                    if filter_result is not None and filter_result.human_visible is not None
                    else 0
                ),
            })
            continue

        # --- Step 4: Run pipeline on each sub-clip ---
        from vllatent.ingest.pipeline import process_clip

        n_clip_ok = 0
        n_clip_err = 0

        for sub_id, seg_paths in sub_clips:
            sub_frames_dir = frames_dir / sub_id
            sub_frames_dir.mkdir(parents=True, exist_ok=True)
            for j, src in enumerate(seg_paths):
                dst = sub_frames_dir / f"{j:06d}.jpg"
                if not dst.exists():
                    shutil.copy2(src, dst)

            _log(f"    pipeline {sub_id} ({len(seg_paths)} frames)...")
            try:
                segment_results = process_clip(
                    url=url,
                    clip_id=sub_id,
                    cfg=cfg,
                    skip_download=True,
                    skip_megasam=args.skip_megasam,
                    device=args.device,
                    track_persons=not args.no_track_persons,
                )

                for seg_result in segment_results:
                    if seg_result.errors:
                        _log(f"    PIPELINE ERRORS ({seg_result.clip_id}): {seg_result.errors}")
                        results_summary.append({
                            "clip_id": seg_result.clip_id,
                            "status": "pipeline_error",
                            "errors": seg_result.errors,
                        })
                        n_clip_err += 1
                    else:
                        _log(f"    OK: {seg_result.clip_id} — {seg_result.n_accepted}/{seg_result.n_frames} frames")
                        results_summary.append({
                            "clip_id": seg_result.clip_id,
                            "status": "ok",
                            "n_frames": seg_result.n_frames,
                            "n_accepted": seg_result.n_accepted,
                            "latent_path": seg_result.latent_path,
                        })
                        n_clip_ok += 1

            except Exception as e:
                _log(f"    PIPELINE FAILED: {e}")
                results_summary.append({"clip_id": sub_id, "status": "pipeline_failed", "error": str(e)})
                n_clip_err += 1

        _log(f"  clip summary: {n_clip_ok} OK, {n_clip_err} failed out of {len(sub_clips)}")

    # --- Summary ---
    _log(f"\n{'='*60}")
    _log("PILOT INGEST SUMMARY")
    _log(f"{'='*60}")

    n_ok = sum(1 for r in results_summary if r.get("status") == "ok")
    n_filter_ok = sum(1 for r in results_summary if r.get("status") == "filter_accepted")
    n_rejected = sum(1 for r in results_summary if r.get("status") == "content_rejected")
    n_failed = len(results_summary) - n_ok - n_filter_ok - n_rejected
    n_fpv_ranges_total = sum(r.get("fpv_ranges", 0) for r in results_summary)
    n_sub_clips_total = sum(r.get("n_sub_clips", 0) for r in results_summary)

    _log(f"  Total clips: {len(clips)}")
    _log(f"  Pipeline OK (sub-clips): {n_ok}")
    if n_filter_ok:
        _log(f"  Filter accepted (no pipeline): {n_filter_ok}")
        _log(f"    FPV ranges: {n_fpv_ranges_total}, 10s sub-clips: {n_sub_clips_total}")
    _log(f"  Content-rejected: {n_rejected}")
    _log(f"  Failed: {n_failed}")

    for r in results_summary:
        status = r.get("status", "unknown")
        cid = r.get("clip_id", "?")
        if status == "ok":
            _log(f"    OK  {cid}: {r.get('n_accepted')}/{r.get('n_frames')} frames -> {r.get('latent_path')}")
        elif status == "filter_accepted":
            _log(f"    FLT {cid}: {r.get('fpv_ranges')} FPV ranges, {r.get('n_sub_clips')} sub-clips")
        elif status == "content_rejected":
            _log(f"    REJ {cid}: {r.get('fpv_shots')}/{r.get('total_shots')} FPV shots")
        else:
            _log(f"    ERR {cid}: {status} — {r.get('error', r.get('errors', ''))}")

    summary_path = cache_dir / "pilot_summary.json"
    summary_path.write_text(json.dumps(results_summary, indent=2))
    _log(f"\nSummary: {summary_path}")

    return 0 if (n_ok > 0 or n_filter_ok > 0) else 1


if __name__ == "__main__":
    sys.exit(main())
