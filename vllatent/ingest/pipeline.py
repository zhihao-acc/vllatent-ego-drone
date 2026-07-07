"""End-to-end per-clip processing pipeline (ORCH tier).

Wires: acquire -> extract -> quality-gate -> megasam -> ego_motion -> encode -> cache.

Quality scoring is a GATE, not a label. Only contiguous segments of accepted
frames are processed through MegaSaM and DINOv3. Each segment becomes its own
.npz cache file. MegaSaM needs temporal continuity — segments are never split
mid-run; rejected frames at segment boundaries define the split points.
"""
from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vllatent.config import IngestConfig
from vllatent.io import load_rgb
from vllatent.schemas import (
    DELTA_DTYPE,
    EMBED_DIM,
    HISTORY,
    HORIZON,
    LATENT_DTYPE,
    PATCH_TOKENS,
)

MIN_SEGMENT_FRAMES = HISTORY + HORIZON + 1


@dataclass(frozen=True)
class ClipPipelineResult:
    clip_id: str
    n_frames: int
    n_accepted: int
    latent_path: str
    stages_skipped: list[str]
    errors: list[str]


def _log(msg: str) -> None:
    print(f"[ingest-pipeline] {msg}", file=sys.stderr)


def _build_clip_npz(
    *,
    latents: np.ndarray,
    deltas: np.ndarray,
    vo_confidence: np.ndarray,
    frame_quality: np.ndarray,
    timestamps: np.ndarray,
    person_bbox: np.ndarray | None = None,
    person_visible: np.ndarray | None = None,
    person_conf: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Validate and return the arrays dict for a single segment's .npz."""
    from vllatent.ingest.person_tracking import empty_person_tracks, validate_person_track_arrays

    n = latents.shape[0]
    if latents.shape != (n, PATCH_TOKENS, EMBED_DIM):
        raise ValueError(f"latents: expected (N, {PATCH_TOKENS}, {EMBED_DIM}), got {latents.shape}")
    if deltas.shape != (n - 1, 4):
        raise ValueError(f"deltas: expected ({n - 1}, 4), got {deltas.shape}")
    if vo_confidence.shape != (n,):
        raise ValueError(f"vo_confidence: expected ({n},), got {vo_confidence.shape}")
    if frame_quality.shape != (n,):
        raise ValueError(f"frame_quality: expected ({n},), got {frame_quality.shape}")
    if timestamps.shape != (n,):
        raise ValueError(f"timestamps: expected ({n},), got {timestamps.shape}")

    if person_bbox is None or person_visible is None or person_conf is None:
        tracks = empty_person_tracks(n)
        person_bbox = tracks.person_bbox
        person_visible = tracks.person_visible
        person_conf = tracks.person_conf
    validate_person_track_arrays(
        n_frames=n,
        person_bbox=person_bbox,
        person_visible=person_visible,
        person_conf=person_conf,
    )

    return {
        "latents": latents.astype(LATENT_DTYPE),
        "deltas": deltas.astype(DELTA_DTYPE),
        "vo_confidence": vo_confidence.astype(np.float32),
        "frame_quality": frame_quality.astype(np.float32),
        "timestamps": timestamps.astype(np.float64),
        "person_bbox": person_bbox.astype(np.float32),
        "person_visible": person_visible.astype(np.bool_),
        "person_conf": person_conf.astype(np.float32),
        "person_bbox_space": np.array("encoder_crop"),
    }


def _write_clip_npz(
    arrays: dict[str, np.ndarray],
    out_path: str | Path,
) -> Path:
    """Write a segment's arrays to a .npz file."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(p), **arrays)
    return p


def _prepare_segment_frames(
    frame_paths: list[Path],
    seg_start: int,
    seg_end: int,
    seg_dir: Path,
) -> list[Path]:
    """Copy a contiguous segment's frames into a clean directory.

    Frames are renumbered sequentially (000000.jpg, 000001.jpg, ...) so
    MegaSaM sees a gapless continuous sequence.
    """
    seg_dir.mkdir(parents=True, exist_ok=True)
    seg_paths: list[Path] = []
    for i, src_idx in enumerate(range(seg_start, seg_end)):
        dst = seg_dir / f"{i:06d}.jpg"
        if not dst.exists():
            shutil.copy2(frame_paths[src_idx], dst)
        seg_paths.append(dst)
    return seg_paths


def _process_segment(
    *,
    segment_id: str,
    segment_frame_paths: list[Path],
    segment_qualities: np.ndarray,
    cfg: IngestConfig,
    cache_dir: Path,
    skip_megasam: bool,
    track_persons: bool,
    device: str,
) -> ClipPipelineResult:
    """Process a single quality-accepted segment through MegaSaM → DINOv3 → cache."""
    from vllatent.encode.batch import encode_frames
    from vllatent.ingest.ego_motion import normalize_scale, se3_sequence_to_deltas
    from vllatent.ingest.megasam import parse_megasam_output, run_megasam
    from vllatent.ingest.person_tracking import empty_person_tracks, track_persons_from_paths

    errors: list[str] = []
    skipped: list[str] = []
    n_seg = len(segment_frame_paths)
    seg_frames_dir = segment_frame_paths[0].parent

    megasam_out = seg_frames_dir.parent / f"{segment_id}_megasam"

    # --- MegaSaM ego-motion ---
    if skip_megasam:
        skipped.append("megasam")
        _log(f"  {segment_id}: using existing MegaSaM output at {megasam_out}")
    else:
        _log(f"  {segment_id}: running MegaSaM ({n_seg} frames)")
        run_megasam(str(seg_frames_dir), str(megasam_out), clip_id=segment_id)

    megasam_result = parse_megasam_output(megasam_out)

    # --- SE(3) → body-frame deltas ---
    deltas = se3_sequence_to_deltas(megasam_result.poses, fps=cfg.target_fps)
    deltas = normalize_scale(deltas, mode="median_speed")

    n_pose = megasam_result.poses.shape[0]
    if n_pose != n_seg:
        _log(f"  {segment_id}: pose count ({n_pose}) != frame count ({n_seg}), truncating")
        n = min(n_pose, n_seg)
    else:
        n = n_seg

    # --- DINOv3 encoding ---
    _log(f"  {segment_id}: encoding {n} frames with DINOv3")
    latents = encode_frames(seg_frames_dir, device=device)

    n = min(n, latents.shape[0])
    latents = latents[:n]
    deltas = deltas[:n - 1]
    qualities = segment_qualities[:n]
    vo_conf = megasam_result.confidences[:n].astype(np.float32)
    timestamps = np.arange(n, dtype=np.float64) / cfg.target_fps
    if track_persons:
        _log(f"  {segment_id}: tracking person subject with YOLO-World/ByteTrack")
        person_tracks = track_persons_from_paths(segment_frame_paths[:n], device=device)
    else:
        person_tracks = empty_person_tracks(n)

    # --- Cache assembly ---
    arrays = _build_clip_npz(
        latents=latents,
        deltas=deltas,
        vo_confidence=vo_conf,
        frame_quality=qualities,
        timestamps=timestamps,
        person_bbox=person_tracks.person_bbox,
        person_visible=person_tracks.person_visible,
        person_conf=person_tracks.person_conf,
    )

    latent_rel = f"{segment_id}.npz"
    _write_clip_npz(arrays, cache_dir / latent_rel)
    _log(f"  {segment_id}: wrote {cache_dir / latent_rel} ({n} frames)")

    return ClipPipelineResult(
        clip_id=segment_id,
        n_frames=n,
        n_accepted=n,
        latent_path=latent_rel,
        stages_skipped=skipped,
        errors=errors,
    )


def process_clip(
    *,
    url: str,
    clip_id: str,
    cfg: IngestConfig,
    skip_download: bool = False,
    skip_megasam: bool = False,
    device: str = "cuda",
    camera_K: np.ndarray | None = None,
    camera_D: np.ndarray | None = None,
    track_persons: bool = False,
) -> list[ClipPipelineResult]:
    """Process a single clip end-to-end.

    Quality scoring gates downstream processing: only contiguous segments of
    accepted frames reach MegaSaM and DINOv3. Returns one result per segment
    (may be empty if quality rejects everything).
    """
    from vllatent.ingest.acquire import download_clip, validate_clip
    from vllatent.ingest.preprocess import batch_undistort, extract_frames
    from vllatent.ingest.quality import composite_quality, filter_frames, find_accepted_segments

    errors: list[str] = []
    skipped: list[str] = []

    raw_dir = Path(cfg.raw_dir)
    frames_dir = Path(cfg.frames_dir) / clip_id
    cache_dir = Path(cfg.cache_dir)

    # --- Stage 1: Download ---
    if skip_download:
        skipped.append("download")
        # The video is only needed to extract frames (Stage 2). When the
        # orchestrator has already cut a sub-clip's frames into frames_dir
        # (no per-sub-clip video exists on disk), we proceed from those frames.
        video_exts = {".mp4", ".mkv", ".webm", ".avi", ".mov"}
        candidates = [
            p for p in raw_dir.glob(f"{clip_id}.*") if p.suffix.lower() in video_exts
        ]
        video_path = candidates[0] if candidates else None
        frames_present = frames_dir.exists() and bool(list(frames_dir.glob("*.jpg")))
        if video_path is None and not frames_present:
            return [ClipPipelineResult(
                clip_id=clip_id, n_frames=0, n_accepted=0,
                latent_path="", stages_skipped=skipped,
                errors=[
                    f"No video in {raw_dir} nor pre-extracted frames in "
                    f"{frames_dir} for {clip_id}"
                ],
            )]
    else:
        _log(f"downloading {clip_id}")
        raw_dir.mkdir(parents=True, exist_ok=True)
        meta = download_clip(url, str(raw_dir), clip_id=clip_id, max_height=cfg.resolution_h)
        video_path = meta.path
        if not validate_clip(video_path):
            errors.append(f"downloaded video failed validation: {video_path}")
            return [ClipPipelineResult(
                clip_id=clip_id, n_frames=0, n_accepted=0,
                latent_path="", stages_skipped=skipped, errors=errors,
            )]

    # --- Stage 2: Frame extraction ---
    if frames_dir.exists() and list(frames_dir.glob("*.jpg")):
        skipped.append("extract_frames")
        _log(f"frames already extracted for {clip_id}")
    else:
        _log(f"extracting frames @ {cfg.target_fps} fps")
        extract_frames(
            video_path, str(frames_dir),
            target_fps=cfg.target_fps,
            resolution_hw=(cfg.resolution_h, cfg.resolution_w),
        )

    n_frames_on_disk = len(list(frames_dir.glob("*.jpg")))
    if n_frames_on_disk < MIN_SEGMENT_FRAMES:
        errors.append(f"too few frames extracted: {n_frames_on_disk}")
        return [ClipPipelineResult(
            clip_id=clip_id, n_frames=n_frames_on_disk, n_accepted=0,
            latent_path="", stages_skipped=skipped, errors=errors,
        )]

    # --- Stage 2b: Fisheye undistortion (optional) ---
    if cfg.undistort_model != "pinhole" and camera_K is not None and camera_D is not None:
        undistort_dir = Path(cfg.frames_dir) / f"{clip_id}_undistorted"
        if undistort_dir.exists() and list(undistort_dir.glob("*.jpg")):
            skipped.append("undistort")
            _log(f"undistorted frames already exist for {clip_id}")
        else:
            _log(f"undistorting frames ({cfg.undistort_model})")
            batch_undistort(frames_dir, undistort_dir, camera_K, camera_D)
        frames_dir = undistort_dir
    elif cfg.undistort_model != "pinhole":
        skipped.append("undistort")
        _log("undistort_model != pinhole but no K/D provided, skipping undistortion")

    # --- Stage 3: Quality scoring (GATE) ---
    _log("scoring frame quality")
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    qualities = np.zeros(len(frame_paths), dtype=np.float32)
    for i, fp in enumerate(frame_paths):
        frame = load_rgb(fp)
        qualities[i] = composite_quality(frame)
    quality_mask = filter_frames(qualities, cfg.quality_threshold)
    n_accepted = int(quality_mask.sum())
    _log(f"quality: {n_accepted}/{len(qualities)} frames accepted (threshold={cfg.quality_threshold})")

    # --- Stage 3b: Find contiguous accepted segments ---
    segments = find_accepted_segments(quality_mask, min_length=MIN_SEGMENT_FRAMES)
    if not segments:
        _log(f"no accepted segments >= {MIN_SEGMENT_FRAMES} frames, rejecting {clip_id}")
        return [ClipPipelineResult(
            clip_id=clip_id, n_frames=len(frame_paths), n_accepted=0,
            latent_path="", stages_skipped=skipped,
            errors=[f"all segments rejected (need >= {MIN_SEGMENT_FRAMES} contiguous accepted frames)"],
        )]
    _log(f"found {len(segments)} accepted segment(s): {segments}")

    # --- Stage 4-7: Process each segment ---
    results: list[ClipPipelineResult] = []
    for seg_idx, (seg_start, seg_end) in enumerate(segments):
        segment_id = f"{clip_id}_seg{seg_idx:02d}" if len(segments) > 1 else clip_id
        seg_frames_dir = Path(cfg.frames_dir) / segment_id

        seg_paths = _prepare_segment_frames(frame_paths, seg_start, seg_end, seg_frames_dir)
        seg_qualities = qualities[seg_start:seg_end]

        try:
            result = _process_segment(
                segment_id=segment_id,
                segment_frame_paths=seg_paths,
                segment_qualities=seg_qualities,
                cfg=cfg,
                cache_dir=cache_dir,
                skip_megasam=skip_megasam,
                track_persons=track_persons,
                device=device,
            )
            results.append(result)
        except Exception as exc:
            _log(f"  {segment_id}: FAILED: {exc}")
            results.append(ClipPipelineResult(
                clip_id=segment_id, n_frames=seg_end - seg_start, n_accepted=0,
                latent_path="", stages_skipped=skipped, errors=[str(exc)],
            ))

    return results


def process_batch(
    clips_yaml: str | Path,
    cfg: IngestConfig,
    *,
    skip_existing: bool = True,
    device: str = "cuda",
    track_persons: bool = False,
) -> list[ClipPipelineResult]:
    """Process all clips in a YAML clip list."""
    from vllatent.ingest.acquire import load_clips_yaml

    clips = load_clips_yaml(str(clips_yaml))
    results: list[ClipPipelineResult] = []
    cache_dir = Path(cfg.cache_dir)

    for clip_info in clips:
        clip_id = clip_info.get("clip_id", clip_info.get("id", ""))
        url = clip_info.get("url", "")
        if not clip_id or not url:
            _log(f"skipping clip entry missing clip_id or url: {clip_info}")
            continue

        if skip_existing and (cache_dir / f"{clip_id}.npz").exists():
            _log(f"skipping {clip_id}: already cached")
            continue

        clip_results = process_clip(
            url=url,
            clip_id=clip_id,
            cfg=cfg,
            device=device,
            track_persons=track_persons,
        )
        results.extend(clip_results)

        for r in clip_results:
            if r.errors:
                _log(f"WARNING: {r.clip_id} had errors: {r.errors}")
            else:
                _log(f"OK: {r.clip_id} -> {r.n_accepted}/{r.n_frames} frames")

    return results


def update_manifest_from_results(
    results: list[ClipPipelineResult],
    cfg: IngestConfig,
    encoder_model_id: str = "vit_base_patch16_dinov3.lvd1689m",
    person_tracker: dict[str, Any] | None = None,
) -> Path:
    """Build or update the manifest from pipeline results."""
    from vllatent.manifest import build_manifest_wild_video, validate_manifest, write_manifest

    cache_dir = Path(cfg.cache_dir)
    manifest_path = cache_dir / "manifest.json"

    existing_entries: list[dict[str, Any]] = []
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        existing_entries = existing.get("entries", [])

    existing_ids = {e.get("clip_id", e.get("episode_id", "")) for e in existing_entries}
    for r in results:
        if r.errors or not r.latent_path:
            continue
        if r.clip_id not in existing_ids:
            existing_entries.append({
                "clip_id": r.clip_id,
                "n_frames": r.n_frames,
                "latent_path": r.latent_path,
            })

    m = build_manifest_wild_video(
        encoder_model_id=encoder_model_id,
        motion_method="megasam",
        motion_model=cfg.megasam_model or "megasam_base",
        scale_mode="normalized",
        source_fps=cfg.target_fps,
        person_tracker=person_tracker,
        entries=existing_entries,
    )

    errs = validate_manifest(m)
    if errs:
        _log(f"manifest validation errors: {errs}")

    return write_manifest(m, str(cache_dir))


__all__ = [
    "ClipPipelineResult",
    "MIN_SEGMENT_FRAMES",
    "process_clip",
    "process_batch",
    "update_manifest_from_results",
]
