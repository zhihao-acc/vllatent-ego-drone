"""End-to-end per-clip processing pipeline (ORCH tier).

Wires: acquire -> extract -> quality -> megasam -> ego_motion -> encode -> cache -> manifest.

Each stage writes intermediate outputs to disk so the pipeline is resumable.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vllatent.config import IngestConfig
from vllatent.io import load_rgb
from vllatent.schemas import DELTA_DTYPE, EMBED_DIM, LATENT_DTYPE, MASK_DTYPE, PATCH_TOKENS


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
    quality_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Validate and return the arrays dict for a single clip's .npz."""
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
    if quality_mask.shape != (n,):
        raise ValueError(f"quality_mask: expected ({n},), got {quality_mask.shape}")

    return {
        "latents": latents.astype(LATENT_DTYPE),
        "deltas": deltas.astype(DELTA_DTYPE),
        "vo_confidence": vo_confidence.astype(np.float32),
        "frame_quality": frame_quality.astype(np.float32),
        "timestamps": timestamps.astype(np.float64),
        "quality_mask": quality_mask.astype(MASK_DTYPE),
    }


def _write_clip_npz(
    arrays: dict[str, np.ndarray],
    out_path: str | Path,
) -> Path:
    """Write a clip's arrays to a .npz file."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(p), **arrays)
    return p


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
) -> ClipPipelineResult:
    """Process a single clip end-to-end."""
    from vllatent.encode.batch import encode_frames
    from vllatent.ingest.acquire import download_clip, validate_clip
    from vllatent.ingest.ego_motion import normalize_scale, se3_sequence_to_deltas
    from vllatent.ingest.megasam import parse_megasam_output, run_megasam
    from vllatent.ingest.preprocess import batch_undistort, extract_frames
    from vllatent.ingest.quality import composite_quality, filter_frames

    errors: list[str] = []
    skipped: list[str] = []

    raw_dir = Path(cfg.raw_dir)
    frames_dir = Path(cfg.frames_dir) / clip_id
    cache_dir = Path(cfg.cache_dir)
    megasam_out = Path(cfg.frames_dir) / f"{clip_id}_megasam"

    # --- Stage 1: Download ---
    if skip_download:
        skipped.append("download")
        candidates = list(raw_dir.glob(f"{clip_id}.*"))
        if not candidates:
            return ClipPipelineResult(
                clip_id=clip_id, n_frames=0, n_accepted=0,
                latent_path="", stages_skipped=skipped,
                errors=[f"No video found for {clip_id} in {raw_dir}"],
            )
        video_path = candidates[0]
    else:
        _log(f"downloading {clip_id}")
        raw_dir.mkdir(parents=True, exist_ok=True)
        meta = download_clip(url, str(raw_dir), clip_id=clip_id, max_height=cfg.resolution_h)
        video_path = meta.path
        if not validate_clip(video_path):
            errors.append(f"downloaded video failed validation: {video_path}")
            return ClipPipelineResult(
                clip_id=clip_id, n_frames=0, n_accepted=0,
                latent_path="", stages_skipped=skipped, errors=errors,
            )

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
    if n_frames_on_disk < 2:
        errors.append(f"too few frames extracted: {n_frames_on_disk}")
        return ClipPipelineResult(
            clip_id=clip_id, n_frames=n_frames_on_disk, n_accepted=0,
            latent_path="", stages_skipped=skipped, errors=errors,
        )

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

    # --- Stage 3: Quality scoring ---
    _log("scoring frame quality")
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    qualities = np.zeros(len(frame_paths), dtype=np.float32)
    for i, fp in enumerate(frame_paths):
        frame = load_rgb(fp)
        qualities[i] = composite_quality(frame)
    quality_mask = filter_frames(qualities, cfg.quality_threshold)
    n_accepted = int(quality_mask.sum())
    _log(f"quality: {n_accepted}/{len(qualities)} frames accepted (threshold={cfg.quality_threshold})")

    # --- Stage 4: MegaSaM ego-motion ---
    if skip_megasam:
        skipped.append("megasam")
        _log(f"using existing MegaSaM output at {megasam_out}")
    else:
        _log("running MegaSaM ego-motion extraction")
        megasam_model = cfg.megasam_model if cfg.megasam_model else "megasam_base"
        run_megasam(str(frames_dir), str(megasam_out), model=megasam_model)

    megasam_result = parse_megasam_output(megasam_out)

    # --- Stage 5: SE(3) -> body-frame deltas ---
    _log("converting SE(3) poses to body-frame deltas")
    deltas = se3_sequence_to_deltas(megasam_result.poses, fps=cfg.target_fps)
    deltas = normalize_scale(deltas, mode="median_speed")

    n_pose_frames = megasam_result.poses.shape[0]
    if n_pose_frames != len(frame_paths):
        _log(f"WARNING: pose count ({n_pose_frames}) != frame count ({len(frame_paths)}), truncating")
        n = min(n_pose_frames, len(frame_paths))
        deltas = deltas[:n - 1]
        qualities = qualities[:n]
        quality_mask = quality_mask[:n]
        n_accepted = int(quality_mask.sum())

    # --- Stage 6: DINOv3 encoding ---
    _log("encoding frames with DINOv3")
    latents = encode_frames(frames_dir, device=device)

    n_latent = latents.shape[0]
    n = min(n_latent, len(qualities), deltas.shape[0] + 1)
    latents = latents[:n]
    deltas = deltas[:n - 1]
    qualities = qualities[:n]
    quality_mask = quality_mask[:n]

    # --- Stage 7: Cache assembly ---
    _log(f"assembling cache for {clip_id}: {n} frames")
    vo_conf = megasam_result.confidences[:n].astype(np.float32)
    timestamps = np.arange(n, dtype=np.float64) / cfg.target_fps

    arrays = _build_clip_npz(
        latents=latents,
        deltas=deltas,
        vo_confidence=vo_conf,
        frame_quality=qualities,
        timestamps=timestamps,
        quality_mask=quality_mask,
    )

    latent_rel = f"{clip_id}.npz"
    _write_clip_npz(arrays, cache_dir / latent_rel)
    _log(f"wrote {cache_dir / latent_rel}")

    return ClipPipelineResult(
        clip_id=clip_id,
        n_frames=n,
        n_accepted=int(quality_mask.sum()),
        latent_path=latent_rel,
        stages_skipped=skipped,
        errors=errors,
    )


def process_batch(
    clips_yaml: str | Path,
    cfg: IngestConfig,
    *,
    skip_existing: bool = True,
    device: str = "cuda",
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

        result = process_clip(url=url, clip_id=clip_id, cfg=cfg, device=device)
        results.append(result)

        if result.errors:
            _log(f"WARNING: {clip_id} had errors: {result.errors}")
        else:
            _log(f"OK: {clip_id} -> {result.n_accepted}/{result.n_frames} frames accepted")

    return results


def update_manifest_from_results(
    results: list[ClipPipelineResult],
    cfg: IngestConfig,
    encoder_model_id: str = "vit_base_patch16_dinov3.lvd1689m",
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
        entries=existing_entries,
    )

    errs = validate_manifest(m)
    if errs:
        _log(f"manifest validation errors: {errs}")

    return write_manifest(m, str(cache_dir))


__all__ = [
    "ClipPipelineResult",
    "process_clip",
    "process_batch",
    "update_manifest_from_results",
]
