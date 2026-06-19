"""End-to-end per-clip processing pipeline (ORCH tier) — Phase B1 step 10.

Wires: acquire → extract → quality → megasam → ego_motion → encode → cache → manifest.

Each stage writes intermediate outputs to disk so the pipeline is resumable.
If an intermediate exists (e.g. frames already extracted), the stage is skipped.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vllatent.sports.config import SportsDataConfig


@dataclass(frozen=True)
class ClipPipelineResult:
    clip_id: str
    n_frames: int
    n_accepted: int
    latent_path: str
    stages_skipped: list[str]
    errors: list[str]


def _log(msg: str) -> None:
    print(f"[sports-pipeline] {msg}", file=sys.stderr)


def process_clip(
    *,
    url: str,
    clip_id: str,
    cfg: SportsDataConfig,
    skip_download: bool = False,
    skip_megasam: bool = False,
    device: str = "cuda",
) -> ClipPipelineResult:
    """Process a single clip end-to-end.

    Parameters
    ----------
    url : source video URL (or local path if skip_download=True)
    clip_id : unique clip identifier
    cfg : sports data config
    skip_download : if True, expect video already at raw_dir/clip_id.*
    skip_megasam : if True, expect MegaSaM output already at megasam_out_dir
    device : torch device for DINOv3 encoding
    """
    from vllatent.sports.acquire import download_clip, validate_clip
    from vllatent.sports.cache import build_clip_npz, write_clip_npz
    from vllatent.sports.ego_motion import normalize_scale, se3_sequence_to_deltas
    from vllatent.sports.encode import encode_frames
    from vllatent.sports.megasam import parse_megasam_output, run_megasam
    from vllatent.sports.preprocess import extract_frames
    from vllatent.sports.quality import composite_quality, filter_frames

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

    # --- Stage 3: Quality scoring ---
    _log("scoring frame quality")
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    qualities = np.zeros(len(frame_paths), dtype=np.float32)
    for i, fp in enumerate(frame_paths):
        frame = _load_rgb_for_quality(fp)
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

    # --- Stage 5: SE(3) → body-frame deltas ---
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

    arrays = build_clip_npz(
        latents=latents,
        deltas=deltas,
        vo_confidence=vo_conf,
        frame_quality=qualities,
        timestamps=timestamps,
        quality_mask=quality_mask,
    )

    latent_rel = f"{clip_id}.npz"
    write_clip_npz(arrays, cache_dir / latent_rel)
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
    cfg: SportsDataConfig,
    *,
    skip_existing: bool = True,
    device: str = "cuda",
) -> list[ClipPipelineResult]:
    """Process all clips in a YAML clip list."""
    from vllatent.sports.acquire import load_clips_yaml

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
            _log(f"OK: {clip_id} → {result.n_accepted}/{result.n_frames} frames accepted")

    return results


def update_manifest_from_results(
    results: list[ClipPipelineResult],
    cfg: SportsDataConfig,
    encoder_model_id: str = "vit_base_patch16_dinov3.lvd1689m",
) -> Path:
    """Build or update the manifest from pipeline results."""
    from vllatent.sports.cache import (
        build_sports_manifest,
        validate_sports_manifest,
        write_sports_manifest,
    )

    cache_dir = Path(cfg.cache_dir)
    manifest_path = cache_dir / "manifest.json"

    existing_entries: list[dict[str, Any]] = []
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        existing_entries = existing.get("entries", [])

    existing_ids = {e["clip_id"] for e in existing_entries}
    for r in results:
        if r.errors or not r.latent_path:
            continue
        if r.clip_id not in existing_ids:
            existing_entries.append({
                "clip_id": r.clip_id,
                "n_frames": r.n_frames,
                "latent_path": r.latent_path,
            })

    m = build_sports_manifest(
        encoder_model_id=encoder_model_id,
        sport=cfg.sport,
        megasam_model=cfg.megasam_model or "megasam_base",
        scale_mode="normalized",
        source_fps=cfg.target_fps,
        entries=existing_entries,
    )

    errs = validate_sports_manifest(m)
    if errs:
        _log(f"manifest validation errors: {errs}")

    return write_sports_manifest(m, str(cache_dir))


def _load_rgb_for_quality(path: Path) -> np.ndarray:
    """Load a frame as RGB uint8 for quality scoring."""
    try:
        import cv2
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is not None:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except ImportError:
        pass
    from PIL import Image
    return np.array(Image.open(str(path)).convert("RGB"))


__all__ = [
    "ClipPipelineResult",
    "process_clip",
    "process_batch",
    "update_manifest_from_results",
]
