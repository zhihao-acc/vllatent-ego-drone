"""Person-track labels and data screens for B3 human-conditioned world models.

The core helpers are numpy-only and fixture-testable. YOLO-World/ByteTrack is
loaded lazily by ``track_persons_from_paths`` so this module still imports in the
pure lane.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PERSON_BBOX_KEY = "person_bbox"
PERSON_VISIBLE_KEY = "person_visible"
PERSON_CONF_KEY = "person_conf"
PERSON_BBOX_DIM = 4
PERSON_TRACK_CLASSES = ("person", "skier", "snowboarder")
PERSON_TRACKER_ID = "yolov8s-worldv2.pt+bytetrack"
PERSON_TRACK_CONFIDENCE = 0.15


@dataclass(frozen=True)
class TrackedDetection:
    """One tracked person-like detection in pixel xyxy coordinates."""

    frame_idx: int
    track_id: int
    xyxy: np.ndarray
    confidence: float


@dataclass(frozen=True)
class PersonTrackResult:
    """Per-frame subject track labels in normalized cxcywh coordinates."""

    person_bbox: np.ndarray
    person_visible: np.ndarray
    person_conf: np.ndarray
    provenance: dict[str, Any]


@dataclass(frozen=True)
class ScreenReport:
    """Clip-level and window-level data-screen counts."""

    n_frames: int
    n_windows: int
    duplicate_frame_runs: int
    time_remap_flags: int
    accel_outlier_frames: int
    person_visible_frames: int
    person_valid_windows: int


def empty_person_tracks(n_frames: int) -> PersonTrackResult:
    """Return invisible-person defaults for old caches or detector failures."""
    return PersonTrackResult(
        person_bbox=np.zeros((n_frames, PERSON_BBOX_DIM), dtype=np.float32),
        person_visible=np.zeros(n_frames, dtype=np.bool_),
        person_conf=np.zeros(n_frames, dtype=np.float32),
        provenance={
            "detector": "none",
            "tracker": "none",
            "classes": list(PERSON_TRACK_CLASSES),
            "fallback": "invisible_defaults",
        },
    )


def validate_person_track_arrays(
    *,
    n_frames: int,
    person_bbox: np.ndarray,
    person_visible: np.ndarray,
    person_conf: np.ndarray,
) -> None:
    """Validate B3 cache person-track arrays."""
    if person_bbox.shape != (n_frames, PERSON_BBOX_DIM):
        raise ValueError(f"person_bbox: expected ({n_frames}, {PERSON_BBOX_DIM}), got {person_bbox.shape}")
    if person_visible.shape != (n_frames,):
        raise ValueError(f"person_visible: expected ({n_frames},), got {person_visible.shape}")
    if person_conf.shape != (n_frames,):
        raise ValueError(f"person_conf: expected ({n_frames},), got {person_conf.shape}")
    if not np.all(np.isfinite(person_bbox)):
        raise ValueError("person_bbox contains non-finite values")
    if not np.all(np.isfinite(person_conf)):
        raise ValueError("person_conf contains non-finite values")


def person_tracks_from_cache(clip: dict[str, np.ndarray]) -> PersonTrackResult:
    """Read person-track labels from a loaded .npz dict, with old-cache fallback."""
    n_frames = int(clip["latents"].shape[0])
    if (
        PERSON_BBOX_KEY not in clip
        or PERSON_VISIBLE_KEY not in clip
        or PERSON_CONF_KEY not in clip
    ):
        return empty_person_tracks(n_frames)

    bbox = np.asarray(clip[PERSON_BBOX_KEY], dtype=np.float32)
    visible = np.asarray(clip[PERSON_VISIBLE_KEY]).astype(np.bool_)
    conf = np.asarray(clip[PERSON_CONF_KEY], dtype=np.float32)
    validate_person_track_arrays(
        n_frames=n_frames,
        person_bbox=bbox,
        person_visible=visible,
        person_conf=conf,
    )
    return PersonTrackResult(
        person_bbox=bbox,
        person_visible=visible,
        person_conf=conf,
        provenance={"source": "cache"},
    )


def _xyxy_to_norm_cxcywh(xyxy: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    h, w = image_hw
    x1, y1, x2, y2 = xyxy.astype(np.float32)
    x1 = float(np.clip(x1, 0, w))
    x2 = float(np.clip(x2, 0, w))
    y1 = float(np.clip(y1, 0, h))
    y2 = float(np.clip(y2, 0, h))
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = x1 + 0.5 * bw
    cy = y1 + 0.5 * bh
    return np.array([cx / w, cy / h, bw / w, bh / h], dtype=np.float32)


def select_subject_track(
    detections: list[TrackedDetection],
    *,
    n_frames: int,
    image_hw: tuple[int, int],
) -> PersonTrackResult:
    """Select the longest track, breaking ties by centrality and size."""
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    if not detections:
        return empty_person_tracks(n_frames)

    by_track: dict[int, list[TrackedDetection]] = {}
    for det in detections:
        if 0 <= det.frame_idx < n_frames:
            by_track.setdefault(det.track_id, []).append(det)
    if not by_track:
        return empty_person_tracks(n_frames)

    def _score(items: list[TrackedDetection]) -> tuple[int, float, float]:
        boxes = np.stack([_xyxy_to_norm_cxcywh(d.xyxy, image_hw) for d in items])
        centers = boxes[:, :2]
        centrality = -float(np.mean(np.sum((centers - 0.5) ** 2, axis=1)))
        area = float(np.mean(boxes[:, 2] * boxes[:, 3]))
        return (len(items), centrality, area)

    selected_id, selected = max(by_track.items(), key=lambda kv: _score(kv[1]))
    _ = selected_id

    bbox = np.zeros((n_frames, PERSON_BBOX_DIM), dtype=np.float32)
    visible = np.zeros(n_frames, dtype=np.bool_)
    conf = np.zeros(n_frames, dtype=np.float32)
    for det in selected:
        bbox[det.frame_idx] = _xyxy_to_norm_cxcywh(det.xyxy, image_hw)
        visible[det.frame_idx] = True
        conf[det.frame_idx] = np.float32(det.confidence)

    return PersonTrackResult(
        person_bbox=bbox,
        person_visible=visible,
        person_conf=conf,
        provenance={
            "detector": PERSON_TRACKER_ID,
            "tracker": "bytetrack",
            "classes": list(PERSON_TRACK_CLASSES),
            "selection": "longest_then_central_then_largest",
        },
    )


def _detections_from_ultralytics_results(results: list[Any]) -> list[TrackedDetection]:
    detections: list[TrackedDetection] = []
    for frame_idx, result in enumerate(results):
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.detach().cpu().numpy()
        conf = boxes.conf.detach().cpu().numpy()
        ids = boxes.id
        if ids is None:
            ids_np = np.arange(len(xyxy), dtype=np.int64)
        else:
            ids_np = ids.detach().cpu().numpy().astype(np.int64)
        for box, score, track_id in zip(xyxy, conf, ids_np, strict=False):
            detections.append(
                TrackedDetection(
                    frame_idx=frame_idx,
                    track_id=int(track_id),
                    xyxy=np.asarray(box, dtype=np.float32),
                    confidence=float(score),
                )
            )
    return detections


def track_persons_from_paths(
    frame_paths: list[str | Path],
    *,
    device: str = "cpu",
    model_id: str = "yolov8s-worldv2.pt",
    confidence: float = PERSON_TRACK_CONFIDENCE,
    classes: tuple[str, ...] = PERSON_TRACK_CLASSES,
) -> PersonTrackResult:
    """Run YOLO-World/ByteTrack on an ordered frame sequence.

    This is intended for fixture/dry-run tests and user-gated full backfills.
    Heavy dependencies are imported inside the function.
    """
    if not frame_paths:
        raise ValueError("frame_paths: expected a non-empty list")

    from PIL import Image
    from ultralytics import YOLOWorld

    paths = [Path(p) for p in frame_paths]
    first = Image.open(paths[0])
    image_hw = (int(first.height), int(first.width))

    model = YOLOWorld(model_id)
    model.to(device)
    model.set_classes(list(classes))
    results = model.track(
        source=[str(p) for p in paths],
        tracker="bytetrack.yaml",
        persist=True,
        conf=confidence,
        verbose=False,
        stream=False,
    )
    detections = _detections_from_ultralytics_results(list(results))
    result = select_subject_track(detections, n_frames=len(paths), image_hw=image_hw)
    return PersonTrackResult(
        person_bbox=result.person_bbox,
        person_visible=result.person_visible,
        person_conf=result.person_conf,
        provenance={
            **result.provenance,
            "detector": model_id,
            "confidence": confidence,
            "classes": list(classes),
        },
    )


def person_state_from_bbox(
    person_bbox: np.ndarray,
    person_visible: np.ndarray,
) -> np.ndarray:
    """Convert normalized cxcywh labels to B3 person state cx,cy,log_h,visibility."""
    bbox = np.asarray(person_bbox, dtype=np.float32)
    visible = np.asarray(person_visible).astype(np.bool_)
    state = np.zeros((bbox.shape[0], 4), dtype=np.float32)
    state[:, 0:2] = bbox[:, 0:2]
    state[:, 2] = np.log(np.clip(bbox[:, 3], 1e-6, 1.0)).astype(np.float32)
    state[:, 3] = visible.astype(np.float32)
    state[~visible, 0:3] = 0.0
    return state


def time_remap_flags_from_deltas(
    deltas: np.ndarray,
    *,
    window: int = 10,
    jump_ratio: float = 1.5,
) -> np.ndarray:
    """Flag speed-ramp-like windows from abrupt robust log-speed jumps."""
    if deltas.size == 0:
        return np.zeros(0, dtype=np.bool_)
    speed = np.linalg.norm(deltas[:, :3].astype(np.float64), axis=1)
    log_speed = np.log(np.clip(speed, 1e-8, None))
    flags = np.zeros(len(speed), dtype=np.bool_)
    if len(speed) < 3:
        return flags
    jump = abs(np.log(jump_ratio))
    diffs = np.abs(np.diff(log_speed, prepend=log_speed[0]))
    flags |= diffs > jump
    if len(speed) >= window:
        for i in range(0, len(speed) - window + 1):
            segment = log_speed[i : i + window]
            if float(np.max(segment) - np.min(segment)) > jump:
                flags[i : i + window] = True
    return flags


def duplicate_frame_runs_from_latents(
    latents: np.ndarray,
    *,
    cosine_distance_threshold: float = 1e-5,
) -> np.ndarray:
    """Flag frames whose cached latent is nearly identical to its predecessor."""
    n = int(latents.shape[0])
    flags = np.zeros(n, dtype=np.bool_)
    if n < 2:
        return flags
    flat = latents.reshape(n, -1).astype(np.float32)
    norms = np.linalg.norm(flat, axis=1)
    denom = np.clip(norms[1:] * norms[:-1], 1e-8, None)
    cos = np.sum(flat[1:] * flat[:-1], axis=1) / denom
    flags[1:] = (1.0 - cos) <= cosine_distance_threshold
    return flags


def accel_outlier_flags_from_deltas(
    deltas: np.ndarray,
    *,
    threshold_mad: float = 3.0,
) -> np.ndarray:
    """MAD-robust acceleration outlier flags on translation deltas."""
    n = int(deltas.shape[0])
    flags = np.zeros(n, dtype=np.bool_)
    if n < 3:
        return flags
    accel = np.diff(deltas[:, :3].astype(np.float64), axis=0)
    mag = np.linalg.norm(accel, axis=1)
    med = float(np.median(mag))
    mad = float(np.median(np.abs(mag - med)))
    scale = 1.4826 * mad if mad > 1e-8 else 1e-8
    flags[1:] = mag > (med + threshold_mad * scale)
    return flags


def screen_clip_arrays(
    *,
    latents: np.ndarray,
    deltas: np.ndarray,
    person_visible: np.ndarray,
    history: int,
    horizon: int,
) -> ScreenReport:
    """Compute B3 data-screen counts for one cached clip."""
    n_frames = int(latents.shape[0])
    duplicate = duplicate_frame_runs_from_latents(latents)
    time_remap = time_remap_flags_from_deltas(deltas)
    accel = accel_outlier_flags_from_deltas(deltas)
    n_windows = max(0, n_frames - horizon)
    person_valid_windows = 0
    visible = np.asarray(person_visible).astype(np.bool_)
    for t in range(n_windows):
        hist = visible[max(0, t - history + 1) : t + 1]
        fut = visible[t + 1 : t + 1 + horizon]
        hist_ok = hist.size > 0 and float(np.mean(hist)) >= (2.0 / 3.0)
        fut_ok = fut.size > 0 and float(np.mean(fut)) >= 0.5
        if hist_ok and fut_ok:
            person_valid_windows += 1
    return ScreenReport(
        n_frames=n_frames,
        n_windows=n_windows,
        duplicate_frame_runs=int(np.sum(duplicate)),
        time_remap_flags=int(np.sum(time_remap)),
        accel_outlier_frames=int(np.sum(accel)),
        person_visible_frames=int(np.sum(visible)),
        person_valid_windows=person_valid_windows,
    )


def _load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(str(path)) as data:
        return {k: data[k] for k in data.files}


def screen_cache_dir(
    cache_dir: str | Path,
    *,
    history: int,
    horizon: int,
    limit: int | None = None,
) -> dict[str, Any]:
    """Screen cached clips and return clip/window/source counts plus flags."""
    paths = sorted(Path(cache_dir).glob("*.npz"))
    if limit is not None:
        paths = paths[:limit]

    clips: list[dict[str, Any]] = []
    sources: dict[str, dict[str, int]] = {}
    totals = {
        "clips": 0,
        "windows": 0,
        "sources": 0,
        "person_valid_windows": 0,
        "person_visible_frames": 0,
        "duplicate_frame_runs": 0,
        "time_remap_flags": 0,
        "accel_outlier_frames": 0,
        "flagged_clips": 0,
    }

    for path in paths:
        arrays = _load_npz_arrays(path)
        tracks = person_tracks_from_cache(arrays)
        report = screen_clip_arrays(
            latents=arrays["latents"],
            deltas=arrays["deltas"],
            person_visible=tracks.person_visible,
            history=history,
            horizon=horizon,
        )
        source = path.stem.split("_")[0]
        source_entry = sources.setdefault(source, {"clips": 0, "windows": 0, "person_valid_windows": 0})
        source_entry["clips"] += 1
        source_entry["windows"] += report.n_windows
        source_entry["person_valid_windows"] += report.person_valid_windows

        flags: list[str] = []
        if report.duplicate_frame_runs:
            flags.append("duplicate_frames")
        if report.time_remap_flags:
            flags.append("time_remap")
        if report.accel_outlier_frames:
            flags.append("accel_outliers")
        if report.n_windows and report.person_valid_windows == 0:
            flags.append("person_absent_windows")

        clip_record = {
            "clip_id": path.stem,
            "source": source,
            "n_frames": report.n_frames,
            "n_windows": report.n_windows,
            "person_valid_windows": report.person_valid_windows,
            "person_visible_frames": report.person_visible_frames,
            "duplicate_frame_runs": report.duplicate_frame_runs,
            "time_remap_flags": report.time_remap_flags,
            "accel_outlier_frames": report.accel_outlier_frames,
            "flags": flags,
        }
        clips.append(clip_record)

        totals["clips"] += 1
        totals["windows"] += report.n_windows
        totals["person_valid_windows"] += report.person_valid_windows
        totals["person_visible_frames"] += report.person_visible_frames
        totals["duplicate_frame_runs"] += report.duplicate_frame_runs
        totals["time_remap_flags"] += report.time_remap_flags
        totals["accel_outlier_frames"] += report.accel_outlier_frames
        totals["flagged_clips"] += int(bool(flags))

    totals["sources"] = len(sources)
    return {
        "totals": totals,
        "sources": sources,
        "clips": clips,
    }


__all__ = [
    "PERSON_BBOX_DIM",
    "PERSON_BBOX_KEY",
    "PERSON_CONF_KEY",
    "PERSON_TRACK_CLASSES",
    "PERSON_TRACKER_ID",
    "PERSON_VISIBLE_KEY",
    "PersonTrackResult",
    "ScreenReport",
    "TrackedDetection",
    "accel_outlier_flags_from_deltas",
    "duplicate_frame_runs_from_latents",
    "empty_person_tracks",
    "person_state_from_bbox",
    "person_tracks_from_cache",
    "screen_clip_arrays",
    "screen_cache_dir",
    "select_subject_track",
    "time_remap_flags_from_deltas",
    "track_persons_from_paths",
    "validate_person_track_arrays",
]
