"""Motion + YOLO-World open-vocabulary content filter + PySceneDetect shot boundaries (B1.7c).

Two-signal FPV filter for sports footage:
1. **Motion** (primary): frame-to-frame pixel difference rejects static/product shots.
2. **YOLO-World** (semantic): open-vocabulary object detection rejects frames containing
   drones, cameras, gear, text overlays, etc. — objects that should never appear in
   first-person training data.

Shot boundaries detected via PySceneDetect AdaptiveDetector. Per-shot majority vote
produces a whole-video verdict (ACCEPT / PARTIAL / REJECT).

**Tier: TORCH** (YOLO inference). All heavy imports (torch, ultralytics, scenedetect)
are LAZY — inside functions — so the module imports on a torch-free box (pure CI lane).
"""
from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

# --- Rejected object classes (YOLO-World open-vocabulary) ---

REJECTED_CLASSES: list[str] = [
    # --- drone body + parts ---
    "drone",
    "quadcopter",
    "hexacopter",
    "octocopter",
    "multirotor",
    "rotor",
    "propeller",
    "drone arm",
    "drone motor",
    "landing gear",
    "gimbal",
    "flight controller",
    "drone battery",
    "RC controller",
    "remote controller",
    # --- camera + filming gear ---
    "camera",
    "GoPro",
    "action camera",
    "camera lens",
    "tripod",
    "monopod",
    "stabilizer",
    "microphone",
    "selfie stick",
    # --- electronics / non-FPV ---
    "laptop",
    "monitor",
    "television",
    "phone screen",
    # --- overlays / graphics ---
    "text overlay",
    "title card",
    "logo",
    "watermark",
    "subtitle",
]

_YOLO_CONFIDENCE_THRESHOLD = 0.15
_YOLO_MODEL_ID = "yolov8s-worldv2.pt"

# Thresholds
_ACCEPT_SHOT_FRAC = 0.60   # >= 60% FPV shots => ACCEPT whole video
_REJECT_SHOT_FRAC = 0.30   # < 30% FPV shots => REJECT whole video
_MIN_SEGMENT_FRAMES = 10   # 2s at 5fps — discard shorter accepted runs


class VideoVerdict(enum.Enum):
    ACCEPT = "accept"
    PARTIAL = "partial"
    REJECT = "reject"


@dataclass(frozen=True)
class ShotInfo:
    start: int
    end: int
    is_fpv: bool
    mean_score: float


@dataclass(frozen=True)
class ShotClassification:
    shots: list[ShotInfo]
    n_shots: int
    n_fpv: int
    fpv_fraction: float


@dataclass(frozen=True)
class FilterResult:
    verdict: VideoVerdict
    n_frames: int
    n_fpv_frames: int
    fpv_mask: np.ndarray
    shot_boundaries: list[int]
    shots: list[ShotInfo]
    per_frame_scores: np.ndarray
    # Diagnostic per-frame signals (the "why" behind each reject) — optional so
    # older constructions stay valid; populated by the filter_video* functions.
    motion_scores: np.ndarray | None = None
    rejected_objects: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Shot boundary detection
# ---------------------------------------------------------------------------


def detect_shot_boundaries(
    frames: list[np.ndarray],
    *,
    adaptive_threshold: float = 2.0,
    min_scene_len: int = 2,
    fps: float = 30.0,
) -> list[int]:
    """Detect shot boundaries using PySceneDetect AdaptiveDetector on in-memory frames.

    Returns a sorted list of frame indices where shot transitions occur.
    """
    if not frames:
        raise ValueError("frames: expected a non-empty list of RGB arrays")

    from scenedetect import AdaptiveDetector, FrameTimecode

    detector = AdaptiveDetector(
        adaptive_threshold=adaptive_threshold,
        min_scene_len=min_scene_len,
    )
    boundaries: list[int] = []

    for i, frame in enumerate(frames):
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"frame {i}: expected (H,W,3) RGB, got shape {frame.shape}")

        tc = FrameTimecode(i, fps=fps)
        cuts = detector.process_frame(tc, frame)
        for cut in cuts:
            boundaries.append(cut.frame_num)

    boundaries.sort()
    return boundaries


def detect_shot_boundaries_from_paths(
    frame_paths: list,
    *,
    adaptive_threshold: float = 2.0,
    min_scene_len: int = 2,
    fps: float = 30.0,
) -> list[int]:
    """Detect shot boundaries by loading frames one at a time from file paths."""
    if not frame_paths:
        raise ValueError("frame_paths: expected a non-empty list of paths")

    from PIL import Image
    from scenedetect import AdaptiveDetector, FrameTimecode

    detector = AdaptiveDetector(
        adaptive_threshold=adaptive_threshold,
        min_scene_len=min_scene_len,
    )
    boundaries: list[int] = []

    for i, path in enumerate(frame_paths):
        frame = np.array(Image.open(path))
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"frame {i}: expected (H,W,3) RGB, got shape {frame.shape}")

        tc = FrameTimecode(i, fps=fps)
        cuts = detector.process_frame(tc, frame)
        for cut in cuts:
            boundaries.append(cut.frame_num)

    boundaries.sort()
    return boundaries


# ---------------------------------------------------------------------------
# YOLO-World open-vocabulary object detection
# ---------------------------------------------------------------------------


def _get_yolo_detector(
    device: str = "cpu",
    confidence: float = _YOLO_CONFIDENCE_THRESHOLD,
) -> Callable[[list[np.ndarray]], np.ndarray]:
    """Build a YOLO-World detector that flags frames containing rejected objects.

    Lazy-loads ultralytics. Text embeddings are computed once via set_classes().
    Returns a callable: list[np.ndarray] → np.ndarray[bool] (True = rejected).
    """
    from ultralytics import YOLOWorld

    model = YOLOWorld(_YOLO_MODEL_ID)
    model.to(device)
    model.set_classes(REJECTED_CLASSES)

    def _detect_batch(frames_batch: list[np.ndarray]) -> np.ndarray:
        from PIL import Image

        images = [Image.fromarray(f) for f in frames_batch]
        results = model.predict(images, conf=confidence, verbose=False)
        rejected = np.zeros(len(frames_batch), dtype=np.bool_)
        for i, result in enumerate(results):
            if len(result.boxes) > 0:
                rejected[i] = True
        return rejected

    return _detect_batch


def detect_rejected_objects(
    frames: list[np.ndarray],
    *,
    device: str = "cpu",
    batch_size: int = 32,
    confidence: float = _YOLO_CONFIDENCE_THRESHOLD,
) -> np.ndarray:
    """Detect rejected objects in frames using YOLO-World.

    Returns per-frame boolean mask: True = frame contains a rejected object.
    """
    if not frames:
        raise ValueError("frames: expected a non-empty list of RGB arrays")

    detector = _get_yolo_detector(device=device, confidence=confidence)

    all_rejected: list[np.ndarray] = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i : i + batch_size]
        rejected = detector(batch)
        all_rejected.append(rejected)

    return np.concatenate(all_rejected)


def detect_rejected_objects_from_paths(
    frame_paths: list,
    *,
    device: str = "cpu",
    batch_size: int = 32,
    confidence: float = _YOLO_CONFIDENCE_THRESHOLD,
) -> np.ndarray:
    """Detect rejected objects by loading frames from file paths in bounded batches."""
    if not frame_paths:
        raise ValueError("frame_paths: expected a non-empty list of paths")

    from PIL import Image

    detector = _get_yolo_detector(device=device, confidence=confidence)

    all_rejected: list[np.ndarray] = []
    for i in range(0, len(frame_paths), batch_size):
        batch_paths = frame_paths[i : i + batch_size]
        batch_frames = [np.array(Image.open(p)) for p in batch_paths]
        rejected = detector(batch_frames)
        all_rejected.append(rejected)

    return np.concatenate(all_rejected)


# ---------------------------------------------------------------------------
# Per-shot classification
# ---------------------------------------------------------------------------


def classify_shots(
    scores: np.ndarray,
    boundaries: list[int],
    *,
    threshold: float = 0.65,
) -> ShotClassification:
    """Classify each shot as FPV or non-FPV via majority vote on per-frame scores."""
    n = len(scores)
    cuts = sorted(set([0] + boundaries + [n]))
    if cuts[0] != 0:
        cuts = [0] + cuts
    if cuts[-1] != n:
        cuts.append(n)

    shots: list[ShotInfo] = []
    for i in range(len(cuts) - 1):
        start, end = cuts[i], cuts[i + 1]
        if start >= end:
            continue
        shot_scores = scores[start:end]
        n_fpv_frames = int(np.sum(shot_scores >= threshold))
        is_fpv = n_fpv_frames > len(shot_scores) / 2
        mean_score = float(np.mean(shot_scores))
        shots.append(ShotInfo(start=start, end=end, is_fpv=is_fpv, mean_score=mean_score))

    n_fpv = sum(1 for s in shots if s.is_fpv)
    fpv_frac = n_fpv / len(shots) if shots else 0.0

    return ShotClassification(
        shots=shots,
        n_shots=len(shots),
        n_fpv=n_fpv,
        fpv_fraction=fpv_frac,
    )


# ---------------------------------------------------------------------------
# Whole-video verdict
# ---------------------------------------------------------------------------


def video_verdict(classification: ShotClassification) -> VideoVerdict:
    """Classify whole video: ACCEPT (>=60% FPV), REJECT (<30%), else PARTIAL."""
    if classification.fpv_fraction >= _ACCEPT_SHOT_FRAC:
        return VideoVerdict.ACCEPT
    if classification.fpv_fraction < _REJECT_SHOT_FRAC:
        return VideoVerdict.REJECT
    return VideoVerdict.PARTIAL


# ---------------------------------------------------------------------------
# Per-frame FPV mask
# ---------------------------------------------------------------------------


def fpv_frame_mask(classification: ShotClassification) -> np.ndarray:
    """Build a per-frame boolean mask from shot classification."""
    if not classification.shots:
        return np.array([], dtype=np.bool_)

    total = max(s.end for s in classification.shots)
    mask = np.zeros(total, dtype=np.bool_)
    for shot in classification.shots:
        if shot.is_fpv:
            mask[shot.start : shot.end] = True
    return mask


# ---------------------------------------------------------------------------
# Thumbnail grid data
# ---------------------------------------------------------------------------


def thumbnail_grid_data(
    frames: list[np.ndarray],
    classification: ShotClassification,
    *,
    max_thumbs: int = 12,
) -> list[dict[str, Any]]:
    """Select representative frames for a thumbnail grid with accept/reject labels."""
    entries: list[dict[str, Any]] = []

    for shot in classification.shots:
        mid = (shot.start + shot.end) // 2
        if mid < len(frames):
            entries.append({
                "frame_idx": mid,
                "is_fpv": shot.is_fpv,
                "score": shot.mean_score,
                "frame": frames[mid],
            })

    if len(entries) > max_thumbs:
        step = len(entries) / max_thumbs
        entries = [entries[int(i * step)] for i in range(max_thumbs)]

    return entries


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def extract_fpv_ranges(
    shots: list[ShotInfo],
    fpv_mask: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Return contiguous FPV frame ranges, respecting both shot and frame verdicts.

    Shot boundaries are editing cuts — frames on either side are NOT
    temporally continuous. MegaSaM needs continuous sequences, so we
    must NOT merge across shot boundaries even when consecutive shots
    are both FPV.

    When *fpv_mask* is provided (per-frame bool from motion + YOLO), each FPV
    shot is further split at frame-level rejections. This prevents individual
    non-FPV frames (YOLO-detected gear, static frames) from leaking into
    sub-clips that MegaSaM and DINOv3 process.
    """
    if fpv_mask is None:
        return [(shot.start, shot.end) for shot in shots if shot.is_fpv]

    ranges: list[tuple[int, int]] = []
    for shot in shots:
        if not shot.is_fpv:
            continue
        i = shot.start
        while i < shot.end:
            if fpv_mask[i]:
                run_start = i
                while i < shot.end and fpv_mask[i]:
                    i += 1
                ranges.append((run_start, i))
            else:
                i += 1
    return ranges


def compute_motion_scores(frame_paths: list, *, downsample: int = 4) -> np.ndarray:
    """Per-frame motion score: mean absolute pixel difference vs previous frame.

    Returns (N,) float32. First frame gets score 0 (no predecessor).
    Loads two frames at a time — O(1) memory. ``downsample`` shrinks frames
    before differencing to save compute (default 4x).
    """
    if not frame_paths:
        raise ValueError("frame_paths: expected a non-empty list of paths")

    from PIL import Image

    scores = np.zeros(len(frame_paths), dtype=np.float32)
    prev: np.ndarray | None = None

    for i, path in enumerate(frame_paths):
        frame = np.array(Image.open(path))
        if downsample > 1:
            frame = frame[::downsample, ::downsample]
        if prev is not None:
            scores[i] = np.mean(np.abs(frame.astype(np.float32) - prev.astype(np.float32)))
        prev = frame

    return scores


def _compute_motion_from_arrays(frames: list[np.ndarray], *, downsample: int = 4) -> np.ndarray:
    """Per-frame motion score from in-memory arrays (no disk I/O)."""
    scores = np.zeros(len(frames), dtype=np.float32)
    prev: np.ndarray | None = None

    for i, frame in enumerate(frames):
        small = frame[::downsample, ::downsample] if downsample > 1 else frame
        if prev is not None:
            scores[i] = np.mean(np.abs(small.astype(np.float32) - prev.astype(np.float32)))
        prev = small

    return scores


_MOTION_THRESHOLD = 8.0  # mean abs pixel diff; FPV at 5fps typically 15-50+


def filter_short_segments(mask: np.ndarray, min_length: int) -> np.ndarray:
    """Zero out contiguous True runs shorter than ``min_length``.

    Prevents tiny accepted fragments between rejected regions from leaking
    into training data. Returns a new array (immutable).
    """
    if min_length <= 1 or len(mask) == 0:
        return mask.copy()

    out = mask.copy()
    n = len(out)
    i = 0
    while i < n:
        if out[i]:
            run_start = i
            while i < n and out[i]:
                i += 1
            if i - run_start < min_length:
                out[run_start:i] = False
        else:
            i += 1
    return out


def filter_video_from_paths(
    frame_paths: list,
    *,
    device: str = "cpu",
    adaptive_threshold: float = 2.0,
    motion_threshold: float = _MOTION_THRESHOLD,
    min_segment_frames: int = _MIN_SEGMENT_FRAMES,
    yolo_confidence: float = _YOLO_CONFIDENCE_THRESHOLD,
    batch_size: int = 32,
) -> FilterResult:
    """Run full content filter on file paths: motion + YOLO + SBD + verdict.

    Three-stage filter:
    1. Motion: reject static frames (below ``motion_threshold``).
    2. YOLO-World: reject frames containing drones, cameras, gear, etc.
    3. Minimum segment: discard accepted runs shorter than ``min_segment_frames``.

    Shot boundaries (AdaptiveDetector, threshold=2.0) split the video
    into independent camera recordings.  Each shot is classified as FPV
    or non-FPV via majority vote.

    The FPV mask covers EVERY frame — no stride sampling.
    """
    if not frame_paths:
        raise ValueError("frame_paths: expected a non-empty list of paths")

    boundaries = detect_shot_boundaries_from_paths(
        frame_paths, adaptive_threshold=adaptive_threshold,
    )
    motion_scores = compute_motion_scores(frame_paths)
    rejected_objects = detect_rejected_objects_from_paths(
        frame_paths, device=device, batch_size=batch_size,
        confidence=yolo_confidence,
    )

    has_motion = motion_scores >= motion_threshold
    fpv_mask = filter_short_segments(has_motion & ~rejected_objects, min_segment_frames)

    scores = np.where(fpv_mask, 1.0, 0.0).astype(np.float32)

    classification = classify_shots(scores, boundaries, threshold=0.5)
    verdict = video_verdict(classification)

    return FilterResult(
        verdict=verdict,
        n_frames=len(frame_paths),
        n_fpv_frames=int(fpv_mask.sum()),
        fpv_mask=fpv_mask,
        shot_boundaries=boundaries,
        shots=classification.shots,
        per_frame_scores=scores,
        motion_scores=motion_scores,
        rejected_objects=rejected_objects,
    )


def filter_video(
    frames: list[np.ndarray],
    *,
    device: str = "cpu",
    adaptive_threshold: float = 2.0,
    motion_threshold: float = _MOTION_THRESHOLD,
    min_segment_frames: int = _MIN_SEGMENT_FRAMES,
    yolo_confidence: float = _YOLO_CONFIDENCE_THRESHOLD,
    batch_size: int = 32,
) -> FilterResult:
    """Run full content filter on in-memory frames: motion + YOLO + SBD + verdict."""
    if not frames:
        raise ValueError("frames: expected a non-empty list of RGB arrays")

    boundaries = detect_shot_boundaries(frames, adaptive_threshold=adaptive_threshold)
    motion_scores = _compute_motion_from_arrays(frames)
    rejected_objects = detect_rejected_objects(
        frames, device=device, batch_size=batch_size,
        confidence=yolo_confidence,
    )

    has_motion = motion_scores >= motion_threshold
    fpv_mask = filter_short_segments(has_motion & ~rejected_objects, min_segment_frames)

    scores = np.where(fpv_mask, 1.0, 0.0).astype(np.float32)

    classification = classify_shots(scores, boundaries, threshold=0.5)
    verdict = video_verdict(classification)

    return FilterResult(
        verdict=verdict,
        n_frames=len(frames),
        n_fpv_frames=int(fpv_mask.sum()),
        fpv_mask=fpv_mask,
        shot_boundaries=boundaries,
        shots=classification.shots,
        per_frame_scores=scores,
        motion_scores=motion_scores,
        rejected_objects=rejected_objects,
    )


# ---------------------------------------------------------------------------
# Provenance persistence (QC) — write the per-frame decisions to disk so the
# QC report reads the EXACT verdict the run made, with no YOLO re-run.
# ---------------------------------------------------------------------------


def save_filter_result(frames_dir, result: FilterResult):
    """Persist a FilterResult as ``<frames_dir>/_filter.json`` (pure JSON)."""
    import json
    from pathlib import Path

    ms = result.motion_scores
    ro = result.rejected_objects
    payload = {
        "verdict": result.verdict.value,
        "n_frames": int(result.n_frames),
        "n_fpv_frames": int(result.n_fpv_frames),
        "motion_threshold": float(_MOTION_THRESHOLD),
        "fpv_mask": [int(x) for x in np.asarray(result.fpv_mask).tolist()],
        "shot_boundaries": [int(x) for x in result.shot_boundaries],
        "shots": [
            {"start": int(s.start), "end": int(s.end),
             "is_fpv": bool(s.is_fpv), "mean_score": float(s.mean_score)}
            for s in result.shots
        ],
        "motion_scores": [round(float(x), 3) for x in np.asarray(ms).tolist()] if ms is not None else None,
        "rejected_objects": [int(x) for x in np.asarray(ro).tolist()] if ro is not None else None,
    }
    out = Path(frames_dir) / "_filter.json"
    out.write_text(json.dumps(payload))
    return out


def load_filter_result(frames_dir) -> dict | None:
    """Load ``<frames_dir>/_filter.json`` if present, else None."""
    import json
    from pathlib import Path

    p = Path(frames_dir) / "_filter.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


__all__ = [
    "REJECTED_CLASSES",
    "VideoVerdict",
    "ShotInfo",
    "ShotClassification",
    "FilterResult",
    "save_filter_result",
    "load_filter_result",
    "compute_motion_scores",
    "detect_shot_boundaries",
    "detect_shot_boundaries_from_paths",
    "detect_rejected_objects",
    "detect_rejected_objects_from_paths",
    "filter_short_segments",
    "classify_shots",
    "video_verdict",
    "fpv_frame_mask",
    "extract_fpv_ranges",
    "thumbnail_grid_data",
    "filter_video",
    "filter_video_from_paths",
]
