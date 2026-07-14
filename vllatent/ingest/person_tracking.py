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

from vllatent.schemas import HISTORY, HORIZON

PERSON_BBOX_KEY = "person_bbox"
PERSON_VISIBLE_KEY = "person_visible"
PERSON_STATE_VALID_KEY = "person_state_valid"
PERSON_CONF_KEY = "person_conf"
PERSON_SELECTED_TRACK_ID_KEY = "person_selected_track_id"
PERSON_SECOND_BEST_TRACK_ID_KEY = "person_second_best_track_id"
PERSON_SUBJECT_AMBIGUITY_MARGIN_KEY = "person_subject_ambiguity_margin"
PERSON_SUBJECT_IS_AMBIGUOUS_KEY = "person_subject_is_ambiguous"
PERSON_BBOX_SPACE_KEY = "person_bbox_space"
PERSON_BBOX_SPACE_ENCODER_CROP = "encoder_crop"
PERSON_BBOX_SPACE_RAW_FRAME = "raw_frame"
PERSON_BBOX_DIM = 4
PERSON_MIN_BBOX_AREA = 0.0025
PERSON_EDGE_MARGIN = 1e-6
PERSON_DINO_PATCH_GRID = 14
PERSON_TRACKABLE_MIN_AREA_PATCHES = 4.0
PERSON_TRACKABLE_MIN_AREA = PERSON_TRACKABLE_MIN_AREA_PATCHES / float(PERSON_DINO_PATCH_GRID * PERSON_DINO_PATCH_GRID)
PERSON_TRACKABLE_MAX_CENTER_JUMP = 0.25
PERSON_TRACKABLE_MIN_RUN = 3
PERSON_TRACK_CLASSES = ("person", "skier", "snowboarder")
PERSON_TRACKER_ID = "yolov8s-worldv2.pt+bytetrack"
PERSON_TRACK_CONFIDENCE = 0.15
PERSON_SUBJECT_AMBIGUITY_MARGIN_MIN = 0.20
PERSON_SUBJECT_MIN_COVISIBLE_FRAMES = 3
PERSON_SUBJECT_MIN_COVISIBLE_RATIO = 0.25


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
    person_state_valid: np.ndarray
    person_conf: np.ndarray
    provenance: dict[str, Any]
    selected_track_id: int = -1
    second_best_track_id: int = -1
    subject_ambiguity_margin: float = 1.0
    subject_is_ambiguous: bool = False


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
    person_trackable_frames: int


def empty_person_tracks(n_frames: int) -> PersonTrackResult:
    """Return invisible-person defaults for old caches or detector failures."""
    return PersonTrackResult(
        person_bbox=np.zeros((n_frames, PERSON_BBOX_DIM), dtype=np.float32),
        person_visible=np.zeros(n_frames, dtype=np.bool_),
        person_state_valid=np.zeros(n_frames, dtype=np.bool_),
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


def validate_person_state_valid(*, n_frames: int, person_state_valid: np.ndarray) -> None:
    """Validate optional B3 trackable-person supervision mask."""
    if person_state_valid.shape != (n_frames,):
        raise ValueError(f"person_state_valid: expected ({n_frames},), got {person_state_valid.shape}")


def valid_encoder_crop_bbox_mask(
    person_bbox: np.ndarray,
    *,
    min_area: float = PERSON_MIN_BBOX_AREA,
) -> np.ndarray:
    """Return boxes with non-degenerate area fully inside normalized crop coords."""
    bbox = np.asarray(person_bbox, dtype=np.float32)
    if bbox.ndim != 2 or bbox.shape[1] != PERSON_BBOX_DIM:
        raise ValueError(f"person_bbox: expected (N,{PERSON_BBOX_DIM}), got {bbox.shape}")
    cx = bbox[:, 0]
    cy = bbox[:, 1]
    bw = bbox[:, 2]
    bh = bbox[:, 3]
    x1 = cx - 0.5 * bw
    x2 = cx + 0.5 * bw
    y1 = cy - 0.5 * bh
    y2 = cy + 0.5 * bh
    area = bw * bh
    return (
        np.all(np.isfinite(bbox), axis=1)
        & (bw > 0.0)
        & (bh > 0.0)
        & (area >= float(min_area))
        & (x1 >= 0.0)
        & (y1 >= 0.0)
        & (x2 <= 1.0)
        & (y2 <= 1.0)
    )


def non_edge_encoder_crop_bbox_mask(
    person_bbox: np.ndarray,
    *,
    edge_margin: float = PERSON_EDGE_MARGIN,
) -> np.ndarray:
    """Return boxes that do not touch the normalized encoder-crop boundary."""
    bbox = np.asarray(person_bbox, dtype=np.float32)
    if bbox.ndim != 2 or bbox.shape[1] != PERSON_BBOX_DIM:
        raise ValueError(f"person_bbox: expected (N,{PERSON_BBOX_DIM}), got {bbox.shape}")
    cx = bbox[:, 0]
    cy = bbox[:, 1]
    bw = bbox[:, 2]
    bh = bbox[:, 3]
    x1 = cx - 0.5 * bw
    x2 = cx + 0.5 * bw
    y1 = cy - 0.5 * bh
    y2 = cy + 0.5 * bh
    return (
        (x1 > float(edge_margin))
        & (y1 > float(edge_margin))
        & (x2 < 1.0 - float(edge_margin))
        & (y2 < 1.0 - float(edge_margin))
    )


def _keep_runs(mask: np.ndarray, *, min_run: int) -> np.ndarray:
    """Keep only True runs with at least ``min_run`` consecutive frames."""
    arr = np.asarray(mask).astype(np.bool_)
    if min_run <= 1 or arr.size == 0:
        return arr.copy()
    out = np.zeros_like(arr, dtype=np.bool_)
    start: int | None = None
    for i, value in enumerate(arr):
        if bool(value) and start is None:
            start = i
        if (not bool(value) or i == arr.size - 1) and start is not None:
            end = i + 1 if bool(value) and i == arr.size - 1 else i
            if end - start >= min_run:
                out[start:end] = True
            start = None
    return out


def person_trackable_mask(
    person_bbox: np.ndarray,
    person_visible: np.ndarray,
    *,
    min_area: float = PERSON_TRACKABLE_MIN_AREA,
    max_center_jump: float = PERSON_TRACKABLE_MAX_CENTER_JUMP,
    min_run: int = PERSON_TRACKABLE_MIN_RUN,
) -> np.ndarray:
    """Return frames safe for person-state supervision, stricter than detector visibility."""
    bbox = np.asarray(person_bbox, dtype=np.float32)
    visible = np.asarray(person_visible).astype(np.bool_)
    if bbox.ndim != 2 or bbox.shape[1] != PERSON_BBOX_DIM:
        raise ValueError(f"person_bbox: expected (N,{PERSON_BBOX_DIM}), got {bbox.shape}")
    if visible.shape != (bbox.shape[0],):
        raise ValueError(f"person_visible: expected ({bbox.shape[0]},), got {visible.shape}")

    valid_geom = valid_encoder_crop_bbox_mask(bbox, min_area=min_area)
    non_edge = non_edge_encoder_crop_bbox_mask(bbox)
    trackable = visible & valid_geom & non_edge

    if max_center_jump > 0.0 and np.any(trackable):
        idx = np.flatnonzero(trackable)
        centers = bbox[idx, :2]
        for j in range(1, len(idx)):
            if idx[j] != idx[j - 1] + 1:
                continue
            jump = float(np.linalg.norm((centers[j] - centers[j - 1]).astype(np.float64)))
            if jump > max_center_jump:
                trackable[idx[j]] = False

    return _keep_runs(trackable, min_run=min_run)


def strict_person_window_mask(
    person_state_valid: np.ndarray,
    *,
    history: int,
    horizon: int,
) -> np.ndarray:
    """Return anchor-frame mask requiring full valid person history and future."""
    if history <= 0:
        raise ValueError(f"history must be > 0, got {history}")
    if horizon <= 0:
        raise ValueError(f"horizon must be > 0, got {horizon}")
    state_valid = np.asarray(person_state_valid).astype(np.bool_)
    n_windows = max(0, state_valid.size - horizon)
    valid_windows = np.zeros(n_windows, dtype=np.bool_)
    for t in range(history - 1, n_windows):
        hist = state_valid[t - history + 1 : t + 1]
        fut = state_valid[t + 1 : t + 1 + horizon]
        valid_windows[t] = bool(
            hist.size == history
            and fut.size == horizon
            and np.all(hist)
            and np.all(fut)
        )
    return valid_windows


def sanitize_person_track_arrays(
    *,
    person_bbox: np.ndarray,
    person_visible: np.ndarray,
    person_conf: np.ndarray,
    min_area: float = PERSON_MIN_BBOX_AREA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mask out visible labels whose crop-space boxes are zero-area or tiny."""
    bbox = np.asarray(person_bbox, dtype=np.float32).copy()
    visible = np.asarray(person_visible).astype(np.bool_, copy=True)
    conf = np.asarray(person_conf, dtype=np.float32).copy()
    validate_person_track_arrays(
        n_frames=int(bbox.shape[0]),
        person_bbox=bbox,
        person_visible=visible,
        person_conf=conf,
    )
    valid = valid_encoder_crop_bbox_mask(bbox, min_area=min_area)
    visible &= valid
    bbox[~visible] = 0.0
    conf[~visible] = 0.0
    return bbox, visible, conf


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
    raw_visible = int(np.sum(visible))
    bbox, visible, conf = sanitize_person_track_arrays(
        person_bbox=bbox,
        person_visible=visible,
        person_conf=conf,
    )
    if PERSON_STATE_VALID_KEY in clip:
        state_valid = np.asarray(clip[PERSON_STATE_VALID_KEY]).astype(np.bool_)
        validate_person_state_valid(n_frames=n_frames, person_state_valid=state_valid)
        state_valid &= person_trackable_mask(bbox, visible)
    else:
        state_valid = person_trackable_mask(bbox, visible)
    ambiguity_known = all(
        key in clip
        for key in (
            PERSON_SELECTED_TRACK_ID_KEY,
            PERSON_SECOND_BEST_TRACK_ID_KEY,
            PERSON_SUBJECT_AMBIGUITY_MARGIN_KEY,
            PERSON_SUBJECT_IS_AMBIGUOUS_KEY,
        )
    )
    selected_track_id = int(np.asarray(clip.get(PERSON_SELECTED_TRACK_ID_KEY, -1)).item())
    second_best_track_id = int(np.asarray(clip.get(PERSON_SECOND_BEST_TRACK_ID_KEY, -1)).item())
    ambiguity_margin = float(np.asarray(clip.get(PERSON_SUBJECT_AMBIGUITY_MARGIN_KEY, 1.0)).item())
    subject_is_ambiguous = bool(np.asarray(clip.get(PERSON_SUBJECT_IS_AMBIGUOUS_KEY, False)).item())
    if subject_is_ambiguous:
        state_valid[:] = False
    return PersonTrackResult(
        person_bbox=bbox,
        person_visible=visible,
        person_state_valid=state_valid,
        person_conf=conf,
        provenance={
            "source": "cache",
            "sanitized_invisible_frames": raw_visible - int(np.sum(visible)),
            "computed_state_valid_frames": int(np.sum(state_valid)),
            "state_valid_source": "cache" if PERSON_STATE_VALID_KEY in clip else "computed",
            "subject_ambiguity_known": ambiguity_known,
        },
        selected_track_id=selected_track_id,
        second_best_track_id=second_best_track_id,
        subject_ambiguity_margin=ambiguity_margin,
        subject_is_ambiguous=subject_is_ambiguous,
    )


def _center_crop_bounds(image_hw: tuple[int, int]) -> tuple[int, int, int]:
    h, w = image_hw
    crop = min(h, w)
    top = (h - crop) // 2
    left = (w - crop) // 2
    return top, left, crop


def xyxy_to_encoder_crop_cxcywh(xyxy: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    """Convert raw-frame pixel xyxy into DINO encoder center-crop normalized cxcywh."""
    h, w = image_hw
    top, left, crop = _center_crop_bounds(image_hw)
    x1, y1, x2, y2 = xyxy.astype(np.float32)
    x1 = float(np.clip(x1, left, left + crop))
    x2 = float(np.clip(x2, left, left + crop))
    y1 = float(np.clip(y1, top, top + crop))
    y2 = float(np.clip(y2, top, top + crop))
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = x1 - left + 0.5 * bw
    cy = y1 - top + 0.5 * bh
    return np.array([cx / crop, cy / crop, bw / crop, bh / crop], dtype=np.float32)


def raw_frame_cxcywh_to_encoder_crop(
    person_bbox: np.ndarray,
    image_hw: tuple[int, int],
) -> np.ndarray:
    """Convert raw-frame normalized cxcywh boxes into encoder-crop coordinates."""
    bbox = np.asarray(person_bbox, dtype=np.float32)
    if bbox.ndim != 2 or bbox.shape[1] != PERSON_BBOX_DIM:
        raise ValueError(f"person_bbox: expected (N,{PERSON_BBOX_DIM}), got {bbox.shape}")
    h, w = image_hw
    cx = bbox[:, 0] * float(w)
    cy = bbox[:, 1] * float(h)
    bw = bbox[:, 2] * float(w)
    bh = bbox[:, 3] * float(h)
    xyxy = np.stack(
        [
            cx - 0.5 * bw,
            cy - 0.5 * bh,
            cx + 0.5 * bw,
            cy + 0.5 * bh,
        ],
        axis=1,
    )
    return np.stack([xyxy_to_encoder_crop_cxcywh(row, image_hw) for row in xyxy]).astype(np.float32)


def select_subject_track(
    detections: list[TrackedDetection],
    *,
    n_frames: int,
    image_hw: tuple[int, int],
    history: int = HISTORY,
    horizon: int = HORIZON,
) -> PersonTrackResult:
    """Select the track with the most strict B3 windows and reject ambiguous runners-up."""
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    if history <= 0:
        raise ValueError(f"history must be > 0, got {history}")
    if horizon <= 0:
        raise ValueError(f"horizon must be > 0, got {horizon}")
    if not detections:
        return empty_person_tracks(n_frames)

    by_track: dict[int, list[TrackedDetection]] = {}
    for det in detections:
        if 0 <= det.frame_idx < n_frames:
            by_track.setdefault(det.track_id, []).append(det)
    if not by_track:
        return empty_person_tracks(n_frames)

    def _evidence(items: list[TrackedDetection]) -> dict[str, Any]:
        boxes = np.stack([xyxy_to_encoder_crop_cxcywh(d.xyxy, image_hw) for d in items])
        valid = valid_encoder_crop_bbox_mask(boxes)
        if not np.any(valid):
            return {
                "valid_frames": 0,
                "frame_ids": set(),
                "centrality": float("-inf"),
                "area": 0.0,
                "confidence": 0.0,
                "strict_windows": 0,
            }
        valid_boxes = boxes[valid]
        valid_items = [item for item, keep in zip(items, valid, strict=False) if bool(keep)]
        centers = valid_boxes[:, :2]
        centrality = -float(np.mean(np.sum((centers - 0.5) ** 2, axis=1)))
        area = float(np.mean(valid_boxes[:, 2] * valid_boxes[:, 3]))
        confidence = float(np.mean([item.confidence for item in valid_items]))
        track_bbox = np.zeros((n_frames, PERSON_BBOX_DIM), dtype=np.float32)
        track_visible = np.zeros(n_frames, dtype=np.bool_)
        for item, box in zip(valid_items, valid_boxes, strict=False):
            track_bbox[item.frame_idx] = box
            track_visible[item.frame_idx] = True
        trackable = person_trackable_mask(track_bbox, track_visible)
        strict_window_mask = strict_person_window_mask(
            trackable,
            history=history,
            horizon=horizon,
        )
        strict_windows = int(np.sum(strict_window_mask))
        return {
            "valid_frames": int(np.sum(valid)),
            "frame_ids": {item.frame_idx for item in valid_items},
            "centrality": centrality,
            "area": area,
            "confidence": confidence,
            "strict_windows": strict_windows,
        }

    evidence = {track_id: _evidence(items) for track_id, items in by_track.items()}

    def _score(track_id: int) -> tuple[int, int, float, float]:
        item = evidence[track_id]
        return (
            item["strict_windows"],
            item["valid_frames"],
            item["centrality"],
            item["area"],
        )

    ranked_ids = sorted(by_track, key=_score, reverse=True)
    selected_id = ranked_ids[0]
    selected = by_track[selected_id]
    top = evidence[selected_id]

    def _runner_up_evidence(track_id: int) -> tuple[float, int, float]:
        second = evidence[track_id]
        top_count = int(top["valid_frames"])
        second_count = int(second["valid_frames"])
        overlap = len(top["frame_ids"] & second["frame_ids"])
        overlap_ratio = overlap / max(1, min(top_count, second_count))
        duration_similarity = min(1.0, second_count / max(1, top_count))
        centrality_similarity = 1.0 - min(
            1.0,
            abs(float(top["centrality"]) - float(second["centrality"])) / 0.5,
        )
        max_area = max(float(top["area"]), float(second["area"]), 1e-8)
        area_similarity = min(float(top["area"]), float(second["area"])) / max_area
        confidence_similarity = 1.0 - min(
            1.0,
            abs(float(top["confidence"]) - float(second["confidence"])),
        )
        plausibility_similarity = (
            0.70 * duration_similarity
            + 0.15 * centrality_similarity
            + 0.10 * area_similarity
            + 0.05 * confidence_similarity
        )
        margin = float(np.clip(1.0 - plausibility_similarity, 0.0, 1.0))
        return margin, overlap, overlap_ratio

    runner_up_evidence = {
        track_id: _runner_up_evidence(track_id) for track_id in ranked_ids[1:]
    }
    covisible_ids = [
        track_id
        for track_id in ranked_ids[1:]
        if runner_up_evidence[track_id][1] >= PERSON_SUBJECT_MIN_COVISIBLE_FRAMES
        and runner_up_evidence[track_id][2] >= PERSON_SUBJECT_MIN_COVISIBLE_RATIO
    ]
    if covisible_ids:
        second_best_id = min(covisible_ids, key=lambda track_id: runner_up_evidence[track_id][0])
    else:
        second_best_id = ranked_ids[1] if len(ranked_ids) > 1 else -1

    ambiguity_margin = 1.0
    covisible_frames = 0
    covisible_ratio = 0.0
    if second_best_id >= 0:
        ambiguity_margin, covisible_frames, covisible_ratio = runner_up_evidence[second_best_id]

    subject_is_ambiguous = bool(
        second_best_id >= 0
        and covisible_frames >= PERSON_SUBJECT_MIN_COVISIBLE_FRAMES
        and covisible_ratio >= PERSON_SUBJECT_MIN_COVISIBLE_RATIO
        and ambiguity_margin < PERSON_SUBJECT_AMBIGUITY_MARGIN_MIN
    )

    bbox = np.zeros((n_frames, PERSON_BBOX_DIM), dtype=np.float32)
    visible = np.zeros(n_frames, dtype=np.bool_)
    conf = np.zeros(n_frames, dtype=np.float32)
    for det in selected:
        det_bbox = xyxy_to_encoder_crop_cxcywh(det.xyxy, image_hw)
        if not bool(valid_encoder_crop_bbox_mask(det_bbox[None, :])[0]):
            continue
        bbox[det.frame_idx] = det_bbox
        visible[det.frame_idx] = True
        conf[det.frame_idx] = np.float32(det.confidence)

    state_valid = person_trackable_mask(bbox, visible)
    if subject_is_ambiguous:
        state_valid[:] = False

    return PersonTrackResult(
        person_bbox=bbox,
        person_visible=visible,
        person_state_valid=state_valid,
        person_conf=conf,
        provenance={
            "detector": PERSON_TRACKER_ID,
            "tracker": "bytetrack",
            "classes": list(PERSON_TRACK_CLASSES),
            "bbox_space": PERSON_BBOX_SPACE_ENCODER_CROP,
            "selection": (
                "strict_windows_then_longest_then_central_then_largest; "
                "strongest_covisible_runner_up"
            ),
            "selection_history": history,
            "selection_horizon": horizon,
            "selected_strict_windows": int(top["strict_windows"]),
            "selected_track_id": selected_id,
            "second_best_track_id": second_best_id,
            "subject_ambiguity_margin": ambiguity_margin,
            "subject_is_ambiguous": subject_is_ambiguous,
            "second_best_covisible_frames": covisible_frames,
            "second_best_covisible_ratio": covisible_ratio,
            "ambiguity_margin_min": PERSON_SUBJECT_AMBIGUITY_MARGIN_MIN,
        },
        selected_track_id=selected_id,
        second_best_track_id=second_best_id,
        subject_ambiguity_margin=ambiguity_margin,
        subject_is_ambiguous=subject_is_ambiguous,
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
    history: int = HISTORY,
    horizon: int = HORIZON,
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
    result = select_subject_track(
        detections,
        n_frames=len(paths),
        image_hw=image_hw,
        history=history,
        horizon=horizon,
    )
    return PersonTrackResult(
        person_bbox=result.person_bbox,
        person_visible=result.person_visible,
        person_state_valid=result.person_state_valid,
        person_conf=result.person_conf,
        provenance={
            **result.provenance,
            "detector": model_id,
            "confidence": confidence,
            "classes": list(classes),
            "bbox_space": PERSON_BBOX_SPACE_ENCODER_CROP,
        },
        selected_track_id=result.selected_track_id,
        second_best_track_id=result.second_best_track_id,
        subject_ambiguity_margin=result.subject_ambiguity_margin,
        subject_is_ambiguous=result.subject_is_ambiguous,
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
    person_state_valid: np.ndarray | None = None,
    history: int,
    horizon: int,
) -> ScreenReport:
    """Compute B3 data-screen counts for one cached clip."""
    n_frames = int(latents.shape[0])
    duplicate = duplicate_frame_runs_from_latents(latents)
    time_remap = time_remap_flags_from_deltas(deltas)
    accel = accel_outlier_flags_from_deltas(deltas)
    n_windows = max(0, n_frames - horizon)
    visible = np.asarray(person_visible).astype(np.bool_)
    state_valid = visible if person_state_valid is None else np.asarray(person_state_valid).astype(np.bool_)
    person_valid_windows = int(np.sum(strict_person_window_mask(
        state_valid,
        history=history,
        horizon=horizon,
    )))
    return ScreenReport(
        n_frames=n_frames,
        n_windows=n_windows,
        duplicate_frame_runs=int(np.sum(duplicate)),
        time_remap_flags=int(np.sum(time_remap)),
        accel_outlier_frames=int(np.sum(accel)),
        person_visible_frames=int(np.sum(visible)),
        person_valid_windows=person_valid_windows,
        person_trackable_frames=int(np.sum(state_valid)),
    )


def person_label_quality_stats(
    person_bbox: np.ndarray,
    person_visible: np.ndarray,
    *,
    min_area: float = PERSON_MIN_BBOX_AREA,
    edge_margin: float = PERSON_EDGE_MARGIN,
) -> dict[str, Any]:
    """Summarize visible-label geometry before any read-time sanitization."""
    bbox = np.asarray(person_bbox, dtype=np.float32)
    visible = np.asarray(person_visible).astype(np.bool_)
    if bbox.ndim != 2 or bbox.shape[1] != PERSON_BBOX_DIM:
        raise ValueError(f"person_bbox: expected (N,{PERSON_BBOX_DIM}), got {bbox.shape}")
    if visible.shape != (bbox.shape[0],):
        raise ValueError(f"person_visible: expected ({bbox.shape[0]},), got {visible.shape}")

    cx = bbox[:, 0]
    cy = bbox[:, 1]
    bw = bbox[:, 2]
    bh = bbox[:, 3]
    x1 = cx - 0.5 * bw
    x2 = cx + 0.5 * bw
    y1 = cy - 0.5 * bh
    y2 = cy + 0.5 * bh
    area = bw * bh
    visible_area = area[visible]
    degenerate = visible & ((bw <= 0.0) | (bh <= 0.0) | ~np.all(np.isfinite(bbox), axis=1))
    tiny = visible & (area < float(min_area))
    edge = visible & (
        (x1 <= float(edge_margin))
        | (y1 <= float(edge_margin))
        | (x2 >= 1.0 - float(edge_margin))
        | (y2 >= 1.0 - float(edge_margin))
    )
    valid = visible & valid_encoder_crop_bbox_mask(bbox, min_area=min_area)
    trackable = person_trackable_mask(bbox, visible)

    idx = np.flatnonzero(visible)
    jumps = np.zeros(0, dtype=np.float32)
    if idx.size >= 2:
        consecutive = np.diff(idx) == 1
        if np.any(consecutive):
            centers = bbox[idx][:, :2]
            jumps = np.linalg.norm(np.diff(centers, axis=0)[consecutive], axis=1).astype(np.float32)

    def _percentile(values: np.ndarray, q: float) -> float:
        if values.size == 0:
            return 0.0
        return float(np.percentile(values.astype(np.float64), q))

    return {
        "visible_frames_raw": int(np.sum(visible)),
        "visible_frames_sanitized": int(np.sum(valid)),
        "trackable_frames": int(np.sum(trackable)),
        "invalid_visible_frames": int(np.sum(visible & ~valid)),
        "degenerate_visible_frames": int(np.sum(degenerate)),
        "tiny_visible_frames": int(np.sum(tiny)),
        "edge_visible_frames": int(np.sum(edge)),
        "flicker_transitions": int(np.sum(visible[1:] != visible[:-1])) if visible.size >= 2 else 0,
        "center_jump_p95": _percentile(jumps, 95.0),
        "center_jump_p99": _percentile(jumps, 99.0),
        "area_min": float(np.min(visible_area)) if visible_area.size else 0.0,
        "area_p05": _percentile(visible_area, 5.0),
        "area_p50": _percentile(visible_area, 50.0),
        "area_p95": _percentile(visible_area, 95.0),
        "area_max": float(np.max(visible_area)) if visible_area.size else 0.0,
    }


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
        "person_trackable_frames": 0,
        "duplicate_frame_runs": 0,
        "time_remap_flags": 0,
        "accel_outlier_frames": 0,
        "flagged_clips": 0,
        "person_invalid_visible_frames": 0,
        "person_degenerate_visible_frames": 0,
        "person_tiny_visible_frames": 0,
        "person_edge_visible_frames": 0,
        "person_flicker_transitions": 0,
        "subject_ambiguity_known_clips": 0,
        "subject_ambiguity_unknown_clips": 0,
        "subject_ambiguous_clips": 0,
    }

    for path in paths:
        arrays = _load_npz_arrays(path)
        qc = (
            person_label_quality_stats(arrays[PERSON_BBOX_KEY], arrays[PERSON_VISIBLE_KEY])
            if PERSON_BBOX_KEY in arrays and PERSON_VISIBLE_KEY in arrays
            else {}
        )
        tracks = person_tracks_from_cache(arrays)
        report = screen_clip_arrays(
            latents=arrays["latents"],
            deltas=arrays["deltas"],
            person_visible=tracks.person_visible,
            person_state_valid=tracks.person_state_valid,
            history=history,
            horizon=horizon,
        )
        source = path.stem.split("_")[0]
        source_entry = sources.setdefault(
            source,
            {
                "clips": 0,
                "windows": 0,
                "person_valid_windows": 0,
                "person_trackable_frames": 0,
                "subject_ambiguity_known_clips": 0,
                "subject_ambiguity_unknown_clips": 0,
                "subject_ambiguous_clips": 0,
            },
        )
        ambiguity_known = bool(tracks.provenance.get("subject_ambiguity_known", False))
        source_entry["clips"] += 1
        source_entry["windows"] += report.n_windows
        source_entry["person_valid_windows"] += report.person_valid_windows
        source_entry["person_trackable_frames"] += report.person_trackable_frames
        source_entry["subject_ambiguity_known_clips"] += int(ambiguity_known)
        source_entry["subject_ambiguity_unknown_clips"] += int(not ambiguity_known)
        source_entry["subject_ambiguous_clips"] += int(tracks.subject_is_ambiguous)
        for src_key, qc_key in (
            ("person_invalid_visible_frames", "invalid_visible_frames"),
            ("person_degenerate_visible_frames", "degenerate_visible_frames"),
            ("person_tiny_visible_frames", "tiny_visible_frames"),
            ("person_edge_visible_frames", "edge_visible_frames"),
            ("person_flicker_transitions", "flicker_transitions"),
        ):
            source_entry[src_key] = source_entry.get(src_key, 0) + int(qc.get(qc_key, 0))

        flags: list[str] = []
        if report.duplicate_frame_runs:
            flags.append("duplicate_frames")
        if report.time_remap_flags:
            flags.append("time_remap")
        if report.accel_outlier_frames:
            flags.append("accel_outliers")
        if report.n_windows and report.person_valid_windows == 0:
            flags.append("person_untrackable_windows")
        if int(qc.get("invalid_visible_frames", 0)):
            flags.append("person_invalid_labels")
        if int(qc.get("edge_visible_frames", 0)):
            flags.append("person_edge_labels")
        if not ambiguity_known:
            flags.append("subject_ambiguity_unknown")
        if tracks.subject_is_ambiguous:
            flags.append("subject_ambiguous")

        clip_record = {
            "clip_id": path.stem,
            "source": source,
            "n_frames": report.n_frames,
            "n_windows": report.n_windows,
            "person_valid_windows": report.person_valid_windows,
            "person_visible_frames": report.person_visible_frames,
            "person_trackable_frames": report.person_trackable_frames,
            "subject_ambiguity_known": ambiguity_known,
            "selected_track_id": tracks.selected_track_id,
            "second_best_track_id": tracks.second_best_track_id,
            "subject_ambiguity_margin": tracks.subject_ambiguity_margin,
            "subject_is_ambiguous": tracks.subject_is_ambiguous,
            "duplicate_frame_runs": report.duplicate_frame_runs,
            "time_remap_flags": report.time_remap_flags,
            "accel_outlier_frames": report.accel_outlier_frames,
            "flags": flags,
        }
        if qc:
            clip_record["person_label_qc"] = qc
        clips.append(clip_record)

        totals["clips"] += 1
        totals["windows"] += report.n_windows
        totals["person_valid_windows"] += report.person_valid_windows
        totals["person_visible_frames"] += report.person_visible_frames
        totals["person_trackable_frames"] += report.person_trackable_frames
        totals["duplicate_frame_runs"] += report.duplicate_frame_runs
        totals["time_remap_flags"] += report.time_remap_flags
        totals["accel_outlier_frames"] += report.accel_outlier_frames
        totals["flagged_clips"] += int(bool(flags))
        totals["person_invalid_visible_frames"] += int(qc.get("invalid_visible_frames", 0))
        totals["person_degenerate_visible_frames"] += int(qc.get("degenerate_visible_frames", 0))
        totals["person_tiny_visible_frames"] += int(qc.get("tiny_visible_frames", 0))
        totals["person_edge_visible_frames"] += int(qc.get("edge_visible_frames", 0))
        totals["person_flicker_transitions"] += int(qc.get("flicker_transitions", 0))
        totals["subject_ambiguity_known_clips"] += int(ambiguity_known)
        totals["subject_ambiguity_unknown_clips"] += int(not ambiguity_known)
        totals["subject_ambiguous_clips"] += int(tracks.subject_is_ambiguous)

    totals["sources"] = len(sources)
    return {
        "totals": totals,
        "sources": sources,
        "clips": clips,
    }


__all__ = [
    "PERSON_BBOX_DIM",
    "PERSON_BBOX_KEY",
    "PERSON_EDGE_MARGIN",
    "PERSON_MIN_BBOX_AREA",
    "PERSON_BBOX_SPACE_ENCODER_CROP",
    "PERSON_BBOX_SPACE_KEY",
    "PERSON_BBOX_SPACE_RAW_FRAME",
    "PERSON_CONF_KEY",
    "PERSON_DINO_PATCH_GRID",
    "PERSON_TRACK_CLASSES",
    "PERSON_TRACKER_ID",
    "PERSON_STATE_VALID_KEY",
    "PERSON_SELECTED_TRACK_ID_KEY",
    "PERSON_SECOND_BEST_TRACK_ID_KEY",
    "PERSON_SUBJECT_AMBIGUITY_MARGIN_KEY",
    "PERSON_SUBJECT_AMBIGUITY_MARGIN_MIN",
    "PERSON_SUBJECT_IS_AMBIGUOUS_KEY",
    "PERSON_TRACKABLE_MAX_CENTER_JUMP",
    "PERSON_TRACKABLE_MIN_AREA",
    "PERSON_TRACKABLE_MIN_AREA_PATCHES",
    "PERSON_TRACKABLE_MIN_RUN",
    "PERSON_VISIBLE_KEY",
    "PersonTrackResult",
    "ScreenReport",
    "TrackedDetection",
    "accel_outlier_flags_from_deltas",
    "duplicate_frame_runs_from_latents",
    "empty_person_tracks",
    "person_state_from_bbox",
    "person_label_quality_stats",
    "person_trackable_mask",
    "person_tracks_from_cache",
    "raw_frame_cxcywh_to_encoder_crop",
    "screen_clip_arrays",
    "screen_cache_dir",
    "select_subject_track",
    "strict_person_window_mask",
    "time_remap_flags_from_deltas",
    "track_persons_from_paths",
    "sanitize_person_track_arrays",
    "validate_person_track_arrays",
    "validate_person_state_valid",
    "valid_encoder_crop_bbox_mask",
    "xyxy_to_encoder_crop_cxcywh",
]
