"""CLIP ViT-B/32 zero-shot FPV content filter + PySceneDetect shot boundaries (B1.7b).

Classifies frames as first-person-view (FPV) sports footage vs non-FPV (talking heads,
text overlays, static scenes, etc.) using CLIP zero-shot similarity. Shot boundaries
detected via PySceneDetect AdaptiveDetector. Per-shot majority vote produces a whole-video
verdict (ACCEPT / PARTIAL / REJECT).

**Tier: TORCH** (CLIP inference). All heavy imports (torch, transformers, scenedetect) are
LAZY — inside functions — so the module imports on a torch-free box (pure CI lane).

Reuses the existing CLIP ViT-B/32 model from ``vllatent/encode/text.py``.
"""
from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

# --- FPV prompt ensembles (zero-shot CLIP classification) ---

_FPV_POSITIVE = [
    "first-person view skiing down a steep snowy mountain slope",
    "POV footage looking down at snow rushing beneath ski tips",
    "wide-angle fisheye perspective of a ski run from the skier's point of view",
    "head-mounted perspective descending through a snowy forest",
    "barrel-distorted horizon with snow surface filling the lower frame",
    "fast-moving first-person ski run with motion blur on the snow",
    "skier's eye-level view of a groomed piste with tracks visible",
    "egocentric view of a powdery off-piste descent, snow spray visible",
    "immersive downhill skiing perspective, trees rushing past on both sides",
    "helmet-mounted view looking forward down a ski slope",
    "subjective viewpoint of high-speed carving turns on a ski run",
    "dynamic first-person snowboarding descent with mountain in background",
]

_FPV_NEGATIVE = [
    "a person sitting on a chairlift or gondola with mountain scenery",
    "third-person view of a skier filmed from the slope side",
    "a skiing instructor talking directly to the camera explaining technique",
    "aerial shot of a ski resort or mountain from very high above",
    "a person standing still at the top of a ski run looking at scenery",
    "slow-motion close-up of ski equipment, bindings, or boots on snow",
    "a group of skiers posing for a photo at the base of a mountain",
    "behind-the-shoulder follow-cam shot of a skier filmed by another skier",
]

_NEGATIVE_WEIGHT = 0.75

# Thresholds
_ACCEPT_SHOT_FRAC = 0.60   # >= 60% FPV shots => ACCEPT whole video
_REJECT_SHOT_FRAC = 0.30   # < 30% FPV shots => REJECT whole video


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


# ---------------------------------------------------------------------------
# Shot boundary detection
# ---------------------------------------------------------------------------


def detect_shot_boundaries(
    frames: list[np.ndarray],
    *,
    adaptive_threshold: float = 3.0,
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


# ---------------------------------------------------------------------------
# CLIP zero-shot FPV scoring
# ---------------------------------------------------------------------------


def _get_clip_scorer(device: str = "cpu") -> Callable[[list[np.ndarray]], np.ndarray]:
    """Build a CLIP zero-shot FPV scorer. Lazy-loads torch/transformers."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    from vllatent.config import EncoderConfig
    model_id = EncoderConfig().text_model_id

    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_id)
    for p in model.parameters():
        p.requires_grad_(False)

    n_pos = len(_FPV_POSITIVE)
    n_neg = len(_FPV_NEGATIVE)

    pos_inputs = processor(text=_FPV_POSITIVE, return_tensors="pt", padding=True, truncation=True)
    pos_inputs = {k: v.to(device) for k, v in pos_inputs.items() if isinstance(v, torch.Tensor)}
    neg_inputs = processor(text=_FPV_NEGATIVE, return_tensors="pt", padding=True, truncation=True)
    neg_inputs = {k: v.to(device) for k, v in neg_inputs.items() if isinstance(v, torch.Tensor)}

    logit_scale = model.logit_scale.exp()

    with torch.no_grad():
        pos_out = model.get_text_features(**pos_inputs)
        pos_features = pos_out.pooler_output if hasattr(pos_out, "pooler_output") else pos_out
        pos_features = pos_features / pos_features.norm(dim=-1, keepdim=True)

        neg_out = model.get_text_features(**neg_inputs)
        neg_features = neg_out.pooler_output if hasattr(neg_out, "pooler_output") else neg_out
        neg_features = neg_features / neg_features.norm(dim=-1, keepdim=True)

    def _score_batch(frames_batch: list[np.ndarray]) -> np.ndarray:
        from PIL import Image

        images = [Image.fromarray(f) for f in frames_batch]
        image_inputs = processor(images=images, return_tensors="pt", padding=True)
        image_inputs = {k: v.to(device) for k, v in image_inputs.items() if isinstance(v, torch.Tensor)}

        with torch.no_grad():
            image_out = model.get_image_features(**image_inputs)
            image_features = image_out.pooler_output if hasattr(image_out, "pooler_output") else image_out
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            pos_sim = (image_features @ pos_features.T) * logit_scale  # (B, n_pos)
            neg_sim = (image_features @ neg_features.T) * logit_scale  # (B, n_neg)

            pos_class = pos_sim.mean(dim=-1)  # (B,)
            neg_class = neg_sim.mean(dim=-1) * _NEGATIVE_WEIGHT  # (B,)
            logits = torch.stack([pos_class, neg_class], dim=-1)  # (B, 2)
            fpv_prob = torch.softmax(logits, dim=-1)[:, 0]  # P(FPV)

        return fpv_prob.float().cpu().numpy().astype(np.float32)

    return _score_batch


def score_frames_fpv(
    frames: list[np.ndarray],
    *,
    device: str = "cpu",
    batch_size: int = 32,
) -> np.ndarray:
    """Score frames for FPV content using CLIP zero-shot classification.

    Returns per-frame scores in [0, 1] where higher = more likely FPV.
    """
    if not frames:
        raise ValueError("frames: expected a non-empty list of RGB arrays")

    scorer = _get_clip_scorer(device=device)

    all_scores: list[np.ndarray] = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i : i + batch_size]
        scores = scorer(batch)
        all_scores.append(scores)

    return np.concatenate(all_scores).astype(np.float32)


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


def extract_fpv_ranges(shots: list[ShotInfo]) -> list[tuple[int, int]]:
    """Merge consecutive FPV shots into contiguous frame ranges."""
    ranges: list[tuple[int, int]] = []
    current_start: int | None = None
    current_end: int = 0

    for shot in shots:
        if shot.is_fpv:
            if current_start is None:
                current_start = shot.start
            current_end = shot.end
        else:
            if current_start is not None:
                ranges.append((current_start, current_end))
                current_start = None

    if current_start is not None:
        ranges.append((current_start, current_end))

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


def score_frames_from_paths(
    frame_paths: list,
    *,
    device: str = "cpu",
    batch_size: int = 32,
) -> np.ndarray:
    """Score frames for FPV content by loading from file paths in bounded batches.

    Never holds more than ``batch_size`` frames in memory at once.
    Returns per-frame scores in [0, 1] where higher = more likely FPV.
    """
    if not frame_paths:
        raise ValueError("frame_paths: expected a non-empty list of paths")

    from PIL import Image

    scorer = _get_clip_scorer(device=device)

    all_scores: list[np.ndarray] = []
    for i in range(0, len(frame_paths), batch_size):
        batch_paths = frame_paths[i : i + batch_size]
        batch_frames = [np.array(Image.open(p)) for p in batch_paths]
        scores = scorer(batch_frames)
        all_scores.append(scores)

    return np.concatenate(all_scores).astype(np.float32)


def detect_shot_boundaries_from_paths(
    frame_paths: list,
    *,
    adaptive_threshold: float = 3.0,
    min_scene_len: int = 2,
    fps: float = 30.0,
) -> list[int]:
    """Detect shot boundaries by loading frames one at a time from file paths.

    Memory-efficient: never holds more than 1 frame in memory.
    Returns a sorted list of frame indices where shot transitions occur.
    """
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


_MOTION_THRESHOLD = 8.0  # mean abs pixel diff; FPV at 5fps typically 15-50+


def filter_video_from_paths(
    frame_paths: list,
    *,
    device: str = "cpu",
    adaptive_threshold: float = 3.0,
    fpv_threshold: float = 0.65,
    motion_threshold: float = _MOTION_THRESHOLD,
    batch_size: int = 32,
) -> FilterResult:
    """Run full content filter on file paths: CLIP + motion + SBD + verdict.

    Two-signal filter: a frame is FPV only if its CLIP semantic score AND its
    temporal motion score both pass their thresholds. This catches content that
    CLIP alone misses (e.g. drone close-ups that semantically match "drone"
    prompts but have near-zero inter-frame motion).

    The FPV mask covers EVERY frame — no stride sampling.
    """
    if not frame_paths:
        raise ValueError("frame_paths: expected a non-empty list of paths")

    boundaries = detect_shot_boundaries_from_paths(
        frame_paths, adaptive_threshold=adaptive_threshold,
    )
    clip_scores = score_frames_from_paths(
        frame_paths, device=device, batch_size=batch_size,
    )
    motion_scores = compute_motion_scores(frame_paths)

    combined = np.where(
        motion_scores >= motion_threshold, clip_scores, 0.0,
    ).astype(np.float32)

    mask = combined >= fpv_threshold

    classification = classify_shots(combined, boundaries, threshold=fpv_threshold)
    verdict = video_verdict(classification)

    return FilterResult(
        verdict=verdict,
        n_frames=len(frame_paths),
        n_fpv_frames=int(mask.sum()),
        fpv_mask=mask,
        shot_boundaries=boundaries,
        shots=classification.shots,
        per_frame_scores=combined,
    )


def filter_video(
    frames: list[np.ndarray],
    *,
    device: str = "cpu",
    adaptive_threshold: float = 3.0,
    fpv_threshold: float = 0.65,
    batch_size: int = 32,
) -> FilterResult:
    """Run full content filter: SBD + CLIP scoring + shot voting + verdict."""
    if not frames:
        raise ValueError("frames: expected a non-empty list of RGB arrays")

    boundaries = detect_shot_boundaries(frames, adaptive_threshold=adaptive_threshold)
    scores = score_frames_fpv(frames, device=device, batch_size=batch_size)
    classification = classify_shots(scores, boundaries, threshold=fpv_threshold)
    verdict = video_verdict(classification)
    mask = fpv_frame_mask(classification)

    return FilterResult(
        verdict=verdict,
        n_frames=len(frames),
        n_fpv_frames=int(mask.sum()),
        fpv_mask=mask,
        shot_boundaries=boundaries,
        shots=classification.shots,
        per_frame_scores=scores,
    )


__all__ = [
    "VideoVerdict",
    "ShotInfo",
    "ShotClassification",
    "FilterResult",
    "compute_motion_scores",
    "detect_shot_boundaries",
    "detect_shot_boundaries_from_paths",
    "score_frames_fpv",
    "score_frames_from_paths",
    "classify_shots",
    "video_verdict",
    "fpv_frame_mask",
    "extract_fpv_ranges",
    "thumbnail_grid_data",
    "filter_video",
    "filter_video_from_paths",
]
