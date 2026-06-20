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
    "a first person view from a drone flying over terrain",
    "point of view footage from an action camera while skiing",
    "first person perspective of mountain biking on a trail",
    "POV drone footage following an athlete",
    "egocentric camera view of outdoor sports activity",
    "GoPro footage of skiing downhill",
    "FPV drone racing through a landscape",
    "body-mounted camera view during extreme sports",
]

_FPV_NEGATIVE = [
    "a person talking to the camera in a room",
    "text overlay on a dark background",
    "video title screen with graphics",
    "static establishing shot of a landscape",
    "interview scene with two people sitting",
    "product advertisement with logos",
    "social media subscribe button animation",
    "drone hovering stationary above a building",
]

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

    all_prompts = _FPV_POSITIVE + _FPV_NEGATIVE
    n_pos = len(_FPV_POSITIVE)

    text_inputs = processor(text=all_prompts, return_tensors="pt", padding=True, truncation=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items() if isinstance(v, torch.Tensor)}

    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    def _score_batch(frames_batch: list[np.ndarray]) -> np.ndarray:
        from PIL import Image

        images = [Image.fromarray(f) for f in frames_batch]
        image_inputs = processor(images=images, return_tensors="pt", padding=True)
        image_inputs = {k: v.to(device) for k, v in image_inputs.items() if isinstance(v, torch.Tensor)}

        with torch.no_grad():
            image_features = model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T)  # (B, n_prompts)

        sim_np = similarity.float().cpu().numpy()
        pos_score = sim_np[:, :n_pos].mean(axis=1)
        neg_score = sim_np[:, n_pos:].mean(axis=1)

        fpv_prob = np.clip(pos_score - neg_score + 0.5, 0.0, 1.0)
        return fpv_prob.astype(np.float32)

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
    threshold: float = 0.25,
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


def filter_video(
    frames: list[np.ndarray],
    *,
    device: str = "cpu",
    adaptive_threshold: float = 3.0,
    fpv_threshold: float = 0.25,
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
    "detect_shot_boundaries",
    "score_frames_fpv",
    "classify_shots",
    "video_verdict",
    "fpv_frame_mask",
    "thumbnail_grid_data",
    "filter_video",
]
