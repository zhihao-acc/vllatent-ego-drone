"""Frame extraction and preprocessing (TOOL tier).

ffmpeg subprocess for frame extraction at uniform FPS. Optional fisheye
undistortion via cv2 (lazy import — not required for pinhole cameras).
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FrameExtraction:
    """Result of extracting frames from a video clip."""

    frame_dir: Path
    n_frames: int
    fps: float
    width: int
    height: int


def extract_frames(
    video_path: str | Path,
    out_dir: str | Path,
    target_fps: float = 5.0,
    resolution_hw: tuple[int, int] | None = None,
) -> FrameExtraction:
    """Extract frames from a video at uniform FPS via ffmpeg."""
    vpath = Path(video_path)
    odir = Path(out_dir)
    odir.mkdir(parents=True, exist_ok=True)

    vf_parts = [f"fps={target_fps}"]
    if resolution_hw is not None:
        h, w = resolution_hw
        vf_parts.append(f"scale={w}:{h}")
    vf = ",".join(vf_parts)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(vpath),
        "-vf", vf,
        "-q:v", "2",
        "-start_number", "0",
        str(odir / "%06d.jpg"),
    ]

    subprocess.run(cmd, check=True, capture_output=True, timeout=300)

    frames = sorted(odir.glob("*.jpg"))
    n_frames = len(frames)
    if n_frames == 0:
        raise RuntimeError(f"No frames extracted from {vpath}")

    w_out, h_out = _probe_frame_size(frames[0])

    return FrameExtraction(
        frame_dir=odir,
        n_frames=n_frames,
        fps=target_fps,
        width=w_out,
        height=h_out,
    )


def _probe_frame_size(frame_path: Path) -> tuple[int, int]:
    """Get (width, height) of an image via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json",
            str(frame_path),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    return int(stream["width"]), int(stream["height"])


def load_frame(path: str | Path) -> np.ndarray:
    """Load a JPEG frame as RGB uint8 numpy array."""
    from vllatent.io import load_rgb
    return load_rgb(path)


def load_frames(frame_dir: str | Path) -> np.ndarray:
    """Load all JPEG frames from a directory as a (N, H, W, 3) uint8 array."""
    paths = sorted(Path(frame_dir).glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No .jpg frames in {frame_dir}")
    frames = [load_frame(p) for p in paths]
    return np.stack(frames)


def cut_fixed_clips(
    frame_paths: list[Path],
    clip_length_frames: int,
    min_usable_frames: int = 7,
) -> list[list[Path]]:
    """Split frame paths into non-overlapping segments of fixed length.

    Trailing segments shorter than ``min_usable_frames`` (default = H+T = 7) are discarded.

    Args:
        frame_paths: Sorted list of frame file paths.
        clip_length_frames: Number of frames per segment (= clip_length_seconds * fps).
        min_usable_frames: Minimum frames for a segment to be kept.

    Returns:
        List of frame-path lists, one per usable segment.
    """
    if clip_length_frames < min_usable_frames:
        raise ValueError(
            f"clip_length_frames ({clip_length_frames}) must be >= "
            f"min_usable_frames ({min_usable_frames})"
        )
    segments: list[list[Path]] = []
    for start in range(0, len(frame_paths), clip_length_frames):
        seg = frame_paths[start:start + clip_length_frames]
        if len(seg) >= min_usable_frames:
            segments.append(seg)
    return segments


def undistort_fisheye(
    frame: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    new_K: np.ndarray | None = None,
) -> np.ndarray:
    """Undistort a fisheye frame using OpenCV's fisheye model."""
    import cv2

    if new_K is None:
        new_K = K
    h, w = frame.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2,
    )
    return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)


def batch_undistort(
    frame_dir: str | Path,
    out_dir: str | Path,
    K: np.ndarray,
    D: np.ndarray,
) -> int:
    """Undistort all frames in a directory, writing to out_dir. Returns frame count."""
    import cv2

    fdir = Path(frame_dir)
    odir = Path(out_dir)
    odir.mkdir(parents=True, exist_ok=True)

    paths = sorted(fdir.glob("*.jpg"))
    new_K = K.copy()

    for p in paths:
        frame = load_frame(p)
        undistorted = undistort_fisheye(frame, K, D, new_K)
        bgr = cv2.cvtColor(undistorted, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(odir / p.name), bgr)

    return len(paths)


__all__ = [
    "FrameExtraction",
    "cut_fixed_clips",
    "extract_frames",
    "load_frame",
    "load_frames",
    "undistort_fisheye",
    "batch_undistort",
]
