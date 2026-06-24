"""CosFly-Track adapter — converts CARLA GT 6-DoF traces to ingest .npz cache (PURE tier).

CosFly-Track (AutelRobotics/CosFly on HuggingFace) is a CARLA drone-tracking dataset
with GT 6-DoF poses. Each trace has a ``trajectory.json`` with per-frame ``drone_pose``
fields ``(x, y, z, pitch, yaw, roll)`` in metres/degrees. The adapter converts these
GT poses to the ingest ``.npz`` cache contract, skipping MegaSaM entirely since the
poses are ground truth (``vo_confidence = 1.0``).

Directory layout (HF download)::

    data_v7/Town01/trajectory_0000/ORI/
        trajectory.json          # waypoints with drone_pose + timing
        frames_playback/frame_00000/rgb.png   # CARLA rendered frame

Deltas are computed as position differences + yaw differences between consecutive
GT poses. No monocular VO noise — this is the "clean data curriculum" that gets
oversampled to ~40% of training batches (per Phase B plan).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vllatent.manifest import build_manifest_wild_video
from vllatent.schemas import DELTA_DTYPE, MASK_DTYPE


@dataclass(frozen=True)
class TrajectoryData:
    """Parsed CosFly trace — GT poses + timestamps + frame paths."""
    poses: np.ndarray         # (N, 6) f64 — x, y, z, pitch, yaw, roll (m/deg)
    timestamps: np.ndarray    # (N,) f64 — seconds
    frame_paths: list[Path]   # sorted RGB paths
    fps: float                # inferred from timestamps


def parse_trajectory(trace_dir: Path | str) -> TrajectoryData:
    """Parse a single CosFly trace directory into structured data.

    Args:
        trace_dir: path to e.g. ``data_v7/Town01/trajectory_0000/ORI/``

    Returns:
        TrajectoryData with GT poses, timestamps, and frame paths.

    Raises:
        FileNotFoundError: if trajectory.json is missing.
    """
    trace_dir = Path(trace_dir)
    traj_path = trace_dir / "trajectory.json"
    if not traj_path.exists():
        raise FileNotFoundError(f"trajectory.json not found in {trace_dir}")

    traj = json.loads(traj_path.read_text())
    waypoints = traj["waypoints"]

    n = len(waypoints)
    poses = np.empty((n, 6), dtype=np.float64)
    timestamps = np.empty(n, dtype=np.float64)

    for i, wp in enumerate(waypoints):
        dp = wp["drone_pose"]
        poses[i] = [dp["x"], dp["y"], dp["z"], dp["pitch"], dp["yaw"], dp["roll"]]
        timestamps[i] = wp["timing"]["timestamp"]

    frames_dir = trace_dir / "frames_playback"
    frame_paths: list[Path] = []
    if frames_dir.exists():
        frame_paths = sorted(
            (d / "rgb.png" for d in sorted(frames_dir.iterdir()) if d.is_dir()),
            key=lambda p: p.parent.name,
        )

    fps = _infer_fps(timestamps)

    return TrajectoryData(poses=poses, timestamps=timestamps, frame_paths=frame_paths, fps=fps)


def _infer_fps(timestamps: np.ndarray) -> float:
    """Infer FPS from timestamp deltas (median of 1/dt)."""
    if len(timestamps) < 2:
        return 2.0
    dts = np.diff(timestamps)
    dts = dts[dts > 0]
    if len(dts) == 0:
        return 2.0
    return float(1.0 / np.median(dts))


def poses_to_deltas(poses: np.ndarray) -> np.ndarray:
    """Convert GT 6-DoF poses to body-frame deltas (dx, dy, dz, dyaw_deg).

    CosFly poses are ``(x, y, z, pitch, yaw, roll)`` in world frame. Deltas
    are simple differences (GT world frame, not relative body frame — CARLA
    world coords are sufficient for the training target since the student
    learns body-frame prediction from the delta pattern, and the scale is
    exact).

    Args:
        poses: (N, 6) f64 — x, y, z, pitch, yaw, roll

    Returns:
        (N-1, 4) f32 — dx, dy, dz, dyaw_deg

    Raises:
        ValueError: if fewer than 2 poses.
    """
    if poses.shape[0] < 2:
        raise ValueError(f"need >= 2 poses for deltas, got {poses.shape[0]}")

    dx = np.diff(poses[:, 0])
    dy = np.diff(poses[:, 1])
    dz = np.diff(poses[:, 2])
    dyaw = np.diff(poses[:, 4])

    deltas = np.stack([dx, dy, dz, dyaw], axis=1).astype(DELTA_DTYPE)
    return deltas


@dataclass(frozen=True)
class ConvertResult:
    """Result of converting a single CosFly trace."""
    clip_id: str
    n_frames: int
    npz_path: Path


def convert_trace(
    trace_dir: Path | str,
    out_dir: Path | str,
    *,
    clip_id: str | None = None,
) -> ConvertResult:
    """Convert a single CosFly trace to the ingest .npz cache contract.

    Writes: ``<out_dir>/<clip_id>.npz`` with keys matching
    ``_build_clip_npz`` in ``pipeline.py``:

    - ``deltas`` (N-1, 4) f32
    - ``vo_confidence`` (N,) f32 — all 1.0 (GT)
    - ``frame_quality`` (N,) f32 — all 1.0 (perfect CARLA renders)
    - ``timestamps`` (N,) f64
    - ``quality_mask`` (N,) bool — all True

    Note: ``latents`` are NOT written here — DINOv3 encoding is done
    separately (B1.7a ``encode_frames``). This adapter produces the
    motion/metadata half of the cache.
    """
    trace_dir = Path(trace_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = parse_trajectory(trace_dir)

    if clip_id is None:
        clip_id = _clip_id_from_path(trace_dir)

    n = data.poses.shape[0]
    deltas = poses_to_deltas(data.poses)

    arrays = {
        "deltas": deltas,
        "vo_confidence": np.ones(n, dtype=np.float32),
        "frame_quality": np.ones(n, dtype=np.float32),
        "timestamps": data.timestamps,
        "quality_mask": np.ones(n, dtype=MASK_DTYPE),
    }

    npz_path = out_dir / f"{clip_id}.npz"
    np.savez(str(npz_path), **arrays)

    return ConvertResult(clip_id=clip_id, n_frames=n, npz_path=npz_path)


def _clip_id_from_path(trace_dir: Path) -> str:
    """Derive a clip ID from the directory path.

    e.g. ``data_v7/Town03/trajectory_0042/ORI`` → ``cosfly_Town03_0042_ORI``
    """
    parts = trace_dir.parts
    variant = parts[-1] if parts else "unknown"
    traj_part = parts[-2] if len(parts) >= 2 else "traj_0000"
    town_part = parts[-3] if len(parts) >= 3 else "Town00"

    traj_num = traj_part.replace("trajectory_", "")
    return f"cosfly_{town_part}_{traj_num}_{variant}"


def discover_traces(data_dir: Path | str) -> list[Path]:
    """Find all CosFly trace directories under a data_v7 root.

    Returns sorted list of trace dirs (each containing trajectory.json).
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []

    traces = sorted(
        p.parent for p in data_dir.rglob("trajectory.json")
    )
    return traces


def build_cosfly_manifest(
    *,
    entries: list[dict[str, Any]] | None = None,
    source_fps: float = 2.0,
) -> dict[str, Any]:
    """Build a wild-video manifest for CosFly-Track data."""
    m = build_manifest_wild_video(
        encoder_model_id="vit_base_patch16_dinov3.lvd1689m",
        motion_method="cosfly_gt",
        motion_model="carla_gt",
        scale_mode="exact",
        source_fps=source_fps,
        entries=entries,
    )
    m["dataset"]["name"] = "cosfly_track"
    m["dataset"]["license"] = "Apache-2.0"
    return m


def _log(msg: str) -> None:
    print(f"[cosfly] {msg}", file=sys.stderr)


def convert_dataset(
    data_dir: Path | str,
    out_dir: Path | str,
    *,
    limit: int | None = None,
    skip_existing: bool = True,
) -> list[ConvertResult]:
    """Convert all CosFly traces in a data_v7 directory.

    Args:
        data_dir: path to the ``data_v7/`` directory.
        out_dir: directory for .npz output files.
        limit: max traces to convert (None = all).
        skip_existing: skip traces whose .npz already exists.

    Returns:
        List of ConvertResult for each successfully converted trace.
    """
    traces = discover_traces(data_dir)
    if limit is not None:
        traces = traces[:limit]

    _log(f"discovered {len(traces)} traces")

    results: list[ConvertResult] = []
    out_dir = Path(out_dir)

    for trace_dir in traces:
        clip_id = _clip_id_from_path(trace_dir)
        npz_path = out_dir / f"{clip_id}.npz"

        if skip_existing and npz_path.exists():
            _log(f"skipping {clip_id}: already cached")
            continue

        try:
            result = convert_trace(trace_dir, out_dir, clip_id=clip_id)
            results.append(result)
            _log(f"OK: {clip_id} ({result.n_frames} frames)")
        except Exception as exc:
            _log(f"ERROR: {clip_id}: {exc}")

    return results


__all__ = [
    "TrajectoryData",
    "ConvertResult",
    "parse_trajectory",
    "poses_to_deltas",
    "convert_trace",
    "discover_traces",
    "build_cosfly_manifest",
    "convert_dataset",
]
