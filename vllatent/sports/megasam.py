"""MegaSaM ego-motion extraction wrapper (TOOL tier) — Phase B1 step 7.

Subprocess wrapper for MegaSaM (CVPR 2025). Parses SE(3) camera poses and
per-frame confidence from MegaSaM's output directory.

MegaSaM handles dynamic foreground by downweighting moving objects during
bundle adjustment — essential for skiing footage where the followed person
occupies 20-40% of the frame.

The wrapper is structured so the output parsing can adapt once MegaSaM's actual
format is confirmed. If MegaSaM proves unsuitable, the wrapper can be swapped
for DPVO/DROID-SLAM with the same ``MegaSamResult`` interface.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, eq=False)
class MegaSamResult:
    """Parsed MegaSaM output: camera trajectory + per-frame confidence."""

    poses: np.ndarray          # (N, 4, 4) f64 — SE(3) camera-to-world
    confidences: np.ndarray    # (N,) f64 — per-frame VO confidence in [0, 1]
    intrinsics: np.ndarray     # (3, 3) f64 — estimated camera intrinsics

    def __post_init__(self) -> None:
        if self.poses.ndim != 3 or self.poses.shape[1:] != (4, 4):
            raise ValueError(f"poses: expected (N, 4, 4), got {self.poses.shape}")
        n = self.poses.shape[0]
        if self.confidences.shape != (n,):
            raise ValueError(
                f"confidences: expected ({n},), got {self.confidences.shape}"
            )
        if self.intrinsics.shape != (3, 3):
            raise ValueError(
                f"intrinsics: expected (3, 3), got {self.intrinsics.shape}"
            )


def run_megasam(
    frame_dir: str | Path,
    out_dir: str | Path,
    *,
    megasam_path: str | Path | None = None,
    model: str = "megasam_base",
    extra_args: list[str] | None = None,
) -> MegaSamResult:
    """Run MegaSaM on a directory of frames and return parsed results.

    Parameters
    ----------
    frame_dir : directory containing sequential JPEG frames
    out_dir : directory for MegaSaM output
    megasam_path : path to MegaSaM installation (autodetected if None)
    model : MegaSaM model checkpoint name
    extra_args : additional CLI arguments
    """
    fdir = Path(frame_dir)
    odir = Path(out_dir)
    odir.mkdir(parents=True, exist_ok=True)

    megasam_root = Path(megasam_path) if megasam_path else _find_megasam()
    script = megasam_root / "run.py"
    if not script.exists():
        script = megasam_root / "demo.py"
    if not script.exists():
        raise FileNotFoundError(
            f"MegaSaM script not found in {megasam_root}. "
            "Expected run.py or demo.py."
        )

    cmd = [
        "python", str(script),
        "--input_dir", str(fdir),
        "--output_dir", str(odir),
        "--model", model,
    ]
    if extra_args:
        cmd.extend(extra_args)

    subprocess.run(cmd, check=True, timeout=1800)

    return parse_megasam_output(odir)


def _find_megasam() -> Path:
    """Try common locations for MegaSaM installation."""
    candidates = [
        Path.home() / "CODE" / "MegaSaM",
        Path.home() / "CODE" / "mega-sam",
        Path("/opt/megasam"),
    ]
    for p in candidates:
        if p.is_dir():
            return p
    raise FileNotFoundError(
        "MegaSaM not found. Set megasam_path explicitly or clone to ~/CODE/MegaSaM"
    )


def parse_megasam_output(out_dir: str | Path) -> MegaSamResult:
    """Parse MegaSaM output directory into a structured result.

    Handles multiple output formats (adapt as MegaSaM evolves):
    1. ``poses.npy`` + ``confidences.npy`` + ``intrinsics.npy``
    2. ``cameras.npz`` with keys ``poses``, ``confidences``, ``intrinsics``
    3. ``results.json`` with serialized arrays
    """
    odir = Path(out_dir)

    # Format 1: separate .npy files
    poses_path = odir / "poses.npy"
    if poses_path.exists():
        poses = np.load(str(poses_path))
        conf_path = odir / "confidences.npy"
        confidences = np.load(str(conf_path)) if conf_path.exists() else np.ones(poses.shape[0])
        intr_path = odir / "intrinsics.npy"
        intrinsics = np.load(str(intr_path)) if intr_path.exists() else _default_intrinsics()
        return MegaSamResult(
            poses=poses.astype(np.float64),
            confidences=confidences.astype(np.float64),
            intrinsics=intrinsics.astype(np.float64),
        )

    # Format 2: single .npz
    npz_path = odir / "cameras.npz"
    if npz_path.exists():
        with np.load(str(npz_path)) as data:
            poses = data["poses"].astype(np.float64)
            confidences = data.get("confidences", np.ones(poses.shape[0])).astype(np.float64)
            intrinsics = data.get("intrinsics", _default_intrinsics()).astype(np.float64)
        return MegaSamResult(poses=poses, confidences=confidences, intrinsics=intrinsics)

    # Format 3: JSON
    json_path = odir / "results.json"
    if json_path.exists():
        raw = json.loads(json_path.read_text())
        poses = np.array(raw["poses"], dtype=np.float64)
        confidences = np.array(raw.get("confidences", [1.0] * len(poses)), dtype=np.float64)
        intrinsics = np.array(raw.get("intrinsics", _default_intrinsics().tolist()), dtype=np.float64)
        return MegaSamResult(poses=poses, confidences=confidences, intrinsics=intrinsics)

    raise FileNotFoundError(
        f"No recognized MegaSaM output in {odir}. "
        "Expected poses.npy, cameras.npz, or results.json"
    )


def _default_intrinsics() -> np.ndarray:
    """Default pinhole intrinsics (placeholder; overwritten by MegaSaM's estimate)."""
    return np.array([
        [500.0, 0.0, 640.0],
        [0.0, 500.0, 360.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def validate_megasam_result(result: MegaSamResult) -> list[str]:
    """Validate a MegaSaM result. Returns list of errors (empty = valid)."""
    errors: list[str] = []
    n = result.poses.shape[0]

    if n < 2:
        errors.append(f"Too few poses: {n} (need >= 2)")

    for i in range(n):
        R = result.poses[i, :3, :3]
        det = np.linalg.det(R)
        if abs(det - 1.0) > 0.01:
            errors.append(f"Pose {i}: det(R) = {det:.4f}, expected ~1.0")
            break
        orth = np.max(np.abs(R @ R.T - np.eye(3)))
        if orth > 0.01:
            errors.append(f"Pose {i}: R not orthogonal (max err {orth:.4f})")
            break

    identity_count = sum(
        1 for i in range(n)
        if np.allclose(result.poses[i], np.eye(4), atol=1e-6)
    )
    if identity_count > n * 0.5:
        errors.append(f"Degenerate: {identity_count}/{n} poses are identity")

    low_conf = np.sum(result.confidences < 0.1)
    if low_conf > n * 0.5:
        errors.append(f"Low confidence: {low_conf}/{n} frames below 0.1")

    return errors


__all__ = [
    "MegaSamResult",
    "run_megasam",
    "parse_megasam_output",
    "validate_megasam_result",
]
