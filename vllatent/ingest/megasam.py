"""MegaSaM ego-motion extraction wrapper (TOOL tier).

Subprocess wrapper for MegaSaM (CVPR 2025). Parses SE(3) camera poses and
per-frame confidence from MegaSaM's output directory.

MegaSaM handles dynamic foreground by downweighting moving objects during
bundle adjustment — essential for footage where the followed person
occupies 20-40% of the frame.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CONFIDENCE_SOURCES = ("real", "default")
_MEGASAM_LIE7_DIM = 7  # [x, y, z, qx, qy, qz, qw]


@dataclass(frozen=True, eq=False)
class MegaSamResult:
    """Parsed MegaSaM output: camera trajectory + per-frame confidence."""

    poses: np.ndarray          # (N, 4, 4) f64 — SE(3) camera-to-world
    confidences: np.ndarray    # (N,) f64 — per-frame VO confidence in [0, 1]
    intrinsics: np.ndarray     # (3, 3) f64 — estimated camera intrinsics
    confidence_source: str = "real"  # "real" = from MegaSaM output, "default" = np.ones fallback

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
        if self.confidence_source not in CONFIDENCE_SOURCES:
            raise ValueError(
                f"confidence_source must be one of {CONFIDENCE_SOURCES}, "
                f"got {self.confidence_source!r}"
            )


def _quat_xyzw_to_rotation(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [qx, qy, qz, qw] to 3x3 rotation matrix."""
    qx, qy, qz, qw = q[0], q[1], q[2], q[3]
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)


def lie7_to_c2w_matrices(lie7: np.ndarray) -> np.ndarray:
    """Convert MegaSaM's (T, 7) w2c Lie group poses to (T, 4, 4) c2w SE(3).

    MegaSaM stores poses as [x, y, z, qx, qy, qz, qw] in world-to-camera
    convention. This function converts each to a 4x4 homogeneous matrix and
    inverts to camera-to-world (matching our pipeline's expectation).
    """
    if lie7.ndim != 2 or lie7.shape[1] != _MEGASAM_LIE7_DIM:
        raise ValueError(
            f"expected (T, {_MEGASAM_LIE7_DIM}), got {lie7.shape}"
        )
    n = lie7.shape[0]
    c2w = np.empty((n, 4, 4), dtype=np.float64)
    for i in range(n):
        t_w2c = lie7[i, :3]
        q_w2c = lie7[i, 3:]
        R_w2c = _quat_xyzw_to_rotation(q_w2c)
        # Invert: c2w = [R^T, -R^T @ t; 0 0 0 1]
        R_c2w = R_w2c.T
        t_c2w = -R_c2w @ t_w2c
        c2w[i, :3, :3] = R_c2w
        c2w[i, :3, 3] = t_c2w
        c2w[i, 3, :] = [0, 0, 0, 1]
    return c2w


def aggregate_motion_prob(motion_prob: np.ndarray) -> np.ndarray:
    """Aggregate per-pixel motion probability to per-frame confidence.

    Input: (T, H, W) motion probability from MegaSaM's bundle adjustment.
    Output: (T,) confidence in [0, 1] — spatial mean per frame.
    """
    return np.mean(motion_prob, axis=(1, 2)).astype(np.float64)


def _intrinsics_4vec_to_3x3(intr_4vec: np.ndarray) -> np.ndarray:
    """Convert MegaSaM's [fx, fy, cx, cy] (×8.0 scaled) to (3, 3) K matrix."""
    fx, fy, cx, cy = intr_4vec / 8.0
    return np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def run_megasam(
    frame_dir: str | Path,
    out_dir: str | Path,
    *,
    clip_id: str = "",
    megasam_path: str | Path | None = None,
    gpu: int = 0,
    encoder: str = "vitl",
    conda_env: str = "mega_sam",
) -> MegaSamResult:
    """Run MegaSaM 3-step pipeline on a directory of frames.

    Uses ``scripts/run_megasam_pipeline.sh`` which orchestrates:
    DepthAnything → UniDepth → camera_tracking.
    """
    fdir = Path(frame_dir)
    odir = Path(out_dir)

    megasam_root = Path(megasam_path) if megasam_path else _find_megasam()
    pipeline_script = Path(__file__).resolve().parents[2] / "scripts" / "run_megasam_pipeline.sh"
    if not pipeline_script.exists():
        raise FileNotFoundError(
            f"Pipeline script not found: {pipeline_script}"
        )

    if not clip_id:
        clip_id = fdir.name

    cmd = [
        "bash", str(pipeline_script),
        "--clip-id", clip_id,
        "--frames-dir", str(fdir),
        "--megasam-dir", str(megasam_root),
        "--gpu", str(gpu),
        "--encoder", encoder,
        "--out-dir", str(odir),
        "--conda-env", conda_env,
    ]

    subprocess.run(cmd, check=True, timeout=3600)

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

    Supports three output layouts (tried in priority order):

    1. **Real MegaSaM format** (``reconstructions/{scene}/``):
       - ``poses.npy`` — ``(T, 7)`` w2c Lie group ``[x,y,z,qx,qy,qz,qw]``
       - ``motion_prob.npy`` — ``(T, H/8, W/8)`` per-pixel confidence
       - ``intrinsics.npy`` — ``(T, 4)`` ``[fx,fy,cx,cy]`` × 8.0
    2. **droid.npz** (``outputs/{scene}_droid.npz``):
       - ``cam_c2w`` — ``(T, 4, 4)`` c2w matrices
       - ``intrinsic`` — ``(3, 3)`` K matrix
    3. **Legacy flat formats** (backward compat):
       - ``poses.npy`` ``(N, 4, 4)`` / ``cameras.npz`` / ``results.json``
    """
    import logging
    log = logging.getLogger(__name__)
    odir = Path(out_dir)

    # --- Format 1: Real MegaSaM reconstructions/ directory ---
    poses_path = odir / "poses.npy"
    if poses_path.exists():
        poses_raw = np.load(str(poses_path))

        # Detect real MegaSaM format: (T, 7) Lie group
        if poses_raw.ndim == 2 and poses_raw.shape[1] == _MEGASAM_LIE7_DIM:
            log.info("Parsing MegaSaM real format (T,7) from %s", odir)
            poses_c2w = lie7_to_c2w_matrices(poses_raw)

            # Motion probability → per-frame confidence
            mp_path = odir / "motion_prob.npy"
            if mp_path.exists():
                motion_prob = np.load(str(mp_path))
                confidences = aggregate_motion_prob(motion_prob)
                conf_source = "real"
            else:
                log.warning("motion_prob.npy missing in %s — using np.ones fallback", odir)
                confidences = np.ones(poses_c2w.shape[0], dtype=np.float64)
                conf_source = "default"

            # Intrinsics: (T, 4) vector → (3, 3) K matrix from first frame
            intr_path = odir / "intrinsics.npy"
            if intr_path.exists():
                intr_raw = np.load(str(intr_path))
                if intr_raw.ndim == 2 and intr_raw.shape[1] == 4:
                    intrinsics = _intrinsics_4vec_to_3x3(intr_raw[0])
                elif intr_raw.ndim == 1 and intr_raw.shape[0] == 4:
                    intrinsics = _intrinsics_4vec_to_3x3(intr_raw)
                else:
                    intrinsics = intr_raw.astype(np.float64)
            else:
                intrinsics = _default_intrinsics()

            return MegaSamResult(
                poses=poses_c2w,
                confidences=confidences.astype(np.float64),
                intrinsics=intrinsics.astype(np.float64),
                confidence_source=conf_source,
            )

        # Legacy format: (N, 4, 4) SE(3) matrices
        if poses_raw.ndim == 3 and poses_raw.shape[1:] == (4, 4):
            log.info("Parsing legacy (N,4,4) poses from %s", odir)
            poses = poses_raw
            conf_path = odir / "confidences.npy"
            if conf_path.exists():
                confidences = np.load(str(conf_path))
                conf_source = "real"
            else:
                log.warning("confidences.npy missing in %s — using np.ones fallback", odir)
                confidences = np.ones(poses.shape[0])
                conf_source = "default"
            intr_path = odir / "intrinsics.npy"
            if intr_path.exists():
                intr_raw = np.load(str(intr_path))
                if intr_raw.ndim == 2 and intr_raw.shape[1] == 4:
                    intrinsics = _intrinsics_4vec_to_3x3(intr_raw[0])
                elif intr_raw.ndim == 1 and intr_raw.shape[0] == 4:
                    intrinsics = _intrinsics_4vec_to_3x3(intr_raw)
                else:
                    intrinsics = intr_raw
            else:
                intrinsics = _default_intrinsics()
            return MegaSamResult(
                poses=poses.astype(np.float64),
                confidences=confidences.astype(np.float64),
                intrinsics=intrinsics.astype(np.float64),
                confidence_source=conf_source,
            )

    # --- Format 2: droid.npz (direct path to .npz file or dir containing it) ---
    if odir.suffix == ".npz" and odir.exists():
        npz_path = odir
    else:
        npz_candidates = list(odir.glob("*_droid.npz"))
        npz_path = npz_candidates[0] if npz_candidates else None

    if npz_path and npz_path.exists():
        with np.load(str(npz_path)) as data:
            if "cam_c2w" in data:
                log.info("Parsing droid.npz format from %s", npz_path)
                poses = data["cam_c2w"].astype(np.float64)
                if "intrinsic" in data:
                    intrinsics = data["intrinsic"].astype(np.float64)
                else:
                    intrinsics = _default_intrinsics()
                confidences = np.ones(poses.shape[0], dtype=np.float64)
                return MegaSamResult(
                    poses=poses,
                    confidences=confidences,
                    intrinsics=intrinsics,
                    confidence_source="default",
                )
            # Legacy cameras.npz
            if "poses" in data:
                log.info("Parsing legacy cameras.npz from %s", npz_path)
                poses = data["poses"].astype(np.float64)
                if "confidences" in data:
                    confidences = data["confidences"].astype(np.float64)
                    conf_source = "real"
                else:
                    log.warning("cameras.npz missing 'confidences' in %s — using np.ones", npz_path)
                    confidences = np.ones(poses.shape[0], dtype=np.float64)
                    conf_source = "default"
                intrinsics = data.get("intrinsics", _default_intrinsics()).astype(np.float64)
                return MegaSamResult(
                    poses=poses, confidences=confidences, intrinsics=intrinsics,
                    confidence_source=conf_source,
                )

    # --- Format 3: Legacy cameras.npz / results.json ---
    npz_path = odir / "cameras.npz"
    if npz_path.exists():
        with np.load(str(npz_path)) as data:
            poses = data["poses"].astype(np.float64)
            if "confidences" in data:
                confidences = data["confidences"].astype(np.float64)
                conf_source = "real"
            else:
                log.warning("cameras.npz missing 'confidences' in %s — using np.ones", odir)
                confidences = np.ones(poses.shape[0], dtype=np.float64)
                conf_source = "default"
            intrinsics = data.get("intrinsics", _default_intrinsics()).astype(np.float64)
        return MegaSamResult(
            poses=poses, confidences=confidences, intrinsics=intrinsics,
            confidence_source=conf_source,
        )

    json_path = odir / "results.json"
    if json_path.exists():
        raw = json.loads(json_path.read_text())
        poses = np.array(raw["poses"], dtype=np.float64)
        if "confidences" in raw:
            confidences = np.array(raw["confidences"], dtype=np.float64)
            conf_source = "real"
        else:
            log.warning("results.json missing 'confidences' in %s — using np.ones", odir)
            confidences = np.ones(len(poses), dtype=np.float64)
            conf_source = "default"
        intrinsics = np.array(raw.get("intrinsics", _default_intrinsics().tolist()), dtype=np.float64)
        return MegaSamResult(
            poses=poses, confidences=confidences, intrinsics=intrinsics,
            confidence_source=conf_source,
        )

    raise FileNotFoundError(
        f"No recognized MegaSaM output in {odir}. "
        "Expected reconstructions/{scene}/ with poses.npy (T,7), "
        "*_droid.npz with cam_c2w, or legacy poses.npy (N,4,4) / cameras.npz / results.json"
    )


def _default_intrinsics() -> np.ndarray:
    """Default pinhole intrinsics (placeholder)."""
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
    "CONFIDENCE_SOURCES",
    "MegaSamResult",
    "aggregate_motion_prob",
    "lie7_to_c2w_matrices",
    "run_megasam",
    "parse_megasam_output",
    "validate_megasam_result",
]
