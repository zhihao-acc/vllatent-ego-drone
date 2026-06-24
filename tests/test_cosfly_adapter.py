"""Tests for CosFly-Track adapter (PURE tier — no network, no torch).

CosFly-Track is a CARLA drone-tracking dataset from AutelRobotics/CosFly on HF.
Each trace has a trajectory.json with GT 6-DoF drone poses. The adapter converts
these to the ingest .npz cache contract (latents placeholder, GT deltas,
vo_confidence=1.0).
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from vllatent.schemas import DELTA_DTYPE

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_cosfly_trace(tmp_path: Path, *, n_frames: int = 20, town: str = "Town01",
                       traj_id: int = 0, variant: str = "ORI") -> Path:
    """Create a minimal CosFly trace directory with trajectory.json + frame PNGs."""
    root = tmp_path / "data_v7" / town / f"trajectory_{traj_id:04d}" / variant
    frames_dir = root / "frames_playback"
    frames_dir.mkdir(parents=True)

    waypoints = []
    for i in range(n_frames):
        frame_dir = frames_dir / f"frame_{i:05d}"
        frame_dir.mkdir()
        # Minimal 1x1 black PNG (8 bytes header + minimal IDAT)
        (frame_dir / "rgb.png").write_bytes(
            b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
        )
        waypoints.append({
            "index": i,
            "drone_pose": {
                "x": float(i * 0.5),
                "y": float(i * 0.1),
                "z": 10.0 + float(i * 0.05),
                "pitch": 0.0,
                "yaw": float(i * 2.0),
                "roll": 0.0,
            },
            "timing": {"timestamp": float(i * 0.5)},
        })

    traj = {"waypoints": waypoints}
    (root / "trajectory.json").write_text(json.dumps(traj))
    return root


def _make_cosfly_dataset(tmp_path: Path, *, n_traces: int = 3,
                         n_frames: int = 10) -> Path:
    """Create a multi-trace CosFly dataset under tmp_path."""
    for i in range(n_traces):
        for variant in ("ORI", "aug_001"):
            _make_cosfly_trace(
                tmp_path, n_frames=n_frames,
                town=f"Town{(i % 3) + 1:02d}", traj_id=i, variant=variant,
            )
    return tmp_path / "data_v7"


# ---------------------------------------------------------------------------
# Tests — parse_trajectory
# ---------------------------------------------------------------------------

class TestParseTrajectory:
    def test_parses_poses_shape(self, tmp_path: Path) -> None:
        from vllatent.ingest.cosfly_adapter import parse_trajectory
        root = _make_cosfly_trace(tmp_path, n_frames=15)
        result = parse_trajectory(root)
        assert result.poses.shape == (15, 6), f"Expected (15, 6), got {result.poses.shape}"

    def test_pose_columns_order(self, tmp_path: Path) -> None:
        """Poses should be (x, y, z, pitch, yaw, roll) — matching trajectory.json."""
        from vllatent.ingest.cosfly_adapter import parse_trajectory
        root = _make_cosfly_trace(tmp_path, n_frames=5)
        result = parse_trajectory(root)
        # Frame 2: x=1.0, y=0.2, z=10.1, pitch=0, yaw=4.0, roll=0
        np.testing.assert_allclose(result.poses[2, 0], 1.0, atol=1e-6)
        np.testing.assert_allclose(result.poses[2, 1], 0.2, atol=1e-6)
        np.testing.assert_allclose(result.poses[2, 4], 4.0, atol=1e-6)

    def test_timestamps_from_timing(self, tmp_path: Path) -> None:
        from vllatent.ingest.cosfly_adapter import parse_trajectory
        root = _make_cosfly_trace(tmp_path, n_frames=10)
        result = parse_trajectory(root)
        assert result.timestamps.shape == (10,)
        np.testing.assert_allclose(result.timestamps[3], 1.5, atol=1e-6)

    def test_frame_paths_sorted(self, tmp_path: Path) -> None:
        from vllatent.ingest.cosfly_adapter import parse_trajectory
        root = _make_cosfly_trace(tmp_path, n_frames=8)
        result = parse_trajectory(root)
        assert len(result.frame_paths) == 8
        for i in range(len(result.frame_paths) - 1):
            assert result.frame_paths[i] < result.frame_paths[i + 1]

    def test_missing_trajectory_json_raises(self, tmp_path: Path) -> None:
        from vllatent.ingest.cosfly_adapter import parse_trajectory
        root = tmp_path / "empty_trace"
        root.mkdir()
        with pytest.raises(FileNotFoundError):
            parse_trajectory(root)

    def test_infers_fps_from_timestamps(self, tmp_path: Path) -> None:
        from vllatent.ingest.cosfly_adapter import parse_trajectory
        root = _make_cosfly_trace(tmp_path, n_frames=10)
        result = parse_trajectory(root)
        assert result.fps == pytest.approx(2.0, abs=0.1)


# ---------------------------------------------------------------------------
# Tests — poses_to_deltas
# ---------------------------------------------------------------------------

class TestPosesToDeltas:
    def test_output_shape(self) -> None:
        from vllatent.ingest.cosfly_adapter import poses_to_deltas
        poses = np.zeros((10, 6), dtype=np.float64)
        poses[:, 0] = np.arange(10) * 0.5  # x moves forward
        deltas = poses_to_deltas(poses)
        assert deltas.shape == (9, 4)
        assert deltas.dtype == DELTA_DTYPE

    def test_pure_forward_motion(self) -> None:
        from vllatent.ingest.cosfly_adapter import poses_to_deltas
        poses = np.zeros((5, 6), dtype=np.float64)
        poses[:, 0] = np.arange(5) * 1.0  # x increments by 1.0
        deltas = poses_to_deltas(poses)
        # dx should be 1.0 for all transitions
        np.testing.assert_allclose(deltas[:, 0], 1.0, atol=1e-5)
        # dy, dz should be 0
        np.testing.assert_allclose(deltas[:, 1], 0.0, atol=1e-5)
        np.testing.assert_allclose(deltas[:, 2], 0.0, atol=1e-5)

    def test_yaw_delta_in_degrees(self) -> None:
        from vllatent.ingest.cosfly_adapter import poses_to_deltas
        poses = np.zeros((4, 6), dtype=np.float64)
        poses[:, 4] = [0.0, 10.0, 25.0, 30.0]  # yaw in degrees
        deltas = poses_to_deltas(poses)
        np.testing.assert_allclose(deltas[:, 3], [10.0, 15.0, 5.0], atol=1e-5)

    def test_too_few_poses_raises(self) -> None:
        from vllatent.ingest.cosfly_adapter import poses_to_deltas
        with pytest.raises(ValueError, match="need >= 2"):
            poses_to_deltas(np.zeros((1, 6)))


# ---------------------------------------------------------------------------
# Tests — convert_trace (full adapter)
# ---------------------------------------------------------------------------

class TestConvertTrace:
    def test_produces_npz_arrays(self, tmp_path: Path) -> None:
        from vllatent.ingest.cosfly_adapter import convert_trace
        root = _make_cosfly_trace(tmp_path, n_frames=20)
        out_dir = tmp_path / "cache"
        result = convert_trace(root, out_dir)

        assert result.clip_id is not None
        npz_path = out_dir / f"{result.clip_id}.npz"
        assert npz_path.exists()

        with np.load(str(npz_path)) as data:
            n = data["deltas"].shape[0] + 1
            assert data["deltas"].shape == (n - 1, 4)
            assert data["deltas"].dtype == DELTA_DTYPE
            assert data["vo_confidence"].shape == (n,)
            assert data["frame_quality"].shape == (n,)
            assert data["timestamps"].shape == (n,)
            assert data["quality_mask"].shape == (n,)

    def test_vo_confidence_all_ones(self, tmp_path: Path) -> None:
        """CosFly has GT poses → vo_confidence must be 1.0 everywhere."""
        from vllatent.ingest.cosfly_adapter import convert_trace
        root = _make_cosfly_trace(tmp_path, n_frames=15)
        result = convert_trace(root, tmp_path / "cache")

        with np.load(str(tmp_path / "cache" / f"{result.clip_id}.npz")) as data:
            np.testing.assert_array_equal(data["vo_confidence"], 1.0)

    def test_frame_quality_all_ones(self, tmp_path: Path) -> None:
        """GT CARLA data → frame_quality = 1.0 (perfect synthetic frames)."""
        from vllatent.ingest.cosfly_adapter import convert_trace
        root = _make_cosfly_trace(tmp_path, n_frames=10)
        result = convert_trace(root, tmp_path / "cache")

        with np.load(str(tmp_path / "cache" / f"{result.clip_id}.npz")) as data:
            np.testing.assert_array_equal(data["frame_quality"], 1.0)

    def test_clip_id_encodes_town_and_traj(self, tmp_path: Path) -> None:
        from vllatent.ingest.cosfly_adapter import convert_trace
        root = _make_cosfly_trace(tmp_path, n_frames=10, town="Town03", traj_id=42)
        result = convert_trace(root, tmp_path / "cache")
        assert "Town03" in result.clip_id
        assert "0042" in result.clip_id


# ---------------------------------------------------------------------------
# Tests — discover_traces
# ---------------------------------------------------------------------------

class TestDiscoverTraces:
    def test_finds_all_traces(self, tmp_path: Path) -> None:
        from vllatent.ingest.cosfly_adapter import discover_traces
        _make_cosfly_dataset(tmp_path, n_traces=3, n_frames=5)
        traces = discover_traces(tmp_path / "data_v7")
        assert len(traces) == 6  # 3 traj × 2 variants (ORI + aug_001)

    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        from vllatent.ingest.cosfly_adapter import discover_traces
        empty = tmp_path / "empty"
        empty.mkdir()
        traces = discover_traces(empty)
        assert traces == []


# ---------------------------------------------------------------------------
# Tests — build_cosfly_manifest
# ---------------------------------------------------------------------------

class TestBuildCoslfyManifest:
    def test_manifest_valid(self, tmp_path: Path) -> None:
        from vllatent.ingest.cosfly_adapter import build_cosfly_manifest
        from vllatent.manifest import validate_manifest

        entries = [{"clip_id": "cosfly_Town01_0000_ORI", "n_frames": 20, "latent_path": "cosfly_Town01_0000_ORI.npz"}]
        m = build_cosfly_manifest(entries=entries)
        errors = validate_manifest(m)
        assert errors == [], f"Manifest validation errors: {errors}"

    def test_motion_method_is_cosfly_gt(self, tmp_path: Path) -> None:
        from vllatent.ingest.cosfly_adapter import build_cosfly_manifest
        m = build_cosfly_manifest()
        assert m["motion_source"]["method"] == "cosfly_gt"

    def test_license_is_apache(self, tmp_path: Path) -> None:
        from vllatent.ingest.cosfly_adapter import build_cosfly_manifest
        m = build_cosfly_manifest()
        assert m["dataset"]["license"] == "Apache-2.0"


# ---------------------------------------------------------------------------
# Import purity — no torch/transformers/ultralytics at module level
# ---------------------------------------------------------------------------

class TestImportPurity:
    def test_no_heavy_imports(self) -> None:
        src = Path(__file__).resolve().parent.parent / "vllatent" / "ingest" / "cosfly_adapter.py"
        tree = ast.parse(src.read_text())
        heavy = {"torch", "transformers", "timm", "ultralytics", "cv2", "PIL"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in heavy, f"top-level import of {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in heavy, f"top-level from-import of {node.module}"

    def test_module_importable_without_torch(self) -> None:
        saved = {}
        for mod in ("torch", "transformers", "timm", "ultralytics"):
            if mod in sys.modules:
                saved[mod] = sys.modules.pop(mod)
        try:
            import vllatent.ingest.cosfly_adapter  # noqa: F401
        finally:
            sys.modules.update(saved)
