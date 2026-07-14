"""Tests for vllatent.ingest.megasam — MegaSaM wrapper."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vllatent.ingest.megasam import (
    CONFIDENCE_SOURCES,
    MegaSamResult,
    aggregate_motion_prob,
    lie7_to_c2w_matrices,
    parse_megasam_output,
    validate_megasam_result,
)


def _valid_poses(n: int = 5) -> np.ndarray:
    poses = np.zeros((n, 4, 4), dtype=np.float64)
    for i in range(n):
        poses[i] = np.eye(4)
        poses[i, 0, 3] = float(i)
    return poses


class TestMegaSamResult:
    def test_valid(self) -> None:
        r = MegaSamResult(
            poses=_valid_poses(3),
            confidences=np.ones(3, dtype=np.float64),
            intrinsics=np.eye(3, dtype=np.float64),
        )
        assert r.poses.shape == (3, 4, 4)
        assert r.confidence_source == "real"

    def test_valid_with_default_confidence(self) -> None:
        r = MegaSamResult(
            poses=_valid_poses(3),
            confidences=np.ones(3, dtype=np.float64),
            intrinsics=np.eye(3, dtype=np.float64),
            confidence_source="default",
        )
        assert r.confidence_source == "default"

    def test_rejects_bad_confidence_source(self) -> None:
        with pytest.raises(ValueError, match="confidence_source"):
            MegaSamResult(
                poses=_valid_poses(3),
                confidences=np.ones(3, dtype=np.float64),
                intrinsics=np.eye(3, dtype=np.float64),
                confidence_source="bogus",
            )

    def test_confidence_sources_enum(self) -> None:
        assert "real" in CONFIDENCE_SOURCES
        assert "default" in CONFIDENCE_SOURCES

    def test_rejects_bad_poses(self) -> None:
        with pytest.raises(ValueError, match="expected"):
            MegaSamResult(
                poses=np.zeros((3, 3), dtype=np.float64),
                confidences=np.ones(3),
                intrinsics=np.eye(3),
            )

    def test_rejects_mismatched_confidence(self) -> None:
        with pytest.raises(ValueError, match="confidences"):
            MegaSamResult(
                poses=_valid_poses(3),
                confidences=np.ones(5, dtype=np.float64),
                intrinsics=np.eye(3),
            )

    def test_rejects_bad_intrinsics(self) -> None:
        with pytest.raises(ValueError, match="intrinsics"):
            MegaSamResult(
                poses=_valid_poses(3),
                confidences=np.ones(3),
                intrinsics=np.zeros((2, 2)),
            )


class TestParseMegasamOutput:
    def test_npy_format(self, tmp_path: Path) -> None:
        poses = _valid_poses(4)
        np.save(str(tmp_path / "poses.npy"), poses)
        result = parse_megasam_output(tmp_path)
        assert result.poses.shape == (4, 4, 4)
        assert result.confidences.shape == (4,)

    def test_npz_format(self, tmp_path: Path) -> None:
        poses = _valid_poses(3)
        conf = np.ones(3, dtype=np.float64)
        np.savez(str(tmp_path / "cameras.npz"), poses=poses, confidences=conf)
        result = parse_megasam_output(tmp_path)
        assert result.poses.shape == (3, 4, 4)

    def test_json_format(self, tmp_path: Path) -> None:
        poses = _valid_poses(2)
        data = {"poses": poses.tolist(), "confidences": [0.9, 0.8]}
        (tmp_path / "results.json").write_text(json.dumps(data))
        result = parse_megasam_output(tmp_path)
        assert result.poses.shape == (2, 4, 4)

    def test_no_output_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No recognized"):
            parse_megasam_output(tmp_path)

    def test_npy_without_confidences_defaults_to_ones(self, tmp_path: Path) -> None:
        np.save(str(tmp_path / "poses.npy"), _valid_poses(3))
        result = parse_megasam_output(tmp_path)
        assert np.allclose(result.confidences, 1.0)
        assert result.confidence_source == "default"

    def test_npy_with_confidences_is_real(self, tmp_path: Path) -> None:
        np.save(str(tmp_path / "poses.npy"), _valid_poses(3))
        np.save(str(tmp_path / "confidences.npy"), np.array([0.9, 0.8, 0.7]))
        result = parse_megasam_output(tmp_path)
        assert result.confidence_source == "real"
        assert not np.allclose(result.confidences, 1.0)

    def test_npz_without_confidences_is_default(self, tmp_path: Path) -> None:
        np.savez(str(tmp_path / "cameras.npz"), poses=_valid_poses(3))
        result = parse_megasam_output(tmp_path)
        assert result.confidence_source == "default"

    def test_json_without_confidences_is_default(self, tmp_path: Path) -> None:
        import json as json_mod
        data = {"poses": _valid_poses(2).tolist()}
        (tmp_path / "results.json").write_text(json_mod.dumps(data))
        result = parse_megasam_output(tmp_path)
        assert result.confidence_source == "default"

    def test_json_with_confidences_is_real(self, tmp_path: Path) -> None:
        import json as json_mod
        data = {"poses": _valid_poses(2).tolist(), "confidences": [0.95, 0.85]}
        (tmp_path / "results.json").write_text(json_mod.dumps(data))
        result = parse_megasam_output(tmp_path)
        assert result.confidence_source == "real"


# ---------------------------------------------------------------------------
# Helpers for real MegaSaM format (Lie group + motion_prob)
# ---------------------------------------------------------------------------

def _identity_lie7(n: int = 5) -> np.ndarray:
    """Identity poses in MegaSaM's (T, 7) Lie group format [x,y,z,qx,qy,qz,qw]."""
    poses = np.zeros((n, 7), dtype=np.float64)
    poses[:, 6] = 1.0  # qw = 1 for identity quaternion
    for i in range(n):
        poses[i, 0] = float(i) * 0.5  # translate along x
    return poses


def _motion_prob_fixture(n: int = 5, h: int = 30, w: int = 40) -> np.ndarray:
    """Synthetic motion probability (T, H/8, W/8) — high confidence everywhere."""
    return np.full((n, h, w), 0.8, dtype=np.float32)


def _intrinsics_vec4_fixture(n: int = 5) -> np.ndarray:
    """Intrinsics in MegaSaM's (T, 4) format [fx, fy, cx, cy] already ×8.0."""
    intr = np.zeros((n, 4), dtype=np.float64)
    intr[:, 0] = 500.0 * 8.0  # fx
    intr[:, 1] = 500.0 * 8.0  # fy
    intr[:, 2] = 320.0 * 8.0  # cx
    intr[:, 3] = 240.0 * 8.0  # cy
    return intr


def _write_reconstructions(tmp_path: Path, scene: str, n: int = 5) -> Path:
    """Write a full MegaSaM reconstructions/{scene}/ directory."""
    recon_dir = tmp_path / "reconstructions" / scene
    recon_dir.mkdir(parents=True)
    np.save(str(recon_dir / "poses.npy"), _identity_lie7(n))
    np.save(str(recon_dir / "motion_prob.npy"), _motion_prob_fixture(n))
    np.save(str(recon_dir / "intrinsics.npy"), _intrinsics_vec4_fixture(n))
    return recon_dir


# ---------------------------------------------------------------------------
# Tests — lie7_to_c2w_matrices
# ---------------------------------------------------------------------------

class TestLie7ToC2wMatrices:
    def test_identity_gives_identity_matrices(self) -> None:
        lie7 = np.zeros((3, 7), dtype=np.float64)
        lie7[:, 6] = 1.0  # qw = 1
        c2w = lie7_to_c2w_matrices(lie7)
        assert c2w.shape == (3, 4, 4)
        for i in range(3):
            np.testing.assert_allclose(c2w[i], np.eye(4), atol=1e-10)

    def test_pure_translation(self) -> None:
        lie7 = np.zeros((2, 7), dtype=np.float64)
        lie7[:, 6] = 1.0
        lie7[1, 0] = 3.0  # x=3 in w2c
        c2w = lie7_to_c2w_matrices(lie7)
        # w2c has t=[3,0,0], so c2w has t=[-3,0,0]
        assert c2w.shape == (2, 4, 4)
        np.testing.assert_allclose(c2w[1, 0, 3], -3.0, atol=1e-10)

    def test_orthogonal_rotations(self) -> None:
        lie7 = np.zeros((5, 7), dtype=np.float64)
        lie7[:, 6] = 1.0
        lie7[2, :4] = [1.0, 0, 0, 0]  # 90° yaw (x-axis rotation as quaternion)
        lie7[2, 4:] = [0, 0, 1]
        c2w = lie7_to_c2w_matrices(lie7)
        for i in range(5):
            R = c2w[i, :3, :3]
            det = np.linalg.det(R)
            np.testing.assert_allclose(det, 1.0, atol=1e-6)
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)

    def test_output_dtype_float64(self) -> None:
        c2w = lie7_to_c2w_matrices(_identity_lie7(3))
        assert c2w.dtype == np.float64

    def test_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="expected"):
            lie7_to_c2w_matrices(np.zeros((3, 4)))


# ---------------------------------------------------------------------------
# Tests — aggregate_motion_prob
# ---------------------------------------------------------------------------

class TestAggregateMotionProb:
    def test_uniform_high_returns_high(self) -> None:
        mp = np.full((10, 30, 40), 0.9, dtype=np.float32)
        conf = aggregate_motion_prob(mp)
        assert conf.shape == (10,)
        np.testing.assert_allclose(conf, 0.9, atol=1e-6)

    def test_uniform_low_returns_low(self) -> None:
        mp = np.full((5, 20, 20), 0.1, dtype=np.float32)
        conf = aggregate_motion_prob(mp)
        np.testing.assert_allclose(conf, 0.1, atol=1e-6)

    def test_mixed_spatial_aggregates(self) -> None:
        mp = np.zeros((3, 10, 10), dtype=np.float32)
        mp[0] = 1.0
        mp[1] = 0.5
        mp[2] = 0.0
        conf = aggregate_motion_prob(mp)
        np.testing.assert_allclose(conf[0], 1.0, atol=1e-6)
        np.testing.assert_allclose(conf[1], 0.5, atol=1e-6)
        np.testing.assert_allclose(conf[2], 0.0, atol=1e-6)

    def test_output_in_01(self) -> None:
        rng = np.random.RandomState(42)
        mp = rng.rand(8, 15, 20).astype(np.float32)
        conf = aggregate_motion_prob(mp)
        assert np.all(conf >= 0.0)
        assert np.all(conf <= 1.0)


# ---------------------------------------------------------------------------
# Tests — parse real MegaSaM format (reconstructions/{scene}/)
# ---------------------------------------------------------------------------

class TestParseMegasamRealFormat:
    def test_reconstructions_dir_parsed(self, tmp_path: Path) -> None:
        recon_dir = _write_reconstructions(tmp_path, "ski01", n=10)
        result = parse_megasam_output(recon_dir)
        assert result.poses.shape == (10, 4, 4)
        assert result.confidences.shape == (10,)
        assert result.intrinsics.shape == (3, 3)
        assert result.confidence_source == "real"

    def test_c2w_poses_are_inverted(self, tmp_path: Path) -> None:
        """The (T,7) w2c poses should be inverted to produce c2w matrices."""
        recon_dir = _write_reconstructions(tmp_path, "ski01", n=3)
        result = parse_megasam_output(recon_dir)
        for i in range(3):
            R = result.poses[i, :3, :3]
            det = np.linalg.det(R)
            np.testing.assert_allclose(det, 1.0, atol=1e-6)

    def test_intrinsics_4vec_to_3x3(self, tmp_path: Path) -> None:
        recon_dir = _write_reconstructions(tmp_path, "ski01", n=3)
        result = parse_megasam_output(recon_dir)
        K = result.intrinsics
        assert K[0, 0] == pytest.approx(500.0, abs=0.1)  # fx (÷8)
        assert K[1, 1] == pytest.approx(500.0, abs=0.1)  # fy (÷8)
        assert K[0, 2] == pytest.approx(320.0, abs=0.1)  # cx (÷8)
        assert K[1, 2] == pytest.approx(240.0, abs=0.1)  # cy (÷8)

    def test_motion_prob_becomes_confidence(self, tmp_path: Path) -> None:
        recon_dir = _write_reconstructions(tmp_path, "ski01", n=5)
        result = parse_megasam_output(recon_dir)
        assert result.confidence_source == "real"
        np.testing.assert_allclose(result.confidences, 0.8, atol=0.05)

    def test_no_motion_prob_falls_back(self, tmp_path: Path) -> None:
        recon_dir = _write_reconstructions(tmp_path, "ski01", n=3)
        (recon_dir / "motion_prob.npy").unlink()
        result = parse_megasam_output(recon_dir)
        assert result.confidence_source == "default"
        np.testing.assert_allclose(result.confidences, 1.0)

    def test_droid_npz_format(self, tmp_path: Path) -> None:
        """Test the outputs/{scene}_droid.npz alternative path."""
        n = 5
        c2w = np.stack([np.eye(4)] * n).astype(np.float64)
        for i in range(n):
            c2w[i, 0, 3] = float(i)
        K = np.eye(3, dtype=np.float64)
        K[0, 0] = K[1, 1] = 500.0
        K[0, 2], K[1, 2] = 320.0, 240.0
        np.savez(
            str(tmp_path / "droid_output.npz"),
            cam_c2w=c2w,
            intrinsic=K,
            images=np.zeros((n, 240, 320, 3), dtype=np.uint8),
            depths=np.zeros((n, 30, 40), dtype=np.float32),
        )
        result = parse_megasam_output(tmp_path / "droid_output.npz")
        assert result.poses.shape == (n, 4, 4)
        assert result.intrinsics.shape == (3, 3)
        assert result.confidence_source == "default"  # droid.npz has no per-frame confidence

    def test_priority_reconstructions_over_legacy(self, tmp_path: Path) -> None:
        """If both poses.npy (T,7) and old (N,4,4) exist, the (T,7) path wins."""
        _write_reconstructions(tmp_path, "test", n=4)
        recon_dir = tmp_path / "reconstructions" / "test"
        result = parse_megasam_output(recon_dir)
        assert result.poses.shape == (4, 4, 4)


class TestValidateMegasamResult:
    def test_valid_result(self) -> None:
        r = MegaSamResult(poses=_valid_poses(5), confidences=np.ones(5), intrinsics=np.eye(3))
        assert validate_megasam_result(r) == []

    def test_too_few_poses(self) -> None:
        r = MegaSamResult(poses=_valid_poses(1), confidences=np.ones(1), intrinsics=np.eye(3))
        errors = validate_megasam_result(r)
        assert any("Too few" in e for e in errors)

    def test_degenerate_poses(self) -> None:
        poses = np.stack([np.eye(4)] * 10).astype(np.float64)
        r = MegaSamResult(poses=poses, confidences=np.ones(10), intrinsics=np.eye(3))
        errors = validate_megasam_result(r)
        assert any("Degenerate" in e for e in errors)

    def test_low_confidence(self) -> None:
        r = MegaSamResult(
            poses=_valid_poses(5),
            confidences=np.full(5, 0.01, dtype=np.float64),
            intrinsics=np.eye(3),
        )
        errors = validate_megasam_result(r)
        assert any("Low confidence" in e for e in errors)

    def test_bad_rotation_det(self) -> None:
        poses = _valid_poses(3)
        poses[1, :3, :3] *= 2.0
        r = MegaSamResult(poses=poses, confidences=np.ones(3), intrinsics=np.eye(3))
        errors = validate_megasam_result(r)
        assert any("det" in e for e in errors)


def test_pipeline_does_not_copy_redundant_droid_archive() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_megasam_pipeline.sh"
    source = script.read_text()
    assert "DROID_NPZ" not in source
