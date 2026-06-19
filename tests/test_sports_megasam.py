"""Tests for vllatent.sports.megasam — MegaSaM wrapper (mocked for CI)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vllatent.sports.megasam import (
    MegaSamResult,
    parse_megasam_output,
    validate_megasam_result,
)


def _make_poses(n: int = 10) -> np.ndarray:
    """Generate N valid SE(3) poses along a straight line."""
    poses = np.zeros((n, 4, 4), dtype=np.float64)
    for i in range(n):
        poses[i] = np.eye(4)
        poses[i, 2, 3] = float(i) * 2.0  # translate along Z (camera forward)
    return poses


class TestMegaSamResult:
    def test_valid_construction(self) -> None:
        poses = _make_poses(5)
        conf = np.ones(5, dtype=np.float64)
        intr = np.eye(3, dtype=np.float64)
        r = MegaSamResult(poses=poses, confidences=conf, intrinsics=intr)
        assert r.poses.shape == (5, 4, 4)

    def test_bad_poses_shape(self) -> None:
        with pytest.raises(ValueError, match="poses"):
            MegaSamResult(
                poses=np.zeros((5, 3, 3)),
                confidences=np.ones(5),
                intrinsics=np.eye(3),
            )

    def test_confidence_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="confidences"):
            MegaSamResult(
                poses=_make_poses(5),
                confidences=np.ones(3),
                intrinsics=np.eye(3),
            )

    def test_bad_intrinsics_shape(self) -> None:
        with pytest.raises(ValueError, match="intrinsics"):
            MegaSamResult(
                poses=_make_poses(5),
                confidences=np.ones(5),
                intrinsics=np.eye(4),
            )


class TestParseMegaSamOutput:
    def test_npy_format(self, tmp_path: Path) -> None:
        poses = _make_poses(10)
        conf = np.ones(10, dtype=np.float64) * 0.9
        intr = np.eye(3, dtype=np.float64) * 500

        np.save(str(tmp_path / "poses.npy"), poses)
        np.save(str(tmp_path / "confidences.npy"), conf)
        np.save(str(tmp_path / "intrinsics.npy"), intr)

        result = parse_megasam_output(tmp_path)
        assert result.poses.shape == (10, 4, 4)
        assert result.confidences.shape == (10,)
        assert result.intrinsics.shape == (3, 3)

    def test_npz_format(self, tmp_path: Path) -> None:
        poses = _make_poses(8)
        conf = np.ones(8, dtype=np.float64)
        intr = np.eye(3, dtype=np.float64)

        np.savez(str(tmp_path / "cameras.npz"), poses=poses, confidences=conf, intrinsics=intr)

        result = parse_megasam_output(tmp_path)
        assert result.poses.shape == (8, 4, 4)

    def test_npy_without_confidence(self, tmp_path: Path) -> None:
        poses = _make_poses(5)
        np.save(str(tmp_path / "poses.npy"), poses)

        result = parse_megasam_output(tmp_path)
        np.testing.assert_array_equal(result.confidences, np.ones(5))

    def test_missing_output_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No recognized"):
            parse_megasam_output(tmp_path)


class TestValidateMegaSamResult:
    def test_valid_result(self) -> None:
        r = MegaSamResult(
            poses=_make_poses(10),
            confidences=np.ones(10) * 0.9,
            intrinsics=np.eye(3),
        )
        errors = validate_megasam_result(r)
        assert errors == []

    def test_too_few_poses(self) -> None:
        r = MegaSamResult(
            poses=_make_poses(1),
            confidences=np.ones(1),
            intrinsics=np.eye(3),
        )
        errors = validate_megasam_result(r)
        assert any("Too few" in e for e in errors)

    def test_degenerate_identity_poses(self) -> None:
        poses = np.tile(np.eye(4), (10, 1, 1))
        r = MegaSamResult(poses=poses, confidences=np.ones(10), intrinsics=np.eye(3))
        errors = validate_megasam_result(r)
        assert any("identity" in e.lower() or "Degenerate" in e for e in errors)

    def test_low_confidence_flagged(self) -> None:
        r = MegaSamResult(
            poses=_make_poses(10),
            confidences=np.ones(10) * 0.01,
            intrinsics=np.eye(3),
        )
        errors = validate_megasam_result(r)
        assert any("confidence" in e.lower() for e in errors)

    def test_bad_rotation_det(self) -> None:
        poses = _make_poses(5)
        poses[2, :3, :3] *= 2.0  # break det(R)=1
        r = MegaSamResult(poses=poses, confidences=np.ones(5), intrinsics=np.eye(3))
        errors = validate_megasam_result(r)
        assert any("det(R)" in e for e in errors)
