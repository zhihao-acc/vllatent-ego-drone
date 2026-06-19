"""Tests for vllatent.ingest.megasam — MegaSaM wrapper."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vllatent.ingest.megasam import (
    CONFIDENCE_SOURCES,
    MegaSamResult,
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
