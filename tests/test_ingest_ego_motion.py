"""Tests for vllatent.ingest.ego_motion — SE(3) to body-frame delta conversion."""
from __future__ import annotations

import numpy as np
import pytest

from vllatent.ingest.ego_motion import (
    R_BODY_FROM_CAM,
    AlignmentResult,
    align_to_gps,
    camera_to_drone_body,
    normalize_scale,
    rotation_to_yaw,
    se3_sequence_to_deltas,
    se3_to_body_delta,
    validate_scale_consistency,
)
from vllatent.schemas import DELTA_DTYPE


def _identity_se3() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def _translation_se3(dx: float, dy: float, dz: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = [dx, dy, dz]
    return T


class TestRBodyFromCam:
    def test_is_rotation(self) -> None:
        det = np.linalg.det(R_BODY_FROM_CAM)
        assert abs(det - 1.0) < 1e-10

    def test_orthogonal(self) -> None:
        assert np.allclose(R_BODY_FROM_CAM @ R_BODY_FROM_CAM.T, np.eye(3), atol=1e-10)

    def test_camera_z_to_body_x(self) -> None:
        cam_forward = np.array([0, 0, 1.0])
        body = R_BODY_FROM_CAM @ cam_forward
        assert np.allclose(body, [1, 0, 0], atol=1e-10)


class TestRotationToYaw:
    def test_identity_zero_yaw(self) -> None:
        assert abs(rotation_to_yaw(np.eye(3))) < 1e-10

    def test_90_deg_yaw(self) -> None:
        c, s = 0.0, 1.0
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        yaw = rotation_to_yaw(R)
        assert abs(yaw - np.pi / 2) < 1e-10


class TestCameraToDroneBody:
    def test_identity_maps_to_identity(self) -> None:
        R_body, t_body = camera_to_drone_body(np.eye(3), np.zeros(3))
        assert np.allclose(R_body, np.eye(3), atol=1e-10)
        assert np.allclose(t_body, np.zeros(3), atol=1e-10)


class TestSe3ToBodyDelta:
    def test_identity_pair_gives_zero(self) -> None:
        delta = se3_to_body_delta(_identity_se3(), _identity_se3())
        assert delta.shape == (4,)
        assert delta.dtype == DELTA_DTYPE
        assert np.allclose(delta, 0.0, atol=1e-6)

    def test_pure_forward_translation(self) -> None:
        T0 = _identity_se3()
        T1 = _translation_se3(0, 0, 1.0)  # cam Z = forward = body X
        delta = se3_to_body_delta(T0, T1)
        assert delta[0] > 0  # body X (forward)
        assert abs(delta[3]) < 1e-6  # no yaw

    def test_output_dtype(self) -> None:
        delta = se3_to_body_delta(_identity_se3(), _translation_se3(1, 0, 0))
        assert delta.dtype == DELTA_DTYPE


class TestSe3SequenceToDeltas:
    def test_basic_sequence(self) -> None:
        poses = np.stack([_identity_se3(), _translation_se3(0, 0, 1), _translation_se3(0, 0, 2)])
        deltas = se3_sequence_to_deltas(poses)
        assert deltas.shape == (2, 4)
        assert deltas.dtype == DELTA_DTYPE

    def test_rejects_bad_shape(self) -> None:
        with pytest.raises(ValueError, match="expected"):
            se3_sequence_to_deltas(np.zeros((3, 3)))

    def test_rejects_single_pose(self) -> None:
        with pytest.raises(ValueError, match="need >= 2"):
            se3_sequence_to_deltas(np.eye(4).reshape(1, 4, 4))


class TestNormalizeScale:
    def test_median_speed(self) -> None:
        deltas = np.array([[1, 0, 0, 10], [2, 0, 0, 20], [3, 0, 0, 30]], dtype=np.float32)
        normed = normalize_scale(deltas, mode="median_speed")
        assert normed.shape == deltas.shape
        assert normed.dtype == deltas.dtype
        assert abs(np.median(np.linalg.norm(normed[:, :3], axis=1)) - 1.0) < 1e-6

    def test_unit_max(self) -> None:
        deltas = np.array([[1, 0, 0, 0], [3, 0, 0, 0]], dtype=np.float32)
        normed = normalize_scale(deltas, mode="unit_max")
        assert abs(np.max(np.linalg.norm(normed[:, :3], axis=1)) - 1.0) < 1e-6

    def test_does_not_mutate_input(self) -> None:
        deltas = np.array([[1, 0, 0, 0]], dtype=np.float32)
        original = deltas.copy()
        normalize_scale(deltas)
        assert np.array_equal(deltas, original)

    def test_near_zero_scale_passthrough(self) -> None:
        deltas = np.array([[0, 0, 0, 10]], dtype=np.float32)
        normed = normalize_scale(deltas)
        assert np.allclose(normed[:, :3], 0.0)

    def test_bad_mode(self) -> None:
        with pytest.raises(ValueError, match="unknown mode"):
            normalize_scale(np.zeros((2, 4), dtype=np.float32), mode="nope")

    def test_bad_shape(self) -> None:
        with pytest.raises(ValueError, match="expected"):
            normalize_scale(np.zeros((2, 3), dtype=np.float32))


class TestAlignToGps:
    def test_raises_not_implemented(self) -> None:
        poses = np.stack([np.eye(4)] * 3)
        gps = np.zeros((3, 3))
        with pytest.raises(NotImplementedError, match="GPS Sim"):
            align_to_gps(poses, gps)

    def test_alignment_result_fields(self) -> None:
        r = AlignmentResult(
            aligned_poses=np.stack([np.eye(4)] * 2),
            scale=1.5,
            rotation=np.eye(3),
            translation=np.zeros(3),
            rmse=0.1,
        )
        assert r.scale == 1.5
        assert r.rmse == 0.1


class TestValidateScaleConsistency:
    def test_returns_stats(self) -> None:
        deltas = np.random.default_rng(42).standard_normal((20, 4)).astype(np.float32)
        stats = validate_scale_consistency(deltas)
        assert "mean" in stats and "n_outliers" in stats


