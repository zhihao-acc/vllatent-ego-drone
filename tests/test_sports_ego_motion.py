"""Tests for vllatent.sports.ego_motion — SE(3) to body-frame delta conversion."""
from __future__ import annotations

import numpy as np
import pytest

from vllatent.schemas import DELTA_DTYPE
from vllatent.sports.ego_motion import (
    R_BODY_FROM_CAM,
    camera_to_drone_body,
    normalize_scale,
    rotation_to_yaw,
    se3_sequence_to_deltas,
    se3_to_body_delta,
    sim3_align,
    validate_scale_consistency,
)


def _se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build a 4x4 SE(3) matrix from R (3x3) and t (3,)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _yaw_rotation_cam(yaw_rad: float) -> np.ndarray:
    """Create a camera-convention rotation corresponding to a yaw in body frame.

    Camera Z = body X (forward), so yaw rotates in the camera XZ plane.
    """
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ], dtype=np.float64)


class TestRotationToYaw:
    def test_identity(self) -> None:
        assert rotation_to_yaw(np.eye(3)) == pytest.approx(0.0)

    def test_90_degrees(self) -> None:
        c, s = 0.0, 1.0
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
        assert rotation_to_yaw(R) == pytest.approx(np.pi / 2, abs=1e-10)


class TestCameraToDroneBody:
    def test_rotation_is_orthogonal(self) -> None:
        assert np.allclose(R_BODY_FROM_CAM @ R_BODY_FROM_CAM.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R_BODY_FROM_CAM) == pytest.approx(1.0)

    def test_camera_forward_is_body_forward(self) -> None:
        cam_forward = np.array([0, 0, 1], dtype=np.float64)
        _, t_body = camera_to_drone_body(np.eye(3), cam_forward)
        assert t_body[0] == pytest.approx(1.0)
        assert t_body[1] == pytest.approx(0.0)
        assert t_body[2] == pytest.approx(0.0)

    def test_camera_right_is_body_right(self) -> None:
        cam_right = np.array([1, 0, 0], dtype=np.float64)
        _, t_body = camera_to_drone_body(np.eye(3), cam_right)
        assert t_body[1] == pytest.approx(1.0)

    def test_camera_down_is_body_down(self) -> None:
        cam_down = np.array([0, 1, 0], dtype=np.float64)
        _, t_body = camera_to_drone_body(np.eye(3), cam_down)
        assert t_body[2] == pytest.approx(1.0)


class TestSe3ToBodyDelta:
    def test_identity_produces_zero_delta(self) -> None:
        T = np.eye(4, dtype=np.float64)
        delta = se3_to_body_delta(T, T)
        np.testing.assert_allclose(delta, [0, 0, 0, 0], atol=1e-10)

    def test_pure_forward_translation(self) -> None:
        T1 = np.eye(4, dtype=np.float64)
        T2 = _se3(np.eye(3), np.array([0, 0, 5.0]))
        delta = se3_to_body_delta(T1, T2)
        assert delta[0] == pytest.approx(5.0, abs=1e-6)
        assert abs(delta[1]) < 1e-6
        assert abs(delta[2]) < 1e-6
        assert abs(delta[3]) < 1e-4

    def test_pure_right_translation(self) -> None:
        T1 = np.eye(4, dtype=np.float64)
        T2 = _se3(np.eye(3), np.array([3.0, 0, 0]))
        delta = se3_to_body_delta(T1, T2)
        assert delta[1] == pytest.approx(3.0, abs=1e-6)

    def test_pure_yaw_rotation(self) -> None:
        T1 = np.eye(4, dtype=np.float64)
        yaw_rad = np.radians(15.0)
        T2 = _se3(_yaw_rotation_cam(yaw_rad), np.zeros(3))
        delta = se3_to_body_delta(T1, T2)
        assert abs(delta[0]) < 1e-6
        assert abs(delta[1]) < 1e-6
        assert abs(delta[2]) < 1e-6
        assert delta[3] == pytest.approx(15.0, abs=0.5)

    def test_output_dtype(self) -> None:
        T = np.eye(4, dtype=np.float64)
        delta = se3_to_body_delta(T, T)
        assert delta.dtype == DELTA_DTYPE
        assert delta.shape == (4,)


class TestSe3SequenceToDeltas:
    def test_straight_line(self) -> None:
        poses = np.zeros((5, 4, 4), dtype=np.float64)
        for i in range(5):
            poses[i] = _se3(np.eye(3), np.array([0, 0, float(i) * 2.0]))
        deltas = se3_sequence_to_deltas(poses)
        assert deltas.shape == (4, 4)
        for i in range(4):
            assert deltas[i, 0] == pytest.approx(2.0, abs=1e-5)

    def test_bad_shape(self) -> None:
        with pytest.raises(ValueError, match="poses"):
            se3_sequence_to_deltas(np.eye(4))

    def test_too_few_poses(self) -> None:
        with pytest.raises(ValueError, match="need >= 2"):
            se3_sequence_to_deltas(np.eye(4).reshape(1, 4, 4))


class TestNormalizeScale:
    def test_median_speed(self) -> None:
        deltas = np.array([
            [1, 0, 0, 0],
            [2, 0, 0, 0],
            [3, 0, 0, 0],
            [4, 0, 0, 0],
            [5, 0, 0, 0],
        ], dtype=DELTA_DTYPE)
        normed = normalize_scale(deltas, mode="median_speed")
        magnitudes = np.linalg.norm(normed[:, :3], axis=1)
        assert np.median(magnitudes) == pytest.approx(1.0, abs=1e-6)

    def test_unit_max(self) -> None:
        deltas = np.array([
            [1, 0, 0, 0],
            [3, 0, 0, 0],
            [5, 0, 0, 10],
        ], dtype=DELTA_DTYPE)
        normed = normalize_scale(deltas, mode="unit_max")
        magnitudes = np.linalg.norm(normed[:, :3], axis=1)
        assert np.max(magnitudes) == pytest.approx(1.0, abs=1e-6)

    def test_yaw_preserved(self) -> None:
        deltas = np.array([[2, 0, 0, 15.0]], dtype=DELTA_DTYPE)
        normed = normalize_scale(deltas, mode="median_speed")
        assert normed[0, 3] == pytest.approx(15.0)

    def test_zero_magnitude(self) -> None:
        deltas = np.array([[0, 0, 0, 5.0]], dtype=DELTA_DTYPE)
        normed = normalize_scale(deltas, mode="median_speed")
        np.testing.assert_array_equal(normed, deltas)

    def test_input_not_mutated(self) -> None:
        deltas = np.array([[3, 0, 0, 0]], dtype=DELTA_DTYPE)
        original = deltas.copy()
        normalize_scale(deltas, mode="median_speed")
        np.testing.assert_array_equal(deltas, original)

    def test_bad_mode(self) -> None:
        deltas = np.array([[1, 0, 0, 0]], dtype=DELTA_DTYPE)
        with pytest.raises(ValueError, match="unknown mode"):
            normalize_scale(deltas, mode="invalid")

    def test_bad_shape(self) -> None:
        with pytest.raises(ValueError, match="deltas"):
            normalize_scale(np.zeros((3, 3), dtype=DELTA_DTYPE))


class TestValidateScaleConsistency:
    def test_basic_stats(self) -> None:
        deltas = np.array([
            [1, 0, 0, 0],
            [2, 0, 0, 0],
            [3, 0, 0, 0],
        ], dtype=DELTA_DTYPE)
        stats = validate_scale_consistency(deltas)
        assert stats["mean"] == pytest.approx(2.0, abs=1e-6)
        assert stats["median"] == pytest.approx(2.0, abs=1e-6)
        assert stats["min"] == pytest.approx(1.0, abs=1e-6)
        assert stats["max"] == pytest.approx(3.0, abs=1e-6)

    def test_outlier_detection(self) -> None:
        deltas = np.array(
            [[1, 0, 0, 0]] * 10 + [[100, 0, 0, 0]],
            dtype=DELTA_DTYPE,
        )
        stats = validate_scale_consistency(deltas)
        assert stats["n_outliers"] >= 1


class TestSim3Align:
    def test_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError, match="GPS or IMU"):
            sim3_align(np.zeros((10, 4, 4)), np.zeros((10, 3)))
