"""Analytic float64 SE(3) and signed image-geometry tests for B3-CS1."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vllatent.sim.contracts import (
    R_CAM_FROM_RIG,
    BranchId,
    default_camera_contract,
    expected_image_effect,
    program_by_id,
)
from vllatent.sim.frames import (
    NamedSE3,
    body_frd_twist_increment,
    compose_se3,
    identity_se3,
    integrate_requested_relative,
    invert_se3,
    kinematic_camera_trajectory,
    project_blender_camera,
    requested_achieved_se3_error,
    rotation_geodesic_angle,
)


def _f64(values: object) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def test_named_float64_transform_inverse_compose_and_validation() -> None:
    matrix = body_frd_twist_increment(_f64([1.0, 0.5, -0.2, 0.1]), 0.2)
    rig1_from_rig0 = NamedSE3(matrix, target_frame="rig0", source_frame="rig1")
    rig0_from_rig1 = invert_se3(rig1_from_rig0)
    identity = compose_se3(rig1_from_rig0, rig0_from_rig1)
    np.testing.assert_allclose(identity.matrix, np.eye(4), atol=1e-12)
    assert identity.target_frame == "rig0"
    assert identity.source_frame == "rig0"
    with pytest.raises(ValueError, match="frame mismatch"):
        compose_se3(rig1_from_rig0, rig1_from_rig0)
    with pytest.raises(ValueError, match="dtype float64"):
        NamedSE3(np.eye(4, dtype=np.float32), "a", "b")
    bad = np.eye(4, dtype=np.float64)
    bad[0, 0] = -1.0
    with pytest.raises(ValueError, match="determinant"):
        NamedSE3(bad, "a", "b")
    named_identity = identity_se3(target_frame="world", source_frame="rig")
    np.testing.assert_array_equal(named_identity.matrix, np.eye(4))


def test_zero_and_pure_yaw_integrate_analytically_about_semantic_down() -> None:
    zero = integrate_requested_relative(program_by_id(BranchId.ZERO))
    np.testing.assert_array_equal(
        zero,
        np.repeat(np.eye(4, dtype=np.float64)[None, :, :], 8, axis=0),
    )

    yaw = integrate_requested_relative(program_by_id(BranchId.YAW_PLUS))
    theta = 8 * 0.2 * math.pi / 15.0
    expected_rotation = _f64(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(yaw[-1, :3, :3], expected_rotation, atol=1e-15)
    np.testing.assert_array_equal(yaw[-1, :3, 3], np.zeros(3))
    np.testing.assert_array_equal(R_CAM_FROM_RIG @ _f64([0.0, 0.0, 1.0]), [0, -1, 0])


def test_constant_body_twist_uses_se3_exponential_not_euler_translation() -> None:
    increment = body_frd_twist_increment(_f64([1.0, 0.0, 0.0, 1.0]), 0.2)
    assert increment[0, 3] == pytest.approx(math.sin(0.2), abs=1e-15)
    assert increment[1, 3] == pytest.approx(1.0 - math.cos(0.2), abs=1e-15)
    assert increment[0, 3] != pytest.approx(0.2)


def test_requested_and_achieved_poses_are_distinct_immutable_records() -> None:
    trajectory = kinematic_camera_trajectory(program_by_id(BranchId.FORWARD_PLUS))
    assert not np.shares_memory(
        trajectory.requested_T_rig0_from_rig_t,
        trajectory.achieved_T_rig0_from_rig_t,
    )
    assert trajectory.requested_T_rig0_from_rig_t is not trajectory.achieved_T_rig0_from_rig_t
    np.testing.assert_array_equal(
        trajectory.requested_T_rig0_from_rig_t,
        trajectory.achieved_T_rig0_from_rig_t,
    )
    assert not trajectory.requested_T_rig0_from_rig_t.flags.writeable
    assert not trajectory.achieved_T_rig0_from_rig_t.flags.writeable
    assert not trajectory.record_valid.flags.writeable
    with pytest.raises(ValueError, match="WRITEABLE"):
        trajectory.requested_T_rig0_from_rig_t.setflags(write=True)
    with pytest.raises(ValueError, match="WRITEABLE"):
        trajectory.achieved_T_rig0_from_rig_t.setflags(write=True)
    with pytest.raises(ValueError, match="WRITEABLE"):
        trajectory.record_valid.setflags(write=True)
    translation_error, rotation_error = requested_achieved_se3_error(trajectory)
    np.testing.assert_array_equal(translation_error, np.zeros(8))
    np.testing.assert_array_equal(rotation_error, np.zeros(8))


def test_rotation_geodesic_retains_small_angle_precision() -> None:
    angle = 1.0e-8
    rotated = _f64(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert rotation_geodesic_angle(np.eye(3, dtype=np.float64), rotated) == pytest.approx(angle, abs=1.0e-16)


def test_world_from_camera_is_derived_in_the_named_direction() -> None:
    initial = np.eye(4, dtype=np.float64)
    initial[:3, 3] = _f64([10.0, 20.0, 30.0])
    trajectory = kinematic_camera_trajectory(program_by_id(BranchId.ZERO), initial_T_world_from_rig=initial)
    world_from_camera = trajectory.T_world_from_cam()
    np.testing.assert_array_equal(world_from_camera[:, :3, 3], np.tile(initial[:3, 3], (8, 1)))
    np.testing.assert_array_equal(world_from_camera[0, :3, :3], default_camera_contract().R_rig_from_cam)


def _target_projection(branch_id: BranchId) -> tuple[np.ndarray, float, float]:
    """Project a fixed two-point target after one branch's eighth action."""
    relative = integrate_requested_relative(program_by_id(branch_id))[-1]
    rig_t_from_rig0 = np.linalg.inv(relative)
    points_rig0 = _f64([[10.0, 0.0, -1.0], [10.0, 0.0, 1.0]])
    points_rig_t = points_rig0 @ rig_t_from_rig0[:3, :3].T + rig_t_from_rig0[:3, 3]
    points_camera = points_rig_t @ R_CAM_FROM_RIG.T
    pixels = project_blender_camera(points_camera, default_camera_contract().K)
    center = pixels.mean(axis=0)
    height = float(abs(pixels[1, 1] - pixels[0, 1]))
    depth = float(np.mean(-points_camera[:, 2]))
    return center, math.log(height / 224.0), depth


@pytest.mark.parametrize(
    "plus,minus",
    [
        (BranchId.YAW_PLUS, BranchId.YAW_MINUS),
        (BranchId.FORWARD_PLUS, BranchId.FORWARD_MINUS),
        (BranchId.LATERAL_PLUS, BranchId.LATERAL_MINUS),
        (BranchId.VERTICAL_PLUS, BranchId.VERTICAL_MINUS),
    ],
)
def test_analytic_pinhole_geometry_matches_preregistered_signed_effects(plus: BranchId, minus: BranchId) -> None:
    zero_center, zero_log_h, _ = _target_projection(BranchId.ZERO)
    plus_center, plus_log_h, plus_depth = _target_projection(plus)
    minus_center, minus_log_h, minus_depth = _target_projection(minus)
    effect = expected_image_effect(plus)
    if effect.field == "cx":
        plus_delta = plus_center[0] - zero_center[0]
        minus_delta = minus_center[0] - zero_center[0]
    elif effect.field == "cy":
        plus_delta = plus_center[1] - zero_center[1]
        minus_delta = minus_center[1] - zero_center[1]
    else:
        plus_delta = plus_log_h - zero_log_h
        minus_delta = minus_log_h - zero_log_h
    assert plus_delta * effect.sign > 0.0
    assert minus_delta * effect.sign < 0.0
    assert plus_depth > 2.0
    assert minus_depth > 2.0
