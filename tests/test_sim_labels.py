"""Mechanical tests for the PURE B3-CS3 amodal/visible label contract."""

from __future__ import annotations

import ast
import inspect
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vllatent.sim.labels import (
    CROP_HEIGHT_PX,
    CROP_WIDTH_PX,
    LABEL_SCHEMA_VERSION,
    OccluderGeometry,
    TargetGeometry,
    amodal_regression_is_valid,
    compute_frame_labels,
    rasterize_triangles_unbounded,
)


def _f64(values: object) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _i64(values: object) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


K = _f64([[100.0, 0.0, 112.0], [0.0, 100.0, 112.0], [0.0, 0.0, 1.0]])
T_CAM_FROM_WORLD = np.eye(4, dtype=np.float64)


def _rectangle_vertices(
    u_min: float,
    v_min: float,
    u_max: float,
    v_max: float,
    *,
    depth_m: float,
) -> np.ndarray:
    pixels = ((u_min, v_min), (u_max, v_min), (u_max, v_max), (u_min, v_max))
    return _f64(
        [
            (
                (u - K[0, 2]) * depth_m / K[0, 0],
                (K[1, 2] - v) * depth_m / K[1, 1],
                -depth_m,
            )
            for u, v in pixels
        ]
    )


RECTANGLE_TRIANGLES = _i64([[0, 1, 2], [0, 2, 3]])


def _target_rectangle(
    u_min: float,
    v_min: float,
    u_max: float,
    v_max: float,
    *,
    depth_m: float = 10.0,
) -> TargetGeometry:
    return TargetGeometry(
        vertices_world_m=_rectangle_vertices(
            u_min,
            v_min,
            u_max,
            v_max,
            depth_m=depth_m,
        ),
        triangles=RECTANGLE_TRIANGLES,
    )


def _occluder_rectangle(
    object_id: str,
    u_min: float,
    v_min: float,
    u_max: float,
    v_max: float,
    *,
    depth_m: float = 5.0,
) -> OccluderGeometry:
    return OccluderGeometry(
        object_id=object_id,
        vertices_world_m=_rectangle_vertices(
            u_min,
            v_min,
            u_max,
            v_max,
            depth_m=depth_m,
        ),
        triangles=RECTANGLE_TRIANGLES,
    )


def test_unbounded_raster_has_explicit_integer_origin_and_never_clips_extrema() -> None:
    projected = _f64(
        [
            [[-10.0, 100.0], [10.0, 100.0], [10.0, 120.0]],
            [[-10.0, 100.0], [10.0, 120.0], [-10.0, 120.0]],
        ]
    )
    full = rasterize_triangles_unbounded(projected)

    assert full.area == 400
    np.testing.assert_array_equal(full.bbox_px, _i64([-10, 100, 10, 120]))
    assert int(full.u_start.min()) == -10
    assert int(full.u_stop.max()) == 10
    assert int(full.rows.min()) == 100
    assert int(full.rows.max()) == 119
    assert not full.rows.flags.writeable
    assert not full.u_start.flags.writeable
    assert not full.u_stop.flags.writeable


def test_crop_crossing_uses_exact_normative_areas_fractions_and_amodal_box() -> None:
    labels = compute_frame_labels(
        _target_rectangle(-10.0, 100.0, 10.0, 120.0),
        T_cam_from_world=T_CAM_FROM_WORLD,
        K=K,
    )

    assert labels.schema_version == LABEL_SCHEMA_VERSION
    assert (CROP_WIDTH_PX, CROP_HEIGHT_PX) == (224, 224)
    assert labels.A_full == 400
    assert labels.A_in == 200
    assert labels.A_vis == 200
    assert labels.frame_fraction == 200 / 400
    assert labels.visible_fraction == 200 / 400
    assert labels.occlusion_fraction == 0.0
    assert labels.in_frame is True
    assert labels.p_visible_target == 1
    assert labels.occlusion_flag is False
    assert labels.modal_front_object_id is None
    np.testing.assert_array_equal(labels.amodal_bbox_px, _i64([-10, 100, 10, 120]))
    assert labels.cx == 0.0
    assert labels.cy == 110.0 / 224.0
    assert labels.log_h == math.log(20.0 / 224.0)
    np.testing.assert_array_equal(labels.visible_bbox_px, _i64([0, 100, 10, 120]))
    assert labels.visible_cx == 5.0 / 224.0
    assert labels.visible_cy == 110.0 / 224.0
    assert labels.visible_log_h == math.log(20.0 / 224.0)
    np.testing.assert_array_equal(labels.m_in, labels.target_only_mask)
    np.testing.assert_array_equal(labels.m_vis, labels.m_in)


def test_depth_tested_complete_occlusion_exports_modal_front_object() -> None:
    target = _target_rectangle(100.0, 100.0, 120.0, 120.0, depth_m=10.0)
    rock = _occluder_rectangle("fixed-rock", 100.0, 100.0, 120.0, 120.0, depth_m=5.0)
    labels = compute_frame_labels(
        target,
        T_cam_from_world=T_CAM_FROM_WORLD,
        K=K,
        occluders=(rock,),
    )

    assert labels.A_full == 400
    assert labels.A_in == 400
    assert labels.A_vis == 0
    assert labels.frame_fraction == 1.0
    assert labels.visible_fraction == 0.0
    assert labels.occlusion_fraction == 1.0
    assert labels.in_frame is True
    assert labels.p_visible_target == 0
    assert labels.occlusion_flag is True
    assert labels.modal_front_object_id == "fixed-rock"
    assert labels.visible_bbox_px is None
    assert labels.visible_cx is None
    assert labels.visible_cy is None
    assert labels.visible_log_h is None
    assert not labels.m_vis.any()
    np.testing.assert_array_equal(labels.m_in, labels.target_only_mask)


def test_modal_front_object_is_counted_over_occluded_amodal_pixels_order_independently() -> None:
    target = _target_rectangle(100.0, 100.0, 120.0, 120.0)
    smaller = _occluder_rectangle("a-smaller", 100.0, 100.0, 105.0, 120.0)
    larger = _occluder_rectangle("z-larger", 110.0, 100.0, 120.0, 120.0)
    first = compute_frame_labels(
        target,
        T_cam_from_world=T_CAM_FROM_WORLD,
        K=K,
        occluders=(larger, smaller),
    )
    second = compute_frame_labels(
        target,
        T_cam_from_world=T_CAM_FROM_WORLD,
        K=K,
        occluders=(smaller, larger),
    )

    assert first.A_vis == 100
    assert first.modal_front_object_id == "z-larger"
    assert first.canonical_bytes() == second.canonical_bytes()


def test_fully_off_crop_target_uses_normative_zero_in_frame_formulas() -> None:
    labels = compute_frame_labels(
        _target_rectangle(-40.0, 80.0, -20.0, 100.0),
        T_cam_from_world=T_CAM_FROM_WORLD,
        K=K,
    )

    assert labels.A_full == 400
    assert labels.A_in == 0
    assert labels.A_vis == 0
    assert labels.frame_fraction == 0.0
    assert labels.visible_fraction == 0.0
    assert labels.occlusion_fraction == 1.0
    assert labels.in_frame is False
    assert labels.p_visible_target == 0
    assert labels.occlusion_flag is False
    assert labels.modal_front_object_id is None


@pytest.mark.parametrize(
    ("occluded_width_px", "expected_visible_area", "expected_target"),
    [(8.0, 20, 1), (9.0, 10, 0)],
)
def test_p_visible_uses_full_amodal_area_and_frozen_point_two_threshold(
    occluded_width_px: float,
    expected_visible_area: int,
    expected_target: int,
) -> None:
    target = _target_rectangle(100.0, 100.0, 110.0, 110.0)
    blocker = _occluder_rectangle(
        "blocker",
        100.0,
        100.0,
        100.0 + occluded_width_px,
        110.0,
    )
    labels = compute_frame_labels(
        target,
        T_cam_from_world=T_CAM_FROM_WORLD,
        K=K,
        occluders=(blocker,),
    )

    assert labels.A_full == 100
    assert labels.A_vis == expected_visible_area
    assert labels.visible_fraction == expected_visible_area / 100
    assert labels.p_visible_target == expected_target


def test_amodal_regression_validity_checks_every_depth_height_and_padded_center() -> None:
    good_depths = _f64([3.0, 4.0, 5.0])
    assert amodal_regression_is_valid(good_depths, _i64([0, 0, 224, 224]))
    assert not amodal_regression_is_valid(_f64([3.0, -1.0, 5.0]), _i64([0, 0, 224, 224]))
    assert not amodal_regression_is_valid(good_depths, _i64([0, 10, 20, 10]))
    assert not amodal_regression_is_valid(good_depths, _i64([-100, 0, -80, 20]))

    vertices = np.concatenate(
        [_rectangle_vertices(100.0, 100.0, 110.0, 110.0, depth_m=10.0), _f64([[0.0, 0.0, 1.0]])],
        axis=0,
    )
    labels = compute_frame_labels(
        TargetGeometry(vertices_world_m=vertices, triangles=RECTANGLE_TRIANGLES),
        T_cam_from_world=T_CAM_FROM_WORLD,
        K=K,
    )
    assert labels.whole_target_positive_depth is False
    assert labels.amodal_regression_valid is False


def test_geometry_and_labels_are_immutable_canonical_and_deterministic() -> None:
    source_vertices = _rectangle_vertices(70.0, 80.0, 90.0, 110.0, depth_m=8.0)
    source_triangles = RECTANGLE_TRIANGLES.copy()
    target = TargetGeometry(vertices_world_m=source_vertices, triangles=source_triangles)
    source_vertices[:] = 0.0
    source_triangles[:] = 0

    first = compute_frame_labels(target, T_cam_from_world=T_CAM_FROM_WORLD, K=K)
    second = compute_frame_labels(target, T_cam_from_world=T_CAM_FROM_WORLD.copy(), K=K.copy())
    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.canonical_sha256() == second.canonical_sha256()
    assert first.canonical_sha256() == "fbc033dd9630b39e08db9da46f93653ab1b1d10ca637e574015a1659fd1b804a"
    assert not target.vertices_world_m.flags.writeable
    assert not target.triangles.flags.writeable
    assert not first.m_in.flags.writeable
    assert not first.target_only_mask.flags.writeable
    assert not first.m_vis.flags.writeable
    with pytest.raises(ValueError, match="WRITEABLE"):
        target.vertices_world_m.setflags(write=True)
    with pytest.raises(ValueError, match="WRITEABLE"):
        first.m_vis.setflags(write=True)
    with pytest.raises(TypeError, match="integer 0 or 1"):
        replace(first, p_visible_target=1.0)


@pytest.mark.parametrize("forbidden_key", ["branch_id", "camera_transform", "visibility"])
def test_target_geometry_payload_forbids_branch_camera_and_observation_state(forbidden_key: str) -> None:
    payload: dict[str, object] = {
        "schema_version": "b3-cs3-target-world-triangles-v1",
        "vertices_world_m": _rectangle_vertices(10.0, 10.0, 20.0, 20.0, depth_m=4.0),
        "triangles": RECTANGLE_TRIANGLES,
        forbidden_key: "forbidden",
    }
    with pytest.raises(ValueError, match="unexpected fields"):
        TargetGeometry.from_payload(payload)

    allowed = set(TargetGeometry.__dataclass_fields__)
    assert allowed == {"vertices_world_m", "triangles"}
    assert not any("camera" in field or "branch" in field for field in allowed)


def test_labels_module_is_stdlib_numpy_only_and_has_no_renderer_import() -> None:
    path = Path(inspect.getfile(compute_frame_labels))
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots <= {
        "__future__",
        "collections",
        "dataclasses",
        "hashlib",
        "math",
        "numpy",
        "typing",
        "unicodedata",
        "vllatent",
    }
    assert "bpy" not in path.read_text(encoding="utf-8")
