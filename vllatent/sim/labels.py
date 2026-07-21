"""PURE deterministic B3-CS3 target projection, rasterization, and labels.

Target and occluder geometry is expressed in a camera-independent world frame.
The logged world-to-camera transform and intrinsics are separate inputs.  The
unbounded amodal mask is represented by absolute integer scanline runs, so area
and extrema outside the 224-square crop are retained rather than clipped.

Pixel cells are indexed on the unbounded integer lattice and sampled at
``(u + 0.5, v + 0.5)``.  Triangle edges are inclusive.  At a target pixel, a
non-target object occludes the target only when its perspective-correct optical
depth is strictly smaller.  Equal-depth ties therefore remain target-visible;
equal-depth non-target ties are resolved by normalized object ID order.
"""

from __future__ import annotations

import hashlib
import math
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np

from vllatent.sim.contracts import canonical_bytes

LABEL_SCHEMA_VERSION: Final[str] = "b3-cs3-frame-labels-f64-v1"
RASTER_SCHEMA_VERSION: Final[str] = "b3-cs3-unbounded-inclusive-center-runs-v1"
TARGET_GEOMETRY_SCHEMA_VERSION: Final[str] = "b3-cs3-target-world-triangles-v1"
OCCLUDER_GEOMETRY_SCHEMA_VERSION: Final[str] = "b3-cs3-occluder-world-triangles-v1"
DEPTH_TEST_SCHEMA_VERSION: Final[str] = "b3-cs3-strict-front-perspective-depth-v1"

CROP_WIDTH_PX: Final[int] = 224
CROP_HEIGHT_PX: Final[int] = 224
CROP_ORIGIN_U_PX: Final[int] = 0
CROP_ORIGIN_V_PX: Final[int] = 0
P_VISIBLE_THRESHOLD: Final[float] = 0.20
OCCLUSION_RATIO_THRESHOLD: Final[float] = 0.80
PADDED_CENTER_MIN: Final[float] = -0.25
PADDED_CENTER_MAX: Final[float] = 1.25

TARGET_COMPONENTS: Final[tuple[str, ...]] = (
    "skinned_person",
    "clothing",
    "helmet",
    "boots",
)
EXCLUDED_TARGET_COMPONENTS: Final[tuple[str, ...]] = (
    "skis",
    "poles",
    "detached_equipment",
    "terrain",
    "obstacles",
)

_F64 = np.dtype("<f8")
_I64 = np.dtype("<i8")
_BOOL = np.dtype(np.bool_)
_INT64_MIN = int(np.iinfo(np.int64).min)
_INT64_MAX = int(np.iinfo(np.int64).max)
_SE3_ATOL = 1.0e-10


def _immutable_array(
    name: str,
    value: object,
    *,
    dtype: np.dtype[np.generic],
    shape: tuple[int, ...] | None = None,
    ndim: int | None = None,
    finite: bool = True,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name}: expected np.ndarray, got {type(value).__name__}")
    expected = np.dtype(dtype)
    if value.dtype != expected:
        raise ValueError(f"{name}: expected dtype {expected}, got {value.dtype}")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {value.shape}")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name}: expected {ndim} dimensions, got {value.ndim}")
    if finite and value.dtype.kind == "f" and not np.all(np.isfinite(value)):
        raise ValueError(f"{name}: expected finite values")
    contiguous = np.array(value, dtype=expected, order="C", copy=True)
    if contiguous.dtype.kind == "f":
        contiguous[contiguous == 0.0] = 0.0
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=expected).reshape(contiguous.shape)


def _geometry_arrays(
    vertices_world_m: object,
    triangles: object,
) -> tuple[np.ndarray, np.ndarray]:
    vertices = _immutable_array(
        "vertices_world_m",
        vertices_world_m,
        dtype=_F64,
        ndim=2,
    )
    faces = _immutable_array("triangles", triangles, dtype=_I64, ndim=2, finite=False)
    if vertices.shape[1:] != (3,) or vertices.shape[0] < 3:
        raise ValueError(f"vertices_world_m: expected shape (N>=3,3), got {vertices.shape}")
    if faces.shape[1:] != (3,) or faces.shape[0] < 1:
        raise ValueError(f"triangles: expected shape (M>=1,3), got {faces.shape}")
    if np.any(faces < 0) or np.any(faces >= vertices.shape[0]):
        raise ValueError("triangles: vertex index out of range")
    return vertices, faces


@dataclass(frozen=True, eq=False)
class TargetGeometry:
    """Camera-independent union of exactly the frozen target components."""

    vertices_world_m: np.ndarray
    triangles: np.ndarray

    def __post_init__(self) -> None:
        vertices, triangles = _geometry_arrays(self.vertices_world_m, self.triangles)
        object.__setattr__(self, "vertices_world_m", vertices)
        object.__setattr__(self, "triangles", triangles)

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": TARGET_GEOMETRY_SCHEMA_VERSION,
            "vertices_world_m": self.vertices_world_m,
            "triangles": self.triangles,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> TargetGeometry:
        if not isinstance(payload, Mapping):
            raise TypeError("payload: expected mapping")
        expected = {"schema_version", "vertices_world_m", "triangles"}
        if set(payload) != expected:
            raise ValueError(f"payload: unexpected fields {sorted(set(payload) - expected)}")
        if payload["schema_version"] != TARGET_GEOMETRY_SCHEMA_VERSION:
            raise ValueError("schema_version: unsupported target geometry schema")
        return cls(
            vertices_world_m=payload["vertices_world_m"],  # type: ignore[arg-type]
            triangles=payload["triangles"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, eq=False)
class OccluderGeometry:
    """One named non-target triangle mesh in the camera-independent world frame."""

    object_id: str
    vertices_world_m: np.ndarray
    triangles: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.object_id, str):
            raise TypeError("object_id: expected str")
        object_id = unicodedata.normalize("NFC", self.object_id)
        if not object_id:
            raise ValueError("object_id: expected non-empty string")
        vertices, triangles = _geometry_arrays(self.vertices_world_m, self.triangles)
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "vertices_world_m", vertices)
        object.__setattr__(self, "triangles", triangles)

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": OCCLUDER_GEOMETRY_SCHEMA_VERSION,
            "object_id": self.object_id,
            "vertices_world_m": self.vertices_world_m,
            "triangles": self.triangles,
        }


@dataclass(frozen=True, eq=False)
class RasterRuns:
    """Unbounded binary mask as sorted absolute ``[u_start,u_stop)`` row runs."""

    rows: np.ndarray
    u_start: np.ndarray
    u_stop: np.ndarray

    def __post_init__(self) -> None:
        rows = _immutable_array("rows", self.rows, dtype=_I64, ndim=1, finite=False)
        starts = _immutable_array("u_start", self.u_start, dtype=_I64, ndim=1, finite=False)
        stops = _immutable_array("u_stop", self.u_stop, dtype=_I64, ndim=1, finite=False)
        if rows.shape != starts.shape or rows.shape != stops.shape or rows.size == 0:
            raise ValueError("runs: expected equal non-empty one-dimensional arrays")
        previous: tuple[int, int] | None = None
        for row, start, stop in zip(rows.tolist(), starts.tolist(), stops.tolist(), strict=True):
            if start >= stop:
                raise ValueError("runs: expected u_start < u_stop")
            if previous is not None:
                previous_row, previous_stop = previous
                if row < previous_row or (row == previous_row and start <= previous_stop):
                    raise ValueError("runs: expected sorted, disjoint, non-adjacent spans")
            previous = (row, stop)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "u_start", starts)
        object.__setattr__(self, "u_stop", stops)

    @property
    def area(self) -> int:
        return sum(
            int(stop) - int(start)
            for start, stop in zip(self.u_start, self.u_stop, strict=True)
        )

    @property
    def bbox_px(self) -> np.ndarray:
        minimum_u = min(int(value) for value in self.u_start)
        minimum_v = int(self.rows[0])
        maximum_u = max(int(value) for value in self.u_stop)
        maximum_v = int(self.rows[-1]) + 1
        bbox = np.asarray([minimum_u, minimum_v, maximum_u, maximum_v], dtype=_I64)
        return _immutable_array("bbox_px", bbox, dtype=_I64, shape=(4,), finite=False)

    def crop(
        self,
        *,
        origin_u_px: int = CROP_ORIGIN_U_PX,
        origin_v_px: int = CROP_ORIGIN_V_PX,
        width_px: int = CROP_WIDTH_PX,
        height_px: int = CROP_HEIGHT_PX,
    ) -> np.ndarray:
        for name, value in (
            ("origin_u_px", origin_u_px),
            ("origin_v_px", origin_v_px),
            ("width_px", width_px),
            ("height_px", height_px),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name}: expected int")
        if width_px <= 0 or height_px <= 0:
            raise ValueError("crop: expected positive width and height")
        mask = np.zeros((height_px, width_px), dtype=np.bool_)
        final_u = origin_u_px + width_px
        final_v = origin_v_px + height_px
        for row, start, stop in zip(self.rows, self.u_start, self.u_stop, strict=True):
            absolute_v = int(row)
            if absolute_v < origin_v_px or absolute_v >= final_v:
                continue
            left = max(int(start), origin_u_px)
            right = min(int(stop), final_u)
            if left < right:
                mask[absolute_v - origin_v_px, left - origin_u_px : right - origin_u_px] = True
        return _immutable_array("crop mask", mask, dtype=_BOOL, shape=(height_px, width_px), finite=False)

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": RASTER_SCHEMA_VERSION,
            "rows_v_px": self.rows,
            "u_start_px": self.u_start,
            "u_stop_px_exclusive": self.u_stop,
            "area_px": self.area,
            "bbox_px": self.bbox_px,
        }


def _checked_i64(name: str, value: int) -> int:
    if value < _INT64_MIN or value > _INT64_MAX:
        raise OverflowError(f"{name}: pixel index outside signed int64")
    return value


def _edge(first: np.ndarray, second: np.ndarray, point: np.ndarray) -> float:
    return float(
        (second[0] - first[0]) * (point[1] - first[1])
        - (second[1] - first[1]) * (point[0] - first[0])
    )


def _oriented_triangle(
    projected: np.ndarray,
    optical_depth_m: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None] | None:
    area = _edge(projected[0], projected[1], projected[2])
    if area == 0.0:
        return None
    order = np.asarray([0, 1, 2] if area > 0.0 else [0, 2, 1], dtype=np.int64)
    triangle = projected[order]
    depths = None if optical_depth_m is None else optical_depth_m[order]
    return triangle, depths


def _triangle_scanline_spans(
    projected: np.ndarray,
    *,
    row_min: int | None = None,
    row_max: int | None = None,
) -> list[tuple[int, int, int]]:
    oriented = _oriented_triangle(projected)
    if oriented is None:
        return []
    triangle = oriented[0]
    first_row = math.ceil(float(np.min(triangle[:, 1])) - 0.5)
    final_row = math.floor(float(np.max(triangle[:, 1])) - 0.5)
    if row_min is not None:
        first_row = max(first_row, row_min)
    if row_max is not None:
        final_row = min(final_row, row_max)
    if first_row > final_row:
        return []
    _checked_i64("first row", first_row)
    _checked_i64("final row", final_row)

    spans: list[tuple[int, int, int]] = []
    for row in range(first_row, final_row + 1):
        y = float(row) + 0.5
        lower = -math.inf
        upper = math.inf
        covered = True
        for index in range(3):
            first = triangle[index]
            second = triangle[(index + 1) % 3]
            dx = float(second[0] - first[0])
            dy = float(second[1] - first[1])
            coefficient = -dy
            constant = dx * (y - float(first[1])) + dy * float(first[0])
            if coefficient > 0.0:
                lower = max(lower, -constant / coefficient)
            elif coefficient < 0.0:
                upper = min(upper, -constant / coefficient)
            elif constant < 0.0:
                covered = False
                break
        if not covered or lower > upper:
            continue
        start = math.ceil(lower - 0.5)
        stop = math.floor(upper - 0.5) + 1
        if start < stop:
            spans.append(
                (
                    _checked_i64("row", row),
                    _checked_i64("u_start", start),
                    _checked_i64("u_stop", stop),
                )
            )
    return spans


def rasterize_triangles_unbounded(projected_triangles_px: np.ndarray) -> RasterRuns:
    """Rasterize finite float64 ``(N,3,2)`` triangles without crop clipping."""
    triangles = _immutable_array(
        "projected_triangles_px",
        projected_triangles_px,
        dtype=_F64,
        ndim=3,
    )
    if triangles.shape[1:] != (3, 2) or triangles.shape[0] < 1:
        raise ValueError(f"projected_triangles_px: expected shape (N>=1,3,2), got {triangles.shape}")

    by_row: dict[int, list[tuple[int, int]]] = {}
    for triangle in triangles:
        for row, start, stop in _triangle_scanline_spans(triangle):
            by_row.setdefault(row, []).append((start, stop))
    if not by_row:
        raise ValueError("projected_triangles_px: no pixel-center coverage")

    merged: list[tuple[int, int, int]] = []
    for row in sorted(by_row):
        spans = sorted(by_row[row])
        current_start, current_stop = spans[0]
        for start, stop in spans[1:]:
            if start <= current_stop:
                current_stop = max(current_stop, stop)
            else:
                merged.append((row, current_start, current_stop))
                current_start, current_stop = start, stop
        merged.append((row, current_start, current_stop))
    return RasterRuns(
        rows=np.asarray([item[0] for item in merged], dtype=_I64),
        u_start=np.asarray([item[1] for item in merged], dtype=_I64),
        u_stop=np.asarray([item[2] for item in merged], dtype=_I64),
    )


def _validate_camera_inputs(T_cam_from_world: object, K: object) -> tuple[np.ndarray, np.ndarray]:
    transform = _immutable_array("T_cam_from_world", T_cam_from_world, dtype=_F64, shape=(4, 4))
    intrinsic = _immutable_array("K", K, dtype=_F64, shape=(3, 3))
    if not np.array_equal(transform[3], np.asarray([0.0, 0.0, 0.0, 1.0], dtype=_F64)):
        raise ValueError("T_cam_from_world: invalid homogeneous bottom row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=_SE3_ATOL):
        raise ValueError("T_cam_from_world: expected orthonormal rotation")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, rel_tol=0.0, abs_tol=_SE3_ATOL):
        raise ValueError("T_cam_from_world: expected determinant +1")
    if intrinsic[0, 0] <= 0.0 or intrinsic[1, 1] <= 0.0:
        raise ValueError("K: expected positive focal lengths")
    if not np.array_equal(
        intrinsic,
        np.asarray(
            [
                [intrinsic[0, 0], 0.0, intrinsic[0, 2]],
                [0.0, intrinsic[1, 1], intrinsic[1, 2]],
                [0.0, 0.0, 1.0],
            ],
            dtype=_F64,
        ),
    ):
        raise ValueError("K: expected zero-skew pinhole intrinsics")
    return transform, intrinsic


def _project_mesh(
    vertices_world_m: np.ndarray,
    triangles: np.ndarray,
    transform: np.ndarray,
    intrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices_cam = vertices_world_m @ transform[:3, :3].T + transform[:3, 3]
    optical_depth = -vertices_cam[:, 2]
    projected: list[np.ndarray] = []
    triangle_depths: list[np.ndarray] = []
    for face in triangles:
        camera_triangle = vertices_cam[face]
        depths = optical_depth[face]
        if np.all(depths > 0.0):
            pixels = np.empty((3, 2), dtype=np.float64)
            pixels[:, 0] = (
                intrinsic[0, 0] * camera_triangle[:, 0] / depths + intrinsic[0, 2]
            )
            pixels[:, 1] = (
                intrinsic[1, 2] - intrinsic[1, 1] * camera_triangle[:, 1] / depths
            )
            if not np.all(np.isfinite(pixels)):
                raise ValueError("projection: non-finite projected coordinate")
            projected.append(pixels)
            triangle_depths.append(np.array(depths, dtype=np.float64, copy=True))
        elif np.all(depths <= 0.0):
            continue
        else:
            raise ValueError(
                "triangle crosses the camera plane; its unbounded projection has no finite pixel area"
            )
    if not projected:
        return (
            np.empty((0, 3, 2), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            optical_depth,
        )
    return np.stack(projected), np.stack(triangle_depths), optical_depth


def _crop_depth_buffer(
    projected_triangles_px: np.ndarray,
    optical_depth_m: np.ndarray,
) -> np.ndarray:
    depth_buffer = np.full((CROP_HEIGHT_PX, CROP_WIDTH_PX), np.inf, dtype=np.float64)
    for projected, depths in zip(projected_triangles_px, optical_depth_m, strict=True):
        oriented = _oriented_triangle(projected, depths)
        if oriented is None:
            continue
        triangle, oriented_depths = oriented
        assert oriented_depths is not None
        area = _edge(triangle[0], triangle[1], triangle[2])
        for row, start, stop in _triangle_scanline_spans(
            triangle,
            row_min=CROP_ORIGIN_V_PX,
            row_max=CROP_ORIGIN_V_PX + CROP_HEIGHT_PX - 1,
        ):
            if row < CROP_ORIGIN_V_PX or row >= CROP_ORIGIN_V_PX + CROP_HEIGHT_PX:
                continue
            left = max(start, CROP_ORIGIN_U_PX)
            right = min(stop, CROP_ORIGIN_U_PX + CROP_WIDTH_PX)
            if left >= right:
                continue
            x = np.arange(left, right, dtype=np.float64) + 0.5
            y = float(row) + 0.5
            weight0 = (
                (triangle[2, 0] - triangle[1, 0]) * (y - triangle[1, 1])
                - (triangle[2, 1] - triangle[1, 1]) * (x - triangle[1, 0])
            ) / area
            weight1 = (
                (triangle[0, 0] - triangle[2, 0]) * (y - triangle[2, 1])
                - (triangle[0, 1] - triangle[2, 1]) * (x - triangle[2, 0])
            ) / area
            weight2 = 1.0 - weight0 - weight1
            reciprocal_depth = (
                weight0 / oriented_depths[0]
                + weight1 / oriented_depths[1]
                + weight2 / oriented_depths[2]
            )
            depth = 1.0 / reciprocal_depth
            if not np.all(np.isfinite(depth)) or np.any(depth <= 0.0):
                raise AssertionError("perspective depth interpolation produced an invalid depth")
            row_index = row - CROP_ORIGIN_V_PX
            slice_start = left - CROP_ORIGIN_U_PX
            slice_stop = right - CROP_ORIGIN_U_PX
            current = depth_buffer[row_index, slice_start:slice_stop]
            np.minimum(current, depth, out=current)
    return depth_buffer


def amodal_regression_is_valid(
    target_optical_depth_m: np.ndarray,
    amodal_bbox_px: np.ndarray,
) -> bool:
    """Apply the frozen whole-depth, height, and padded-center validity rule."""
    if not isinstance(target_optical_depth_m, np.ndarray):
        raise TypeError("target_optical_depth_m: expected np.ndarray")
    if target_optical_depth_m.dtype != _F64 or target_optical_depth_m.size == 0:
        raise ValueError("target_optical_depth_m: expected non-empty float64 array")
    bbox = _immutable_array("amodal_bbox_px", amodal_bbox_px, dtype=_I64, shape=(4,), finite=False)
    whole_target_positive = bool(
        np.all(np.isfinite(target_optical_depth_m)) and np.all(target_optical_depth_m > 0.0)
    )
    u_min, v_min, u_max, v_max = (int(item) for item in bbox)
    height_positive = v_max - v_min > 0
    cx = (u_min + u_max) / (2.0 * CROP_WIDTH_PX)
    cy = (v_min + v_max) / (2.0 * CROP_HEIGHT_PX)
    center_padded = (
        PADDED_CENTER_MIN <= cx <= PADDED_CENTER_MAX
        and PADDED_CENTER_MIN <= cy <= PADDED_CENTER_MAX
    )
    return whole_target_positive and height_positive and center_padded


def _tight_mask_bbox(mask: np.ndarray) -> np.ndarray | None:
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        return None
    minimum_v, minimum_u = coordinates.min(axis=0)
    maximum_v, maximum_u = coordinates.max(axis=0)
    bbox = np.asarray(
        [minimum_u, minimum_v, maximum_u + 1, maximum_v + 1],
        dtype=_I64,
    )
    return _immutable_array("visible_bbox_px", bbox, dtype=_I64, shape=(4,), finite=False)


def _normalized_box_values(bbox_px: np.ndarray) -> tuple[float, float, float]:
    u_min, v_min, u_max, v_max = (int(item) for item in bbox_px)
    height = v_max - v_min
    if height <= 0:
        raise ValueError("bbox_px: expected positive height")
    return (
        (u_min + u_max) / (2.0 * CROP_WIDTH_PX),
        (v_min + v_max) / (2.0 * CROP_HEIGHT_PX),
        math.log(height / CROP_HEIGHT_PX),
    )


@dataclass(frozen=True, eq=False)
class FrameLabels:
    """One immutable canonical amodal, crop, depth-visible label record."""

    m_full: RasterRuns
    m_in: np.ndarray
    target_only_mask: np.ndarray
    m_vis: np.ndarray
    A_full: int
    A_in: int
    A_vis: int
    frame_fraction: float
    visible_fraction: float
    occlusion_fraction: float
    amodal_bbox_px: np.ndarray
    cx: float
    cy: float
    log_h: float
    visible_bbox_px: np.ndarray | None
    visible_cx: float | None
    visible_cy: float | None
    visible_log_h: float | None
    whole_target_positive_depth: bool
    amodal_regression_valid: bool
    in_frame: bool
    p_visible_target: int
    occlusion_flag: bool
    modal_front_object_id: str | None
    schema_version: str = LABEL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LABEL_SCHEMA_VERSION:
            raise ValueError("schema_version: unsupported label schema")
        if not isinstance(self.m_full, RasterRuns):
            raise TypeError("m_full: expected RasterRuns")
        m_in = _immutable_array(
            "m_in",
            self.m_in,
            dtype=_BOOL,
            shape=(CROP_HEIGHT_PX, CROP_WIDTH_PX),
            finite=False,
        )
        target_only = _immutable_array(
            "target_only_mask",
            self.target_only_mask,
            dtype=_BOOL,
            shape=(CROP_HEIGHT_PX, CROP_WIDTH_PX),
            finite=False,
        )
        m_vis = _immutable_array(
            "m_vis",
            self.m_vis,
            dtype=_BOOL,
            shape=(CROP_HEIGHT_PX, CROP_WIDTH_PX),
            finite=False,
        )
        if not np.array_equal(m_in, target_only):
            raise ValueError("target-only ID mask must agree with M_in pixel-for-pixel")
        if np.any(m_vis & ~m_in):
            raise ValueError("M_vis must be a subset of M_in")
        object.__setattr__(self, "m_in", m_in)
        object.__setattr__(self, "target_only_mask", target_only)
        object.__setattr__(self, "m_vis", m_vis)

        for area_name, area_value in (("A_full", self.A_full), ("A_in", self.A_in), ("A_vis", self.A_vis)):
            if isinstance(area_value, bool) or not isinstance(area_value, int):
                raise TypeError(f"{area_name}: expected int")
        if self.A_full != self.m_full.area or self.A_full <= 0:
            raise ValueError("A_full: expected exact positive M_full area")
        if self.A_in != int(np.count_nonzero(m_in)):
            raise ValueError("A_in: expected exact M_in area")
        if self.A_vis != int(np.count_nonzero(m_vis)):
            raise ValueError("A_vis: expected exact M_vis area")

        bbox = _immutable_array(
            "amodal_bbox_px",
            self.amodal_bbox_px,
            dtype=_I64,
            shape=(4,),
            finite=False,
        )
        if not np.array_equal(bbox, self.m_full.bbox_px):
            raise ValueError("amodal_bbox_px: expected tight M_full bbox")
        object.__setattr__(self, "amodal_bbox_px", bbox)
        expected_cx, expected_cy, expected_log_h = _normalized_box_values(bbox)
        for box_name, box_value, box_expected in (
            ("cx", self.cx, expected_cx),
            ("cy", self.cy, expected_cy),
            ("log_h", self.log_h, expected_log_h),
        ):
            if not isinstance(box_value, float) or not math.isfinite(box_value) or box_value != box_expected:
                raise ValueError(f"{box_name}: normative amodal formula mismatch")

        expected_visible_bbox = _tight_mask_bbox(m_vis)
        if expected_visible_bbox is None:
            if any(
                value is not None
                for value in (
                    self.visible_bbox_px,
                    self.visible_cx,
                    self.visible_cy,
                    self.visible_log_h,
                )
            ):
                raise ValueError("visible bbox/center/height must be invalid when A_vis=0")
        else:
            visible_bbox = _immutable_array(
                "visible_bbox_px",
                self.visible_bbox_px,
                dtype=_I64,
                shape=(4,),
                finite=False,
            )
            if not np.array_equal(visible_bbox, expected_visible_bbox):
                raise ValueError("visible_bbox_px: expected tight M_vis bbox")
            visible_values = _normalized_box_values(visible_bbox)
            for visible_name, visible_value, visible_expected in zip(
                ("visible_cx", "visible_cy", "visible_log_h"),
                (self.visible_cx, self.visible_cy, self.visible_log_h),
                visible_values,
                strict=True,
            ):
                if (
                    not isinstance(visible_value, float)
                    or not math.isfinite(visible_value)
                    or visible_value != visible_expected
                ):
                    raise ValueError(f"{visible_name}: normative visible formula mismatch")
            object.__setattr__(self, "visible_bbox_px", visible_bbox)

        expected_frame = self.A_in / self.A_full
        expected_visible = self.A_vis / self.A_full
        expected_occlusion = 1.0 - self.A_vis / self.A_in if self.A_in > 0 else 1.0
        for fraction_name, fraction_value, fraction_expected in (
            ("frame_fraction", self.frame_fraction, expected_frame),
            ("visible_fraction", self.visible_fraction, expected_visible),
            ("occlusion_fraction", self.occlusion_fraction, expected_occlusion),
        ):
            if (
                not isinstance(fraction_value, float)
                or not math.isfinite(fraction_value)
                or fraction_value != fraction_expected
            ):
                raise ValueError(f"{fraction_name}: normative area formula mismatch")

        for flag_name, flag_value in (
            ("whole_target_positive_depth", self.whole_target_positive_depth),
            ("amodal_regression_valid", self.amodal_regression_valid),
            ("in_frame", self.in_frame),
            ("occlusion_flag", self.occlusion_flag),
        ):
            if not isinstance(flag_value, bool):
                raise TypeError(f"{flag_name}: expected bool")
        expected_valid = bool(
            self.whole_target_positive_depth
            and bbox[3] > bbox[1]
            and PADDED_CENTER_MIN <= self.cx <= PADDED_CENTER_MAX
            and PADDED_CENTER_MIN <= self.cy <= PADDED_CENTER_MAX
        )
        if self.amodal_regression_valid is not expected_valid:
            raise ValueError("amodal_regression_valid: normative rule mismatch")
        if self.in_frame is not (self.A_in > 0):
            raise ValueError("in_frame: normative rule mismatch")
        if (
            isinstance(self.p_visible_target, bool)
            or not isinstance(self.p_visible_target, int)
            or self.p_visible_target not in (0, 1)
        ):
            raise TypeError("p_visible_target: expected integer 0 or 1")
        expected_target = int(self.in_frame and self.visible_fraction >= P_VISIBLE_THRESHOLD)
        if self.p_visible_target != expected_target:
            raise ValueError("p_visible_target: normative threshold mismatch")

        if self.modal_front_object_id is not None:
            if not isinstance(self.modal_front_object_id, str):
                raise TypeError("modal_front_object_id: expected str or None")
            normalized_id = unicodedata.normalize("NFC", self.modal_front_object_id)
            if not normalized_id:
                raise ValueError("modal_front_object_id: expected non-empty string")
            object.__setattr__(self, "modal_front_object_id", normalized_id)
        if self.A_vis < self.A_in and self.modal_front_object_id is None:
            raise ValueError("modal_front_object_id: required for depth-occluded amodal pixels")
        if self.A_vis == self.A_in and self.modal_front_object_id is not None:
            raise ValueError("modal_front_object_id: forbidden without occluded amodal pixels")
        expected_flag = bool(
            self.A_in > 0
            and self.A_vis / self.A_in < OCCLUSION_RATIO_THRESHOLD
            and self.modal_front_object_id is not None
        )
        if self.occlusion_flag is not expected_flag:
            raise ValueError("occlusion_flag: normative threshold/front-object rule mismatch")

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "raster_schema_version": RASTER_SCHEMA_VERSION,
            "depth_test_schema_version": DEPTH_TEST_SCHEMA_VERSION,
            "crop_origin_uv_px": np.asarray(
                [CROP_ORIGIN_U_PX, CROP_ORIGIN_V_PX], dtype=_I64
            ),
            "crop_size_wh_px": np.asarray([CROP_WIDTH_PX, CROP_HEIGHT_PX], dtype=_I64),
            "target_components": TARGET_COMPONENTS,
            "excluded_target_components": EXCLUDED_TARGET_COMPONENTS,
            "thresholds": {
                "p_visible": P_VISIBLE_THRESHOLD,
                "occlusion_ratio": OCCLUSION_RATIO_THRESHOLD,
                "padded_center_min": PADDED_CENTER_MIN,
                "padded_center_max": PADDED_CENTER_MAX,
            },
            "M_full": self.m_full.payload(),
            "M_in": self.m_in,
            "target_only_ID_mask": self.target_only_mask,
            "M_vis": self.m_vis,
            "A_full": self.A_full,
            "A_in": self.A_in,
            "A_vis": self.A_vis,
            "frame_fraction": self.frame_fraction,
            "visible_fraction": self.visible_fraction,
            "occlusion_fraction": self.occlusion_fraction,
            "amodal_bbox_px": self.amodal_bbox_px,
            "cx": self.cx,
            "cy": self.cy,
            "log_h": self.log_h,
            "visible_bbox_px": self.visible_bbox_px,
            "visible_cx": self.visible_cx,
            "visible_cy": self.visible_cy,
            "visible_log_h": self.visible_log_h,
            "whole_target_positive_depth": self.whole_target_positive_depth,
            "amodal_regression_valid": self.amodal_regression_valid,
            "in_frame": self.in_frame,
            "p_visible_target": self.p_visible_target,
            "occlusion_flag": self.occlusion_flag,
            "modal_front_object_id": self.modal_front_object_id,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.payload())

    def canonical_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def compute_frame_labels(
    target: TargetGeometry,
    *,
    T_cam_from_world: np.ndarray,
    K: np.ndarray,
    occluders: tuple[OccluderGeometry, ...] = (),
) -> FrameLabels:
    """Project and label one target frame under the frozen B3-CS3 formulas."""
    if not isinstance(target, TargetGeometry):
        raise TypeError("target: expected TargetGeometry")
    if not isinstance(occluders, tuple) or not all(
        isinstance(item, OccluderGeometry) for item in occluders
    ):
        raise TypeError("occluders: expected tuple[OccluderGeometry, ...]")
    object_ids = [item.object_id for item in occluders]
    if len(set(object_ids)) != len(object_ids):
        raise ValueError("occluders: object IDs must be unique")
    transform, intrinsic = _validate_camera_inputs(T_cam_from_world, K)

    projected, target_triangle_depths, target_vertex_depths = _project_mesh(
        target.vertices_world_m,
        target.triangles,
        transform,
        intrinsic,
    )
    if projected.shape[0] == 0:
        raise ValueError("target: no positive-depth triangle can define M_full")
    m_full = rasterize_triangles_unbounded(projected)
    m_in = m_full.crop()
    target_depth = _crop_depth_buffer(projected, target_triangle_depths)
    target_only = np.isfinite(target_depth)
    if not np.array_equal(m_in, target_only):
        raise AssertionError("target-only depth pass disagrees with cropped M_full")

    nearest_non_target = np.full(
        (CROP_HEIGHT_PX, CROP_WIDTH_PX),
        np.inf,
        dtype=np.float64,
    )
    nearest_object_index = np.full(
        (CROP_HEIGHT_PX, CROP_WIDTH_PX),
        -1,
        dtype=np.int64,
    )
    ordered_occluders = tuple(sorted(occluders, key=lambda item: item.object_id))
    for object_index, occluder in enumerate(ordered_occluders):
        object_projected, object_triangle_depths, _depths = _project_mesh(
            occluder.vertices_world_m,
            occluder.triangles,
            transform,
            intrinsic,
        )
        if object_projected.shape[0] == 0:
            continue
        object_depth = _crop_depth_buffer(object_projected, object_triangle_depths)
        strictly_nearer = object_depth < nearest_non_target
        nearest_non_target[strictly_nearer] = object_depth[strictly_nearer]
        nearest_object_index[strictly_nearer] = object_index

    occluded = target_only & (nearest_non_target < target_depth)
    m_vis = target_only & ~occluded
    occluded_indices = nearest_object_index[occluded]
    modal_front_object_id: str | None = None
    if occluded_indices.size:
        counts = np.bincount(occluded_indices, minlength=len(ordered_occluders))
        maximum = int(counts.max())
        modal_index = next(index for index, count in enumerate(counts) if int(count) == maximum)
        modal_front_object_id = ordered_occluders[modal_index].object_id

    A_full = m_full.area
    A_in = int(np.count_nonzero(m_in))
    A_vis = int(np.count_nonzero(m_vis))
    frame_fraction = A_in / A_full
    visible_fraction = A_vis / A_full
    occlusion_fraction = 1.0 - A_vis / A_in if A_in > 0 else 1.0
    amodal_bbox = m_full.bbox_px
    cx, cy, log_h = _normalized_box_values(amodal_bbox)
    visible_bbox = _tight_mask_bbox(m_vis)
    if visible_bbox is None:
        visible_cx = visible_cy = visible_log_h = None
    else:
        visible_cx, visible_cy, visible_log_h = _normalized_box_values(visible_bbox)
    whole_target_positive = bool(
        np.all(np.isfinite(target_vertex_depths)) and np.all(target_vertex_depths > 0.0)
    )
    regression_valid = amodal_regression_is_valid(target_vertex_depths, amodal_bbox)
    in_frame = A_in > 0
    p_visible_target = int(in_frame and visible_fraction >= P_VISIBLE_THRESHOLD)
    occlusion_flag = bool(
        A_in > 0
        and A_vis / A_in < OCCLUSION_RATIO_THRESHOLD
        and modal_front_object_id is not None
    )
    return FrameLabels(
        m_full=m_full,
        m_in=m_in,
        target_only_mask=target_only,
        m_vis=m_vis,
        A_full=A_full,
        A_in=A_in,
        A_vis=A_vis,
        frame_fraction=frame_fraction,
        visible_fraction=visible_fraction,
        occlusion_fraction=occlusion_fraction,
        amodal_bbox_px=amodal_bbox,
        cx=cx,
        cy=cy,
        log_h=log_h,
        visible_bbox_px=visible_bbox,
        visible_cx=visible_cx,
        visible_cy=visible_cy,
        visible_log_h=visible_log_h,
        whole_target_positive_depth=whole_target_positive,
        amodal_regression_valid=regression_valid,
        in_frame=in_frame,
        p_visible_target=p_visible_target,
        occlusion_flag=occlusion_flag,
        modal_front_object_id=modal_front_object_id,
    )


__all__ = [
    "CROP_HEIGHT_PX",
    "CROP_ORIGIN_U_PX",
    "CROP_ORIGIN_V_PX",
    "CROP_WIDTH_PX",
    "DEPTH_TEST_SCHEMA_VERSION",
    "EXCLUDED_TARGET_COMPONENTS",
    "FrameLabels",
    "LABEL_SCHEMA_VERSION",
    "OCCLUDER_GEOMETRY_SCHEMA_VERSION",
    "OCCLUSION_RATIO_THRESHOLD",
    "OccluderGeometry",
    "PADDED_CENTER_MAX",
    "PADDED_CENTER_MIN",
    "P_VISIBLE_THRESHOLD",
    "RASTER_SCHEMA_VERSION",
    "RasterRuns",
    "TARGET_COMPONENTS",
    "TARGET_GEOMETRY_SCHEMA_VERSION",
    "TargetGeometry",
    "amodal_regression_is_valid",
    "compute_frame_labels",
    "rasterize_triangles_unbounded",
]
