"""PURE versioned scene, camera, and canonical-root contracts for B3-CS3.

This module is deliberately renderer-neutral.  It binds a parsed, already
audited scene/asset/rig/pose-table set to the complete deterministic root input
that both the isolated Blender bridge and an independent verifier can rebuild.
No file path, wall-clock value, camera branch result, visibility result, or
pixel value is allowed to influence the root identity.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Final

import numpy as np

from vllatent.sim.contracts import (
    CONTRACT_SCHEMA_VERSION,
    FIXED_DT_SECONDS,
    HORIZON_STEPS,
    SKIER_POSE_ROOT_SCHEMA_VERSION,
    SKIER_ROOT_SCHEMA_VERSION,
    BranchId,
    DatasetSplit,
    RootSiblingIdentity,
    camera_contract_sha256,
    canonical_branch_programs,
    canonical_bytes,
    canonical_skier_digest,
    default_camera_contract,
    sha256_canonical,
    validate_sibling_group,
)
from vllatent.sim.pose import (
    IK_SCHEMA_VERSION,
    LOCAL_BONE_TRANSFORM_SEMANTICS,
    POSE_ROOT_SCHEMA_VERSION,
    POSE_SCHEMA_VERSION,
    POSE_TABLE_EXPORT_SCHEMA_VERSION,
    construct_pose_root,
)
from vllatent.sim.rig import (
    ASSET_MANIFEST_SCHEMA_VERSION,
    BLENDER_BINARY_SHA256,
    RIG_MANIFEST_SCHEMA_VERSION,
    RigManifest,
)
from vllatent.sim.skier import (
    ANIMATION_PARAMETER_VERSION,
    INTEGRATOR_VERSION,
    SCHEDULE_SCHEMA_VERSION,
    SKI_CONSTRUCTION_VERSION,
    SkierFrameRecord,
    default_slope_frame,
)
from vllatent.sim.skier_audit import (
    CONTINUATION_AUDIT_VERSION,
    TERMINAL_KEY_VERSION,
    ContinuationAuditResult,
    audit_forecast_continuation,
    audit_terminal_key_collisions,
)
from vllatent.sim.skier_fixtures import (
    FIXTURE_IDS,
    FIXTURE_SCHEMA_VERSION,
    CanonicalSkierFixture,
    canonical_skier_fixtures,
)

SCENE_SCHEMA_VERSION: Final[str] = "b3-cs3-authored-slope-scene-v1"
CAMERA_ROOT_SCHEMA_VERSION: Final[str] = "b3-cs3-static-root-relative-camera-v1"
LABEL_TARGET_SET_VERSION: Final[str] = "b3-cs3-target-person-clothing-helmet-boots-v1"
ROOT_RECORD_SCHEMA_VERSION: Final[str] = "b3-cs3-canonical-root-record-v1"
ROOT_ENVELOPE_SCHEMA_VERSION: Final[str] = "b3-cs3-root-sibling-envelope-v1"
RENDERER_SCHEMA_VERSION: Final[str] = "b3-cs3-cycles-cpu-renderer-v1"
OBSTACLE_SCHEMA_VERSION: Final[str] = "b3-cs3-fixed-occluder-policy-v1"
PNG_SCHEMA_VERSION: Final[str] = "b3-cs3-lossless-png-fixed-filter-v1"
NEAR_CLIP_SCHEMA_VERSION: Final[str] = "b3-cs3-camera-near-plane-triangle-clip-v1"
ID_PASS_SCHEMA_VERSION: Final[str] = "b3-cs3-blender-depsgraph-center-ray-id-v1"

SLOPE_LENGTH_M: Final[float] = 80.0
SLOPE_WIDTH_M: Final[float] = 30.0
CAMERA_UPHILL_OFFSET_M: Final[float] = 8.0
CAMERA_NORMAL_OFFSET_M: Final[float] = 3.0

TARGET_OBJECT_NAMES: Final[tuple[str, ...]] = (
    "CS3_Body",
    "CS3_Eyes",
    "CS3_Eyebrows",
    "CS3_Boot_L",
    "CS3_Boot_R",
    "CS3_Helmet",
)
EXCLUDED_EQUIPMENT_NAMES: Final[tuple[str, ...]] = (
    "CS3_Ski_L",
    "CS3_Ski_R",
    "CS3_Binding_L",
    "CS3_Binding_R",
    "CS3_Pole_L",
    "CS3_Pole_R",
)

_SHA256_HEX_LENGTH: Final[int] = 64
_HISTORY_TICKS: Final[tuple[int, ...]] = (-2, -1, 0)
_FUTURE_TICKS: Final[tuple[int, ...]] = tuple(range(1, HORIZON_STEPS + 1))
_ROOT_RECORD_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "versions",
        "provenance",
        "renderer",
        "identity_basis",
        "time",
        "units",
        "numeric_dtypes",
        "episode_static",
        "continuation",
        "initial_camera",
        "obstacle",
        "history",
        "canonical_skier_pose_table",
        "branch_programs",
    }
)
_ROOT_ENVELOPE_KEYS: Final[frozenset[str]] = frozenset(
    {"schema_version", "root_id", "split_group_id", "split", "record"}
)
_SCENE_RENDER_KEYS: Final[frozenset[str]] = frozenset(
    {
        "engine",
        "device",
        "resolution",
        "samples",
        "adaptive_sampling",
        "seed",
        "animated_seed",
        "denoising",
        "threads",
        "motion_blur",
        "color_transform",
        "exposure",
        "gamma",
        "dither",
    }
)
_OCCLUDER_KEYS: Final[frozenset[str]] = frozenset(
    {"object", "id", "center_world_m", "dimensions_m", "center_fraction", "local_z_offset_m"}
)
_POSE_EXPORT_ROW_KEYS: Final[frozenset[str]] = frozenset(
    {
        "fixture_id",
        "record_index",
        "absolute_tick",
        "source_record_sha256",
        "source_skier_digest",
        "pose_sha256",
        "pose_record_canonical_hex",
        "posed_record_sha256",
        "posed_skier_digest",
        "posed_record_canonical_hex",
        "T_world_from_armature",
        "parent_bone_names",
        "T_root_from_parent_bone",
        "bone_names",
        "T_root_from_bone",
        "local_transform_semantics",
        "local_bone_transforms",
    }
)
_CANONICAL_ARRAY_KEYS: Final[frozenset[str]] = frozenset({"data_hex", "dtype", "shape"})
_CANONICAL_DTYPES: Final[frozenset[str]] = frozenset(
    {"<f4", "<f8", "|i1", "<i2", "<i4", "<i8", "|u1", "<u2", "<u4", "<u8", "|b1"}
)
_CANONICAL_INT_PATTERN: Final[re.Pattern[str]] = re.compile(r"(?:0|-[1-9][0-9]*|[1-9][0-9]*)\Z")
_LOWER_HEX_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]*\Z")


def _require_mapping(name: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name}: expected mapping")
    for key in value:
        if not isinstance(key, str):
            raise TypeError(f"{name}: expected string keys")
    return value


def _require_keys(name: str, value: Mapping[str, object], required: frozenset[str]) -> None:
    missing = required - frozenset(value)
    if missing:
        raise ValueError(f"{name}: missing required keys {sorted(missing)}")


def _require_exact_keys(name: str, value: Mapping[str, object], expected: frozenset[str]) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise ValueError(
            f"{name}: key mismatch: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or len(value) != _SHA256_HEX_LENGTH:
        raise ValueError(f"{name}: expected 64 lowercase hexadecimal characters")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name}: expected 64 lowercase hexadecimal characters")
    return value


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name}: expected non-empty string")
    return value


def _finite_vector(name: str, value: object, size: int) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.dtype("<f8"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name}: expected numeric length-{size} vector") from error
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name}: expected finite numeric length-{size} vector")
    contiguous = np.array(array, dtype=np.dtype("<f8"), order="C", copy=True)
    contiguous[contiguous == 0.0] = 0.0
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=np.dtype("<f8"))


def _deep_freeze(value: object) -> object:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value).copy()
        if array.dtype.kind == "f":
            if not np.all(np.isfinite(array)):
                raise ValueError("root record: non-finite array")
            array[array == 0.0] = 0.0
        return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("root record: expected string mapping keys")
            frozen[key] = _deep_freeze(item)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float, np.generic)):
        if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            raise ValueError("root record: non-finite scalar")
        return value
    # Enum values (for example ManeuverType) are intentionally retained; the
    # shared canonical serializer owns their exact string representation.
    if isinstance(value, Enum):
        return value
    raise TypeError(f"root record: unsupported immutable value {type(value).__name__}")


def _canonical_object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"canonical payload: duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> object:
    raise ValueError(f"canonical payload: non-finite JSON number {value!r}")


def _decode_canonical_value(value: object, *, path: str) -> object:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, dict):
        keys = frozenset(value)
        if keys == {"$int"}:
            encoded = value["$int"]
            if not isinstance(encoded, str) or _CANONICAL_INT_PATTERN.fullmatch(encoded) is None:
                raise ValueError(f"{path}.$int: non-canonical integer encoding")
            return int(encoded)
        if keys == {"$ndarray"}:
            descriptor = value["$ndarray"]
            if not isinstance(descriptor, dict):
                raise ValueError(f"{path}.$ndarray: expected descriptor object")
            _require_exact_keys(f"{path}.$ndarray", descriptor, _CANONICAL_ARRAY_KEYS)
            data_hex = descriptor["data_hex"]
            dtype_text = descriptor["dtype"]
            shape_value = descriptor["shape"]
            if (
                not isinstance(data_hex, str)
                or len(data_hex) % 2
                or _LOWER_HEX_PATTERN.fullmatch(data_hex) is None
            ):
                raise ValueError(f"{path}.$ndarray.data_hex: expected lowercase hexadecimal bytes")
            if not isinstance(dtype_text, str) or dtype_text not in _CANONICAL_DTYPES:
                raise ValueError(f"{path}.$ndarray.dtype: unsupported canonical little-endian dtype")
            if not isinstance(shape_value, list) or any(
                isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in shape_value
            ):
                raise ValueError(f"{path}.$ndarray.shape: expected non-negative integer dimensions")
            shape = tuple(shape_value)
            element_count = math.prod(shape)
            expected_bytes = element_count * np.dtype(dtype_text).itemsize
            if len(data_hex) != 2 * expected_bytes:
                raise ValueError(f"{path}.$ndarray.data_hex: byte length does not match dtype/shape")
            raw = bytes.fromhex(data_hex)
            array = np.frombuffer(raw, dtype=np.dtype(dtype_text)).reshape(shape)
            if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
                raise ValueError(f"{path}.$ndarray: non-finite floating value")
            return array
        if "$int" in keys or "$ndarray" in keys or any(key.startswith("$") for key in keys):
            raise ValueError(f"{path}: canonical tag must be the sole exact mapping key")
        return {
            key: _decode_canonical_value(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, list):
        return tuple(
            _decode_canonical_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, (int, float)):
        raise ValueError(f"{path}: untagged JSON number is not canonical")
    raise TypeError(f"{path}: unsupported decoded JSON value {type(value).__name__}")


def decode_canonical_mapping_bytes(data: bytes, *, label: str = "canonical payload") -> Mapping[str, object]:
    """Decode one exact tagged canonical mapping and prove byte-identical reserialization."""
    if not isinstance(data, bytes):
        raise TypeError("data: expected bytes")
    try:
        parsed = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_canonical_object_pairs,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label}: invalid canonical UTF-8 JSON") from error
    decoded = _decode_canonical_value(parsed, path=label)
    if not isinstance(decoded, Mapping):
        raise ValueError(f"{label}: expected top-level mapping")
    if canonical_bytes(decoded) != data:
        raise ValueError(f"{label}: decoded value does not have byte-identical canonical serialization")
    return decoded


def _canonical_hex_bytes(name: str, value: object) -> bytes:
    if (
        not isinstance(value, str)
        or len(value) % 2
        or _LOWER_HEX_PATTERN.fullmatch(value) is None
    ):
        raise ValueError(f"{name}: expected lowercase hexadecimal canonical bytes")
    return bytes.fromhex(value)


def canonical_renderer_contract() -> dict[str, object]:
    """Return the exact renderer/artifact mapping shared by writer and verifier."""
    return {
        "schema_version": RENDERER_SCHEMA_VERSION,
        "engine": "CYCLES",
        "device": "CPU",
        "resolution": (224, 224),
        "samples": 32,
        "adaptive_sampling": False,
        "seed": 1729,
        "animated_seed": False,
        "denoising": False,
        "threads": 1,
        "motion_blur": False,
        "color_transform": "Standard",
        "exposure": 0.0,
        "gamma": 1.0,
        "dither": 0.0,
        "png_schema_version": PNG_SCHEMA_VERSION,
        "near_clip_schema_version": NEAR_CLIP_SCHEMA_VERSION,
        "id_pass_schema_version": ID_PASS_SCHEMA_VERSION,
    }


def _immutable_transform(value: np.ndarray) -> np.ndarray:
    if value.shape != (4, 4) or value.dtype != np.dtype("<f8") or not np.all(np.isfinite(value)):
        raise ValueError("camera transform must be finite float64 shape (4,4)")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1.0e-12):
        raise ValueError("camera transform rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("camera transform rotation must have determinant +1")
    contiguous = np.array(value, dtype=np.dtype("<f8"), order="C", copy=True)
    contiguous[contiguous == 0.0] = 0.0
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=np.dtype("<f8")).reshape((4, 4))


def initial_camera_rig_transform(record: SkierFrameRecord) -> np.ndarray:
    """Construct one static initial rig aimed once at tick-0 pelvis; no follow constraint."""
    if not isinstance(record, SkierFrameRecord):
        raise TypeError("record: expected SkierFrameRecord")
    slope = default_slope_frame()
    ground = record.root.ground_point_world_m
    origin = (
        ground
        - CAMERA_UPHILL_OFFSET_M * slope.downhill_world
        + CAMERA_NORMAL_OFFSET_M * slope.normal_world
    )
    forward = construct_pose_root(record).pelvis_point_world_m - origin
    forward /= np.linalg.norm(forward)
    right = slope.right_world.copy()
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.column_stack((forward, right, down))
    result[:3, 3] = origin
    return _immutable_transform(result)


def _validate_pose_export_rows(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 88:
        raise ValueError("pose_rows: expected exact ordered 88-row sequence")
    expected_order = [
        (fixture_id, record_index, tick)
        for fixture_id in FIXTURE_IDS
        for record_index, tick in enumerate(_HISTORY_TICKS + _FUTURE_TICKS)
    ]
    rows: list[Mapping[str, object]] = []
    for row_index, (raw_row, expected) in enumerate(zip(value, expected_order, strict=True)):
        row = _require_mapping(f"pose_rows[{row_index}]", raw_row)
        _require_exact_keys(f"pose_rows[{row_index}]", row, _POSE_EXPORT_ROW_KEYS)
        fixture_id, record_index, absolute_tick = expected
        if (
            row["fixture_id"] != fixture_id
            or isinstance(row["record_index"], bool)
            or row["record_index"] != record_index
            or isinstance(row["absolute_tick"], bool)
            or row["absolute_tick"] != absolute_tick
        ):
            raise ValueError(f"pose_rows[{row_index}]: fixture/index/absolute-tick order drift")
        for hash_field in (
            "source_record_sha256",
            "source_skier_digest",
            "pose_sha256",
            "posed_record_sha256",
            "posed_skier_digest",
        ):
            _sha256(f"pose_rows[{row_index}].{hash_field}", row[hash_field])
        pose_bytes = _canonical_hex_bytes(
            f"pose_rows[{row_index}].pose_record_canonical_hex",
            row["pose_record_canonical_hex"],
        )
        posed_bytes = _canonical_hex_bytes(
            f"pose_rows[{row_index}].posed_record_canonical_hex",
            row["posed_record_canonical_hex"],
        )
        if hashlib.sha256(pose_bytes).hexdigest() != row["pose_sha256"]:
            raise ValueError(f"pose_rows[{row_index}]: pose canonical bytes/hash mismatch")
        if hashlib.sha256(posed_bytes).hexdigest() != row["posed_record_sha256"]:
            raise ValueError(f"pose_rows[{row_index}]: posed canonical bytes/hash mismatch")
        if row["local_transform_semantics"] != LOCAL_BONE_TRANSFORM_SEMANTICS:
            raise ValueError(f"pose_rows[{row_index}]: local transform semantics drift")
        numeric_shapes = {
            "T_world_from_armature": (4, 4),
            "T_root_from_parent_bone": (5, 4, 4),
            "T_root_from_bone": (17, 4, 4),
            "local_bone_transforms": (17, 4, 4),
        }
        for field_name, shape in numeric_shapes.items():
            try:
                array = np.asarray(row[field_name], dtype=np.dtype("<f8"))
            except (TypeError, ValueError) as error:
                raise ValueError(f"pose_rows[{row_index}].{field_name}: expected numeric array") from error
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ValueError(f"pose_rows[{row_index}].{field_name}: expected finite shape {shape}")
        if (
            not isinstance(row["parent_bone_names"], Sequence)
            or isinstance(row["parent_bone_names"], (str, bytes))
            or len(row["parent_bone_names"]) != 5
            or not all(isinstance(name, str) and name for name in row["parent_bone_names"])
            or not isinstance(row["bone_names"], Sequence)
            or isinstance(row["bone_names"], (str, bytes))
            or len(row["bone_names"]) != 17
            or not all(isinstance(name, str) and name for name in row["bone_names"])
        ):
            raise ValueError(f"pose_rows[{row_index}]: manifested bone-name order drift")
        frozen = _deep_freeze(row)
        assert isinstance(frozen, Mapping)
        rows.append(frozen)
    return tuple(rows)


@dataclass(frozen=True, eq=False)
class RootRecordBindings:
    """Strict hash/version bridge from parsed tracked manifests to a PURE root."""

    asset_manifest_schema_version: str
    asset_manifest_file_sha256: str
    source_archive_sha256: str
    source_asset_sha256: str
    license_file_sha256: str
    license_manifest_sha256: str
    rig_manifest_schema_version: str
    rig_manifest_file_sha256: str
    rig_manifest_canonical_sha256: str
    derived_rig_sha256: str
    pose_export_schema_version: str
    pose_export_file_sha256: str
    canonical_pose_table_sha256: str
    animation_clip_table_sha256: str
    scene_manifest_schema_version: str
    scene_manifest_file_sha256: str
    scene_sha256: str
    blender_version: str
    blender_build_hash: str
    blender_binary_sha256: str
    camera_setup_sha256: str
    pose_rows: tuple[Mapping[str, object], ...] = field(repr=False)
    renderer: Mapping[str, object]
    occluder: Mapping[str, object]

    def __post_init__(self) -> None:
        expected_versions = {
            "asset_manifest_schema_version": ASSET_MANIFEST_SCHEMA_VERSION,
            "rig_manifest_schema_version": RIG_MANIFEST_SCHEMA_VERSION,
            "pose_export_schema_version": POSE_TABLE_EXPORT_SCHEMA_VERSION,
            "scene_manifest_schema_version": SCENE_SCHEMA_VERSION,
        }
        for name, expected in expected_versions.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name}: expected {expected!r}")
        for name in (
            "asset_manifest_file_sha256",
            "source_archive_sha256",
            "source_asset_sha256",
            "license_file_sha256",
            "license_manifest_sha256",
            "rig_manifest_file_sha256",
            "rig_manifest_canonical_sha256",
            "derived_rig_sha256",
            "pose_export_file_sha256",
            "canonical_pose_table_sha256",
            "animation_clip_table_sha256",
            "scene_manifest_file_sha256",
            "scene_sha256",
            "blender_binary_sha256",
            "camera_setup_sha256",
        ):
            object.__setattr__(self, name, _sha256(name, getattr(self, name)))
        _text("blender_version", self.blender_version)
        _text("blender_build_hash", self.blender_build_hash)
        object.__setattr__(self, "pose_rows", _validate_pose_export_rows(self.pose_rows))
        renderer = _require_mapping("renderer", self.renderer)
        expected_renderer = canonical_renderer_contract()
        if dict(renderer) != expected_renderer:
            raise ValueError("renderer: exact Cycles CPU/artifact contract drift")
        object.__setattr__(self, "renderer", _deep_freeze(renderer))
        occluder = _require_mapping("occluder", self.occluder)
        _require_exact_keys("occluder", occluder, _OCCLUDER_KEYS)
        center = _finite_vector("occluder.center_world_m", occluder["center_world_m"], 3)
        dimensions = _finite_vector("occluder.dimensions_m", occluder["dimensions_m"], 3)
        if not np.array_equal(dimensions, np.array([0.4, 1.1, 1.0], dtype=np.float64)):
            raise ValueError("occluder.dimensions_m: fixed obstacle dimensions drift")
        if occluder["object"] != "CS3_FixedOccluder" or occluder["id"] != "fixed_occluder_101":
            raise ValueError("occluder: fixed object/ID drift")
        if occluder["center_fraction"] != 0.58 or occluder["local_z_offset_m"] != 0.15:
            raise ValueError("occluder: fixed placement policy drift")
        object.__setattr__(
            self,
            "occluder",
            _deep_freeze(
                {
                    "object": occluder["object"],
                    "id": occluder["id"],
                    "center_world_m": center,
                    "dimensions_m": dimensions,
                    "center_fraction": 0.58,
                    "local_z_offset_m": 0.15,
                }
            ),
        )
        if self.license_manifest_sha256 != self.asset_manifest_file_sha256:
            raise ValueError("license_manifest_sha256: the audited asset/license manifest must be bound exactly")
        if self.animation_clip_table_sha256 != self.canonical_pose_table_sha256:
            raise ValueError("animation_clip_table_sha256: must bind the authoritative pose/clip table")

    @classmethod
    def from_manifests(
        cls,
        *,
        asset_manifest: Mapping[str, object],
        asset_manifest_file_sha256: str,
        rig_manifest: RigManifest,
        rig_manifest_file_sha256: str,
        scene_manifest: Mapping[str, object],
        scene_manifest_file_sha256: str,
        pose_export_manifest: Mapping[str, object],
        pose_export_file_sha256: str,
    ) -> RootRecordBindings:
        """Extract and cross-check all root bindings from parsed tracked manifests."""
        asset = _require_mapping("asset_manifest", asset_manifest)
        scene = _require_mapping("scene_manifest", scene_manifest)
        pose_export = _require_mapping("pose_export_manifest", pose_export_manifest)
        if not isinstance(rig_manifest, RigManifest):
            raise TypeError("rig_manifest: expected RigManifest")
        _require_keys(asset.__class__.__name__, asset, frozenset({"schema_version", "pack", "blender"}))
        _require_keys(
            "scene_manifest",
            scene,
            frozenset(
                {
                    "schema_version",
                    "scene_sha256",
                    "derived_rig_sha256",
                    "rig_manifest_canonical_sha256",
                    "asset_manifest_file_sha256",
                    "blender_version",
                    "blender_build_hash",
                    "render",
                    "camera_contract_sha256",
                    "occluder",
                }
            ),
        )
        _require_keys(
            "pose_export_manifest",
            pose_export,
            frozenset(
                {
                    "schema_version",
                    "rig_manifest_canonical_sha256",
                    "canonical_pose_table_sha256",
                    "fixture_count",
                    "sample_count",
                    "rows",
                }
            ),
        )
        pack = _require_mapping("asset_manifest.pack", asset["pack"])
        blender = _require_mapping("asset_manifest.blender", asset["blender"])
        _require_keys(
            "asset_manifest.pack",
            pack,
            frozenset({"archive_sha256", "selected_member_sha256", "license_sha256"}),
        )
        _require_keys(
            "asset_manifest.blender",
            blender,
            frozenset({"version", "binary_sha256", "build_hash", "render_device"}),
        )
        asset_file_hash = _sha256("asset_manifest_file_sha256", asset_manifest_file_sha256)
        rig_file_hash = _sha256("rig_manifest_file_sha256", rig_manifest_file_sha256)
        scene_file_hash = _sha256("scene_manifest_file_sha256", scene_manifest_file_sha256)
        pose_file_hash = _sha256("pose_export_file_sha256", pose_export_file_sha256)
        rig_canonical_hash = rig_manifest.canonical_sha256()
        if scene["asset_manifest_file_sha256"] != asset_file_hash:
            raise ValueError("scene_manifest.asset_manifest_file_sha256: parsed asset manifest mismatch")
        if scene["derived_rig_sha256"] != rig_manifest.derived_rig_sha256:
            raise ValueError("scene_manifest.derived_rig_sha256: parsed rig mismatch")
        if scene["rig_manifest_canonical_sha256"] != rig_canonical_hash:
            raise ValueError("scene_manifest.rig_manifest_canonical_sha256: parsed rig mismatch")
        if pose_export["rig_manifest_canonical_sha256"] != rig_canonical_hash:
            raise ValueError("pose_export_manifest.rig_manifest_canonical_sha256: parsed rig mismatch")
        if pack["selected_member_sha256"] != rig_manifest.source_asset_sha256:
            raise ValueError("asset_manifest.pack.selected_member_sha256: parsed rig source mismatch")
        if blender["version"] != scene["blender_version"] or blender["build_hash"] != scene["blender_build_hash"]:
            raise ValueError("asset/scene Blender build mismatch")
        if blender["binary_sha256"] != BLENDER_BINARY_SHA256 or blender["render_device"] != "CPU":
            raise ValueError("asset_manifest.blender: pinned binary/device drift")
        scene_render = _require_mapping("scene_manifest.render", scene["render"])
        _require_exact_keys("scene_manifest.render", scene_render, _SCENE_RENDER_KEYS)
        expected_scene_render = {
            key: value
            for key, value in canonical_renderer_contract().items()
            if key in _SCENE_RENDER_KEYS
        }
        normalized_scene_render = dict(scene_render)
        resolution = normalized_scene_render.get("resolution")
        if isinstance(resolution, list):
            normalized_scene_render["resolution"] = tuple(resolution)
        if normalized_scene_render != expected_scene_render:
            raise ValueError("scene_manifest.render: exact Cycles CPU settings drift")
        camera_setup_hash = hashlib.sha256(canonical_bytes(default_camera_contract().manifest())).hexdigest()
        if scene["camera_contract_sha256"] != camera_setup_hash:
            raise ValueError("scene_manifest.camera_contract_sha256: frozen camera setup drift")
        if pose_export["fixture_count"] != 8 or pose_export["sample_count"] != 88:
            raise ValueError("pose_export_manifest: expected canonical 8-root/88-row fixture table")
        rows = pose_export["rows"]
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or len(rows) != 88:
            raise ValueError("pose_export_manifest.rows: expected 88 rows")
        return cls(
            asset_manifest_schema_version=_text("asset schema_version", asset["schema_version"]),
            asset_manifest_file_sha256=asset_file_hash,
            source_archive_sha256=_sha256("pack.archive_sha256", pack["archive_sha256"]),
            source_asset_sha256=_sha256("pack.selected_member_sha256", pack["selected_member_sha256"]),
            license_file_sha256=_sha256("pack.license_sha256", pack["license_sha256"]),
            license_manifest_sha256=asset_file_hash,
            rig_manifest_schema_version=RIG_MANIFEST_SCHEMA_VERSION,
            rig_manifest_file_sha256=rig_file_hash,
            rig_manifest_canonical_sha256=rig_canonical_hash,
            derived_rig_sha256=rig_manifest.derived_rig_sha256,
            pose_export_schema_version=_text("pose export schema_version", pose_export["schema_version"]),
            pose_export_file_sha256=pose_file_hash,
            canonical_pose_table_sha256=_sha256(
                "pose_export.canonical_pose_table_sha256",
                pose_export["canonical_pose_table_sha256"],
            ),
            animation_clip_table_sha256=_sha256(
                "pose_export.canonical_pose_table_sha256",
                pose_export["canonical_pose_table_sha256"],
            ),
            scene_manifest_schema_version=_text("scene schema_version", scene["schema_version"]),
            scene_manifest_file_sha256=scene_file_hash,
            scene_sha256=_sha256("scene.scene_sha256", scene["scene_sha256"]),
            blender_version=_text("blender.version", blender["version"]),
            blender_build_hash=_text("blender.build_hash", blender["build_hash"]),
            blender_binary_sha256=_sha256("blender.binary_sha256", blender["binary_sha256"]),
            camera_setup_sha256=camera_setup_hash,
            pose_rows=tuple(rows),
            renderer=canonical_renderer_contract(),
            occluder=_require_mapping("scene_manifest.occluder", scene["occluder"]),
        )

    def fixture_pose_rows(self, fixture_id: str) -> tuple[Mapping[str, object], ...]:
        """Return the exact eleven authoritative rows for one canonical fixture."""
        if fixture_id not in FIXTURE_IDS:
            raise ValueError(f"fixture_id: unknown canonical fixture {fixture_id!r}")
        start = FIXTURE_IDS.index(fixture_id) * len(_HISTORY_TICKS + _FUTURE_TICKS)
        rows = self.pose_rows[start : start + len(_HISTORY_TICKS + _FUTURE_TICKS)]
        if tuple(row["fixture_id"] for row in rows) != (fixture_id,) * 11:
            raise ValueError("pose_rows: fixture grouping drift")
        return rows

    def payload(self) -> dict[str, object]:
        """Return the exact provenance/version mapping included in every root."""
        return {
            "asset_manifest_schema_version": self.asset_manifest_schema_version,
            "asset_manifest_file_sha256": self.asset_manifest_file_sha256,
            "source_archive_sha256": self.source_archive_sha256,
            "source_asset_sha256": self.source_asset_sha256,
            "license_file_sha256": self.license_file_sha256,
            "license_manifest_sha256": self.license_manifest_sha256,
            "rig_manifest_schema_version": self.rig_manifest_schema_version,
            "rig_manifest_file_sha256": self.rig_manifest_file_sha256,
            "rig_manifest_canonical_sha256": self.rig_manifest_canonical_sha256,
            "derived_rig_sha256": self.derived_rig_sha256,
            "pose_export_schema_version": self.pose_export_schema_version,
            "pose_export_file_sha256": self.pose_export_file_sha256,
            "canonical_pose_table_sha256": self.canonical_pose_table_sha256,
            "animation_clip_table_sha256": self.animation_clip_table_sha256,
            "scene_manifest_schema_version": self.scene_manifest_schema_version,
            "scene_manifest_file_sha256": self.scene_manifest_file_sha256,
            "scene_sha256": self.scene_sha256,
            "blender_version": self.blender_version,
            "blender_build_hash": self.blender_build_hash,
            "blender_binary_sha256": self.blender_binary_sha256,
            "camera_setup_sha256": self.camera_setup_sha256,
        }


def _mapping_field(value: Mapping[str, object], field_name: str, *, path: str) -> Mapping[str, object]:
    field_value = value.get(field_name)
    if not isinstance(field_value, Mapping):
        raise ValueError(f"{path}.{field_name}: expected mapping")
    return field_value


def _decoded_int(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path}: expected canonical integer")
    return value


def _decoded_float(value: object, *, path: str) -> float:
    if (
        not isinstance(value, np.ndarray)
        or value.shape != (1,)
        or value.dtype != np.dtype("<f8")
        or not np.all(np.isfinite(value))
    ):
        raise ValueError(f"{path}: expected canonical float64 scalar")
    return float(value[0])


def _decoded_array(value: object, *, shape: tuple[int, ...], path: str) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.shape != shape
        or value.dtype != np.dtype("<f8")
        or not np.all(np.isfinite(value))
    ):
        raise ValueError(f"{path}: expected canonical float64 shape {shape}")
    return value


def _decoded_skier_digest(payload: Mapping[str, object], *, path: str) -> str:
    root = _mapping_field(payload, "root", path=path)
    skis = _mapping_field(payload, "skis", path=path)
    left = _mapping_field(skis, "left", path=f"{path}.skis")
    right = _mapping_field(skis, "right", path=f"{path}.skis")
    maneuver = _mapping_field(payload, "maneuver", path=path)
    animation = _mapping_field(payload, "animation", path=path)
    schema_version = payload.get("schema_version")
    if schema_version != SKIER_POSE_ROOT_SCHEMA_VERSION:
        raise ValueError(f"{path}.schema_version: expected posed skier-root schema")

    def normalized_ski(ski: Mapping[str, object], side: str) -> dict[str, object]:
        return {
            "side": ski.get("side"),
            "attack_rad": _decoded_float(ski.get("attack_rad"), path=f"{path}.skis.{side}.attack_rad"),
            "edge_rad": _decoded_float(ski.get("edge_rad"), path=f"{path}.skis.{side}.edge_rad"),
            "centerline_origin_world_m": _decoded_array(
                ski.get("centerline_origin_world_m"),
                shape=(3,),
                path=f"{path}.skis.{side}.centerline_origin_world_m",
            ),
            "base_origin_world_m": _decoded_array(
                ski.get("base_origin_world_m"),
                shape=(3,),
                path=f"{path}.skis.{side}.base_origin_world_m",
            ),
            "binding_origin_world_m": _decoded_array(
                ski.get("binding_origin_world_m"),
                shape=(3,),
                path=f"{path}.skis.{side}.binding_origin_world_m",
            ),
            "contact_origin_world_m": _decoded_array(
                ski.get("contact_origin_world_m"),
                shape=(3,),
                path=f"{path}.skis.{side}.contact_origin_world_m",
            ),
            "target_F_world_from_ski": _decoded_array(
                ski.get("target_F_world_from_ski"),
                shape=(3, 3),
                path=f"{path}.skis.{side}.target_F_world_from_ski",
            ),
            "realized_F_world_from_ski": _decoded_array(
                ski.get("realized_F_world_from_ski"),
                shape=(3, 3),
                path=f"{path}.skis.{side}.realized_F_world_from_ski",
            ),
            "analytic_slip_longitudinal_lateral_m_s": _decoded_array(
                ski.get("analytic_slip_longitudinal_lateral_m_s"),
                shape=(2,),
                path=f"{path}.skis.{side}.analytic_slip_longitudinal_lateral_m_s",
            ),
            "realized_slip_longitudinal_lateral_m_s": _decoded_array(
                ski.get("realized_slip_longitudinal_lateral_m_s"),
                shape=(2,),
                path=f"{path}.skis.{side}.realized_slip_longitudinal_lateral_m_s",
            ),
            "realized_attack_rad": _decoded_float(
                ski.get("realized_attack_rad"), path=f"{path}.skis.{side}.realized_attack_rad"
            ),
            "realized_edge_rad": _decoded_float(
                ski.get("realized_edge_rad"), path=f"{path}.skis.{side}.realized_edge_rad"
            ),
            "frame_orientation_residual_rad": _decoded_float(
                ski.get("frame_orientation_residual_rad"),
                path=f"{path}.skis.{side}.frame_orientation_residual_rad",
            ),
        }

    normalized_skis = {
        "dimensions_m": _decoded_array(
            skis.get("dimensions_m"), shape=(3,), path=f"{path}.skis.dimensions_m"
        ),
        "stance_half_width_m": _decoded_float(
            skis.get("stance_half_width_m"), path=f"{path}.skis.stance_half_width_m"
        ),
        "centerline_ordering_m": _decoded_float(
            skis.get("centerline_ordering_m"), path=f"{path}.skis.centerline_ordering_m"
        ),
        "inner_tip_gap_m": _decoded_float(
            skis.get("inner_tip_gap_m"), path=f"{path}.skis.inner_tip_gap_m"
        ),
        "left": normalized_ski(left, "left"),
        "right": normalized_ski(right, "right"),
    }
    clip_ids = animation.get("clip_ids")
    if (
        not isinstance(clip_ids, tuple)
        or not clip_ids
        or not all(isinstance(clip_id, str) and clip_id for clip_id in clip_ids)
    ):
        raise ValueError(f"{path}.animation.clip_ids: expected non-empty canonical string tuple")
    return canonical_skier_digest(
        root={
            "schema_version": schema_version,
            "absolute_tick": _decoded_int(payload.get("absolute_tick"), path=f"{path}.absolute_tick"),
            "position_xy_m": _decoded_array(
                payload.get("position_xy_m"), shape=(2,), path=f"{path}.position_xy_m"
            ),
            "heading_rad": _decoded_float(payload.get("heading_rad"), path=f"{path}.heading_rad"),
            "speed_m_s": _decoded_float(payload.get("speed_m_s"), path=f"{path}.speed_m_s"),
            "acceleration_m_s2": _decoded_float(
                payload.get("acceleration_m_s2"), path=f"{path}.acceleration_m_s2"
            ),
            "curvature_1_m": _decoded_float(
                payload.get("curvature_1_m"), path=f"{path}.curvature_1_m"
            ),
            "omega_rad_s": _decoded_float(payload.get("omega_rad_s"), path=f"{path}.omega_rad_s"),
            "gross_lean_rad": _decoded_float(
                payload.get("gross_lean_rad"), path=f"{path}.gross_lean_rad"
            ),
            "T_world_from_groundroot": _decoded_array(
                root.get("T_world_from_groundroot"),
                shape=(4, 4),
                path=f"{path}.root.T_world_from_groundroot",
            ),
            "T_world_from_armature": _decoded_array(
                root.get("T_world_from_armature"),
                shape=(4, 4),
                path=f"{path}.root.T_world_from_armature",
            ),
            "tracked_joint_positions_root_m": _decoded_array(
                payload.get("tracked_joint_positions_root_m"),
                shape=(17, 3),
                path=f"{path}.tracked_joint_positions_root_m",
            ),
        },
        skis=normalized_skis,
        contacts={
            "left_contact_origin_world_m": _decoded_array(
                left.get("contact_origin_world_m"),
                shape=(3,),
                path=f"{path}.skis.left.contact_origin_world_m",
            ),
            "right_contact_origin_world_m": _decoded_array(
                right.get("contact_origin_world_m"),
                shape=(3,),
                path=f"{path}.skis.right.contact_origin_world_m",
            ),
        },
        phases={
            "maneuver_id": maneuver.get("id"),
            "maneuver_phase": _decoded_float(
                maneuver.get("phase"), path=f"{path}.maneuver.phase"
            ),
            "animation_clip_ids": clip_ids,
            "animation_phase": _decoded_float(
                animation.get("phase"), path=f"{path}.animation.phase"
            ),
            "animation_blend_weights": _decoded_array(
                animation.get("blend_weights"),
                shape=(len(clip_ids),),
                path=f"{path}.animation.blend_weights",
            ),
        },
        local_bone_transforms=_decoded_array(
            payload.get("local_bone_transforms"),
            shape=(17, 4, 4),
            path=f"{path}.local_bone_transforms",
        ),
        randomness={
            "seed": _decoded_int(payload.get("randomness_seed"), path=f"{path}.randomness_seed")
        },
    )


def _require_export_array_match(
    row: Mapping[str, object],
    row_field: str,
    payload: Mapping[str, object],
    payload_field: str,
    shape: tuple[int, ...],
    *,
    path: str,
) -> None:
    exported = np.asarray(row[row_field], dtype=np.dtype("<f8"))
    canonical = _decoded_array(payload.get(payload_field), shape=shape, path=f"{path}.{payload_field}")
    if not np.array_equal(exported, canonical):
        raise ValueError(f"{path}: exported {row_field} differs from canonical pose payload")


def _canonical_table_row(
    fixture_id: str,
    record_index: int,
    source_record: SkierFrameRecord,
    authoritative_row: Mapping[str, object],
    rig_manifest_sha256: str,
) -> dict[str, object]:
    path = f"pose_rows[{fixture_id}][{record_index}]"
    source_bytes = source_record.canonical_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    source_digest = source_record.skier_digest()
    if (
        authoritative_row["fixture_id"] != fixture_id
        or authoritative_row["record_index"] != record_index
        or authoritative_row["absolute_tick"] != source_record.state.absolute_tick
    ):
        raise ValueError(f"{path}: fixture/index/tick mismatch")
    if authoritative_row["source_record_sha256"] != source_sha256:
        raise ValueError(f"{path}: independently recomputed source-record hash mismatch")
    if authoritative_row["source_skier_digest"] != source_digest:
        raise ValueError(f"{path}: independently recomputed source-skier digest mismatch")
    pose_bytes = _canonical_hex_bytes(
        f"{path}.pose_record_canonical_hex", authoritative_row["pose_record_canonical_hex"]
    )
    posed_bytes = _canonical_hex_bytes(
        f"{path}.posed_record_canonical_hex", authoritative_row["posed_record_canonical_hex"]
    )
    pose = decode_canonical_mapping_bytes(pose_bytes, label=f"{path}.pose_record")
    posed = decode_canonical_mapping_bytes(posed_bytes, label=f"{path}.posed_record")
    if hashlib.sha256(pose_bytes).hexdigest() != authoritative_row["pose_sha256"]:
        raise ValueError(f"{path}: pose canonical hash mismatch")
    if hashlib.sha256(posed_bytes).hexdigest() != authoritative_row["posed_record_sha256"]:
        raise ValueError(f"{path}: posed canonical hash mismatch")
    if (
        _decoded_int(pose.get("absolute_tick"), path=f"{path}.pose_record.absolute_tick")
        != source_record.state.absolute_tick
        or pose.get("rig_manifest_sha256") != rig_manifest_sha256
        or pose.get("source_skier_digest") != source_digest
        or _decoded_int(posed.get("absolute_tick"), path=f"{path}.posed_record.absolute_tick")
        != source_record.state.absolute_tick
    ):
        raise ValueError(f"{path}: canonical pose/posed fixture binding mismatch")
    if _decoded_skier_digest(posed, path=f"{path}.posed_record") != authoritative_row["posed_skier_digest"]:
        raise ValueError(f"{path}: independently recomputed posed-skier digest mismatch")
    _require_export_array_match(
        authoritative_row,
        "T_world_from_armature",
        pose,
        "T_world_from_armature",
        (4, 4),
        path=f"{path}.pose_record",
    )
    _require_export_array_match(
        authoritative_row,
        "T_root_from_parent_bone",
        pose,
        "T_root_from_parent_bone",
        (5, 4, 4),
        path=f"{path}.pose_record",
    )
    _require_export_array_match(
        authoritative_row,
        "T_root_from_bone",
        pose,
        "T_root_from_bone",
        (17, 4, 4),
        path=f"{path}.pose_record",
    )
    _require_export_array_match(
        authoritative_row,
        "local_bone_transforms",
        pose,
        "local_bone_transforms",
        (17, 4, 4),
        path=f"{path}.pose_record",
    )
    exported_parent_names = authoritative_row["parent_bone_names"]
    exported_bone_names = authoritative_row["bone_names"]
    if not isinstance(exported_parent_names, tuple) or not isinstance(exported_bone_names, tuple):
        raise ValueError(f"{path}: frozen exported bone names must be tuples")
    if (
        pose.get("parent_bone_names") != exported_parent_names
        or pose.get("bone_names") != exported_bone_names
        or pose.get("local_transform_semantics") != authoritative_row["local_transform_semantics"]
    ):
        raise ValueError(f"{path}: exported pose names/semantics differ from canonical payload")
    posed_root = _mapping_field(posed, "root", path=f"{path}.posed_record")
    if not np.array_equal(
        _decoded_array(
            posed_root.get("T_world_from_armature"),
            shape=(4, 4),
            path=f"{path}.posed_record.root.T_world_from_armature",
        ),
        _decoded_array(
            pose.get("T_world_from_armature"),
            shape=(4, 4),
            path=f"{path}.pose_record.T_world_from_armature",
        ),
    ) or not np.array_equal(
        _decoded_array(
            posed.get("local_bone_transforms"),
            shape=(17, 4, 4),
            path=f"{path}.posed_record.local_bone_transforms",
        ),
        _decoded_array(
            pose.get("local_bone_transforms"),
            shape=(17, 4, 4),
            path=f"{path}.pose_record.local_bone_transforms",
        ),
    ):
        raise ValueError(f"{path}: canonical pose and posed-record transforms disagree")
    return {
        "record_index": record_index,
        "absolute_tick": source_record.state.absolute_tick,
        "source_record_sha256": source_sha256,
        "source_skier_digest": source_digest,
        "source_record": source_record.payload(),
        "pose_record_sha256": authoritative_row["pose_sha256"],
        "pose_record": pose,
        "posed_record_sha256": authoritative_row["posed_record_sha256"],
        "posed_skier_digest": authoritative_row["posed_skier_digest"],
        "posed_record": posed,
    }


def _camera_root_payload(tick_zero_record: SkierFrameRecord, bindings: RootRecordBindings) -> dict[str, object]:
    camera = default_camera_contract()
    T_rig_from_cam = np.eye(4, dtype=np.float64)
    T_rig_from_cam[:3, :3] = camera.R_rig_from_cam
    T_cam_from_rig = np.eye(4, dtype=np.float64)
    T_cam_from_rig[:3, :3] = camera.R_cam_from_rig
    T_world_from_rig = initial_camera_rig_transform(tick_zero_record)
    return {
        "schema_version": CAMERA_ROOT_SCHEMA_VERSION,
        "camera_contract_schema_version": CONTRACT_SCHEMA_VERSION,
        "camera_setup_sha256": bindings.camera_setup_sha256,
        "camera_manifest": camera.manifest(),
        "T_world_from_rig": T_world_from_rig,
        "T_rig_from_cam": T_rig_from_cam,
        "T_cam_from_rig": T_cam_from_rig,
        "T_world_from_cam": T_world_from_rig @ T_rig_from_cam,
        "K": camera.K,
        "extrinsic_semantics": {
            "T_rig_from_cam": "camera-local coordinates to semantic body-FRD rig coordinates",
            "T_cam_from_rig": "semantic body-FRD rig coordinates to Blender camera-local coordinates",
        },
        "history_hold_ticks": _HISTORY_TICKS,
        "history_requested_command": np.zeros((len(_HISTORY_TICKS), 4), dtype=np.dtype("<f8")),
        "history_record_valid": np.ones(len(_HISTORY_TICKS), dtype=np.bool_),
    }


def _continuation_result_payload(result: ContinuationAuditResult) -> dict[str, object]:
    return {
        "audit_version": result.audit_version,
        "continuation_law_id": result.continuation_law_id,
        "continuation_target_sha256": result.continuation_target_sha256,
        "terminal_state_key": result.terminal_key.payload(),
        "nonsteady": result.nonsteady,
        "history_visible_count": result.history_visible_count,
        "satisfied_cues": result.satisfied_cues,
    }


def build_canonical_root_record(
    fixture: CanonicalSkierFixture,
    rig_manifest: RigManifest,
    bindings: RootRecordBindings,
) -> CanonicalRootRecord:
    """Recompute one complete immutable eight-root CS3 identity from PURE inputs."""
    if not isinstance(fixture, CanonicalSkierFixture):
        raise TypeError("fixture: expected CanonicalSkierFixture")
    if not isinstance(rig_manifest, RigManifest):
        raise TypeError("rig_manifest: expected RigManifest")
    if not isinstance(bindings, RootRecordBindings):
        raise TypeError("bindings: expected RootRecordBindings")
    if rig_manifest.canonical_sha256() != bindings.rig_manifest_canonical_sha256:
        raise ValueError("rig_manifest: canonical hash does not match bindings")
    if rig_manifest.derived_rig_sha256 != bindings.derived_rig_sha256:
        raise ValueError("rig_manifest: derived-rig hash does not match bindings")
    records = fixture.records()
    expected_ticks = _HISTORY_TICKS + _FUTURE_TICKS
    if tuple(record.state.absolute_tick for record in records) != expected_ticks:
        raise ValueError("fixture.records: expected exact absolute ticks -2..8")
    pose_rows = bindings.fixture_pose_rows(fixture.fixture_id)
    table = tuple(
        _canonical_table_row(
            fixture.fixture_id,
            record_index,
            source_record,
            authoritative_row,
            bindings.rig_manifest_canonical_sha256,
        )
        for record_index, (source_record, authoritative_row) in enumerate(
            zip(records, pose_rows, strict=True)
        )
    )
    future_laws = tuple(fixture.schedule.evaluate(tick).continuation_law_id for tick in _FUTURE_TICKS)
    if len(set(future_laws)) != 1:
        raise ValueError("fixture: future ticks do not have exactly one continuation law")
    continuation_audit = audit_forecast_continuation(records, fixture.schedule, fixture.history_visible)
    catalog = canonical_skier_fixtures()
    catalog_audits = tuple(
        audit_forecast_continuation(candidate.records(), candidate.schedule, candidate.history_visible)
        for candidate in catalog
    )
    audit_terminal_key_collisions(catalog_audits)
    catalog_audit_rows = tuple(
        {
            "fixture_id": candidate.fixture_id,
            **_continuation_result_payload(result),
        }
        for candidate, result in zip(catalog, catalog_audits, strict=True)
    )
    programs = canonical_branch_programs()
    branch_rows = tuple(
        {
            "branch_id": program.branch_id,
            "requested_command": program.requested_command,
            "dt_seconds": program.dt_seconds,
            "record_valid": program.record_valid,
            "capture_absolute_ticks": np.asarray(_FUTURE_TICKS, dtype=np.dtype("<i8")),
            "capture_timestamp_seconds": np.asarray(_FUTURE_TICKS, dtype=np.dtype("<f8")) * FIXED_DT_SECONDS,
            "camera_contract_sha256": camera_contract_sha256(program),
        }
        for program in programs
    )
    record_payload: dict[str, object] = {
        "schema_version": ROOT_RECORD_SCHEMA_VERSION,
        "versions": {
            "fixture_schema_version": FIXTURE_SCHEMA_VERSION,
            "camera_contract_schema_version": CONTRACT_SCHEMA_VERSION,
            "camera_root_schema_version": CAMERA_ROOT_SCHEMA_VERSION,
            "skier_root_schema_version": SKIER_ROOT_SCHEMA_VERSION,
            "skier_pose_root_schema_version": SKIER_POSE_ROOT_SCHEMA_VERSION,
            "integrator_version": INTEGRATOR_VERSION,
            "schedule_schema_version": SCHEDULE_SCHEMA_VERSION,
            "ski_construction_version": SKI_CONSTRUCTION_VERSION,
            "animation_parameter_version": ANIMATION_PARAMETER_VERSION,
            "pose_schema_version": POSE_SCHEMA_VERSION,
            "pose_root_schema_version": POSE_ROOT_SCHEMA_VERSION,
            "ik_schema_version": IK_SCHEMA_VERSION,
            "renderer_schema_version": RENDERER_SCHEMA_VERSION,
            "scene_schema_version": SCENE_SCHEMA_VERSION,
            "obstacle_schema_version": OBSTACLE_SCHEMA_VERSION,
        },
        "provenance": bindings.payload(),
        "renderer": dict(bindings.renderer),
        "identity_basis": {
            "fixture_id": fixture.fixture_id,
            "split": DatasetSplit.TEST,
            "branch_grouping": "all-nine-branches-one-indivisible-root",
        },
        "time": {
            "absolute_time_origin_tick": 0,
            "absolute_time_origin_seconds": 0.0,
            "dt_seconds": FIXED_DT_SECONDS,
            "history_ticks": np.asarray(_HISTORY_TICKS, dtype=np.dtype("<i8")),
            "future_ticks": np.asarray(_FUTURE_TICKS, dtype=np.dtype("<i8")),
            "history_timestamp_seconds": np.asarray(_HISTORY_TICKS, dtype=np.dtype("<f8")) * FIXED_DT_SECONDS,
            "future_timestamp_seconds": np.asarray(_FUTURE_TICKS, dtype=np.dtype("<f8")) * FIXED_DT_SECONDS,
            "future_capture_semantics": "integrate action row k once, then capture absolute tick k",
        },
        "units": {
            "position": "metre",
            "linear_velocity": "metre_per_second",
            "linear_acceleration": "metre_per_second_squared",
            "angle": "radian",
            "angular_velocity": "radian_per_second",
            "curvature": "inverse_metre",
            "time": "second",
            "camera_pixel": "pixel",
            "frames": {
                "world": "slope-fixed right-handed",
                "rig": "body-FRD +X forward,+Y right,+Z down",
                "camera": "Blender +X right,+Y up,-Z optical-forward",
            },
        },
        "numeric_dtypes": {
            "simulator_floating": "<f8",
            "absolute_tick": "<i8",
            "randomness_seed": "<i8",
            "record_valid": "|b1",
            "canonical_byte_order": "little-endian",
            "canonical_mapping_keys": "NFC UTF-8 sorted",
        },
        "episode_static": {
            **fixture.root_payload(),
            "seed": fixture.initial_state.randomness_seed,
        },
        "continuation": {
            **_continuation_result_payload(continuation_audit),
            "terminal_key_version": TERMINAL_KEY_VERSION,
            "forecast_continuation_law_id": future_laws[0],
            "future_continuation_law_ids": future_laws,
            "history_continuation_law_ids": tuple(
                fixture.schedule.evaluate(tick).continuation_law_id for tick in _HISTORY_TICKS
            ),
            "sorted_maneuver_schedule": fixture.schedule.payload(),
            "history_visible_cue": fixture.history_visible,
            "future_boundary_policy": "no maneuver/target/ramp/trigger boundary may begin at ticks 1..8",
            "catalog_collision_audit": {
                "audit_version": CONTINUATION_AUDIT_VERSION,
                "fixture_count": len(catalog_audit_rows),
                "collision_free": True,
                "results_sha256": sha256_canonical(
                    {
                        "audit_version": CONTINUATION_AUDIT_VERSION,
                        "terminal_key_version": TERMINAL_KEY_VERSION,
                        "results": catalog_audit_rows,
                    }
                ),
                "results": catalog_audit_rows,
            },
        },
        "initial_camera": _camera_root_payload(records[2], bindings),
        "obstacle": {
            "schema_version": OBSTACLE_SCHEMA_VERSION,
            "enabled": fixture.fixture_id == "occlusion_path",
            "policy": "enabled only for occlusion_path; fixed across all ticks and camera siblings",
            **dict(bindings.occluder),
        },
        "history": tuple(table[index] for index in range(len(_HISTORY_TICKS))),
        "canonical_skier_pose_table": {
            "absolute_ticks": np.asarray(expected_ticks, dtype=np.dtype("<i8")),
            "row_count": len(table),
            "rows": table,
        },
        "branch_programs": {
            "branch_count": len(branch_rows),
            "ordered_branch_ids": tuple(BranchId),
            "rows": branch_rows,
        },
    }
    return CanonicalRootRecord(record_payload)


@dataclass(frozen=True, eq=False)
class CanonicalRootRecord:
    """Immutable root envelope whose IDs hash only the complete record payload."""

    _record: Mapping[str, object] = field(repr=False)
    root_id: str = field(init=False)
    split_group_id: str = field(init=False)
    split: DatasetSplit = field(init=False, default=DatasetSplit.TEST)

    def __post_init__(self) -> None:
        record = _require_mapping("record", self._record)
        _require_exact_keys("record", record, _ROOT_RECORD_KEYS)
        if record["schema_version"] != ROOT_RECORD_SCHEMA_VERSION:
            raise ValueError("record.schema_version: unsupported canonical root record")
        frozen = _deep_freeze(record)
        assert isinstance(frozen, Mapping)
        root_id = sha256_canonical(frozen)
        object.__setattr__(self, "_record", frozen)
        object.__setattr__(self, "root_id", root_id)
        object.__setattr__(self, "split_group_id", root_id)

    @property
    def record(self) -> Mapping[str, object]:
        """Return the recursively immutable record mapping used by the ID hash."""
        return self._record

    def sibling_identities(self) -> tuple[RootSiblingIdentity, ...]:
        """Return and validate the indivisible nine-branch test sibling group."""
        identities = tuple(
            RootSiblingIdentity(self.root_id, self.split_group_id, branch_id, self.split)
            for branch_id in BranchId
        )
        validate_sibling_group(identities)
        return identities

    def payload(self) -> Mapping[str, object]:
        """Return the immutable exact envelope written by the bridge."""
        payload = {
            "schema_version": ROOT_ENVELOPE_SCHEMA_VERSION,
            "root_id": self.root_id,
            "split_group_id": self.split_group_id,
            "split": self.split,
            "record": self._record,
        }
        frozen = _deep_freeze(payload)
        assert isinstance(frozen, Mapping)
        return frozen

    def record_canonical_bytes(self) -> bytes:
        return canonical_bytes(self._record)

    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.payload())

    def canonical_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> CanonicalRootRecord:
        """Fail closed on an ordinary decoded envelope and recompute both IDs."""
        envelope = _require_mapping("root envelope", value)
        _require_exact_keys("root envelope", envelope, _ROOT_ENVELOPE_KEYS)
        if envelope["schema_version"] != ROOT_ENVELOPE_SCHEMA_VERSION:
            raise ValueError("root envelope.schema_version: unsupported version")
        if envelope["split"] not in (DatasetSplit.TEST, DatasetSplit.TEST.value):
            raise ValueError("root envelope.split: canonical CS3 roots are test-only")
        record = cls(_require_mapping("root envelope.record", envelope["record"]))
        if envelope["root_id"] != record.root_id:
            raise ValueError("root envelope.root_id: canonical payload hash mismatch")
        if envelope["split_group_id"] != record.root_id:
            raise ValueError("root envelope.split_group_id: must equal recomputed root_id")
        return record


def validate_canonical_root_bytes(data: bytes, expected: CanonicalRootRecord) -> None:
    """Validate exact canonical bytes against an independently rebuilt root."""
    if not isinstance(data, bytes):
        raise TypeError("data: expected bytes")
    if not isinstance(expected, CanonicalRootRecord):
        raise TypeError("expected: expected CanonicalRootRecord")
    if data != expected.canonical_bytes():
        raise ValueError("canonical root bytes: independently rebuilt bytes mismatch")
    decoded = json.loads(data.decode("utf-8"))
    if not isinstance(decoded, dict) or set(decoded) != _ROOT_ENVELOPE_KEYS:
        raise ValueError("canonical root bytes: malformed envelope")


__all__ = [
    "CAMERA_NORMAL_OFFSET_M",
    "CAMERA_ROOT_SCHEMA_VERSION",
    "CAMERA_UPHILL_OFFSET_M",
    "CanonicalRootRecord",
    "EXCLUDED_EQUIPMENT_NAMES",
    "ID_PASS_SCHEMA_VERSION",
    "LABEL_TARGET_SET_VERSION",
    "NEAR_CLIP_SCHEMA_VERSION",
    "OBSTACLE_SCHEMA_VERSION",
    "PNG_SCHEMA_VERSION",
    "RENDERER_SCHEMA_VERSION",
    "ROOT_ENVELOPE_SCHEMA_VERSION",
    "ROOT_RECORD_SCHEMA_VERSION",
    "RootRecordBindings",
    "SCENE_SCHEMA_VERSION",
    "SLOPE_LENGTH_M",
    "SLOPE_WIDTH_M",
    "TARGET_OBJECT_NAMES",
    "build_canonical_root_record",
    "canonical_renderer_contract",
    "decode_canonical_mapping_bytes",
    "initial_camera_rig_transform",
    "validate_canonical_root_bytes",
]
