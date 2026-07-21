#!/usr/bin/env python3
"""Fail-closed same-host replay audit for the B3-CS3 eight-root proof.

The renderer writes one output-independent relative path table per fresh run.
This verifier does not trust renderer-declared artifact hashes: it resolves every
path beneath its replay directory, recomputes SHA-256, validates PNG containers,
and byte-compares corresponding files from the two processes.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import struct
import sys
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final
from xml.sax.saxutils import escape

import numpy as np

from vllatent.sim.contracts import FIXED_DT_SECONDS, canonical_bytes, default_camera_contract
from vllatent.sim.frames import rotation_geodesic_angle
from vllatent.sim.labels import FrameLabels, RasterRuns
from vllatent.sim.pose import (
    PINNED_CANONICAL_POSE_TABLE_SHA256,
    PINNED_POSE_TABLE_EXPORT_SHA256,
    canonical_pose_export_json_bytes,
    canonical_pose_export_payload,
    evaluate_pose,
    record_with_pose,
)
from vllatent.sim.rig import load_asset_manifest, load_rig_manifest
from vllatent.sim.scene import (
    CAMERA_ROOT_SCHEMA_VERSION,
    ROOT_ENVELOPE_SCHEMA_VERSION,
    ROOT_RECORD_SCHEMA_VERSION,
    RootRecordBindings,
    build_canonical_root_record,
    canonical_renderer_contract,
    initial_camera_rig_transform,
)
from vllatent.sim.skier_fixtures import FIXTURE_IDS, canonical_skier_fixtures

REPLAY_SCHEMA_VERSION: Final[str] = "b3-cs3-eight-root-render-replay-v2"
FRAME_SCHEMA_VERSION: Final[str] = "b3-cs3-rendered-frame-metadata-v2"
AUDIT_SCHEMA_VERSION: Final[str] = "b3-cs3-two-process-replay-audit-v2"
FIXED_OCCLUDER_ID: Final[str] = "fixed_occluder_101"
EXPECTED_TICKS: Final[tuple[int, ...]] = tuple(range(-2, 9))
VISUAL_STRIP_TICKS: Final[tuple[int, ...]] = (0, 4, 8)

_TOP_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "blender_version",
        "blender_build_hash",
        "scene_sha256",
        "scene_manifest_file_sha256",
        "asset_manifest_file_sha256",
        "rig_manifest_file_sha256",
        "rig_manifest_canonical_sha256",
        "authoritative_pose_table_file_sha256",
        "canonical_pose_table_sha256",
        "renderer",
        "root_record_schema_version",
        "root_envelope_schema_version",
        "root_count",
        "roots",
        "fixture_count",
        "frame_count",
        "rows",
    }
)
_ROW_KEYS: Final[frozenset[str]] = frozenset(
    {
        "fixture_id",
        "root_id",
        "split_group_id",
        "split",
        "root_record_sha256",
        "record_index",
        "absolute_tick",
        "branch_id",
        "requested_command",
        "requested_command_valid",
        "record_valid",
        "dt_seconds",
        "frame_metadata_path",
        "frame_metadata_sha256",
        "state_record_sha256",
        "pose_record_sha256",
        "source_skier_digest",
        "posed_skier_digest",
        "pose_sha256",
        "camera_transform_sha256",
        "rgb_sha256",
        "mask_full_sha256",
        "mask_in_sha256",
        "mask_visible_sha256",
        "target_only_id_sha256",
        "target_id_sha256",
        "labels_sha256",
        "target_geometry_sha256",
        "projection_parity_max_px",
        "occlusion_flag",
        "modal_front_object_id",
        "p_visible_target",
        "fixed_occluder_observation_changed",
    }
)
_ROW_HASH_FIELDS: Final[tuple[str, ...]] = (
    "frame_metadata_sha256",
    "state_record_sha256",
    "pose_record_sha256",
    "source_skier_digest",
    "posed_skier_digest",
    "pose_sha256",
    "camera_transform_sha256",
    "rgb_sha256",
    "mask_full_sha256",
    "mask_in_sha256",
    "mask_visible_sha256",
    "target_only_id_sha256",
    "target_id_sha256",
    "labels_sha256",
    "target_geometry_sha256",
)
_TOP_HASH_FIELDS: Final[tuple[str, ...]] = (
    "scene_sha256",
    "scene_manifest_file_sha256",
    "asset_manifest_file_sha256",
    "rig_manifest_file_sha256",
    "rig_manifest_canonical_sha256",
    "authoritative_pose_table_file_sha256",
    "canonical_pose_table_sha256",
)
_METADATA_MATCH_FIELDS: Final[tuple[str, ...]] = (
    "fixture_id",
    "root_id",
    "split_group_id",
    "split",
    "root_record_sha256",
    "record_index",
    "absolute_tick",
    "branch_id",
    "requested_command",
    "requested_command_valid",
    "record_valid",
    "dt_seconds",
    "state_record_sha256",
    "pose_record_sha256",
    "source_skier_digest",
    "posed_skier_digest",
    "pose_sha256",
    "camera_transform_sha256",
    "rgb_sha256",
    "mask_full_sha256",
    "mask_in_sha256",
    "mask_visible_sha256",
    "target_only_id_sha256",
    "target_id_sha256",
    "labels_sha256",
    "target_geometry_sha256",
    "projection_parity_max_px",
    "occlusion_flag",
    "modal_front_object_id",
    "p_visible_target",
    "fixed_occluder_observation_changed",
)
_FILE_ARTIFACTS: Final[tuple[tuple[str, str, str], ...]] = (
    ("state_record", "state_record_path", "state_record_sha256"),
    ("pose_record", "pose_record_path", "pose_record_sha256"),
    ("rgb", "rgb_path", "rgb_sha256"),
    ("mask_full", "mask_full_path", "mask_full_sha256"),
    ("mask_in", "mask_in_path", "mask_in_sha256"),
    ("mask_visible", "mask_visible_path", "mask_visible_sha256"),
    ("target_only_id", "target_only_id_path", "target_only_id_sha256"),
    ("target_id", "target_id_path", "target_id_sha256"),
)
_PNG_KINDS: Final[frozenset[str]] = frozenset(
    {"rgb", "mask_full", "mask_in", "mask_visible", "target_only_id", "target_id"}
)
_ROOT_ROW_KEYS: Final[frozenset[str]] = frozenset(
    {
        "fixture_id",
        "root_id",
        "split_group_id",
        "split",
        "root_record_schema_version",
        "root_envelope_schema_version",
        "root_record_path",
        "root_record_sha256",
    }
)
_METADATA_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "fixture_id",
        "root_id",
        "split_group_id",
        "split",
        "root_record_sha256",
        "record_index",
        "absolute_tick",
        "timestamp_seconds",
        "branch_id",
        "requested_command",
        "requested_command_valid",
        "record_valid",
        "dt_seconds",
        "source_skier_digest",
        "posed_skier_digest",
        "pose_sha256",
        "camera_transform_sha256",
        "camera",
        "target_geometry_sha256",
        "labels_sha256",
        "labels",
        "fixed_occluder_enabled",
        "fixed_occluder_observation_changed",
        "occlusion_flag",
        "modal_front_object_id",
        "p_visible_target",
        "without_fixed_occluder_labels_sha256",
        "projection_parity_max_px",
        "id_pass_schema_version",
        "local_pose_reconstruction",
        "pose_parity_maxima",
        "equipment_metrics",
        "mask_full_origin_uv_px",
        "state_record_path",
        "state_record_sha256",
        "pose_record_path",
        "pose_record_sha256",
        "rgb_path",
        "rgb_sha256",
        "mask_full_path",
        "mask_full_sha256",
        "mask_in_path",
        "mask_in_sha256",
        "mask_visible_path",
        "mask_visible_sha256",
        "target_only_id_path",
        "target_only_id_sha256",
        "target_id_path",
        "target_id_sha256",
    }
)
_CAMERA_KEYS: Final[frozenset[str]] = frozenset(
    {
        "camera_root_schema_version",
        "T_world_from_rig_requested",
        "T_world_from_rig_achieved",
        "T_world_from_cam_requested",
        "T_world_from_cam_achieved",
        "T_cam_from_world_achieved",
        "T_rig0_from_rig_t_requested",
        "T_rig0_from_rig_t_achieved",
        "T_rig_from_cam_requested",
        "T_cam_from_rig_requested",
        "T_rig_from_cam_achieved",
        "T_cam_from_rig_achieved",
        "K_requested",
        "K_achieved_blender",
        "achieved_intrinsic_residual_px",
        "achieved_camera_position_residual_m",
        "achieved_camera_rotation_residual_rad",
        "achieved_rig_action_position_residual_m",
        "achieved_rig_action_rotation_residual_rad",
    }
)
_LABEL_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "A_full",
        "A_in",
        "A_vis",
        "frame_fraction",
        "visible_fraction",
        "occlusion_fraction",
        "amodal_bbox_px",
        "cx",
        "cy",
        "log_h",
        "visible_bbox_px",
        "visible_cx",
        "visible_cy",
        "visible_log_h",
        "whole_target_positive_depth",
        "amodal_regression_valid",
        "in_frame",
        "p_visible_target",
        "occlusion_flag",
        "modal_front_object_id",
    }
)
_LOCAL_POSE_KEYS: Final[frozenset[str]] = frozenset({"position_m", "rotation_frobenius"})
_POSE_MAXIMA_KEYS: Final[frozenset[str]] = frozenset(
    {
        "object_position_m",
        "object_rotation_rad",
        "root_position_m",
        "root_rotation_rad",
        "world_position_m",
        "world_rotation_rad",
    }
)
_EQUIPMENT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "max_equipment_origin_residual_m",
        "max_ski_frame_residual_rad",
        "max_contact_origin_residual_m",
        "inner_tip_gap_residual_m",
        "inner_tip_gap_m",
        "max_boot_binding_position_m",
        "max_boot_binding_rotation_rad",
        "max_binding_relative_position_residual_m",
        "max_binding_relative_rotation_residual_rad",
        "max_attack_residual_rad",
        "max_edge_residual_rad",
        "max_slip_component_residual_m_s",
    }
)


class ReplayAuditError(ValueError):
    """One replay violates the frozen CS3 comparison contract."""


@dataclass(frozen=True)
class _AuditAuthority:
    asset_manifest_file_sha256: str
    rig_manifest_file_sha256: str
    scene_manifest_file_sha256: str
    pose_table_file_sha256: str
    asset_manifest: Mapping[str, object]
    rig_manifest: Any
    scene_manifest: Mapping[str, object]
    pose_table: Mapping[str, object]
    fixtures: tuple[Any, ...]
    root_records: Mapping[str, Any]


def _reject_constant(value: str) -> object:
    raise ReplayAuditError(f"non-finite JSON number is forbidden: {value}")


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ReplayAuditError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReplayAuditError(f"{label}: cannot load JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ReplayAuditError(f"{label}: expected a JSON object")
    return value


def _require_exact_keys(value: Mapping[str, object], expected: frozenset[str], *, label: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ReplayAuditError(f"{label}: key mismatch; missing={missing}, extra={extra}")


def _require_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReplayAuditError(f"{label}: expected integer")
    return value


def _require_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReplayAuditError(f"{label}: expected non-empty string")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    text = _require_text(value, label=label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ReplayAuditError(f"{label}: expected lowercase SHA-256 hex")
    return text


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _build_authority(
    *,
    asset_manifest_path: Path,
    rig_manifest_path: Path,
    scene_manifest_path: Path,
    pose_table_path: Path,
) -> _AuditAuthority:
    """Load tracked PURE authority and independently rebuild all eight roots."""
    try:
        asset_path = asset_manifest_path.resolve(strict=True)
        rig_path = rig_manifest_path.resolve(strict=True)
        scene_path = scene_manifest_path.resolve(strict=True)
        pose_path = pose_table_path.resolve(strict=True)
        asset = load_asset_manifest(asset_path)
        rig = load_rig_manifest(rig_path)
        scene = _load_json_object(scene_path, label="authoritative scene manifest")
        expected_pose_bytes = canonical_pose_export_json_bytes(rig)
        actual_pose_bytes = pose_path.read_bytes()
        if actual_pose_bytes != expected_pose_bytes:
            raise ReplayAuditError("authoritative pose table differs from independent PURE export")
        if _sha256_file(pose_path) != PINNED_POSE_TABLE_EXPORT_SHA256:
            raise ReplayAuditError("authoritative pose-table file SHA-256 differs from the pin")
        pose = _load_json_object(pose_path, label="authoritative pose table")
        if pose.get("canonical_pose_table_sha256") != PINNED_CANONICAL_POSE_TABLE_SHA256:
            raise ReplayAuditError("authoritative canonical pose-table SHA-256 differs from the pin")
        if pose != canonical_pose_export_payload(rig):
            raise ReplayAuditError("authoritative pose-table mapping differs from PURE reconstruction")
        bindings = RootRecordBindings.from_manifests(
            asset_manifest=asset,
            asset_manifest_file_sha256=_sha256_file(asset_path),
            rig_manifest=rig,
            rig_manifest_file_sha256=_sha256_file(rig_path),
            scene_manifest=scene,
            scene_manifest_file_sha256=_sha256_file(scene_path),
            pose_export_manifest=pose,
            pose_export_file_sha256=_sha256_file(pose_path),
        )
        fixtures = canonical_skier_fixtures()
        roots = {
            fixture.fixture_id: build_canonical_root_record(fixture, rig, bindings)
            for fixture in fixtures
        }
    except ReplayAuditError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise ReplayAuditError(f"authoritative CS3 dependency audit failed: {error}") from error
    return _AuditAuthority(
        asset_manifest_file_sha256=_sha256_file(asset_path),
        rig_manifest_file_sha256=_sha256_file(rig_path),
        scene_manifest_file_sha256=_sha256_file(scene_path),
        pose_table_file_sha256=_sha256_file(pose_path),
        asset_manifest=asset,
        rig_manifest=rig,
        scene_manifest=scene,
        pose_table=pose,
        fixtures=fixtures,
        root_records=roots,
    )


def _same_file_bytes(first: Path, second: Path) -> bool:
    if first.stat().st_size != second.stat().st_size:
        return False
    with first.open("rb") as first_handle, second.open("rb") as second_handle:
        while True:
            first_chunk = first_handle.read(1024 * 1024)
            second_chunk = second_handle.read(1024 * 1024)
            if first_chunk != second_chunk:
                return False
            if not first_chunk:
                return True


def _relative_artifact(root: Path, value: object, *, label: str) -> tuple[str, Path]:
    relative = _require_text(value, label=label)
    if "\\" in relative:
        raise ReplayAuditError(f"{label}: expected normalized POSIX relative path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise ReplayAuditError(f"{label}: path must stay beneath the replay directory")
    if pure.as_posix() != relative:
        raise ReplayAuditError(f"{label}: path is not normalized")
    replay_root = root.resolve(strict=True)
    try:
        resolved = (replay_root / Path(*pure.parts)).resolve(strict=True)
    except OSError as error:
        raise ReplayAuditError(f"{label}: missing artifact {relative}: {error}") from error
    if not resolved.is_relative_to(replay_root) or not resolved.is_file():
        raise ReplayAuditError(f"{label}: artifact escapes replay directory or is not a file")
    return relative, resolved


def _validate_png(path: Path, *, kind: str) -> np.ndarray:
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ReplayAuditError(f"{kind}: not a PNG: {path}")
    offset = 8
    chunks: list[tuple[bytes, bytes]] = []
    while offset < len(data):
        if offset + 12 > len(data):
            raise ReplayAuditError(f"{kind}: truncated PNG chunk: {path}")
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        final = offset + 12 + length
        if final > len(data):
            raise ReplayAuditError(f"{kind}: truncated PNG payload: {path}")
        chunk_type = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack(">I", data[offset + 8 + length : final])[0]
        actual_crc = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise ReplayAuditError(f"{kind}: PNG CRC mismatch: {path}")
        chunks.append((chunk_type, payload))
        offset = final
        if chunk_type == b"IEND":
            break
    if offset != len(data) or not chunks or chunks[0][0] != b"IHDR" or chunks[-1][0] != b"IEND":
        raise ReplayAuditError(f"{kind}: malformed PNG chunk sequence: {path}")
    header = chunks[0][1]
    if len(header) != 13 or not any(chunk_type == b"IDAT" for chunk_type, _ in chunks):
        raise ReplayAuditError(f"{kind}: missing PNG IHDR/IDAT: {path}")
    width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(">IIBBBBB", header)
    if width <= 0 or height <= 0 or (bit_depth, compression, filtering, interlace) != (8, 0, 0, 0):
        raise ReplayAuditError(f"{kind}: unsupported deterministic PNG format: {path}")
    expected_color_type = 2 if kind == "rgb" else 0
    if color_type != expected_color_type:
        raise ReplayAuditError(f"{kind}: expected PNG color type {expected_color_type}: {path}")
    if kind != "mask_full" and (width, height) != (224, 224):
        raise ReplayAuditError(f"{kind}: expected 224x224 PNG: {path}")
    channels = 3 if kind == "rgb" else 1
    compressed = b"".join(payload for chunk_type, payload in chunks if chunk_type == b"IDAT")
    try:
        raw = zlib.decompress(compressed)
    except zlib.error as error:
        raise ReplayAuditError(f"{kind}: invalid compressed PNG data: {path}") from error
    row_bytes = width * channels
    if len(raw) != height * (row_bytes + 1):
        raise ReplayAuditError(f"{kind}: deterministic PNG scanline size mismatch: {path}")
    decoded = np.empty((height, row_bytes), dtype=np.uint8)
    for row_index in range(height):
        offset = row_index * (row_bytes + 1)
        if raw[offset] != 0:
            raise ReplayAuditError(f"{kind}: expected fixed PNG filter 0: {path}")
        decoded[row_index] = np.frombuffer(raw[offset + 1 : offset + 1 + row_bytes], dtype=np.uint8)
    shape = (height, width, channels) if channels == 3 else (height, width)
    return decoded.reshape(shape)


def _matrix(value: object, *, shape: tuple[int, ...], label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ReplayAuditError(f"{label}: expected finite numeric shape {shape}") from error
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ReplayAuditError(f"{label}: expected finite numeric shape {shape}")
    return array


def _validate_camera_metadata(
    metadata: Mapping[str, object],
    *,
    fixture: Any,
    row_number: int,
) -> None:
    value = metadata["camera"]
    if not isinstance(value, dict):
        raise ReplayAuditError(f"rows[{row_number}].camera: expected object")
    _require_exact_keys(value, _CAMERA_KEYS, label=f"rows[{row_number}].camera")
    if value["camera_root_schema_version"] != CAMERA_ROOT_SCHEMA_VERSION:
        raise ReplayAuditError(f"rows[{row_number}].camera: schema drift")
    transforms = {
        key: _matrix(value[key], shape=(4, 4), label=f"rows[{row_number}].camera.{key}")
        for key in _CAMERA_KEYS
        if key.startswith("T_")
    }
    contract = default_camera_contract()
    expected_world_rig = initial_camera_rig_transform(fixture.records()[2])
    expected_rig_cam = np.eye(4, dtype=np.float64)
    expected_rig_cam[:3, :3] = contract.R_rig_from_cam
    expected_cam_rig = np.eye(4, dtype=np.float64)
    expected_cam_rig[:3, :3] = contract.R_cam_from_rig
    expected_world_cam = expected_world_rig @ expected_rig_cam
    requested = {
        "T_world_from_rig_requested": expected_world_rig,
        "T_world_from_cam_requested": expected_world_cam,
        "T_rig0_from_rig_t_requested": np.eye(4, dtype=np.float64),
        "T_rig_from_cam_requested": expected_rig_cam,
        "T_cam_from_rig_requested": expected_cam_rig,
    }
    for key, expected in requested.items():
        if not np.allclose(transforms[key], expected, rtol=0.0, atol=1.0e-12):
            raise ReplayAuditError(f"rows[{row_number}].camera.{key}: requested transform drift")
    inverse_pairs = (
        ("T_world_from_cam_achieved", "T_cam_from_world_achieved"),
        ("T_rig_from_cam_achieved", "T_cam_from_rig_achieved"),
    )
    for forward, inverse in inverse_pairs:
        if not np.allclose(
            transforms[forward] @ transforms[inverse],
            np.eye(4),
            rtol=0.0,
            atol=1.0e-10,
        ):
            raise ReplayAuditError(f"rows[{row_number}].camera: achieved inverse pair drift")
    achieved_world_rig = transforms["T_world_from_rig_achieved"]
    achieved_world_cam = transforms["T_world_from_cam_achieved"]
    achieved_action = transforms["T_rig0_from_rig_t_achieved"]
    achieved_rig_cam = transforms["T_rig_from_cam_achieved"]
    if (
        np.linalg.norm(achieved_world_rig[:3, 3] - expected_world_rig[:3, 3]) > 1.0e-6
        or rotation_geodesic_angle(achieved_world_rig[:3, :3], expected_world_rig[:3, :3]) > 1.0e-6
        or np.linalg.norm(achieved_world_cam[:3, 3] - expected_world_cam[:3, 3]) > 1.0e-6
        or rotation_geodesic_angle(achieved_world_cam[:3, :3], expected_world_cam[:3, :3]) > 1.0e-6
        or np.linalg.norm(achieved_action[:3, 3]) > 1.0e-6
        or rotation_geodesic_angle(achieved_action[:3, :3], np.eye(3)) > 1.0e-6
        or not np.allclose(
            np.linalg.inv(achieved_world_rig) @ achieved_world_cam,
            achieved_rig_cam,
            rtol=0.0,
            atol=1.0e-10,
        )
    ):
        raise ReplayAuditError(f"rows[{row_number}].camera: achieved SE(3) contract drift")
    K_requested = _matrix(value["K_requested"], shape=(3, 3), label="K_requested")
    K_achieved = _matrix(value["K_achieved_blender"], shape=(3, 3), label="K_achieved_blender")
    if not np.array_equal(K_requested, contract.K) or np.max(np.abs(K_achieved - contract.K)) > 1.0e-4:
        raise ReplayAuditError(f"rows[{row_number}].camera: intrinsic contract drift")
    numeric_limits = {
        "achieved_intrinsic_residual_px": 1.0e-4,
        "achieved_camera_position_residual_m": 1.0e-6,
        "achieved_camera_rotation_residual_rad": 1.0e-6,
        "achieved_rig_action_position_residual_m": 1.0e-6,
        "achieved_rig_action_rotation_residual_rad": 1.0e-6,
    }
    for key, limit in numeric_limits.items():
        scalar = value[key]
        if isinstance(scalar, bool) or not isinstance(scalar, (int, float)) or not math.isfinite(float(scalar)):
            raise ReplayAuditError(f"rows[{row_number}].camera.{key}: expected finite scalar")
        if not 0.0 <= float(scalar) <= limit:
            raise ReplayAuditError(f"rows[{row_number}].camera.{key}: residual gate failed")
    camera_payload = {
        key: transforms[key] if key in transforms else K_requested if key == "K_requested" else K_achieved if key == "K_achieved_blender" else value[key]
        for key in _CAMERA_KEYS
    }
    expected_hash = hashlib.sha256(canonical_bytes(camera_payload)).hexdigest()
    if metadata["camera_transform_sha256"] != expected_hash:
        raise ReplayAuditError(f"rows[{row_number}].camera: canonical transform hash drift")


def _validate_nested_metadata_schema(metadata: Mapping[str, object], *, row_number: int) -> None:
    nested_sets = (
        ("labels", _LABEL_KEYS),
        ("local_pose_reconstruction", _LOCAL_POSE_KEYS),
        ("pose_parity_maxima", _POSE_MAXIMA_KEYS),
        ("equipment_metrics", _EQUIPMENT_KEYS),
    )
    for key, expected in nested_sets:
        value = metadata[key]
        if not isinstance(value, dict):
            raise ReplayAuditError(f"rows[{row_number}].{key}: expected object")
        _require_exact_keys(value, expected, label=f"rows[{row_number}].{key}")
    local = metadata["local_pose_reconstruction"]
    pose = metadata["pose_parity_maxima"]
    equipment = metadata["equipment_metrics"]
    assert isinstance(local, dict) and isinstance(pose, dict) and isinstance(equipment, dict)
    for key, value in (*local.items(), *pose.items(), *equipment.items()):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ReplayAuditError(f"rows[{row_number}]: non-finite metric {key}")
    if max(float(value) for value in local.values()) > 1.0e-10:
        raise ReplayAuditError(f"rows[{row_number}]: local pose reconstruction gate failed")
    if max(float(pose[key]) for key in _POSE_MAXIMA_KEYS if key.endswith("position_m")) > 1.0e-6:
        raise ReplayAuditError(f"rows[{row_number}]: Blender pose position parity failed")
    if max(float(pose[key]) for key in _POSE_MAXIMA_KEYS if key.endswith("rotation_rad")) > 1.0e-6:
        raise ReplayAuditError(f"rows[{row_number}]: Blender pose rotation parity failed")
    equipment_limits = {
        "max_equipment_origin_residual_m": 0.01,
        "max_contact_origin_residual_m": 0.01,
        "inner_tip_gap_residual_m": 0.01,
        "max_boot_binding_position_m": 0.01,
        "max_binding_relative_position_residual_m": 0.01,
        "max_slip_component_residual_m_s": 0.02,
        "max_ski_frame_residual_rad": math.radians(1.0),
        "max_binding_relative_rotation_residual_rad": math.radians(1.0),
        "max_attack_residual_rad": math.radians(1.0),
        "max_edge_residual_rad": math.radians(1.0),
        "max_boot_binding_rotation_rad": math.radians(2.0),
    }
    if any(float(equipment[key]) > limit for key, limit in equipment_limits.items()):
        raise ReplayAuditError(f"rows[{row_number}]: realized equipment gate failed")
    if float(equipment["inner_tip_gap_m"]) < 0.05:
        raise ReplayAuditError(f"rows[{row_number}]: realized inner-tip gap gate failed")
    origin = metadata["mask_full_origin_uv_px"]
    if (
        not isinstance(origin, list)
        or len(origin) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in origin)
    ):
        raise ReplayAuditError(f"rows[{row_number}].mask_full_origin_uv_px: expected two integers")


def _runs_from_mask(mask: np.ndarray, *, origin_u: int, origin_v: int) -> RasterRuns:
    rows: list[int] = []
    starts: list[int] = []
    stops: list[int] = []
    for local_v, row in enumerate(mask):
        padded = np.pad(row.astype(np.int8), (1, 1))
        transitions = np.diff(padded)
        for start, stop in zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1), strict=True):
            rows.append(origin_v + local_v)
            starts.append(origin_u + int(start))
            stops.append(origin_u + int(stop))
    if not rows:
        raise ReplayAuditError("mask_full: expected non-empty amodal mask")
    return RasterRuns(
        rows=np.asarray(rows, dtype=np.int64),
        u_start=np.asarray(starts, dtype=np.int64),
        u_stop=np.asarray(stops, dtype=np.int64),
    )


def _validate_label_artifacts(
    metadata: Mapping[str, object],
    frame_paths: Mapping[str, Path],
    *,
    row_number: int,
) -> None:
    mask_full = _validate_png(frame_paths["mask_full"], kind="mask_full") > 0
    mask_in = _validate_png(frame_paths["mask_in"], kind="mask_in") > 0
    mask_visible = _validate_png(frame_paths["mask_visible"], kind="mask_visible") > 0
    target_only = _validate_png(frame_paths["target_only_id"], kind="target_only_id") > 0
    target_id = _validate_png(frame_paths["target_id"], kind="target_id") > 0
    if not np.array_equal(mask_in, target_only) or not np.array_equal(mask_visible, target_id):
        raise ReplayAuditError(f"rows[{row_number}]: Blender ID passes disagree with PURE masks")
    labels = metadata["labels"]
    assert isinstance(labels, dict)
    origin = metadata["mask_full_origin_uv_px"]
    assert isinstance(origin, list)
    runs = _runs_from_mask(mask_full, origin_u=int(origin[0]), origin_v=int(origin[1]))
    visible_bbox_value = labels["visible_bbox_px"]
    visible_bbox = (
        None
        if visible_bbox_value is None
        else np.asarray(visible_bbox_value, dtype=np.int64)
    )
    try:
        reconstructed = FrameLabels(
            m_full=runs,
            m_in=np.asarray(mask_in, dtype=np.bool_),
            target_only_mask=np.asarray(target_only, dtype=np.bool_),
            m_vis=np.asarray(mask_visible, dtype=np.bool_),
            A_full=_require_int(labels["A_full"], label="labels.A_full"),
            A_in=_require_int(labels["A_in"], label="labels.A_in"),
            A_vis=_require_int(labels["A_vis"], label="labels.A_vis"),
            frame_fraction=float(labels["frame_fraction"]),
            visible_fraction=float(labels["visible_fraction"]),
            occlusion_fraction=float(labels["occlusion_fraction"]),
            amodal_bbox_px=np.asarray(labels["amodal_bbox_px"], dtype=np.int64),
            cx=float(labels["cx"]),
            cy=float(labels["cy"]),
            log_h=float(labels["log_h"]),
            visible_bbox_px=visible_bbox,
            visible_cx=None if labels["visible_cx"] is None else float(labels["visible_cx"]),
            visible_cy=None if labels["visible_cy"] is None else float(labels["visible_cy"]),
            visible_log_h=None if labels["visible_log_h"] is None else float(labels["visible_log_h"]),
            whole_target_positive_depth=labels["whole_target_positive_depth"],
            amodal_regression_valid=labels["amodal_regression_valid"],
            in_frame=labels["in_frame"],
            p_visible_target=_require_int(labels["p_visible_target"], label="labels.p_visible_target"),
            occlusion_flag=labels["occlusion_flag"],
            modal_front_object_id=labels["modal_front_object_id"],
            schema_version=_require_text(labels["schema_version"], label="labels.schema_version"),
        )
    except (TypeError, ValueError) as error:
        raise ReplayAuditError(f"rows[{row_number}]: mechanical label reconstruction failed: {error}") from error
    if reconstructed.canonical_sha256() != metadata["labels_sha256"]:
        raise ReplayAuditError(f"rows[{row_number}]: reconstructed canonical label hash drift")
    if (
        reconstructed.p_visible_target != metadata["p_visible_target"]
        or reconstructed.occlusion_flag is not metadata["occlusion_flag"]
        or reconstructed.modal_front_object_id != metadata["modal_front_object_id"]
    ):
        raise ReplayAuditError(f"rows[{row_number}]: top-level label summary drift")


def _validate_manifest(
    manifest_path: Path,
    authority: _AuditAuthority,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    dict[tuple[str, int], dict[str, Path]],
    dict[str, Path],
]:
    manifest_path = manifest_path.resolve(strict=True)
    manifest = _load_json_object(manifest_path, label="replay manifest")
    _require_exact_keys(manifest, _TOP_KEYS, label="replay manifest")
    if manifest["schema_version"] != REPLAY_SCHEMA_VERSION:
        raise ReplayAuditError("replay manifest: unsupported schema_version")
    if manifest["blender_version"] != authority.scene_manifest.get("blender_version"):
        raise ReplayAuditError("replay manifest: expected Blender 4.5.11 LTS")
    if manifest["blender_build_hash"] != authority.scene_manifest.get("blender_build_hash"):
        raise ReplayAuditError("replay manifest: Blender build hash drift")
    for field in _TOP_HASH_FIELDS:
        _require_sha256(manifest[field], label=field)
    expected_top_hashes = {
        "scene_sha256": authority.scene_manifest.get("scene_sha256"),
        "scene_manifest_file_sha256": authority.scene_manifest_file_sha256,
        "asset_manifest_file_sha256": authority.asset_manifest_file_sha256,
        "rig_manifest_file_sha256": authority.rig_manifest_file_sha256,
        "rig_manifest_canonical_sha256": authority.rig_manifest.canonical_sha256(),
        "authoritative_pose_table_file_sha256": authority.pose_table_file_sha256,
        "canonical_pose_table_sha256": PINNED_CANONICAL_POSE_TABLE_SHA256,
    }
    if any(manifest[field] != value for field, value in expected_top_hashes.items()):
        raise ReplayAuditError("replay manifest: authoritative dependency hash drift")
    renderer = manifest["renderer"]
    if not isinstance(renderer, dict):
        raise ReplayAuditError("renderer: expected exact JSON object")
    expected_renderer = canonical_renderer_contract()
    normalized_renderer = dict(renderer)
    if isinstance(normalized_renderer.get("resolution"), list):
        normalized_renderer["resolution"] = tuple(normalized_renderer["resolution"])
    if normalized_renderer != expected_renderer:
        raise ReplayAuditError("renderer: exact Cycles CPU contract drift")
    if (
        manifest["root_record_schema_version"] != ROOT_RECORD_SCHEMA_VERSION
        or manifest["root_envelope_schema_version"] != ROOT_ENVELOPE_SCHEMA_VERSION
    ):
        raise ReplayAuditError("replay manifest: root schema drift")
    if _require_int(manifest["root_count"], label="root_count") != len(FIXTURE_IDS):
        raise ReplayAuditError("root_count: expected exactly 8")
    if _require_int(manifest["fixture_count"], label="fixture_count") != len(FIXTURE_IDS):
        raise ReplayAuditError("fixture_count: expected exactly 8")
    expected_frame_count = len(FIXTURE_IDS) * len(EXPECTED_TICKS)
    if _require_int(manifest["frame_count"], label="frame_count") != expected_frame_count:
        raise ReplayAuditError("frame_count: expected exactly 88")
    rows_value = manifest["rows"]
    if not isinstance(rows_value, list) or len(rows_value) != expected_frame_count:
        raise ReplayAuditError("rows: expected exactly 88 ordered frame rows")

    root_values = manifest["roots"]
    if not isinstance(root_values, list) or len(root_values) != len(FIXTURE_IDS):
        raise ReplayAuditError("roots: expected exactly 8 ordered canonical root rows")
    rows: list[dict[str, object]] = []
    artifact_paths: dict[tuple[str, int], dict[str, Path]] = {}
    root_paths: dict[str, Path] = {}
    seen_relative_paths: set[str] = set()
    for root_index, (root_value, fixture_id) in enumerate(zip(root_values, FIXTURE_IDS, strict=True)):
        if not isinstance(root_value, dict):
            raise ReplayAuditError(f"roots[{root_index}]: expected JSON object")
        _require_exact_keys(root_value, _ROOT_ROW_KEYS, label=f"roots[{root_index}]")
        expected_root = authority.root_records[fixture_id]
        expected_row = {
            "fixture_id": fixture_id,
            "root_id": expected_root.root_id,
            "split_group_id": expected_root.split_group_id,
            "split": expected_root.split.value,
            "root_record_schema_version": ROOT_RECORD_SCHEMA_VERSION,
            "root_envelope_schema_version": ROOT_ENVELOPE_SCHEMA_VERSION,
            "root_record_path": f"roots/{fixture_id}.canonical",
            "root_record_sha256": expected_root.canonical_sha256(),
        }
        if root_value != expected_row:
            raise ReplayAuditError(f"roots[{root_index}]: independently rebuilt root identity drift")
        relative, root_path = _relative_artifact(
            manifest_path.parent,
            root_value["root_record_path"],
            label=f"roots[{root_index}].root_record_path",
        )
        if relative in seen_relative_paths:
            raise ReplayAuditError(f"duplicate artifact path: {relative}")
        seen_relative_paths.add(relative)
        if root_path.read_bytes() != expected_root.canonical_bytes():
            raise ReplayAuditError(f"roots[{root_index}]: canonical root bytes mismatch")
        if _sha256_file(root_path) != root_value["root_record_sha256"]:
            raise ReplayAuditError(f"roots[{root_index}]: canonical root SHA-256 mismatch")
        root_paths[fixture_id] = root_path
    root_by_fixture: dict[str, str] = {}
    camera_by_fixture: dict[str, str] = {}
    expected_order = [
        (fixture_id, record_index, tick)
        for fixture_id in FIXTURE_IDS
        for record_index, tick in enumerate(EXPECTED_TICKS)
    ]
    for row_number, (row_value, expected) in enumerate(zip(rows_value, expected_order, strict=True)):
        if not isinstance(row_value, dict):
            raise ReplayAuditError(f"rows[{row_number}]: expected JSON object")
        row = row_value
        _require_exact_keys(row, _ROW_KEYS, label=f"rows[{row_number}]")
        fixture_id, expected_record_index, expected_tick = expected
        actual_order = (
            row["fixture_id"],
            _require_int(row["record_index"], label=f"rows[{row_number}].record_index"),
            _require_int(row["absolute_tick"], label=f"rows[{row_number}].absolute_tick"),
        )
        if actual_order != expected:
            raise ReplayAuditError(f"rows[{row_number}]: expected ordered row {expected}, got {actual_order}")
        root_id = _require_text(row["root_id"], label=f"rows[{row_number}].root_id")
        expected_root = authority.root_records[fixture_id]
        if (
            root_id != expected_root.root_id
            or row["split_group_id"] != expected_root.split_group_id
            or row["split"] != expected_root.split.value
            or row["root_record_sha256"] != expected_root.canonical_sha256()
        ):
            raise ReplayAuditError(f"rows[{row_number}]: canonical root/split identity drift")
        if fixture_id in root_by_fixture and root_by_fixture[fixture_id] != root_id:
            raise ReplayAuditError(f"{fixture_id}: root_id changes within one fixture")
        root_by_fixture[fixture_id] = root_id
        if (
            row["branch_id"] != "zero"
            or row["requested_command"] != [0.0, 0.0, 0.0, 0.0]
            or row["requested_command_valid"] is not True
            or row["record_valid"] is not True
            or row["dt_seconds"] != FIXED_DT_SECONDS
        ):
            raise ReplayAuditError(f"rows[{row_number}]: zero action/validity/dt contract drift")
        fixture = authority.fixtures[FIXTURE_IDS.index(fixture_id)]
        source_record = fixture.records()[expected_record_index]
        pure_pose = evaluate_pose(source_record, authority.rig_manifest)
        posed_record = record_with_pose(source_record, pure_pose)
        expected_pose_rows = authority.pose_table["rows"]
        assert isinstance(expected_pose_rows, list)
        expected_pose_row = expected_pose_rows[row_number]
        expected_semantics = {
            "source_skier_digest": source_record.skier_digest(),
            "posed_skier_digest": posed_record.skier_digest(),
            "pose_sha256": pure_pose.canonical_sha256(),
        }
        if any(row[field] != value for field, value in expected_semantics.items()):
            raise ReplayAuditError(f"rows[{row_number}]: independently reconstructed skier/pose drift")
        for field in _ROW_HASH_FIELDS:
            _require_sha256(row[field], label=f"rows[{row_number}].{field}")
        camera_hash = str(row["camera_transform_sha256"])
        if fixture_id in camera_by_fixture and camera_by_fixture[fixture_id] != camera_hash:
            raise ReplayAuditError(f"{fixture_id}: camera moved during zero-branch replay")
        camera_by_fixture[fixture_id] = camera_hash
        projection = row["projection_parity_max_px"]
        if (
            isinstance(projection, bool)
            or not isinstance(projection, (int, float))
            or not math.isfinite(float(projection))
            or not 0.0 <= float(projection) <= 1.0
        ):
            raise ReplayAuditError(f"rows[{row_number}].projection_parity_max_px: expected [0,1]")
        if not isinstance(row["occlusion_flag"], bool):
            raise ReplayAuditError(f"rows[{row_number}].occlusion_flag: expected bool")
        modal_id = row["modal_front_object_id"]
        if modal_id is not None and (not isinstance(modal_id, str) or not modal_id):
            raise ReplayAuditError(f"rows[{row_number}].modal_front_object_id: invalid")
        if _require_int(row["p_visible_target"], label=f"rows[{row_number}].p_visible_target") not in (0, 1):
            raise ReplayAuditError(f"rows[{row_number}].p_visible_target: expected 0 or 1")
        if not isinstance(row["fixed_occluder_observation_changed"], bool):
            raise ReplayAuditError(f"rows[{row_number}].fixed_occluder_observation_changed: expected bool")

        metadata_relative, metadata_path = _relative_artifact(
            manifest_path.parent,
            row["frame_metadata_path"],
            label=f"rows[{row_number}].frame_metadata_path",
        )
        if metadata_relative in seen_relative_paths:
            raise ReplayAuditError(f"duplicate artifact path: {metadata_relative}")
        seen_relative_paths.add(metadata_relative)
        if _sha256_file(metadata_path) != row["frame_metadata_sha256"]:
            raise ReplayAuditError(f"rows[{row_number}]: frame metadata SHA-256 mismatch")
        metadata = _load_json_object(metadata_path, label=f"rows[{row_number}] frame metadata")
        _require_exact_keys(metadata, _METADATA_KEYS, label=f"rows[{row_number}] frame metadata")
        if metadata.get("schema_version") != FRAME_SCHEMA_VERSION:
            raise ReplayAuditError(f"rows[{row_number}]: frame metadata schema drift")
        for field in _METADATA_MATCH_FIELDS:
            if metadata.get(field) != row[field]:
                raise ReplayAuditError(f"rows[{row_number}]: metadata field {field} disagrees with row")
        if metadata.get("branch_id") != "zero":
            raise ReplayAuditError(f"rows[{row_number}]: CS3 replay must use only the zero branch")
        expected_fixed_occluder_enabled = fixture_id == "occlusion_path"
        if metadata.get("fixed_occluder_enabled") is not expected_fixed_occluder_enabled:
            raise ReplayAuditError(
                f"rows[{row_number}]: fixed occluder intervention must be enabled "
                "only for occlusion_path"
            )
        command = metadata.get("requested_command")
        if (
            not isinstance(command, list)
            or len(command) != 4
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) != 0.0
                for value in command
            )
        ):
            raise ReplayAuditError(f"rows[{row_number}]: requested command must be four exact zeros")
        if (
            metadata["requested_command_valid"] is not True
            or metadata["record_valid"] is not True
            or metadata["dt_seconds"] != FIXED_DT_SECONDS
            or metadata["timestamp_seconds"] != expected_tick * FIXED_DT_SECONDS
        ):
            raise ReplayAuditError(f"rows[{row_number}]: action validity/time contract drift")
        _validate_camera_metadata(
            metadata,
            fixture=fixture,
            row_number=row_number,
        )
        _validate_nested_metadata_schema(metadata, row_number=row_number)

        frame_paths = {"frame_metadata": metadata_path}
        for kind, path_field, sha_field in _FILE_ARTIFACTS:
            relative, artifact_path = _relative_artifact(
                manifest_path.parent,
                metadata.get(path_field),
                label=f"rows[{row_number}] metadata.{path_field}",
            )
            if relative in seen_relative_paths:
                raise ReplayAuditError(f"duplicate artifact path: {relative}")
            seen_relative_paths.add(relative)
            if metadata.get(sha_field) != row[sha_field]:
                raise ReplayAuditError(f"rows[{row_number}]: metadata {sha_field} disagrees with row")
            if _sha256_file(artifact_path) != row[sha_field]:
                raise ReplayAuditError(f"rows[{row_number}]: {kind} SHA-256 mismatch")
            if kind in _PNG_KINDS:
                _validate_png(artifact_path, kind=kind)
            frame_paths[kind] = artifact_path
        if frame_paths["state_record"].read_bytes() != posed_record.canonical_bytes():
            raise ReplayAuditError(f"rows[{row_number}]: canonical state bytes differ from PURE reconstruction")
        if frame_paths["pose_record"].read_bytes() != _canonical_json_bytes(expected_pose_row):
            raise ReplayAuditError(f"rows[{row_number}]: exported pose bytes differ from PURE reconstruction")
        _validate_label_artifacts(metadata, frame_paths, row_number=row_number)
        artifact_paths[(fixture_id, expected_record_index)] = frame_paths
        rows.append(row)

    if len(set(root_by_fixture.values())) != len(FIXTURE_IDS):
        raise ReplayAuditError("root_id: expected one distinct root per canonical fixture")
    _validate_occlusion(rows)
    return manifest, rows, artifact_paths, root_paths


def _validate_occlusion(rows: Sequence[Mapping[str, object]]) -> None:
    for fixture_id in FIXTURE_IDS:
        fixture_rows = [row for row in rows if row["fixture_id"] == fixture_id]
        changed = [row for row in fixture_rows if row["fixed_occluder_observation_changed"] is True]
        if fixture_id != "occlusion_path":
            if changed:
                raise ReplayAuditError(f"{fixture_id}: fixed occluder may alter observations only for occlusion_path")
            if any(row["modal_front_object_id"] == FIXED_OCCLUDER_ID for row in fixture_rows):
                raise ReplayAuditError(
                    f"{fixture_id}: disabled fixed occluder appears as a modal front object"
                )
            continue
        negative = [
            row
            for row in fixture_rows
            if row["fixed_occluder_observation_changed"] is True
            and row["occlusion_flag"] is True
            and row["p_visible_target"] == 0
            and row["modal_front_object_id"] == FIXED_OCCLUDER_ID
        ]
        if not negative:
            raise ReplayAuditError("occlusion_path: expected a fixed-obstacle negative frame")
        negative_ticks = [
            _require_int(row["absolute_tick"], label="occlusion_path.absolute_tick") for row in negative
        ]
        before = [
            row
            for row in fixture_rows
            if _require_int(row["absolute_tick"], label="occlusion_path.absolute_tick") < min(negative_ticks)
            and row["p_visible_target"] == 1
            and row["occlusion_flag"] is False
        ]
        after = [
            row
            for row in fixture_rows
            if _require_int(row["absolute_tick"], label="occlusion_path.absolute_tick") > max(negative_ticks)
            and row["p_visible_target"] == 1
            and row["occlusion_flag"] is False
        ]
        if not before or not after:
            raise ReplayAuditError("occlusion_path: temporary occlusion requires positive, unoccluded pre/post frames")
        for row in changed:
            if row["modal_front_object_id"] != FIXED_OCCLUDER_ID:
                raise ReplayAuditError("occlusion_path: changed observation must name the preregistered fixed occluder")


def _write_visual_strip(
    output: Path,
    rows: Sequence[Mapping[str, object]],
    artifacts: Mapping[tuple[str, int], Mapping[str, Path]],
) -> dict[str, object]:
    selected = [
        row
        for row in rows
        if _require_int(row["absolute_tick"], label="visual strip absolute_tick") in VISUAL_STRIP_TICKS
    ]
    cell_width = 224
    image_height = 224
    label_height = 22
    header_height = 32
    width = cell_width * len(VISUAL_STRIP_TICKS)
    height = header_height + len(FIXTURE_IDS) * (image_height + label_height)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'),
        '<rect width="100%" height="100%" fill="#111"/>',
        '<text x="8" y="22" fill="#fff" font-family="monospace" font-size="15">'
        "B3-CS3 deterministic contact strip: ticks 0, 4, 8</text>",
    ]
    selected_keys: list[dict[str, object]] = []
    for fixture_index, fixture_id in enumerate(FIXTURE_IDS):
        y = header_height + fixture_index * (image_height + label_height)
        for tick_index, tick in enumerate(VISUAL_STRIP_TICKS):
            row = next(item for item in selected if item["fixture_id"] == fixture_id and item["absolute_tick"] == tick)
            record_index = _require_int(row["record_index"], label="visual strip record_index")
            rgb = artifacts[(fixture_id, record_index)]["rgb"].read_bytes()
            encoded = base64.b64encode(rgb).decode("ascii")
            x = tick_index * cell_width
            lines.append(f'<image x="{x}" y="{y}" width="224" height="224" href="data:image/png;base64,{encoded}"/>')
            label = escape(f"{fixture_id} tick={tick} rgb={str(row['rgb_sha256'])[:12]}")
            lines.append(
                f'<text x="{x + 4}" y="{y + image_height + 16}" fill="#fff" '
                f'font-family="monospace" font-size="11">{label}</text>'
            )
            selected_keys.append(
                {
                    "fixture_id": fixture_id,
                    "absolute_tick": tick,
                    "rgb_sha256": row["rgb_sha256"],
                }
            )
    lines.append("</svg>")
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    return {
        "format": "self-contained-svg",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "selection": selected_keys,
    }


def audit_replays(
    first_manifest_path: Path,
    second_manifest_path: Path,
    *,
    asset_manifest_path: Path,
    rig_manifest_path: Path,
    scene_manifest_path: Path,
    pose_table_path: Path,
    visual_strip_path: Path | None = None,
) -> dict[str, object]:
    """Audit two fresh-process replay trees and return a deterministic report."""
    authority = _build_authority(
        asset_manifest_path=asset_manifest_path,
        rig_manifest_path=rig_manifest_path,
        scene_manifest_path=scene_manifest_path,
        pose_table_path=pose_table_path,
    )
    first_manifest, first_rows, first_artifacts, first_roots = _validate_manifest(
        first_manifest_path, authority
    )
    second_manifest, second_rows, second_artifacts, second_roots = _validate_manifest(
        second_manifest_path, authority
    )
    if first_manifest != second_manifest:
        raise ReplayAuditError("fresh replay manifests are not structurally identical")

    compared_files = 0
    for fixture_id in FIXTURE_IDS:
        if not _same_file_bytes(first_roots[fixture_id], second_roots[fixture_id]):
            raise ReplayAuditError(f"{fixture_id}: canonical root files are not byte-identical")
        compared_files += 1
    for fixture_id in FIXTURE_IDS:
        for record_index in range(len(EXPECTED_TICKS)):
            first_frame = first_artifacts[(fixture_id, record_index)]
            second_frame = second_artifacts[(fixture_id, record_index)]
            if tuple(first_frame) != tuple(second_frame):
                raise ReplayAuditError(f"{fixture_id}/{record_index}: artifact structure differs")
            for kind in first_frame:
                if not _same_file_bytes(first_frame[kind], second_frame[kind]):
                    raise ReplayAuditError(f"{fixture_id}/{record_index}: {kind} files are not byte-identical")
                compared_files += 1

    ordered_replay_hash = hashlib.sha256(
        json.dumps(
            [[row[field] for field in _ROW_HASH_FIELDS] for row in first_rows],
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    report: dict[str, object] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "PASS",
        "replay_schema_version": REPLAY_SCHEMA_VERSION,
        "manifest_structure_equal": True,
        "fixture_count": len(FIXTURE_IDS),
        "frame_count": len(first_rows),
        "compared_file_count": compared_files,
        "ordered_replay_sha256": ordered_replay_hash,
        "static_camera_sha256_by_fixture": {
            fixture_id: next(row["camera_transform_sha256"] for row in first_rows if row["fixture_id"] == fixture_id)
            for fixture_id in FIXTURE_IDS
        },
        "zero_branch_only": True,
        "temporary_fixed_occlusion": True,
        "root_records_recomputed": True,
        "semantic_state_pose_reconstruction": True,
        "mechanical_label_reconstruction": True,
        "exact_recursive_schema": True,
    }
    if visual_strip_path is not None:
        report["visual_contact_strip"] = _write_visual_strip(
            visual_strip_path,
            first_rows,
            first_artifacts,
        )
    return report


def _write_report(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", required=True, type=Path, help="first fresh-run replay manifest")
    parser.add_argument("--second", required=True, type=Path, help="second fresh-run replay manifest")
    parser.add_argument("--asset-manifest", required=True, type=Path)
    parser.add_argument("--rig-manifest", required=True, type=Path)
    parser.add_argument("--scene-manifest", required=True, type=Path)
    parser.add_argument("--pose-table", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path, help="output PASS report JSON")
    parser.add_argument(
        "--visual-strip",
        type=Path,
        help="optional deterministic self-contained SVG contact strip",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = audit_replays(
            args.first,
            args.second,
            asset_manifest_path=args.asset_manifest,
            rig_manifest_path=args.rig_manifest,
            scene_manifest_path=args.scene_manifest,
            pose_table_path=args.pose_table,
            visual_strip_path=args.visual_strip,
        )
    except (OSError, ReplayAuditError) as error:
        print(f"B3_CS3_REPLAY_AUDIT_FAIL {error}", file=sys.stderr)
        return 1
    _write_report(args.report, report)
    print("B3_CS3_REPLAY_AUDIT_OK " + json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
