from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from collections.abc import Callable, Mapping
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.audit_b3_cs3_replays import (
    AUDIT_SCHEMA_VERSION,
    FRAME_SCHEMA_VERSION,
    REPLAY_SCHEMA_VERSION,
    ReplayAuditError,
    audit_replays,
)
from vllatent.sim.contracts import FIXED_DT_SECONDS, canonical_bytes, default_camera_contract
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
from vllatent.sim.skier_fixtures import canonical_skier_fixtures

REPO_ROOT = Path(__file__).parents[1]
ASSET_MANIFEST = REPO_ROOT / "manifests" / "b3_cs3" / "asset.json"
RIG_MANIFEST = REPO_ROOT / "manifests" / "b3_cs3" / "rig.json"
SCENE_MANIFEST = REPO_ROOT / "manifests" / "b3_cs3" / "scene.json"


def _sha(value: bytes | str) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha(path.read_bytes())


def _json_bytes(value: object, *, compact: bool = False) -> bytes:
    separators = (",", ":") if compact else None
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=None if compact else 2,
            separators=separators,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    body = chunk_type + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def _png(pixels: np.ndarray) -> bytes:
    if pixels.dtype != np.uint8 or pixels.ndim not in (2, 3):
        raise TypeError("pixels must be uint8 HxW or HxWx3")
    height, width = pixels.shape[:2]
    channels = 1 if pixels.ndim == 2 else pixels.shape[2]
    if channels not in (1, 3):
        raise ValueError("only grayscale/RGB fixtures are supported")
    raw = b"".join(b"\x00" + np.ascontiguousarray(pixels[row]).tobytes() for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 0 if channels == 1 else 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(raw, level=9))
        + _png_chunk(b"IEND", b"")
    )


def _label(occluded: bool) -> FrameLabels:
    rows = np.arange(80, 120, dtype=np.int64)
    runs = RasterRuns(
        rows=rows,
        u_start=np.full(rows.shape, 100, dtype=np.int64),
        u_stop=np.full(rows.shape, 120, dtype=np.int64),
    )
    m_in = np.zeros((224, 224), dtype=np.bool_)
    m_in[80:120, 100:120] = True
    m_vis = np.zeros_like(m_in) if occluded else m_in.copy()
    bbox = np.array([100, 80, 120, 120], dtype=np.int64)
    cx = (100 + 120) / (2.0 * 224.0)
    cy = (80 + 120) / (2.0 * 224.0)
    log_h = math.log(40.0 / 224.0)
    return FrameLabels(
        m_full=runs,
        m_in=m_in,
        target_only_mask=m_in,
        m_vis=m_vis,
        A_full=800,
        A_in=800,
        A_vis=0 if occluded else 800,
        frame_fraction=1.0,
        visible_fraction=0.0 if occluded else 1.0,
        occlusion_fraction=1.0 if occluded else 0.0,
        amodal_bbox_px=bbox,
        cx=cx,
        cy=cy,
        log_h=log_h,
        visible_bbox_px=None if occluded else bbox,
        visible_cx=None if occluded else cx,
        visible_cy=None if occluded else cy,
        visible_log_h=None if occluded else log_h,
        whole_target_positive_depth=True,
        amodal_regression_valid=True,
        in_frame=True,
        p_visible_target=0 if occluded else 1,
        occlusion_flag=occluded,
        modal_front_object_id="fixed_occluder_101" if occluded else None,
    )


def _label_summary(labels: FrameLabels) -> dict[str, object]:
    return {
        "schema_version": labels.schema_version,
        "A_full": labels.A_full,
        "A_in": labels.A_in,
        "A_vis": labels.A_vis,
        "frame_fraction": labels.frame_fraction,
        "visible_fraction": labels.visible_fraction,
        "occlusion_fraction": labels.occlusion_fraction,
        "amodal_bbox_px": labels.amodal_bbox_px.tolist(),
        "cx": labels.cx,
        "cy": labels.cy,
        "log_h": labels.log_h,
        "visible_bbox_px": None if labels.visible_bbox_px is None else labels.visible_bbox_px.tolist(),
        "visible_cx": labels.visible_cx,
        "visible_cy": labels.visible_cy,
        "visible_log_h": labels.visible_log_h,
        "whole_target_positive_depth": labels.whole_target_positive_depth,
        "amodal_regression_valid": labels.amodal_regression_valid,
        "in_frame": labels.in_frame,
        "p_visible_target": labels.p_visible_target,
        "occlusion_flag": labels.occlusion_flag,
        "modal_front_object_id": labels.modal_front_object_id,
    }


@cache
def _authority_data() -> tuple[Any, dict[str, object], dict[str, object], dict[str, object], tuple[Any, ...], dict[str, Any]]:
    rig = load_rig_manifest(RIG_MANIFEST)
    asset = load_asset_manifest(ASSET_MANIFEST)
    scene = json.loads(SCENE_MANIFEST.read_text(encoding="utf-8"))
    pose = canonical_pose_export_payload(rig)
    bindings = RootRecordBindings.from_manifests(
        asset_manifest=asset,
        asset_manifest_file_sha256=_sha_file(ASSET_MANIFEST),
        rig_manifest=rig,
        rig_manifest_file_sha256=_sha_file(RIG_MANIFEST),
        scene_manifest=scene,
        scene_manifest_file_sha256=_sha_file(SCENE_MANIFEST),
        pose_export_manifest=pose,
        pose_export_file_sha256=PINNED_POSE_TABLE_EXPORT_SHA256,
    )
    fixtures = canonical_skier_fixtures()
    roots = {fixture.fixture_id: build_canonical_root_record(fixture, rig, bindings) for fixture in fixtures}
    return rig, asset, scene, pose, fixtures, roots


def _write(root: Path, relative: str, payload: bytes) -> str:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return _sha(payload)


def _artifact_paths(fixture_id: str, record_index: int) -> dict[str, str]:
    prefix = f"frames/{fixture_id}/{record_index:02d}"
    return {
        "frame_metadata_path": f"{prefix}/frame.json",
        "state_record_path": f"{prefix}/state.canonical",
        "pose_record_path": f"{prefix}/pose.json",
        "rgb_path": f"{prefix}/rgb.png",
        "mask_full_path": f"{prefix}/mask_full.png",
        "mask_in_path": f"{prefix}/mask_in.png",
        "mask_visible_path": f"{prefix}/mask_visible.png",
        "target_only_id_path": f"{prefix}/target_only_id.png",
        "target_id_path": f"{prefix}/target_id.png",
    }


def _camera_payload(fixture: Any) -> dict[str, object]:
    contract = default_camera_contract()
    world_rig = initial_camera_rig_transform(fixture.records()[2])
    rig_cam = np.eye(4, dtype=np.float64)
    rig_cam[:3, :3] = contract.R_rig_from_cam
    cam_rig = np.eye(4, dtype=np.float64)
    cam_rig[:3, :3] = contract.R_cam_from_rig
    world_cam = world_rig @ rig_cam
    identity = np.eye(4, dtype=np.float64)
    return {
        "camera_root_schema_version": CAMERA_ROOT_SCHEMA_VERSION,
        "T_world_from_rig_requested": world_rig,
        "T_world_from_rig_achieved": world_rig,
        "T_world_from_cam_requested": world_cam,
        "T_world_from_cam_achieved": world_cam,
        "T_cam_from_world_achieved": np.linalg.inv(world_cam),
        "T_rig0_from_rig_t_requested": identity,
        "T_rig0_from_rig_t_achieved": identity,
        "T_rig_from_cam_requested": rig_cam,
        "T_cam_from_rig_requested": cam_rig,
        "T_rig_from_cam_achieved": rig_cam,
        "T_cam_from_rig_achieved": cam_rig,
        "K_requested": contract.K,
        "K_achieved_blender": contract.K,
        "achieved_intrinsic_residual_px": 0.0,
        "achieved_camera_position_residual_m": 0.0,
        "achieved_camera_rotation_residual_rad": 0.0,
        "achieved_rig_action_position_residual_m": 0.0,
        "achieved_rig_action_rotation_residual_rad": 0.0,
    }


def _camera_json(payload: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in payload.items()
    }


def _camera_sha_from_json(value: Mapping[str, object]) -> str:
    payload = {
        key: np.asarray(item, dtype=np.float64) if key.startswith("T_") or key.startswith("K_") else item
        for key, item in value.items()
    }
    return _sha(canonical_bytes(payload))


def _write_replay(root: Path) -> Path:
    rig, _asset, scene, pose, fixtures, roots = _authority_data()
    root.mkdir(parents=True)
    root_rows: list[dict[str, object]] = []
    for fixture in fixtures:
        root_record = roots[fixture.fixture_id]
        relative = f"roots/{fixture.fixture_id}.canonical"
        root_sha = _write(root, relative, root_record.canonical_bytes())
        root_rows.append(
            {
                "fixture_id": fixture.fixture_id,
                "root_id": root_record.root_id,
                "split_group_id": root_record.split_group_id,
                "split": "test",
                "root_record_schema_version": ROOT_RECORD_SCHEMA_VERSION,
                "root_envelope_schema_version": ROOT_ENVELOPE_SCHEMA_VERSION,
                "root_record_path": relative,
                "root_record_sha256": root_sha,
            }
        )

    rows: list[dict[str, object]] = []
    pose_rows = pose["rows"]
    assert isinstance(pose_rows, list)
    flat_index = 0
    visible_labels = _label(False)
    rgb_payload = _png(np.zeros((224, 224, 3), dtype=np.uint8))
    for fixture in fixtures:
        root_record = roots[fixture.fixture_id]
        camera_payload = _camera_payload(fixture)
        camera_sha = _sha(canonical_bytes(camera_payload))
        for record_index, source_record in enumerate(fixture.records()):
            absolute_tick = source_record.state.absolute_tick
            pure_pose = evaluate_pose(source_record, rig)
            posed_record = record_with_pose(source_record, pure_pose)
            pose_row = pose_rows[flat_index]
            paths = _artifact_paths(fixture.fixture_id, record_index)
            state_sha = _write(root, paths["state_record_path"], posed_record.canonical_bytes())
            pose_record_sha = _write(root, paths["pose_record_path"], _json_bytes(pose_row, compact=True))
            rgb_sha = _write(root, paths["rgb_path"], rgb_payload)
            during_occlusion = fixture.fixture_id == "occlusion_path" and absolute_tick == 4
            labels = _label(during_occlusion)
            m_in = np.asarray(labels.m_in, dtype=np.uint8) * 255
            m_vis = np.asarray(labels.m_vis, dtype=np.uint8) * 255
            mask_full = np.asarray(labels.m_full.crop(origin_u_px=100, origin_v_px=80, width_px=20, height_px=40), dtype=np.uint8) * 255
            mask_full_sha = _write(root, paths["mask_full_path"], _png(mask_full))
            mask_in_sha = _write(root, paths["mask_in_path"], _png(m_in))
            mask_visible_sha = _write(root, paths["mask_visible_path"], _png(m_vis))
            target_only_sha = _write(root, paths["target_only_id_path"], _png(m_in))
            target_id_sha = _write(root, paths["target_id_path"], _png(m_vis))
            target_geometry_sha = _sha(f"geometry:{fixture.fixture_id}:{absolute_tick}")
            artifact_hashes = {
                "state_record_sha256": state_sha,
                "pose_record_sha256": pose_record_sha,
                "rgb_sha256": rgb_sha,
                "mask_full_sha256": mask_full_sha,
                "mask_in_sha256": mask_in_sha,
                "mask_visible_sha256": mask_visible_sha,
                "target_only_id_sha256": target_only_sha,
                "target_id_sha256": target_id_sha,
            }
            metadata: dict[str, object] = {
                "schema_version": FRAME_SCHEMA_VERSION,
                "fixture_id": fixture.fixture_id,
                "root_id": root_record.root_id,
                "split_group_id": root_record.split_group_id,
                "split": "test",
                "root_record_sha256": root_record.canonical_sha256(),
                "record_index": record_index,
                "absolute_tick": absolute_tick,
                "timestamp_seconds": absolute_tick * FIXED_DT_SECONDS,
                "branch_id": "zero",
                "requested_command": [0.0, 0.0, 0.0, 0.0],
                "requested_command_valid": True,
                "record_valid": True,
                "dt_seconds": FIXED_DT_SECONDS,
                "source_skier_digest": source_record.skier_digest(),
                "posed_skier_digest": posed_record.skier_digest(),
                "pose_sha256": pure_pose.canonical_sha256(),
                "camera_transform_sha256": camera_sha,
                "camera": _camera_json(camera_payload),
                "target_geometry_sha256": target_geometry_sha,
                "labels_sha256": labels.canonical_sha256(),
                "labels": _label_summary(labels),
                "fixed_occluder_enabled": fixture.fixture_id == "occlusion_path",
                "fixed_occluder_observation_changed": during_occlusion,
                "occlusion_flag": labels.occlusion_flag,
                "modal_front_object_id": labels.modal_front_object_id,
                "p_visible_target": labels.p_visible_target,
                "without_fixed_occluder_labels_sha256": (
                    visible_labels.canonical_sha256() if fixture.fixture_id == "occlusion_path" else None
                ),
                "projection_parity_max_px": 0.25,
                "id_pass_schema_version": "b3-cs3-blender-depsgraph-center-ray-id-v1",
                "local_pose_reconstruction": {"position_m": 0.0, "rotation_frobenius": 0.0},
                "pose_parity_maxima": {
                    "object_position_m": 0.0,
                    "object_rotation_rad": 0.0,
                    "root_position_m": 0.0,
                    "root_rotation_rad": 0.0,
                    "world_position_m": 0.0,
                    "world_rotation_rad": 0.0,
                },
                "equipment_metrics": {
                    "max_equipment_origin_residual_m": 0.0,
                    "max_ski_frame_residual_rad": 0.0,
                    "max_contact_origin_residual_m": 0.0,
                    "inner_tip_gap_residual_m": 0.0,
                    "inner_tip_gap_m": 0.1,
                    "max_boot_binding_position_m": 0.0,
                    "max_boot_binding_rotation_rad": 0.0,
                    "max_binding_relative_position_residual_m": 0.0,
                    "max_binding_relative_rotation_residual_rad": 0.0,
                    "max_attack_residual_rad": 0.0,
                    "max_edge_residual_rad": 0.0,
                    "max_slip_component_residual_m_s": 0.0,
                },
                "mask_full_origin_uv_px": [100, 80],
                **{key: value for key, value in paths.items() if key != "frame_metadata_path"},
                **artifact_hashes,
            }
            metadata_bytes = _json_bytes(metadata)
            metadata_sha = _write(root, paths["frame_metadata_path"], metadata_bytes)
            rows.append(
                {
                    "fixture_id": fixture.fixture_id,
                    "root_id": root_record.root_id,
                    "split_group_id": root_record.split_group_id,
                    "split": "test",
                    "root_record_sha256": root_record.canonical_sha256(),
                    "record_index": record_index,
                    "absolute_tick": absolute_tick,
                    "branch_id": "zero",
                    "requested_command": [0.0, 0.0, 0.0, 0.0],
                    "requested_command_valid": True,
                    "record_valid": True,
                    "dt_seconds": FIXED_DT_SECONDS,
                    "frame_metadata_path": paths["frame_metadata_path"],
                    "frame_metadata_sha256": metadata_sha,
                    **artifact_hashes,
                    "source_skier_digest": source_record.skier_digest(),
                    "posed_skier_digest": posed_record.skier_digest(),
                    "pose_sha256": pure_pose.canonical_sha256(),
                    "camera_transform_sha256": camera_sha,
                    "labels_sha256": labels.canonical_sha256(),
                    "target_geometry_sha256": target_geometry_sha,
                    "projection_parity_max_px": 0.25,
                    "occlusion_flag": labels.occlusion_flag,
                    "modal_front_object_id": labels.modal_front_object_id,
                    "p_visible_target": labels.p_visible_target,
                    "fixed_occluder_observation_changed": during_occlusion,
                }
            )
            flat_index += 1
    manifest = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "blender_version": scene["blender_version"],
        "blender_build_hash": scene["blender_build_hash"],
        "scene_sha256": scene["scene_sha256"],
        "scene_manifest_file_sha256": _sha_file(SCENE_MANIFEST),
        "asset_manifest_file_sha256": _sha_file(ASSET_MANIFEST),
        "rig_manifest_file_sha256": _sha_file(RIG_MANIFEST),
        "rig_manifest_canonical_sha256": rig.canonical_sha256(),
        "authoritative_pose_table_file_sha256": PINNED_POSE_TABLE_EXPORT_SHA256,
        "canonical_pose_table_sha256": PINNED_CANONICAL_POSE_TABLE_SHA256,
        "renderer": canonical_renderer_contract(),
        "root_record_schema_version": ROOT_RECORD_SCHEMA_VERSION,
        "root_envelope_schema_version": ROOT_ENVELOPE_SCHEMA_VERSION,
        "root_count": 8,
        "roots": root_rows,
        "fixture_count": 8,
        "frame_count": 88,
        "rows": rows,
    }
    manifest_path = root / "replay.json"
    manifest_path.write_bytes(_json_bytes(manifest))
    return manifest_path


def _pair(tmp_path: Path) -> tuple[Path, Path, Path]:
    pose_path = tmp_path / "pose_table_authoritative.json"
    pose_path.parent.mkdir(parents=True, exist_ok=True)
    pose_path.write_bytes(canonical_pose_export_json_bytes(load_rig_manifest(RIG_MANIFEST)))
    return _write_replay(tmp_path / "first"), _write_replay(tmp_path / "second"), pose_path


def _audit(first: Path, second: Path, pose_path: Path, **kwargs: object) -> dict[str, object]:
    return audit_replays(
        first,
        second,
        asset_manifest_path=ASSET_MANIFEST,
        rig_manifest_path=RIG_MANIFEST,
        scene_manifest_path=SCENE_MANIFEST,
        pose_table_path=pose_path,
        **kwargs,  # type: ignore[arg-type]
    )


def _rewrite_manifest(path: Path, mutate: Callable[[dict[str, object]], None]) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    path.write_bytes(_json_bytes(value))


def _rewrite_row_metadata(
    manifest_path: Path,
    row_index: int,
    mutate: Callable[[dict[str, object], dict[str, object]], None],
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = manifest["rows"][row_index]
    metadata_path = manifest_path.parent / row["frame_metadata_path"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mutate(row, metadata)
    payload = _json_bytes(metadata)
    metadata_path.write_bytes(payload)
    row["frame_metadata_sha256"] = _sha(payload)
    manifest_path.write_bytes(_json_bytes(manifest))


def test_v2_replay_audit_reconstructs_800_files_and_visual_strip(tmp_path: Path) -> None:
    first, second, pose_path = _pair(tmp_path)
    first_strip = tmp_path / "contact-first.svg"
    second_strip = tmp_path / "contact-second.svg"
    report = _audit(first, second, pose_path, visual_strip_path=first_strip)
    repeated = _audit(first, second, pose_path, visual_strip_path=second_strip)
    assert report["schema_version"] == AUDIT_SCHEMA_VERSION
    assert report["status"] == "PASS"
    assert report["compared_file_count"] == 800
    assert report["root_records_recomputed"] is True
    assert report["semantic_state_pose_reconstruction"] is True
    assert report["mechanical_label_reconstruction"] is True
    assert first_strip.read_bytes() == second_strip.read_bytes()
    assert first_strip.read_text(encoding="utf-8").count("data:image/png;base64,") == 24
    assert report == repeated


def test_audit_rejects_artifact_bytes_even_when_manifest_is_unchanged(tmp_path: Path) -> None:
    first, second, pose_path = _pair(tmp_path)
    rgb = second.parent / "frames/straight/00/rgb.png"
    rgb.write_bytes(rgb.read_bytes() + b"corrupt")
    with pytest.raises(ReplayAuditError, match="rgb SHA-256 mismatch"):
        _audit(first, second, pose_path)


def test_audit_rejects_nonzero_or_invalid_command_even_when_both_runs_agree(tmp_path: Path) -> None:
    first, second, pose_path = _pair(tmp_path)

    def mutate(row: dict[str, object], metadata: dict[str, object]) -> None:
        row["requested_command"] = [0.1, 0.0, 0.0, 0.0]
        metadata["requested_command"] = row["requested_command"]
        row["requested_command_valid"] = False
        metadata["requested_command_valid"] = False

    for path in (first, second):
        _rewrite_row_metadata(path, 0, mutate)
    with pytest.raises(ReplayAuditError, match="zero action/validity/dt contract drift"):
        _audit(first, second, pose_path)


def test_audit_rejects_root_bytes_or_split_identity_drift(tmp_path: Path) -> None:
    first, second, pose_path = _pair(tmp_path)
    for manifest_path in (first, second):
        root_path = manifest_path.parent / "roots/straight.canonical"
        root_path.write_bytes(root_path.read_bytes() + b"corrupt")
    with pytest.raises(ReplayAuditError, match="canonical root bytes mismatch"):
        _audit(first, second, pose_path)

    first, second, pose_path = _pair(tmp_path / "split")

    def split_drift(manifest: dict[str, object]) -> None:
        manifest["roots"][0]["split"] = "train"

    for path in (first, second):
        _rewrite_manifest(path, split_drift)
    with pytest.raises(ReplayAuditError, match="root identity drift"):
        _audit(first, second, pose_path)


def test_audit_rejects_named_camera_extrinsic_drift(tmp_path: Path) -> None:
    first, second, pose_path = _pair(tmp_path)

    def mutate(row: dict[str, object], metadata: dict[str, object]) -> None:
        camera = metadata["camera"]
        camera["T_rig_from_cam_requested"][0][3] = 0.1
        changed = _camera_sha_from_json(camera)
        metadata["camera_transform_sha256"] = changed
        row["camera_transform_sha256"] = changed

    for path in (first, second):
        _rewrite_row_metadata(path, 0, mutate)
    with pytest.raises(ReplayAuditError, match="requested transform drift"):
        _audit(first, second, pose_path)


def test_audit_rejects_extra_metadata_key_and_semantically_changed_state(tmp_path: Path) -> None:
    first, second, pose_path = _pair(tmp_path)

    def extra(_row: dict[str, object], metadata: dict[str, object]) -> None:
        metadata["undeclared"] = True

    for path in (first, second):
        _rewrite_row_metadata(path, 0, extra)
    with pytest.raises(ReplayAuditError, match="key mismatch"):
        _audit(first, second, pose_path)

    first, second, pose_path = _pair(tmp_path / "state")
    for manifest_path in (first, second):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        row = manifest["rows"][0]
        metadata_path = manifest_path.parent / row["frame_metadata_path"]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        state_path = manifest_path.parent / metadata["state_record_path"]
        state_path.write_bytes(b"semantic drift")
        changed = _sha_file(state_path)
        row["state_record_sha256"] = changed
        metadata["state_record_sha256"] = changed
        metadata_path.write_bytes(_json_bytes(metadata))
        row["frame_metadata_sha256"] = _sha_file(metadata_path)
        manifest_path.write_bytes(_json_bytes(manifest))
    with pytest.raises(ReplayAuditError, match="canonical state bytes differ"):
        _audit(first, second, pose_path)


def test_audit_requires_fixed_occluder_only_on_preregistered_temporary_path(tmp_path: Path) -> None:
    first, second, pose_path = _pair(tmp_path)

    def enable_straight(_row: dict[str, object], metadata: dict[str, object]) -> None:
        metadata["fixed_occluder_enabled"] = True

    for path in (first, second):
        _rewrite_row_metadata(path, 0, enable_straight)
    with pytest.raises(ReplayAuditError, match="enabled only for occlusion_path"):
        _audit(first, second, pose_path)


def test_audit_rejects_pose_authority_file_drift(tmp_path: Path) -> None:
    first, second, pose_path = _pair(tmp_path)
    pose_path.write_bytes(pose_path.read_bytes() + b" ")
    with pytest.raises(ReplayAuditError, match="differs from independent PURE export"):
        _audit(first, second, pose_path)
