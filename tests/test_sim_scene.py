"""PURE geometry tests for the fixed B3-CS3 scene camera root."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vllatent.sim.contracts import BranchId, DatasetSplit, canonical_bytes, default_camera_contract
from vllatent.sim.frames import project_blender_camera
from vllatent.sim.pose import (
    canonical_pose_export_payload,
    canonical_pose_export_sha256,
    construct_pose_root,
)
from vllatent.sim.rig import RigManifest, load_rig_manifest
from vllatent.sim.scene import (
    ROOT_ENVELOPE_SCHEMA_VERSION,
    TARGET_OBJECT_NAMES,
    CanonicalRootRecord,
    RootRecordBindings,
    build_canonical_root_record,
    canonical_renderer_contract,
    decode_canonical_mapping_bytes,
    initial_camera_rig_transform,
    validate_canonical_root_bytes,
)
from vllatent.sim.skier_fixtures import canonical_skier_fixtures

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bindings() -> tuple[RootRecordBindings, RigManifest]:
    manifest_root = _REPO_ROOT / "manifests" / "b3_cs3"
    asset_path = manifest_root / "asset.json"
    rig_path = manifest_root / "rig.json"
    scene_path = manifest_root / "scene.json"
    asset = json.loads(asset_path.read_text(encoding="utf-8"))
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    rig = load_rig_manifest(rig_path)
    pose_export = canonical_pose_export_payload(rig)
    return (
        RootRecordBindings.from_manifests(
            asset_manifest=asset,
            asset_manifest_file_sha256=_file_sha256(asset_path),
            rig_manifest=rig,
            rig_manifest_file_sha256=_file_sha256(rig_path),
            scene_manifest=scene,
            scene_manifest_file_sha256=_file_sha256(scene_path),
            pose_export_manifest=pose_export,
            pose_export_file_sha256=canonical_pose_export_sha256(rig),
        ),
        rig,
    )


def test_initial_camera_is_proper_immutable_and_centers_each_tick_zero_pelvis() -> None:
    camera = default_camera_contract()
    T_rig_from_cam = np.eye(4, dtype=np.float64)
    T_rig_from_cam[:3, :3] = camera.R_rig_from_cam
    for fixture in canonical_skier_fixtures():
        record = fixture.records()[2]
        T_world_from_rig = initial_camera_rig_transform(record)
        assert not T_world_from_rig.flags.writeable
        assert math.isclose(
            float(np.linalg.det(T_world_from_rig[:3, :3])),
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        T_world_from_cam = T_world_from_rig @ T_rig_from_cam
        pelvis_world = np.append(construct_pose_root(record).pelvis_point_world_m, 1.0)
        pelvis_cam = np.linalg.inv(T_world_from_cam) @ pelvis_world
        pixel = project_blender_camera(pelvis_cam[None, :3].astype(np.float64), camera.K)[0]
        assert np.allclose(pixel, np.array([112.0, 112.0]), rtol=0.0, atol=1.0e-10)
        assert -pelvis_cam[2] > 2.0


def test_scene_target_set_is_exact_and_camera_api_has_no_branch_or_observation_input() -> None:
    assert TARGET_OBJECT_NAMES == (
        "CS3_Body",
        "CS3_Eyes",
        "CS3_Eyebrows",
        "CS3_Boot_L",
        "CS3_Boot_R",
        "CS3_Helmet",
    )
    assert tuple(inspect.signature(initial_camera_rig_transform).parameters) == ("record",)


def test_renderer_contract_is_exact_cpu_deterministic_and_artifact_versioned() -> None:
    assert canonical_renderer_contract() == {
        "schema_version": "b3-cs3-cycles-cpu-renderer-v1",
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
        "png_schema_version": "b3-cs3-lossless-png-fixed-filter-v1",
        "near_clip_schema_version": "b3-cs3-camera-near-plane-triangle-clip-v1",
        "id_pass_schema_version": "b3-cs3-blender-depsgraph-center-ray-id-v1",
    }


def test_complete_root_record_is_deterministic_immutable_and_reconstructable() -> None:
    bindings, rig = _bindings()
    fixture = canonical_skier_fixtures()[0]
    root = build_canonical_root_record(fixture, rig, bindings)
    repeated = build_canonical_root_record(fixture, rig, bindings)
    assert root.canonical_bytes() == repeated.canonical_bytes()
    assert root.root_id == hashlib.sha256(root.record_canonical_bytes()).hexdigest()
    assert root.split_group_id == root.root_id
    assert root.split is DatasetSplit.TEST
    identities = root.sibling_identities()
    assert tuple(identity.branch_id for identity in identities) == tuple(BranchId)
    assert all(identity.root_id == root.root_id for identity in identities)
    assert all(identity.split_group_id == root.root_id for identity in identities)

    record = root.record
    assert record["provenance"] == bindings.payload()
    table = record["canonical_skier_pose_table"]
    assert isinstance(table, Mapping)
    assert np.array_equal(table["absolute_ticks"], np.arange(-2, 9, dtype=np.int64))
    assert table["row_count"] == 11
    rows = table["rows"]
    assert isinstance(rows, tuple) and len(rows) == 11
    for source_record, row in zip(fixture.records(), rows, strict=True):
        assert isinstance(row, Mapping)
        assert row["source_record_sha256"] == hashlib.sha256(source_record.canonical_bytes()).hexdigest()
        assert row["source_skier_digest"] == source_record.skier_digest()
        assert row["posed_skier_digest"] != source_record.skier_digest()
    for row, authority_row in zip(rows, bindings.fixture_pose_rows(fixture.fixture_id), strict=True):
        pose_hex = authority_row["pose_record_canonical_hex"]
        posed_hex = authority_row["posed_record_canonical_hex"]
        assert isinstance(pose_hex, str)
        assert isinstance(posed_hex, str)
        assert canonical_bytes(row["pose_record"]) == bytes.fromhex(pose_hex)
        assert canonical_bytes(row["posed_record"]) == bytes.fromhex(posed_hex)
    history = record["history"]
    assert isinstance(history, tuple) and tuple(row["absolute_tick"] for row in history) == (-2, -1, 0)
    branches = record["branch_programs"]
    assert isinstance(branches, Mapping)
    assert branches["branch_count"] == 9
    assert tuple(row["branch_id"] for row in branches["rows"]) == tuple(BranchId)
    assert all(bool(np.all(row["record_valid"])) for row in branches["rows"])
    continuation = record["continuation"]
    assert isinstance(continuation, Mapping)
    assert continuation["audit_version"] == "b3-cs2-continuation-audit-v1"
    assert continuation["terminal_key_version"] == "b3-cs2-terminal-key-half-away-v1"
    assert len(continuation["continuation_target_sha256"]) == 64
    collision = continuation["catalog_collision_audit"]
    assert isinstance(collision, Mapping)
    assert collision["collision_free"] is True
    assert collision["fixture_count"] == 8
    camera = record["initial_camera"]
    assert isinstance(camera, Mapping)
    assert np.array_equal(camera["T_cam_from_rig"] @ camera["T_rig_from_cam"], np.eye(4))
    assert not camera["T_world_from_rig"].flags.writeable
    with pytest.raises(TypeError):
        record["provenance"] = {}  # type: ignore[index]
    with pytest.raises(ValueError):
        camera["K"][0, 0] = 0.0
    validate_canonical_root_bytes(root.canonical_bytes(), root)


def test_root_record_binds_obstacle_policy_without_contaminating_skier_rows() -> None:
    bindings, rig = _bindings()
    fixtures = canonical_skier_fixtures()
    straight = build_canonical_root_record(fixtures[0], rig, bindings)
    occlusion = build_canonical_root_record(fixtures[6], rig, bindings)
    straight_obstacle = straight.record["obstacle"]
    occlusion_obstacle = occlusion.record["obstacle"]
    assert isinstance(straight_obstacle, Mapping)
    assert isinstance(occlusion_obstacle, Mapping)
    assert straight_obstacle["enabled"] is False
    assert occlusion_obstacle["enabled"] is True
    for root, fixture in ((straight, fixtures[0]), (occlusion, fixtures[6])):
        table = root.record["canonical_skier_pose_table"]
        assert isinstance(table, Mapping)
        rows = table["rows"]
        assert isinstance(rows, tuple)
        assert [row["source_skier_digest"] for row in rows] == [
            source.skier_digest() for source in fixture.records()
        ]


def test_root_envelope_and_binding_corruption_fail_closed() -> None:
    bindings, rig = _bindings()
    root = build_canonical_root_record(canonical_skier_fixtures()[0], rig, bindings)
    envelope = dict(root.payload())
    envelope["root_id"] = "0" * 64
    with pytest.raises(ValueError, match="root_id"):
        CanonicalRootRecord.from_mapping(envelope)
    envelope = dict(root.payload())
    envelope["split_group_id"] = "1" * 64
    with pytest.raises(ValueError, match="split_group_id"):
        CanonicalRootRecord.from_mapping(envelope)
    envelope = dict(root.payload())
    envelope["extra"] = True
    with pytest.raises(ValueError, match="key mismatch"):
        CanonicalRootRecord.from_mapping(envelope)
    corrupted = bytearray(root.canonical_bytes())
    corrupted[-2] = ord(" ")
    with pytest.raises(ValueError, match="bytes mismatch"):
        validate_canonical_root_bytes(bytes(corrupted), root)

    with pytest.raises(ValueError, match="license_manifest_sha256"):
        replace(bindings, license_manifest_sha256="2" * 64)


def test_root_payload_hash_excludes_envelope_ids() -> None:
    bindings, rig = _bindings()
    root = build_canonical_root_record(canonical_skier_fixtures()[0], rig, bindings)
    assert b"root_id" not in root.record_canonical_bytes()
    assert b"split_group_id" not in root.record_canonical_bytes()
    assert root.payload()["schema_version"] == ROOT_ENVELOPE_SCHEMA_VERSION
    assert canonical_bytes(root.record) == root.record_canonical_bytes()


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        (b'{"a":{"$int":"1"},"a":{"$int":"1"}}', "duplicate JSON key"),
        (b'{"a":{"$int":"01"}}', "non-canonical integer"),
        (b'{"a":{"$int":"1","extra":true}}', "sole exact mapping key"),
        (b'{"a":1}', "untagged JSON number"),
        (b'{"z":true,"a":false}', "byte-identical canonical serialization"),
        (
            b'{"a":{"$ndarray":{"data_hex":"000000000000f03f","dtype":">f8","shape":[1]}}}',
            "little-endian dtype",
        ),
        (
            b'{"a":{"$ndarray":{"data_hex":"00","dtype":"<f8","shape":[1]}}}',
            "byte length",
        ),
    ),
)
def test_canonical_tagged_decoder_rejects_ambiguous_or_noncanonical_bytes(
    payload: bytes, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        decode_canonical_mapping_bytes(payload)


def test_pose_authority_tampering_fails_before_or_during_root_reconstruction() -> None:
    bindings, rig = _bindings()
    fixture = canonical_skier_fixtures()[0]
    rows = list(bindings.pose_rows)
    first = dict(rows[0])
    encoded = first["pose_record_canonical_hex"]
    assert isinstance(encoded, str)
    first["pose_record_canonical_hex"] = encoded[:-1] + ("0" if encoded[-1] != "0" else "1")
    with pytest.raises(ValueError, match="pose canonical bytes/hash mismatch"):
        replace(bindings, pose_rows=(first, *rows[1:]))

    first = dict(rows[0])
    pose_hex = first["pose_record_canonical_hex"]
    assert isinstance(pose_hex, str)
    pose_payload = dict(
        decode_canonical_mapping_bytes(bytes.fromhex(pose_hex))
    )
    pose_payload["absolute_tick"] = -1
    changed_pose_bytes = canonical_bytes(pose_payload)
    first["pose_record_canonical_hex"] = changed_pose_bytes.hex()
    first["pose_sha256"] = hashlib.sha256(changed_pose_bytes).hexdigest()
    semantically_tampered = replace(bindings, pose_rows=(first, *rows[1:]))
    with pytest.raises(ValueError, match="canonical pose/posed fixture binding mismatch"):
        build_canonical_root_record(fixture, rig, semantically_tampered)

    first = dict(rows[0])
    first["posed_skier_digest"] = "0" * 64
    digest_tampered = replace(bindings, pose_rows=(first, *rows[1:]))
    with pytest.raises(ValueError, match="posed-skier digest mismatch"):
        build_canonical_root_record(fixture, rig, digest_tampered)
