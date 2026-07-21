"""PURE contract tests for the pre-animation B3-CS3 rig freeze."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vllatent.sim.rig import (
    AMPLITUDE_EXCLUDED_SEMANTICS,
    BONE_BY_SEMANTIC,
    LEFT_RIGHT_PAIRS,
    R_ROOT_FROM_SOURCE,
    SOURCE_BLEND_SHA256,
    TRACKED_SEMANTICS,
    audit_asset_manifest,
    load_asset_manifest,
    load_rig_manifest,
    rig_manifest_from_mapping,
)

REPO_ROOT = Path(__file__).parents[1]
ASSET_MANIFEST = REPO_ROOT / "manifests" / "b3_cs3" / "asset.json"
RIG_MANIFEST = REPO_ROOT / "manifests" / "b3_cs3" / "rig.json"


def test_asset_manifest_is_exactly_one_cc0_pack_with_two_allowlisted_members() -> None:
    manifest = load_asset_manifest(ASSET_MANIFEST)
    assert manifest["external_pack_count"] == 1
    assert manifest["allowlisted_external_members"] == [
        "License_Source.txt",
        "Base Characters/Regular_Male_FullBody.blend",
    ]
    assert manifest["pack"]["selected_member_sha256"] == SOURCE_BLEND_SHA256  # type: ignore[index]
    assert manifest["blender"]["render_device"] == "CPU"  # type: ignore[index]
    assert [row["id"] for row in manifest["saved_license_evidence"]] == [  # type: ignore[index]
        "pack_page",
        "cc0_legal_code",
        "quaternius_faq",
    ]
    assert manifest["import_conversion_settings"]["plugins_required"] == 0  # type: ignore[index]
    assert set(manifest["derived_artifacts"]) == {  # type: ignore[arg-type]
        "derived_rig_sha256",
        "rig_manifest_canonical_sha256",
        "root_free_clip_canonical_sha256",
        "authoritative_pose_table_file_sha256",
        "authored_scene_sha256",
    }


def test_asset_manifest_fails_closed_on_pack_or_license_expansion() -> None:
    manifest = json.loads(ASSET_MANIFEST.read_text(encoding="utf-8"))
    manifest["external_pack_count"] = 2
    with pytest.raises(ValueError, match="exactly one external pack"):
        audit_asset_manifest(manifest)
    manifest = json.loads(ASSET_MANIFEST.read_text(encoding="utf-8"))
    manifest["allowlisted_external_members"].append("Engine Projects/Unity.zip")
    with pytest.raises(ValueError, match="selected blend and license only"):
        audit_asset_manifest(manifest)


@pytest.mark.parametrize(
    ("section", "field", "replacement", "error"),
    (
        ("saved_license_evidence", 0, "0" * 64, "path/hash evidence drift"),
        ("import_conversion_settings", "source_operation", "link", "conversion drift"),
        ("derived_artifacts", "authored_scene_sha256", "not-a-sha", "expected 64"),
    ),
)
def test_asset_manifest_rejects_missing_provenance_or_derived_binding(
    section: str,
    field: str | int,
    replacement: str,
    error: str,
) -> None:
    manifest = json.loads(ASSET_MANIFEST.read_text(encoding="utf-8"))
    if section == "saved_license_evidence":
        manifest[section][field]["sha256"] = replacement
    else:
        manifest[section][field] = replacement
    with pytest.raises(ValueError, match=error):
        audit_asset_manifest(manifest)


@pytest.mark.parametrize(
    ("domain", "field", "replacement"),
    (
        ("pack", "name", "different pack"),
        ("pack", "source_url", "https://example.invalid"),
        ("pack", "upstream_filename", "different.zip"),
        ("pack", "displayed_pack_date", "unknown"),
        ("pack", "acquisition_date", "2099-01-01"),
        ("pack", "selected_member_timestamp_utc", "2099-01-01T00:00:00Z"),
        ("blender", "binary_sha256", "0" * 64),
        ("blender", "build_hash", "different"),
    ),
)
def test_asset_manifest_rejects_provenance_metadata_drift(
    domain: str,
    field: str,
    replacement: str,
) -> None:
    manifest = json.loads(ASSET_MANIFEST.read_text(encoding="utf-8"))
    manifest[domain][field] = replacement
    with pytest.raises(ValueError, match="frozen value drift|pinned build drift"):
        audit_asset_manifest(manifest)


def test_root_frame_is_proper_frd_and_semantic_mapping_is_frozen() -> None:
    np.testing.assert_array_equal(
        R_ROOT_FROM_SOURCE,
        np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64),
    )
    np.testing.assert_allclose(R_ROOT_FROM_SOURCE.T @ R_ROOT_FROM_SOURCE, np.eye(3), atol=0.0)
    assert np.linalg.det(R_ROOT_FROM_SOURCE) == 1.0
    assert tuple(BONE_BY_SEMANTIC) == TRACKED_SEMANTICS
    assert BONE_BY_SEMANTIC["left_boot"] == "boot_bind_l"
    assert BONE_BY_SEMANTIC["right_boot"] == "boot_bind_r"
    assert AMPLITUDE_EXCLUDED_SEMANTICS == (
        "left_ankle",
        "right_ankle",
        "left_boot",
        "right_boot",
    )


def test_rig_manifest_has_exact_order_rigid_rest_frames_and_pelvis_origin() -> None:
    manifest = load_rig_manifest(RIG_MANIFEST)
    assert tuple(bone.semantic for bone in manifest.bones) == TRACKED_SEMANTICS
    assert manifest.bone("pelvis").blender_name == "pelvis"
    np.testing.assert_allclose(manifest.bone("pelvis").rest_matrix_root_m[:3, 3], np.zeros(3), atol=1.0e-7)
    for bone in manifest.bones:
        rotation = bone.rest_matrix_root_m[:3, :3]
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1.0e-10)
        assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1.0e-10)
        assert 0.0 <= bone.rigidization_frobenius_residual <= 1.0e-3
        assert not bone.raw_rest_matrix_root_m.flags.writeable
        assert not bone.rest_matrix_root_m.flags.writeable
        with pytest.raises(ValueError, match="WRITEABLE"):
            bone.rest_matrix_root_m.setflags(write=True)


def test_manifested_left_right_joint_origins_mirror_in_root_local_metres() -> None:
    manifest = load_rig_manifest(RIG_MANIFEST)
    sagittal_reflection = np.diag([1.0, -1.0, 1.0])
    for left_semantic, right_semantic in LEFT_RIGHT_PAIRS:
        left = manifest.bone(left_semantic).rest_matrix_root_m[:3, 3]
        right = manifest.bone(right_semantic).rest_matrix_root_m[:3, 3]
        np.testing.assert_allclose(left, sagittal_reflection @ right, rtol=0.0, atol=2.0e-5)


def test_rig_manifest_rejects_semantic_reordering_or_raw_rest_evidence_loss() -> None:
    manifest = json.loads(RIG_MANIFEST.read_text(encoding="utf-8"))
    manifest["bones"][0], manifest["bones"][1] = manifest["bones"][1], manifest["bones"][0]
    with pytest.raises(ValueError, match="semantic order"):
        rig_manifest_from_mapping(manifest)
    manifest = json.loads(RIG_MANIFEST.read_text(encoding="utf-8"))
    del manifest["bones"][0]["raw_rest_matrix_root_m"]
    with pytest.raises(ValueError, match="key mismatch"):
        rig_manifest_from_mapping(manifest)
