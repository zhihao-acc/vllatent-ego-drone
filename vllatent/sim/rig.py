"""PURE B3-CS3 skier-rig and external-asset manifest contracts.

The evaluated rig is frozen before pose authoring.  This module deliberately
contains no Blender dependency: it validates the exact semantic bone order,
root-local metric frame, rest frames, left/right pairing, and the one-pack CC0
allowlist emitted by the separately isolated Blender bridge.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from vllatent.sim.contracts import canonical_bytes

RIG_MANIFEST_SCHEMA_VERSION: Final[str] = "b3-cs3-rig-manifest-v1"
ASSET_MANIFEST_SCHEMA_VERSION: Final[str] = "b3-cs3-asset-manifest-v2"
RIG_ID: Final[str] = "quaternius-regular-male-cs3-v1"
REST_FRAME_EXTRACTION_VERSION: Final[str] = "closest-so3-svd-f64-det-correct-v1"
RIG_METRIC_SCALE: Final[float] = 1.15

SOURCE_ARCHIVE_SHA256: Final[str] = "631426cdc133f32ac5cee182e779487636ab4720e82d9e297a0daf16063bc00a"
SOURCE_BLEND_SHA256: Final[str] = "f35bee0656510d3832affe6fef4d00d61e1739021a713faa21d76dfde8b69cf2"
SOURCE_LICENSE_SHA256: Final[str] = "173831869ea6f9e270e1d13604c54a4dcf5af62e3593c47599510c929b892c99"
SOURCE_INVENTORY_SHA256: Final[str] = "d57004aaef1eb76aebf80307a74b198111abc174451e173f870c86538fa63745"
BLENDER_ARCHIVE_SHA256: Final[str] = "05ed7bd41bf3e61ae4f4a7cdc364c43088bf8b3fed702c2269c018fdf63a2188"
BLENDER_BINARY_SHA256: Final[str] = "dc72290ee8651c93c4a946c012c5f2a034946fd320e6c3ab214fa23181427428"
BLENDER_BUILD_HASH: Final[str] = "4db51e9d1e1e"
PACK_PAGE_EVIDENCE_SHA256: Final[str] = "f1df9c599ffb45ef095814829b38d67b40da699f5531372fde4a54bbfa25941a"
CC0_PAGE_EVIDENCE_SHA256: Final[str] = "4ceb8ae6835f2f5263caa0e39c9e1adca9469686c267475049b33521dabbe339"
FAQ_PAGE_EVIDENCE_SHA256: Final[str] = "83fabcc06fe168d61418c7681fdf15a4c11b558a7c246dabc713acdf5b8770e8"

TRACKED_SEMANTICS: Final[tuple[str, ...]] = (
    "pelvis",
    "chest",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_boot",
    "right_boot",
)

BONE_BY_SEMANTIC: Final[dict[str, str]] = {
    "pelvis": "pelvis",
    "chest": "spine_03",
    "head": "Head",
    "left_shoulder": "upperarm_l",
    "right_shoulder": "upperarm_r",
    "left_elbow": "lowerarm_l",
    "right_elbow": "lowerarm_r",
    "left_wrist": "hand_l",
    "right_wrist": "hand_r",
    "left_hip": "thigh_l",
    "right_hip": "thigh_r",
    "left_knee": "calf_l",
    "right_knee": "calf_r",
    "left_ankle": "foot_l",
    "right_ankle": "foot_r",
    "left_boot": "boot_bind_l",
    "right_boot": "boot_bind_r",
}

LEFT_RIGHT_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
    ("left_boot", "right_boot"),
)

AMPLITUDE_EXCLUDED_SEMANTICS: Final[tuple[str, ...]] = (
    "left_ankle",
    "right_ankle",
    "left_boot",
    "right_boot",
)

# Source Regular uses +X anatomical-left, -Y forward, +Z up.  The dynamics
# armature uses the proper body frame +X forward, +Y right, +Z down.
R_ROOT_FROM_SOURCE: Final[np.ndarray] = np.frombuffer(
    np.array(
        [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
        dtype=np.dtype("<f8"),
    ).tobytes(order="C"),
    dtype=np.dtype("<f8"),
).reshape((3, 3))

_SHA256_HEX_LENGTH = 64
_TRANSFORM_ATOL = 1.0e-10


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


def _exact_keys(name: str, value: object, expected: set[str]) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{name}: expected object")
    keys = set(value)
    if keys != expected:
        raise ValueError(f"{name}: key mismatch: missing={sorted(expected - keys)}, extra={sorted(keys - expected)}")
    return value


def _f64_transform(name: str, value: object) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.dtype("<f8"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name}: expected numeric 4x4 transform") from error
    if array.shape != (4, 4) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name}: expected finite 4x4 transform")
    if not np.allclose(array[3], np.array([0.0, 0.0, 0.0, 1.0]), rtol=0.0, atol=_TRANSFORM_ATOL):
        raise ValueError(f"{name}: invalid homogeneous bottom row")
    rotation = array[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=_TRANSFORM_ATOL):
        raise ValueError(f"{name}: rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, rel_tol=0.0, abs_tol=_TRANSFORM_ATOL):
        raise ValueError(f"{name}: rotation determinant must be +1")
    contiguous = np.array(array, dtype=np.dtype("<f8"), order="C", copy=True)
    contiguous[contiguous == 0.0] = 0.0
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=np.dtype("<f8")).reshape((4, 4))


def _f64_matrix(name: str, value: object) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.dtype("<f8"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name}: expected numeric 4x4 matrix") from error
    if array.shape != (4, 4) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name}: expected finite 4x4 matrix")
    contiguous = np.array(array, dtype=np.dtype("<f8"), order="C", copy=True)
    contiguous[contiguous == 0.0] = 0.0
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=np.dtype("<f8")).reshape((4, 4))


@dataclass(frozen=True, eq=False)
class EvaluatedBone:
    """One manifested semantic and immutable armature-space rest transform."""

    semantic: str
    blender_name: str
    parent_name: str | None
    parent_raw_rest_matrix_root_m: np.ndarray
    parent_rest_matrix_root_m: np.ndarray
    parent_rigidization_frobenius_residual: float
    raw_rest_matrix_root_m: np.ndarray
    rest_matrix_root_m: np.ndarray
    rigidization_frobenius_residual: float

    def __post_init__(self) -> None:
        semantic = _text("semantic", self.semantic)
        if semantic not in BONE_BY_SEMANTIC:
            raise ValueError(f"semantic: unrecognized evaluated semantic {semantic!r}")
        expected = BONE_BY_SEMANTIC[semantic]
        if self.blender_name != expected:
            raise ValueError(f"blender_name: {semantic!r} requires {expected!r}, got {self.blender_name!r}")
        if self.parent_name is not None:
            _text("parent_name", self.parent_name)
        object.__setattr__(
            self,
            "parent_raw_rest_matrix_root_m",
            _f64_matrix("parent_raw_rest_matrix_root_m", self.parent_raw_rest_matrix_root_m),
        )
        object.__setattr__(
            self,
            "parent_rest_matrix_root_m",
            _f64_transform("parent_rest_matrix_root_m", self.parent_rest_matrix_root_m),
        )
        parent_residual = float(self.parent_rigidization_frobenius_residual)
        if not math.isfinite(parent_residual) or parent_residual < 0.0 or parent_residual > 1.0e-3:
            raise ValueError("parent_rigidization_frobenius_residual: expected finite value in [0,1e-3]")
        object.__setattr__(self, "parent_rigidization_frobenius_residual", parent_residual)
        object.__setattr__(
            self,
            "raw_rest_matrix_root_m",
            _f64_matrix("raw_rest_matrix_root_m", self.raw_rest_matrix_root_m),
        )
        object.__setattr__(self, "rest_matrix_root_m", _f64_transform("rest_matrix_root_m", self.rest_matrix_root_m))
        residual = float(self.rigidization_frobenius_residual)
        if not math.isfinite(residual) or residual < 0.0 or residual > 1.0e-3:
            raise ValueError("rigidization_frobenius_residual: expected finite value in [0,1e-3]")
        object.__setattr__(self, "rigidization_frobenius_residual", residual)


@dataclass(frozen=True, eq=False)
class RigManifest:
    """Immutable renderer-neutral view of the pre-animation rig freeze."""

    rig_id: str
    metric_scale: float
    source_asset_sha256: str
    derived_rig_sha256: str
    normalization_max_position_residual_m: float
    bones: tuple[EvaluatedBone, ...]
    deform_bone_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.rig_id != RIG_ID:
            raise ValueError(f"rig_id: expected {RIG_ID!r}")
        if isinstance(self.metric_scale, bool) or float(self.metric_scale) != RIG_METRIC_SCALE:
            raise ValueError(f"metric_scale: expected frozen source-metre scale {RIG_METRIC_SCALE}")
        object.__setattr__(self, "metric_scale", RIG_METRIC_SCALE)
        if self.source_asset_sha256 != SOURCE_BLEND_SHA256:
            raise ValueError("source_asset_sha256: selected source drift")
        _sha256("derived_rig_sha256", self.derived_rig_sha256)
        normalization_residual = float(self.normalization_max_position_residual_m)
        if not math.isfinite(normalization_residual) or not 0.0 <= normalization_residual <= 1.0e-6:
            raise ValueError("normalization_max_position_residual_m: expected value in [0,1e-6]")
        object.__setattr__(self, "normalization_max_position_residual_m", normalization_residual)
        if tuple(bone.semantic for bone in self.bones) != TRACKED_SEMANTICS:
            raise ValueError("bones: expected exact evaluated semantic order")
        names = tuple(_text("deform_bone_name", name) for name in self.deform_bone_names)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError("deform_bone_names: expected unique sorted tuple")
        required_deform = set(BONE_BY_SEMANTIC.values()) - {"boot_bind_l", "boot_bind_r"}
        if not required_deform <= set(names):
            raise ValueError("deform_bone_names: missing evaluated source deform bones")

    def bone(self, semantic: str) -> EvaluatedBone:
        """Return one evaluated bone by exact semantic name."""
        for bone in self.bones:
            if bone.semantic == semantic:
                return bone
        raise KeyError(semantic)

    def canonical_sha256(self) -> str:
        """Hash the frozen semantic/rest-frame content, independent of JSON whitespace."""
        return hashlib.sha256(
            canonical_bytes(
                {
                    "schema_version": RIG_MANIFEST_SCHEMA_VERSION,
                    "rig_id": self.rig_id,
                    "metric_scale": self.metric_scale,
                    "source_asset_sha256": self.source_asset_sha256,
                    "derived_rig_sha256": self.derived_rig_sha256,
                    "normalization_max_position_residual_m": self.normalization_max_position_residual_m,
                    "root_local_frame": {"x": "forward", "y": "right", "z": "down"},
                    "R_root_from_source": R_ROOT_FROM_SOURCE,
                    "rest_frame_extraction": REST_FRAME_EXTRACTION_VERSION,
                    "left_right_pairs": LEFT_RIGHT_PAIRS,
                    "animation_amplitude_excluded": AMPLITUDE_EXCLUDED_SEMANTICS,
                    "bones": tuple(
                        {
                            "semantic": bone.semantic,
                            "blender_name": bone.blender_name,
                            "parent_name": bone.parent_name,
                            "parent_raw_rest_matrix_root_m": bone.parent_raw_rest_matrix_root_m,
                            "parent_rest_matrix_root_m": bone.parent_rest_matrix_root_m,
                            "parent_rigidization_frobenius_residual": (
                                bone.parent_rigidization_frobenius_residual
                            ),
                            "raw_rest_matrix_root_m": bone.raw_rest_matrix_root_m,
                            "rest_matrix_root_m": bone.rest_matrix_root_m,
                            "rigidization_frobenius_residual": bone.rigidization_frobenius_residual,
                        }
                        for bone in self.bones
                    ),
                    "deform_bone_names": self.deform_bone_names,
                }
            )
        ).hexdigest()


def rig_manifest_from_mapping(value: object) -> RigManifest:
    """Validate the exact tracked JSON representation of the CS3 rig freeze."""
    manifest = _exact_keys(
        "rig manifest",
        value,
        {
            "schema_version",
            "rig_id",
            "units",
            "metric_scale",
            "source_asset_sha256",
            "derived_rig_sha256",
            "normalization_max_position_residual_m",
            "source_frame",
            "root_local_frame",
            "R_root_from_source",
            "rest_frame_extraction",
            "origin_semantic",
            "origin_root_m",
            "left_right_pairs",
            "animation_amplitude_excluded",
            "contact_required",
            "objects",
            "deform_bone_names",
            "bones",
            "sanitization",
        },
    )
    if manifest["schema_version"] != RIG_MANIFEST_SCHEMA_VERSION:
        raise ValueError("schema_version: unsupported rig manifest")
    if manifest["units"] != "metre" or manifest["origin_semantic"] != "pelvis":
        raise ValueError("rig manifest: expected metre units and pelvis origin")
    if manifest["source_frame"] != {"x": "anatomical_left", "y": "rear", "z": "up"}:
        raise ValueError("source_frame: frozen axis contract drift")
    if manifest["root_local_frame"] != {
        "x": "forward",
        "y": "right",
        "z": "down",
        "handedness": "proper",
    }:
        raise ValueError("root_local_frame: frozen FRD contract drift")
    if manifest["rest_frame_extraction"] != REST_FRAME_EXTRACTION_VERSION:
        raise ValueError("rest_frame_extraction: frozen rigidization contract drift")
    source_rotation = np.asarray(manifest["R_root_from_source"], dtype=np.float64)
    if source_rotation.shape != (3, 3) or not np.array_equal(source_rotation, R_ROOT_FROM_SOURCE):
        raise ValueError("R_root_from_source: frozen rotation drift")
    if manifest["origin_root_m"] != [0.0, 0.0, 0.0]:
        raise ValueError("origin_root_m: expected exact pelvis origin")
    if manifest["left_right_pairs"] != [list(pair) for pair in LEFT_RIGHT_PAIRS]:
        raise ValueError("left_right_pairs: expected exact semantic mapping")
    if manifest["animation_amplitude_excluded"] != list(AMPLITUDE_EXCLUDED_SEMANTICS):
        raise ValueError("animation_amplitude_excluded: ankles/boots contract drift")
    if manifest["contact_required"] != ["left_ankle", "right_ankle", "left_boot", "right_boot"]:
        raise ValueError("contact_required: ankle/boot contact contract drift")
    if manifest["objects"] != {
        "armature": "CS3_SkierArmature",
        "target_meshes": ["CS3_Body", "CS3_Eyes", "CS3_Eyebrows"],
    }:
        raise ValueError("objects: sanitized target-object allowlist drift")
    sanitization = manifest["sanitization"]
    if sanitization != {
        "actions": 0,
        "external_libraries": 0,
        "images": 0,
        "non_armature_modifiers": 0,
        "plugins_required": 0,
        "source_texts": 0,
        "source_worlds": 0,
    }:
        raise ValueError("sanitization: derived rig is not closed and plugin-free")
    bones_raw = manifest["bones"]
    if not isinstance(bones_raw, list):
        raise TypeError("bones: expected array")
    bones: list[EvaluatedBone] = []
    for index, item in enumerate(bones_raw):
        bone = _exact_keys(
            f"bones[{index}]",
            item,
            {
                "semantic",
                "blender_name",
                "parent_name",
                "parent_raw_rest_matrix_root_m",
                "parent_rest_matrix_root_m",
                "parent_rigidization_frobenius_residual",
                "raw_rest_matrix_root_m",
                "rest_matrix_root_m",
                "rigidization_frobenius_residual",
            },
        )
        bones.append(
            EvaluatedBone(
                semantic=bone["semantic"],  # type: ignore[arg-type]
                blender_name=bone["blender_name"],  # type: ignore[arg-type]
                parent_name=bone["parent_name"],  # type: ignore[arg-type]
                parent_raw_rest_matrix_root_m=np.asarray(
                    bone["parent_raw_rest_matrix_root_m"], dtype=np.float64
                ),
                parent_rest_matrix_root_m=np.asarray(bone["parent_rest_matrix_root_m"], dtype=np.float64),
                parent_rigidization_frobenius_residual=bone[  # type: ignore[arg-type]
                    "parent_rigidization_frobenius_residual"
                ],
                raw_rest_matrix_root_m=np.asarray(bone["raw_rest_matrix_root_m"], dtype=np.float64),
                rest_matrix_root_m=np.asarray(bone["rest_matrix_root_m"], dtype=np.float64),
                rigidization_frobenius_residual=bone["rigidization_frobenius_residual"],  # type: ignore[arg-type]
            )
        )
    deform_names = manifest["deform_bone_names"]
    if not isinstance(deform_names, list):
        raise TypeError("deform_bone_names: expected array")
    return RigManifest(
        rig_id=manifest["rig_id"],  # type: ignore[arg-type]
        metric_scale=manifest["metric_scale"],  # type: ignore[arg-type]
        source_asset_sha256=manifest["source_asset_sha256"],  # type: ignore[arg-type]
        derived_rig_sha256=manifest["derived_rig_sha256"],  # type: ignore[arg-type]
        normalization_max_position_residual_m=manifest["normalization_max_position_residual_m"],  # type: ignore[arg-type]
        bones=tuple(bones),
        deform_bone_names=tuple(deform_names),
    )


def load_rig_manifest(path: str | Path) -> RigManifest:
    """Read and validate a UTF-8 CS3 rig manifest."""
    manifest_path = Path(path)
    return rig_manifest_from_mapping(json.loads(manifest_path.read_text(encoding="utf-8")))


def audit_asset_manifest(value: object) -> None:
    """Fail closed unless the manifest names exactly one selected CC0 source pack."""
    manifest = _exact_keys(
        "asset manifest",
        value,
        {
            "schema_version",
            "external_pack_count",
            "pack",
            "blender",
            "saved_license_evidence",
            "import_conversion_settings",
            "derived_artifacts",
            "allowlisted_external_members",
            "rejected_external_content",
            "authored_content_policy",
        },
    )
    if manifest["schema_version"] != ASSET_MANIFEST_SCHEMA_VERSION or manifest["external_pack_count"] != 1:
        raise ValueError("asset manifest: expected exactly one external pack")
    pack = _exact_keys(
        "asset manifest pack",
        manifest["pack"],
        {
            "name",
            "source_url",
            "displayed_pack_date",
            "acquisition_date",
            "upstream_filename",
            "archive_size_bytes",
            "archive_sha256",
            "inventory_sha256",
            "license_member",
            "license_sha256",
            "license_spdx",
            "selected_member",
            "selected_member_size_bytes",
            "selected_member_sha256",
            "selected_member_timestamp_utc",
        },
    )
    expected = {
        "name": "Quaternius Universal Base Characters",
        "source_url": "https://quaternius.com/packs/universalbasecharacters.html",
        "displayed_pack_date": "August 2025",
        "acquisition_date": "2026-07-20",
        "upstream_filename": "Universal Base Characters[Source].zip",
        "archive_sha256": SOURCE_ARCHIVE_SHA256,
        "inventory_sha256": SOURCE_INVENTORY_SHA256,
        "license_sha256": SOURCE_LICENSE_SHA256,
        "selected_member_sha256": SOURCE_BLEND_SHA256,
        "license_spdx": "CC0-1.0",
        "selected_member": "Base Characters/Regular_Male_FullBody.blend",
        "license_member": "License_Source.txt",
        "selected_member_timestamp_utc": "2025-11-14T20:30:19Z",
    }
    for key, expected_value in expected.items():
        if pack[key] != expected_value:
            raise ValueError(f"asset manifest pack.{key}: frozen value drift")
    if pack["archive_size_bytes"] != 629_923_189 or pack["selected_member_size_bytes"] != 4_454_184:
        raise ValueError("asset manifest pack: byte-size drift")
    blender = _exact_keys(
        "asset manifest blender",
        manifest["blender"],
        {"version", "archive_sha256", "binary_sha256", "build_hash", "render_device"},
    )
    expected_blender = {
        "version": "4.5.11 LTS",
        "archive_sha256": BLENDER_ARCHIVE_SHA256,
        "binary_sha256": BLENDER_BINARY_SHA256,
        "build_hash": BLENDER_BUILD_HASH,
        "render_device": "CPU",
    }
    if any(blender[key] != expected_value for key, expected_value in expected_blender.items()):
        raise ValueError("asset manifest blender: pinned build drift")
    evidence = manifest["saved_license_evidence"]
    expected_evidence = [
        {
            "id": "pack_page",
            "relative_path": "evidence/quaternius-universal-base-characters.html",
            "sha256": PACK_PAGE_EVIDENCE_SHA256,
        },
        {
            "id": "cc0_legal_code",
            "relative_path": "evidence/cc0-1.0.html",
            "sha256": CC0_PAGE_EVIDENCE_SHA256,
        },
        {
            "id": "quaternius_faq",
            "relative_path": "evidence/quaternius-faq.html",
            "sha256": FAQ_PAGE_EVIDENCE_SHA256,
        },
    ]
    if evidence != expected_evidence:
        raise ValueError("saved_license_evidence: frozen path/hash evidence drift")
    settings = _exact_keys(
        "asset manifest import_conversion_settings",
        manifest["import_conversion_settings"],
        {
            "source_operation",
            "source_objects",
            "source_transform_precondition",
            "root_normalization",
            "metric_scale",
            "rig_sanitization",
            "boot_binding_policy",
            "material_policy",
            "data_purge_policy",
            "save_policy",
            "plugins_required",
        },
    )
    expected_settings = {
        "source_operation": "append selected objects from the selected source blend",
        "source_objects": ["Armature", "RegularMale", "Eyes", "Eyebrows"],
        "source_transform_precondition": "all four appended object transforms must be identity",
        "root_normalization": "pelvis-origin body FRD metres using frozen R_root_from_source and metric_scale",
        "metric_scale": RIG_METRIC_SCALE,
        "rig_sanitization": "remove Rigify/control bones, constraints, drivers, custom properties, actions, and text blocks",
        "boot_binding_policy": "author deform boot_bind_l and boot_bind_r bones before pose evaluation",
        "material_policy": "replace source materials with authored deterministic materials",
        "data_purge_policy": "remove source scenes, worlds, cameras, lights, actions, images, texts, libraries, and unused data",
        "save_policy": "save uncompressed with relative-path remapping disabled",
        "plugins_required": 0,
    }
    if settings != expected_settings:
        raise ValueError("import_conversion_settings: frozen conversion drift")
    derived = _exact_keys(
        "asset manifest derived_artifacts",
        manifest["derived_artifacts"],
        {
            "derived_rig_sha256",
            "rig_manifest_canonical_sha256",
            "root_free_clip_canonical_sha256",
            "authoritative_pose_table_file_sha256",
            "authored_scene_sha256",
        },
    )
    for key, value in derived.items():
        _sha256(f"asset manifest derived_artifacts.{key}", value)
    if manifest["allowlisted_external_members"] != [
        "License_Source.txt",
        "Base Characters/Regular_Male_FullBody.blend",
    ]:
        raise ValueError("allowlisted_external_members: expected selected blend and license only")
    rejected = manifest["rejected_external_content"]
    if not isinstance(rejected, list) or not rejected:
        raise ValueError("rejected_external_content: expected explicit exclusions")
    if manifest["authored_content_policy"] != (
        "terrain, obstacles, equipment, materials, and animation are authored in-project"
    ):
        raise ValueError("authored_content_policy: external-content boundary drift")


def load_asset_manifest(path: str | Path) -> dict[str, object]:
    """Read and audit the tracked one-pack asset manifest."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    audit_asset_manifest(value)
    return value


__all__ = [
    "AMPLITUDE_EXCLUDED_SEMANTICS",
    "ASSET_MANIFEST_SCHEMA_VERSION",
    "BONE_BY_SEMANTIC",
    "EvaluatedBone",
    "LEFT_RIGHT_PAIRS",
    "RIG_MANIFEST_SCHEMA_VERSION",
    "RIG_METRIC_SCALE",
    "PACK_PAGE_EVIDENCE_SHA256",
    "CC0_PAGE_EVIDENCE_SHA256",
    "FAQ_PAGE_EVIDENCE_SHA256",
    "REST_FRAME_EXTRACTION_VERSION",
    "R_ROOT_FROM_SOURCE",
    "RigManifest",
    "TRACKED_SEMANTICS",
    "audit_asset_manifest",
    "load_asset_manifest",
    "load_rig_manifest",
    "rig_manifest_from_mapping",
]
