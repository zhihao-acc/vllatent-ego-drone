# SPDX-License-Identifier: GPL-3.0-or-later
"""Isolated Blender 4.5.11 bridge for the B3-CS3 feasibility proof.

Run only with ``--factory-startup --background --disable-autoexec``.  The source
blend is never saved: ``freeze-rig`` appends four allowlisted objects into an
empty scene, removes Rigify controls/scripts/external dependencies, normalizes
the armature to pelvis-origin body FRD, authors explicit boot-binding bones, and
writes a pre-animation derived rig plus its renderer-neutral manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
import traceback
import zlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllatent.sim.contracts import (  # noqa: E402
    FIXED_DT_SECONDS,
    canonical_bytes,
    default_camera_contract,
)
from vllatent.sim.labels import (  # noqa: E402
    OccluderGeometry,
    TargetGeometry,
    compute_frame_labels,
)
from vllatent.sim.pose import (  # noqa: E402
    CARVE_CYCLE_SAMPLE_COUNT,
    CARVE_CYCLE_SCHEMA_VERSION,
    INTERMEDIATE_PARENT_NAMES,
    LOCAL_BONE_TRANSFORM_SEMANTICS,
    PINNED_CANONICAL_POSE_TABLE_SHA256,
    PINNED_POSE_TABLE_EXPORT_SHA256,
    POSE_TABLE_EXPORT_SCHEMA_VERSION,
    authored_carve_cycle,
    authored_carve_schedule,
    canonical_pose_table_sha256,
    construct_pose_root,
    evaluate_pose,
    record_with_pose,
)
from vllatent.sim.rig import (  # noqa: E402
    AMPLITUDE_EXCLUDED_SEMANTICS,
    BONE_BY_SEMANTIC,
    LEFT_RIGHT_PAIRS,
    R_ROOT_FROM_SOURCE,
    REST_FRAME_EXTRACTION_VERSION,
    RIG_ID,
    RIG_MANIFEST_SCHEMA_VERSION,
    RIG_METRIC_SCALE,
    SOURCE_BLEND_SHA256,
    TRACKED_SEMANTICS,
    load_asset_manifest,
    load_rig_manifest,
    rig_manifest_from_mapping,
)
from vllatent.sim.scene import (  # noqa: E402
    CAMERA_ROOT_SCHEMA_VERSION,
    EXCLUDED_EQUIPMENT_NAMES,
    LABEL_TARGET_SET_VERSION,
    ROOT_ENVELOPE_SCHEMA_VERSION,
    ROOT_RECORD_SCHEMA_VERSION,
    SCENE_SCHEMA_VERSION,
    SLOPE_LENGTH_M,
    SLOPE_WIDTH_M,
    TARGET_OBJECT_NAMES,
    RootRecordBindings,
    build_canonical_root_record,
    canonical_renderer_contract,
    initial_camera_rig_transform,
)
from vllatent.sim.skier import (  # noqa: E402
    BINDING_HEIGHT_M,
    SKI_LENGTH_M,
    SKI_THICKNESS_M,
    SKI_WIDTH_M,
    SKIER_POSE_ROOT_SCHEMA_VERSION,
    default_slope_frame,
)
from vllatent.sim.skier_fixtures import canonical_skier_fixtures  # noqa: E402

BLENDER_VERSION = (4, 5, 11)
SOURCE_OBJECT_NAMES = ("Armature", "RegularMale", "Eyes", "Eyebrows")
DERIVED_ARMATURE_NAME = "CS3_SkierArmature"
DERIVED_TARGET_NAMES = ("CS3_Body", "CS3_Eyes", "CS3_Eyebrows")
SOURCE_TO_DERIVED_NAME = {
    "RegularMale": "CS3_Body",
    "Eyes": "CS3_Eyes",
    "Eyebrows": "CS3_Eyebrows",
}
BOOT_BIND_BONES = ("boot_bind_l", "boot_bind_r")
POSE_PARITY_POSITION_ATOL_M = 1.0e-6
POSE_PARITY_ROTATION_ATOL_RAD = 1.0e-6
STANCE_CONTACT_POSITION_ATOL_M = 0.01
SKI_FRAME_ROTATION_ATOL_RAD = math.radians(1.0)
BOOT_BINDING_ROTATION_ATOL_RAD = math.radians(2.0)
SLIP_COMPONENT_ATOL_M_S = 0.02
REALIZATION_LATERAL_CLEARANCE_MARGIN_M = 1.0e-5
FIXED_OCCLUDER_CENTER_FRACTION = 0.58
FIXED_OCCLUDER_DIMENSIONS_M = (0.40, 1.10, 1.00)
FIXED_OCCLUDER_LOCAL_Z_OFFSET_M = 0.15
SCENE_TARGET_PASS_INDEX = 1
SCENE_OCCLUDER_PASS_INDEX = 101
REPLAY_SCHEMA_VERSION = "b3-cs3-eight-root-render-replay-v2"
FRAME_SCHEMA_VERSION = "b3-cs3-rendered-frame-metadata-v2"
NEAR_CLIP_SCHEMA_VERSION = "b3-cs3-camera-near-plane-triangle-clip-v1"
PNG_SCHEMA_VERSION = "b3-cs3-lossless-png-fixed-filter-v1"
ID_PASS_SCHEMA_VERSION = "b3-cs3-blender-depsgraph-center-ray-id-v1"
HELMET_HEAD_BONE_FRACTION = 0.65
DYNAMIC_EQUIPMENT_PASS_INDICES = {
    "CS3_Binding_L": 301,
    "CS3_Binding_R": 302,
    "CS3_Pole_L": 303,
    "CS3_Pole_R": 304,
    "CS3_Ski_L": 305,
    "CS3_Ski_R": 306,
}

POSE_TABLE_TOP_LEVEL_KEYS = {
    "schema_version",
    "rig_manifest_canonical_sha256",
    "canonical_pose_table_sha256",
    "fixture_count",
    "sample_count",
    "rows",
    "carve_cycle_schema_version",
    "carve_cycle_count",
    "carve_cycle_sample_count",
    "carve_cycles",
    "carve_cycle_rows",
}
POSE_TABLE_FIXTURE_ROW_KEYS = {
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
POSE_TABLE_CYCLE_ROW_KEYS = {
    "cycle_id",
    "sign",
    "sample_index",
    "absolute_tick",
    "elapsed_seconds",
    "maneuver_phase",
    "maneuver_phase_name",
    "maneuver_weight",
    "target_curvature_1_m",
    "target_left_edge_rad",
    "target_right_edge_rad",
    "gross_lean_rad",
    "source_record_sha256",
    "source_skier_digest",
    "pose_sha256",
    "posed_record_sha256",
    "posed_skier_digest",
    "skis",
    "T_world_from_armature",
    "parent_bone_names",
    "T_root_from_parent_bone",
    "bone_names",
    "T_root_from_bone",
    "local_transform_semantics",
    "local_bone_transforms",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _camera_setup_sha256() -> str:
    return hashlib.sha256(canonical_bytes(default_camera_contract().manifest())).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _matrix_rows(matrix: Matrix | np.ndarray) -> list[list[float]]:
    array = np.asarray(matrix, dtype=np.float64)
    return [[float(array[row, column]) for column in range(4)] for row in range(4)]


def _verify_runtime() -> None:
    if tuple(bpy.app.version) != BLENDER_VERSION:
        raise RuntimeError(f"expected Blender {BLENDER_VERSION}, got {tuple(bpy.app.version)}")
    if not bpy.app.background:
        raise RuntimeError("B3-CS3 bridge must run in background mode")


def _clear_id_properties(block: Any) -> None:
    if not hasattr(block, "keys"):
        return
    for key in list(block.keys()):
        try:
            del block[key]
        except (KeyError, TypeError):
            continue


def _clear_animation(block: Any) -> None:
    if hasattr(block, "animation_data_clear"):
        block.animation_data_clear()


def _remove_blocks(collection: Any) -> None:
    for block in tuple(collection):
        collection.remove(block, do_unlink=True)


def _reset_to_empty_scene() -> bpy.types.Scene:
    _remove_blocks(bpy.data.objects)
    for collection in tuple(bpy.data.collections):
        bpy.data.collections.remove(collection)
    for scene in tuple(bpy.data.scenes)[1:]:
        bpy.data.scenes.remove(scene)
    scene = bpy.context.scene
    scene.name = "CS3_RigFreeze"
    scene.world = None
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    _remove_blocks(bpy.data.worlds)
    _remove_blocks(bpy.data.cameras)
    _remove_blocks(bpy.data.lights)
    _remove_blocks(bpy.data.libraries)
    _remove_blocks(bpy.data.actions)
    _remove_blocks(bpy.data.texts)
    return scene


def _append_source_objects(source: Path, scene: bpy.types.Scene) -> dict[str, bpy.types.Object]:
    with bpy.data.libraries.load(str(source), link=False, assets_only=False) as (data_from, data_to):
        missing = sorted(set(SOURCE_OBJECT_NAMES) - set(data_from.objects))
        if missing:
            raise RuntimeError(f"source object allowlist missing: {missing}")
        data_to.objects = list(SOURCE_OBJECT_NAMES)
    loaded = {obj.name: obj for obj in data_to.objects if obj is not None}
    if set(loaded) != set(SOURCE_OBJECT_NAMES):
        raise RuntimeError(f"appended object mismatch: {sorted(loaded)}")
    for obj in loaded.values():
        if not obj.users_collection:
            scene.collection.objects.link(obj)
    return loaded


def _identity_matrix(matrix: Matrix, *, atol: float = 1.0e-7) -> bool:
    return bool(np.allclose(np.asarray(matrix, dtype=np.float64), np.eye(4), rtol=0.0, atol=atol))


def _strip_rigify_and_controls(
    armature: bpy.types.Object,
    selected_objects: set[bpy.types.Object],
) -> tuple[str, ...]:
    for pose_bone in armature.pose.bones:
        for constraint in tuple(pose_bone.constraints):
            pose_bone.constraints.remove(constraint)
        pose_bone.custom_shape = None
        pose_bone.custom_shape_transform = None
        _clear_id_properties(pose_bone)
        _clear_id_properties(pose_bone.bone)
    _clear_id_properties(armature)
    _clear_id_properties(armature.data)
    _clear_animation(armature)
    _clear_animation(armature.data)

    keep_deform = {bone.name for bone in armature.data.bones if bone.use_deform}
    required = set(BONE_BY_SEMANTIC.values()) - set(BOOT_BIND_BONES)
    missing = sorted(required - keep_deform)
    if missing:
        raise RuntimeError(f"source deform bones missing evaluated mapping: {missing}")

    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    for edit_bone in tuple(armature.data.edit_bones):
        if edit_bone.name not in keep_deform:
            armature.data.edit_bones.remove(edit_bone)
    bpy.ops.object.mode_set(mode="OBJECT")

    for obj in tuple(bpy.data.objects):
        if obj not in selected_objects:
            bpy.data.objects.remove(obj, do_unlink=True)
    return tuple(sorted(keep_deform))


def _normalization_transform(armature: bpy.types.Object) -> Matrix:
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="EDIT")
    pelvis = armature.data.edit_bones.get("pelvis")
    if pelvis is None:
        raise RuntimeError("source pelvis bone missing")
    pelvis_head = pelvis.head.copy()
    bpy.ops.object.mode_set(mode="OBJECT")
    rotation = Matrix(tuple(tuple(float(value) for value in row) for row in R_ROOT_FROM_SOURCE)).to_4x4()
    return Matrix.Scale(RIG_METRIC_SCALE, 4) @ rotation @ Matrix.Translation(-pelvis_head)


def _author_boot_bind_bone(
    edit_bones: bpy.types.ArmatureEditBones,
    *,
    side_suffix: str,
) -> None:
    foot = edit_bones[f"foot_{side_suffix}"]
    ball = edit_bones[f"ball_{side_suffix}"]
    name = f"boot_bind_{side_suffix}"
    boot = edit_bones.new(name)
    boot.use_deform = False
    boot.use_connect = False
    boot.head = 0.5 * (foot.head + foot.tail)
    forward = ball.tail - foot.head
    if forward.length <= 1.0e-6:
        raise RuntimeError(f"{name}: degenerate foot rest direction")
    forward.normalize()
    boot.tail = boot.head + 0.25 * forward
    boot.roll = 0.0
    boot.parent = foot


def _transform_armature_and_meshes(
    armature: bpy.types.Object,
    meshes: tuple[bpy.types.Object, ...],
    transform: Matrix,
) -> float:
    source_segments = {
        bone.name: (bone.head_local.copy(), bone.tail_local.copy()) for bone in armature.data.bones
    }
    armature.data.transform(transform)
    residuals: list[float] = []
    for name, (source_head, source_tail) in source_segments.items():
        bone = armature.data.bones[name]
        expected_head = (transform @ source_head.to_4d()).to_3d()
        expected_tail = (transform @ source_tail.to_4d()).to_3d()
        residuals.extend(((bone.head_local - expected_head).length, (bone.tail_local - expected_tail).length))
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="EDIT")
    _author_boot_bind_bone(armature.data.edit_bones, side_suffix="l")
    _author_boot_bind_bone(armature.data.edit_bones, side_suffix="r")
    bpy.ops.object.mode_set(mode="OBJECT")
    for mesh_object in meshes:
        mesh_object.data.transform(transform, shape_keys=True)
        mesh_object.data.update()
    return max(residuals)


def _material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    material.node_tree.nodes.clear()
    output = material.node_tree.nodes.new("ShaderNodeOutputMaterial")
    principled = material.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
    principled.inputs["Base Color"].default_value = color
    principled.inputs["Roughness"].default_value = 0.72
    material.node_tree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    return material


def _replace_materials(meshes: tuple[bpy.types.Object, ...]) -> None:
    authored = {
        "CS3_Body": _material("CS3_SuitMaterial", (0.035, 0.11, 0.42, 1.0)),
        "CS3_Eyes": _material("CS3_EyeMaterial", (0.82, 0.88, 0.95, 1.0)),
        "CS3_Eyebrows": _material("CS3_EyebrowMaterial", (0.025, 0.018, 0.012, 1.0)),
    }
    for obj in meshes:
        obj.data.materials.clear()
        obj.data.materials.append(authored[obj.name])
        for polygon in obj.data.polygons:
            polygon.material_index = 0


def _bind_meshes(
    armature: bpy.types.Object,
    meshes: tuple[bpy.types.Object, ...],
    deform_bones: tuple[str, ...],
) -> None:
    deform_set = set(deform_bones)
    for mesh in meshes:
        for modifier in tuple(mesh.modifiers):
            mesh.modifiers.remove(modifier)
        modifier = mesh.modifiers.new(name="CS3_Armature", type="ARMATURE")
        modifier.object = armature
        modifier.use_deform_preserve_volume = True
        mesh.parent = armature
        mesh.parent_type = "OBJECT"
        mesh.matrix_parent_inverse = Matrix.Identity(4)
        mesh.matrix_basis = Matrix.Identity(4)
        for group in tuple(mesh.vertex_groups):
            if group.name not in deform_set:
                mesh.vertex_groups.remove(group)
        _clear_animation(mesh)
        _clear_animation(mesh.data)
        if mesh.data.shape_keys is not None:
            _clear_animation(mesh.data.shape_keys)
        _clear_id_properties(mesh)
        _clear_id_properties(mesh.data)
        mesh["cs3_target_part"] = True


def _purge_unapproved_data(authored_material_names: set[str]) -> None:
    _remove_blocks(bpy.data.actions)
    _remove_blocks(bpy.data.texts)
    _remove_blocks(bpy.data.images)
    _remove_blocks(bpy.data.worlds)
    _remove_blocks(bpy.data.cameras)
    _remove_blocks(bpy.data.lights)
    _remove_blocks(bpy.data.libraries)
    for material in tuple(bpy.data.materials):
        if material.name not in authored_material_names:
            bpy.data.materials.remove(material, do_unlink=True)
    for node_group in tuple(bpy.data.node_groups):
        if node_group.users == 0:
            bpy.data.node_groups.remove(node_group)
    for collection in tuple(bpy.data.collections):
        if collection.users == 0:
            bpy.data.collections.remove(collection)
    for _ in range(3):
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)


def _rigidize_rest_matrix(raw_matrix: Matrix) -> tuple[np.ndarray, float]:
    raw = np.asarray(raw_matrix, dtype=np.float64)
    rotation = raw[:3, :3]
    left, _singular, right_t = np.linalg.svd(rotation)
    closest = left @ right_t
    if float(np.linalg.det(closest)) < 0.0:
        left[:, -1] *= -1.0
        closest = left @ right_t
    rigid = np.eye(4, dtype=np.float64)
    rigid[:3, :3] = closest
    rigid[:3, 3] = raw[:3, 3]
    residual = float(np.linalg.norm(rotation - closest, ord="fro"))
    return rigid, residual


def _rotation_residual_rad(first: np.ndarray, second: np.ndarray) -> float:
    first_rigid, _ = _rigidize_rest_matrix(Matrix(first.tolist()))
    second_rigid, _ = _rigidize_rest_matrix(Matrix(second.tolist()))
    relative = first_rigid[:3, :3].T @ second_rigid[:3, :3]
    skew = np.array(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ],
        dtype=np.float64,
    )
    sine = 0.5 * float(np.linalg.norm(skew))
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.arctan2(sine, cosine))


def _pose_basis_residual(armature: bpy.types.Object) -> float:
    return max(
        float(np.linalg.norm(np.asarray(bone.matrix_basis, dtype=np.float64) - np.eye(4), ord="fro"))
        for bone in armature.pose.bones
    )


def _sanitization_snapshot(
    armature: bpy.types.Object,
    target_meshes: tuple[bpy.types.Object, ...],
) -> dict[str, int]:
    non_armature_modifiers = sum(
        modifier.type != "ARMATURE" for obj in target_meshes for modifier in obj.modifiers
    )
    valid_armature_modifiers = all(
        len(obj.modifiers) == 1 and obj.modifiers[0].type == "ARMATURE" and obj.modifiers[0].object == armature
        for obj in target_meshes
    )
    if not valid_armature_modifiers:
        raise RuntimeError("target meshes do not have exactly one valid armature modifier")
    snapshot = {
        "actions": len(bpy.data.actions),
        "external_libraries": len(bpy.data.libraries),
        "images": len(bpy.data.images),
        "non_armature_modifiers": int(non_armature_modifiers),
        "plugins_required": 0,
        "source_texts": len(bpy.data.texts),
        "source_worlds": len(bpy.data.worlds),
    }
    if any(snapshot.values()):
        raise RuntimeError(f"derived rig sanitization failed: {snapshot}")
    if _pose_basis_residual(armature) > 1.0e-12:
        raise RuntimeError("derived rig pose basis is not exact identity before pose authoring")
    return snapshot


def _build_rig_manifest(
    *,
    derived_sha256: str,
    armature: bpy.types.Object,
    deform_bones: tuple[str, ...],
    sanitization: dict[str, int],
    normalization_residual_m: float,
) -> dict[str, object]:
    bones: list[dict[str, object]] = []
    for semantic in TRACKED_SEMANTICS:
        name = BONE_BY_SEMANTIC[semantic]
        bone = armature.data.bones.get(name)
        if bone is None:
            raise RuntimeError(f"evaluated bone missing after derivation: {name}")
        if bone.parent is None:
            raise RuntimeError(f"evaluated bone has no frozen parent: {name}")
        raw = np.asarray(bone.matrix_local, dtype=np.float64)
        rigid, residual = _rigidize_rest_matrix(bone.matrix_local)
        parent_raw = np.asarray(bone.parent.matrix_local, dtype=np.float64)
        parent_rigid, parent_residual = _rigidize_rest_matrix(bone.parent.matrix_local)
        bones.append(
            {
                "semantic": semantic,
                "blender_name": name,
                "parent_name": bone.parent.name if bone.parent is not None else None,
                "parent_raw_rest_matrix_root_m": _matrix_rows(parent_raw),
                "parent_rest_matrix_root_m": _matrix_rows(parent_rigid),
                "parent_rigidization_frobenius_residual": parent_residual,
                "raw_rest_matrix_root_m": _matrix_rows(raw),
                "rest_matrix_root_m": _matrix_rows(rigid),
                "rigidization_frobenius_residual": residual,
            }
        )
    manifest: dict[str, object] = {
        "schema_version": RIG_MANIFEST_SCHEMA_VERSION,
        "rig_id": RIG_ID,
        "units": "metre",
        "metric_scale": RIG_METRIC_SCALE,
        "source_asset_sha256": SOURCE_BLEND_SHA256,
        "derived_rig_sha256": derived_sha256,
        "normalization_max_position_residual_m": normalization_residual_m,
        "source_frame": {"x": "anatomical_left", "y": "rear", "z": "up"},
        "root_local_frame": {"x": "forward", "y": "right", "z": "down", "handedness": "proper"},
        "R_root_from_source": [[float(value) for value in row] for row in R_ROOT_FROM_SOURCE],
        "rest_frame_extraction": REST_FRAME_EXTRACTION_VERSION,
        "origin_semantic": "pelvis",
        "origin_root_m": [0.0, 0.0, 0.0],
        "left_right_pairs": [list(pair) for pair in LEFT_RIGHT_PAIRS],
        "animation_amplitude_excluded": list(AMPLITUDE_EXCLUDED_SEMANTICS),
        "contact_required": ["left_ankle", "right_ankle", "left_boot", "right_boot"],
        "objects": {"armature": DERIVED_ARMATURE_NAME, "target_meshes": list(DERIVED_TARGET_NAMES)},
        "deform_bone_names": list(deform_bones),
        "bones": bones,
        "sanitization": sanitization,
    }
    rig_manifest_from_mapping(manifest)
    return manifest


def freeze_rig(args: argparse.Namespace) -> None:
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve()
    audit_path = Path(args.audit).resolve()
    if _sha256_file(source) != SOURCE_BLEND_SHA256:
        raise RuntimeError("selected source blend SHA-256 mismatch")

    scene = _reset_to_empty_scene()
    loaded = _append_source_objects(source, scene)
    armature = loaded["Armature"]
    if armature.type != "ARMATURE":
        raise RuntimeError("source Armature object has wrong type")
    meshes = tuple(loaded[name] for name in ("RegularMale", "Eyes", "Eyebrows"))
    if any(obj.type != "MESH" for obj in meshes):
        raise RuntimeError("source target object has wrong type")
    if any(not _identity_matrix(obj.matrix_world) for obj in (armature, *meshes)):
        raise RuntimeError("source object transforms must be identity before normalization")

    deform_bones = _strip_rigify_and_controls(armature, set(loaded.values()))
    transform = _normalization_transform(armature)
    normalization_residual_m = _transform_armature_and_meshes(armature, meshes, transform)
    if normalization_residual_m > 1.0e-6:
        raise RuntimeError(f"armature normalization residual too large: {normalization_residual_m}")

    armature.name = DERIVED_ARMATURE_NAME
    armature.data.name = "CS3_SkierArmatureData"
    renamed_meshes: list[bpy.types.Object] = []
    for source_name, mesh in zip(("RegularMale", "Eyes", "Eyebrows"), meshes, strict=True):
        mesh.name = SOURCE_TO_DERIVED_NAME[source_name]
        mesh.data.name = f"{mesh.name}Mesh"
        renamed_meshes.append(mesh)
    target_meshes = tuple(renamed_meshes)
    _bind_meshes(armature, target_meshes, deform_bones)
    _replace_materials(target_meshes)
    _purge_unapproved_data({"CS3_SuitMaterial", "CS3_EyeMaterial", "CS3_EyebrowMaterial"})

    armature.matrix_world = Matrix.Identity(4)
    armature["cs3_rig_id"] = RIG_ID
    scene["cs3_ready_for_pose"] = True
    scene["cs3_source_sha256"] = SOURCE_BLEND_SHA256
    scene["cs3_rest_frame_extraction"] = REST_FRAME_EXTRACTION_VERSION
    sanitization = _sanitization_snapshot(armature, target_meshes)

    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output), check_existing=False, compress=False, relative_remap=False)
    derived_sha256 = _sha256_file(output)
    manifest = _build_rig_manifest(
        derived_sha256=derived_sha256,
        armature=armature,
        deform_bones=deform_bones,
        sanitization=sanitization,
        normalization_residual_m=normalization_residual_m,
    )
    _write_json(manifest_path, manifest)
    loaded_manifest = load_rig_manifest(manifest_path)
    audit = {
        "operation": "freeze-rig",
        "ready_for_pose": True,
        "blender_version": bpy.app.version_string,
        "blender_build_hash": bpy.app.build_hash.decode("ascii"),
        "source_sha256": SOURCE_BLEND_SHA256,
        "derived_rig_sha256": derived_sha256,
        "rig_manifest_canonical_sha256": loaded_manifest.canonical_sha256(),
        "all_bone_count": len(armature.data.bones),
        "deform_bone_count": len(deform_bones),
        "evaluated_bone_count": len(TRACKED_SEMANTICS),
        "pose_basis_max_frobenius_residual": _pose_basis_residual(armature),
        "normalization_max_position_residual_m": normalization_residual_m,
        "rest_rigidization_max_frobenius_residual": max(
            bone.rigidization_frobenius_residual for bone in loaded_manifest.bones
        ),
        "objects": sorted(obj.name for obj in bpy.data.objects),
        "sanitization": sanitization,
    }
    _write_json(audit_path, audit)
    print("B3_CS3_FREEZE_RIG_OK", json.dumps(audit, sort_keys=True, separators=(",", ":")))


def audit_derived(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).resolve()
    audit_path = Path(args.audit).resolve()
    manifest = load_rig_manifest(manifest_path)
    current = Path(bpy.data.filepath).resolve()
    if not current.is_file() or _sha256_file(current) != manifest.derived_rig_sha256:
        raise RuntimeError("loaded derived rig does not match manifest SHA-256")
    armature = bpy.data.objects.get(DERIVED_ARMATURE_NAME)
    if armature is None or armature.type != "ARMATURE":
        raise RuntimeError("derived armature missing")
    target_meshes = tuple(bpy.data.objects.get(name) for name in DERIVED_TARGET_NAMES)
    if any(obj is None or obj.type != "MESH" for obj in target_meshes):
        raise RuntimeError("derived target mesh allowlist missing")
    typed_meshes = tuple(obj for obj in target_meshes if obj is not None)
    sanitization = _sanitization_snapshot(armature, typed_meshes)
    if sorted(obj.name for obj in bpy.data.objects) != sorted((DERIVED_ARMATURE_NAME, *DERIVED_TARGET_NAMES)):
        raise RuntimeError("derived object allowlist drift")
    for evaluated in manifest.bones:
        bone = armature.data.bones.get(evaluated.blender_name)
        if bone is None:
            raise RuntimeError(f"derived evaluated bone missing: {evaluated.blender_name}")
        raw = np.asarray(bone.matrix_local, dtype=np.float64)
        rigid, residual = _rigidize_rest_matrix(bone.matrix_local)
        if not np.array_equal(raw, evaluated.raw_rest_matrix_root_m):
            raise RuntimeError(f"raw rest matrix drift: {evaluated.blender_name}")
        if not np.array_equal(rigid, evaluated.rest_matrix_root_m):
            raise RuntimeError(f"rigidized rest matrix drift: {evaluated.blender_name}")
        if residual != evaluated.rigidization_frobenius_residual:
            raise RuntimeError(f"rigidization residual drift: {evaluated.blender_name}")
        if bone.parent is None or bone.parent.name != evaluated.parent_name:
            raise RuntimeError(f"evaluated parent drift: {evaluated.blender_name}")
        parent_raw = np.asarray(bone.parent.matrix_local, dtype=np.float64)
        parent_rigid, parent_residual = _rigidize_rest_matrix(bone.parent.matrix_local)
        if not np.array_equal(parent_raw, evaluated.parent_raw_rest_matrix_root_m):
            raise RuntimeError(f"raw parent rest matrix drift: {evaluated.blender_name}")
        if not np.array_equal(parent_rigid, evaluated.parent_rest_matrix_root_m):
            raise RuntimeError(f"rigidized parent rest matrix drift: {evaluated.blender_name}")
        if parent_residual != evaluated.parent_rigidization_frobenius_residual:
            raise RuntimeError(f"parent rigidization residual drift: {evaluated.blender_name}")
    audit = {
        "operation": "audit-derived",
        "ready_for_pose": bool(bpy.context.scene.get("cs3_ready_for_pose", False)),
        "derived_rig_sha256": manifest.derived_rig_sha256,
        "rig_manifest_canonical_sha256": manifest.canonical_sha256(),
        "pose_basis_max_frobenius_residual": _pose_basis_residual(armature),
        "sanitization": sanitization,
    }
    if not audit["ready_for_pose"]:
        raise RuntimeError("derived scene is not marked ready for pose")
    _write_json(audit_path, audit)
    print("B3_CS3_AUDIT_DERIVED_OK", json.dumps(audit, sort_keys=True, separators=(",", ":")))


def _apply_and_measure_pose_parity(
    armature: bpy.types.Object,
    T_world_from_armature: np.ndarray,
    parent_bone_names: tuple[str, ...],
    T_root_from_parent_bone: np.ndarray,
    bone_names: tuple[str, ...],
    T_root_from_bone: np.ndarray,
) -> tuple[dict[str, float], dict[str, str], dict[str, dict[str, float]]]:
    for pose_bone in armature.pose.bones:
        pose_bone.matrix_basis = Matrix.Identity(4)
    armature.matrix_world = Matrix(T_world_from_armature.tolist())

    expected_root_local: dict[str, np.ndarray] = {
        name: T_root_from_parent_bone[index]
        for index, name in enumerate(parent_bone_names)
    }
    expected_root_local.update(
        {
            bone_names[index]: T_root_from_bone[index]
            for index in range(len(bone_names))
        }
    )
    missing = sorted(set(expected_root_local) - set(armature.pose.bones.keys()))
    if missing:
        raise RuntimeError(f"pose audit bone closure missing: {missing}")
    ordered_names = sorted(
        expected_root_local,
        key=lambda name: (len(armature.pose.bones[name].parent_recursive), name),
    )
    for name in ordered_names:
        armature.pose.bones[name].matrix = Matrix(expected_root_local[name].tolist())
        bpy.context.view_layer.update()

    root_position_residuals: list[float] = []
    root_rotation_residuals: list[float] = []
    world_position_residuals: list[float] = []
    world_rotation_residuals: list[float] = []
    per_bone: dict[str, dict[str, float]] = {}
    armature_world = np.asarray(armature.matrix_world, dtype=np.float64)
    for name in ordered_names:
        actual_root = np.asarray(armature.pose.bones[name].matrix, dtype=np.float64)
        expected_root = expected_root_local[name]
        actual_world = armature_world @ actual_root
        expected_world = T_world_from_armature @ expected_root
        root_position = float(np.linalg.norm(actual_root[:3, 3] - expected_root[:3, 3]))
        root_rotation = _rotation_residual_rad(actual_root, expected_root)
        world_position = float(np.linalg.norm(actual_world[:3, 3] - expected_world[:3, 3]))
        world_rotation = _rotation_residual_rad(actual_world, expected_world)
        root_position_residuals.append(root_position)
        root_rotation_residuals.append(root_rotation)
        world_position_residuals.append(world_position)
        world_rotation_residuals.append(world_rotation)
        per_bone[name] = {
            "root_position_m": root_position,
            "root_rotation_rad": root_rotation,
            "world_position_m": world_position,
            "world_rotation_rad": world_rotation,
        }

    object_position = float(
        np.linalg.norm(armature_world[:3, 3] - T_world_from_armature[:3, 3])
    )
    object_rotation = _rotation_residual_rad(armature_world, T_world_from_armature)
    maxima = {
        "object_position_m": object_position,
        "object_rotation_rad": object_rotation,
        "root_position_m": max(root_position_residuals),
        "root_rotation_rad": max(root_rotation_residuals),
        "world_position_m": max(world_position_residuals),
        "world_rotation_rad": max(world_rotation_residuals),
    }
    worst = {
        metric: max(per_bone, key=lambda name: per_bone[name][metric])
        for metric in (
            "root_position_m",
            "root_rotation_rad",
            "world_position_m",
            "world_rotation_rad",
        )
    }
    if max(maxima["object_position_m"], maxima["root_position_m"], maxima["world_position_m"]) > (
        POSE_PARITY_POSITION_ATOL_M
    ):
        raise RuntimeError(f"Blender/PURE pose position parity failed: maxima={maxima}, worst={worst}")
    if max(maxima["object_rotation_rad"], maxima["root_rotation_rad"], maxima["world_rotation_rad"]) > (
        POSE_PARITY_ROTATION_ATOL_RAD
    ):
        raise RuntimeError(f"Blender/PURE pose rotation parity failed: maxima={maxima}, worst={worst}")
    return maxima, worst, per_bone


def _pose_audit_inputs(args: argparse.Namespace) -> tuple[Any, bpy.types.Object]:
    manifest = load_rig_manifest(Path(args.manifest).resolve())
    current = Path(bpy.data.filepath).resolve()
    if not current.is_file() or _sha256_file(current) != manifest.derived_rig_sha256:
        raise RuntimeError("loaded derived rig does not match manifest SHA-256")
    armature = bpy.data.objects.get(DERIVED_ARMATURE_NAME)
    if armature is None or armature.type != "ARMATURE":
        raise RuntimeError("derived armature missing")
    return manifest, armature


def _read_authoritative_pose_table(
    manifest: Any,
    pose_table_path: Path,
) -> dict[str, object]:
    """Load the exact pinned v3 fixture and actual carve-cycle export."""
    if _sha256_file(pose_table_path) != PINNED_POSE_TABLE_EXPORT_SHA256:
        raise RuntimeError("authoritative pose table file SHA-256 mismatch")
    try:
        value = json.loads(pose_table_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("authoritative pose table is unreadable") from error
    if not isinstance(value, dict) or set(value) != POSE_TABLE_TOP_LEVEL_KEYS:
        raise RuntimeError("authoritative pose table top-level schema drift")
    if value["schema_version"] != POSE_TABLE_EXPORT_SCHEMA_VERSION:
        raise RuntimeError("authoritative pose table version drift")
    if value["rig_manifest_canonical_sha256"] != manifest.canonical_sha256():
        raise RuntimeError("authoritative pose table rig-manifest mismatch")
    if value["canonical_pose_table_sha256"] != PINNED_CANONICAL_POSE_TABLE_SHA256:
        raise RuntimeError("authoritative canonical pose-table SHA-256 mismatch")
    if value["carve_cycle_schema_version"] != CARVE_CYCLE_SCHEMA_VERSION:
        raise RuntimeError("authoritative carve-cycle schema drift")
    if (
        value["fixture_count"] != 8
        or value["sample_count"] != 88
        or value["carve_cycle_count"] != 2
        or value["carve_cycle_sample_count"] != 2 * CARVE_CYCLE_SAMPLE_COUNT
    ):
        raise RuntimeError("authoritative pose table count drift")
    fixture_rows = value["rows"]
    cycle_descriptors = value["carve_cycles"]
    cycle_rows = value["carve_cycle_rows"]
    if not isinstance(fixture_rows, list) or len(fixture_rows) != 88:
        raise RuntimeError("authoritative fixture pose rows drift")
    if not isinstance(cycle_descriptors, list) or len(cycle_descriptors) != 2:
        raise RuntimeError("authoritative carve-cycle descriptor drift")
    if not isinstance(cycle_rows, list) or len(cycle_rows) != 2 * CARVE_CYCLE_SAMPLE_COUNT:
        raise RuntimeError("authoritative carve-cycle rows drift")
    return value


def _reconstruct_exported_bones_from_local(
    manifest: Any,
    parent_bone_names: tuple[str, ...],
    T_root_from_parent_bone: np.ndarray,
    bone_names: tuple[str, ...],
    local_bone_transforms: np.ndarray,
) -> np.ndarray:
    """Independently reconstruct the exported globals from digest-facing local deltas."""
    parent_globals = {
        name: T_root_from_parent_bone[index]
        for index, name in enumerate(parent_bone_names)
    }
    semantic_by_name = {bone.blender_name: bone.semantic for bone in manifest.bones}
    local_by_name = {
        name: local_bone_transforms[index]
        for index, name in enumerate(bone_names)
    }
    reconstructed: dict[str, np.ndarray] = {}
    remaining = set(bone_names)
    while remaining:
        progressed = False
        for name in bone_names:
            if name not in remaining:
                continue
            bone = manifest.bone(semantic_by_name[name])
            if bone.parent_name in parent_globals:
                parent_pose = parent_globals[bone.parent_name]
            elif bone.parent_name in reconstructed:
                parent_pose = reconstructed[bone.parent_name]
            else:
                continue
            local_rest = np.linalg.inv(bone.parent_rest_matrix_root_m) @ bone.rest_matrix_root_m
            reconstructed[name] = parent_pose @ local_by_name[name] @ local_rest
            remaining.remove(name)
            progressed = True
        if not progressed:
            raise RuntimeError("authoritative pose table local transform graph is not reconstructible")
    return np.stack([reconstructed[name] for name in bone_names]).astype(np.float64)


def _validate_cycle_descriptors(pose_table: dict[str, object]) -> None:
    descriptors = pose_table["carve_cycles"]
    assert isinstance(descriptors, list)
    for descriptor, sign in zip(descriptors, (-1, 1), strict=True):
        if not isinstance(descriptor, dict):
            raise RuntimeError("authoritative carve-cycle descriptor type drift")
        expected_id = f"authoritative-carve-{'left' if sign < 0 else 'right'}-cycle-v2"
        schedule_payload = authored_carve_schedule(sign).payload()
        expected = {
            "cycle_id": expected_id,
            "sign": sign,
            "sample_rate_hz": 1.0 / FIXED_DT_SECONDS,
            "duration_seconds": (CARVE_CYCLE_SAMPLE_COUNT - 1) * FIXED_DT_SECONDS,
            "sample_count": CARVE_CYCLE_SAMPLE_COUNT,
            "sample_ticks": list(range(CARVE_CYCLE_SAMPLE_COUNT)),
            "root_motion_policy": "fixed_position_heading_speed_root_removed_proof",
            "schedule": json.loads(json.dumps(schedule_payload, default=lambda value: value.value)),
            "schedule_sha256": hashlib.sha256(canonical_bytes(schedule_payload)).hexdigest(),
        }
        if descriptor != expected:
            raise RuntimeError("authoritative carve-cycle descriptor content drift")


def _cycle_row_export_inputs(
    row: object,
    source_record: Any,
    manifest: Any,
    *,
    sign: int,
    sample_index: int,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, tuple[str, ...], np.ndarray, np.ndarray, Any]:
    """Verify a cycle source/equipment row and reconstruct its exported posed record."""
    if not isinstance(row, dict) or set(row) != POSE_TABLE_CYCLE_ROW_KEYS:
        raise RuntimeError("authoritative carve-cycle row schema drift")
    cycle_id = f"authoritative-carve-{'left' if sign < 0 else 'right'}-cycle-v2"
    expected_scalars = {
        "cycle_id": cycle_id,
        "sign": sign,
        "sample_index": sample_index,
        "absolute_tick": source_record.state.absolute_tick,
        "elapsed_seconds": sample_index * FIXED_DT_SECONDS,
        "maneuver_phase": source_record.evaluated_maneuver.maneuver_phase,
        "maneuver_phase_name": source_record.evaluated_maneuver.phase_name,
        "maneuver_weight": source_record.evaluated_maneuver.weight,
        "target_curvature_1_m": source_record.evaluated_maneuver.targets.curvature_1_m,
        "target_left_edge_rad": source_record.evaluated_maneuver.targets.left_edge_rad,
        "target_right_edge_rad": source_record.evaluated_maneuver.targets.right_edge_rad,
        "gross_lean_rad": source_record.gross_lean_rad,
        "source_record_sha256": hashlib.sha256(source_record.canonical_bytes()).hexdigest(),
        "source_skier_digest": source_record.skier_digest(),
    }
    if any(row[key] != value for key, value in expected_scalars.items()):
        raise RuntimeError("authoritative carve-cycle source row drift")
    skis = row["skis"]
    expected_ski_keys = {
        "attack_rad",
        "edge_rad",
        "realized_attack_rad",
        "realized_edge_rad",
        "centerline_origin_world_m",
        "base_origin_world_m",
        "binding_origin_world_m",
        "contact_origin_world_m",
        "commanded_F_world_from_ski",
        "realized_F_world_from_ski",
    }
    if not isinstance(skis, dict) or set(skis) != {"left", "right"}:
        raise RuntimeError("authoritative carve-cycle ski-side schema drift")
    for side in ("left", "right"):
        exported = skis[side]
        if not isinstance(exported, dict) or set(exported) != expected_ski_keys:
            raise RuntimeError("authoritative carve-cycle ski schema drift")
        actual = getattr(source_record.skis, side)
        expected_ski = {
            "attack_rad": actual.attack_rad,
            "edge_rad": actual.edge_rad,
            "realized_attack_rad": actual.realized_attack_rad,
            "realized_edge_rad": actual.realized_edge_rad,
            "centerline_origin_world_m": actual.centerline_origin_world_m.tolist(),
            "base_origin_world_m": actual.base_origin_world_m.tolist(),
            "binding_origin_world_m": actual.binding_origin_world_m.tolist(),
            "contact_origin_world_m": actual.contact_origin_world_m.tolist(),
            "commanded_F_world_from_ski": actual.commanded_F_world_from_ski.tolist(),
            "realized_F_world_from_ski": actual.realized_F_world_from_ski.tolist(),
        }
        if exported != expected_ski:
            raise RuntimeError("authoritative carve-cycle ski construction drift")

    parent_names = tuple(row["parent_bone_names"])
    bone_names = tuple(row["bone_names"])
    if parent_names != INTERMEDIATE_PARENT_NAMES:
        raise RuntimeError("authoritative carve-cycle parent order drift")
    if bone_names != tuple(BONE_BY_SEMANTIC[semantic] for semantic in TRACKED_SEMANTICS):
        raise RuntimeError("authoritative carve-cycle bone order drift")
    if row["local_transform_semantics"] != LOCAL_BONE_TRANSFORM_SEMANTICS:
        raise RuntimeError("authoritative carve-cycle local semantics drift")
    T_world = np.asarray(row["T_world_from_armature"], dtype=np.float64)
    T_parent = np.asarray(row["T_root_from_parent_bone"], dtype=np.float64)
    T_bone = np.asarray(row["T_root_from_bone"], dtype=np.float64)
    T_local = np.asarray(row["local_bone_transforms"], dtype=np.float64)
    if (
        T_world.shape != (4, 4)
        or T_parent.shape != (len(parent_names), 4, 4)
        or T_bone.shape != (len(bone_names), 4, 4)
        or T_local.shape != (len(bone_names), 4, 4)
        or not all(np.all(np.isfinite(value)) for value in (T_world, T_parent, T_bone, T_local))
    ):
        raise RuntimeError("authoritative carve-cycle transform drift")
    reconstructed = _reconstruct_exported_bones_from_local(
        manifest, parent_names, T_parent, bone_names, T_local
    )
    if not np.allclose(reconstructed, T_bone, rtol=0.0, atol=1.0e-10):
        raise RuntimeError("authoritative carve-cycle local/global reconstruction drift")
    posed_record = _posed_record_from_export(source_record, T_world, T_bone, T_local)
    if hashlib.sha256(posed_record.canonical_bytes()).hexdigest() != row["posed_record_sha256"]:
        raise RuntimeError("authoritative carve-cycle posed record SHA-256 drift")
    if posed_record.skier_digest() != row["posed_skier_digest"]:
        raise RuntimeError("authoritative carve-cycle posed skier digest drift")
    for digest_name in ("pose_sha256", "posed_record_sha256", "posed_skier_digest"):
        digest = row[digest_name]
        if not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError(f"authoritative carve-cycle invalid {digest_name}")
    return T_world, parent_names, T_parent, bone_names, T_bone, reconstructed, posed_record


def audit_pose(args: argparse.Namespace) -> None:
    """Apply one PURE pose and audit Blender's evaluated ancestor/joint matrices."""
    audit_path = Path(args.audit).resolve()
    manifest, armature = _pose_audit_inputs(args)
    fixtures = canonical_skier_fixtures()
    fixture_index = int(args.fixture_index)
    record_index = int(args.record_index)
    if fixture_index < 0 or fixture_index >= len(fixtures):
        raise RuntimeError("fixture index out of range")
    records = fixtures[fixture_index].records()
    if record_index < 0 or record_index >= len(records):
        raise RuntimeError("record index out of range")
    source_record = records[record_index]
    pure_pose = evaluate_pose(source_record, manifest)
    posed_record = record_with_pose(source_record, pure_pose)
    maxima, worst, per_bone = _apply_and_measure_pose_parity(
        armature,
        pure_pose.T_world_from_armature,
        pure_pose.parent_bone_names,
        pure_pose.T_root_from_parent_bone,
        pure_pose.bone_names,
        pure_pose.T_root_from_bone,
    )
    audit = {
        "operation": "audit-pose",
        "blender_version": bpy.app.version_string,
        "derived_rig_sha256": manifest.derived_rig_sha256,
        "rig_manifest_canonical_sha256": manifest.canonical_sha256(),
        "canonical_pose_table_sha256": canonical_pose_table_sha256(manifest),
        "fixture_id": fixtures[fixture_index].fixture_id,
        "record_index": record_index,
        "absolute_tick": source_record.state.absolute_tick,
        "source_skier_digest": pure_pose.source_skier_digest,
        "pose_sha256": pure_pose.canonical_sha256(),
        "posed_skier_digest": posed_record.skier_digest(),
        "evaluated_parent_count": len(INTERMEDIATE_PARENT_NAMES),
        "evaluated_bone_count": len(TRACKED_SEMANTICS),
        "position_tolerance_m": POSE_PARITY_POSITION_ATOL_M,
        "rotation_tolerance_rad": POSE_PARITY_ROTATION_ATOL_RAD,
        "maxima": maxima,
        "worst": worst,
        "per_bone": per_bone,
    }
    _write_json(audit_path, audit)
    print("B3_CS3_AUDIT_POSE_OK", json.dumps(audit, sort_keys=True, separators=(",", ":")))


def audit_pose_table(args: argparse.Namespace) -> None:
    """Audit the authoritative exported CPython pose table in isolated Blender."""
    audit_path = Path(args.audit).resolve()
    manifest, armature = _pose_audit_inputs(args)
    pose_table_path = Path(args.pose_table).resolve()
    pose_table_file_sha256 = _sha256_file(pose_table_path)
    pose_table = _read_authoritative_pose_table(manifest, pose_table_path)
    rows = pose_table["rows"]
    if not isinstance(rows, list):
        raise RuntimeError("authoritative pose table rows must be a list")
    aggregate = {
        "object_position_m": 0.0,
        "object_rotation_rad": 0.0,
        "root_position_m": 0.0,
        "root_rotation_rad": 0.0,
        "world_position_m": 0.0,
        "world_rotation_rad": 0.0,
    }
    worst_sample: dict[str, str] = {}
    replay_rows: list[dict[str, object]] = []
    fixtures = canonical_skier_fixtures()
    expected_order = [
        (fixture.fixture_id, record_index, source_record.state.absolute_tick)
        for fixture in fixtures
        for record_index, source_record in enumerate(fixture.records())
    ]
    if pose_table["fixture_count"] != len(fixtures) or pose_table["sample_count"] != len(expected_order):
        raise RuntimeError("authoritative pose table count drift")
    if len(rows) != len(expected_order):
        raise RuntimeError("authoritative pose table row count drift")
    for row, expected_identity in zip(rows, expected_order, strict=True):
        if not isinstance(row, dict) or set(row) != POSE_TABLE_FIXTURE_ROW_KEYS:
            raise RuntimeError("authoritative pose table row schema drift")
        identity = (row["fixture_id"], row["record_index"], row["absolute_tick"])
        if identity != expected_identity:
            raise RuntimeError("authoritative pose table row order drift")
        for digest_name in (
            "source_record_sha256",
            "source_skier_digest",
            "pose_sha256",
            "posed_record_sha256",
            "posed_skier_digest",
        ):
            digest = row[digest_name]
            if not isinstance(digest, str) or len(digest) != 64:
                raise RuntimeError(f"authoritative pose table invalid {digest_name}")
        for canonical_name in ("pose_record_canonical_hex", "posed_record_canonical_hex"):
            canonical_hex = row[canonical_name]
            if (
                not isinstance(canonical_hex, str)
                or len(canonical_hex) % 2 != 0
                or any(character not in "0123456789abcdef" for character in canonical_hex)
            ):
                raise RuntimeError(f"authoritative pose table invalid {canonical_name}")
        parent_names = tuple(row["parent_bone_names"])
        bone_names = tuple(row["bone_names"])
        if parent_names != INTERMEDIATE_PARENT_NAMES:
            raise RuntimeError("authoritative pose table parent order drift")
        expected_bone_names = tuple(BONE_BY_SEMANTIC[semantic] for semantic in TRACKED_SEMANTICS)
        if bone_names != expected_bone_names:
            raise RuntimeError("authoritative pose table bone order drift")
        T_world = np.asarray(row["T_world_from_armature"], dtype=np.float64)
        T_parent = np.asarray(row["T_root_from_parent_bone"], dtype=np.float64)
        T_bone = np.asarray(row["T_root_from_bone"], dtype=np.float64)
        T_local = np.asarray(row["local_bone_transforms"], dtype=np.float64)
        expected_shapes = (
            (4, 4),
            (len(parent_names), 4, 4),
            (len(bone_names), 4, 4),
            (len(bone_names), 4, 4),
        )
        if (T_world.shape, T_parent.shape, T_bone.shape, T_local.shape) != expected_shapes:
            raise RuntimeError("authoritative pose table transform shape drift")
        if row["local_transform_semantics"] != LOCAL_BONE_TRANSFORM_SEMANTICS:
            raise RuntimeError("authoritative pose table local transform semantics drift")
        if not all(np.all(np.isfinite(value)) for value in (T_world, T_parent, T_bone, T_local)):
            raise RuntimeError("authoritative pose table contains nonfinite transforms")
        reconstructed_bones = _reconstruct_exported_bones_from_local(
            manifest,
            parent_names,
            T_parent,
            bone_names,
            T_local,
        )
        reconstruction_position_m = float(
            np.max(np.linalg.norm(reconstructed_bones[:, :3, 3] - T_bone[:, :3, 3], axis=1))
        )
        reconstruction_rotation_frobenius = float(
            np.max(
                np.linalg.norm(
                    reconstructed_bones[:, :3, :3] - T_bone[:, :3, :3],
                    axis=(1, 2),
                )
            )
        )
        if max(reconstruction_position_m, reconstruction_rotation_frobenius) > 1.0e-10:
            raise RuntimeError("authoritative local/global pose reconstruction mismatch")
        maxima, _worst_bones, _per_bone = _apply_and_measure_pose_parity(
            armature,
            T_world,
            parent_names,
            T_parent,
            bone_names,
            reconstructed_bones,
        )
        sample_id = f"{identity[0]}:{identity[1]}:{identity[2]}"
        for metric, value in maxima.items():
            if value > aggregate[metric]:
                aggregate[metric] = value
                worst_sample[metric] = sample_id
        replay_rows.append(
            {
                "fixture_id": identity[0],
                "record_index": identity[1],
                "absolute_tick": identity[2],
                "source_skier_digest": row["source_skier_digest"],
                "pose_sha256": row["pose_sha256"],
                "posed_skier_digest": row["posed_skier_digest"],
            }
        )
    _validate_cycle_descriptors(pose_table)
    cycle_rows_value = pose_table["carve_cycle_rows"]
    assert isinstance(cycle_rows_value, list)
    cycle_aggregate = {metric: 0.0 for metric in aggregate}
    cycle_summaries: list[dict[str, object]] = []
    flat_cycle_index = 0
    for sign in (-1, 1):
        source_cycle = authored_carve_cycle(sign)
        edge_degrees: list[float] = []
        lean_degrees: list[float] = []
        for sample_index, source_record in enumerate(source_cycle):
            row = cycle_rows_value[flat_cycle_index]
            (
                T_world,
                parent_names,
                T_parent,
                bone_names,
                _T_bone,
                reconstructed,
                _posed_record,
            ) = _cycle_row_export_inputs(
                row,
                source_record,
                manifest,
                sign=sign,
                sample_index=sample_index,
            )
            assert isinstance(row, dict)
            maxima, _worst_bones, _per_bone = _apply_and_measure_pose_parity(
                armature,
                T_world,
                parent_names,
                T_parent,
                bone_names,
                reconstructed,
            )
            for metric, value in maxima.items():
                cycle_aggregate[metric] = max(cycle_aggregate[metric], value)
            edge_degrees.append(math.degrees(float(row["target_left_edge_rad"])))
            lean_degrees.append(math.degrees(float(row["gross_lean_rad"])))
            flat_cycle_index += 1
        edge_peak_index = int(np.argmax(np.abs(edge_degrees)))
        lean_peak_index = int(np.argmax(np.abs(lean_degrees)))
        if (
            abs(edge_degrees[0]) >= 5.0
            or max(abs(value) for value in edge_degrees) < 45.0
            or abs(edge_degrees[-1]) >= 5.0
            or abs(edge_peak_index - lean_peak_index) > 1
        ):
            raise RuntimeError("authoritative carve-cycle edge/lean timing gate failed")
        cycle_summaries.append(
            {
                "sign": sign,
                "edge_deg": edge_degrees,
                "gross_lean_deg": lean_degrees,
                "edge_peak_index": edge_peak_index,
                "lean_peak_index": lean_peak_index,
            }
        )
    audit = {
        "operation": "audit-pose-table",
        "blender_version": bpy.app.version_string,
        "derived_rig_sha256": manifest.derived_rig_sha256,
        "rig_manifest_canonical_sha256": manifest.canonical_sha256(),
        "authoritative_pose_table_file_sha256": pose_table_file_sha256,
        "canonical_pose_table_sha256": pose_table["canonical_pose_table_sha256"],
        "fixture_count": len(fixtures),
        "sample_count": len(rows),
        "carve_cycle_count": 2,
        "carve_cycle_sample_count": len(cycle_rows_value),
        "evaluated_parent_count": len(INTERMEDIATE_PARENT_NAMES),
        "evaluated_bone_count": len(TRACKED_SEMANTICS),
        "position_tolerance_m": POSE_PARITY_POSITION_ATOL_M,
        "rotation_tolerance_rad": POSE_PARITY_ROTATION_ATOL_RAD,
        "maxima": aggregate,
        "carve_cycle_maxima": cycle_aggregate,
        "carve_cycles": cycle_summaries,
        "worst_sample": worst_sample,
        "rows": replay_rows,
    }
    _write_json(audit_path, audit)
    print("B3_CS3_AUDIT_POSE_TABLE_OK", json.dumps(audit, sort_keys=True, separators=(",", ":")))


def _new_mesh_object(
    scene: bpy.types.Scene,
    *,
    name: str,
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, ...]],
    material: bpy.types.Material,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"{name}Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.materials.append(material)
    mesh.update(calc_edges=True)
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    return obj


def _box_geometry(
    dimensions: tuple[float, float, float],
    *,
    z_min: float | None = None,
    z_max: float | None = None,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, ...]]]:
    x_half = 0.5 * dimensions[0]
    y_half = 0.5 * dimensions[1]
    low = -0.5 * dimensions[2] if z_min is None else z_min
    high = 0.5 * dimensions[2] if z_max is None else z_max
    vertices = [
        (-x_half, -y_half, low),
        (x_half, -y_half, low),
        (x_half, y_half, low),
        (-x_half, y_half, low),
        (-x_half, -y_half, high),
        (x_half, -y_half, high),
        (x_half, y_half, high),
        (-x_half, y_half, high),
    ]
    faces = [
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    return vertices, faces


def _matrix_from_axes(axes: np.ndarray, origin: np.ndarray) -> Matrix:
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = axes
    value[:3, 3] = origin
    return Matrix(value.tolist())


def _snow_material() -> bpy.types.Material:
    material = bpy.data.materials.new("CS3_SnowMaterial")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    coordinates = nodes.new("ShaderNodeTexCoord")
    noise = nodes.new("ShaderNodeTexNoise")
    ramp = nodes.new("ShaderNodeValToRGB")
    noise.noise_dimensions = "3D"
    noise.inputs["Scale"].default_value = 2.0
    noise.inputs["Detail"].default_value = 3.0
    noise.inputs["Roughness"].default_value = 0.55
    ramp.color_ramp.elements[0].position = 0.25
    ramp.color_ramp.elements[0].color = (0.52, 0.63, 0.76, 1.0)
    ramp.color_ramp.elements[1].position = 0.78
    ramp.color_ramp.elements[1].color = (0.98, 0.99, 1.0, 1.0)
    principled.inputs["Roughness"].default_value = 0.82
    material.node_tree.links.new(coordinates.outputs["Generated"], noise.inputs["Vector"])
    material.node_tree.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    material.node_tree.links.new(ramp.outputs["Color"], principled.inputs["Base Color"])
    material.node_tree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    material["cs3_authored_procedural"] = True
    return material


def _configure_scene_render(scene: bpy.types.Scene) -> None:
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 32
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.seed = 1729
    scene.cycles.use_animated_seed = False
    scene.cycles.use_denoising = False
    scene.render.resolution_x = 224
    scene.render.resolution_y = 224
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 15
    scene.render.film_transparent = False
    scene.render.use_file_extension = True
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 1
    if hasattr(scene.render, "use_motion_blur"):
        scene.render.use_motion_blur = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.display_settings.display_device = "sRGB"
    scene.view_settings.use_curve_mapping = False
    scene.render.dither_intensity = 0.0


def _author_scene_geometry(scene: bpy.types.Scene) -> dict[str, object]:
    slope = default_slope_frame()
    snow = _snow_material()
    rock_material = _material("CS3_RockMaterial", (0.14, 0.12, 0.11, 1.0))
    wood_material = _material("CS3_WoodMaterial", (0.20, 0.07, 0.025, 1.0))
    foliage_material = _material("CS3_FoliageMaterial", (0.025, 0.19, 0.055, 1.0))
    occluder_material = _material("CS3_OccluderMaterial", (0.42, 0.055, 0.035, 1.0))
    ski_material = _material("CS3_SkiMaterial", (0.82, 0.075, 0.025, 1.0))
    binding_material = _material("CS3_BindingMaterial", (0.055, 0.055, 0.065, 1.0))
    boot_material = _material("CS3_BootMaterial", (0.025, 0.028, 0.035, 1.0))
    helmet_material = _material("CS3_HelmetMaterial", (0.92, 0.31, 0.035, 1.0))
    pole_material = _material("CS3_PoleMaterial", (0.12, 0.13, 0.15, 1.0))

    x_min = -10.0
    x_max = x_min + SLOPE_LENGTH_M
    y_half = 0.5 * SLOPE_WIDTH_M
    slope_vertices = [
        tuple(x_min * slope.downhill_world - y_half * slope.right_world),
        tuple(x_min * slope.downhill_world + y_half * slope.right_world),
        tuple(x_max * slope.downhill_world + y_half * slope.right_world),
        tuple(x_max * slope.downhill_world - y_half * slope.right_world),
    ]
    terrain = _new_mesh_object(
        scene,
        name="CS3_Slope",
        vertices=slope_vertices,
        faces=[(0, 1, 2, 3)],
        material=snow,
    )
    terrain["cs3_authored_terrain"] = True
    terrain.pass_index = 200

    ground_axes = np.column_stack((slope.downhill_world, slope.right_world, -slope.normal_world))
    obstacle_specs = (
        ("CS3_Rock_A", np.array([12.0, -4.0, -0.35]), (1.1, 1.4, 0.7), rock_material, 201),
        ("CS3_Rock_B", np.array([28.0, 5.5, -0.45]), (1.7, 1.1, 0.9), rock_material, 202),
        ("CS3_ParallaxPole", np.array([18.0, -7.0, -1.4]), (0.16, 0.16, 2.8), wood_material, 203),
        ("CS3_TreeTrunk", np.array([37.0, 6.5, -1.8]), (0.42, 0.42, 3.6), wood_material, 204),
        ("CS3_TreeCrown", np.array([37.0, 6.5, -4.1]), (2.2, 2.2, 2.6), foliage_material, 205),
    )
    for name, slope_coordinates, dimensions, material, pass_index in obstacle_specs:
        vertices, faces = _box_geometry(dimensions)
        obj = _new_mesh_object(scene, name=name, vertices=vertices, faces=faces, material=material)
        origin = (
            slope_coordinates[0] * slope.downhill_world
            + slope_coordinates[1] * slope.right_world
            + slope_coordinates[2] * (-slope.normal_world)
        )
        obj.matrix_world = _matrix_from_axes(ground_axes, origin)
        obj["cs3_authored_obstacle"] = True
        obj.pass_index = pass_index

    fixtures = canonical_skier_fixtures()
    occlusion_fixture = next(fixture for fixture in fixtures if fixture.fixture_id == "occlusion_path")
    occlusion_records = occlusion_fixture.records()
    camera_origin = initial_camera_rig_transform(occlusion_records[2])[:3, 3]
    occlusion_target = construct_pose_root(occlusion_records[6]).pelvis_point_world_m
    occluder_center = (
        camera_origin
        + FIXED_OCCLUDER_CENTER_FRACTION * (occlusion_target - camera_origin)
        + FIXED_OCCLUDER_LOCAL_Z_OFFSET_M * (-slope.normal_world)
    )
    vertices, faces = _box_geometry(FIXED_OCCLUDER_DIMENSIONS_M)
    occluder = _new_mesh_object(
        scene,
        name="CS3_FixedOccluder",
        vertices=vertices,
        faces=faces,
        material=occluder_material,
    )
    occluder.matrix_world = _matrix_from_axes(ground_axes, occluder_center)
    occluder["cs3_authored_obstacle"] = True
    occluder["cs3_occluder_id"] = "fixed_occluder_101"
    occluder["cs3_center_fraction"] = FIXED_OCCLUDER_CENTER_FRACTION
    occluder["cs3_dimensions_m"] = list(FIXED_OCCLUDER_DIMENSIONS_M)
    occluder["cs3_local_z_offset_m"] = FIXED_OCCLUDER_LOCAL_Z_OFFSET_M
    occluder.pass_index = SCENE_OCCLUDER_PASS_INDEX

    dynamic_materials = {
        "ski": ski_material,
        "binding": binding_material,
        "boot": boot_material,
        "helmet": helmet_material,
        "pole": pole_material,
    }
    ski_vertices, ski_faces = _box_geometry(
        (SKI_LENGTH_M, SKI_WIDTH_M, SKI_THICKNESS_M),
        z_min=-SKI_THICKNESS_M,
        z_max=0.0,
    )
    boot_vertices, boot_faces = _box_geometry((0.34, 0.14, 0.18), z_min=-0.18, z_max=0.0)
    binding_vertices, binding_faces = _box_geometry((0.26, 0.12, 0.05))
    pole_vertices, pole_faces = _box_geometry((0.025, 0.025, 1.45))
    for suffix in ("L", "R"):
        ski = _new_mesh_object(
            scene,
            name=f"CS3_Ski_{suffix}",
            vertices=ski_vertices,
            faces=ski_faces,
            material=dynamic_materials["ski"],
        )
        binding = _new_mesh_object(
            scene,
            name=f"CS3_Binding_{suffix}",
            vertices=binding_vertices,
            faces=binding_faces,
            material=dynamic_materials["binding"],
        )
        boot = _new_mesh_object(
            scene,
            name=f"CS3_Boot_{suffix}",
            vertices=boot_vertices,
            faces=boot_faces,
            material=dynamic_materials["boot"],
        )
        pole = _new_mesh_object(
            scene,
            name=f"CS3_Pole_{suffix}",
            vertices=pole_vertices,
            faces=pole_faces,
            material=dynamic_materials["pole"],
        )
        for equipment in (ski, binding, pole):
            equipment["cs3_equipment_excluded_from_target"] = True
            equipment.hide_render = True
            equipment.pass_index = DYNAMIC_EQUIPMENT_PASS_INDICES[equipment.name]
        boot["cs3_target_part"] = True
        boot.pass_index = SCENE_TARGET_PASS_INDEX
        boot.hide_render = True

    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=0.17, location=(0.0, 0.0, 0.0))
    helmet = bpy.context.object
    helmet.name = "CS3_Helmet"
    helmet.data.name = "CS3_HelmetMesh"
    helmet.data.materials.append(dynamic_materials["helmet"])
    helmet["cs3_target_part"] = True
    helmet.pass_index = SCENE_TARGET_PASS_INDEX
    helmet.hide_render = True

    for name in DERIVED_TARGET_NAMES:
        target = bpy.data.objects[name]
        target["cs3_target_part"] = True
        target.pass_index = SCENE_TARGET_PASS_INDEX

    return {
        "occluder_id": "fixed_occluder_101",
        "occluder_center_world_m": [float(value) for value in occluder_center],
        "occluder_dimensions_m": list(FIXED_OCCLUDER_DIMENSIONS_M),
        "occluder_center_fraction": FIXED_OCCLUDER_CENTER_FRACTION,
        "occluder_local_z_offset_m": FIXED_OCCLUDER_LOCAL_Z_OFFSET_M,
        "authored_materials": sorted(
            material.name
            for material in (
                snow,
                rock_material,
                wood_material,
                foliage_material,
                occluder_material,
                ski_material,
                binding_material,
                boot_material,
                helmet_material,
                pole_material,
            )
        ),
    }


def _author_camera_and_lighting(scene: bpy.types.Scene) -> None:
    contract = default_camera_contract()
    camera_manifest = contract.manifest()
    rig = bpy.data.objects.new("DroneRig", None)
    rig.empty_display_type = "PLAIN_AXES"
    scene.collection.objects.link(rig)
    camera_data = bpy.data.cameras.new("CS3_CameraData")
    camera_data.type = "PERSP"
    camera_data.lens = camera_manifest["lens_mm"]
    camera_data.sensor_width = camera_manifest["sensor_width_mm"]
    camera_data.sensor_fit = camera_manifest["sensor_fit"]
    camera_data.shift_x = camera_manifest["shift_x"]
    camera_data.shift_y = camera_manifest["shift_y"]
    camera_data.clip_start = camera_manifest["clip_start_m"]
    camera_data.clip_end = camera_manifest["clip_end_m"]
    camera_data.dof.use_dof = camera_manifest["depth_of_field_enabled"]
    camera = bpy.data.objects.new("CS3_Camera", camera_data)
    scene.collection.objects.link(camera)
    camera.parent = rig
    camera.matrix_parent_inverse = Matrix.Identity(4)
    T_rig_from_cam = np.eye(4, dtype=np.float64)
    T_rig_from_cam[:3, :3] = contract.R_rig_from_cam
    camera.matrix_basis = Matrix(T_rig_from_cam.tolist())
    rig.matrix_world = Matrix.Identity(4)
    scene.camera = camera

    world = bpy.data.worlds.new("CS3_World")
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.34, 0.46, 0.64, 1.0)
    background.inputs["Strength"].default_value = 0.35
    scene.world = world
    sun_data = bpy.data.lights.new("CS3_SunData", type="SUN")
    sun_data.energy = 3.0
    sun_data.angle = math.radians(4.0)
    sun = bpy.data.objects.new("CS3_Sun", sun_data)
    scene.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(28.0), math.radians(-18.0), math.radians(24.0))


def build_scene(args: argparse.Namespace) -> None:
    rig_manifest = load_rig_manifest(Path(args.manifest).resolve())
    asset_path = Path(args.asset_manifest).resolve()
    asset_manifest = load_asset_manifest(asset_path)
    current = Path(bpy.data.filepath).resolve()
    if not current.is_file() or _sha256_file(current) != rig_manifest.derived_rig_sha256:
        raise RuntimeError("build-scene must load the exact pinned derived rig")
    scene = bpy.context.scene
    scene.name = "CS3_CausalSkiScene"
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    _configure_scene_render(scene)
    geometry = _author_scene_geometry(scene)
    _author_camera_and_lighting(scene)
    scene["cs3_scene_schema_version"] = SCENE_SCHEMA_VERSION
    scene["cs3_camera_root_schema_version"] = CAMERA_ROOT_SCHEMA_VERSION
    scene["cs3_label_target_set_version"] = LABEL_TARGET_SET_VERSION
    scene["cs3_camera_contract_sha256"] = _camera_setup_sha256()
    scene["cs3_external_pack_count"] = int(asset_manifest["external_pack_count"])

    target_names = tuple(sorted(obj.name for obj in bpy.data.objects if bool(obj.get("cs3_target_part", False))))
    if target_names != tuple(sorted(TARGET_OBJECT_NAMES)):
        raise RuntimeError(f"scene target object set drift: {target_names}")
    equipment_names = tuple(
        sorted(obj.name for obj in bpy.data.objects if bool(obj.get("cs3_equipment_excluded_from_target", False)))
    )
    if equipment_names != tuple(sorted(EXCLUDED_EQUIPMENT_NAMES)):
        raise RuntimeError(f"scene excluded-equipment set drift: {equipment_names}")
    if len(bpy.data.actions) or len(bpy.data.texts) or len(bpy.data.images) or len(bpy.data.libraries):
        raise RuntimeError("authored scene retained unapproved executable/external data")

    output = Path(args.output).resolve()
    scene_manifest_path = Path(args.scene_manifest).resolve()
    audit_path = Path(args.audit).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output), check_existing=False, compress=False, relative_remap=False)
    scene_sha256 = _sha256_file(output)
    slope = default_slope_frame()
    object_rows = [
        {
            "name": obj.name,
            "type": obj.type,
            "target_part": bool(obj.get("cs3_target_part", False)),
            "excluded_equipment": bool(obj.get("cs3_equipment_excluded_from_target", False)),
            "pass_index": int(obj.pass_index),
        }
        for obj in sorted(bpy.data.objects, key=lambda item: item.name)
    ]
    scene_manifest = {
        "schema_version": SCENE_SCHEMA_VERSION,
        "scene_sha256": scene_sha256,
        "derived_rig_sha256": rig_manifest.derived_rig_sha256,
        "rig_manifest_canonical_sha256": rig_manifest.canonical_sha256(),
        "asset_manifest_file_sha256": _sha256_file(asset_path),
        "external_pack_count": asset_manifest["external_pack_count"],
        "authored_content_policy": asset_manifest["authored_content_policy"],
        "blender_version": bpy.app.version_string,
        "blender_build_hash": bpy.app.build_hash.decode("ascii"),
        "scene_units": "metre",
        "slope": {
            "angle_deg": 15.0,
            "length_m": SLOPE_LENGTH_M,
            "width_m": SLOPE_WIDTH_M,
            "downhill_world": [float(value) for value in slope.downhill_world],
            "right_world": [float(value) for value in slope.right_world],
            "normal_world": [float(value) for value in slope.normal_world],
        },
        "render": {
            "engine": "CYCLES",
            "device": "CPU",
            "resolution": [224, 224],
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
        },
        "camera_root_schema_version": CAMERA_ROOT_SCHEMA_VERSION,
        "camera_contract_sha256": _camera_setup_sha256(),
        "camera_objects": {"rig": "DroneRig", "camera": "CS3_Camera"},
        "label_target_set_version": LABEL_TARGET_SET_VERSION,
        "target_objects": list(TARGET_OBJECT_NAMES),
        "excluded_equipment": list(EXCLUDED_EQUIPMENT_NAMES),
        "dynamic_equipment_pass_indices": DYNAMIC_EQUIPMENT_PASS_INDICES,
        "helmet_head_bone_fraction": HELMET_HEAD_BONE_FRACTION,
        "occluder": {
            "object": "CS3_FixedOccluder",
            "id": geometry["occluder_id"],
            "center_world_m": geometry["occluder_center_world_m"],
            "dimensions_m": geometry["occluder_dimensions_m"],
            "center_fraction": geometry["occluder_center_fraction"],
            "local_z_offset_m": geometry["occluder_local_z_offset_m"],
        },
        "authored_materials": geometry["authored_materials"],
        "objects": object_rows,
        "plugins_required": 0,
        "actions": len(bpy.data.actions),
        "texts": len(bpy.data.texts),
        "images": len(bpy.data.images),
        "external_libraries": len(bpy.data.libraries),
    }
    _write_json(scene_manifest_path, scene_manifest)
    audit = {
        "operation": "build-scene",
        "scene_sha256": scene_sha256,
        "object_count": len(object_rows),
        "target_objects": list(target_names),
        "excluded_equipment": list(equipment_names),
        "camera_contract_sha256": _camera_setup_sha256(),
        "external_pack_count": asset_manifest["external_pack_count"],
        "render": scene_manifest["render"],
    }
    _write_json(audit_path, audit)
    print("B3_CS3_BUILD_SCENE_OK", json.dumps(audit, sort_keys=True, separators=(",", ":")))


def audit_scene(args: argparse.Namespace) -> None:
    scene_manifest_path = Path(args.scene_manifest).resolve()
    audit_path = Path(args.audit).resolve()
    scene_manifest = json.loads(scene_manifest_path.read_text(encoding="utf-8"))
    current = Path(bpy.data.filepath).resolve()
    if _sha256_file(current) != scene_manifest["scene_sha256"]:
        raise RuntimeError("loaded scene SHA-256 mismatch")
    scene = bpy.context.scene
    if scene.get("cs3_scene_schema_version") != SCENE_SCHEMA_VERSION:
        raise RuntimeError("scene schema marker drift")
    if scene.render.engine != "CYCLES" or scene.cycles.device != "CPU":
        raise RuntimeError("scene CPU Cycles contract drift")
    settings = (
        scene.render.resolution_x,
        scene.render.resolution_y,
        scene.cycles.samples,
        scene.cycles.use_adaptive_sampling,
        scene.cycles.seed,
        scene.cycles.use_animated_seed,
        scene.cycles.use_denoising,
        scene.render.threads,
    )
    if settings != (224, 224, 32, False, 1729, False, False, 1):
        raise RuntimeError(f"scene render settings drift: {settings}")
    color_settings = (
        scene.view_settings.view_transform,
        scene.view_settings.look,
        scene.view_settings.exposure,
        scene.view_settings.gamma,
        scene.render.dither_intensity,
    )
    if color_settings != ("Standard", "None", 0.0, 1.0, 0.0):
        raise RuntimeError(f"scene color settings drift: {color_settings}")
    target_names = tuple(sorted(obj.name for obj in bpy.data.objects if bool(obj.get("cs3_target_part", False))))
    equipment_names = tuple(
        sorted(obj.name for obj in bpy.data.objects if bool(obj.get("cs3_equipment_excluded_from_target", False)))
    )
    if target_names != tuple(sorted(TARGET_OBJECT_NAMES)):
        raise RuntimeError("fresh scene target set drift")
    if equipment_names != tuple(sorted(EXCLUDED_EQUIPMENT_NAMES)):
        raise RuntimeError("fresh scene equipment exclusion drift")
    non_target_indices = [
        int(obj.pass_index)
        for obj in bpy.data.objects
        if obj.type == "MESH" and obj.name not in TARGET_OBJECT_NAMES
    ]
    if 0 in non_target_indices or len(non_target_indices) != len(set(non_target_indices)):
        raise RuntimeError("fresh scene non-target object-index allowlist drift")
    current_objects = [obj.name for obj in sorted(bpy.data.objects, key=lambda item: item.name)]
    manifested_objects = [row["name"] for row in scene_manifest["objects"]]
    if current_objects != manifested_objects:
        raise RuntimeError("fresh scene object allowlist drift")
    if len(bpy.data.actions) or len(bpy.data.texts) or len(bpy.data.images) or len(bpy.data.libraries):
        raise RuntimeError("fresh scene executable/external-data isolation drift")
    camera = bpy.data.objects.get("CS3_Camera")
    rig = bpy.data.objects.get("DroneRig")
    if camera is None or rig is None or camera.parent != rig or scene.camera != camera:
        raise RuntimeError("fresh scene camera hierarchy drift")
    contract = default_camera_contract()
    camera_manifest = contract.manifest()
    camera_settings = (
        camera.data.type,
        camera.data.lens,
        camera.data.sensor_width,
        camera.data.sensor_fit,
        camera.data.shift_x,
        camera.data.shift_y,
        camera.data.clip_start,
        camera.data.clip_end,
        camera.data.dof.use_dof,
    )
    expected_camera_settings = (
        "PERSP",
        camera_manifest["lens_mm"],
        camera_manifest["sensor_width_mm"],
        camera_manifest["sensor_fit"],
        camera_manifest["shift_x"],
        camera_manifest["shift_y"],
        camera_manifest["clip_start_m"],
        camera_manifest["clip_end_m"],
        camera_manifest["depth_of_field_enabled"],
    )
    if camera_settings[:4] != expected_camera_settings[:4] or camera_settings[8] is not expected_camera_settings[8]:
        raise RuntimeError(f"fresh scene camera contract drift: {camera_settings}")
    if not np.allclose(
        np.asarray(camera_settings[4:8], dtype=np.float64),
        np.asarray(expected_camera_settings[4:8], dtype=np.float64),
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise RuntimeError(f"fresh scene camera numeric contract drift: {camera_settings}")
    expected_local = np.eye(4, dtype=np.float64)
    expected_local[:3, :3] = contract.R_rig_from_cam
    if not np.allclose(
        np.asarray(camera.matrix_basis, dtype=np.float64),
        expected_local,
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise RuntimeError("fresh scene co-located camera transform drift")
    terrain = bpy.data.objects.get("CS3_Slope")
    occluder = bpy.data.objects.get("CS3_FixedOccluder")
    if terrain is None or not bool(terrain.get("cs3_authored_terrain", False)):
        raise RuntimeError("fresh scene authored slope drift")
    if occluder is None or occluder.get("cs3_occluder_id") != "fixed_occluder_101":
        raise RuntimeError("fresh scene occluder identity drift")
    occluder_manifest = scene_manifest.get("occluder")
    if not isinstance(occluder_manifest, dict):
        raise RuntimeError("fresh scene occluder manifest drift")
    occluder_vertices = np.asarray(
        [tuple(vertex.co) for vertex in occluder.data.vertices],
        dtype=np.float64,
    )
    occluder_dimensions = np.ptp(occluder_vertices, axis=0)
    if not np.allclose(
        occluder_dimensions,
        np.asarray(FIXED_OCCLUDER_DIMENSIONS_M, dtype=np.float64),
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise RuntimeError("fresh scene occluder dimensions drift")
    if not np.allclose(
        np.asarray(occluder.matrix_world.translation, dtype=np.float64),
        np.asarray(occluder_manifest.get("center_world_m"), dtype=np.float64),
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise RuntimeError("fresh scene occluder center drift")
    if (
        list(occluder_manifest.get("dimensions_m", ()))
        != list(FIXED_OCCLUDER_DIMENSIONS_M)
        or occluder_manifest.get("center_fraction")
        != FIXED_OCCLUDER_CENTER_FRACTION
        or occluder_manifest.get("local_z_offset_m")
        != FIXED_OCCLUDER_LOCAL_Z_OFFSET_M
        or float(occluder.get("cs3_center_fraction", math.nan))
        != FIXED_OCCLUDER_CENTER_FRACTION
        or list(occluder.get("cs3_dimensions_m", ()))
        != list(FIXED_OCCLUDER_DIMENSIONS_M)
        or float(occluder.get("cs3_local_z_offset_m", math.nan))
        != FIXED_OCCLUDER_LOCAL_Z_OFFSET_M
    ):
        raise RuntimeError("fresh scene occluder construction contract drift")
    snow = bpy.data.materials.get("CS3_SnowMaterial")
    if snow is None or not bool(snow.get("cs3_authored_procedural", False)):
        raise RuntimeError("fresh scene procedural snow drift")
    audit = {
        "operation": "audit-scene",
        "scene_sha256": scene_manifest["scene_sha256"],
        "scene_manifest_file_sha256": _sha256_file(scene_manifest_path),
        "object_count": len(current_objects),
        "target_objects": list(target_names),
        "excluded_equipment": list(equipment_names),
        "camera_contract_sha256": scene.get("cs3_camera_contract_sha256"),
        "render_settings": list(settings),
        "color_settings": list(color_settings),
        "camera_settings": list(camera_settings),
        "actions": len(bpy.data.actions),
        "texts": len(bpy.data.texts),
        "images": len(bpy.data.images),
        "external_libraries": len(bpy.data.libraries),
    }
    _write_json(audit_path, audit)
    print("B3_CS3_AUDIT_SCENE_OK", json.dumps(audit, sort_keys=True, separators=(",", ":")))


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


def _load_render_pose_rows(manifest: Any, pose_table_path: Path) -> list[dict[str, object]]:
    pose_table = _read_authoritative_pose_table(manifest, pose_table_path)
    rows = pose_table["rows"]
    if not isinstance(rows, list) or len(rows) != 88:
        raise RuntimeError("render pose-table rows drift")
    if any(not isinstance(row, dict) or set(row) != POSE_TABLE_FIXTURE_ROW_KEYS for row in rows):
        raise RuntimeError("render pose-table fixture row schema drift")
    return rows


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    body = chunk_type + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def _write_u8_png(path: Path, pixels: np.ndarray) -> None:
    if not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8 or pixels.ndim != 2:
        raise TypeError("PNG pixels must be a uint8 HxW array")
    height, width = pixels.shape
    if width <= 0 or height <= 0:
        raise ValueError("PNG pixels must have positive dimensions")
    raw = b"".join(
        b"\x00" + np.ascontiguousarray(pixels[row]).tobytes()
        for row in range(height)
    )
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    encoded = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(raw, level=9))
        + _png_chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def _write_rgb_png(path: Path, pixels: np.ndarray) -> None:
    if (
        not isinstance(pixels, np.ndarray)
        or pixels.dtype != np.uint8
        or pixels.ndim != 3
        or pixels.shape[2] != 3
    ):
        raise TypeError("RGB PNG pixels must be a uint8 HxWx3 array")
    height, width, _channels = pixels.shape
    raw = b"".join(
        b"\x00" + np.ascontiguousarray(pixels[row]).tobytes()
        for row in range(height)
    )
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    encoded = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(raw, level=9))
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(encoded)


def _paeth_predictor(left: int, above: int, upper_left: int) -> int:
    estimate = left + above - upper_left
    left_distance = abs(estimate - left)
    above_distance = abs(estimate - above)
    upper_left_distance = abs(estimate - upper_left)
    if left_distance <= above_distance and left_distance <= upper_left_distance:
        return left
    if above_distance <= upper_left_distance:
        return above
    return upper_left


def _decode_png_u8(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"invalid PNG signature: {path}")
    offset = 8
    header: tuple[int, int, int, int, int, int, int] | None = None
    compressed = bytearray()
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack(">I", data[offset + 8 + length : offset + 12 + length])[0]
        if zlib.crc32(chunk_type + payload) & 0xFFFFFFFF != expected_crc:
            raise RuntimeError(f"PNG CRC mismatch: {path}")
        if chunk_type == b"IHDR":
            header = struct.unpack(">IIBBBBB", payload)
        elif chunk_type == b"IDAT":
            compressed.extend(payload)
        elif chunk_type == b"IEND":
            break
        offset += 12 + length
    if header is None:
        raise RuntimeError(f"PNG missing IHDR: {path}")
    width, height, bit_depth, color_type, compression, filtering, interlace = header
    if (bit_depth, compression, filtering, interlace) != (8, 0, 0, 0):
        raise RuntimeError(f"unsupported PNG format: {path}")
    channels = {0: 1, 2: 3}.get(color_type)
    if channels is None:
        raise RuntimeError(f"unsupported PNG color type: {path}")
    row_size = width * channels
    raw = zlib.decompress(bytes(compressed))
    if len(raw) != height * (row_size + 1):
        raise RuntimeError(f"PNG scanline size mismatch: {path}")
    decoded = np.empty((height, row_size), dtype=np.uint8)
    previous = np.zeros(row_size, dtype=np.uint8)
    cursor = 0
    for row_index in range(height):
        filter_type = raw[cursor]
        cursor += 1
        filtered = raw[cursor : cursor + row_size]
        cursor += row_size
        row = np.empty(row_size, dtype=np.uint8)
        for index, encoded_value in enumerate(filtered):
            left = int(row[index - channels]) if index >= channels else 0
            above = int(previous[index])
            upper_left = int(previous[index - channels]) if index >= channels else 0
            if filter_type == 0:
                predictor = 0
            elif filter_type == 1:
                predictor = left
            elif filter_type == 2:
                predictor = above
            elif filter_type == 3:
                predictor = (left + above) // 2
            elif filter_type == 4:
                predictor = _paeth_predictor(left, above, upper_left)
            else:
                raise RuntimeError(f"unsupported PNG filter {filter_type}: {path}")
            row[index] = (encoded_value + predictor) & 0xFF
        decoded[row_index] = row
        previous = row
    shape = (height, width) if channels == 1 else (height, width, channels)
    return decoded.reshape(shape)


def _png_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise RuntimeError(f"invalid PNG output: {path}")
    return struct.unpack(">II", data[16:24])


def _object_transform(rotation: np.ndarray, origin: np.ndarray) -> Matrix:
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = rotation
    value[:3, 3] = origin
    return Matrix(value.tolist())


def _pole_transform(hand_world: np.ndarray, slope_normal: np.ndarray) -> Matrix:
    length_m = 1.45
    direction = -slope_normal
    z_axis = direction / np.linalg.norm(direction)
    reference = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(reference, z_axis))) > 0.9:
        reference = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = np.cross(reference, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    center = hand_world + 0.5 * length_m * direction
    return _object_transform(rotation, center)


def _place_render_equipment(
    record: Any,
    T_world_from_armature: np.ndarray,
    bone_names: tuple[str, ...],
    T_root_from_bone: np.ndarray,
) -> dict[str, float]:
    for suffix, side in (("L", "left"), ("R", "right")):
        ski = getattr(record.skis, side)
        frame = ski.realized_F_world_from_ski
        side_sign = -1.0 if side == "left" else 1.0
        realization_offset = (
            side_sign
            * REALIZATION_LATERAL_CLEARANCE_MARGIN_M
            * record.root.lateral_world
        )
        bpy.data.objects[f"CS3_Ski_{suffix}"].matrix_world = _object_transform(
            frame, ski.base_origin_world_m + realization_offset
        )
        bpy.data.objects[f"CS3_Binding_{suffix}"].matrix_world = _object_transform(
            frame, ski.binding_origin_world_m + realization_offset
        )
        bpy.data.objects[f"CS3_Boot_{suffix}"].matrix_world = _object_transform(
            frame, ski.binding_origin_world_m + realization_offset
        )

    index_by_name = {name: index for index, name in enumerate(bone_names)}
    head_bone = bpy.data.objects[DERIVED_ARMATURE_NAME].pose.bones["Head"]
    head_base_world = T_world_from_armature @ np.append(
        np.asarray(head_bone.head, dtype=np.float64), 1.0
    )
    head_tip_world = T_world_from_armature @ np.append(
        np.asarray(head_bone.tail, dtype=np.float64), 1.0
    )
    helmet_center = head_base_world[:3] + HELMET_HEAD_BONE_FRACTION * (
        head_tip_world[:3] - head_base_world[:3]
    )
    head_world = T_world_from_armature @ T_root_from_bone[index_by_name["Head"]]
    bpy.data.objects["CS3_Helmet"].matrix_world = _object_transform(
        head_world[:3, :3], helmet_center
    )
    slope = default_slope_frame()
    for suffix, hand_name in (("L", "hand_l"), ("R", "hand_r")):
        hand_world = T_world_from_armature @ T_root_from_bone[index_by_name[hand_name]]
        bpy.data.objects[f"CS3_Pole_{suffix}"].matrix_world = _pole_transform(
            hand_world[:3, 3], slope.normal_world
        )
    for name in (*EXCLUDED_EQUIPMENT_NAMES, "CS3_Boot_L", "CS3_Boot_R", "CS3_Helmet"):
        bpy.data.objects[name].hide_render = False
    bpy.context.view_layer.update()

    origin_residuals: list[float] = []
    frame_residuals: list[float] = []
    contact_residuals: list[float] = []
    boot_binding_position: list[float] = []
    boot_binding_rotation: list[float] = []
    binding_relative_position: list[float] = []
    binding_relative_rotation: list[float] = []
    attack_residuals: list[float] = []
    edge_residuals: list[float] = []
    slip_residuals: list[float] = []
    ski_matrices: dict[str, np.ndarray] = {}
    for suffix, side in (("L", "left"), ("R", "right")):
        ski = getattr(record.skis, side)
        ski_matrix = np.asarray(
            bpy.data.objects[f"CS3_Ski_{suffix}"].matrix_world,
            dtype=np.float64,
        )
        binding_matrix = np.asarray(
            bpy.data.objects[f"CS3_Binding_{suffix}"].matrix_world,
            dtype=np.float64,
        )
        boot_matrix = np.asarray(
            bpy.data.objects[f"CS3_Boot_{suffix}"].matrix_world,
            dtype=np.float64,
        )
        ski_matrices[side] = ski_matrix
        origin_residuals.extend(
            [
                float(np.linalg.norm(ski_matrix[:3, 3] - ski.base_origin_world_m)),
                float(np.linalg.norm(binding_matrix[:3, 3] - ski.binding_origin_world_m)),
                float(np.linalg.norm(boot_matrix[:3, 3] - ski.binding_origin_world_m)),
            ]
        )
        expected_ski_frame = np.eye(4, dtype=np.float64)
        expected_ski_frame[:3, :3] = ski.realized_F_world_from_ski
        frame_residuals.append(_rotation_residual_rad(ski_matrix, expected_ski_frame))
        sigma = 0.0 if ski.edge_rad == 0.0 else math.copysign(1.0, ski.edge_rad)
        realized_contact = (
            ski_matrix[:3, 3] + sigma * SKI_WIDTH_M * 0.5 * ski_matrix[:3, 1]
        )
        contact_residuals.append(
            float(np.linalg.norm(realized_contact - ski.contact_origin_world_m))
        )
        boot_binding_position.append(
            float(np.linalg.norm(boot_matrix[:3, 3] - binding_matrix[:3, 3]))
        )
        boot_binding_rotation.append(_rotation_residual_rad(boot_matrix, binding_matrix))
        binding_relative = np.linalg.inv(ski_matrix) @ binding_matrix
        binding_relative_position.append(
            float(
                np.linalg.norm(
                    binding_relative[:3, 3]
                    - np.asarray([0.0, 0.0, -BINDING_HEIGHT_M], dtype=np.float64)
                )
            )
        )
        binding_relative_rotation.append(
            _rotation_residual_rad(binding_relative, np.eye(4, dtype=np.float64))
        )
        forward = ski_matrix[:3, 0]
        actual_attack = math.atan2(
            float(np.dot(forward, record.root.lateral_world)),
            float(np.dot(forward, record.root.tangent_world)),
        )
        zero_edge_right = (
            -math.sin(actual_attack) * record.root.tangent_world
            + math.cos(actual_attack) * record.root.lateral_world
        )
        actual_edge = math.atan2(
            -float(np.dot(ski_matrix[:3, 1], default_slope_frame().normal_world)),
            float(np.dot(ski_matrix[:3, 1], zero_edge_right)),
        )
        attack_residuals.append(abs(actual_attack - ski.attack_rad))
        edge_residuals.append(abs(actual_edge - ski.edge_rad))
        actual_slip = np.asarray(
            [
                np.dot(record.world_velocity_m_s, forward),
                np.dot(record.world_velocity_m_s, zero_edge_right),
            ],
            dtype=np.float64,
        )
        slip_residuals.append(
            float(np.max(np.abs(actual_slip - ski.realized_slip_longitudinal_lateral_m_s)))
        )
    left_tip = (
        ski_matrices["left"][:3, 3]
        + SKI_LENGTH_M * 0.5 * ski_matrices["left"][:3, 0]
    )
    right_tip = (
        ski_matrices["right"][:3, 3]
        + SKI_LENGTH_M * 0.5 * ski_matrices["right"][:3, 0]
    )
    actual_tip_gap = (
        float(np.dot(right_tip - left_tip, record.root.lateral_world)) - SKI_WIDTH_M
    )
    return {
        "max_equipment_origin_residual_m": max(origin_residuals),
        "max_ski_frame_residual_rad": max(frame_residuals),
        "max_contact_origin_residual_m": max(contact_residuals),
        "inner_tip_gap_residual_m": abs(actual_tip_gap - record.skis.inner_tip_gap_m),
        "inner_tip_gap_m": actual_tip_gap,
        "max_boot_binding_position_m": max(boot_binding_position),
        "max_boot_binding_rotation_rad": max(boot_binding_rotation),
        "max_binding_relative_position_residual_m": max(binding_relative_position),
        "max_binding_relative_rotation_residual_rad": max(binding_relative_rotation),
        "max_attack_residual_rad": max(attack_residuals),
        "max_edge_residual_rad": max(edge_residuals),
        "max_slip_component_residual_m_s": max(slip_residuals),
    }


def _validate_equipment_metrics(equipment: dict[str, float]) -> None:
    position_limits = {
        "max_equipment_origin_residual_m": STANCE_CONTACT_POSITION_ATOL_M,
        "max_contact_origin_residual_m": STANCE_CONTACT_POSITION_ATOL_M,
        "inner_tip_gap_residual_m": STANCE_CONTACT_POSITION_ATOL_M,
        "max_boot_binding_position_m": STANCE_CONTACT_POSITION_ATOL_M,
        "max_binding_relative_position_residual_m": STANCE_CONTACT_POSITION_ATOL_M,
        "max_slip_component_residual_m_s": SLIP_COMPONENT_ATOL_M_S,
    }
    for key, limit in position_limits.items():
        if equipment[key] > limit:
            raise RuntimeError(f"render equipment metric failed: {key}={equipment[key]}")
    for key in (
        "max_ski_frame_residual_rad",
        "max_binding_relative_rotation_residual_rad",
        "max_attack_residual_rad",
        "max_edge_residual_rad",
    ):
        if equipment[key] > SKI_FRAME_ROTATION_ATOL_RAD:
            raise RuntimeError(f"render equipment rotation metric failed: {key}={equipment[key]}")
    if equipment["max_boot_binding_rotation_rad"] > BOOT_BINDING_ROTATION_ATOL_RAD:
        raise RuntimeError(
            "render equipment rotation metric failed: "
            f"max_boot_binding_rotation_rad={equipment['max_boot_binding_rotation_rad']}"
        )
    if equipment["inner_tip_gap_m"] < 0.05:
        raise RuntimeError(
            "render realized inner-tip clearance failed: "
            f"{equipment['inner_tip_gap_m']}"
        )


def audit_carve_cycle(args: argparse.Namespace) -> None:
    """Apply all exported cycle poses/equipment in the authored scene and gate timing."""
    audit_path = Path(args.audit).resolve()
    rig_manifest = load_rig_manifest(Path(args.manifest).resolve())
    scene_manifest_path = Path(args.scene_manifest).resolve()
    scene_manifest = json.loads(scene_manifest_path.read_text(encoding="utf-8"))
    current = Path(bpy.data.filepath).resolve()
    if _sha256_file(current) != scene_manifest.get("scene_sha256"):
        raise RuntimeError("carve-cycle audit loaded scene SHA-256 mismatch")
    if scene_manifest.get("derived_rig_sha256") != rig_manifest.derived_rig_sha256:
        raise RuntimeError("carve-cycle audit scene/rig mismatch")
    pose_table_path = Path(args.pose_table).resolve()
    pose_table = _read_authoritative_pose_table(rig_manifest, pose_table_path)
    _validate_cycle_descriptors(pose_table)
    rows = pose_table["carve_cycle_rows"]
    assert isinstance(rows, list)
    armature = bpy.data.objects.get(DERIVED_ARMATURE_NAME)
    if armature is None or armature.type != "ARMATURE":
        raise RuntimeError("carve-cycle audit armature missing")
    if any(bpy.data.objects.get(name) is None for name in EXCLUDED_EQUIPMENT_NAMES):
        raise RuntimeError("carve-cycle audit equipment set incomplete")

    metric_maxima: dict[str, float] = {}
    minimum_tip_gap_m = math.inf
    cycles: list[dict[str, object]] = []
    flat_index = 0
    for sign in (-1, 1):
        edge_degrees: list[float] = []
        lean_degrees: list[float] = []
        source_records = authored_carve_cycle(sign)
        for sample_index, source_record in enumerate(source_records):
            row = rows[flat_index]
            (
                T_world,
                parent_names,
                T_parent,
                bone_names,
                _T_bone,
                reconstructed,
                _posed_record,
            ) = _cycle_row_export_inputs(
                row,
                source_record,
                rig_manifest,
                sign=sign,
                sample_index=sample_index,
            )
            _apply_and_measure_pose_parity(
                armature,
                T_world,
                parent_names,
                T_parent,
                bone_names,
                reconstructed,
            )
            equipment = _place_render_equipment(
                source_record,
                T_world,
                bone_names,
                reconstructed,
            )
            _validate_equipment_metrics(equipment)
            minimum_tip_gap_m = min(minimum_tip_gap_m, equipment["inner_tip_gap_m"])
            for metric, value in equipment.items():
                if metric != "inner_tip_gap_m":
                    metric_maxima[metric] = max(metric_maxima.get(metric, 0.0), value)
            assert isinstance(row, dict)
            edge_degrees.append(math.degrees(float(row["target_left_edge_rad"])))
            lean_degrees.append(math.degrees(float(row["gross_lean_rad"])))
            flat_index += 1
        edge_peak_index = int(np.argmax(np.abs(edge_degrees)))
        lean_peak_index = int(np.argmax(np.abs(lean_degrees)))
        timing_pass = (
            abs(edge_degrees[0]) < 5.0
            and max(abs(value) for value in edge_degrees) >= 45.0
            and abs(edge_degrees[-1]) < 5.0
            and abs(edge_peak_index - lean_peak_index) <= 1
        )
        if not timing_pass:
            raise RuntimeError("Blender carve-cycle edge/lean timing gate failed")
        cycles.append(
            {
                "sign": sign,
                "edge_deg": edge_degrees,
                "gross_lean_deg": lean_degrees,
                "edge_peak_index": edge_peak_index,
                "lean_peak_index": lean_peak_index,
                "timing_pass": timing_pass,
            }
        )
    if not np.allclose(
        np.asarray(cycles[0]["edge_deg"], dtype=np.float64),
        -np.asarray(cycles[1]["edge_deg"], dtype=np.float64),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("Blender carve-cycle signed mirror gate failed")
    audit = {
        "operation": "audit-carve-cycle",
        "blender_version": bpy.app.version_string,
        "scene_sha256": scene_manifest["scene_sha256"],
        "scene_manifest_file_sha256": _sha256_file(scene_manifest_path),
        "rig_manifest_canonical_sha256": rig_manifest.canonical_sha256(),
        "authoritative_pose_table_file_sha256": _sha256_file(pose_table_path),
        "canonical_pose_table_sha256": pose_table["canonical_pose_table_sha256"],
        "cycle_count": 2,
        "sample_count": len(rows),
        "cycles": cycles,
        "equipment_metric_maxima": metric_maxima,
        "minimum_inner_tip_gap_m": minimum_tip_gap_m,
        "status": "PASS",
    }
    _write_json(audit_path, audit)
    print("B3_CS3_AUDIT_CARVE_CYCLE_OK", json.dumps(audit, sort_keys=True, separators=(",", ":")))


def _evaluated_mesh_arrays(
    obj: bpy.types.Object,
    depsgraph: bpy.types.Depsgraph,
) -> tuple[np.ndarray, np.ndarray]:
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
    try:
        mesh.calc_loop_triangles()
        transform = np.asarray(evaluated.matrix_world, dtype=np.float64)
        vertices_local = np.asarray([tuple(vertex.co) for vertex in mesh.vertices], dtype=np.float64)
        vertices_world = vertices_local @ transform[:3, :3].T + transform[:3, 3]
        triangles = np.asarray(
            [tuple(triangle.vertices) for triangle in mesh.loop_triangles],
            dtype=np.int64,
        )
    finally:
        evaluated.to_mesh_clear()
    if vertices_world.shape[0] < 3 or triangles.shape[0] < 1:
        raise RuntimeError(f"render mesh has no triangle geometry: {obj.name}")
    return vertices_world, triangles


def _union_target_geometry(
    depsgraph: bpy.types.Depsgraph,
) -> TargetGeometry:
    vertices: list[np.ndarray] = []
    triangles: list[np.ndarray] = []
    offset = 0
    for name in TARGET_OBJECT_NAMES:
        obj = bpy.data.objects[name]
        object_vertices, object_triangles = _evaluated_mesh_arrays(obj, depsgraph)
        vertices.append(object_vertices)
        triangles.append(object_triangles + offset)
        offset += object_vertices.shape[0]
    return TargetGeometry(
        vertices_world_m=np.concatenate(vertices, axis=0).astype(np.float64),
        triangles=np.concatenate(triangles, axis=0).astype(np.int64),
    )


def _clip_polygon_to_near_plane(
    polygon_cam: list[np.ndarray],
    near_clip_m: float,
) -> list[np.ndarray]:
    clipped: list[np.ndarray] = []
    previous = polygon_cam[-1]
    previous_depth = -float(previous[2])
    previous_inside = previous_depth >= near_clip_m
    for current in polygon_cam:
        current_depth = -float(current[2])
        current_inside = current_depth >= near_clip_m
        if current_inside != previous_inside:
            weight = (near_clip_m - previous_depth) / (current_depth - previous_depth)
            clipped.append(previous + weight * (current - previous))
        if current_inside:
            clipped.append(current)
        previous = current
        previous_depth = current_depth
        previous_inside = current_inside
    return clipped


def _clip_occluder_geometry(
    *,
    object_id: str,
    vertices_world: np.ndarray,
    triangles: np.ndarray,
    T_cam_from_world: np.ndarray,
    near_clip_m: float,
) -> OccluderGeometry | None:
    vertices_cam = vertices_world @ T_cam_from_world[:3, :3].T + T_cam_from_world[:3, 3]
    T_world_from_cam = np.linalg.inv(T_cam_from_world)
    clipped_vertices_world: list[np.ndarray] = []
    clipped_triangles: list[tuple[int, int, int]] = []
    for face in triangles:
        polygon = _clip_polygon_to_near_plane(
            [np.array(vertices_cam[index], dtype=np.float64, copy=True) for index in face],
            near_clip_m,
        )
        if len(polygon) < 3:
            continue
        world_polygon = [
            point @ T_world_from_cam[:3, :3].T + T_world_from_cam[:3, 3]
            for point in polygon
        ]
        base = len(clipped_vertices_world)
        clipped_vertices_world.extend(world_polygon)
        for index in range(1, len(world_polygon) - 1):
            clipped_triangles.append((base, base + index, base + index + 1))
    if not clipped_triangles:
        return None
    return OccluderGeometry(
        object_id=object_id,
        vertices_world_m=np.asarray(clipped_vertices_world, dtype=np.float64),
        triangles=np.asarray(clipped_triangles, dtype=np.int64),
    )


def _scene_occluders(
    depsgraph: bpy.types.Depsgraph,
    T_cam_from_world: np.ndarray,
    *,
    omit_fixed_occluder: bool = False,
) -> tuple[OccluderGeometry, ...]:
    near_clip_m = float(default_camera_contract().manifest()["clip_start_m"])
    occluders: list[OccluderGeometry] = []
    for obj in sorted(bpy.data.objects, key=lambda item: item.name):
        if obj.type != "MESH" or obj.name in TARGET_OBJECT_NAMES or obj.hide_render:
            continue
        if omit_fixed_occluder and obj.name == "CS3_FixedOccluder":
            continue
        vertices, triangles = _evaluated_mesh_arrays(obj, depsgraph)
        object_id = str(obj.get("cs3_occluder_id", obj.name))
        clipped = _clip_occluder_geometry(
            object_id=object_id,
            vertices_world=vertices,
            triangles=triangles,
            T_cam_from_world=T_cam_from_world,
            near_clip_m=near_clip_m,
        )
        if clipped is not None:
            occluders.append(clipped)
    return tuple(occluders)


def _projection_parity_residual_px(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    target: TargetGeometry,
    T_cam_from_world: np.ndarray,
    K: np.ndarray,
) -> float:
    vertices = target.vertices_world_m
    stride = max(1, vertices.shape[0] // 128)
    sampled = vertices[::stride]
    camera_points = sampled @ T_cam_from_world[:3, :3].T + T_cam_from_world[:3, 3]
    depth = -camera_points[:, 2]
    if np.any(depth <= 0.0):
        raise RuntimeError("target contains a non-positive-depth vertex")
    analytic = np.column_stack(
        (
            K[0, 0] * camera_points[:, 0] / depth + K[0, 2],
            K[1, 2] - K[1, 1] * camera_points[:, 1] / depth,
        )
    )
    blender = np.asarray(
        [
            (
                world_to_camera_view(scene, camera, Vector(point)).x * scene.render.resolution_x,
                (1.0 - world_to_camera_view(scene, camera, Vector(point)).y)
                * scene.render.resolution_y,
            )
            for point in sampled
        ],
        dtype=np.float64,
    )
    return float(np.max(np.linalg.norm(analytic - blender, axis=1)))


def _blender_center_ray_id_passes(
    scene: bpy.types.Scene,
    depsgraph: bpy.types.Depsgraph,
    target: TargetGeometry,
    *,
    T_cam_from_world: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normal-scene and target-only Blender ID passes at pixel centers."""
    world_from_cam = np.linalg.inv(T_cam_from_world)
    origin = Vector(world_from_cam[:3, 3].tolist())
    target_bvh = BVHTree.FromPolygons(
        [Vector(vertex.tolist()) for vertex in target.vertices_world_m],
        [tuple(int(index) for index in face) for face in target.triangles],
        all_triangles=True,
    )
    if target_bvh is None:
        raise RuntimeError("Blender target-only BVH construction failed")
    normal_mask = np.zeros((224, 224), dtype=np.bool_)
    target_only_mask = np.zeros((224, 224), dtype=np.bool_)
    target_names = frozenset(TARGET_OBJECT_NAMES)
    clip_end_m = float(default_camera_contract().manifest()["clip_end_m"])
    for v_px in range(224):
        for u_px in range(224):
            direction_cam = np.asarray(
                [
                    (float(u_px) + 0.5 - K[0, 2]) / K[0, 0],
                    (K[1, 2] - float(v_px) - 0.5) / K[1, 1],
                    -1.0,
                ],
                dtype=np.float64,
            )
            direction_world = world_from_cam[:3, :3] @ direction_cam
            direction_world /= np.linalg.norm(direction_world)
            direction = Vector(direction_world.tolist())
            target_location, _target_normal, _target_face, _target_distance = (
                target_bvh.ray_cast(origin, direction, clip_end_m)
            )
            target_only_mask[v_px, u_px] = target_location is not None
            hit, _location, _normal, _face, hit_object, _matrix = scene.ray_cast(
                depsgraph,
                origin,
                direction,
                distance=clip_end_m,
            )
            normal_mask[v_px, u_px] = bool(
                hit and hit_object is not None and hit_object.name in target_names
            )
    return normal_mask, target_only_mask


def _render_rgb_and_id_passes(
    scene: bpy.types.Scene,
    frame_dir: Path,
    depsgraph: bpy.types.Depsgraph,
    target: TargetGeometry,
    *,
    T_cam_from_world: np.ndarray,
    K: np.ndarray,
) -> tuple[Path, Path, Path, np.ndarray, np.ndarray]:
    frame_dir.mkdir(parents=True, exist_ok=False)
    scene.frame_set(1)
    rgb_path = frame_dir / "rgb.png"
    target_id_path = frame_dir / "target_id.png"
    target_only_id_path = frame_dir / "target_only_id.png"
    scene.use_nodes = False
    scene.render.filepath = str(rgb_path)
    bpy.ops.render.render(write_still=True)
    if _png_dimensions(rgb_path) != (224, 224):
        raise RuntimeError("RGB output dimension drift")
    rgb_pixels = _decode_png_u8(rgb_path)
    if rgb_pixels.shape != (224, 224, 3):
        raise RuntimeError("RGB output format drift")
    _write_rgb_png(rgb_path, rgb_pixels)
    normal_mask, target_only_mask = _blender_center_ray_id_passes(
        scene,
        depsgraph,
        target,
        T_cam_from_world=T_cam_from_world,
        K=K,
    )
    _write_u8_png(target_id_path, np.asarray(normal_mask, dtype=np.uint8) * 255)
    _write_u8_png(
        target_only_id_path,
        np.asarray(target_only_mask, dtype=np.uint8) * 255,
    )
    return rgb_path, target_id_path, target_only_id_path, normal_mask, target_only_mask


def _posed_record_from_export(
    source_record: Any,
    T_world_from_armature: np.ndarray,
    T_root_from_bone: np.ndarray,
    local_bone_transforms: np.ndarray,
) -> Any:
    pose_root = replace(
        source_record.root,
        pelvis_point_world_m=np.asarray(
            T_world_from_armature[:3, 3], dtype=np.float64
        ),
        T_world_from_armature=np.asarray(T_world_from_armature, dtype=np.float64),
    )
    posed_state = replace(
        source_record.state,
        tracked_joint_positions_root_m=np.asarray(
            T_root_from_bone[:, :3, 3], dtype=np.float64
        ),
        local_bone_transforms=np.asarray(local_bone_transforms, dtype=np.float64),
    )
    return replace(
        source_record,
        state=posed_state,
        root=pose_root,
        root_schema_version=SKIER_POSE_ROOT_SCHEMA_VERSION,
    )


def _label_summary(labels: Any) -> dict[str, object]:
    return {
        "schema_version": labels.schema_version,
        "A_full": labels.A_full,
        "A_in": labels.A_in,
        "A_vis": labels.A_vis,
        "frame_fraction": labels.frame_fraction,
        "visible_fraction": labels.visible_fraction,
        "occlusion_fraction": labels.occlusion_fraction,
        "amodal_bbox_px": [int(value) for value in labels.amodal_bbox_px],
        "cx": labels.cx,
        "cy": labels.cy,
        "log_h": labels.log_h,
        "visible_bbox_px": (
            None
            if labels.visible_bbox_px is None
            else [int(value) for value in labels.visible_bbox_px]
        ),
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


def _write_software_masks(frame_dir: Path, labels: Any) -> dict[str, Path]:
    u_min, v_min, u_max, v_max = (int(value) for value in labels.amodal_bbox_px)
    full = labels.m_full.crop(
        origin_u_px=u_min,
        origin_v_px=v_min,
        width_px=u_max - u_min,
        height_px=v_max - v_min,
    )
    paths = {
        "mask_full": frame_dir / "mask_full.png",
        "mask_in": frame_dir / "mask_in.png",
        "mask_visible": frame_dir / "mask_visible.png",
    }
    _write_u8_png(paths["mask_full"], np.asarray(full, dtype=np.uint8) * 255)
    _write_u8_png(paths["mask_in"], np.asarray(labels.m_in, dtype=np.uint8) * 255)
    _write_u8_png(paths["mask_visible"], np.asarray(labels.m_vis, dtype=np.uint8) * 255)
    return paths


def _render_one_authoritative_row(
    *,
    output_root: Path,
    fixture: Any,
    root_record: Any,
    source_record: Any,
    row: dict[str, object],
    manifest: Any,
    scene: bpy.types.Scene,
    armature: bpy.types.Object,
    camera: bpy.types.Object,
    T_world_from_rig: np.ndarray,
    T_world_from_rig0_achieved: np.ndarray,
    T_world_from_rig_achieved: np.ndarray,
    T_world_from_cam: np.ndarray,
    T_world_from_cam_achieved: np.ndarray,
    T_cam_from_world: np.ndarray,
) -> dict[str, object]:
    identity = (row["fixture_id"], row["record_index"], row["absolute_tick"])
    expected_identity = (
        fixture.fixture_id,
        source_record.state.absolute_tick + 2,
        source_record.state.absolute_tick,
    )
    if identity != expected_identity:
        raise RuntimeError(f"render pose-row identity drift: {identity} != {expected_identity}")
    if source_record.skier_digest() != row["source_skier_digest"]:
        raise RuntimeError("Blender runtime source skier digest differs from authoritative row")

    parent_names = tuple(row["parent_bone_names"])
    bone_names = tuple(row["bone_names"])
    T_world = np.asarray(row["T_world_from_armature"], dtype=np.float64)
    T_parent = np.asarray(row["T_root_from_parent_bone"], dtype=np.float64)
    T_bone = np.asarray(row["T_root_from_bone"], dtype=np.float64)
    T_local = np.asarray(row["local_bone_transforms"], dtype=np.float64)
    reconstructed = _reconstruct_exported_bones_from_local(
        manifest, parent_names, T_parent, bone_names, T_local
    )
    local_position_residual = float(
        np.max(np.linalg.norm(reconstructed[:, :3, 3] - T_bone[:, :3, 3], axis=1))
    )
    local_rotation_residual = float(
        np.max(
            np.linalg.norm(
                reconstructed[:, :3, :3] - T_bone[:, :3, :3],
                axis=(1, 2),
            )
        )
    )
    if max(local_position_residual, local_rotation_residual) > 1.0e-10:
        raise RuntimeError("render local/global authoritative pose reconstruction failed")

    posed_record = _posed_record_from_export(source_record, T_world, T_bone, T_local)
    if not np.allclose(
        posed_record.root.T_world_from_armature, T_world, rtol=0.0, atol=1.0e-10
    ):
        raise RuntimeError("render authoritative pose-root transform drift")
    if posed_record.skier_digest() != row["posed_skier_digest"]:
        raise RuntimeError("Blender runtime posed skier digest differs from authoritative row")
    pose_maxima, _worst, _per_bone = _apply_and_measure_pose_parity(
        armature,
        T_world,
        parent_names,
        T_parent,
        bone_names,
        reconstructed,
    )
    equipment = _place_render_equipment(
        source_record, T_world, bone_names, reconstructed
    )
    _validate_equipment_metrics(equipment)

    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    target = _union_target_geometry(depsgraph)
    camera_contract = default_camera_contract()
    K_requested = camera_contract.K
    projection_matrix = np.asarray(
        camera.calc_matrix_camera(
            depsgraph,
            x=scene.render.resolution_x,
            y=scene.render.resolution_y,
            scale_x=scene.render.pixel_aspect_x,
            scale_y=scene.render.pixel_aspect_y,
        ),
        dtype=np.float64,
    )
    width_px = float(scene.render.resolution_x)
    height_px = float(scene.render.resolution_y)
    K_achieved_blender = np.asarray(
        [
            [
                projection_matrix[0, 0] * width_px / 2.0,
                0.0,
                (1.0 - projection_matrix[0, 2]) * width_px / 2.0,
            ],
            [
                0.0,
                projection_matrix[1, 1] * height_px / 2.0,
                (1.0 + projection_matrix[1, 2]) * height_px / 2.0,
            ],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    intrinsic_residual_px = float(
        np.max(np.abs(K_achieved_blender - K_requested))
    )
    if intrinsic_residual_px > 1.0e-4:
        raise RuntimeError(f"render achieved camera intrinsics drift: {intrinsic_residual_px}")
    K = K_requested
    projection_residual = _projection_parity_residual_px(
        scene, camera, target, T_cam_from_world, K
    )
    if projection_residual > 1.0:
        raise RuntimeError(f"render projection parity exceeds one pixel: {projection_residual}")
    occluders = _scene_occluders(depsgraph, T_cam_from_world)
    labels = compute_frame_labels(
        target,
        T_cam_from_world=T_cam_from_world,
        K=K,
        occluders=occluders,
    )
    labels_without_fixed = None
    fixed_observation_changed = False
    if fixture.fixture_id == "occlusion_path":
        labels_without_fixed = compute_frame_labels(
            target,
            T_cam_from_world=T_cam_from_world,
            K=K,
            occluders=_scene_occluders(
                depsgraph, T_cam_from_world, omit_fixed_occluder=True
            ),
        )
        if (
            labels.A_full != labels_without_fixed.A_full
            or labels.A_in != labels_without_fixed.A_in
            or not np.array_equal(labels.m_in, labels_without_fixed.m_in)
        ):
            raise RuntimeError("fixed occluder changed amodal/in-crop target geometry")
        fixed_observation_changed = not np.array_equal(
            labels.m_vis, labels_without_fixed.m_vis
        )

    record_index = int(row["record_index"])
    frame_rel = Path(fixture.fixture_id) / (
        f"frame_{record_index:02d}_tick_{source_record.state.absolute_tick:+03d}"
    )
    frame_dir = output_root / frame_rel
    rgb_path, target_id_path, target_only_id_path, normal_id, target_only_id = (
        _render_rgb_and_id_passes(
            scene,
            frame_dir,
            depsgraph,
            target,
            T_cam_from_world=T_cam_from_world,
            K=K,
        )
    )
    # Persist the independent masks before parity assertions so a failed
    # Blender/PURE boundary check leaves inspectable, immutable evidence.
    mask_paths = _write_software_masks(frame_dir, labels)
    if not np.array_equal(target_only_id, labels.m_in):
        mismatch = int(np.count_nonzero(target_only_id != labels.m_in))
        raise RuntimeError(f"Blender target-only ID disagrees with M_in at {mismatch} pixels")
    if not np.array_equal(normal_id, labels.m_vis):
        mismatch = int(np.count_nonzero(normal_id != labels.m_vis))
        raise RuntimeError(f"Blender normal target ID disagrees with M_vis at {mismatch} pixels")

    state_path = frame_dir / "state.canonical"
    pose_path = frame_dir / "pose.json"
    state_path.write_bytes(posed_record.canonical_bytes())
    pose_path.write_bytes(_canonical_json_bytes(row))
    root_id = root_record.root_id
    if root_record.split_group_id != root_id:
        raise RuntimeError("render root/split identity drift")
    T_rig_from_cam_requested = np.eye(4, dtype=np.float64)
    T_rig_from_cam_requested[:3, :3] = camera_contract.R_rig_from_cam
    T_cam_from_rig_requested = np.eye(4, dtype=np.float64)
    T_cam_from_rig_requested[:3, :3] = camera_contract.R_cam_from_rig
    T_rig_from_cam_achieved = np.linalg.inv(T_world_from_rig_achieved) @ T_world_from_cam_achieved
    T_cam_from_rig_achieved = np.linalg.inv(T_rig_from_cam_achieved)
    T_rig0_from_rig_t_requested = np.eye(4, dtype=np.float64)
    T_rig0_from_rig_t_achieved = np.linalg.inv(T_world_from_rig0_achieved) @ T_world_from_rig_achieved
    camera_payload = {
        "camera_root_schema_version": CAMERA_ROOT_SCHEMA_VERSION,
        "T_world_from_rig_requested": T_world_from_rig,
        "T_world_from_rig_achieved": T_world_from_rig_achieved,
        "T_world_from_cam_requested": T_world_from_cam,
        "T_world_from_cam_achieved": T_world_from_cam_achieved,
        "T_cam_from_world_achieved": T_cam_from_world,
        "T_rig0_from_rig_t_requested": T_rig0_from_rig_t_requested,
        "T_rig0_from_rig_t_achieved": T_rig0_from_rig_t_achieved,
        "T_rig_from_cam_requested": T_rig_from_cam_requested,
        "T_cam_from_rig_requested": T_cam_from_rig_requested,
        "T_rig_from_cam_achieved": T_rig_from_cam_achieved,
        "T_cam_from_rig_achieved": T_cam_from_rig_achieved,
        "K_requested": K,
        "K_achieved_blender": K_achieved_blender,
        "achieved_intrinsic_residual_px": intrinsic_residual_px,
        "achieved_camera_position_residual_m": float(
            np.linalg.norm(T_world_from_cam_achieved[:3, 3] - T_world_from_cam[:3, 3])
        ),
        "achieved_camera_rotation_residual_rad": _rotation_residual_rad(
            T_world_from_cam_achieved, T_world_from_cam
        ),
        "achieved_rig_action_position_residual_m": float(
            np.linalg.norm(T_rig0_from_rig_t_achieved[:3, 3])
        ),
        "achieved_rig_action_rotation_residual_rad": _rotation_residual_rad(
            T_rig0_from_rig_t_achieved, T_rig0_from_rig_t_requested
        ),
    }
    camera_transform_sha256 = hashlib.sha256(canonical_bytes(camera_payload)).hexdigest()
    artifact_paths = {
        "state_record_path": state_path.relative_to(output_root).as_posix(),
        "pose_record_path": pose_path.relative_to(output_root).as_posix(),
        "rgb_path": rgb_path.relative_to(output_root).as_posix(),
        "mask_full_path": mask_paths["mask_full"].relative_to(output_root).as_posix(),
        "mask_in_path": mask_paths["mask_in"].relative_to(output_root).as_posix(),
        "mask_visible_path": mask_paths["mask_visible"].relative_to(output_root).as_posix(),
        "target_only_id_path": target_only_id_path.relative_to(output_root).as_posix(),
        "target_id_path": target_id_path.relative_to(output_root).as_posix(),
    }
    artifact_hashes = {
        key.removesuffix("_path") + "_sha256": _sha256_file(output_root / value)
        for key, value in artifact_paths.items()
    }
    metadata = {
        "schema_version": FRAME_SCHEMA_VERSION,
        "fixture_id": fixture.fixture_id,
        "root_id": root_id,
        "split_group_id": root_record.split_group_id,
        "split": root_record.split.value,
        "root_record_sha256": root_record.canonical_sha256(),
        "record_index": record_index,
        "absolute_tick": source_record.state.absolute_tick,
        "timestamp_seconds": source_record.state.absolute_tick * 0.2,
        "branch_id": "zero",
        "requested_command": [0.0, 0.0, 0.0, 0.0],
        "requested_command_valid": True,
        "record_valid": True,
        "dt_seconds": FIXED_DT_SECONDS,
        "source_skier_digest": row["source_skier_digest"],
        "posed_skier_digest": row["posed_skier_digest"],
        "pose_sha256": row["pose_sha256"],
        "camera_transform_sha256": camera_transform_sha256,
        "camera": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in camera_payload.items()
        },
        "target_geometry_sha256": hashlib.sha256(
            canonical_bytes(target.payload())
        ).hexdigest(),
        "labels_sha256": labels.canonical_sha256(),
        "labels": _label_summary(labels),
        "fixed_occluder_enabled": fixture.fixture_id == "occlusion_path",
        "fixed_occluder_observation_changed": fixed_observation_changed,
        "occlusion_flag": labels.occlusion_flag,
        "modal_front_object_id": labels.modal_front_object_id,
        "p_visible_target": labels.p_visible_target,
        "without_fixed_occluder_labels_sha256": (
            None
            if labels_without_fixed is None
            else labels_without_fixed.canonical_sha256()
        ),
        "projection_parity_max_px": projection_residual,
        "id_pass_schema_version": ID_PASS_SCHEMA_VERSION,
        "local_pose_reconstruction": {
            "position_m": local_position_residual,
            "rotation_frobenius": local_rotation_residual,
        },
        "pose_parity_maxima": pose_maxima,
        "equipment_metrics": equipment,
        "mask_full_origin_uv_px": [
            int(labels.amodal_bbox_px[0]),
            int(labels.amodal_bbox_px[1]),
        ],
        **artifact_paths,
        **artifact_hashes,
    }
    metadata_path = frame_dir / "frame.json"
    _write_json(metadata_path, metadata)
    metadata_sha256 = _sha256_file(metadata_path)
    return {
        "fixture_id": fixture.fixture_id,
        "root_id": root_id,
        "split_group_id": root_record.split_group_id,
        "split": root_record.split.value,
        "root_record_sha256": root_record.canonical_sha256(),
        "record_index": record_index,
        "absolute_tick": source_record.state.absolute_tick,
        "branch_id": "zero",
        "requested_command": [0.0, 0.0, 0.0, 0.0],
        "requested_command_valid": True,
        "record_valid": True,
        "dt_seconds": FIXED_DT_SECONDS,
        "frame_metadata_path": metadata_path.relative_to(output_root).as_posix(),
        "frame_metadata_sha256": metadata_sha256,
        **artifact_hashes,
        "source_skier_digest": row["source_skier_digest"],
        "pose_sha256": row["pose_sha256"],
        "posed_skier_digest": row["posed_skier_digest"],
        "camera_transform_sha256": camera_transform_sha256,
        "labels_sha256": labels.canonical_sha256(),
        "target_geometry_sha256": metadata["target_geometry_sha256"],
        "projection_parity_max_px": projection_residual,
        "occlusion_flag": labels.occlusion_flag,
        "modal_front_object_id": labels.modal_front_object_id,
        "p_visible_target": labels.p_visible_target,
        "fixed_occluder_observation_changed": fixed_observation_changed,
    }


def render_authoritative_replay(args: argparse.Namespace) -> None:
    rig_manifest_path = Path(args.manifest).resolve()
    rig_manifest = load_rig_manifest(rig_manifest_path)
    asset_manifest_path = Path(args.asset_manifest).resolve()
    asset_manifest = load_asset_manifest(asset_manifest_path)
    scene_manifest_path = Path(args.scene_manifest).resolve()
    scene_manifest = json.loads(scene_manifest_path.read_text(encoding="utf-8"))
    current_scene = Path(bpy.data.filepath).resolve()
    if _sha256_file(current_scene) != scene_manifest["scene_sha256"]:
        raise RuntimeError("render replay loaded scene SHA-256 mismatch")
    if scene_manifest["derived_rig_sha256"] != rig_manifest.derived_rig_sha256:
        raise RuntimeError("render replay scene/rig manifest mismatch")
    pose_table_path = Path(args.pose_table).resolve()
    pose_table_manifest = _read_authoritative_pose_table(rig_manifest, pose_table_path)
    pose_rows = pose_table_manifest["rows"]
    assert isinstance(pose_rows, list)
    fixtures = canonical_skier_fixtures()
    bindings = RootRecordBindings.from_manifests(
        asset_manifest=asset_manifest,
        asset_manifest_file_sha256=_sha256_file(asset_manifest_path),
        rig_manifest=rig_manifest,
        rig_manifest_file_sha256=_sha256_file(rig_manifest_path),
        scene_manifest=scene_manifest,
        scene_manifest_file_sha256=_sha256_file(scene_manifest_path),
        pose_export_manifest=pose_table_manifest,
        pose_export_file_sha256=_sha256_file(pose_table_path),
    )
    root_records = {
        fixture.fixture_id: build_canonical_root_record(fixture, rig_manifest, bindings)
        for fixture in fixtures
    }
    expected = [
        (fixture, record_index, record)
        for fixture in fixtures
        for record_index, record in enumerate(fixture.records())
    ]
    if args.operation == "render-frame":
        fixture_index = int(args.fixture_index)
        record_index = int(args.record_index)
        if fixture_index < 0 or fixture_index >= len(fixtures):
            raise RuntimeError("render-frame fixture index out of range")
        if record_index < 0 or record_index >= len(fixtures[fixture_index].records()):
            raise RuntimeError("render-frame record index out of range")
        flat_index = fixture_index * 11 + record_index
        selected = [(flat_index, expected[flat_index])]
    else:
        selected = list(enumerate(expected))

    output_root = Path(args.output).resolve()
    if output_root.exists():
        raise RuntimeError("render replay output must not already exist")
    output_root.mkdir(parents=True, exist_ok=False)
    root_rows: list[dict[str, object]] = []
    for fixture in fixtures:
        root_record = root_records[fixture.fixture_id]
        root_path = output_root / "roots" / f"{fixture.fixture_id}.canonical"
        root_path.parent.mkdir(parents=True, exist_ok=True)
        root_path.write_bytes(root_record.canonical_bytes())
        root_rows.append(
            {
                "fixture_id": fixture.fixture_id,
                "root_id": root_record.root_id,
                "split_group_id": root_record.split_group_id,
                "split": root_record.split.value,
                "root_record_schema_version": ROOT_RECORD_SCHEMA_VERSION,
                "root_envelope_schema_version": ROOT_ENVELOPE_SCHEMA_VERSION,
                "root_record_path": root_path.relative_to(output_root).as_posix(),
                "root_record_sha256": _sha256_file(root_path),
            }
        )
    scene = bpy.context.scene
    scene.cycles.use_denoising = False
    scene.render.use_persistent_data = False
    armature = bpy.data.objects.get(DERIVED_ARMATURE_NAME)
    camera = bpy.data.objects.get("CS3_Camera")
    rig = bpy.data.objects.get("DroneRig")
    if armature is None or camera is None or rig is None:
        raise RuntimeError("render replay scene hierarchy incomplete")

    replay_rows: list[dict[str, object]] = []
    camera_hash_by_fixture: dict[str, str] = {}
    active_fixture_id: str | None = None
    T_world_from_rig = np.eye(4, dtype=np.float64)
    T_world_from_rig0_achieved = np.eye(4, dtype=np.float64)
    T_world_from_rig_achieved = np.eye(4, dtype=np.float64)
    T_world_from_cam = np.eye(4, dtype=np.float64)
    T_world_from_cam_achieved = np.eye(4, dtype=np.float64)
    T_cam_from_world = np.eye(4, dtype=np.float64)
    camera_contract = default_camera_contract()
    expected_local = np.eye(4, dtype=np.float64)
    expected_local[:3, :3] = camera_contract.R_rig_from_cam
    fixed_occluder = bpy.data.objects.get("CS3_FixedOccluder")
    if fixed_occluder is None:
        raise RuntimeError("render replay fixed occluder missing")
    for flat_index, (fixture, _record_index, source_record) in selected:
        if fixture.fixture_id != active_fixture_id:
            fixed_occluder_enabled = fixture.fixture_id == "occlusion_path"
            fixed_occluder.hide_render = not fixed_occluder_enabled
            fixed_occluder.hide_set(not fixed_occluder_enabled)
            T_world_from_rig = initial_camera_rig_transform(fixture.records()[2])
            rig.matrix_world = Matrix(T_world_from_rig.tolist())
            bpy.context.view_layer.update()
            T_world_from_rig0_achieved, rig_rigidization_residual = _rigidize_rest_matrix(
                rig.matrix_world
            )
            if rig_rigidization_residual > 1.0e-6:
                raise RuntimeError("render replay initial rig is not rigid")
            active_fixture_id = fixture.fixture_id
        T_world_from_rig_achieved, rig_rigidization_residual = _rigidize_rest_matrix(
            rig.matrix_world
        )
        achieved_world_from_cam, camera_rigidization_residual = _rigidize_rest_matrix(
            camera.matrix_world
        )
        if max(rig_rigidization_residual, camera_rigidization_residual) > 1.0e-6:
            raise RuntimeError("render replay achieved rig/camera is not rigid")
        T_world_from_cam_achieved = achieved_world_from_cam
        expected_world_from_cam = T_world_from_rig @ expected_local
        camera_position_residual = float(
            np.linalg.norm(
                achieved_world_from_cam[:3, 3] - expected_world_from_cam[:3, 3]
            )
        )
        camera_rotation_residual = _rotation_residual_rad(
            achieved_world_from_cam, expected_world_from_cam
        )
        rig_position_residual = float(
            np.linalg.norm(T_world_from_rig_achieved[:3, 3] - T_world_from_rig[:3, 3])
        )
        rig_rotation_residual = _rotation_residual_rad(
            T_world_from_rig_achieved, T_world_from_rig
        )
        if max(
            camera_position_residual,
            camera_rotation_residual,
            rig_position_residual,
            rig_rotation_residual,
        ) > 1.0e-6:
            raise RuntimeError("render replay achieved rig/camera SE(3) drift")
        T_world_from_cam = expected_world_from_cam
        # Labels are defined by the camera that produced the pixels.  Keep the
        # requested transform above for the achieved-SE(3) audit, but
        # project/rasterize through Blender's independently measured camera.
        T_cam_from_world = np.linalg.inv(achieved_world_from_cam)
        rendered = _render_one_authoritative_row(
            output_root=output_root,
            fixture=fixture,
            root_record=root_records[fixture.fixture_id],
            source_record=source_record,
            row=pose_rows[flat_index],
            manifest=rig_manifest,
            scene=scene,
            armature=armature,
            camera=camera,
            T_world_from_rig=T_world_from_rig,
            T_world_from_rig0_achieved=T_world_from_rig0_achieved,
            T_world_from_rig_achieved=T_world_from_rig_achieved,
            T_world_from_cam=T_world_from_cam,
            T_world_from_cam_achieved=T_world_from_cam_achieved,
            T_cam_from_world=T_cam_from_world,
        )
        previous_hash = camera_hash_by_fixture.setdefault(
            fixture.fixture_id, rendered["camera_transform_sha256"]
        )
        if previous_hash != rendered["camera_transform_sha256"]:
            raise RuntimeError("render replay camera moved within fixture")
        replay_rows.append(rendered)

    complete = len(replay_rows) == 88
    if complete:
        occlusion_rows = [
            row for row in replay_rows if row["fixture_id"] == "occlusion_path"
        ]
        changed_indices = [
            index
            for index, row in enumerate(occlusion_rows)
            if row["fixed_occluder_observation_changed"]
        ]
        negatives = [
            index
            for index, row in enumerate(occlusion_rows)
            if row["occlusion_flag"]
            and row["p_visible_target"] == 0
            and row["modal_front_object_id"] == "fixed_occluder_101"
        ]
        if not changed_indices or not negatives:
            raise RuntimeError("render replay fixed-obstacle negative gate failed")
        first_negative = min(negatives)
        last_negative = max(negatives)
        def positive(row: dict[str, object]) -> bool:
            return row["p_visible_target"] == 1 and not row["occlusion_flag"]

        if not any(positive(row) for row in occlusion_rows[:first_negative]):
            raise RuntimeError("render replay lacks positive pre-occlusion frame")
        if not any(positive(row) for row in occlusion_rows[last_negative + 1 :]):
            raise RuntimeError("render replay lacks positive post-occlusion frame")

    replay = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "blender_version": bpy.app.version_string,
        "blender_build_hash": bpy.app.build_hash.decode("ascii"),
        "scene_sha256": scene_manifest["scene_sha256"],
        "scene_manifest_file_sha256": _sha256_file(scene_manifest_path),
        "asset_manifest_file_sha256": _sha256_file(asset_manifest_path),
        "rig_manifest_file_sha256": _sha256_file(rig_manifest_path),
        "rig_manifest_canonical_sha256": rig_manifest.canonical_sha256(),
        "authoritative_pose_table_file_sha256": _sha256_file(pose_table_path),
        "canonical_pose_table_sha256": PINNED_CANONICAL_POSE_TABLE_SHA256,
        "renderer": canonical_renderer_contract(),
        "root_record_schema_version": ROOT_RECORD_SCHEMA_VERSION,
        "root_envelope_schema_version": ROOT_ENVELOPE_SCHEMA_VERSION,
        "root_count": len(root_rows),
        "roots": root_rows,
        "fixture_count": 8 if complete else len({row["fixture_id"] for row in replay_rows}),
        "frame_count": len(replay_rows),
        "rows": replay_rows,
    }
    replay_path = output_root / "replay.json"
    _write_json(replay_path, replay)
    result = {
        "operation": args.operation,
        "complete": complete,
        "frame_count": len(replay_rows),
        "replay_sha256": _sha256_file(replay_path),
        "output": str(output_root),
    }
    print("B3_CS3_RENDER_REPLAY_OK", json.dumps(result, sort_keys=True, separators=(",", ":")))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="operation", required=True)
    freeze = subparsers.add_parser("freeze-rig")
    freeze.add_argument("--source", required=True)
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--manifest", required=True)
    freeze.add_argument("--audit", required=True)
    derived = subparsers.add_parser("audit-derived")
    derived.add_argument("--manifest", required=True)
    derived.add_argument("--audit", required=True)
    pose = subparsers.add_parser("audit-pose")
    pose.add_argument("--manifest", required=True)
    pose.add_argument("--fixture-index", required=True, type=int)
    pose.add_argument("--record-index", required=True, type=int)
    pose.add_argument("--audit", required=True)
    pose_table = subparsers.add_parser("audit-pose-table")
    pose_table.add_argument("--manifest", required=True)
    pose_table.add_argument("--pose-table", required=True)
    pose_table.add_argument("--audit", required=True)
    carve_cycle = subparsers.add_parser("audit-carve-cycle")
    carve_cycle.add_argument("--manifest", required=True)
    carve_cycle.add_argument("--scene-manifest", required=True)
    carve_cycle.add_argument("--pose-table", required=True)
    carve_cycle.add_argument("--audit", required=True)
    scene_build = subparsers.add_parser("build-scene")
    scene_build.add_argument("--manifest", required=True)
    scene_build.add_argument("--asset-manifest", required=True)
    scene_build.add_argument("--output", required=True)
    scene_build.add_argument("--scene-manifest", required=True)
    scene_build.add_argument("--audit", required=True)
    scene_audit = subparsers.add_parser("audit-scene")
    scene_audit.add_argument("--scene-manifest", required=True)
    scene_audit.add_argument("--audit", required=True)
    render_frame = subparsers.add_parser("render-frame")
    render_frame.add_argument("--manifest", required=True)
    render_frame.add_argument("--asset-manifest", required=True)
    render_frame.add_argument("--scene-manifest", required=True)
    render_frame.add_argument("--pose-table", required=True)
    render_frame.add_argument("--fixture-index", required=True, type=int)
    render_frame.add_argument("--record-index", required=True, type=int)
    render_frame.add_argument("--output", required=True)
    render_replay = subparsers.add_parser("render-eight-root")
    render_replay.add_argument("--manifest", required=True)
    render_replay.add_argument("--asset-manifest", required=True)
    render_replay.add_argument("--scene-manifest", required=True)
    render_replay.add_argument("--pose-table", required=True)
    render_replay.add_argument("--output", required=True)
    return parser


def main() -> None:
    _verify_runtime()
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = _parser().parse_args(arguments)
    if args.operation == "freeze-rig":
        freeze_rig(args)
    elif args.operation == "audit-derived":
        audit_derived(args)
    elif args.operation == "audit-pose":
        audit_pose(args)
    elif args.operation == "audit-pose-table":
        audit_pose_table(args)
    elif args.operation == "audit-carve-cycle":
        audit_carve_cycle(args)
    elif args.operation == "build-scene":
        build_scene(args)
    elif args.operation == "audit-scene":
        audit_scene(args)
    elif args.operation in ("render-frame", "render-eight-root"):
        render_authoritative_replay(args)
    else:
        raise AssertionError(args.operation)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # Blender otherwise exits zero after a --python exception.
        traceback.print_exc()
        raise SystemExit(1) from None
