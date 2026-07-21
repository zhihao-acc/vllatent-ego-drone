"""PURE absolute-tick B3-CS3 skier pose and stateless binding IK.

The armature object/root remains authoritative in :mod:`vllatent.sim.skier`.
This module evaluates only manifested armature-local bone transforms.  It has no
camera, branch, visibility, renderer, wall-clock, random-generator, or stateful
IK input.  Boots are placed on the already-constructed ski binding frames by an
analytic two-bone solve; animation never changes a ski origin or frame.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from typing import Final

import numpy as np

from vllatent.sim.contracts import FIXED_DT_SECONDS, SKIER_POSE_ROOT_SCHEMA_VERSION, canonical_bytes
from vllatent.sim.frames import rotation_geodesic_angle
from vllatent.sim.rig import (
    AMPLITUDE_EXCLUDED_SEMANTICS,
    BONE_BY_SEMANTIC,
    LEFT_RIGHT_PAIRS,
    TRACKED_SEMANTICS,
    RigManifest,
)
from vllatent.sim.skier import (
    PROOF_CROUCH_DROP_M,
    PROOF_STANDING_PELVIS_HEIGHT_M,
    ManeuverRecord,
    ManeuverSchedule,
    ManeuverTargets,
    ManeuverType,
    RootGeometry,
    SkierFrameRecord,
    SkierParameters,
    SkierState,
    default_slope_frame,
    drag_area_from_crouch,
    frame_record,
    ideal_carve_curvature,
    neutral_targets,
)

POSE_SCHEMA_VERSION: Final[str] = "b3-cs3-absolute-pose-f64-v1"
POSE_TRANSFORM_SEMANTICS: Final[str] = "T_armature_root_from_manifested_bone"
LOCAL_BONE_TRANSFORM_SEMANTICS: Final[str] = (
    "D_parent_rest_frame_where_L_pose_equals_D_times_L_rest"
)
IK_SCHEMA_VERSION: Final[str] = "b3-cs3-stateless-two-bone-binding-v1"
POSE_ROOT_SCHEMA_VERSION: Final[str] = "b3-cs3-authoritative-pelvis-height-v1"
CARVE_CYCLE_SCHEMA_VERSION: Final[str] = "b3-cs3-authoritative-carve-cycle-v2"
POSE_TABLE_EXPORT_SCHEMA_VERSION: Final[str] = "b3-cs3-pose-table-export-v3"
PINNED_CANONICAL_POSE_TABLE_SHA256: Final[str] = (
    "d39624214cee5e4fdddc725a8ee8142a12807ae86c96170827ee9622c13018fd"
)
PINNED_POSE_TABLE_EXPORT_SHA256: Final[str] = (
    "d07a9104ef2c271ef83e54921515a5d0eb9f1bdae7ffd245f0745a0c2b59ae6f"
)

INTERMEDIATE_PARENT_NAMES: Final[tuple[str, ...]] = (
    "root",
    "spine_02",
    "neck_01",
    "clavicle_l",
    "clavicle_r",
)

SIDE_ORDER: Final[tuple[str, str]] = ("left", "right")
MAX_BODY_LEAN_RAD: Final[float] = math.radians(35.0)
CYCLE_FLEXION_M: Final[float] = 0.035
BRAKE_FLEXION_M: Final[float] = 0.050
TRANSITION_FLEXION_M: Final[float] = 0.160
TRANSITION_EDGE_REFERENCE_RAD: Final[float] = math.radians(50.0)
CARVE_CYCLE_SAMPLE_COUNT: Final[int] = 11
CARVE_CYCLE_RAMP_IN_TICKS: Final[int] = 5
CARVE_CYCLE_HOLD_TICKS: Final[int] = 1
CARVE_CYCLE_RAMP_OUT_TICKS: Final[int] = 5
CARVE_CYCLE_TARGET_EDGE_RAD: Final[float] = math.radians(50.0)
CARVE_CYCLE_SPEED_M_S: Final[float] = 8.0
CARVE_CYCLE_RANDOMNESS_SEED: Final[int] = 314159

_SE3_ATOL = 1.0e-10


def _immutable_f64(name: str, value: object, shape: tuple[int, ...]) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name}: expected np.ndarray")
    if value.shape != shape or value.dtype != np.dtype("<f8") or not np.all(np.isfinite(value)):
        raise ValueError(f"{name}: expected finite float64 shape {shape}")
    contiguous = np.array(value, dtype=np.dtype("<f8"), order="C", copy=True)
    contiguous[contiguous == 0.0] = 0.0
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=np.dtype("<f8")).reshape(shape)


def _validate_rotation(name: str, value: np.ndarray) -> None:
    if not np.allclose(value.T @ value, np.eye(3), rtol=0.0, atol=_SE3_ATOL):
        raise ValueError(f"{name}: expected orthonormal rotation")
    if not math.isclose(float(np.linalg.det(value)), 1.0, rel_tol=0.0, abs_tol=_SE3_ATOL):
        raise ValueError(f"{name}: expected determinant +1")


def _validate_transforms(name: str, value: np.ndarray) -> None:
    for index, transform in enumerate(value):
        if not np.allclose(transform[3], np.array([0.0, 0.0, 0.0, 1.0]), rtol=0.0, atol=_SE3_ATOL):
            raise ValueError(f"{name}[{index}]: invalid homogeneous bottom row")
        _validate_rotation(f"{name}[{index}]", transform[:3, :3])


def _normalize(name: str, value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError(f"{name}: expected nonzero finite vector")
    return value / norm


def _axis_rotation(axis: str, angle_rad: float) -> np.ndarray:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]], dtype=np.float64)
    if axis == "y":
        return np.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]], dtype=np.float64)
    raise ValueError(f"axis: unsupported axis {axis!r}")


def _rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    first = _normalize("source", source)
    second = _normalize("target", target)
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    if cosine >= 1.0 - 1.0e-14:
        return np.eye(3, dtype=np.float64)
    if cosine <= -1.0 + 1.0e-14:
        basis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(first, basis))) > 0.8:
            basis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = _normalize("opposite rotation axis", np.cross(first, basis))
        return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float64)
    cross = np.cross(first, second)
    skew = np.array(
        [[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + skew + skew @ skew / (1.0 + cosine)


def _aligned_bone_rotation(rest_rotation: np.ndarray, target_direction: np.ndarray) -> np.ndarray:
    rotation = _rotation_between(rest_rotation[:, 1], target_direction) @ rest_rotation
    left, _singular, right_t = np.linalg.svd(rotation)
    result = left @ right_t
    if float(np.linalg.det(result)) < 0.0:
        left[:, -1] *= -1.0
        result = left @ right_t
    return result


def _transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    _validate_rotation("rotation", rotation)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def _two_bone_solve(
    hip: np.ndarray,
    ankle: np.ndarray,
    upper_length_m: float,
    lower_length_m: float,
    pole: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    displacement = ankle - hip
    distance = float(np.linalg.norm(displacement))
    maximum = upper_length_m + lower_length_m
    minimum = abs(upper_length_m - lower_length_m)
    if not minimum < distance < maximum:
        raise ValueError(f"binding target violates strict non-stretch reach: {minimum} < {distance} < {maximum}")
    direction = displacement / distance
    pole_plane = pole - float(np.dot(pole, direction)) * direction
    if float(np.linalg.norm(pole_plane)) <= 1.0e-10:
        raise ValueError("IK pole is collinear with the binding target")
    bend = _normalize("IK pole plane", pole_plane)
    along = (
        upper_length_m**2 - lower_length_m**2 + distance**2
    ) / (2.0 * distance)
    height = math.sqrt(max(0.0, upper_length_m**2 - along**2))
    knee = hip + along * direction + height * bend
    hip_to_knee = hip - knee
    ankle_from_knee = ankle - knee
    internal = math.acos(
        float(
            np.clip(
                np.dot(hip_to_knee, ankle_from_knee)
                / (np.linalg.norm(hip_to_knee) * np.linalg.norm(ankle_from_knee)),
                -1.0,
                1.0,
            )
        )
    )
    flexion = math.pi - internal
    return knee, distance / maximum, flexion


@dataclass(frozen=True, eq=False)
class PoseEvaluation:
    """One immutable, camera-independent manifested pose at an absolute tick."""

    absolute_tick: int
    rig_manifest_sha256: str
    source_skier_digest: str
    bone_names: tuple[str, ...]
    parent_bone_names: tuple[str, ...]
    T_world_from_armature: np.ndarray
    T_root_from_parent_bone: np.ndarray
    T_root_from_bone: np.ndarray
    local_bone_transforms: np.ndarray
    tracked_joint_positions_root_m: np.ndarray
    binding_target_root_m: np.ndarray
    F_root_from_binding: np.ndarray
    F_root_from_boot_object: np.ndarray
    ik_reach_ratio: np.ndarray
    knee_flexion_rad: np.ndarray
    pelvis_height_m: float
    shoulder_height_m: float
    transition_flexion_m: float

    def __post_init__(self) -> None:
        if isinstance(self.absolute_tick, bool) or not isinstance(self.absolute_tick, int):
            raise TypeError("absolute_tick: expected int")
        if len(self.rig_manifest_sha256) != 64:
            raise ValueError("rig_manifest_sha256: expected SHA-256 hex")
        if len(self.source_skier_digest) != 64:
            raise ValueError("source_skier_digest: expected SHA-256 hex")
        if self.bone_names != tuple(BONE_BY_SEMANTIC[semantic] for semantic in TRACKED_SEMANTICS):
            raise ValueError("bone_names: manifested order drift")
        if self.parent_bone_names != INTERMEDIATE_PARENT_NAMES:
            raise ValueError("parent_bone_names: evaluated ancestor order drift")
        count = len(TRACKED_SEMANTICS)
        armature = _immutable_f64("T_world_from_armature", self.T_world_from_armature, (4, 4))
        _validate_transforms("T_world_from_armature", armature.reshape((1, 4, 4)))
        object.__setattr__(self, "T_world_from_armature", armature)
        parent_transforms = _immutable_f64(
            "T_root_from_parent_bone",
            self.T_root_from_parent_bone,
            (len(INTERMEDIATE_PARENT_NAMES), 4, 4),
        )
        _validate_transforms("T_root_from_parent_bone", parent_transforms)
        object.__setattr__(self, "T_root_from_parent_bone", parent_transforms)
        transforms = _immutable_f64("T_root_from_bone", self.T_root_from_bone, (count, 4, 4))
        _validate_transforms("T_root_from_bone", transforms)
        object.__setattr__(self, "T_root_from_bone", transforms)
        local = _immutable_f64("local_bone_transforms", self.local_bone_transforms, (count, 4, 4))
        _validate_transforms("local_bone_transforms", local)
        object.__setattr__(self, "local_bone_transforms", local)
        positions = _immutable_f64(
            "tracked_joint_positions_root_m",
            self.tracked_joint_positions_root_m,
            (count, 3),
        )
        if not np.array_equal(positions, transforms[:, :3, 3]):
            raise ValueError("tracked_joint_positions_root_m: must equal manifested bone origins")
        object.__setattr__(self, "tracked_joint_positions_root_m", positions)
        targets = _immutable_f64("binding_target_root_m", self.binding_target_root_m, (2, 3))
        object.__setattr__(self, "binding_target_root_m", targets)
        binding_frames = _immutable_f64("F_root_from_binding", self.F_root_from_binding, (2, 3, 3))
        for index, frame in enumerate(binding_frames):
            _validate_rotation(f"F_root_from_binding[{index}]", frame)
        object.__setattr__(self, "F_root_from_binding", binding_frames)
        frames = _immutable_f64("F_root_from_boot_object", self.F_root_from_boot_object, (2, 3, 3))
        for index, frame in enumerate(frames):
            _validate_rotation(f"F_root_from_boot_object[{index}]", frame)
        object.__setattr__(self, "F_root_from_boot_object", frames)
        reach = _immutable_f64("ik_reach_ratio", self.ik_reach_ratio, (2,))
        if np.any(reach <= 0.0) or np.any(reach > 1.0 + 1.0e-10):
            raise ValueError("ik_reach_ratio: expected non-stretch values in (0,1]")
        object.__setattr__(self, "ik_reach_ratio", reach)
        flexion = _immutable_f64("knee_flexion_rad", self.knee_flexion_rad, (2,))
        if np.any(flexion < 0.0) or np.any(flexion > math.pi):
            raise ValueError("knee_flexion_rad: expected values in [0,pi]")
        object.__setattr__(self, "knee_flexion_rad", flexion)
        for name in ("pelvis_height_m", "shoulder_height_m", "transition_flexion_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name}: expected finite value")
            object.__setattr__(self, name, value)

    def transform(self, semantic: str) -> np.ndarray:
        """Return the immutable root-local transform for one manifested semantic."""
        try:
            index = TRACKED_SEMANTICS.index(semantic)
        except ValueError as error:
            raise KeyError(semantic) from error
        return self.T_root_from_bone[index]

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": POSE_SCHEMA_VERSION,
            "transform_semantics": POSE_TRANSFORM_SEMANTICS,
            "local_transform_semantics": LOCAL_BONE_TRANSFORM_SEMANTICS,
            "ik_schema_version": IK_SCHEMA_VERSION,
            "pose_root_schema_version": POSE_ROOT_SCHEMA_VERSION,
            "absolute_tick": self.absolute_tick,
            "rig_manifest_sha256": self.rig_manifest_sha256,
            "source_skier_digest": self.source_skier_digest,
            "bone_names": self.bone_names,
            "parent_bone_names": self.parent_bone_names,
            "T_world_from_armature": self.T_world_from_armature,
            "T_root_from_parent_bone": self.T_root_from_parent_bone,
            "T_root_from_bone": self.T_root_from_bone,
            "local_bone_transforms": self.local_bone_transforms,
            "tracked_joint_positions_root_m": self.tracked_joint_positions_root_m,
            "binding_target_root_m": self.binding_target_root_m,
            "F_root_from_binding": self.F_root_from_binding,
            "F_root_from_boot_object": self.F_root_from_boot_object,
            "ik_reach_ratio": self.ik_reach_ratio,
            "knee_flexion_rad": self.knee_flexion_rad,
            "pelvis_height_m": self.pelvis_height_m,
            "shoulder_height_m": self.shoulder_height_m,
            "transition_flexion_m": self.transition_flexion_m,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.payload())

    def canonical_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def _record_phase(record: SkierFrameRecord) -> float:
    expected = (record.state.absolute_tick * FIXED_DT_SECONDS) % 1.0
    if record.animation.animation_phase != expected:
        raise ValueError("animation phase is not derived from the record's absolute tick")
    return record.animation.animation_phase


def _pose_flexion_terms(record: SkierFrameRecord, phase: float) -> tuple[float, float, float]:
    cycle_flexion = CYCLE_FLEXION_M * 0.5 * (1.0 + math.cos(2.0 * math.pi * phase))
    brake_flexion = BRAKE_FLEXION_M if record.evaluated_maneuver.maneuver_type is ManeuverType.BRAKE else 0.0
    transition_flexion = 0.0
    if record.evaluated_maneuver.maneuver_type is ManeuverType.TRANSITION:
        mean_edge = 0.5 * (
            record.evaluated_maneuver.targets.left_edge_rad
            + record.evaluated_maneuver.targets.right_edge_rad
        )
        transition_pulse = max(0.0, 1.0 - abs(mean_edge) / TRANSITION_EDGE_REFERENCE_RAD)
        transition_flexion = TRANSITION_FLEXION_M * transition_pulse
    return cycle_flexion, brake_flexion, transition_flexion


def construct_pose_root(record: SkierFrameRecord) -> RootGeometry:
    """Replace CS2 proof height with the versioned CS3 authoritative pelvis height."""
    if not isinstance(record, SkierFrameRecord):
        raise TypeError("record: expected SkierFrameRecord")
    phase = _record_phase(record)
    cycle, brake, transition = _pose_flexion_terms(record, phase)
    pelvis_height = (
        PROOF_STANDING_PELVIS_HEIGHT_M
        - PROOF_CROUCH_DROP_M * record.evaluated_maneuver.targets.crouch
        - cycle
        - brake
        - transition
    )
    if pelvis_height <= 0.0:
        raise ValueError("CS3 pelvis height must remain positive")
    ground = record.root.ground_point_world_m
    ground_transform = record.root.T_world_from_groundroot
    normal = -ground_transform[:3, 2]
    pelvis = ground + pelvis_height * normal
    armature = ground_transform.copy()
    armature[:3, 3] = pelvis
    return RootGeometry(
        ground_point_world_m=ground,
        pelvis_point_world_m=pelvis,
        tangent_world=record.root.tangent_world,
        lateral_world=record.root.lateral_world,
        T_world_from_groundroot=ground_transform,
        T_world_from_armature=armature,
    )


def evaluate_pose(record: SkierFrameRecord, manifest: RigManifest) -> PoseEvaluation:
    """Evaluate one root-free absolute pose and stateless binding IK solution."""
    if not isinstance(record, SkierFrameRecord) or not isinstance(manifest, RigManifest):
        raise TypeError("record/manifest: expected SkierFrameRecord and RigManifest")
    phase = _record_phase(record)
    phase_angle = 2.0 * math.pi * phase
    sine = math.sin(phase_angle)
    cosine = math.cos(phase_angle)
    crouch = record.evaluated_maneuver.targets.crouch
    maneuver_type = record.evaluated_maneuver.maneuver_type
    lean = float(np.clip(record.gross_lean_rad, -MAX_BODY_LEAN_RAD, MAX_BODY_LEAN_RAD))
    _cycle_flexion, _brake_flexion, transition_flexion = _pose_flexion_terms(record, phase)
    pose_root = construct_pose_root(record)

    rest = {semantic: manifest.bone(semantic).rest_matrix_root_m for semantic in TRACKED_SEMANTICS}
    parent_rests: dict[str, np.ndarray] = {}
    for semantic in TRACKED_SEMANTICS:
        bone = manifest.bone(semantic)
        if bone.parent_name not in INTERMEDIATE_PARENT_NAMES:
            continue
        prior = parent_rests.setdefault(bone.parent_name, bone.parent_rest_matrix_root_m)
        if not np.array_equal(prior, bone.parent_rest_matrix_root_m):
            raise ValueError(f"manifested parent rest drift for {bone.parent_name!r}")
    if set(parent_rests) != set(INTERMEDIATE_PARENT_NAMES):
        raise ValueError("manifest does not close the evaluated parent set")
    pelvis_rest = rest["pelvis"][:3, 3]
    pelvis = pelvis_rest.copy()

    lower_rotation = _axis_rotation("x", 0.35 * lean)
    torso_rotation = _axis_rotation("x", 0.75 * lean) @ _axis_rotation("y", -0.28 * crouch)

    transforms: dict[str, np.ndarray] = {}
    transforms["pelvis"] = rest["pelvis"].copy()
    chest_position = rest["chest"][:3, 3].copy()
    transforms["chest"] = _transform(torso_rotation @ rest["chest"][:3, :3], chest_position)
    neck_position = chest_position + torso_rotation @ (
        parent_rests["neck_01"][:3, 3] - rest["chest"][:3, 3]
    )
    head_position = neck_position + torso_rotation @ (
        rest["head"][:3, 3] - parent_rests["neck_01"][:3, 3]
    )
    transforms["head"] = _transform(torso_rotation @ rest["head"][:3, :3], head_position)

    spread = 1.35 if maneuver_type is ManeuverType.BRAKE else 1.0
    for side, side_sign in (("left", -1.0), ("right", 1.0)):
        shoulder_semantic = f"{side}_shoulder"
        elbow_semantic = f"{side}_elbow"
        wrist_semantic = f"{side}_wrist"
        shoulder = chest_position + torso_rotation @ (
            rest[shoulder_semantic][:3, 3] - rest["chest"][:3, 3]
        )
        upper_length = float(
            np.linalg.norm(rest[elbow_semantic][:3, 3] - rest[shoulder_semantic][:3, 3])
        )
        lower_length = float(np.linalg.norm(rest[wrist_semantic][:3, 3] - rest[elbow_semantic][:3, 3]))
        upper_direction = torso_rotation @ _normalize(
            "upper-arm direction",
            np.array(
                [0.42 * sine + 0.55 * crouch, side_sign * spread, 0.22 * cosine + 0.22 * crouch],
                dtype=np.float64,
            ),
        )
        lower_direction = torso_rotation @ _normalize(
            "lower-arm direction",
            np.array(
                [0.60 * sine + 0.75 * crouch, side_sign * 0.70 * spread, 0.32 * cosine + 0.32 * crouch],
                dtype=np.float64,
            ),
        )
        elbow = shoulder + upper_length * upper_direction
        wrist = elbow + lower_length * lower_direction
        hand_direction = torso_rotation @ _normalize(
            "hand direction",
            np.array([0.80 + 0.15 * sine, side_sign * 0.20, 0.25], dtype=np.float64),
        )
        transforms[shoulder_semantic] = _transform(
            _aligned_bone_rotation(rest[shoulder_semantic][:3, :3], upper_direction),
            shoulder,
        )
        transforms[elbow_semantic] = _transform(
            _aligned_bone_rotation(rest[elbow_semantic][:3, :3], lower_direction),
            elbow,
        )
        transforms[wrist_semantic] = _transform(
            _aligned_bone_rotation(rest[wrist_semantic][:3, :3], hand_direction),
            wrist,
        )

    root_rotation_world = pose_root.T_world_from_armature[:3, :3]
    root_origin_world = pose_root.T_world_from_armature[:3, 3]
    binding_targets: list[np.ndarray] = []
    boot_object_frames: list[np.ndarray] = []
    reach_ratios: list[float] = []
    knee_flexions: list[float] = []
    for side, side_sign in (("left", -1.0), ("right", 1.0)):
        hip_semantic = f"{side}_hip"
        knee_semantic = f"{side}_knee"
        ankle_semantic = f"{side}_ankle"
        boot_semantic = f"{side}_boot"
        ski = getattr(record.skis, side)
        binding = root_rotation_world.T @ (ski.binding_origin_world_m - root_origin_world)
        ski_frame = root_rotation_world.T @ ski.realized_F_world_from_ski[:3, :3]
        _validate_rotation(f"{side} ski frame in root", ski_frame)

        boot_rest = rest[boot_semantic]
        foot_rest = rest[ankle_semantic]
        desired_boot = _transform(ski_frame @ boot_rest[:3, :3], binding)
        foot_from_boot_rest = np.linalg.inv(foot_rest) @ boot_rest
        desired_foot = desired_boot @ np.linalg.inv(foot_from_boot_rest)
        ankle = desired_foot[:3, 3]

        hip = pelvis + lower_rotation @ (rest[hip_semantic][:3, 3] - pelvis_rest)
        hip += lower_rotation @ np.array([0.03 * crouch, 0.0, 0.0], dtype=np.float64)
        upper_length = float(np.linalg.norm(rest[knee_semantic][:3, 3] - rest[hip_semantic][:3, 3]))
        lower_length = float(np.linalg.norm(rest[ankle_semantic][:3, 3] - rest[knee_semantic][:3, 3]))
        pole = lower_rotation @ np.array([1.0, 0.12 * side_sign, 0.0], dtype=np.float64)
        knee, reach_ratio, knee_flexion = _two_bone_solve(
            hip,
            ankle,
            upper_length,
            lower_length,
            pole,
        )
        transforms[hip_semantic] = _transform(
            _aligned_bone_rotation(rest[hip_semantic][:3, :3], knee - hip),
            hip,
        )
        transforms[knee_semantic] = _transform(
            _aligned_bone_rotation(rest[knee_semantic][:3, :3], ankle - knee),
            knee,
        )
        transforms[ankle_semantic] = desired_foot
        transforms[boot_semantic] = desired_boot
        binding_targets.append(binding)
        boot_object_frames.append(desired_boot[:3, :3] @ boot_rest[:3, :3].T)
        reach_ratios.append(reach_ratio)
        knee_flexions.append(knee_flexion)

    parent_poses = {
        "root": parent_rests["root"],
        "spine_02": parent_rests["spine_02"],
        "neck_01": _transform(
            torso_rotation @ parent_rests["neck_01"][:3, :3],
            neck_position,
        ),
        "clavicle_l": _transform(
            torso_rotation @ parent_rests["clavicle_l"][:3, :3],
            chest_position
            + torso_rotation @ (parent_rests["clavicle_l"][:3, 3] - rest["chest"][:3, 3]),
        ),
        "clavicle_r": _transform(
            torso_rotation @ parent_rests["clavicle_r"][:3, :3],
            chest_position
            + torso_rotation @ (parent_rests["clavicle_r"][:3, 3] - rest["chest"][:3, 3]),
        ),
    }

    reverse_bone_mapping = {name: semantic for semantic, name in BONE_BY_SEMANTIC.items()}
    local_transforms: dict[str, np.ndarray] = {}
    for semantic in TRACKED_SEMANTICS:
        manifested = manifest.bone(semantic)
        parent_rest = manifested.parent_rest_matrix_root_m
        parent_name = manifested.parent_name
        if parent_name in reverse_bone_mapping:
            parent_pose = transforms[reverse_bone_mapping[parent_name]]
        elif parent_name in parent_poses:
            parent_pose = parent_poses[parent_name]
        else:
            raise ValueError(f"unsupported evaluated parent closure: {parent_name!r}")
        local_pose = np.linalg.inv(parent_pose) @ transforms[semantic]
        local_rest = np.linalg.inv(parent_rest) @ rest[semantic]
        delta = local_pose @ np.linalg.inv(local_rest)
        left, _singular, right_t = np.linalg.svd(delta[:3, :3])
        delta[:3, :3] = left @ right_t
        if float(np.linalg.det(delta[:3, :3])) < 0.0:
            left[:, -1] *= -1.0
            delta[:3, :3] = left @ right_t
        delta[3] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        local_transforms[semantic] = delta

    ordered = np.stack([transforms[semantic] for semantic in TRACKED_SEMANTICS]).astype(np.float64)
    ordered_local = np.stack([local_transforms[semantic] for semantic in TRACKED_SEMANTICS]).astype(np.float64)
    positions = ordered[:, :3, 3].copy()
    armature_height = float(np.linalg.norm(pose_root.pelvis_point_world_m - pose_root.ground_point_world_m))
    shoulder_z = 0.5 * (
        transforms["left_shoulder"][2, 3] + transforms["right_shoulder"][2, 3]
    )
    return PoseEvaluation(
        absolute_tick=record.state.absolute_tick,
        rig_manifest_sha256=manifest.canonical_sha256(),
        source_skier_digest=record.skier_digest(),
        bone_names=tuple(BONE_BY_SEMANTIC[semantic] for semantic in TRACKED_SEMANTICS),
        parent_bone_names=INTERMEDIATE_PARENT_NAMES,
        T_world_from_armature=pose_root.T_world_from_armature,
        T_root_from_parent_bone=np.stack(
            [parent_poses[name] for name in INTERMEDIATE_PARENT_NAMES]
        ).astype(np.float64),
        T_root_from_bone=ordered,
        local_bone_transforms=ordered_local,
        tracked_joint_positions_root_m=positions,
        binding_target_root_m=np.stack(binding_targets).astype(np.float64),
        F_root_from_binding=np.stack(
            [
                root_rotation_world.T @ getattr(record.skis, side).realized_F_world_from_ski[:3, :3]
                for side in SIDE_ORDER
            ]
        ).astype(np.float64),
        F_root_from_boot_object=np.stack(boot_object_frames).astype(np.float64),
        ik_reach_ratio=np.array(reach_ratios, dtype=np.float64),
        knee_flexion_rad=np.array(knee_flexions, dtype=np.float64),
        pelvis_height_m=armature_height,
        shoulder_height_m=armature_height - shoulder_z,
        transition_flexion_m=transition_flexion,
    )


def record_with_pose(record: SkierFrameRecord, pose: PoseEvaluation) -> SkierFrameRecord:
    """Return a canonical record whose digest contains the evaluated body pose."""
    if not isinstance(record, SkierFrameRecord) or not isinstance(pose, PoseEvaluation):
        raise TypeError("record/pose: expected SkierFrameRecord and PoseEvaluation")
    if record.state.absolute_tick != pose.absolute_tick:
        raise ValueError("record/pose: absolute tick mismatch")
    if record.skier_digest() != pose.source_skier_digest:
        raise ValueError("record/pose: pose was evaluated from a different source skier digest")
    root = construct_pose_root(record)
    if not np.array_equal(root.T_world_from_armature, pose.T_world_from_armature):
        raise ValueError("record/pose: authoritative armature root mismatch")
    state = replace(
        record.state,
        tracked_joint_positions_root_m=pose.tracked_joint_positions_root_m,
        local_bone_transforms=pose.local_bone_transforms,
    )
    return replace(
        record,
        state=state,
        root=root,
        root_schema_version=SKIER_POSE_ROOT_SCHEMA_VERSION,
    )


def reconstruct_pose_bones_from_local(pose: PoseEvaluation, manifest: RigManifest) -> np.ndarray:
    """Reconstruct root-global bones from the digest-facing parent-rest deltas."""
    if not isinstance(pose, PoseEvaluation) or not isinstance(manifest, RigManifest):
        raise TypeError("pose/manifest: expected PoseEvaluation/RigManifest")
    if pose.rig_manifest_sha256 != manifest.canonical_sha256():
        raise ValueError("pose/manifest: rig manifest digest mismatch")
    parent_globals = {
        name: pose.T_root_from_parent_bone[index]
        for index, name in enumerate(pose.parent_bone_names)
    }
    reconstructed: dict[str, np.ndarray] = {}
    remaining = set(TRACKED_SEMANTICS)
    while remaining:
        progressed = False
        for semantic in TRACKED_SEMANTICS:
            if semantic not in remaining:
                continue
            bone = manifest.bone(semantic)
            if bone.parent_name in parent_globals:
                parent_pose = parent_globals[bone.parent_name]
            elif bone.parent_name in BONE_BY_SEMANTIC.values():
                parent_semantic = next(
                    item for item, blender_name in BONE_BY_SEMANTIC.items() if blender_name == bone.parent_name
                )
                if parent_semantic not in reconstructed:
                    continue
                parent_pose = reconstructed[parent_semantic]
            else:
                raise ValueError(f"unsupported evaluated parent closure: {bone.parent_name!r}")
            local_rest = np.linalg.inv(bone.parent_rest_matrix_root_m) @ bone.rest_matrix_root_m
            index = TRACKED_SEMANTICS.index(semantic)
            reconstructed[semantic] = parent_pose @ pose.local_bone_transforms[index] @ local_rest
            remaining.remove(semantic)
            progressed = True
        if not progressed:
            raise ValueError("evaluated local transform graph is not reconstructible")
    ordered = np.stack([reconstructed[semantic] for semantic in TRACKED_SEMANTICS]).astype(np.float64)
    contiguous = np.array(ordered, dtype=np.dtype("<f8"), order="C", copy=True)
    contiguous[contiguous == 0.0] = 0.0
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=np.dtype("<f8")).reshape(ordered.shape)


def pose_local_reconstruction_residuals(
    pose: PoseEvaluation,
    manifest: RigManifest,
) -> tuple[float, float]:
    """Return maximum position/Frobenius-rotation residuals from local reconstruction."""
    reconstructed = reconstruct_pose_bones_from_local(pose, manifest)
    position = float(np.max(np.linalg.norm(reconstructed[:, :3, 3] - pose.T_root_from_bone[:, :3, 3], axis=1)))
    rotation = float(
        np.max(
            np.linalg.norm(
                reconstructed[:, :3, :3] - pose.T_root_from_bone[:, :3, :3],
                axis=(1, 2),
            )
        )
    )
    return position, rotation


@dataclass(frozen=True)
class AnimationAmplitudeMetrics:
    upper_joint_peak_to_peak_m: float
    noncontact_joint_time_rms_m: float
    driven_knee_range_rad: float


def animation_amplitude_metrics(samples: tuple[PoseEvaluation, ...]) -> AnimationAmplitudeMetrics:
    """Compute the report's manifested, root-removed animation-amplitude metrics."""
    if len(samples) < 2 or any(not isinstance(sample, PoseEvaluation) for sample in samples):
        raise ValueError("samples: expected at least two PoseEvaluation values")
    positions = np.stack([sample.tracked_joint_positions_root_m for sample in samples])
    upper_indices = [
        TRACKED_SEMANTICS.index(semantic)
        for semantic in (
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
        )
    ]
    upper_peak = 0.0
    for index in upper_indices:
        deltas = positions[:, index, None, :] - positions[None, :, index, :]
        upper_peak = max(upper_peak, float(np.max(np.linalg.norm(deltas, axis=-1))))
    included = [
        index
        for index, semantic in enumerate(TRACKED_SEMANTICS)
        if semantic not in AMPLITUDE_EXCLUDED_SEMANTICS
    ]
    selected = positions[:, included]
    mean = selected.mean(axis=0, keepdims=True)
    rms = float(np.sqrt(np.mean(np.sum((selected - mean) ** 2, axis=-1))))
    knee = np.stack([sample.knee_flexion_rad for sample in samples])
    knee_range = float(np.max(np.ptp(knee, axis=0)))
    return AnimationAmplitudeMetrics(upper_peak, rms, knee_range)


def sample_authored_cycle(record: SkierFrameRecord, manifest: RigManifest) -> tuple[PoseEvaluation, ...]:
    """Sample the one-second root-free clip at the exact five 5-Hz phases."""
    samples: list[PoseEvaluation] = []
    for tick in range(5):
        state = replace(record.state, absolute_tick=tick)
        animation = replace(record.animation, animation_phase=(tick * FIXED_DT_SECONDS) % 1.0)
        samples.append(evaluate_pose(replace(record, state=state, animation=animation), manifest))
    return tuple(samples)


@dataclass(frozen=True)
class MirroredPoseMetrics:
    rotation_rms_rad: float
    rotation_max_rad: float
    joint_position_rms_m: float


def mirrored_pose_metrics(
    left_pose: PoseEvaluation,
    right_pose: PoseEvaluation,
    manifest: RigManifest,
) -> MirroredPoseMetrics:
    """Compare parent-rest joint deltas/displacements after sagittal reflection."""
    if not isinstance(left_pose, PoseEvaluation) or not isinstance(right_pose, PoseEvaluation):
        raise TypeError("left_pose/right_pose: expected PoseEvaluation")
    reflection = np.diag([1.0, -1.0, 1.0])
    counterpart = {semantic: semantic for semantic in TRACKED_SEMANTICS}
    for left, right in LEFT_RIGHT_PAIRS:
        counterpart[left] = right
        counterpart[right] = left
    rotation_errors: list[float] = []
    position_errors: list[float] = []
    for semantic in TRACKED_SEMANTICS:
        other = counterpart[semantic]
        index = TRACKED_SEMANTICS.index(semantic)
        other_index = TRACKED_SEMANTICS.index(other)
        left_transform = left_pose.transform(semantic)
        right_transform = right_pose.transform(other)
        left_rest = manifest.bone(semantic).rest_matrix_root_m
        right_rest = manifest.bone(other).rest_matrix_root_m
        left_parent_rest = manifest.bone(semantic).parent_rest_matrix_root_m[:3, :3]
        right_parent_rest = manifest.bone(other).parent_rest_matrix_root_m[:3, :3]
        left_local_delta = left_pose.local_bone_transforms[index, :3, :3]
        right_local_delta = right_pose.local_bone_transforms[other_index, :3, :3]
        left_delta_root = left_parent_rest @ left_local_delta @ left_parent_rest.T
        right_delta_root = right_parent_rest @ right_local_delta @ right_parent_rest.T
        mirrored_delta = reflection @ right_delta_root @ reflection
        rotation_errors.append(rotation_geodesic_angle(left_delta_root, mirrored_delta))
        left_displacement = left_transform[:3, 3] - left_rest[:3, 3]
        right_displacement = right_transform[:3, 3] - right_rest[:3, 3]
        position_errors.append(
            float(np.linalg.norm(left_displacement - reflection @ right_displacement))
        )
    rotations = np.array(rotation_errors, dtype=np.float64)
    positions = np.array(position_errors, dtype=np.float64)
    return MirroredPoseMetrics(
        rotation_rms_rad=float(np.sqrt(np.mean(rotations**2))),
        rotation_max_rad=float(np.max(rotations)),
        joint_position_rms_m=float(np.sqrt(np.mean(positions**2))),
    )


def binding_pose_residuals(pose: PoseEvaluation) -> dict[str, float]:
    """Return exact boot-object position/frame residuals against ski bindings."""
    residuals: dict[str, float] = {}
    for index, side in enumerate(SIDE_ORDER):
        boot = pose.transform(f"{side}_boot")
        residuals[f"{side}_position_m"] = float(
            np.linalg.norm(boot[:3, 3] - pose.binding_target_root_m[index])
        )
        residuals[f"{side}_orientation_rad"] = rotation_geodesic_angle(
            pose.F_root_from_boot_object[index],
            pose.F_root_from_binding[index],
        )
    return residuals


def ik_segment_residuals(pose: PoseEvaluation, manifest: RigManifest) -> dict[str, float]:
    """Return non-stretch upper/lower leg length residuals in root-local metres."""
    if not isinstance(pose, PoseEvaluation) or not isinstance(manifest, RigManifest):
        raise TypeError("pose/manifest: expected PoseEvaluation/RigManifest")
    residuals: dict[str, float] = {}
    for side in SIDE_ORDER:
        hip = pose.transform(f"{side}_hip")[:3, 3]
        knee = pose.transform(f"{side}_knee")[:3, 3]
        ankle = pose.transform(f"{side}_ankle")[:3, 3]
        rest_hip = manifest.bone(f"{side}_hip").rest_matrix_root_m[:3, 3]
        rest_knee = manifest.bone(f"{side}_knee").rest_matrix_root_m[:3, 3]
        rest_ankle = manifest.bone(f"{side}_ankle").rest_matrix_root_m[:3, 3]
        residuals[f"{side}_upper_m"] = abs(
            float(np.linalg.norm(knee - hip)) - float(np.linalg.norm(rest_knee - rest_hip))
        )
        residuals[f"{side}_lower_m"] = abs(
            float(np.linalg.norm(ankle - knee)) - float(np.linalg.norm(rest_ankle - rest_knee))
        )
    return residuals


def pose_root_residuals(source: SkierFrameRecord, posed: SkierFrameRecord) -> dict[str, float]:
    """Independently audit the CS3 root law and unchanged CS2 ground/ski geometry."""
    if not isinstance(source, SkierFrameRecord) or not isinstance(posed, SkierFrameRecord):
        raise TypeError("source/posed: expected SkierFrameRecord values")
    if posed.root_schema_version != SKIER_POSE_ROOT_SCHEMA_VERSION:
        raise ValueError("posed: expected the CS3 pose-root schema")
    phase = _record_phase(source)
    cycle = CYCLE_FLEXION_M * 0.5 * (1.0 + math.cos(2.0 * math.pi * phase))
    brake = BRAKE_FLEXION_M if source.evaluated_maneuver.maneuver_type is ManeuverType.BRAKE else 0.0
    transition = 0.0
    if source.evaluated_maneuver.maneuver_type is ManeuverType.TRANSITION:
        mean_edge = 0.5 * (
            source.evaluated_maneuver.targets.left_edge_rad
            + source.evaluated_maneuver.targets.right_edge_rad
        )
        transition = TRANSITION_FLEXION_M * max(
            0.0,
            1.0 - abs(mean_edge) / TRANSITION_EDGE_REFERENCE_RAD,
        )
    expected_height = (
        PROOF_STANDING_PELVIS_HEIGHT_M
        - PROOF_CROUCH_DROP_M * source.evaluated_maneuver.targets.crouch
        - cycle
        - brake
        - transition
    )
    ground = source.root.ground_point_world_m
    normal = -source.root.T_world_from_groundroot[:3, 2]
    expected_pelvis = ground + expected_height * normal
    residuals = {
        "pelvis_position_m": float(np.linalg.norm(posed.root.pelvis_point_world_m - expected_pelvis)),
        "armature_origin_m": float(
            np.linalg.norm(posed.root.T_world_from_armature[:3, 3] - expected_pelvis)
        ),
        "armature_frame_rad": rotation_geodesic_angle(
            posed.root.T_world_from_armature[:3, :3],
            source.root.T_world_from_groundroot[:3, :3],
        ),
        "pelvis_height_m": abs(
            float(np.dot(posed.root.pelvis_point_world_m - ground, normal)) - expected_height
        ),
        "ground_root_max_abs": float(
            np.max(np.abs(posed.root.T_world_from_groundroot - source.root.T_world_from_groundroot))
        ),
    }
    ski_position_residual = 0.0
    ski_frame_residual = 0.0
    for side in SIDE_ORDER:
        source_ski = getattr(source.skis, side)
        posed_ski = getattr(posed.skis, side)
        for field in (
            "centerline_origin_world_m",
            "base_origin_world_m",
            "binding_origin_world_m",
            "contact_origin_world_m",
        ):
            ski_position_residual = max(
                ski_position_residual,
                float(np.linalg.norm(getattr(posed_ski, field) - getattr(source_ski, field))),
            )
        ski_frame_residual = max(
            ski_frame_residual,
            rotation_geodesic_angle(
                posed_ski.realized_F_world_from_ski[:3, :3],
                source_ski.realized_F_world_from_ski[:3, :3],
            ),
        )
    residuals["ski_position_m"] = ski_position_residual
    residuals["ski_frame_rad"] = ski_frame_residual
    return residuals


def authored_carve_schedule(sign: int) -> ManeuverSchedule:
    """Return the absolute-tick quintic schedule used by the carve-cycle proof."""
    if isinstance(sign, bool) or sign not in (-1, 1):
        raise ValueError("sign: expected -1 or +1")
    edge = sign * CARVE_CYCLE_TARGET_EDGE_RAD
    maneuver_type = ManeuverType.CARVE_LEFT if sign < 0 else ManeuverType.CARVE_RIGHT
    targets = ManeuverTargets(
        curvature_1_m=ideal_carve_curvature(edge),
        left_edge_rad=edge,
        right_edge_rad=edge,
        left_attack_rad=0.0,
        right_attack_rad=0.0,
        crouch=0.2,
        drag_area_m2=drag_area_from_crouch(0.2),
        brake_cap_m_s2=0.0,
    )
    side = "left" if sign < 0 else "right"
    return ManeuverSchedule(
        baseline_targets=neutral_targets(),
        records=(
            ManeuverRecord(
                maneuver_id=f"authoritative-carve-{side}-cycle-v2",
                maneuver_type=maneuver_type,
                continuation_law_id="authoritative-carve-cycle-quintic-v2",
                start_tick=0,
                ramp_in_ticks=CARVE_CYCLE_RAMP_IN_TICKS,
                hold_ticks=CARVE_CYCLE_HOLD_TICKS,
                ramp_out_ticks=CARVE_CYCLE_RAMP_OUT_TICKS,
                targets=targets,
            ),
        ),
    )


def authored_carve_cycle(sign: int) -> tuple[SkierFrameRecord, ...]:
    """Construct the 11 actual fixed-root records spanning the two-second cycle.

    Every sample goes through :class:`ManeuverSchedule` evaluation, the frozen
    root law, and the analytic ski/contact constructor via :func:`frame_record`.
    Root position, heading, and speed are held fixed deliberately: this is the
    root-removed animation/equipment feasibility proof, not a second trajectory
    integrator or a mutation of the frozen CS2 fixtures.
    """
    schedule = authored_carve_schedule(sign)
    slope = default_slope_frame()
    parameters = SkierParameters()
    records: list[SkierFrameRecord] = []
    for tick in range(CARVE_CYCLE_SAMPLE_COUNT):
        evaluated = schedule.evaluate(tick)
        state = SkierState(
            absolute_tick=tick,
            x_m=0.0,
            y_m=0.0,
            heading_rad=0.0,
            speed_m_s=CARVE_CYCLE_SPEED_M_S,
            curvature_1_m=evaluated.targets.curvature_1_m,
            tracked_joint_positions_root_m=np.empty((0, 3), dtype=np.float64),
            local_bone_transforms=np.empty((0, 4, 4), dtype=np.float64),
            randomness_seed=CARVE_CYCLE_RANDOMNESS_SEED,
        )
        records.append(frame_record(state, schedule, slope, parameters))
    return tuple(records)


def _pose_transform_export_fields(pose: PoseEvaluation) -> dict[str, object]:
    return {
        "T_world_from_armature": pose.T_world_from_armature.tolist(),
        "parent_bone_names": list(pose.parent_bone_names),
        "T_root_from_parent_bone": pose.T_root_from_parent_bone.tolist(),
        "bone_names": list(pose.bone_names),
        "T_root_from_bone": pose.T_root_from_bone.tolist(),
        "local_transform_semantics": LOCAL_BONE_TRANSFORM_SEMANTICS,
        "local_bone_transforms": pose.local_bone_transforms.tolist(),
    }


def _ski_export_fields(record: SkierFrameRecord) -> dict[str, object]:
    result: dict[str, object] = {}
    for side in SIDE_ORDER:
        ski = getattr(record.skis, side)
        result[side] = {
            "attack_rad": ski.attack_rad,
            "edge_rad": ski.edge_rad,
            "realized_attack_rad": ski.realized_attack_rad,
            "realized_edge_rad": ski.realized_edge_rad,
            "centerline_origin_world_m": ski.centerline_origin_world_m.tolist(),
            "base_origin_world_m": ski.base_origin_world_m.tolist(),
            "binding_origin_world_m": ski.binding_origin_world_m.tolist(),
            "contact_origin_world_m": ski.contact_origin_world_m.tolist(),
            "commanded_F_world_from_ski": ski.commanded_F_world_from_ski.tolist(),
            "realized_F_world_from_ski": ski.realized_F_world_from_ski.tolist(),
        }
    return result


def _fixture_pose_export_rows(manifest: RigManifest) -> list[dict[str, object]]:
    from vllatent.sim.skier_fixtures import canonical_skier_fixtures

    rows: list[dict[str, object]] = []
    for fixture in canonical_skier_fixtures():
        for record_index, source_record in enumerate(fixture.records()):
            pose = evaluate_pose(source_record, manifest)
            posed_record = record_with_pose(source_record, pose)
            rows.append(
                {
                    "fixture_id": fixture.fixture_id,
                    "record_index": record_index,
                    "absolute_tick": source_record.state.absolute_tick,
                    "source_record_sha256": hashlib.sha256(
                        source_record.canonical_bytes()
                    ).hexdigest(),
                    "source_skier_digest": pose.source_skier_digest,
                    "pose_sha256": pose.canonical_sha256(),
                    "pose_record_canonical_hex": pose.canonical_bytes().hex(),
                    "posed_record_sha256": hashlib.sha256(
                        posed_record.canonical_bytes()
                    ).hexdigest(),
                    "posed_skier_digest": posed_record.skier_digest(),
                    "posed_record_canonical_hex": posed_record.canonical_bytes().hex(),
                    **_pose_transform_export_fields(pose),
                }
            )
    return rows


def _carve_cycle_export(
    manifest: RigManifest,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    descriptors: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for sign in (-1, 1):
        side = "left" if sign < 0 else "right"
        cycle_id = f"authoritative-carve-{side}-cycle-v2"
        schedule = authored_carve_schedule(sign)
        records = authored_carve_cycle(sign)
        descriptors.append(
            {
                "cycle_id": cycle_id,
                "sign": sign,
                "sample_rate_hz": 1.0 / FIXED_DT_SECONDS,
                "duration_seconds": (CARVE_CYCLE_SAMPLE_COUNT - 1) * FIXED_DT_SECONDS,
                "sample_count": CARVE_CYCLE_SAMPLE_COUNT,
                "sample_ticks": list(range(CARVE_CYCLE_SAMPLE_COUNT)),
                "root_motion_policy": "fixed_position_heading_speed_root_removed_proof",
                "schedule": schedule.payload(),
                "schedule_sha256": hashlib.sha256(canonical_bytes(schedule.payload())).hexdigest(),
            }
        )
        for sample_index, source_record in enumerate(records):
            pose = evaluate_pose(source_record, manifest)
            posed_record = record_with_pose(source_record, pose)
            target = source_record.evaluated_maneuver.targets
            rows.append(
                {
                    "cycle_id": cycle_id,
                    "sign": sign,
                    "sample_index": sample_index,
                    "absolute_tick": source_record.state.absolute_tick,
                    "elapsed_seconds": sample_index * FIXED_DT_SECONDS,
                    "maneuver_phase": source_record.evaluated_maneuver.maneuver_phase,
                    "maneuver_phase_name": source_record.evaluated_maneuver.phase_name,
                    "maneuver_weight": source_record.evaluated_maneuver.weight,
                    "target_curvature_1_m": target.curvature_1_m,
                    "target_left_edge_rad": target.left_edge_rad,
                    "target_right_edge_rad": target.right_edge_rad,
                    "gross_lean_rad": source_record.gross_lean_rad,
                    "source_record_sha256": hashlib.sha256(source_record.canonical_bytes()).hexdigest(),
                    "source_skier_digest": source_record.skier_digest(),
                    "pose_sha256": pose.canonical_sha256(),
                    "posed_record_sha256": hashlib.sha256(posed_record.canonical_bytes()).hexdigest(),
                    "posed_skier_digest": posed_record.skier_digest(),
                    "skis": _ski_export_fields(source_record),
                    **_pose_transform_export_fields(pose),
                }
            )
    return descriptors, rows


def _canonical_pose_table_payload(manifest: RigManifest) -> dict[str, object]:
    fixture_rows = _fixture_pose_export_rows(manifest)
    cycle_descriptors, cycle_rows = _carve_cycle_export(manifest)
    return {
        "schema_version": POSE_SCHEMA_VERSION,
        "pose_table_export_schema_version": POSE_TABLE_EXPORT_SCHEMA_VERSION,
        "rig_manifest_sha256": manifest.canonical_sha256(),
        "fixture_rows": fixture_rows,
        "carve_cycle_schema_version": CARVE_CYCLE_SCHEMA_VERSION,
        "carve_cycles": cycle_descriptors,
        "carve_cycle_rows": cycle_rows,
    }


def canonical_pose_table_sha256(manifest: RigManifest) -> str:
    """Hash the fixture poses and actual schedule/equipment carve-cycle rows."""
    if not isinstance(manifest, RigManifest):
        raise TypeError("manifest: expected RigManifest")
    return hashlib.sha256(canonical_bytes(_canonical_pose_table_payload(manifest))).hexdigest()


def canonical_pose_export_payload(manifest: RigManifest) -> dict[str, object]:
    """Build the Blender-bound fixture table and 22 actual carve-cycle rows."""
    if not isinstance(manifest, RigManifest):
        raise TypeError("manifest: expected RigManifest")
    from vllatent.sim.skier_fixtures import canonical_skier_fixtures

    fixtures = canonical_skier_fixtures()
    table = _canonical_pose_table_payload(manifest)
    rows = table["fixture_rows"]
    cycle_descriptors = table["carve_cycles"]
    cycle_rows = table["carve_cycle_rows"]
    assert isinstance(rows, list)
    assert isinstance(cycle_descriptors, list)
    assert isinstance(cycle_rows, list)
    return {
        "schema_version": POSE_TABLE_EXPORT_SCHEMA_VERSION,
        "rig_manifest_canonical_sha256": manifest.canonical_sha256(),
        "canonical_pose_table_sha256": hashlib.sha256(canonical_bytes(table)).hexdigest(),
        "fixture_count": len(fixtures),
        "sample_count": len(rows),
        "rows": rows,
        "carve_cycle_schema_version": CARVE_CYCLE_SCHEMA_VERSION,
        "carve_cycle_count": len(cycle_descriptors),
        "carve_cycle_sample_count": len(cycle_rows),
        "carve_cycles": cycle_descriptors,
        "carve_cycle_rows": cycle_rows,
    }


def canonical_pose_export_json_bytes(manifest: RigManifest) -> bytes:
    """Serialize the Blender-bound table with one tracked deterministic mapping."""
    return (
        json.dumps(
            canonical_pose_export_payload(manifest),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_pose_export_sha256(manifest: RigManifest) -> str:
    """Hash the exact authoritative bytes supplied to the isolated Blender process."""
    return hashlib.sha256(canonical_pose_export_json_bytes(manifest)).hexdigest()


__all__ = [
    "AnimationAmplitudeMetrics",
    "CARVE_CYCLE_HOLD_TICKS",
    "CARVE_CYCLE_RAMP_IN_TICKS",
    "CARVE_CYCLE_RAMP_OUT_TICKS",
    "CARVE_CYCLE_SAMPLE_COUNT",
    "CARVE_CYCLE_SCHEMA_VERSION",
    "CARVE_CYCLE_TARGET_EDGE_RAD",
    "IK_SCHEMA_VERSION",
    "INTERMEDIATE_PARENT_NAMES",
    "LOCAL_BONE_TRANSFORM_SEMANTICS",
    "MirroredPoseMetrics",
    "POSE_SCHEMA_VERSION",
    "POSE_TABLE_EXPORT_SCHEMA_VERSION",
    "POSE_TRANSFORM_SEMANTICS",
    "PINNED_CANONICAL_POSE_TABLE_SHA256",
    "PINNED_POSE_TABLE_EXPORT_SHA256",
    "PoseEvaluation",
    "animation_amplitude_metrics",
    "authored_carve_cycle",
    "authored_carve_schedule",
    "binding_pose_residuals",
    "canonical_pose_export_json_bytes",
    "canonical_pose_export_payload",
    "canonical_pose_export_sha256",
    "canonical_pose_table_sha256",
    "evaluate_pose",
    "ik_segment_residuals",
    "mirrored_pose_metrics",
    "pose_local_reconstruction_residuals",
    "pose_root_residuals",
    "reconstruct_pose_bones_from_local",
    "record_with_pose",
    "sample_authored_cycle",
]
