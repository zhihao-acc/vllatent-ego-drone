"""Focused PURE tests for the B3-CS1 causal camera contract."""

from __future__ import annotations

import ast
import importlib.util
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vllatent.sim.contracts import (
    BRANCH_IDS,
    COMMAND_FIELDS,
    DEPTH_OF_FIELD_ENABLED,
    FIXED_DT_SECONDS,
    FORWARD_SPEED_MAGNITUDE_M_S,
    HORIZON_STEPS,
    LATERAL_SPEED_MAGNITUDE_M_S,
    R_CAM_FROM_RIG,
    R_RIG_FROM_CAM,
    SKIER_ROOT_SCHEMA_VERSION,
    T_CAM_FROM_RIG_M,
    T_RIG_FROM_CAM_M,
    VERTICAL_SPEED_MAGNITUDE_M_S,
    YAW_RATE_MAGNITUDE_RAD_S,
    BranchId,
    BranchProgram,
    DatasetSplit,
    RootSiblingIdentity,
    camera_contract_sha256,
    canonical_branch_programs,
    canonical_bytes,
    canonical_skier_digest,
    default_camera_contract,
    expected_image_effect,
    program_by_id,
    sign_eligible,
    validate_sibling_group,
)


def _f64(values: object) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _program_kwargs() -> dict[str, object]:
    return {
        "branch_id": BranchId.ZERO,
        "requested_command": np.zeros((8, 4), dtype=np.float64),
        "dt_seconds": np.full(8, 0.2, dtype=np.float64),
        "record_valid": np.ones(8, dtype=np.bool_),
    }


def _digest_kwargs() -> dict[str, object]:
    def ski(side: str, lateral_m: float) -> dict[str, object]:
        return {
            "side": side,
            "attack_rad": 0.0,
            "edge_rad": 0.0,
            "centerline_origin_world_m": _f64([0.0, lateral_m, 0.0]),
            "base_origin_world_m": _f64([0.0, lateral_m, 0.0]),
            "binding_origin_world_m": _f64([0.0, lateral_m, 0.04]),
            "contact_origin_world_m": _f64([0.0, lateral_m, 0.0]),
            "target_F_world_from_ski": np.eye(3, dtype=np.float64),
            "realized_F_world_from_ski": np.eye(3, dtype=np.float64),
            "analytic_slip_longitudinal_lateral_m_s": _f64([8.0, 0.0]),
            "realized_slip_longitudinal_lateral_m_s": _f64([8.0, 0.0]),
            "realized_attack_rad": 0.0,
            "realized_edge_rad": 0.0,
            "frame_orientation_residual_rad": 0.0,
        }

    return {
        "root": {
            "schema_version": SKIER_ROOT_SCHEMA_VERSION,
            "absolute_tick": 7,
            "position_xy_m": _f64([1.0, 2.0]),
            "heading_rad": 0.3,
            "speed_m_s": 8.0,
            "acceleration_m_s2": 0.1,
            "curvature_1_m": 0.05,
            "omega_rad_s": 0.4,
            "gross_lean_rad": 0.2,
            "T_world_from_groundroot": np.eye(4, dtype=np.float64),
            "T_world_from_armature": np.eye(4, dtype=np.float64),
            "tracked_joint_positions_root_m": _f64([[0.0, 0.0, 0.0]]),
        },
        "skis": {
            "dimensions_m": _f64([1.7, 0.1, 0.015]),
            "stance_half_width_m": 0.16,
            "centerline_ordering_m": 0.32,
            "inner_tip_gap_m": 0.22,
            "left": ski("left", -0.16),
            "right": ski("right", 0.16),
        },
        "contacts": {
            "left_contact_origin_world_m": _f64([0.0, -0.16, 0.0]),
            "right_contact_origin_world_m": _f64([0.0, 0.16, 0.0]),
        },
        "phases": {
            "maneuver_id": "carve-right-v1",
            "maneuver_phase": 0.25,
            "animation_clip_ids": ("straight_high", "carve_right"),
            "animation_phase": 0.4,
            "animation_blend_weights": _f64([0.0, 1.0]),
        },
        "local_bone_transforms": np.empty((0, 4, 4), dtype=np.float64),
        "randomness": {"seed": 1729},
    }


def _uses_numpy_random(tree: ast.AST) -> bool:
    numpy_aliases = {"numpy"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "numpy":
                    numpy_aliases.add(alias.asname or alias.name)
                if alias.name.startswith("numpy.random"):
                    return True
        if isinstance(node, ast.ImportFrom):
            if node.module is not None and node.module.startswith("numpy.random"):
                return True
            if node.module == "numpy" and any(alias.name == "random" for alias in node.names):
                return True
    return any(
        isinstance(node, ast.Attribute)
        and node.attr == "random"
        and isinstance(node.value, ast.Name)
        and node.value.id in numpy_aliases
        for node in ast.walk(tree)
    )


def _import_roots(tree: ast.AST) -> set[str]:
    roots = {
        alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    }
    roots.update(
        node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    )
    return roots


def test_sim_modules_are_pure_and_forbid_renderer_or_model_imports() -> None:
    forbidden = {
        "airsim",
        "bpy",
        "cv2",
        "datetime",
        "projectairsim",
        "random",
        "secrets",
        "timm",
        "time",
        "torch",
        "torchvision",
        "transformers",
        "uuid",
    }
    allowed = set(sys.stdlib_module_names) | {"numpy", "vllatent", "yaml"}
    package_dir = Path(__file__).parents[1] / "vllatent" / "sim"
    for path in sorted(package_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = _import_roots(tree)
        assert imported <= allowed, (path, imported - allowed)
        assert imported.isdisjoint(forbidden), (path, imported & forbidden)
        assert not _uses_numpy_random(tree), path

    for module_name in ("vllatent.sim.contracts", "vllatent.sim.frames"):
        spec = importlib.util.find_spec(module_name)
        assert spec is not None and spec.origin is not None


@pytest.mark.parametrize(
    "source",
    (
        "import numpy as numerical\nnumerical.random.default_rng()",
        "import numpy.random as rng",
        "from numpy import random as rng",
        "from numpy.random import default_rng",
    ),
)
def test_pure_ast_guard_rejects_numpy_random_aliases(source: str) -> None:
    assert _uses_numpy_random(ast.parse(source))


@pytest.mark.parametrize("source", ("import pandas", "from requests import get", "import scipy.linalg"))
def test_pure_ast_guard_rejects_unlisted_third_party_imports(source: str) -> None:
    allowed = set(sys.stdlib_module_names) | {"numpy", "vllatent", "yaml"}
    assert _import_roots(ast.parse(source)) - allowed


def test_exact_nine_programs_and_float64_si_values_are_frozen() -> None:
    programs = canonical_branch_programs()
    assert tuple(program.branch_id.value for program in programs) == BRANCH_IDS
    assert BRANCH_IDS == (
        "zero",
        "yaw_plus",
        "yaw_minus",
        "forward_plus",
        "forward_minus",
        "lateral_plus",
        "lateral_minus",
        "vertical_plus",
        "vertical_minus",
    )
    assert len(programs) == 9
    assert COMMAND_FIELDS == (
        "v_forward_m_s",
        "v_right_m_s",
        "v_down_m_s",
        "yaw_rate_rad_s",
    )

    expected_rows = {
        BranchId.ZERO: (0.0, 0.0, 0.0, 0.0),
        BranchId.YAW_PLUS: (0.0, 0.0, 0.0, math.pi / 15.0),
        BranchId.YAW_MINUS: (0.0, 0.0, 0.0, -math.pi / 15.0),
        BranchId.FORWARD_PLUS: (1.0, 0.0, 0.0, 0.0),
        BranchId.FORWARD_MINUS: (-1.0, 0.0, 0.0, 0.0),
        BranchId.LATERAL_PLUS: (0.0, 0.75, 0.0, 0.0),
        BranchId.LATERAL_MINUS: (0.0, -0.75, 0.0, 0.0),
        BranchId.VERTICAL_PLUS: (0.0, 0.0, 0.5, 0.0),
        BranchId.VERTICAL_MINUS: (0.0, 0.0, -0.5, 0.0),
    }
    for program in programs:
        assert program.requested_command.shape == (8, 4)
        assert program.requested_command.dtype == np.dtype("<f8")
        assert program.dt_seconds.shape == (8,)
        assert program.dt_seconds.dtype == np.dtype("<f8")
        assert program.record_valid.dtype == np.bool_
        assert program.record_valid.tolist() == [True] * 8
        np.testing.assert_array_equal(
            program.requested_command,
            np.repeat(_f64(expected_rows[program.branch_id])[None, :], 8, axis=0),
        )
        np.testing.assert_array_equal(program.dt_seconds, np.full(8, 0.2))
        assert not np.signbit(program.requested_command[program.requested_command == 0.0]).any()
        assert not program.requested_command.flags.writeable
        assert not program.dt_seconds.flags.writeable
        with pytest.raises(ValueError, match="WRITEABLE"):
            program.requested_command.setflags(write=True)
        with pytest.raises(ValueError, match="WRITEABLE"):
            program.dt_seconds.setflags(write=True)
    assert YAW_RATE_MAGNITUDE_RAD_S == math.pi / 15.0
    assert FORWARD_SPEED_MAGNITUDE_M_S == 1.0
    assert LATERAL_SPEED_MAGNITUDE_M_S == 0.75
    assert VERTICAL_SPEED_MAGNITUDE_M_S == 0.5


def test_zero_and_pure_yaw_are_valid_and_model_inputs_keep_dt_separate() -> None:
    zero = program_by_id(BranchId.ZERO)
    yaw = program_by_id(BranchId.YAW_PLUS)
    assert zero.record_valid.all()
    assert yaw.record_valid.all()
    assert np.all(yaw.requested_command[:, :3] == 0.0)
    assert np.all(yaw.requested_command[:, 3] == math.pi / 15.0)

    commands, dt = yaw.model_inputs()
    assert commands.shape == (8, 4)
    assert dt.shape == (8,)
    assert commands.dtype == np.float32
    assert dt.dtype == np.float32
    np.testing.assert_array_equal(commands, yaw.requested_command.astype(np.float32))
    np.testing.assert_array_equal(dt, yaw.dt_seconds.astype(np.float32))
    assert not hasattr(yaw, "u_record")


@pytest.mark.parametrize(
    "changes, error",
    [
        ({"requested_command": np.zeros((8, 5), dtype=np.float64)}, ValueError),
        ({"requested_command": np.zeros((8, 6), dtype=np.float64)}, ValueError),
        ({"requested_command": np.zeros((8, 4), dtype=np.float32)}, ValueError),
        ({"requested_command": np.full((8, 4), np.nan, dtype=np.float64)}, ValueError),
        ({"dt_seconds": np.asarray(0.2, dtype=np.float64)}, ValueError),
        ({"dt_seconds": np.full(7, 0.2, dtype=np.float64)}, ValueError),
        ({"dt_seconds": np.full(8, 0.1, dtype=np.float64)}, ValueError),
        ({"record_valid": np.ones(8, dtype=np.uint8)}, ValueError),
        ({"record_valid": np.zeros(8, dtype=np.bool_)}, ValueError),
    ],
)
def test_program_rejects_shape_dtype_time_and_validity_drift(
    changes: dict[str, object], error: type[Exception]
) -> None:
    kwargs = _program_kwargs()
    kwargs.update(changes)
    with pytest.raises(error):
        BranchProgram(**kwargs)  # type: ignore[arg-type]


def test_program_rejects_command_values_that_disagree_with_its_branch_id() -> None:
    kwargs = _program_kwargs()
    kwargs["requested_command"] = np.ones((8, 4), dtype=np.float64)
    with pytest.raises(ValueError, match="frozen eight-step program"):
        BranchProgram(**kwargs)  # type: ignore[arg-type]


def test_camera_axes_transforms_intrinsics_and_zero_mount_are_exact() -> None:
    contract = default_camera_contract()
    expected = _f64([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
    np.testing.assert_array_equal(R_CAM_FROM_RIG, expected)
    np.testing.assert_array_equal(R_RIG_FROM_CAM, expected.T)
    np.testing.assert_array_equal(contract.R_cam_from_rig, expected)
    np.testing.assert_array_equal(contract.R_rig_from_cam, expected.T)
    np.testing.assert_array_equal(T_CAM_FROM_RIG_M, np.zeros(3))
    np.testing.assert_array_equal(T_RIG_FROM_CAM_M, np.zeros(3))
    np.testing.assert_array_equal(contract.t_cam_from_rig_m, np.zeros(3))
    np.testing.assert_array_equal(contract.t_rig_from_cam_m, np.zeros(3))
    np.testing.assert_array_equal(expected @ _f64([1, 0, 0]), [0, 0, -1])
    np.testing.assert_array_equal(expected @ _f64([0, 1, 0]), [1, 0, 0])
    np.testing.assert_array_equal(expected @ _f64([0, 0, 1]), [0, -1, 0])
    np.testing.assert_array_equal(expected @ _f64([0, 0, 1]), [0, -1, 0])
    assert np.linalg.det(expected) == pytest.approx(1.0)
    assert contract.K[0, 0] == pytest.approx(24.0 / 36.0 * 224.0)
    assert contract.K[0, 2] == 112.0
    assert contract.K[1, 2] == 112.0
    assert contract.manifest()["depth_of_field_enabled"] is False
    assert DEPTH_OF_FIELD_ENABLED is False


def test_camera_contract_hash_has_known_answer_and_is_input_sensitive() -> None:
    zero = program_by_id(BranchId.ZERO)
    assert camera_contract_sha256(zero) == ("2a22de62d4249fd9c065f9e7ded8b2540c92b1bcb8f39fa395ca5854ea3b355e")
    assert camera_contract_sha256(program_by_id(BranchId.YAW_PLUS)) != camera_contract_sha256(zero)

    signed_zero_kwargs = _program_kwargs()
    signed_zero_command = signed_zero_kwargs["requested_command"]
    assert isinstance(signed_zero_command, np.ndarray)
    signed_zero_command[0, 0] = -0.0
    signed_zero = BranchProgram(**signed_zero_kwargs)  # type: ignore[arg-type]
    assert not np.signbit(signed_zero.requested_command).any()
    negative_translation = np.array([-0.0, 0.0, 0.0], dtype=np.float64)
    signed_zero_camera = replace(
        default_camera_contract(),
        t_cam_from_rig_m=negative_translation,
    )
    assert not np.signbit(signed_zero_camera.t_cam_from_rig_m).any()
    assert camera_contract_sha256(signed_zero, signed_zero_camera) == camera_contract_sha256(zero)

    with pytest.raises(ValueError, match="frozen 24-mm"):
        replace(
            default_camera_contract(),
            K=default_camera_contract().K + _f64([[1e-12, 0, 0], [0, 0, 0], [0, 0, 0]]),
        )


def test_canonical_bytes_are_order_endian_and_contiguity_stable() -> None:
    little = np.arange(12, dtype="<f8").reshape(3, 4)
    big = little.astype(">f8")
    backing = np.empty((3, 8), dtype="<f8")
    backing[:, ::2] = little
    noncontiguous = backing[:, ::2]
    assert canonical_bytes({"b": 1, "a": little}) == canonical_bytes({"a": big, "b": 1})
    assert canonical_bytes({"a": little}) == canonical_bytes({"a": noncontiguous})
    assert '"dtype":"<f8"' in canonical_bytes({"a": little}).decode("utf-8")
    assert canonical_bytes({"zero": np.asarray(0.0)}) == canonical_bytes({"zero": np.asarray(-0.0)})


def test_expected_image_sign_table_is_exact() -> None:
    expected = {
        BranchId.YAW_PLUS: ("cx", -1),
        BranchId.YAW_MINUS: ("cx", 1),
        BranchId.FORWARD_PLUS: ("log_h", 1),
        BranchId.FORWARD_MINUS: ("log_h", -1),
        BranchId.LATERAL_PLUS: ("cx", -1),
        BranchId.LATERAL_MINUS: ("cx", 1),
        BranchId.VERTICAL_PLUS: ("cy", -1),
        BranchId.VERTICAL_MINUS: ("cy", 1),
    }
    for branch_id, (field, sign) in expected.items():
        effect = expected_image_effect(branch_id)
        assert (effect.field, effect.sign) == (field, sign)
    with pytest.raises(ValueError, match="zero"):
        expected_image_effect(BranchId.ZERO)


def test_sign_eligibility_uses_strict_depth_and_displacement_boundaries() -> None:
    kwargs = {
        "plus_optical_depth_m": 2.01,
        "minus_optical_depth_m": 3.0,
        "plus_center_xy": _f64([0.1, 0.9]),
        "minus_center_xy": _f64([0.9, 0.1]),
        "plus_displacement_px": -1.01,
        "minus_displacement_px": 1.01,
    }
    assert sign_eligible(**kwargs)
    assert not sign_eligible(**{**kwargs, "plus_optical_depth_m": 2.0})
    assert not sign_eligible(**{**kwargs, "minus_displacement_px": 1.0})
    assert not sign_eligible(**{**kwargs, "plus_center_xy": _f64([0.099, 0.5])})


def test_complete_sibling_group_is_indivisible_and_cross_split_is_rejected() -> None:
    root_id = "root-0001"
    identities = tuple(RootSiblingIdentity(root_id, root_id, branch, DatasetSplit.TRAIN) for branch in BranchId)
    validate_sibling_group(identities)
    with pytest.raises(ValueError, match="exactly"):
        validate_sibling_group(identities[:-1])
    duplicated = identities[:-1] + (identities[0],)
    with pytest.raises(ValueError, match="each of the nine"):
        validate_sibling_group(duplicated)
    crossed = identities[:-1] + (RootSiblingIdentity(root_id, root_id, BranchId.VERTICAL_MINUS, DatasetSplit.TEST),)
    with pytest.raises(ValueError, match="crosses"):
        validate_sibling_group(crossed)
    with pytest.raises(ValueError, match="must equal"):
        RootSiblingIdentity(root_id, "different", BranchId.ZERO, DatasetSplit.TRAIN)


def test_skier_digest_is_camera_independent_and_sensitive_to_every_required_domain() -> None:
    base = _digest_kwargs()
    first = canonical_skier_digest(**base)  # type: ignore[arg-type]
    camera_pose_a = np.eye(4, dtype=np.float64)
    camera_pose_b = np.eye(4, dtype=np.float64)
    camera_pose_b[0, 3] = 100.0
    assert not np.array_equal(camera_pose_a, camera_pose_b)
    assert canonical_skier_digest(**base) == first  # type: ignore[arg-type]

    changed_root = dict(base["root"])  # type: ignore[arg-type]
    changed_root["absolute_tick"] = 8
    changed_skis = dict(base["skis"])  # type: ignore[arg-type]
    changed_left = dict(changed_skis["left"])  # type: ignore[arg-type]
    changed_left["realized_F_world_from_ski"] = np.eye(3, dtype=np.float64) * 2.0
    changed_skis["left"] = changed_left
    changed_contacts = dict(base["contacts"])  # type: ignore[arg-type]
    changed_contacts["right_contact_origin_world_m"] = _f64([0.0, 0.16, 0.1])
    changed_phases = dict(base["phases"])  # type: ignore[arg-type]
    changed_phases["maneuver_phase"] = 0.5
    domain_changes = {
        "root": changed_root,
        "skis": changed_skis,
        "contacts": changed_contacts,
        "phases": changed_phases,
        "randomness": {"seed": 1730},
    }
    for field, replacement in domain_changes.items():
        changed = dict(base)
        changed[field] = replacement
        assert canonical_skier_digest(**changed) != first  # type: ignore[arg-type]
    changed_bones = dict(base)
    changed_bones["local_bone_transforms"] = np.ones((1, 4, 4), dtype=np.float64)
    assert canonical_skier_digest(**changed_bones) != first  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "forbidden_key",
    [
        "camera_pose",
        "branch_id",
        "requested_command",
        "visibility",
        "pixel_bbox",
        "rgb_sha256",
        "target_mask",
    ],
)
def test_skier_digest_recursively_rejects_observation_and_branch_fields(
    forbidden_key: str,
) -> None:
    kwargs = _digest_kwargs()
    kwargs["root"] = {"nested": {forbidden_key: 1}}
    with pytest.raises(ValueError, match="cannot enter skier digest"):
        canonical_skier_digest(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "fail_open_alias",
    ["K", "requested_action", "bbox_xyxy", "depth_m", "occlusion_fraction"],
)
def test_skier_digest_rejects_untyped_camera_or_observation_aliases(
    fail_open_alias: str,
) -> None:
    kwargs = _digest_kwargs()
    root = dict(kwargs["root"])  # type: ignore[arg-type]
    root[fail_open_alias] = 1
    kwargs["root"] = root
    with pytest.raises(ValueError, match="typed digest schema"):
        canonical_skier_digest(**kwargs)  # type: ignore[arg-type]


def test_skier_digest_rejects_wrong_typed_domain_values() -> None:
    wrong_state = _digest_kwargs()
    root = dict(wrong_state["root"])  # type: ignore[arg-type]
    root["position_xy_m"] = [1.0, 2.0]
    wrong_state["root"] = root
    with pytest.raises(TypeError, match="root.position_xy_m"):
        canonical_skier_digest(**wrong_state)  # type: ignore[arg-type]

    wrong_bones = _digest_kwargs()
    wrong_bones["local_bone_transforms"] = np.empty((0, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="little-endian float64"):
        canonical_skier_digest(**wrong_bones)  # type: ignore[arg-type]

    wrong_randomness = _digest_kwargs()
    wrong_randomness["randomness"] = {"seed": True}
    with pytest.raises(TypeError, match="randomness.seed"):
        canonical_skier_digest(**wrong_randomness)  # type: ignore[arg-type]


def test_skier_digest_rejects_incomplete_left_only_pose_schema() -> None:
    kwargs = _digest_kwargs()
    skis = dict(kwargs["skis"])  # type: ignore[arg-type]
    skis.pop("right")
    kwargs["skis"] = skis
    with pytest.raises(ValueError, match="typed digest schema"):
        canonical_skier_digest(**kwargs)  # type: ignore[arg-type]


def test_constants_have_exact_horizon_and_dt() -> None:
    assert HORIZON_STEPS == 8
    assert FIXED_DT_SECONDS == 0.2
