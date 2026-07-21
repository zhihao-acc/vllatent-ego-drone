"""Mechanical/property tests for the B3-CS3 absolute pose and binding IK."""

from __future__ import annotations

import hashlib
import inspect
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vllatent.sim.contracts import SKIER_POSE_ROOT_SCHEMA_VERSION
from vllatent.sim.pose import (
    CARVE_CYCLE_SAMPLE_COUNT,
    PINNED_CANONICAL_POSE_TABLE_SHA256,
    PINNED_POSE_TABLE_EXPORT_SHA256,
    animation_amplitude_metrics,
    authored_carve_cycle,
    authored_carve_schedule,
    binding_pose_residuals,
    canonical_pose_export_payload,
    canonical_pose_export_sha256,
    canonical_pose_table_sha256,
    construct_pose_root,
    evaluate_pose,
    ik_segment_residuals,
    mirrored_pose_metrics,
    pose_local_reconstruction_residuals,
    pose_root_residuals,
    record_with_pose,
    sample_authored_cycle,
)
from vllatent.sim.rig import TRACKED_SEMANTICS, load_rig_manifest
from vllatent.sim.skier import (
    SkierParameters,
    construct_root_geometry,
    construct_skis,
    default_slope_frame,
    drag_area_from_crouch,
    quintic_smoothstep,
)
from vllatent.sim.skier_fixtures import canonical_skier_fixtures

REPO_ROOT = Path(__file__).parents[1]
RIG_MANIFEST_PATH = REPO_ROOT / "manifests" / "b3_cs3" / "rig.json"


def _manifest():
    return load_rig_manifest(RIG_MANIFEST_PATH)


def _crouch_record(value: float):
    base = canonical_skier_fixtures()[0].records()[2]
    slope = default_slope_frame()
    parameters = SkierParameters()
    targets = replace(
        base.evaluated_maneuver.targets,
        crouch=value,
        drag_area_m2=drag_area_from_crouch(value),
    )
    evaluated = replace(base.evaluated_maneuver, targets=targets)
    root = construct_root_geometry(base.state, evaluated, slope, parameters)
    skis = construct_skis(base.state, evaluated, root, slope, parameters)
    return replace(base, evaluated_maneuver=evaluated, root=root, skis=skis)


def test_all_eight_roots_have_exact_replay_nonstretch_ik_and_binding_pose() -> None:
    manifest = _manifest()
    for fixture in canonical_skier_fixtures():
        for record in fixture.records():
            first = evaluate_pose(record, manifest)
            second = evaluate_pose(record, manifest)
            assert first.canonical_bytes() == second.canonical_bytes()
            assert first.canonical_sha256() == second.canonical_sha256()
            assert np.max(first.ik_reach_ratio) < 1.0
            assert max(binding_pose_residuals(first).values()) <= 1.0e-10
            assert max(ik_segment_residuals(first, manifest).values()) <= 1.0e-10
            assert first.bone_names == tuple(manifest.bone(item).blender_name for item in TRACKED_SEMANTICS)
            assert not first.T_root_from_bone.flags.writeable
            assert not first.tracked_joint_positions_root_m.flags.writeable
            with pytest.raises(ValueError, match="WRITEABLE"):
                first.T_root_from_bone.setflags(write=True)
            posed_first = record_with_pose(record, first)
            posed_second = record_with_pose(record, second)
            assert posed_first.canonical_bytes() == posed_second.canonical_bytes()
            assert posed_first.skier_digest() == posed_second.skier_digest()
            assert posed_first.root_schema_version == SKIER_POSE_ROOT_SCHEMA_VERSION
            assert np.array_equal(posed_first.root.T_world_from_armature, first.T_world_from_armature)
            assert np.array_equal(posed_first.state.local_bone_transforms, first.local_bone_transforms)
            assert posed_first.state.local_bone_transforms.shape == (17, 4, 4)
            assert record.state.local_bone_transforms.shape == (0, 4, 4)
            assert max(pose_root_residuals(record, posed_first).values()) <= 1.0e-10
            assert np.array_equal(
                first.transform("pelvis")[:3, 3],
                manifest.bone("pelvis").rest_matrix_root_m[:3, 3],
            )


def test_pose_attachment_is_source_bound_and_cs3_schema_rejects_empty_cs2_pose() -> None:
    fixtures = canonical_skier_fixtures()
    source = fixtures[0].records()[2]
    other = fixtures[1].records()[2]
    pose = evaluate_pose(source, _manifest())
    assert source.state.absolute_tick == other.state.absolute_tick
    with pytest.raises(ValueError, match="different source skier digest"):
        record_with_pose(other, pose)
    with pytest.raises(ValueError, match="exactly 17 tracked"):
        replace(source, root_schema_version=SKIER_POSE_ROOT_SCHEMA_VERSION)


def test_pose_root_is_the_only_pelvis_height_and_ground_and_skis_are_unchanged() -> None:
    manifest = _manifest()
    for fixture in canonical_skier_fixtures():
        for source in fixture.records():
            pose = evaluate_pose(source, manifest)
            posed = record_with_pose(source, pose)
            expected_root = construct_pose_root(source)
            assert np.array_equal(posed.root.T_world_from_groundroot, source.root.T_world_from_groundroot)
            assert np.array_equal(posed.root.T_world_from_armature, expected_root.T_world_from_armature)
            assert posed.skis is source.skis
            assert np.array_equal(posed.skis.left.binding_origin_world_m, source.skis.left.binding_origin_world_m)
            assert np.array_equal(posed.skis.right.binding_origin_world_m, source.skis.right.binding_origin_world_m)


def test_pose_api_has_no_observation_or_branch_input_and_rejects_nonabsolute_phase() -> None:
    assert tuple(inspect.signature(evaluate_pose).parameters) == ("record", "manifest")
    record = canonical_skier_fixtures()[0].records()[2]
    corrupted = replace(record, animation=replace(record.animation, animation_phase=0.2))
    with pytest.raises(ValueError, match="absolute tick"):
        evaluate_pose(corrupted, _manifest())


def test_authored_cycles_clear_upper_rms_and_driven_knee_amplitude_gates() -> None:
    manifest = _manifest()
    minimum_knee_range = math.radians(5.0)
    for fixture in canonical_skier_fixtures():
        metrics = animation_amplitude_metrics(sample_authored_cycle(fixture.records()[2], manifest))
        assert metrics.upper_joint_peak_to_peak_m >= 0.05
        assert metrics.noncontact_joint_time_rms_m >= 0.02
        assert metrics.driven_knee_range_rad >= minimum_knee_range


def test_left_right_carve_pose_mirrors_far_inside_report_tolerances() -> None:
    manifest = _manifest()
    left = canonical_skier_fixtures()[3].records()
    right = canonical_skier_fixtures()[4].records()
    for left_record, right_record in zip(left, right, strict=True):
        metrics = mirrored_pose_metrics(
            evaluate_pose(left_record, manifest),
            evaluate_pose(right_record, manifest),
            manifest,
        )
        assert metrics.rotation_rms_rad <= math.radians(2.0)
        assert metrics.rotation_max_rad <= math.radians(5.0)
        assert metrics.joint_position_rms_m <= 0.02


def test_mirror_rotation_gate_uses_parent_rest_local_joint_deltas() -> None:
    manifest = _manifest()
    left_record = canonical_skier_fixtures()[3].records()[2]
    right_record = canonical_skier_fixtures()[4].records()[2]
    left_pose = evaluate_pose(left_record, manifest)
    right_pose = evaluate_pose(right_record, manifest)
    changed = right_pose.local_bone_transforms.copy()
    index = TRACKED_SEMANTICS.index("right_elbow")
    angle = math.radians(12.0)
    rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(angle), -math.sin(angle)],
            [0.0, math.sin(angle), math.cos(angle)],
        ],
        dtype=np.float64,
    )
    changed[index, :3, :3] = rotation @ changed[index, :3, :3]
    redistributed = replace(right_pose, local_bone_transforms=changed)
    baseline = mirrored_pose_metrics(left_pose, right_pose, manifest)
    detected = mirrored_pose_metrics(left_pose, redistributed, manifest)
    assert detected.rotation_max_rad >= baseline.rotation_max_rad + math.radians(10.0)


def test_all_local_deltas_reconstruct_globals_and_mutation_is_detected() -> None:
    manifest = _manifest()
    for fixture in canonical_skier_fixtures():
        for record in fixture.records():
            pose = evaluate_pose(record, manifest)
            position_m, rotation_frobenius = pose_local_reconstruction_residuals(pose, manifest)
            assert position_m <= 1.0e-10
            assert rotation_frobenius <= 1.0e-10
    pose = evaluate_pose(canonical_skier_fixtures()[3].records()[2], manifest)
    corrupted = pose.local_bone_transforms.copy()
    corrupted[TRACKED_SEMANTICS.index("left_elbow"), 0, 3] += 0.05
    bad_pose = replace(pose, local_bone_transforms=corrupted)
    position_m, _rotation_frobenius = pose_local_reconstruction_residuals(bad_pose, manifest)
    assert position_m >= 0.049


def test_authoritative_carve_cycles_drive_schedule_skis_pose_and_digest_proof() -> None:
    manifest = _manifest()
    cycles = {sign: authored_carve_cycle(sign) for sign in (-1, 1)}
    expected_weights = np.array(
        [
            quintic_smoothstep(index / 5.0) for index in range(5)
        ]
        + [1.0]
        + [1.0 - quintic_smoothstep(index / 5.0) for index in range(5)],
        dtype=np.float64,
    )
    for cycle in cycles.values():
        assert len(cycle) == CARVE_CYCLE_SAMPLE_COUNT == 11
        assert [record.state.absolute_tick for record in cycle] == list(range(11))
        assert np.allclose(
            [record.evaluated_maneuver.weight for record in cycle],
            expected_weights,
            rtol=0.0,
            atol=1.0e-15,
        )
        edge = np.abs(
            [record.evaluated_maneuver.targets.left_edge_rad for record in cycle]
        )
        lean = np.abs([record.gross_lean_rad for record in cycle])
        assert edge[0] < math.radians(5.0)
        assert np.max(edge) >= math.radians(45.0)
        assert edge[-1] < math.radians(5.0)
        assert abs(int(np.argmax(edge)) - int(np.argmax(lean))) <= 1
        for record in cycle:
            target = record.evaluated_maneuver.targets
            assert record.state.x_m == 0.0
            assert record.state.y_m == 0.0
            assert record.state.heading_rad == 0.0
            assert record.state.speed_m_s == 8.0
            assert record.state.curvature_1_m == target.curvature_1_m
            assert record.skis.left.edge_rad == target.left_edge_rad
            assert record.skis.right.edge_rad == target.right_edge_rad
            assert record.skis.left.realized_edge_rad == pytest.approx(target.left_edge_rad, abs=1.0e-15)
            assert record.skis.right.realized_edge_rad == pytest.approx(target.right_edge_rad, abs=1.0e-15)
            pose = evaluate_pose(record, manifest)
            posed = record_with_pose(record, pose)
            assert pose.source_skier_digest == record.skier_digest()
            assert posed.skier_digest() != record.skier_digest()
            assert max(binding_pose_residuals(pose).values()) <= 1.0e-10
            assert max(ik_segment_residuals(pose, manifest).values()) <= 1.0e-10
    left_edge = np.array(
        [record.evaluated_maneuver.targets.left_edge_rad for record in cycles[-1]]
    )
    right_edge = np.array(
        [record.evaluated_maneuver.targets.left_edge_rad for record in cycles[1]]
    )
    assert np.array_equal(left_edge, -right_edge)
    assert authored_carve_schedule(-1).records[0].end_tick == 11
    assert authored_carve_schedule(1).records[0].end_tick == 11


def test_crouch_heights_are_monotone_and_full_pose_clears_drop_and_flexion_gates() -> None:
    manifest = _manifest()
    poses = [evaluate_pose(_crouch_record(value), manifest) for value in (0.0, 0.5, 1.0)]
    pelvis_heights = np.array([pose.pelvis_height_m for pose in poses])
    shoulder_heights = np.array([pose.shoulder_height_m for pose in poses])
    assert np.all(np.diff(pelvis_heights) <= 0.001)
    assert np.all(np.diff(shoulder_heights) <= 0.001)
    assert pelvis_heights[0] - pelvis_heights[-1] >= 0.12
    assert shoulder_heights[0] - shoulder_heights[-1] >= 0.10
    assert np.min(poses[-1].knee_flexion_rad - poses[0].knee_flexion_rad) >= math.radians(20.0)


def test_transition_pelvis_minimum_tracks_edge_zero_and_clears_flexion_drop() -> None:
    fixture = canonical_skier_fixtures()[5]
    records = fixture.records()
    poses = [evaluate_pose(record, _manifest()) for record in records]
    edge = np.array([record.evaluated_maneuver.targets.left_edge_rad for record in records])
    height = np.array([pose.pelvis_height_m for pose in poses])
    zero_index = int(np.argmin(np.abs(edge)))
    minimum_index = int(np.argmin(height))
    assert abs(minimum_index - zero_index) <= 1
    adjacent_hold_mean = 0.5 * (height[0] + height[-1])
    assert adjacent_hold_mean - height[minimum_index] >= 0.05
    assert poses[zero_index].transition_flexion_m >= 0.05


def test_pose_table_hash_is_pinned_and_exact_in_two_fresh_python_processes() -> None:
    manifest = _manifest()
    expected = canonical_pose_table_sha256(manifest)
    assert expected == PINNED_CANONICAL_POSE_TABLE_SHA256
    assert canonical_pose_export_sha256(manifest) == PINNED_POSE_TABLE_EXPORT_SHA256
    export = canonical_pose_export_payload(manifest)
    assert export["sample_count"] == 88
    assert all("local_bone_transforms" in row for row in export["rows"])
    assert all("source_record_sha256" in row for row in export["rows"])
    assert all("pose_record_canonical_hex" in row for row in export["rows"])
    assert all("posed_record_sha256" in row for row in export["rows"])
    assert all("posed_record_canonical_hex" in row for row in export["rows"])
    assert export["carve_cycle_count"] == 2
    assert export["carve_cycle_sample_count"] == 22
    assert len(export["carve_cycles"]) == 2
    assert len(export["carve_cycle_rows"]) == 22
    assert all("skis" in row for row in export["carve_cycle_rows"])
    assert all("source_record_sha256" in row for row in export["carve_cycle_rows"])
    assert all("posed_record_sha256" in row for row in export["carve_cycle_rows"])
    assert all("posed_skier_digest" in row for row in export["carve_cycle_rows"])
    code = (
        "from vllatent.sim.pose import canonical_pose_table_sha256;"
        "from vllatent.sim.rig import load_rig_manifest;"
        "print(canonical_pose_table_sha256(load_rig_manifest('manifests/b3_cs3/rig.json')))"
    )
    first = subprocess.check_output([sys.executable, "-c", code], text=True, cwd=REPO_ROOT).strip()
    second = subprocess.check_output([sys.executable, "-c", code], text=True, cwd=REPO_ROOT).strip()
    assert first == second == expected


def test_exported_fixture_rows_bind_complete_exact_pose_and_posed_records() -> None:
    manifest = _manifest()
    rows = canonical_pose_export_payload(manifest)["rows"]
    fixtures = canonical_skier_fixtures()
    expected = [
        (fixture.fixture_id, record_index, record)
        for fixture in fixtures
        for record_index, record in enumerate(fixture.records())
    ]
    assert len(rows) == len(expected) == 88
    for row, (fixture_id, record_index, record) in zip(rows, expected, strict=True):
        pose = evaluate_pose(record, manifest)
        posed = record_with_pose(record, pose)
        assert (row["fixture_id"], row["record_index"], row["absolute_tick"]) == (
            fixture_id,
            record_index,
            record.state.absolute_tick,
        )
        assert row["source_record_sha256"] == hashlib.sha256(
            record.canonical_bytes()
        ).hexdigest()
        assert bytes.fromhex(row["pose_record_canonical_hex"]) == pose.canonical_bytes()
        assert row["pose_sha256"] == hashlib.sha256(pose.canonical_bytes()).hexdigest()
        assert bytes.fromhex(row["posed_record_canonical_hex"]) == posed.canonical_bytes()
        assert row["posed_record_sha256"] == hashlib.sha256(
            posed.canonical_bytes()
        ).hexdigest()


def test_exported_carve_rows_are_exact_recomputable_records_poses_and_equipment() -> None:
    manifest = _manifest()
    export = canonical_pose_export_payload(manifest)
    for sign in (-1, 1):
        rows = [row for row in export["carve_cycle_rows"] if row["sign"] == sign]
        records = authored_carve_cycle(sign)
        assert len(rows) == len(records) == 11
        for row, record in zip(rows, records, strict=True):
            pose = evaluate_pose(record, manifest)
            posed = record_with_pose(record, pose)
            assert row["source_record_sha256"] == hashlib.sha256(record.canonical_bytes()).hexdigest()
            assert row["source_skier_digest"] == record.skier_digest()
            assert row["pose_sha256"] == pose.canonical_sha256()
            assert row["posed_record_sha256"] == hashlib.sha256(posed.canonical_bytes()).hexdigest()
            assert row["posed_skier_digest"] == posed.skier_digest()
            assert row["target_left_edge_rad"] == record.evaluated_maneuver.targets.left_edge_rad
            assert row["target_right_edge_rad"] == record.evaluated_maneuver.targets.right_edge_rad
            assert row["gross_lean_rad"] == record.gross_lean_rad
            for side in ("left", "right"):
                ski = getattr(record.skis, side)
                assert row["skis"][side]["edge_rad"] == ski.edge_rad
                assert row["skis"][side]["realized_edge_rad"] == ski.realized_edge_rad
                assert row["skis"][side]["binding_origin_world_m"] == ski.binding_origin_world_m.tolist()
                assert row["skis"][side]["contact_origin_world_m"] == ski.contact_origin_world_m.tolist()
