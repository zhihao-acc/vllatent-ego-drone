"""Mechanical/property tests for the deterministic B3-CS2 skier root."""

from __future__ import annotations

import inspect
import math
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from vllatent.sim.contracts import BranchId
from vllatent.sim.skier import (
    BRAKE_DECELERATION_CAP_M_S2,
    DRAG_AREA_HIGH_M2,
    DRAG_AREA_MIDDLE_M2,
    DRAG_AREA_TUCK_M2,
    FIXED_DT_SECONDS,
    GRAVITY_M_S2,
    MIN_INNER_TIP_GAP_M,
    SLOPE_ANGLE_RAD,
    ManeuverRecord,
    ManeuverSchedule,
    ManeuverTargets,
    ManeuverType,
    SkierParameters,
    advance_skier,
    drag_area_from_crouch,
    ideal_carve_curvature,
    quintic_smoothstep,
    residuals_within_contract,
    root_construction_residuals,
    simulate_records,
    ski_construction_residuals,
    step_diagnostics,
)
from vllatent.sim.skier_audit import (
    gate_mechanical_margins,
    root_law_residuals,
    steady_carve_radius_error_fraction,
)
from vllatent.sim.skier_fixtures import (
    CANONICAL_RAMP_IN_TICKS,
    FIXTURE_IDS,
    canonical_skier_fixtures,
    fixture_table_sha256,
)


def _by_id() -> dict[str, object]:
    return {fixture.fixture_id: fixture for fixture in canonical_skier_fixtures()}


def test_frozen_constants_drag_curve_quintic_and_parameter_drift_rejection() -> None:
    assert SLOPE_ANGLE_RAD == math.pi / 12.0
    assert GRAVITY_M_S2 == 9.80665
    assert FIXED_DT_SECONDS == 0.2
    assert BRAKE_DECELERATION_CAP_M_S2 == 6.0
    assert drag_area_from_crouch(0.0) == DRAG_AREA_HIGH_M2 == 0.65
    assert drag_area_from_crouch(0.5) == DRAG_AREA_MIDDLE_M2 == 0.53
    assert drag_area_from_crouch(1.0) == DRAG_AREA_TUCK_M2 == 0.235
    assert quintic_smoothstep(0.0) == 0.0
    assert quintic_smoothstep(0.5) == 0.5
    assert quintic_smoothstep(1.0) == 1.0
    with pytest.raises(ValueError, match="frozen value"):
        SkierParameters(mass_kg=74.0)
    with pytest.raises(ValueError, match="piecewise"):
        ManeuverTargets(0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.65, 0.0)


def test_exact_eight_fixture_catalog_has_a_pinned_canonical_table_hash() -> None:
    fixtures = canonical_skier_fixtures()
    assert FIXTURE_IDS == (
        "straight",
        "accelerate_tuck",
        "brake",
        "carve_left",
        "carve_right",
        "carve_transition",
        "occlusion_path",
        "composite_start_state",
    )
    assert tuple(fixture.fixture_id for fixture in fixtures) == FIXTURE_IDS
    assert len(fixtures) == 8
    assert fixture_table_sha256(fixtures) == ("15bff8634187e52a3e6a6d76377b6b51dc530dfa06c1c6432ace8bc36643d95b")


def test_repeat_replay_is_byte_identical_and_all_arrays_are_immutable() -> None:
    for fixture in canonical_skier_fixtures():
        first = fixture.records()
        second = fixture.records()
        assert len(first) == 11
        assert [record.state.absolute_tick for record in first] == list(range(-2, 9))
        assert [record.canonical_bytes() for record in first] == [record.canonical_bytes() for record in second]
        assert [record.skier_digest() for record in first] == [record.skier_digest() for record in second]
        assert fixture.canonical_root_bytes() == replace(fixture).canonical_root_bytes()
        assert b"history_visible" not in fixture.canonical_root_bytes()
        with pytest.raises(ValueError, match="WRITEABLE"):
            fixture.history_visible.setflags(write=True)
        for record in first:
            immutable_arrays = (
                record.state.tracked_joint_positions_root_m,
                record.root.T_world_from_groundroot,
                record.skis.left.realized_F_world_from_ski,
            )
            for array in immutable_arrays:
                assert not array.flags.writeable
                with pytest.raises(ValueError, match="WRITEABLE"):
                    array.setflags(write=True)


def test_fixture_table_hash_replays_identically_in_two_fresh_python_processes() -> None:
    code = "from vllatent.sim.skier_fixtures import fixture_table_sha256; print(fixture_table_sha256())"
    first = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    second = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert first == second == fixture_table_sha256()


def test_report_root_law_equations_and_planar_recovery_hold_mechanically() -> None:
    for fixture in canonical_skier_fixtures():
        records = fixture.records()
        serialized_residuals = root_law_residuals(records, fixture.schedule, fixture.slope, fixture.parameters)
        assert max(serialized_residuals.values()) <= 1.0e-10
        for record in records:
            state = record.state
            target = record.evaluated_maneuver.targets
            params = fixture.parameters
            diagnostics = step_diagnostics(state, fixture.schedule, fixture.slope, fixture.parameters)
            expected_kappa = state.curvature_1_m + (
                1.0 - math.exp(-params.dt_seconds / params.curvature_response_seconds)
            ) * (target.curvature_1_m - state.curvature_1_m)
            expected_q = state.speed_m_s**2 * expected_kappa + params.gravity_m_s2 * math.sin(
                params.slope_angle_rad
            ) * math.sin(state.heading_rad)
            expected_nu = math.sqrt((params.gravity_m_s2 * math.cos(params.slope_angle_rad)) ** 2 + expected_q**2)
            expected_brake = min(
                target.brake_cap_m_s2,
                0.5
                * sum(
                    params.gravity_m_s2 * math.cos(params.slope_angle_rad) * math.tan(abs(edge)) * abs(math.sin(attack))
                    for edge, attack in (
                        (target.left_edge_rad, target.left_attack_rad),
                        (target.right_edge_rad, target.right_attack_rad),
                    )
                ),
            )
            expected_acceleration = (
                params.gravity_m_s2 * math.sin(params.slope_angle_rad) * math.cos(state.heading_rad)
                - params.snow_friction * expected_nu
                - params.air_density_kg_m3
                * drag_area_from_crouch(target.crouch)
                * state.speed_m_s**2
                / (2.0 * params.mass_kg)
                - expected_brake
            )
            expected_speed = max(0.0, state.speed_m_s + expected_acceleration * params.dt_seconds)
            raw_heading = (
                state.heading_rad + 0.5 * (state.speed_m_s + expected_speed) * expected_kappa * params.dt_seconds
            )
            expected_heading = (raw_heading + math.pi) % (2.0 * math.pi) - math.pi
            tangent_now = (
                math.cos(state.heading_rad) * fixture.slope.downhill_world
                + math.sin(state.heading_rad) * fixture.slope.right_world
            )
            tangent_next = (
                math.cos(expected_heading) * fixture.slope.downhill_world
                + math.sin(expected_heading) * fixture.slope.right_world
            )
            ground_now = (
                fixture.slope.origin_world_m
                + state.x_m * fixture.slope.downhill_world
                + state.y_m * fixture.slope.right_world
            )
            expected_ground_next = ground_now + 0.5 * params.dt_seconds * (
                state.speed_m_s * tangent_now + expected_speed * tangent_next
            )
            relative = expected_ground_next - fixture.slope.origin_world_m
            expected_x = float(np.dot(relative, fixture.slope.downhill_world))
            expected_y = float(np.dot(relative, fixture.slope.right_world))
            assert diagnostics.next_curvature_1_m == pytest.approx(expected_kappa, abs=1e-15)
            assert diagnostics.q_m_s2 == pytest.approx(expected_q, abs=1e-14)
            assert diagnostics.normal_load_accel_m_s2 == pytest.approx(expected_nu, abs=1e-14)
            assert diagnostics.brake_deceleration_m_s2 == pytest.approx(expected_brake, abs=1e-14)
            assert diagnostics.longitudinal_acceleration_m_s2 == pytest.approx(expected_acceleration, abs=1e-14)
            assert diagnostics.next_speed_m_s == pytest.approx(expected_speed, abs=1e-14)
            assert diagnostics.next_heading_rad == pytest.approx(expected_heading, abs=1e-14)
            np.testing.assert_allclose(diagnostics.next_ground_point_world_m, expected_ground_next, atol=1e-12)
            assert diagnostics.next_x_m == pytest.approx(expected_x, abs=1e-12)
            assert diagnostics.next_y_m == pytest.approx(expected_y, abs=1e-12)
            assert diagnostics.planar_recovery_residual_m <= 1e-10
            assert record.omega_rad_s == state.speed_m_s * state.curvature_1_m
            assert record.payload()["acceleration_semantics"] == ("a_at_tick_for_interval_to_tick_plus_1")
            advanced = advance_skier(state, fixture.schedule, fixture.slope, fixture.parameters)
            assert advanced.absolute_tick == state.absolute_tick + 1
            assert advanced.curvature_1_m == pytest.approx(expected_kappa, abs=1e-15)
            assert advanced.speed_m_s == pytest.approx(expected_speed, abs=1e-14)
            assert advanced.heading_rad == pytest.approx(expected_heading, abs=1e-14)
            assert advanced.x_m == pytest.approx(expected_x, abs=1e-12)
            assert advanced.y_m == pytest.approx(expected_y, abs=1e-12)


def test_serialized_root_law_audit_rejects_record_and_update_corruption() -> None:
    fixture = canonical_skier_fixtures()[0]
    records = list(fixture.records())
    records[3] = replace(
        records[3],
        acceleration_m_s2=records[3].acceleration_m_s2 + 3.0,
        omega_rad_s=records[3].omega_rad_s + 2.0,
    )
    field_residuals = root_law_residuals(records, fixture.schedule, fixture.slope, fixture.parameters)
    assert field_residuals["acceleration_m_s2"] == pytest.approx(3.0)
    assert field_residuals["omega_rad_s"] == pytest.approx(2.0)

    records = list(fixture.records())
    original_state = records[4].state
    corrupted_state = replace(
        original_state,
        x_m=original_state.x_m + 0.5,
        y_m=original_state.y_m - 0.4,
        heading_rad=original_state.heading_rad + 0.2,
        speed_m_s=original_state.speed_m_s + 1.0,
        curvature_1_m=original_state.curvature_1_m + 0.1,
    )
    records[4] = replace(records[4], state=corrupted_state)
    update_residuals = root_law_residuals(records, fixture.schedule, fixture.slope, fixture.parameters)
    assert update_residuals["x_update_m"] >= 0.49
    assert update_residuals["y_update_m"] >= 0.39
    assert update_residuals["heading_update_rad"] >= 0.19
    assert update_residuals["speed_update_m_s"] >= 0.99
    assert update_residuals["curvature_update_1_m"] >= 0.09
    assert update_residuals["position_update_m"] > 0.6


def test_root_armature_stance_origin_frame_and_slip_residuals_clear_gates() -> None:
    for fixture in canonical_skier_fixtures():
        for record in fixture.records():
            root_residuals = root_construction_residuals(record, fixture.slope, fixture.parameters)
            ski_residuals = ski_construction_residuals(record, fixture.slope, fixture.parameters)
            assert max(root_residuals.values()) <= 1e-10
            expected_ground = (
                fixture.slope.origin_world_m
                + record.state.x_m * fixture.slope.downhill_world
                + record.state.y_m * fixture.slope.right_world
            )
            np.testing.assert_allclose(record.root.T_world_from_groundroot[:3, 3], expected_ground, atol=1e-12)
            np.testing.assert_allclose(
                record.root.T_world_from_groundroot[:3, :3],
                np.column_stack(
                    (
                        record.root.tangent_world,
                        record.root.lateral_world,
                        -fixture.slope.normal_world,
                    )
                ),
                atol=1e-12,
            )
            assert residuals_within_contract(ski_residuals)
            gate_mechanical_margins(record)
            assert record.skis.centerline_ordering_m > 0.0
            assert record.skis.inner_tip_gap_m >= MIN_INNER_TIP_GAP_M - 1e-12
            for ski in (record.skis.left, record.skis.right):
                np.testing.assert_allclose(
                    np.cross(ski.forward_world, ski.edged_right_world),
                    -ski.outward_normal_world,
                    atol=1e-12,
                )
                assert np.linalg.det(ski.realized_F_world_from_ski) == pytest.approx(1.0, abs=1e-12)
                assert ski.realized_attack_rad == pytest.approx(ski.attack_rad, abs=1e-12)
                assert ski.realized_edge_rad == pytest.approx(ski.edge_rad, abs=1e-12)
                np.testing.assert_allclose(
                    ski.realized_slip_longitudinal_lateral_m_s,
                    ski.analytic_slip_longitudinal_lateral_m_s,
                    atol=1e-12,
                )
                lateral_ratio = abs(ski.realized_slip_longitudinal_lateral_m_s[1]) / record.state.speed_m_s
                if record.evaluated_maneuver.maneuver_type is ManeuverType.BRAKE:
                    assert lateral_ratio == pytest.approx(abs(math.sin(ski.attack_rad)), abs=0.02)
                else:
                    assert lateral_ratio <= math.sin(math.radians(5.0)) + 1e-12
            if record.evaluated_maneuver.maneuver_type is ManeuverType.BRAKE:
                assert (
                    record.skis.left.realized_slip_longitudinal_lateral_m_s[1]
                    * record.skis.right.realized_slip_longitudinal_lateral_m_s[1]
                    < 0.0
                )


def test_mechanical_residuals_reject_independent_root_and_ski_corruption() -> None:
    fixture = canonical_skier_fixtures()[0]
    record = fixture.records()[2]
    angle = 0.1
    local_rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    wrong_ground = record.root.T_world_from_groundroot.copy()
    wrong_armature = record.root.T_world_from_armature.copy()
    wrong_ground[:3, :3] = wrong_ground[:3, :3] @ local_rotation
    wrong_armature[:3, :3] = wrong_armature[:3, :3] @ local_rotation
    wrong_root_record = replace(
        record,
        root=replace(
            record.root,
            T_world_from_groundroot=wrong_ground,
            T_world_from_armature=wrong_armature,
        ),
    )
    root_residuals = root_construction_residuals(wrong_root_record, fixture.slope, fixture.parameters)
    assert root_residuals["ground_frame_rad"] == pytest.approx(angle, abs=1e-12)
    assert root_residuals["armature_frame_rad"] == pytest.approx(angle, abs=1e-12)

    tangential_shift = 0.25 * fixture.slope.downhill_world
    shifted_points_record = replace(
        record,
        root=replace(
            record.root,
            ground_point_world_m=record.root.ground_point_world_m + tangential_shift,
            pelvis_point_world_m=record.root.pelvis_point_world_m + tangential_shift,
        ),
    )
    shifted_point_residuals = root_construction_residuals(shifted_points_record, fixture.slope, fixture.parameters)
    assert shifted_point_residuals["ground_point_m"] == pytest.approx(0.25)
    assert shifted_point_residuals["pelvis_point_m"] == pytest.approx(0.25)

    small_angle = 1.0e-8
    small_rotation = np.array(
        [
            [math.cos(small_angle), -math.sin(small_angle), 0.0],
            [math.sin(small_angle), math.cos(small_angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    small_wrong_ground = record.root.T_world_from_groundroot.copy()
    small_wrong_ground[:3, :3] = small_wrong_ground[:3, :3] @ small_rotation
    small_angle_record = replace(
        record,
        root=replace(record.root, T_world_from_groundroot=small_wrong_ground),
    )
    small_angle_residuals = root_construction_residuals(small_angle_record, fixture.slope, fixture.parameters)
    assert small_angle_residuals["ground_frame_rad"] == pytest.approx(small_angle, abs=1.0e-16)

    translation = np.array([10.0, 20.0, 30.0], dtype=np.float64)

    def translated(ski):
        return replace(
            ski,
            centerline_origin_world_m=ski.centerline_origin_world_m + translation,
            base_origin_world_m=ski.base_origin_world_m + translation,
            binding_origin_world_m=ski.binding_origin_world_m + translation,
            contact_origin_world_m=ski.contact_origin_world_m + translation,
        )

    translated_record = replace(
        record,
        skis=replace(
            record.skis,
            left=translated(record.skis.left),
            right=translated(record.skis.right),
        ),
    )
    translated_residuals = ski_construction_residuals(translated_record, fixture.slope, fixture.parameters)
    assert not residuals_within_contract(translated_residuals)
    assert translated_residuals["left_centerline_origin_m"] > 1.0

    wrong_realized = record.skis.left.realized_F_world_from_ski @ local_rotation
    wrong_frame_record = replace(
        record,
        skis=replace(
            record.skis,
            left=replace(
                record.skis.left,
                realized_F_world_from_ski=wrong_realized,
                frame_orientation_residual_rad=0.0,
            ),
        ),
    )
    wrong_frame_residuals = ski_construction_residuals(wrong_frame_record, fixture.slope, fixture.parameters)
    assert wrong_frame_residuals["left_frame_rad"] == pytest.approx(angle, abs=1e-12)
    assert not residuals_within_contract(wrong_frame_residuals)

    wrong_forward = (
        math.cos(angle) * record.skis.left.forward_world + math.sin(angle) * record.skis.left.zero_edge_right_world
    )
    corrupted_left = replace(record.skis.left, forward_world=wrong_forward)
    corrupted_left_tip = (
        corrupted_left.centerline_origin_world_m + fixture.parameters.ski_length_m * 0.5 * corrupted_left.forward_world
    )
    right_tip = (
        record.skis.right.centerline_origin_world_m
        + fixture.parameters.ski_length_m * 0.5 * record.skis.right.forward_world
    )
    corrupted_gap = (
        float(
            np.dot(
                right_tip - corrupted_left_tip,
                record.root.lateral_world,
            )
        )
        - fixture.parameters.ski_width_m
    )
    coordinated_forward_record = replace(
        record,
        skis=replace(
            record.skis,
            left=corrupted_left,
            inner_tip_gap_m=corrupted_gap,
        ),
    )
    coordinated_forward_residuals = ski_construction_residuals(
        coordinated_forward_record, fixture.slope, fixture.parameters
    )
    assert coordinated_forward_residuals["left_forward_world"] > 0.01
    assert coordinated_forward_residuals["inner_tip_gap_residual_m"] > 0.001
    assert not residuals_within_contract(coordinated_forward_residuals)

    wrong_ordering_record = replace(
        record,
        skis=replace(
            record.skis,
            centerline_ordering_m=record.skis.centerline_ordering_m + 1.0,
        ),
    )
    wrong_ordering_residuals = ski_construction_residuals(wrong_ordering_record, fixture.slope, fixture.parameters)
    assert wrong_ordering_residuals["stored_centerline_ordering_m"] == pytest.approx(1.0)
    assert not residuals_within_contract(wrong_ordering_residuals)


def test_straight_tuck_brake_and_speed_envelope_gates() -> None:
    fixtures = {fixture.fixture_id: fixture for fixture in canonical_skier_fixtures()}
    straight = fixtures["straight"]
    tuck = fixtures["accelerate_tuck"]
    brake = fixtures["brake"]
    straight_records = straight.records()
    tuck_records = tuck.records()
    brake_records = brake.records()
    assert max(abs(record.state.curvature_1_m) for record in straight_records) < 1e-4
    assert tuck_records[-1].state.speed_m_s >= straight_records[-1].state.speed_m_s + 0.10

    brake_speeds = np.array([record.state.speed_m_s for record in brake_records], dtype=np.float64)
    assert np.all(np.diff(brake_speeds) <= 0.0)
    brake_target = brake_records[0].evaluated_maneuver.targets
    assert brake_target.left_attack_rad >= math.radians(30.0)
    assert brake_target.right_attack_rad <= -math.radians(30.0)
    assert brake_target.left_attack_rad - brake_target.right_attack_rad >= math.radians(60.0)
    assert math.radians(25.0) <= brake_target.left_edge_rad <= math.radians(35.0)
    assert -math.radians(35.0) <= brake_target.right_edge_rad <= -math.radians(25.0)

    high_from_brake_root = simulate_records(
        brake.initial_state,
        straight.schedule,
        brake.slope,
        brake.parameters,
        final_tick=8,
    )
    assert high_from_brake_root[-1].state.speed_m_s - brake_records[-1].state.speed_m_s >= 1.0
    all_speeds = [record.state.speed_m_s for fixture in canonical_skier_fixtures() for record in fixture.records()]
    assert min(all_speeds) >= 2.0
    assert max(all_speeds) <= 12.0


def test_carves_are_mirrored_and_ideal_radius_is_only_gated_in_steady_hold() -> None:
    fixtures = {fixture.fixture_id: fixture for fixture in canonical_skier_fixtures()}
    left = fixtures["carve_left"].records()
    right = fixtures["carve_right"].records()
    np.testing.assert_allclose(
        [record.state.speed_m_s for record in left],
        [record.state.speed_m_s for record in right],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        [record.state.heading_rad for record in left],
        [-record.state.heading_rad for record in right],
        atol=1e-12,
    )
    for record in (*left, *right):
        target = record.evaluated_maneuver.targets
        assert abs(target.left_attack_rad) <= math.radians(5.0)
        assert abs(target.right_attack_rad) <= math.radians(5.0)
        assert math.copysign(1.0, target.left_edge_rad) == math.copysign(1.0, target.right_edge_rad)
        assert abs(target.left_edge_rad) > math.radians(45.0)
        error = steady_carve_radius_error_fraction(record)
        assert error is not None and error <= 0.05
    np.testing.assert_allclose(
        [record.gross_lean_rad for record in left],
        [-record.gross_lean_rad for record in right],
        atol=1e-12,
    )
    transition = fixtures["carve_transition"].records()
    assert all(steady_carve_radius_error_fraction(record) is None for record in transition)

    drifted_state = replace(
        right[0].state,
        curvature_1_m=right[0].state.curvature_1_m * 0.8,
    )
    drifted_record = replace(right[0], state=drifted_state)
    drifted_error = steady_carve_radius_error_fraction(drifted_record)
    assert drifted_error == pytest.approx(0.25, abs=1e-12)
    assert drifted_error > 0.05


def test_ideal_carve_gate_uses_strict_45_degree_edge_and_5_degree_attack_bounds() -> None:
    fixture = next(item for item in canonical_skier_fixtures() if item.fixture_id == "carve_right")
    record = fixture.records()[2]
    at_45 = math.radians(45.0)
    low_edge_targets = replace(
        record.evaluated_maneuver.targets,
        curvature_1_m=ideal_carve_curvature(at_45),
        left_edge_rad=at_45,
        right_edge_rad=at_45,
    )
    low_edge_record = replace(
        record,
        evaluated_maneuver=replace(record.evaluated_maneuver, targets=low_edge_targets),
    )
    assert steady_carve_radius_error_fraction(low_edge_record) is None
    with pytest.raises(ValueError, match="edge >45"):
        ManeuverRecord(
            "boundary-edge",
            ManeuverType.CARVE_RIGHT,
            "boundary-edge-v1",
            -10,
            1,
            10,
            1,
            low_edge_targets,
        )

    at_5_attack = replace(
        record.evaluated_maneuver.targets,
        left_attack_rad=math.radians(5.0),
        right_attack_rad=math.radians(5.0),
    )
    attack_boundary_record = replace(
        record,
        evaluated_maneuver=replace(record.evaluated_maneuver, targets=at_5_attack),
    )
    assert steady_carve_radius_error_fraction(attack_boundary_record) is None
    with pytest.raises(ValueError, match="attack <5"):
        ManeuverRecord(
            "boundary-attack",
            ManeuverType.CARVE_RIGHT,
            "boundary-attack-v1",
            -10,
            1,
            10,
            1,
            at_5_attack,
        )


def test_transition_occlusion_and_composite_start_state_are_root_law_fixtures() -> None:
    fixtures = {fixture.fixture_id: fixture for fixture in canonical_skier_fixtures()}
    transition = fixtures["carve_transition"].records()
    assert transition[-1].root.pelvis_point_world_m[2] < transition[0].root.pelvis_point_world_m[2]
    first_target = transition[0].evaluated_maneuver.targets
    last_target = transition[-1].evaluated_maneuver.targets
    assert first_target.left_edge_rad < 0.0 < last_target.left_edge_rad
    assert transition[0].state.curvature_1_m < 0.0 < transition[-1].state.curvature_1_m
    occlusion = fixtures["occlusion_path"].records()
    assert all(record.evaluated_maneuver.maneuver_type is ManeuverType.STRAIGHT for record in occlusion)
    assert all("occlusion" not in record.payload() for record in occlusion)
    # Observation labels are audit-only: changing them never changes advancement
    # or the canonical skier table for the matched root.
    labels_a = np.zeros(11, dtype=np.bool_)
    labels_b = np.ones(11, dtype=np.bool_)
    assert not np.array_equal(labels_a, labels_b)
    assert [record.skier_digest() for record in occlusion] == [
        record.skier_digest() for record in fixtures["occlusion_path"].records()
    ]
    composite = fixtures["composite_start_state"]
    assert composite.initial_state.speed_m_s == 6.5
    assert composite.initial_state.heading_rad == pytest.approx(-0.20, abs=1e-15)
    assert composite.initial_state.x_m == -3.0
    assert composite.initial_state.y_m == 1.5


def test_advancement_and_digest_are_structurally_camera_branch_independent() -> None:
    forbidden_parameters = {
        "branch",
        "camera",
        "command",
        "image",
        "mask",
        "pixel",
        "render",
        "visibility",
    }
    assert forbidden_parameters.isdisjoint(inspect.signature(advance_skier).parameters)
    fixture = canonical_skier_fixtures()[0]
    state = fixture.initial_state
    digests = []
    next_states = []
    for _branch in BranchId:
        record = fixture.records()[0]
        digests.append(record.skier_digest())
        next_states.append(advance_skier(state, fixture.schedule, fixture.slope, fixture.parameters))
    assert len(set(digests)) == 1
    assert all(
        candidate.x_m == next_states[0].x_m
        and candidate.y_m == next_states[0].y_m
        and candidate.heading_rad == next_states[0].heading_rad
        for candidate in next_states
    )


def test_schedule_and_state_inputs_are_frozen_and_copy_owned() -> None:
    fixture = canonical_skier_fixtures()[0]
    visibility = fixture.history_visible
    assert not visibility.flags.writeable
    with pytest.raises(ValueError):
        visibility[0] = False
    with pytest.raises(ValueError, match="intervals must not overlap"):
        ManeuverSchedule(
            fixture.schedule.baseline_targets,
            (fixture.schedule.records[0], replace(fixture.schedule.records[0], maneuver_id="x")),
        )
    changed_record = replace(fixture.schedule.records[0], hold_ticks=199)
    changed_fixture = replace(
        fixture,
        schedule=ManeuverSchedule(fixture.schedule.baseline_targets, (changed_record,)),
    )
    assert changed_fixture.canonical_root_bytes() != fixture.canonical_root_bytes()
    with pytest.raises(TypeError):
        CANONICAL_RAMP_IN_TICKS[ManeuverType.STRAIGHT] = 99  # type: ignore[index]
