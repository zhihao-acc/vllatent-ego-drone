"""Causal continuation/history-cue audits for B3-CS2 deterministic roots."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from vllatent.sim.skier import (
    ManeuverRecord,
    ManeuverSchedule,
    ManeuverTargets,
    ManeuverType,
    drag_area_from_crouch,
    neutral_targets,
    simulate_records,
)
from vllatent.sim.skier_audit import (
    ATTACK_CUE_RAD,
    CROUCH_CUE,
    CURVATURE_CUE_1_M,
    EDGE_CUE_RAD,
    HEADING_BIN_RAD,
    SPEED_BIN_M_S,
    SPEED_CUE_M_S,
    ContinuationAuditResult,
    applicable_history_cues,
    audit_forecast_continuation,
    audit_terminal_key_collisions,
    terminal_state_key,
)
from vllatent.sim.skier_fixtures import canonical_skier_fixtures


def _crouch_targets(crouch: float) -> ManeuverTargets:
    return ManeuverTargets(
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        crouch,
        drag_area_from_crouch(crouch),
        0.0,
    )


def _schedule(*, start_tick: int, ramp_in_ticks: int, crouch: float) -> ManeuverSchedule:
    return ManeuverSchedule(
        neutral_targets(),
        (
            ManeuverRecord(
                maneuver_id="audit-crouch-v1",
                maneuver_type=ManeuverType.CROUCH,
                continuation_law_id="audit-crouch-continuation-v1",
                start_tick=start_tick,
                ramp_in_ticks=ramp_in_ticks,
                hold_ticks=100,
                ramp_out_ticks=0,
                targets=_crouch_targets(crouch),
            ),
        ),
    )


def _records(schedule: ManeuverSchedule):
    fixture = canonical_skier_fixtures()[0]
    return simulate_records(
        fixture.initial_state,
        schedule,
        fixture.slope,
        fixture.parameters,
        final_tick=8,
    )


def test_all_eight_forecasts_have_one_causal_law_and_no_terminal_collision() -> None:
    results = []
    expected_nonsteady = {"accelerate_tuck", "carve_transition"}
    for fixture in canonical_skier_fixtures():
        result = audit_forecast_continuation(fixture.records(), fixture.schedule, fixture.history_visible)
        results.append(result)
        assert result.nonsteady is (fixture.fixture_id in expected_nonsteady)
        if result.nonsteady:
            assert result.history_visible_count >= 2
            assert result.satisfied_cues
    audit_terminal_key_collisions(results)


def test_terminal_key_has_every_scalar_and_tracked_joint_bin() -> None:
    fixture = canonical_skier_fixtures()[0]
    origin = fixture.records()[2]
    key = terminal_state_key(origin)
    assert len(key.scalar_bins) == 11
    assert len(key.tracked_joint_bins) == 9
    assert key == terminal_state_key(fixture.records()[2])
    assert "camera" not in key.payload()
    assert "branch" not in key.payload()


def test_terminal_key_uses_half_away_from_zero_at_exact_bin_ties() -> None:
    origin = canonical_skier_fixtures()[0].records()[2]
    positive = replace(
        origin,
        state=replace(
            origin.state,
            speed_m_s=SPEED_BIN_M_S * 0.5,
            heading_rad=HEADING_BIN_RAD * 0.5,
        ),
    )
    negative_heading = replace(
        positive,
        state=replace(positive.state, heading_rad=-HEADING_BIN_RAD * 0.5),
    )
    assert terminal_state_key(positive).scalar_bins[:2] == (1, 1)
    assert terminal_state_key(negative_heading).scalar_bins[:2] == (1, -1)


def test_audit_rejects_a_new_future_maneuver_or_ramp_boundary() -> None:
    visibility = np.ones(3, dtype=np.bool_)
    future_start = _schedule(start_tick=1, ramp_in_ticks=12, crouch=0.7)
    with pytest.raises(ValueError, match="boundary begins in future"):
        audit_forecast_continuation(_records(future_start), future_start, visibility)

    future_ramp_end = _schedule(start_tick=-3, ramp_in_ticks=4, crouch=0.7)
    with pytest.raises(ValueError, match="boundary begins in future"):
        audit_forecast_continuation(_records(future_ramp_end), future_ramp_end, visibility)


def test_audit_rejects_active_ramp_that_started_after_history_tick_minus_two() -> None:
    schedule = _schedule(start_tick=-1, ramp_in_ticks=12, crouch=0.7)
    with pytest.raises(ValueError, match="no later than history tick -2"):
        audit_forecast_continuation(_records(schedule), schedule, np.ones(3, dtype=np.bool_))


def test_audit_rejects_ramp_out_phase_starting_after_history_tick_minus_two() -> None:
    schedule = ManeuverSchedule(
        neutral_targets(),
        (
            ManeuverRecord(
                maneuver_id="late-ramp-out-v1",
                maneuver_type=ManeuverType.CROUCH,
                continuation_law_id="late-ramp-out-continuation-v1",
                start_tick=-20,
                ramp_in_ticks=10,
                hold_ticks=9,
                ramp_out_ticks=20,
                targets=_crouch_targets(0.7),
            ),
        ),
    )
    assert schedule.records[0].ramp_out_start_tick == -1
    assert schedule.evaluate(0).phase_name == "ramp_out"
    with pytest.raises(ValueError, match="ramp phase must start no later"):
        audit_forecast_continuation(_records(schedule), schedule, np.ones(3, dtype=np.bool_))


def test_nonsteady_forecast_requires_two_visible_history_frames() -> None:
    fixture = next(item for item in canonical_skier_fixtures() if item.fixture_id == "accelerate_tuck")
    with pytest.raises(ValueError, match="visible in two history frames"):
        audit_forecast_continuation(
            fixture.records(),
            fixture.schedule,
            np.array([True, False, False], dtype=np.bool_),
        )


def test_animation_phase_alone_is_not_an_applicable_history_cue() -> None:
    schedule = _schedule(start_tick=-3, ramp_in_ticks=12, crouch=0.0)
    with pytest.raises(ValueError, match="animation phase alone"):
        audit_forecast_continuation(_records(schedule), schedule, np.ones(3, dtype=np.bool_))


def test_subthreshold_nonsteady_state_change_is_rejected() -> None:
    schedule = _schedule(start_tick=-3, ramp_in_ticks=12, crouch=0.01)
    with pytest.raises(ValueError, match="lacks an applicable state cue"):
        audit_forecast_continuation(_records(schedule), schedule, np.ones(3, dtype=np.bool_))


def test_equal_terminal_key_cannot_choose_a_different_law_or_target() -> None:
    fixture = canonical_skier_fixtures()[0]
    result = audit_forecast_continuation(fixture.records(), fixture.schedule, fixture.history_visible)
    audit_terminal_key_collisions((result, result))
    different_law = replace(result, continuation_law_id="different-law-v1")
    with pytest.raises(ValueError, match="equal observed keys"):
        audit_terminal_key_collisions((result, different_law))
    different_target = replace(result, continuation_target_sha256="0" * 64)
    with pytest.raises(ValueError, match="equal observed keys"):
        audit_terminal_key_collisions((result, different_target))


def test_collision_audit_rejects_untyped_entries() -> None:
    with pytest.raises(TypeError, match=r"results\[0\]"):
        audit_terminal_key_collisions(("not-a-result",))  # type: ignore[arg-type]
    assert ContinuationAuditResult.__dataclass_fields__["terminal_key"].type is not None


def test_audit_fails_closed_when_a_root_leaves_the_smoke_speed_envelope() -> None:
    fixture = canonical_skier_fixtures()[0]
    too_fast = replace(fixture.initial_state, speed_m_s=12.0)
    records = simulate_records(
        too_fast,
        fixture.schedule,
        fixture.slope,
        fixture.parameters,
        final_tick=8,
    )
    with pytest.raises(ValueError, match=r"2\.\.12 m/s"):
        audit_forecast_continuation(records, fixture.schedule, fixture.history_visible)


def test_records_are_cryptographically_bound_to_the_supplied_schedule() -> None:
    straight = canonical_skier_fixtures()[0]
    crouch_schedule = _schedule(start_tick=-3, ramp_in_ticks=12, crouch=0.7)
    with pytest.raises(ValueError, match="not bound to the supplied schedule"):
        audit_forecast_continuation(straight.records(), crouch_schedule, straight.history_visible)


def test_continuation_hash_excludes_local_maneuver_id_but_includes_ramp_source() -> None:
    fixture = canonical_skier_fixtures()[0]
    original = fixture.schedule.records[0]
    renamed_schedule = ManeuverSchedule(
        fixture.schedule.baseline_targets,
        (replace(original, maneuver_id="renamed-root-local-id"),),
    )
    renamed_records = simulate_records(
        fixture.initial_state,
        renamed_schedule,
        fixture.slope,
        fixture.parameters,
        final_tick=8,
    )
    original_result = audit_forecast_continuation(fixture.records(), fixture.schedule, fixture.history_visible)
    renamed_result = audit_forecast_continuation(renamed_records, renamed_schedule, fixture.history_visible)
    assert original_result.terminal_key == renamed_result.terminal_key
    assert original_result.continuation_target_sha256 == renamed_result.continuation_target_sha256
    audit_terminal_key_collisions((original_result, renamed_result))

    ramp = _schedule(start_tick=-3, ramp_in_ticks=12, crouch=0.7)
    changed_source = ManeuverSchedule(_crouch_targets(0.01), ramp.records)
    base_records = _records(ramp)
    changed_records = _records(changed_source)
    base_result = audit_forecast_continuation(base_records, ramp, np.ones(3, dtype=np.bool_))
    changed_result = audit_forecast_continuation(changed_records, changed_source, np.ones(3, dtype=np.bool_))
    assert base_result.terminal_key == changed_result.terminal_key
    assert base_result.continuation_target_sha256 != changed_result.continuation_target_sha256
    assert base_records[-1].evaluated_maneuver.targets.crouch != changed_records[-1].evaluated_maneuver.targets.crouch
    with pytest.raises(ValueError, match="equal observed keys"):
        audit_terminal_key_collisions((base_result, changed_result))


def test_all_five_history_cue_thresholds_are_inclusive_at_exact_equality() -> None:
    fixture = canonical_skier_fixtures()[0]
    first = fixture.records()[0]
    last = fixture.records()[2]
    speed_last = replace(last, state=replace(last.state, speed_m_s=first.state.speed_m_s + SPEED_CUE_M_S))
    assert "speed" in applicable_history_cues(first, speed_last, ManeuverType.ACCELERATE)
    curvature_last = replace(
        last,
        state=replace(
            last.state,
            curvature_1_m=first.state.curvature_1_m + CURVATURE_CUE_1_M,
        ),
    )
    assert "curvature" in applicable_history_cues(first, curvature_last, ManeuverType.TRANSITION)
    edge_targets = replace(last.evaluated_maneuver.targets, left_edge_rad=EDGE_CUE_RAD)
    edge_last = replace(
        last,
        evaluated_maneuver=replace(last.evaluated_maneuver, targets=edge_targets),
    )
    assert "edge" in applicable_history_cues(first, edge_last, ManeuverType.TRANSITION)
    attack_targets = replace(last.evaluated_maneuver.targets, left_attack_rad=ATTACK_CUE_RAD)
    attack_last = replace(
        last,
        evaluated_maneuver=replace(last.evaluated_maneuver, targets=attack_targets),
    )
    assert "attack" in applicable_history_cues(first, attack_last, ManeuverType.BRAKE)
    crouch_targets = replace(
        last.evaluated_maneuver.targets,
        crouch=CROUCH_CUE,
        drag_area_m2=drag_area_from_crouch(CROUCH_CUE),
    )
    crouch_last = replace(
        last,
        evaluated_maneuver=replace(last.evaluated_maneuver, targets=crouch_targets),
    )
    assert "crouch" in applicable_history_cues(first, crouch_last, ManeuverType.CROUCH)


def test_schedule_dispatch_is_half_open_at_the_serialized_end_tick() -> None:
    schedule = _schedule(start_tick=-3, ramp_in_ticks=12, crouch=0.7)
    record = schedule.records[0]
    assert schedule.evaluate(record.end_tick - 1).maneuver_id == record.maneuver_id
    assert schedule.evaluate(record.end_tick).maneuver_id == "terminal-baseline"
