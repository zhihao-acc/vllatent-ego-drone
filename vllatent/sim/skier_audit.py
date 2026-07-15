"""Causal-identifiability and mechanical gates for B3-CS2 PURE skier records."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from vllatent.sim.contracts import HORIZON_STEPS, sha256_canonical
from vllatent.sim.skier import (
    MIN_INNER_TIP_GAP_M,
    SCHEDULE_SCHEMA_VERSION,
    SKI_SIDECUT_RADIUS_M,
    SMOKE_SPEED_MAX_M_S,
    SMOKE_SPEED_MIN_M_S,
    ManeuverRecord,
    ManeuverSchedule,
    ManeuverType,
    SkierFrameRecord,
    SkierParameters,
    SlopeFrame,
)

CONTINUATION_AUDIT_VERSION: Final[str] = "b3-cs2-continuation-audit-v1"
TERMINAL_KEY_VERSION: Final[str] = "b3-cs2-terminal-key-half-away-v1"

SPEED_BIN_M_S: Final[float] = 0.05
HEADING_BIN_RAD: Final[float] = math.radians(0.5)
CURVATURE_BIN_1_M: Final[float] = 0.002
ACCELERATION_BIN_M_S2: Final[float] = 0.05
EDGE_BIN_RAD: Final[float] = math.radians(1.0)
ATTACK_BIN_RAD: Final[float] = math.radians(2.0)
CROUCH_BIN: Final[float] = 0.02
MANEUVER_PHASE_BIN: Final[float] = 0.02
ANIMATION_PHASE_BIN: Final[float] = 0.02
JOINT_POSITION_BIN_M: Final[float] = 0.01

SPEED_CUE_M_S: Final[float] = 0.10
CURVATURE_CUE_1_M: Final[float] = 0.002
EDGE_CUE_RAD: Final[float] = math.radians(5.0)
ATTACK_CUE_RAD: Final[float] = math.radians(5.0)
CROUCH_CUE: Final[float] = 0.05


def _meets_cue_threshold(value: float, threshold: float) -> bool:
    # The v1 audit treats a mathematically exact threshold as inclusive after
    # float64 subtraction; 1e-15 is the frozen roundoff allowance, not a relaxed
    # physical cue threshold.
    return value >= threshold or math.isclose(value, threshold, rel_tol=0.0, abs_tol=1.0e-15)


def _half_away_bin(value: float, width: float) -> int:
    """Quantize with explicit half-away-from-zero ties for cross-run stability."""
    if not math.isfinite(value) or not math.isfinite(width) or width <= 0.0:
        raise ValueError("quantization: expected finite value and positive width")
    scaled = value / width
    magnitude = math.floor(abs(scaled) + 0.5)
    return -magnitude if scaled < 0.0 else magnitude


@dataclass(frozen=True)
class TerminalStateKey:
    """Quantized observed-state key with every bin frozen by the report."""

    scalar_bins: tuple[int, ...]
    tracked_joint_bins: tuple[int, ...]

    def payload(self) -> dict[str, object]:
        return {
            "version": TERMINAL_KEY_VERSION,
            "scalar_bins": self.scalar_bins,
            "tracked_joint_bins": self.tracked_joint_bins,
        }


def terminal_state_key(record: SkierFrameRecord) -> TerminalStateKey:
    """Build the terminal key without position, camera, branch, or future labels."""
    if not isinstance(record, SkierFrameRecord):
        raise TypeError(f"record: expected SkierFrameRecord, got {type(record).__name__}")
    target = record.evaluated_maneuver.targets
    scalars = (
        _half_away_bin(record.state.speed_m_s, SPEED_BIN_M_S),
        _half_away_bin(record.state.heading_rad, HEADING_BIN_RAD),
        _half_away_bin(record.state.curvature_1_m, CURVATURE_BIN_1_M),
        _half_away_bin(record.acceleration_m_s2, ACCELERATION_BIN_M_S2),
        _half_away_bin(target.left_edge_rad, EDGE_BIN_RAD),
        _half_away_bin(target.right_edge_rad, EDGE_BIN_RAD),
        _half_away_bin(target.left_attack_rad, ATTACK_BIN_RAD),
        _half_away_bin(target.right_attack_rad, ATTACK_BIN_RAD),
        _half_away_bin(target.crouch, CROUCH_BIN),
        _half_away_bin(record.evaluated_maneuver.maneuver_phase, MANEUVER_PHASE_BIN),
        _half_away_bin(record.animation.animation_phase, ANIMATION_PHASE_BIN),
    )
    joint_bins = tuple(
        _half_away_bin(float(value), JOINT_POSITION_BIN_M)
        for value in record.state.tracked_joint_positions_root_m.reshape(-1)
    )
    return TerminalStateKey(scalars, joint_bins)


def _active_record(schedule: ManeuverSchedule, origin: SkierFrameRecord) -> ManeuverRecord | None:
    for record in schedule.records:
        if record.maneuver_id == origin.evaluated_maneuver.maneuver_id:
            return record
    return None


def _record_boundaries(record: ManeuverRecord) -> tuple[int, ...]:
    boundaries = {record.start_tick, record.end_tick}
    if record.ramp_in_ticks > 0:
        boundaries.add(record.ramp_in_end_tick)
    if record.ramp_out_ticks > 0:
        boundaries.add(record.ramp_out_start_tick)
    return tuple(sorted(boundaries))


@dataclass(frozen=True)
class ContinuationAuditResult:
    """Verified continuation identity, terminal key, and applicable history cues."""

    audit_version: str
    continuation_law_id: str
    continuation_target_sha256: str
    terminal_key: TerminalStateKey
    nonsteady: bool
    history_visible_count: int
    satisfied_cues: tuple[str, ...]


def applicable_history_cues(
    first: SkierFrameRecord,
    last: SkierFrameRecord,
    maneuver_type: ManeuverType,
) -> tuple[str, ...]:
    first_target = first.evaluated_maneuver.targets
    last_target = last.evaluated_maneuver.targets
    cues: list[str] = []
    if maneuver_type in (
        ManeuverType.ACCELERATE,
        ManeuverType.BRAKE,
    ) and _meets_cue_threshold(abs(last.state.speed_m_s - first.state.speed_m_s), SPEED_CUE_M_S):
        cues.append("speed")
    if maneuver_type in (
        ManeuverType.CARVE_LEFT,
        ManeuverType.CARVE_RIGHT,
        ManeuverType.TRANSITION,
    ) and _meets_cue_threshold(
        abs(last.state.curvature_1_m - first.state.curvature_1_m),
        CURVATURE_CUE_1_M,
    ):
        cues.append("curvature")
    edge_delta = max(
        abs(last_target.left_edge_rad - first_target.left_edge_rad),
        abs(last_target.right_edge_rad - first_target.right_edge_rad),
    )
    if maneuver_type in (
        ManeuverType.BRAKE,
        ManeuverType.CARVE_LEFT,
        ManeuverType.CARVE_RIGHT,
        ManeuverType.TRANSITION,
    ) and _meets_cue_threshold(edge_delta, EDGE_CUE_RAD):
        cues.append("edge")
    attack_delta = max(
        abs(last_target.left_attack_rad - first_target.left_attack_rad),
        abs(last_target.right_attack_rad - first_target.right_attack_rad),
    )
    if maneuver_type in (
        ManeuverType.BRAKE,
        ManeuverType.TRANSITION,
    ) and _meets_cue_threshold(attack_delta, ATTACK_CUE_RAD):
        cues.append("attack")
    if maneuver_type in (
        ManeuverType.ACCELERATE,
        ManeuverType.CROUCH,
        ManeuverType.TRANSITION,
    ) and _meets_cue_threshold(abs(last_target.crouch - first_target.crouch), CROUCH_CUE):
        cues.append("crouch")
    return tuple(cues)


def audit_forecast_continuation(
    records: Sequence[SkierFrameRecord],
    schedule: ManeuverSchedule,
    history_visible: np.ndarray,
    *,
    forecast_origin_tick: int = 0,
) -> ContinuationAuditResult:
    """Enforce one identifiable history-cued law over history ``H=3``/future ``T=8``."""
    if not isinstance(records, Sequence):
        raise TypeError(f"records: expected sequence, got {type(records).__name__}")
    if not isinstance(schedule, ManeuverSchedule):
        raise TypeError(f"schedule: expected ManeuverSchedule, got {type(schedule).__name__}")
    if not isinstance(history_visible, np.ndarray):
        raise TypeError("history_visible: expected np.ndarray")
    if history_visible.shape != (3,) or history_visible.dtype != np.bool_:
        raise ValueError("history_visible: expected shape (3,) and dtype bool")
    if isinstance(forecast_origin_tick, (bool, np.bool_)) or not isinstance(forecast_origin_tick, (int, np.integer)):
        raise TypeError("forecast_origin_tick: expected integer")
    origin_tick = int(forecast_origin_tick)
    expected_ticks = tuple(range(origin_tick - 2, origin_tick + HORIZON_STEPS + 1))
    by_tick: dict[int, SkierFrameRecord] = {}
    for record in records:
        if not isinstance(record, SkierFrameRecord):
            raise TypeError("records: expected SkierFrameRecord elements")
        tick = record.state.absolute_tick
        if tick in by_tick:
            raise ValueError(f"records: duplicate absolute tick {tick}")
        by_tick[tick] = record
    if tuple(sorted(by_tick)) != expected_ticks:
        raise ValueError(f"records: expected exactly absolute ticks {expected_ticks}")
    for tick in expected_ticks:
        expected_maneuver = schedule.evaluate(tick)
        if by_tick[tick].evaluated_maneuver != expected_maneuver:
            raise ValueError(
                f"records: evaluated maneuver is not bound to the supplied schedule at absolute tick {tick}"
            )
    origin = by_tick[origin_tick]
    continuation_id = origin.evaluated_maneuver.continuation_law_id
    active = _active_record(schedule, origin)
    if active is not None:
        active_phase_start_tick = (
            active.ramp_out_start_tick if origin.evaluated_maneuver.phase_name == "ramp_out" else active.start_tick
        )
        if active_phase_start_tick > origin_tick - 2:
            raise ValueError("continuation: active maneuver/ramp phase must start no later than history tick -2")
    future_ticks = set(range(origin_tick + 1, origin_tick + HORIZON_STEPS + 1))
    for schedule_record in schedule.records:
        future_boundaries = future_ticks.intersection(_record_boundaries(schedule_record))
        if future_boundaries:
            raise ValueError(
                f"continuation: maneuver/target/ramp phase boundary begins in future at {sorted(future_boundaries)}"
            )
    for tick in sorted(future_ticks):
        if by_tick[tick].evaluated_maneuver.continuation_law_id != continuation_id:
            raise ValueError("continuation: future selects a different continuation-law ID")
    for tick in expected_ticks:
        speed = by_tick[tick].state.speed_m_s
        if speed < SMOKE_SPEED_MIN_M_S or speed > SMOKE_SPEED_MAX_M_S:
            raise ValueError("continuation: root leaves the frozen 2..12 m/s smoke speed envelope")

    phases = {
        by_tick[tick].evaluated_maneuver.phase_name for tick in range(origin_tick - 2, origin_tick + HORIZON_STEPS + 1)
    }
    nonsteady = bool(phases.intersection({"ramp_in", "ramp_out"}))
    visible_count = int(np.count_nonzero(history_visible))
    cues = applicable_history_cues(
        by_tick[origin_tick - 2],
        origin,
        origin.evaluated_maneuver.maneuver_type,
    )
    if nonsteady:
        if visible_count < 2:
            raise ValueError("continuation: non-steady target must be visible in two history frames")
        if not cues:
            raise ValueError(
                "continuation: non-steady target lacks an applicable state cue; animation phase alone is invalid"
            )

    active_payload: dict[str, object] | None = None
    source_payload: dict[str, object]
    if active is None:
        source_payload = origin.evaluated_maneuver.targets.payload()
    else:
        active_index = schedule.records.index(active)
        source = schedule.baseline_targets if active_index == 0 else schedule.evaluate(active.start_tick - 1).targets
        source_payload = source.payload()
        active_payload = {
            "maneuver_type": active.maneuver_type,
            "continuation_law_id": active.continuation_law_id,
            "start_tick": active.start_tick,
            "ramp_in_ticks": active.ramp_in_ticks,
            "hold_ticks": active.hold_ticks,
            "ramp_out_ticks": active.ramp_out_ticks,
            "targets": active.targets.payload(),
        }
    future_target_sequence = [
        {
            "absolute_tick": tick,
            "maneuver_type": schedule.evaluate(tick).maneuver_type,
            "continuation_law_id": schedule.evaluate(tick).continuation_law_id,
            "weight": schedule.evaluate(tick).weight,
            "maneuver_phase": schedule.evaluate(tick).maneuver_phase,
            "phase_name": schedule.evaluate(tick).phase_name,
            "targets": schedule.evaluate(tick).targets.payload(),
        }
        for tick in sorted(future_ticks)
    ]
    target_hash = sha256_canonical(
        {
            "audit_version": CONTINUATION_AUDIT_VERSION,
            "schedule_version": SCHEDULE_SCHEMA_VERSION,
            "forecast_origin_tick": origin_tick,
            "source_targets": source_payload,
            "active_law": active_payload,
            "future_target_sequence": future_target_sequence,
        }
    )
    return ContinuationAuditResult(
        CONTINUATION_AUDIT_VERSION,
        continuation_id,
        target_hash,
        terminal_state_key(origin),
        nonsteady,
        visible_count,
        cues,
    )


def audit_terminal_key_collisions(results: Sequence[ContinuationAuditResult]) -> None:
    """Reject one quantized terminal key selecting different laws or target parameters."""
    if not isinstance(results, Sequence):
        raise TypeError(f"results: expected sequence, got {type(results).__name__}")
    seen: dict[TerminalStateKey, tuple[str, str]] = {}
    for index, result in enumerate(results):
        if not isinstance(result, ContinuationAuditResult):
            raise TypeError(f"results[{index}]: expected ContinuationAuditResult")
        signature = (result.continuation_law_id, result.continuation_target_sha256)
        previous = seen.get(result.terminal_key)
        if previous is not None and previous != signature:
            raise ValueError(
                "terminal-state key collision: equal observed keys select different continuation laws or targets"
            )
        seen[result.terminal_key] = signature


def root_law_residuals(
    records: Sequence[SkierFrameRecord],
    schedule: ManeuverSchedule,
    slope: SlopeFrame,
    parameters: SkierParameters,
) -> dict[str, float]:
    """Independently recompute serialized root fields and consecutive updates."""
    if not isinstance(records, Sequence) or not records:
        raise TypeError("records: expected a non-empty sequence")
    if not isinstance(schedule, ManeuverSchedule):
        raise TypeError("schedule: expected ManeuverSchedule")
    if not isinstance(slope, SlopeFrame) or not isinstance(parameters, SkierParameters):
        raise TypeError("slope/parameters: expected SlopeFrame and SkierParameters")
    residuals = {
        "omega_rad_s": 0.0,
        "acceleration_m_s2": 0.0,
        "world_velocity_m_s": 0.0,
        "world_acceleration_m_s2": 0.0,
        "gross_lean_rad": 0.0,
        "curvature_update_1_m": 0.0,
        "speed_update_m_s": 0.0,
        "heading_update_rad": 0.0,
        "x_update_m": 0.0,
        "y_update_m": 0.0,
        "position_update_m": 0.0,
    }

    def retain_max(name: str, value: float) -> None:
        residuals[name] = max(residuals[name], abs(float(value)))

    for index, record in enumerate(records):
        if not isinstance(record, SkierFrameRecord):
            raise TypeError(f"records[{index}]: expected SkierFrameRecord")
        state = record.state
        evaluated = schedule.evaluate(state.absolute_tick)
        if record.evaluated_maneuver != evaluated:
            raise ValueError(f"records[{index}]: evaluated maneuver is not bound to schedule")
        target = evaluated.targets
        gain = 1.0 - math.exp(-parameters.dt_seconds / parameters.curvature_response_seconds)
        next_curvature = state.curvature_1_m + gain * (target.curvature_1_m - state.curvature_1_m)
        q_m_s2 = state.speed_m_s**2 * next_curvature + parameters.gravity_m_s2 * math.sin(
            parameters.slope_angle_rad
        ) * math.sin(state.heading_rad)
        normal_load = math.sqrt((parameters.gravity_m_s2 * math.cos(parameters.slope_angle_rad)) ** 2 + q_m_s2**2)
        skid_sum = sum(
            parameters.gravity_m_s2 * math.cos(parameters.slope_angle_rad) * math.tan(abs(edge)) * abs(math.sin(attack))
            for edge, attack in (
                (target.left_edge_rad, target.left_attack_rad),
                (target.right_edge_rad, target.right_attack_rad),
            )
        )
        brake = min(target.brake_cap_m_s2, 0.5 * skid_sum)
        acceleration = (
            parameters.gravity_m_s2 * math.sin(parameters.slope_angle_rad) * math.cos(state.heading_rad)
            - parameters.snow_friction * normal_load
            - parameters.air_density_kg_m3 * target.drag_area_m2 * state.speed_m_s**2 / (2.0 * parameters.mass_kg)
            - brake
        )
        next_speed = max(0.0, state.speed_m_s + acceleration * parameters.dt_seconds)
        raw_next_heading = (
            state.heading_rad + 0.5 * (state.speed_m_s + next_speed) * next_curvature * parameters.dt_seconds
        )
        next_heading = (raw_next_heading + math.pi) % (2.0 * math.pi) - math.pi
        tangent_now = (
            math.cos(state.heading_rad) * slope.downhill_world + math.sin(state.heading_rad) * slope.right_world
        )
        tangent_next = math.cos(next_heading) * slope.downhill_world + math.sin(next_heading) * slope.right_world
        expected_velocity = state.speed_m_s * tangent_now
        expected_next_velocity = next_speed * tangent_next
        expected_world_acceleration = (expected_next_velocity - expected_velocity) / parameters.dt_seconds
        expected_lean = math.atan2(
            q_m_s2,
            parameters.gravity_m_s2 * math.cos(parameters.slope_angle_rad),
        )
        retain_max("omega_rad_s", record.omega_rad_s - state.speed_m_s * state.curvature_1_m)
        retain_max("acceleration_m_s2", record.acceleration_m_s2 - acceleration)
        retain_max(
            "world_velocity_m_s",
            float(np.linalg.norm(record.world_velocity_m_s - expected_velocity)),
        )
        retain_max(
            "world_acceleration_m_s2",
            float(np.linalg.norm(record.world_acceleration_m_s2 - expected_world_acceleration)),
        )
        retain_max("gross_lean_rad", record.gross_lean_rad - expected_lean)

        if index + 1 >= len(records):
            continue
        next_record = records[index + 1]
        if not isinstance(next_record, SkierFrameRecord):
            raise TypeError(f"records[{index + 1}]: expected SkierFrameRecord")
        next_state = next_record.state
        if next_state.absolute_tick != state.absolute_tick + 1:
            raise ValueError("records: expected consecutive absolute ticks")
        ground_now = slope.origin_world_m + state.x_m * slope.downhill_world + state.y_m * slope.right_world
        expected_ground_next = ground_now + 0.5 * parameters.dt_seconds * (
            state.speed_m_s * tangent_now + next_speed * tangent_next
        )
        relative = expected_ground_next - slope.origin_world_m
        expected_x = float(np.dot(relative, slope.downhill_world))
        expected_y = float(np.dot(relative, slope.right_world))
        actual_ground_next = (
            slope.origin_world_m + next_state.x_m * slope.downhill_world + next_state.y_m * slope.right_world
        )
        heading_error = (next_state.heading_rad - next_heading + math.pi) % (2.0 * math.pi) - math.pi
        retain_max("curvature_update_1_m", next_state.curvature_1_m - next_curvature)
        retain_max("speed_update_m_s", next_state.speed_m_s - next_speed)
        retain_max("heading_update_rad", heading_error)
        retain_max("x_update_m", next_state.x_m - expected_x)
        retain_max("y_update_m", next_state.y_m - expected_y)
        retain_max(
            "position_update_m",
            float(np.linalg.norm(actual_ground_next - expected_ground_next)),
        )
    return residuals


def steady_carve_radius_error_fraction(record: SkierFrameRecord) -> float | None:
    """Check sidecut radius only for a carve's steady hold; return ``None`` otherwise."""
    maneuver = record.evaluated_maneuver
    if maneuver.maneuver_type not in (ManeuverType.CARVE_LEFT, ManeuverType.CARVE_RIGHT):
        return None
    if maneuver.phase_name != "hold":
        return None
    target = maneuver.targets
    if not (
        abs(target.left_edge_rad) > math.radians(45.0)
        and abs(target.right_edge_rad) > math.radians(45.0)
        and abs(target.left_attack_rad) < math.radians(5.0)
        and abs(target.right_attack_rad) < math.radians(5.0)
    ):
        return None
    edge = target.left_edge_rad
    achieved_curvature = record.state.curvature_1_m
    if edge == 0.0 or achieved_curvature == 0.0:
        raise ValueError("steady carve: expected nonzero edge and achieved curvature")
    ideal_radius = SKI_SIDECUT_RADIUS_M * math.cos(abs(edge))
    realized_radius = 1.0 / abs(achieved_curvature)
    return abs(realized_radius - ideal_radius) / ideal_radius


def gate_mechanical_margins(record: SkierFrameRecord) -> None:
    """Require strict L/R ordering and the frozen minimum inner-tip clearance."""
    if record.skis.centerline_ordering_m <= 0.0:
        raise ValueError("ski ordering: left/right centerlines are not strictly ordered")
    if record.skis.inner_tip_gap_m + 1.0e-12 < MIN_INNER_TIP_GAP_M:
        raise ValueError("ski clearance: inner-tip gap is below the frozen minimum")


__all__ = [
    "ACCELERATION_BIN_M_S2",
    "ANIMATION_PHASE_BIN",
    "ATTACK_BIN_RAD",
    "ATTACK_CUE_RAD",
    "CONTINUATION_AUDIT_VERSION",
    "CROUCH_BIN",
    "CROUCH_CUE",
    "CURVATURE_BIN_1_M",
    "CURVATURE_CUE_1_M",
    "ContinuationAuditResult",
    "EDGE_BIN_RAD",
    "EDGE_CUE_RAD",
    "HEADING_BIN_RAD",
    "JOINT_POSITION_BIN_M",
    "MANEUVER_PHASE_BIN",
    "SPEED_BIN_M_S",
    "SPEED_CUE_M_S",
    "TERMINAL_KEY_VERSION",
    "TerminalStateKey",
    "audit_forecast_continuation",
    "audit_terminal_key_collisions",
    "applicable_history_cues",
    "gate_mechanical_margins",
    "root_law_residuals",
    "steady_carve_radius_error_fraction",
    "terminal_state_key",
]
