"""Exactly eight deterministic renderer-neutral root fixtures for B3-CS2."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import numpy as np

from vllatent.sim.contracts import HORIZON_STEPS, canonical_bytes, sha256_canonical
from vllatent.sim.skier import (
    BRAKE_DECELERATION_CAP_M_S2,
    ManeuverRecord,
    ManeuverSchedule,
    ManeuverTargets,
    ManeuverType,
    SkierFrameRecord,
    SkierParameters,
    SkierState,
    SlopeFrame,
    default_slope_frame,
    drag_area_from_crouch,
    ideal_carve_curvature,
    neutral_targets,
    simulate_records,
)

FIXTURE_SCHEMA_VERSION: Final[str] = "b3-cs2-eight-root-fixtures-v1"
FIXTURE_IDS: Final[tuple[str, ...]] = (
    "straight",
    "accelerate_tuck",
    "brake",
    "carve_left",
    "carve_right",
    "carve_transition",
    "occlusion_path",
    "composite_start_state",
)
CANONICAL_RAMP_IN_TICKS: Final[Mapping[ManeuverType, int]] = MappingProxyType(
    {
        ManeuverType.STRAIGHT: 0,
        ManeuverType.ACCELERATE: 12,
        ManeuverType.BRAKE: 10,
        ManeuverType.CARVE_LEFT: 10,
        ManeuverType.CARVE_RIGHT: 10,
        ManeuverType.CROUCH: 12,
        ManeuverType.TRANSITION: 12,
    }
)


def _targets(
    *,
    curvature: float = 0.0,
    left_edge_deg: float = 0.0,
    right_edge_deg: float = 0.0,
    left_attack_deg: float = 0.0,
    right_attack_deg: float = 0.0,
    crouch: float = 0.0,
    brake_cap: float = 0.0,
) -> ManeuverTargets:
    return ManeuverTargets(
        curvature_1_m=curvature,
        left_edge_rad=math.radians(left_edge_deg),
        right_edge_rad=math.radians(right_edge_deg),
        left_attack_rad=math.radians(left_attack_deg),
        right_attack_rad=math.radians(right_attack_deg),
        crouch=crouch,
        drag_area_m2=drag_area_from_crouch(crouch),
        brake_cap_m_s2=brake_cap,
    )


def _record(
    fixture_id: str,
    maneuver_type: ManeuverType,
    targets: ManeuverTargets,
    *,
    start_tick: int,
    ramp_in_ticks: int,
    hold_ticks: int = 200,
) -> ManeuverRecord:
    expected_ramp = CANONICAL_RAMP_IN_TICKS[maneuver_type]
    if ramp_in_ticks != expected_ramp:
        raise ValueError(f"{maneuver_type.value}: ramp_in_ticks must equal frozen {expected_ramp}")
    return ManeuverRecord(
        maneuver_id=f"{fixture_id}-maneuver-v1",
        maneuver_type=maneuver_type,
        continuation_law_id=f"{fixture_id}-continuation-v1",
        start_tick=start_tick,
        ramp_in_ticks=ramp_in_ticks,
        hold_ticks=hold_ticks,
        ramp_out_ticks=0,
        targets=targets,
    )


@dataclass(frozen=True, eq=False)
class CanonicalSkierFixture:
    """One serialized root/schedule plus observation-only history visibility audit data."""

    fixture_id: str
    initial_state: SkierState
    schedule: ManeuverSchedule
    slope: SlopeFrame
    parameters: SkierParameters
    history_visible: np.ndarray

    def __post_init__(self) -> None:
        if self.fixture_id not in FIXTURE_IDS:
            raise ValueError(f"fixture_id: unknown canonical fixture {self.fixture_id!r}")
        if self.initial_state.absolute_tick != -2:
            raise ValueError("initial_state: canonical history must start at absolute tick -2")
        if not isinstance(self.history_visible, np.ndarray):
            raise TypeError("history_visible: expected np.ndarray")
        if self.history_visible.shape != (3,) or self.history_visible.dtype != np.bool_:
            raise ValueError("history_visible: expected shape (3,) and dtype bool")
        visibility = np.frombuffer(
            np.ascontiguousarray(self.history_visible, dtype=np.bool_).tobytes(order="C"),
            dtype=np.bool_,
        ).reshape((3,))
        object.__setattr__(self, "history_visible", visibility)

    def records(self) -> tuple[SkierFrameRecord, ...]:
        """Return history -2..0 and future 1..8 inclusive."""
        return simulate_records(
            self.initial_state,
            self.schedule,
            self.slope,
            self.parameters,
            final_tick=HORIZON_STEPS,
        )

    def root_payload(self) -> dict[str, object]:
        """Serialize every PURE episode-static/root input; visibility stays separate."""
        return {
            "fixture_schema_version": FIXTURE_SCHEMA_VERSION,
            "fixture_id": self.fixture_id,
            "parameters": self.parameters.payload(),
            "slope": self.slope.payload(),
            "schedule": self.schedule.payload(),
            "initial_state": self.initial_state.payload(),
        }

    def canonical_root_bytes(self) -> bytes:
        return canonical_bytes(self.root_payload())


def _state(
    *,
    fixture_index: int,
    curvature: float = 0.0,
    heading_rad: float = 0.0,
    x_m: float = 0.0,
    y_m: float = 0.0,
    speed_m_s: float = 8.0,
) -> SkierState:
    joints = np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.55], [-0.1, 0.0, 0.55]],
        dtype=np.float64,
    )
    return SkierState(
        absolute_tick=-2,
        x_m=x_m,
        y_m=y_m,
        heading_rad=heading_rad,
        speed_m_s=speed_m_s,
        curvature_1_m=curvature,
        tracked_joint_positions_root_m=joints,
        local_bone_transforms=np.empty((0, 4, 4), dtype=np.float64),
        randomness_seed=1729 + fixture_index,
    )


def canonical_skier_fixtures() -> tuple[CanonicalSkierFixture, ...]:
    """Build exactly the eight preregistered deterministic CS2 roots."""
    parameters = SkierParameters()
    slope = default_slope_frame()
    visible = np.ones(3, dtype=np.bool_)
    edge = math.radians(50.0)
    carve_right = _targets(
        curvature=ideal_carve_curvature(edge),
        left_edge_deg=50.0,
        right_edge_deg=50.0,
        crouch=0.2,
    )
    carve_left = _targets(
        curvature=ideal_carve_curvature(-edge),
        left_edge_deg=-50.0,
        right_edge_deg=-50.0,
        crouch=0.2,
    )
    definitions: tuple[tuple[str, ManeuverSchedule, float, float, float, float, float], ...] = (
        (
            "straight",
            ManeuverSchedule(
                neutral_targets(),
                (_record("straight", ManeuverType.STRAIGHT, neutral_targets(), start_tick=-20, ramp_in_ticks=0),),
            ),
            0.0,
            0.0,
            0.0,
            0.0,
            8.0,
        ),
        (
            "accelerate_tuck",
            ManeuverSchedule(
                neutral_targets(),
                (
                    _record(
                        "accelerate_tuck",
                        ManeuverType.ACCELERATE,
                        _targets(crouch=1.0),
                        start_tick=-3,
                        ramp_in_ticks=12,
                    ),
                ),
            ),
            0.0,
            0.0,
            0.0,
            0.0,
            8.0,
        ),
        (
            "brake",
            ManeuverSchedule(
                neutral_targets(),
                (
                    _record(
                        "brake",
                        ManeuverType.BRAKE,
                        _targets(
                            left_edge_deg=30.0,
                            right_edge_deg=-30.0,
                            left_attack_deg=30.0,
                            right_attack_deg=-30.0,
                            brake_cap=BRAKE_DECELERATION_CAP_M_S2,
                        ),
                        start_tick=-12,
                        ramp_in_ticks=10,
                    ),
                ),
            ),
            0.0,
            0.0,
            0.0,
            0.0,
            8.0,
        ),
        (
            "carve_left",
            ManeuverSchedule(
                neutral_targets(),
                (_record("carve_left", ManeuverType.CARVE_LEFT, carve_left, start_tick=-20, ramp_in_ticks=10),),
            ),
            carve_left.curvature_1_m,
            0.0,
            0.0,
            0.0,
            8.0,
        ),
        (
            "carve_right",
            ManeuverSchedule(
                neutral_targets(),
                (_record("carve_right", ManeuverType.CARVE_RIGHT, carve_right, start_tick=-20, ramp_in_ticks=10),),
            ),
            carve_right.curvature_1_m,
            0.0,
            0.0,
            0.0,
            8.0,
        ),
        (
            "carve_transition",
            ManeuverSchedule(
                carve_left,
                (
                    _record(
                        "carve_transition",
                        ManeuverType.TRANSITION,
                        _targets(
                            curvature=carve_right.curvature_1_m,
                            left_edge_deg=50.0,
                            right_edge_deg=50.0,
                            crouch=0.7,
                        ),
                        start_tick=-3,
                        ramp_in_ticks=12,
                    ),
                ),
            ),
            carve_left.curvature_1_m,
            0.0,
            0.0,
            0.0,
            8.0,
        ),
        (
            "occlusion_path",
            ManeuverSchedule(
                neutral_targets(),
                (
                    _record(
                        "occlusion_path",
                        ManeuverType.STRAIGHT,
                        neutral_targets(),
                        start_tick=-20,
                        ramp_in_ticks=0,
                    ),
                ),
            ),
            0.0,
            0.12,
            2.0,
            -1.0,
            8.0,
        ),
        (
            "composite_start_state",
            ManeuverSchedule(
                carve_right,
                (
                    _record(
                        "composite_start_state",
                        ManeuverType.CARVE_RIGHT,
                        carve_right,
                        start_tick=-20,
                        ramp_in_ticks=10,
                    ),
                ),
            ),
            carve_right.curvature_1_m,
            -0.20,
            -3.0,
            1.5,
            6.5,
        ),
    )
    fixtures = tuple(
        CanonicalSkierFixture(
            fixture_id=fixture_id,
            initial_state=_state(
                fixture_index=index,
                curvature=curvature,
                heading_rad=heading,
                x_m=x_m,
                y_m=y_m,
                speed_m_s=speed,
            ),
            schedule=schedule,
            slope=slope,
            parameters=parameters,
            history_visible=visible,
        )
        for index, (
            fixture_id,
            schedule,
            curvature,
            heading,
            x_m,
            y_m,
            speed,
        ) in enumerate(definitions)
    )
    if tuple(fixture.fixture_id for fixture in fixtures) != FIXTURE_IDS:
        raise AssertionError("canonical fixture catalog drift")
    return fixtures


def fixture_table_sha256(fixtures: tuple[CanonicalSkierFixture, ...] | None = None) -> str:
    """Hash every canonical record for a known-answer deterministic fixture table."""
    selected = canonical_skier_fixtures() if fixtures is None else fixtures
    if tuple(fixture.fixture_id for fixture in selected) != FIXTURE_IDS:
        raise ValueError("fixtures: expected exact ordered canonical catalog")
    return sha256_canonical(
        {
            "schema_version": FIXTURE_SCHEMA_VERSION,
            "fixtures": {
                fixture.fixture_id: {
                    "root": fixture.root_payload(),
                    "records": [record.payload() for record in fixture.records()],
                }
                for fixture in selected
            },
        }
    )


__all__ = [
    "CANONICAL_RAMP_IN_TICKS",
    "CanonicalSkierFixture",
    "FIXTURE_IDS",
    "FIXTURE_SCHEMA_VERSION",
    "canonical_skier_fixtures",
    "fixture_table_sha256",
]
