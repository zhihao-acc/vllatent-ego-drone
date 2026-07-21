"""Versioned deterministic float64 skier mechanics for B3-CS2.

The module is renderer-neutral and PURE (stdlib + NumPy).  Advancement depends
only on the serialized slope, fixed smoke parameters, maneuver schedule, skier
state, and absolute integer tick.  It intentionally has no camera, branch,
visibility, wall-clock, random-generator, animation-rig, or renderer input.

Ramp intervals use the versioned half-open convention documented by
``SCHEDULE_SCHEMA_VERSION``: a ramp-in that starts at tick ``a`` evaluates
``H(0)=0`` at ``a`` and reaches its target at ``a + ramp_in_ticks``.  Hold then
continues through the half-open interval ending where ramp-out starts.  The CS2
forecast audit rejects every phase boundary in future ticks 1..8, so forecast
continuations are either an already-visible ramp or one steady hold.
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Final

import numpy as np

from vllatent.sim.contracts import (
    FIXED_DT_SECONDS,
    SKIER_POSE_ROOT_SCHEMA_VERSION,
    SKIER_ROOT_SCHEMA_VERSION,
    canonical_bytes,
    canonical_skier_digest,
)
from vllatent.sim.frames import rotation_geodesic_angle

SKIER_SCHEMA_VERSION: Final[str] = SKIER_ROOT_SCHEMA_VERSION
INTEGRATOR_VERSION: Final[str] = "b3-cs2-slope-plane-f64-v1"
SCHEDULE_SCHEMA_VERSION: Final[str] = "b3-cs2-absolute-tick-half-open-v1"
SKI_CONSTRUCTION_VERSION: Final[str] = "b3-cs2-ski-contact-v1"
ANIMATION_PARAMETER_VERSION: Final[str] = "b3-cs2-absolute-phase-params-v1"

SLOPE_ANGLE_RAD: Final[float] = math.pi / 12.0
GRAVITY_M_S2: Final[float] = 9.80665
MASS_KG: Final[float] = 75.0
AIR_DENSITY_KG_M3: Final[float] = 1.225
SNOW_FRICTION: Final[float] = 0.04
SKI_SIDECUT_RADIUS_M: Final[float] = 15.0
SKI_LENGTH_M: Final[float] = 1.70
SKI_WIDTH_M: Final[float] = 0.10
SKI_THICKNESS_M: Final[float] = 0.015
PARALLEL_STANCE_HALF_WIDTH_M: Final[float] = 0.16
MIN_INNER_TIP_GAP_M: Final[float] = 0.05
BINDING_HEIGHT_M: Final[float] = 0.04
CURVATURE_RESPONSE_SECONDS: Final[float] = 0.4
DRAG_AREA_HIGH_M2: Final[float] = 0.65
DRAG_AREA_MIDDLE_M2: Final[float] = 0.53
DRAG_AREA_TUCK_M2: Final[float] = 0.235
BRAKE_DECELERATION_CAP_M_S2: Final[float] = 6.0
SMOKE_SPEED_MIN_M_S: Final[float] = 2.0
SMOKE_SPEED_MAX_M_S: Final[float] = 12.0

# CS2 needs a deterministic armature origin to prove root construction before a
# real rig exists.  These versioned proof-only geometry values are not claimed as
# biomechanics and are replaced only under the separately gated CS3 rig contract.
PROOF_STANDING_PELVIS_HEIGHT_M: Final[float] = 1.0
PROOF_CROUCH_DROP_M: Final[float] = 0.35

_VECTOR_ATOL = 1.0e-12
_RESIDUAL_ATOL = 1.0e-10


def _f64(name: str, value: object, shape: tuple[int, ...]) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name}: expected np.ndarray, got {type(value).__name__}")
    if value.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {value.shape}")
    if value.dtype != np.dtype("<f8"):
        raise ValueError(f"{name}: expected dtype float64, got {value.dtype}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name}: expected finite values")
    contiguous = np.array(value, dtype=np.dtype("<f8"), order="C", copy=True)
    contiguous[contiguous == 0.0] = 0.0
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=np.dtype("<f8")).reshape(shape)


def _finite(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name}: expected real scalar, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name}: expected finite value")
    return 0.0 if result == 0.0 else result


def _integer(name: str, value: object, *, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name}: expected integer, got {type(value).__name__}")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name}: expected >= {minimum}, got {result}")
    return result


def _identifier(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name}: expected str, got {type(value).__name__}")
    result = unicodedata.normalize("NFC", value)
    if not result:
        raise ValueError(f"{name}: expected non-empty string")
    return result


def _transform(name: str, value: object) -> np.ndarray:
    result = _f64(name, value, (4, 4))
    if not np.allclose(result[3], np.array([0.0, 0.0, 0.0, 1.0]), rtol=0.0, atol=_VECTOR_ATOL):
        raise ValueError(f"{name}: invalid homogeneous bottom row")
    rotation = result[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=_VECTOR_ATOL) or not math.isclose(
        float(np.linalg.det(rotation)), 1.0, rel_tol=0.0, abs_tol=_VECTOR_ATOL
    ):
        raise ValueError(f"{name}: expected proper rotation")
    return result


def wrap_pi(angle_rad: float) -> float:
    """Wrap a finite angle to the half-open interval ``[-pi, pi)``."""
    angle = _finite("angle_rad", angle_rad)
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def quintic_smoothstep(s: float) -> float:
    """Return frozen ``6s^5 - 15s^4 + 10s^3`` for ``s`` in ``[0,1]``."""
    value = _finite("s", s)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"s: expected value in [0,1], got {value}")
    return value**3 * (10.0 + value * (-15.0 + 6.0 * value))


def drag_area_from_crouch(crouch: float) -> float:
    """Piecewise-linear ``C_d A(c)`` through the three frozen report points."""
    value = _finite("crouch", crouch)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"crouch: expected value in [0,1], got {value}")
    if value <= 0.5:
        return DRAG_AREA_HIGH_M2 + (DRAG_AREA_MIDDLE_M2 - DRAG_AREA_HIGH_M2) * (value / 0.5)
    return DRAG_AREA_MIDDLE_M2 + (DRAG_AREA_TUCK_M2 - DRAG_AREA_MIDDLE_M2) * ((value - 0.5) / 0.5)


@dataclass(frozen=True)
class SkierParameters:
    """Frozen smoke parameters from the locked research decision."""

    dt_seconds: float = FIXED_DT_SECONDS
    slope_angle_rad: float = SLOPE_ANGLE_RAD
    gravity_m_s2: float = GRAVITY_M_S2
    mass_kg: float = MASS_KG
    air_density_kg_m3: float = AIR_DENSITY_KG_M3
    snow_friction: float = SNOW_FRICTION
    ski_sidecut_radius_m: float = SKI_SIDECUT_RADIUS_M
    ski_length_m: float = SKI_LENGTH_M
    ski_width_m: float = SKI_WIDTH_M
    ski_thickness_m: float = SKI_THICKNESS_M
    parallel_stance_half_width_m: float = PARALLEL_STANCE_HALF_WIDTH_M
    minimum_inner_tip_gap_m: float = MIN_INNER_TIP_GAP_M
    binding_height_m: float = BINDING_HEIGHT_M
    curvature_response_seconds: float = CURVATURE_RESPONSE_SECONDS
    brake_deceleration_cap_m_s2: float = BRAKE_DECELERATION_CAP_M_S2
    smoke_speed_min_m_s: float = SMOKE_SPEED_MIN_M_S
    smoke_speed_max_m_s: float = SMOKE_SPEED_MAX_M_S
    proof_standing_pelvis_height_m: float = PROOF_STANDING_PELVIS_HEIGHT_M
    proof_crouch_drop_m: float = PROOF_CROUCH_DROP_M

    def __post_init__(self) -> None:
        expected = {
            "dt_seconds": FIXED_DT_SECONDS,
            "slope_angle_rad": SLOPE_ANGLE_RAD,
            "gravity_m_s2": GRAVITY_M_S2,
            "mass_kg": MASS_KG,
            "air_density_kg_m3": AIR_DENSITY_KG_M3,
            "snow_friction": SNOW_FRICTION,
            "ski_sidecut_radius_m": SKI_SIDECUT_RADIUS_M,
            "ski_length_m": SKI_LENGTH_M,
            "ski_width_m": SKI_WIDTH_M,
            "ski_thickness_m": SKI_THICKNESS_M,
            "parallel_stance_half_width_m": PARALLEL_STANCE_HALF_WIDTH_M,
            "minimum_inner_tip_gap_m": MIN_INNER_TIP_GAP_M,
            "binding_height_m": BINDING_HEIGHT_M,
            "curvature_response_seconds": CURVATURE_RESPONSE_SECONDS,
            "brake_deceleration_cap_m_s2": BRAKE_DECELERATION_CAP_M_S2,
            "smoke_speed_min_m_s": SMOKE_SPEED_MIN_M_S,
            "smoke_speed_max_m_s": SMOKE_SPEED_MAX_M_S,
            "proof_standing_pelvis_height_m": PROOF_STANDING_PELVIS_HEIGHT_M,
            "proof_crouch_drop_m": PROOF_CROUCH_DROP_M,
        }
        for name, frozen in expected.items():
            value = _finite(name, getattr(self, name))
            if value != frozen:
                raise ValueError(f"{name}: expected frozen value {frozen}, got {value}")

    def payload(self) -> dict[str, object]:
        """Return every frozen numeric parameter entering the root record."""
        return {
            "dt_seconds": self.dt_seconds,
            "slope_angle_rad": self.slope_angle_rad,
            "gravity_m_s2": self.gravity_m_s2,
            "mass_kg": self.mass_kg,
            "air_density_kg_m3": self.air_density_kg_m3,
            "snow_friction": self.snow_friction,
            "ski_sidecut_radius_m": self.ski_sidecut_radius_m,
            "ski_length_m": self.ski_length_m,
            "ski_width_m": self.ski_width_m,
            "ski_thickness_m": self.ski_thickness_m,
            "parallel_stance_half_width_m": self.parallel_stance_half_width_m,
            "minimum_inner_tip_gap_m": self.minimum_inner_tip_gap_m,
            "binding_height_m": self.binding_height_m,
            "curvature_response_seconds": self.curvature_response_seconds,
            "brake_deceleration_cap_m_s2": self.brake_deceleration_cap_m_s2,
            "smoke_speed_min_m_s": self.smoke_speed_min_m_s,
            "smoke_speed_max_m_s": self.smoke_speed_max_m_s,
            "proof_standing_pelvis_height_m": self.proof_standing_pelvis_height_m,
            "proof_crouch_drop_m": self.proof_crouch_drop_m,
        }


@dataclass(frozen=True, eq=False)
class SlopeFrame:
    """Immutable proper slope basis with ``n = r cross d``."""

    origin_world_m: np.ndarray
    downhill_world: np.ndarray
    right_world: np.ndarray
    normal_world: np.ndarray

    def __post_init__(self) -> None:
        origin = _f64("origin_world_m", self.origin_world_m, (3,))
        downhill = _f64("downhill_world", self.downhill_world, (3,))
        right = _f64("right_world", self.right_world, (3,))
        normal = _f64("normal_world", self.normal_world, (3,))
        for name, vector in (
            ("downhill_world", downhill),
            ("right_world", right),
            ("normal_world", normal),
        ):
            if not math.isclose(float(np.linalg.norm(vector)), 1.0, rel_tol=0.0, abs_tol=_VECTOR_ATOL):
                raise ValueError(f"{name}: expected unit vector")
        basis = np.column_stack((downhill, right, normal))
        if not np.allclose(basis.T @ basis, np.eye(3), rtol=0.0, atol=_VECTOR_ATOL):
            raise ValueError("slope basis: expected mutually orthogonal vectors")
        if not np.allclose(np.cross(right, downhill), normal, rtol=0.0, atol=_VECTOR_ATOL):
            raise ValueError("normal_world: must equal right_world cross downhill_world")
        object.__setattr__(self, "origin_world_m", origin)
        object.__setattr__(self, "downhill_world", downhill)
        object.__setattr__(self, "right_world", right)
        object.__setattr__(self, "normal_world", normal)

    def payload(self) -> dict[str, object]:
        return {
            "origin_world_m": self.origin_world_m,
            "downhill_world": self.downhill_world,
            "right_world": self.right_world,
            "normal_world": self.normal_world,
        }


def default_slope_frame() -> SlopeFrame:
    """Return the canonical 15-degree slope in a fixed world basis."""
    downhill = np.array(
        [math.cos(SLOPE_ANGLE_RAD), 0.0, -math.sin(SLOPE_ANGLE_RAD)],
        dtype=np.float64,
    )
    right = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    normal = np.cross(right, downhill)
    return SlopeFrame(np.zeros(3, dtype=np.float64), downhill, right, normal)


class ManeuverType(str, Enum):
    """Renderer-neutral maneuver laws used by the canonical CS2 fixtures."""

    STRAIGHT = "straight"
    ACCELERATE = "accelerate_tuck"
    BRAKE = "brake"
    CARVE_LEFT = "carve_left"
    CARVE_RIGHT = "carve_right"
    CROUCH = "crouch"
    TRANSITION = "transition"


@dataclass(frozen=True)
class ManeuverTargets:
    """One complete target vector; drag is checked against crouch, not overridden."""

    curvature_1_m: float
    left_edge_rad: float
    right_edge_rad: float
    left_attack_rad: float
    right_attack_rad: float
    crouch: float
    drag_area_m2: float
    brake_cap_m_s2: float

    def __post_init__(self) -> None:
        for name in (
            "curvature_1_m",
            "left_edge_rad",
            "right_edge_rad",
            "left_attack_rad",
            "right_attack_rad",
            "drag_area_m2",
            "brake_cap_m_s2",
        ):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        crouch = _finite("crouch", self.crouch)
        if crouch < 0.0 or crouch > 1.0:
            raise ValueError(f"crouch: expected value in [0,1], got {crouch}")
        object.__setattr__(self, "crouch", crouch)
        expected_drag = drag_area_from_crouch(crouch)
        if not math.isclose(self.drag_area_m2, expected_drag, rel_tol=0.0, abs_tol=1.0e-15):
            raise ValueError("drag_area_m2: must equal the frozen piecewise C_dA(crouch) law")
        if self.brake_cap_m_s2 < 0.0 or self.brake_cap_m_s2 > BRAKE_DECELERATION_CAP_M_S2:
            raise ValueError("brake_cap_m_s2: expected value in [0,6] for the bounded smoke law")

    def payload(self) -> dict[str, object]:
        return {
            "curvature_1_m": self.curvature_1_m,
            "left_edge_rad": self.left_edge_rad,
            "right_edge_rad": self.right_edge_rad,
            "left_attack_rad": self.left_attack_rad,
            "right_attack_rad": self.right_attack_rad,
            "crouch": self.crouch,
            "drag_area_m2": self.drag_area_m2,
            "brake_cap_m_s2": self.brake_cap_m_s2,
        }


def neutral_targets() -> ManeuverTargets:
    """Return the exact high-stance, flat-ski, unbraked target vector."""
    return ManeuverTargets(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, DRAG_AREA_HIGH_M2, 0.0)


def ideal_carve_curvature(edge_rad: float) -> float:
    """Return the report's signed steady high-edge sidecut relationship."""
    edge = _finite("edge_rad", edge_rad)
    if edge == 0.0:
        raise ValueError("edge_rad: ideal carve requires nonzero edge")
    return math.copysign(1.0 / (SKI_SIDECUT_RADIUS_M * math.cos(abs(edge))), edge)


@dataclass(frozen=True)
class ManeuverRecord:
    """One absolute-tick quintic maneuver record."""

    maneuver_id: str
    maneuver_type: ManeuverType
    continuation_law_id: str
    start_tick: int
    ramp_in_ticks: int
    hold_ticks: int
    ramp_out_ticks: int
    targets: ManeuverTargets

    def __post_init__(self) -> None:
        object.__setattr__(self, "maneuver_id", _identifier("maneuver_id", self.maneuver_id))
        object.__setattr__(
            self,
            "continuation_law_id",
            _identifier("continuation_law_id", self.continuation_law_id),
        )
        if not isinstance(self.maneuver_type, ManeuverType):
            raise TypeError(f"maneuver_type: expected ManeuverType, got {type(self.maneuver_type).__name__}")
        object.__setattr__(self, "start_tick", _integer("start_tick", self.start_tick))
        for name in ("ramp_in_ticks", "hold_ticks", "ramp_out_ticks"):
            object.__setattr__(self, name, _integer(name, getattr(self, name), minimum=0))
        if self.ramp_in_ticks + self.hold_ticks + self.ramp_out_ticks <= 0:
            raise ValueError("maneuver duration: expected at least one tick")
        if not isinstance(self.targets, ManeuverTargets):
            raise TypeError(f"targets: expected ManeuverTargets, got {type(self.targets).__name__}")
        self._validate_type_contract()

    @property
    def ramp_in_end_tick(self) -> int:
        return self.start_tick + self.ramp_in_ticks

    @property
    def ramp_out_start_tick(self) -> int:
        return self.ramp_in_end_tick + self.hold_ticks

    @property
    def end_tick(self) -> int:
        return self.ramp_out_start_tick + self.ramp_out_ticks

    def _validate_type_contract(self) -> None:
        target = self.targets
        flat = (
            target.left_edge_rad == 0.0
            and target.right_edge_rad == 0.0
            and target.left_attack_rad == 0.0
            and target.right_attack_rad == 0.0
        )
        if self.maneuver_type in (
            ManeuverType.STRAIGHT,
            ManeuverType.ACCELERATE,
            ManeuverType.CROUCH,
        ):
            if target.curvature_1_m != 0.0 or not flat or target.brake_cap_m_s2 != 0.0:
                raise ValueError(f"{self.maneuver_type.value}: expected straight flat skis")
        elif self.maneuver_type is ManeuverType.BRAKE:
            if abs(target.curvature_1_m) >= 1.0e-4:
                raise ValueError("brake: target curvature must remain near zero")
            if not (
                target.left_attack_rad >= math.radians(30.0)
                and target.right_attack_rad <= -math.radians(30.0)
                and target.left_attack_rad - target.right_attack_rad >= math.radians(60.0)
            ):
                raise ValueError("brake: opposing attacks must form at least a 60-degree wedge")
            if not (
                math.radians(25.0) <= target.left_edge_rad <= math.radians(35.0)
                and -math.radians(35.0) <= target.right_edge_rad <= -math.radians(25.0)
            ):
                raise ValueError("brake: opposing inside edges must be 25..35 degrees")
            if target.brake_cap_m_s2 <= 0.0:
                raise ValueError("brake: expected positive bounded brake cap")
        elif self.maneuver_type in (ManeuverType.CARVE_LEFT, ManeuverType.CARVE_RIGHT):
            expected_sign = -1.0 if self.maneuver_type is ManeuverType.CARVE_LEFT else 1.0
            if not (
                math.copysign(1.0, target.left_edge_rad) == expected_sign
                and math.copysign(1.0, target.right_edge_rad) == expected_sign
                and abs(target.left_edge_rad) > math.radians(45.0)
                and abs(target.right_edge_rad) > math.radians(45.0)
                and abs(target.left_attack_rad) < math.radians(5.0)
                and abs(target.right_attack_rad) < math.radians(5.0)
                and target.brake_cap_m_s2 == 0.0
            ):
                raise ValueError("carve: steady target requires signed edge >45 degrees and parallel attack <5 degrees")
            expected = ideal_carve_curvature(target.left_edge_rad)
            if not math.isclose(target.curvature_1_m, expected, rel_tol=0.0, abs_tol=1.0e-15):
                raise ValueError("carve: steady target must obey the sidecut relationship")
        elif self.maneuver_type is ManeuverType.TRANSITION:
            if target.brake_cap_m_s2 != 0.0:
                raise ValueError("transition: brake cap must be zero")

    def payload(self) -> dict[str, object]:
        """Serialize the complete absolute-tick law, including every duration."""
        return {
            "maneuver_id": self.maneuver_id,
            "maneuver_type": self.maneuver_type,
            "continuation_law_id": self.continuation_law_id,
            "start_tick": self.start_tick,
            "ramp_in_ticks": self.ramp_in_ticks,
            "hold_ticks": self.hold_ticks,
            "ramp_out_ticks": self.ramp_out_ticks,
            "targets": self.targets.payload(),
        }


@dataclass(frozen=True)
class EvaluatedManeuver:
    """Absolute-tick evaluated targets and deterministic phase metadata."""

    maneuver_id: str
    maneuver_type: ManeuverType
    continuation_law_id: str
    targets: ManeuverTargets
    weight: float
    maneuver_phase: float
    phase_name: str


@dataclass(frozen=True)
class ManeuverSchedule:
    """Sorted, non-overlapping absolute-tick records and first-record baseline."""

    baseline_targets: ManeuverTargets
    records: tuple[ManeuverRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.baseline_targets, ManeuverTargets):
            raise TypeError("baseline_targets: expected ManeuverTargets")
        if not isinstance(self.records, tuple) or not self.records:
            raise ValueError("records: expected a non-empty tuple")
        previous: ManeuverRecord | None = None
        ids: set[str] = set()
        for index, record in enumerate(self.records):
            if not isinstance(record, ManeuverRecord):
                raise TypeError(f"records[{index}]: expected ManeuverRecord")
            if record.maneuver_id in ids:
                raise ValueError(f"records: duplicate maneuver_id {record.maneuver_id!r}")
            ids.add(record.maneuver_id)
            if previous is not None and record.start_tick <= previous.end_tick:
                raise ValueError("records: absolute maneuver intervals must not overlap or share a boundary")
            previous = record

    def evaluate(self, absolute_tick: int) -> EvaluatedManeuver:
        """Evaluate schedule deterministically from the absolute integer tick."""
        tick = _integer("absolute_tick", absolute_tick)
        first = self.records[0]
        if tick < first.start_tick:
            return EvaluatedManeuver(
                "baseline",
                ManeuverType.STRAIGHT,
                "baseline-steady-v1",
                self.baseline_targets,
                0.0,
                0.0,
                "baseline",
            )
        for index, record in enumerate(self.records):
            if tick < record.end_tick:
                source = self.baseline_targets if index == 0 else neutral_targets()
                return _evaluate_record(record, source, tick)
            next_index = index + 1
            if next_index < len(self.records) and tick < self.records[next_index].start_tick:
                return EvaluatedManeuver(
                    "between-records",
                    ManeuverType.STRAIGHT,
                    "between-records-steady-v1",
                    neutral_targets(),
                    0.0,
                    0.0,
                    "between",
                )
        return EvaluatedManeuver(
            "terminal-baseline",
            ManeuverType.STRAIGHT,
            "terminal-baseline-steady-v1",
            neutral_targets(),
            0.0,
            1.0,
            "terminal",
        )

    def payload(self) -> dict[str, object]:
        """Return the sorted schedule and its first-record source target."""
        return {
            "schema_version": SCHEDULE_SCHEMA_VERSION,
            "baseline_targets": self.baseline_targets.payload(),
            "records": [record.payload() for record in self.records],
        }


def _interpolate_targets(source: ManeuverTargets, target: ManeuverTargets, weight: float) -> ManeuverTargets:
    w = _finite("weight", weight)
    if w < 0.0 or w > 1.0:
        raise ValueError("weight: expected value in [0,1]")
    crouch = source.crouch + w * (target.crouch - source.crouch)
    return ManeuverTargets(
        curvature_1_m=source.curvature_1_m + w * (target.curvature_1_m - source.curvature_1_m),
        left_edge_rad=source.left_edge_rad + w * (target.left_edge_rad - source.left_edge_rad),
        right_edge_rad=source.right_edge_rad + w * (target.right_edge_rad - source.right_edge_rad),
        left_attack_rad=source.left_attack_rad + w * (target.left_attack_rad - source.left_attack_rad),
        right_attack_rad=source.right_attack_rad + w * (target.right_attack_rad - source.right_attack_rad),
        crouch=crouch,
        drag_area_m2=drag_area_from_crouch(crouch),
        brake_cap_m_s2=source.brake_cap_m_s2 + w * (target.brake_cap_m_s2 - source.brake_cap_m_s2),
    )


def _evaluate_record(record: ManeuverRecord, source: ManeuverTargets, tick: int) -> EvaluatedManeuver:
    elapsed = tick - record.start_tick
    total = record.ramp_in_ticks + record.hold_ticks + record.ramp_out_ticks
    maneuver_phase = min(max(elapsed / total, 0.0), 1.0)
    if elapsed < record.ramp_in_ticks:
        weight = quintic_smoothstep(elapsed / record.ramp_in_ticks)
        targets = _interpolate_targets(source, record.targets, weight)
        phase_name = "ramp_in"
    elif elapsed < record.ramp_in_ticks + record.hold_ticks:
        weight = 1.0
        targets = record.targets
        phase_name = "hold"
    elif elapsed < total and record.ramp_out_ticks > 0:
        progress = (elapsed - record.ramp_in_ticks - record.hold_ticks) / record.ramp_out_ticks
        weight = 1.0 - quintic_smoothstep(progress)
        targets = _interpolate_targets(neutral_targets(), record.targets, weight)
        phase_name = "ramp_out"
    else:
        weight = 0.0
        targets = neutral_targets()
        phase_name = "end"
    return EvaluatedManeuver(
        record.maneuver_id,
        record.maneuver_type,
        record.continuation_law_id,
        targets,
        weight,
        maneuver_phase,
        phase_name,
    )


@dataclass(frozen=True, eq=False)
class SkierState:
    """Minimal authoritative planar state and serialized pose/randomness inputs."""

    absolute_tick: int
    x_m: float
    y_m: float
    heading_rad: float
    speed_m_s: float
    curvature_1_m: float
    tracked_joint_positions_root_m: np.ndarray
    local_bone_transforms: np.ndarray
    randomness_seed: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "absolute_tick", _integer("absolute_tick", self.absolute_tick))
        for name in ("x_m", "y_m", "speed_m_s", "curvature_1_m"):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        if self.speed_m_s < 0.0:
            raise ValueError("speed_m_s: expected nonnegative value")
        object.__setattr__(self, "heading_rad", wrap_pi(self.heading_rad))
        if not isinstance(self.tracked_joint_positions_root_m, np.ndarray):
            raise TypeError("tracked_joint_positions_root_m: expected np.ndarray")
        joints = self.tracked_joint_positions_root_m
        if joints.ndim != 2 or joints.shape[1:] != (3,):
            raise ValueError(f"tracked_joint_positions_root_m: expected shape (J,3), got {joints.shape}")
        object.__setattr__(
            self,
            "tracked_joint_positions_root_m",
            _f64("tracked_joint_positions_root_m", joints, joints.shape),
        )
        if not isinstance(self.local_bone_transforms, np.ndarray):
            raise TypeError("local_bone_transforms: expected np.ndarray")
        bones = self.local_bone_transforms
        if bones.ndim != 3 or bones.shape[1:] != (4, 4):
            raise ValueError(f"local_bone_transforms: expected shape (J,4,4), got {bones.shape}")
        validated = _f64("local_bone_transforms", bones, bones.shape)
        for index, transform in enumerate(validated):
            _transform(f"local_bone_transforms[{index}]", transform.copy())
        object.__setattr__(self, "local_bone_transforms", validated)
        object.__setattr__(self, "randomness_seed", _integer("randomness_seed", self.randomness_seed))

    def payload(self) -> dict[str, object]:
        return {
            "absolute_tick": self.absolute_tick,
            "x_m": self.x_m,
            "y_m": self.y_m,
            "heading_rad": self.heading_rad,
            "speed_m_s": self.speed_m_s,
            "curvature_1_m": self.curvature_1_m,
            "tracked_joint_positions_root_m": self.tracked_joint_positions_root_m,
            "local_bone_transforms": self.local_bone_transforms,
            "randomness_seed": self.randomness_seed,
        }


def tangent_axes(slope: SlopeFrame, heading_rad: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ``t(psi)`` and ``l(psi)`` in the frozen slope basis."""
    if not isinstance(slope, SlopeFrame):
        raise TypeError(f"slope: expected SlopeFrame, got {type(slope).__name__}")
    heading = wrap_pi(heading_rad)
    tangent = math.cos(heading) * slope.downhill_world + math.sin(heading) * slope.right_world
    lateral = -math.sin(heading) * slope.downhill_world + math.cos(heading) * slope.right_world
    tangent.setflags(write=False)
    lateral.setflags(write=False)
    return tangent, lateral


@dataclass(frozen=True, eq=False)
class RootGeometry:
    """Constructed ground-root and pelvis/armature transforms."""

    ground_point_world_m: np.ndarray
    pelvis_point_world_m: np.ndarray
    tangent_world: np.ndarray
    lateral_world: np.ndarray
    T_world_from_groundroot: np.ndarray
    T_world_from_armature: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "ground_point_world_m",
            "pelvis_point_world_m",
            "tangent_world",
            "lateral_world",
        ):
            object.__setattr__(self, name, _f64(name, getattr(self, name), (3,)))
        object.__setattr__(
            self,
            "T_world_from_groundroot",
            _transform("T_world_from_groundroot", self.T_world_from_groundroot),
        )
        object.__setattr__(
            self,
            "T_world_from_armature",
            _transform("T_world_from_armature", self.T_world_from_armature),
        )


def construct_root_geometry(
    state: SkierState,
    evaluated: EvaluatedManeuver,
    slope: SlopeFrame,
    parameters: SkierParameters,
) -> RootGeometry:
    """Construct both world roots from planar state and scheduled crouch."""
    if not isinstance(state, SkierState) or not isinstance(evaluated, EvaluatedManeuver):
        raise TypeError("state/evaluated: expected SkierState and EvaluatedManeuver")
    if not isinstance(slope, SlopeFrame) or not isinstance(parameters, SkierParameters):
        raise TypeError("slope/parameters: expected SlopeFrame and SkierParameters")
    tangent, lateral = tangent_axes(slope, state.heading_rad)
    ground = slope.origin_world_m + state.x_m * slope.downhill_world + state.y_m * slope.right_world
    pelvis_height = (
        parameters.proof_standing_pelvis_height_m - parameters.proof_crouch_drop_m * evaluated.targets.crouch
    )
    pelvis = ground + pelvis_height * slope.normal_world
    rotation = np.column_stack((tangent, lateral, -slope.normal_world))
    ground_transform = np.eye(4, dtype=np.float64)
    ground_transform[:3, :3] = rotation
    ground_transform[:3, 3] = ground
    armature_transform = ground_transform.copy()
    armature_transform[:3, 3] = pelvis
    return RootGeometry(
        ground,
        pelvis,
        tangent,
        lateral,
        ground_transform,
        armature_transform,
    )


@dataclass(frozen=True, eq=False)
class SkiGeometry:
    """One commanded/realized ski, binding, and analytic contact construction."""

    side: str
    attack_rad: float
    edge_rad: float
    centerline_origin_world_m: np.ndarray
    base_origin_world_m: np.ndarray
    binding_origin_world_m: np.ndarray
    contact_origin_world_m: np.ndarray
    forward_world: np.ndarray
    zero_edge_right_world: np.ndarray
    edged_right_world: np.ndarray
    outward_normal_world: np.ndarray
    commanded_F_world_from_ski: np.ndarray
    realized_F_world_from_ski: np.ndarray
    analytic_slip_longitudinal_lateral_m_s: np.ndarray
    realized_slip_longitudinal_lateral_m_s: np.ndarray
    realized_attack_rad: float
    realized_edge_rad: float
    frame_orientation_residual_rad: float

    def __post_init__(self) -> None:
        side = _identifier("side", self.side)
        if side not in ("left", "right"):
            raise ValueError("side: expected 'left' or 'right'")
        object.__setattr__(self, "side", side)
        for name in (
            "attack_rad",
            "edge_rad",
            "realized_attack_rad",
            "realized_edge_rad",
            "frame_orientation_residual_rad",
        ):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        for name in (
            "centerline_origin_world_m",
            "base_origin_world_m",
            "binding_origin_world_m",
            "contact_origin_world_m",
            "forward_world",
            "zero_edge_right_world",
            "edged_right_world",
            "outward_normal_world",
        ):
            object.__setattr__(self, name, _f64(name, getattr(self, name), (3,)))
        for name in ("commanded_F_world_from_ski", "realized_F_world_from_ski"):
            frame = _f64(name, getattr(self, name), (3, 3))
            if not np.allclose(frame.T @ frame, np.eye(3), rtol=0.0, atol=_VECTOR_ATOL):
                raise ValueError(f"{name}: expected orthonormal frame")
            if not math.isclose(float(np.linalg.det(frame)), 1.0, rel_tol=0.0, abs_tol=_VECTOR_ATOL):
                raise ValueError(f"{name}: expected determinant +1")
            object.__setattr__(self, name, frame)
        for name in (
            "analytic_slip_longitudinal_lateral_m_s",
            "realized_slip_longitudinal_lateral_m_s",
        ):
            object.__setattr__(self, name, _f64(name, getattr(self, name), (2,)))


@dataclass(frozen=True, eq=False)
class SkiPairGeometry:
    """Left/right ski pair plus exact stance and inner-tip clearance diagnostics."""

    left: SkiGeometry
    right: SkiGeometry
    stance_half_width_m: float
    centerline_ordering_m: float
    inner_tip_gap_m: float

    def __post_init__(self) -> None:
        if not isinstance(self.left, SkiGeometry) or self.left.side != "left":
            raise TypeError("left: expected left SkiGeometry")
        if not isinstance(self.right, SkiGeometry) or self.right.side != "right":
            raise TypeError("right: expected right SkiGeometry")
        for name in ("stance_half_width_m", "centerline_ordering_m", "inner_tip_gap_m"):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))


def _construct_one_ski(
    *,
    side: str,
    centerline: np.ndarray,
    attack_rad: float,
    edge_rad: float,
    root: RootGeometry,
    slope: SlopeFrame,
    speed_m_s: float,
    parameters: SkierParameters,
) -> SkiGeometry:
    attack = _finite("attack_rad", attack_rad)
    edge = _finite("edge_rad", edge_rad)
    tangent = root.tangent_world
    lateral = root.lateral_world
    normal = slope.normal_world
    forward = math.cos(attack) * tangent + math.sin(attack) * lateral
    zero_edge_right = -math.sin(attack) * tangent + math.cos(attack) * lateral
    edged_right = math.cos(edge) * zero_edge_right - math.sin(edge) * normal
    outward_normal = math.cos(edge) * normal + math.sin(edge) * zero_edge_right
    frame = np.column_stack((forward, edged_right, -outward_normal))
    base = centerline + parameters.ski_width_m * 0.5 * abs(math.sin(edge)) * normal
    sigma = 0.0 if edge == 0.0 else math.copysign(1.0, edge)
    contact = base + sigma * parameters.ski_width_m * 0.5 * edged_right
    binding = base + parameters.binding_height_m * outward_normal
    realized_attack = math.atan2(float(np.dot(forward, lateral)), float(np.dot(forward, tangent)))
    realized_edge = math.atan2(
        -float(np.dot(edged_right, normal)),
        float(np.dot(edged_right, zero_edge_right)),
    )
    velocity = speed_m_s * tangent
    analytic_slip = np.array(
        [speed_m_s * math.cos(attack), -speed_m_s * math.sin(attack)],
        dtype=np.float64,
    )
    realized_slip = np.array(
        [float(np.dot(velocity, forward)), float(np.dot(velocity, zero_edge_right))],
        dtype=np.float64,
    )
    # Commanded and realized frames are byte-identical in the analytic CS2
    # construction.  Preserve the exact zero instead of amplifying roundoff by
    # applying acos near one; independent tests still recompute both frames.
    orientation_residual = 0.0
    return SkiGeometry(
        side=side,
        attack_rad=attack,
        edge_rad=edge,
        centerline_origin_world_m=centerline,
        base_origin_world_m=base,
        binding_origin_world_m=binding,
        contact_origin_world_m=contact,
        forward_world=forward,
        zero_edge_right_world=zero_edge_right,
        edged_right_world=edged_right,
        outward_normal_world=outward_normal,
        commanded_F_world_from_ski=np.array(frame, copy=True),
        realized_F_world_from_ski=np.array(frame, copy=True),
        analytic_slip_longitudinal_lateral_m_s=analytic_slip,
        realized_slip_longitudinal_lateral_m_s=realized_slip,
        realized_attack_rad=realized_attack,
        realized_edge_rad=realized_edge,
        frame_orientation_residual_rad=orientation_residual,
    )


def construct_skis(
    state: SkierState,
    evaluated: EvaluatedManeuver,
    root: RootGeometry,
    slope: SlopeFrame,
    parameters: SkierParameters,
) -> SkiPairGeometry:
    """Construct exact stance, base, binding, contact, frames, and slip values."""
    target = evaluated.targets
    if evaluated.maneuver_type is ManeuverType.BRAKE:
        stance_half_width = max(
            parameters.parallel_stance_half_width_m,
            parameters.ski_length_m
            * 0.5
            * max(abs(math.sin(target.left_attack_rad)), abs(math.sin(target.right_attack_rad)))
            + (parameters.ski_width_m + parameters.minimum_inner_tip_gap_m) * 0.5,
        )
    else:
        stance_half_width = parameters.parallel_stance_half_width_m
    left_center = root.ground_point_world_m - stance_half_width * root.lateral_world
    right_center = root.ground_point_world_m + stance_half_width * root.lateral_world
    left = _construct_one_ski(
        side="left",
        centerline=left_center,
        attack_rad=target.left_attack_rad,
        edge_rad=target.left_edge_rad,
        root=root,
        slope=slope,
        speed_m_s=state.speed_m_s,
        parameters=parameters,
    )
    right = _construct_one_ski(
        side="right",
        centerline=right_center,
        attack_rad=target.right_attack_rad,
        edge_rad=target.right_edge_rad,
        root=root,
        slope=slope,
        speed_m_s=state.speed_m_s,
        parameters=parameters,
    )
    ordering = float(np.dot(right_center - left_center, root.lateral_world))
    left_tip = left_center + parameters.ski_length_m * 0.5 * left.forward_world
    right_tip = right_center + parameters.ski_length_m * 0.5 * right.forward_world
    inner_tip_gap = float(np.dot(right_tip - left_tip, root.lateral_world)) - parameters.ski_width_m
    return SkiPairGeometry(left, right, stance_half_width, ordering, inner_tip_gap)


def bounded_brake_deceleration(targets: ManeuverTargets, parameters: SkierParameters) -> float:
    """Return the exact average two-ski skid term capped by the maneuver record."""
    if not isinstance(targets, ManeuverTargets) or not isinstance(parameters, SkierParameters):
        raise TypeError("targets/parameters: expected ManeuverTargets and SkierParameters")
    skid_sum = 0.0
    for edge, attack in (
        (targets.left_edge_rad, targets.left_attack_rad),
        (targets.right_edge_rad, targets.right_attack_rad),
    ):
        skid_sum += (
            parameters.gravity_m_s2 * math.cos(parameters.slope_angle_rad) * math.tan(abs(edge)) * abs(math.sin(attack))
        )
    return min(targets.brake_cap_m_s2, 0.5 * skid_sum)


@dataclass(frozen=True, eq=False)
class StepDiagnostics:
    """All scalar/vector intermediates in one frozen root-law update."""

    next_curvature_1_m: float
    q_m_s2: float
    normal_load_accel_m_s2: float
    brake_deceleration_m_s2: float
    longitudinal_acceleration_m_s2: float
    next_speed_m_s: float
    next_heading_rad: float
    next_ground_point_world_m: np.ndarray
    next_x_m: float
    next_y_m: float
    planar_recovery_residual_m: float

    def __post_init__(self) -> None:
        for name in (
            "next_curvature_1_m",
            "q_m_s2",
            "normal_load_accel_m_s2",
            "brake_deceleration_m_s2",
            "longitudinal_acceleration_m_s2",
            "next_speed_m_s",
            "next_heading_rad",
            "next_x_m",
            "next_y_m",
            "planar_recovery_residual_m",
        ):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        object.__setattr__(
            self,
            "next_ground_point_world_m",
            _f64("next_ground_point_world_m", self.next_ground_point_world_m, (3,)),
        )


def step_diagnostics(
    state: SkierState,
    schedule: ManeuverSchedule,
    slope: SlopeFrame,
    parameters: SkierParameters,
) -> StepDiagnostics:
    """Evaluate the exact report equation at ``state.absolute_tick``."""
    if not isinstance(state, SkierState) or not isinstance(schedule, ManeuverSchedule):
        raise TypeError("state/schedule: expected SkierState and ManeuverSchedule")
    evaluated = schedule.evaluate(state.absolute_tick)
    target = evaluated.targets
    gain = 1.0 - math.exp(-parameters.dt_seconds / parameters.curvature_response_seconds)
    next_curvature = state.curvature_1_m + gain * (target.curvature_1_m - state.curvature_1_m)
    q = state.speed_m_s**2 * next_curvature + parameters.gravity_m_s2 * math.sin(parameters.slope_angle_rad) * math.sin(
        state.heading_rad
    )
    normal_load = math.sqrt((parameters.gravity_m_s2 * math.cos(parameters.slope_angle_rad)) ** 2 + q**2)
    brake = bounded_brake_deceleration(target, parameters)
    acceleration = (
        parameters.gravity_m_s2 * math.sin(parameters.slope_angle_rad) * math.cos(state.heading_rad)
        - parameters.snow_friction * normal_load
        - parameters.air_density_kg_m3
        * drag_area_from_crouch(target.crouch)
        * state.speed_m_s**2
        / (2.0 * parameters.mass_kg)
        - brake
    )
    next_speed = max(0.0, state.speed_m_s + acceleration * parameters.dt_seconds)
    next_heading = wrap_pi(
        state.heading_rad + 0.5 * (state.speed_m_s + next_speed) * next_curvature * parameters.dt_seconds
    )
    tangent_now, _ = tangent_axes(slope, state.heading_rad)
    tangent_next, _ = tangent_axes(slope, next_heading)
    ground_now = slope.origin_world_m + state.x_m * slope.downhill_world + state.y_m * slope.right_world
    ground_next = ground_now + 0.5 * parameters.dt_seconds * (state.speed_m_s * tangent_now + next_speed * tangent_next)
    relative = ground_next - slope.origin_world_m
    next_x = float(np.dot(relative, slope.downhill_world))
    next_y = float(np.dot(relative, slope.right_world))
    recovered = slope.origin_world_m + next_x * slope.downhill_world + next_y * slope.right_world
    residual = float(np.linalg.norm(recovered - ground_next))
    return StepDiagnostics(
        next_curvature,
        q,
        normal_load,
        brake,
        acceleration,
        next_speed,
        next_heading,
        ground_next,
        next_x,
        next_y,
        residual,
    )


def advance_skier(
    state: SkierState,
    schedule: ManeuverSchedule,
    slope: SlopeFrame,
    parameters: SkierParameters,
) -> SkierState:
    """Advance exactly one fixed step with no observation- or branch-side input."""
    diagnostics = step_diagnostics(state, schedule, slope, parameters)
    return SkierState(
        absolute_tick=state.absolute_tick + 1,
        x_m=diagnostics.next_x_m,
        y_m=diagnostics.next_y_m,
        heading_rad=diagnostics.next_heading_rad,
        speed_m_s=diagnostics.next_speed_m_s,
        curvature_1_m=diagnostics.next_curvature_1_m,
        tracked_joint_positions_root_m=state.tracked_joint_positions_root_m,
        local_bone_transforms=state.local_bone_transforms,
        randomness_seed=state.randomness_seed,
    )


@dataclass(frozen=True, eq=False)
class AnimationParameters:
    """Absolute-tick clip selection parameters; no evaluated rig pose is claimed."""

    clip_ids: tuple[str, ...]
    blend_weights: np.ndarray
    animation_phase: float

    def __post_init__(self) -> None:
        if not isinstance(self.clip_ids, tuple) or not self.clip_ids:
            raise ValueError("clip_ids: expected non-empty tuple")
        normalized = tuple(_identifier("clip_id", value) for value in self.clip_ids)
        object.__setattr__(self, "clip_ids", normalized)
        weights = _f64("blend_weights", self.blend_weights, (len(normalized),))
        if np.any(weights < 0.0) or not math.isclose(float(weights.sum()), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("blend_weights: expected nonnegative weights summing to one")
        object.__setattr__(self, "blend_weights", weights)
        phase = _finite("animation_phase", self.animation_phase)
        if phase < 0.0 or phase >= 1.0:
            raise ValueError("animation_phase: expected value in [0,1)")
        object.__setattr__(self, "animation_phase", phase)


def absolute_animation_parameters(
    evaluated: EvaluatedManeuver,
    absolute_tick: int,
    parameters: SkierParameters,
) -> AnimationParameters:
    """Derive clip IDs, weights, and a one-second phase only from absolute tick."""
    tick = _integer("absolute_tick", absolute_tick)
    if not isinstance(evaluated, EvaluatedManeuver) or not isinstance(parameters, SkierParameters):
        raise TypeError("evaluated/parameters: expected EvaluatedManeuver/SkierParameters")
    clip_by_type = {
        ManeuverType.STRAIGHT: "straight_high",
        ManeuverType.ACCELERATE: "straight_tuck",
        ManeuverType.BRAKE: "brake_wedge",
        ManeuverType.CARVE_LEFT: "carve_left",
        ManeuverType.CARVE_RIGHT: "carve_right",
        ManeuverType.CROUCH: "crouch",
        ManeuverType.TRANSITION: "transition_flexion",
    }
    active_clip = clip_by_type[evaluated.maneuver_type]
    clips: tuple[str, ...]
    if active_clip == "straight_high":
        clips = ("straight_high",)
        weights = np.array([1.0], dtype=np.float64)
    else:
        clips = ("straight_high", active_clip)
        weights = np.array([1.0 - evaluated.weight, evaluated.weight], dtype=np.float64)
    phase = (tick * parameters.dt_seconds) % 1.0
    return AnimationParameters(clips, weights, phase)


@dataclass(frozen=True, eq=False)
class SkierFrameRecord:
    """Complete renderer-neutral per-tick state used by CS2 replay and digest proof."""

    state: SkierState
    evaluated_maneuver: EvaluatedManeuver
    animation: AnimationParameters
    root: RootGeometry
    skis: SkiPairGeometry
    acceleration_m_s2: float
    omega_rad_s: float
    gross_lean_rad: float
    world_velocity_m_s: np.ndarray
    world_acceleration_m_s2: np.ndarray
    root_schema_version: str = SKIER_ROOT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.state, SkierState):
            raise TypeError("state: expected SkierState")
        if not isinstance(self.evaluated_maneuver, EvaluatedManeuver):
            raise TypeError("evaluated_maneuver: expected EvaluatedManeuver")
        if not isinstance(self.animation, AnimationParameters):
            raise TypeError("animation: expected AnimationParameters")
        if not isinstance(self.root, RootGeometry) or not isinstance(self.skis, SkiPairGeometry):
            raise TypeError("root/skis: expected RootGeometry/SkiPairGeometry")
        if self.root_schema_version not in (SKIER_ROOT_SCHEMA_VERSION, SKIER_POSE_ROOT_SCHEMA_VERSION):
            raise ValueError("root_schema_version: unsupported skier root schema")
        if self.root_schema_version == SKIER_POSE_ROOT_SCHEMA_VERSION:
            if self.state.tracked_joint_positions_root_m.shape != (17, 3):
                raise ValueError("CS3 pose root requires exactly 17 tracked joint positions")
            if self.state.local_bone_transforms.shape != (17, 4, 4):
                raise ValueError("CS3 pose root requires exactly 17 local bone transforms")
        object.__setattr__(self, "acceleration_m_s2", _finite("acceleration_m_s2", self.acceleration_m_s2))
        object.__setattr__(self, "omega_rad_s", _finite("omega_rad_s", self.omega_rad_s))
        object.__setattr__(self, "gross_lean_rad", _finite("gross_lean_rad", self.gross_lean_rad))
        object.__setattr__(
            self,
            "world_velocity_m_s",
            _f64("world_velocity_m_s", self.world_velocity_m_s, (3,)),
        )
        object.__setattr__(
            self,
            "world_acceleration_m_s2",
            _f64("world_acceleration_m_s2", self.world_acceleration_m_s2, (3,)),
        )

    def payload(self) -> dict[str, object]:
        """Return the complete versioned canonical PURE record payload."""
        target = self.evaluated_maneuver.targets

        def ski_payload(ski: SkiGeometry) -> dict[str, object]:
            return {
                "side": ski.side,
                "attack_rad": ski.attack_rad,
                "edge_rad": ski.edge_rad,
                "centerline_origin_world_m": ski.centerline_origin_world_m,
                "base_origin_world_m": ski.base_origin_world_m,
                "binding_origin_world_m": ski.binding_origin_world_m,
                "contact_origin_world_m": ski.contact_origin_world_m,
                "target_F_world_from_ski": ski.commanded_F_world_from_ski,
                "realized_F_world_from_ski": ski.realized_F_world_from_ski,
                "analytic_slip_longitudinal_lateral_m_s": (ski.analytic_slip_longitudinal_lateral_m_s),
                "realized_slip_longitudinal_lateral_m_s": (ski.realized_slip_longitudinal_lateral_m_s),
                "realized_attack_rad": ski.realized_attack_rad,
                "realized_edge_rad": ski.realized_edge_rad,
                "frame_orientation_residual_rad": ski.frame_orientation_residual_rad,
            }

        return {
            "schema_version": self.root_schema_version,
            "integrator_version": INTEGRATOR_VERSION,
            "schedule_version": SCHEDULE_SCHEMA_VERSION,
            "ski_construction_version": SKI_CONSTRUCTION_VERSION,
            "animation_parameter_version": ANIMATION_PARAMETER_VERSION,
            "acceleration_semantics": "a_at_tick_for_interval_to_tick_plus_1",
            "slip_component_axes": "ski_forward_and_zero_edge_tangent_right",
            "absolute_tick": self.state.absolute_tick,
            "position_xy_m": np.array([self.state.x_m, self.state.y_m], dtype=np.float64),
            "heading_rad": self.state.heading_rad,
            "speed_m_s": self.state.speed_m_s,
            "acceleration_m_s2": self.acceleration_m_s2,
            "curvature_1_m": self.state.curvature_1_m,
            "omega_rad_s": self.omega_rad_s,
            "gross_lean_rad": self.gross_lean_rad,
            "world_velocity_m_s": self.world_velocity_m_s,
            "world_acceleration_m_s2": self.world_acceleration_m_s2,
            "maneuver": {
                "id": self.evaluated_maneuver.maneuver_id,
                "type": self.evaluated_maneuver.maneuver_type,
                "continuation_law_id": self.evaluated_maneuver.continuation_law_id,
                "weight": self.evaluated_maneuver.weight,
                "phase": self.evaluated_maneuver.maneuver_phase,
                "phase_name": self.evaluated_maneuver.phase_name,
                "targets": target.payload(),
            },
            "animation": {
                "clip_ids": self.animation.clip_ids,
                "blend_weights": self.animation.blend_weights,
                "phase": self.animation.animation_phase,
            },
            "root": {
                "ground_point_world_m": self.root.ground_point_world_m,
                "pelvis_point_world_m": self.root.pelvis_point_world_m,
                "T_world_from_groundroot": self.root.T_world_from_groundroot,
                "T_world_from_armature": self.root.T_world_from_armature,
            },
            "skis": {
                "dimensions_m": np.array([SKI_LENGTH_M, SKI_WIDTH_M, SKI_THICKNESS_M], dtype=np.float64),
                "stance_half_width_m": self.skis.stance_half_width_m,
                "centerline_ordering_m": self.skis.centerline_ordering_m,
                "inner_tip_gap_m": self.skis.inner_tip_gap_m,
                "left": ski_payload(self.skis.left),
                "right": ski_payload(self.skis.right),
            },
            "tracked_joint_positions_root_m": self.state.tracked_joint_positions_root_m,
            "local_bone_transforms": self.state.local_bone_transforms,
            "randomness_seed": self.state.randomness_seed,
        }

    def canonical_bytes(self) -> bytes:
        """Serialize this record with the shared canonical little-endian contract."""
        return canonical_bytes(self.payload())

    def skier_digest(self) -> str:
        """Hash only root/skis/contacts/phases/bone pose/randomness domains."""
        payload = self.payload()
        skis = payload["skis"]
        assert isinstance(skis, dict)
        left = skis["left"]
        right = skis["right"]
        assert isinstance(left, dict) and isinstance(right, dict)
        return canonical_skier_digest(
            root={
                "schema_version": self.root_schema_version,
                "absolute_tick": self.state.absolute_tick,
                "position_xy_m": payload["position_xy_m"],
                "heading_rad": self.state.heading_rad,
                "speed_m_s": self.state.speed_m_s,
                "acceleration_m_s2": self.acceleration_m_s2,
                "curvature_1_m": self.state.curvature_1_m,
                "omega_rad_s": self.omega_rad_s,
                "gross_lean_rad": self.gross_lean_rad,
                "T_world_from_groundroot": self.root.T_world_from_groundroot,
                "T_world_from_armature": self.root.T_world_from_armature,
                "tracked_joint_positions_root_m": (self.state.tracked_joint_positions_root_m),
            },
            skis=skis,
            contacts={
                "left_contact_origin_world_m": left["contact_origin_world_m"],
                "right_contact_origin_world_m": right["contact_origin_world_m"],
            },
            phases={
                "maneuver_id": self.evaluated_maneuver.maneuver_id,
                "maneuver_phase": self.evaluated_maneuver.maneuver_phase,
                "animation_clip_ids": self.animation.clip_ids,
                "animation_phase": self.animation.animation_phase,
                "animation_blend_weights": self.animation.blend_weights,
            },
            local_bone_transforms=self.state.local_bone_transforms,
            randomness={"seed": self.state.randomness_seed},
        )


def frame_record(
    state: SkierState,
    schedule: ManeuverSchedule,
    slope: SlopeFrame,
    parameters: SkierParameters,
) -> SkierFrameRecord:
    """Construct one canonical per-tick record and exact mechanical diagnostics."""
    evaluated = schedule.evaluate(state.absolute_tick)
    diagnostics = step_diagnostics(state, schedule, slope, parameters)
    root = construct_root_geometry(state, evaluated, slope, parameters)
    skis = construct_skis(state, evaluated, root, slope, parameters)
    tangent, _ = tangent_axes(slope, state.heading_rad)
    next_tangent, _ = tangent_axes(slope, diagnostics.next_heading_rad)
    velocity = state.speed_m_s * tangent
    next_velocity = diagnostics.next_speed_m_s * next_tangent
    world_acceleration = (next_velocity - velocity) / parameters.dt_seconds
    animation = absolute_animation_parameters(evaluated, state.absolute_tick, parameters)
    return SkierFrameRecord(
        state=state,
        evaluated_maneuver=evaluated,
        animation=animation,
        root=root,
        skis=skis,
        acceleration_m_s2=diagnostics.longitudinal_acceleration_m_s2,
        omega_rad_s=state.speed_m_s * state.curvature_1_m,
        gross_lean_rad=math.atan2(
            diagnostics.q_m_s2,
            parameters.gravity_m_s2 * math.cos(parameters.slope_angle_rad),
        ),
        world_velocity_m_s=velocity,
        world_acceleration_m_s2=world_acceleration,
    )


def simulate_records(
    initial_state: SkierState,
    schedule: ManeuverSchedule,
    slope: SlopeFrame,
    parameters: SkierParameters,
    *,
    final_tick: int,
) -> tuple[SkierFrameRecord, ...]:
    """Replay inclusive absolute ticks without stateful randomness or hidden time."""
    end = _integer("final_tick", final_tick)
    if end < initial_state.absolute_tick:
        raise ValueError("final_tick: expected >= initial state's absolute_tick")
    state = initial_state
    records: list[SkierFrameRecord] = []
    while state.absolute_tick <= end:
        records.append(frame_record(state, schedule, slope, parameters))
        if state.absolute_tick < end:
            state = advance_skier(state, schedule, slope, parameters)
        else:
            break
    return tuple(records)


def root_construction_residuals(
    record: SkierFrameRecord, slope: SlopeFrame, parameters: SkierParameters
) -> dict[str, float]:
    """Independently reconstruct and compare both root origins and frames."""
    state = record.state
    evaluated = record.evaluated_maneuver
    cosine = math.cos(state.heading_rad)
    sine = math.sin(state.heading_rad)
    expected_tangent = cosine * slope.downhill_world + sine * slope.right_world
    expected_lateral = -sine * slope.downhill_world + cosine * slope.right_world
    expected_rotation = np.column_stack((expected_tangent, expected_lateral, -slope.normal_world))
    expected_ground = slope.origin_world_m + state.x_m * slope.downhill_world + state.y_m * slope.right_world
    pelvis_height = (
        parameters.proof_standing_pelvis_height_m - parameters.proof_crouch_drop_m * evaluated.targets.crouch
    )
    expected_pelvis = expected_ground + pelvis_height * slope.normal_world

    return {
        "ground_point_m": float(np.linalg.norm(record.root.ground_point_world_m - expected_ground)),
        "pelvis_point_m": float(np.linalg.norm(record.root.pelvis_point_world_m - expected_pelvis)),
        "tangent_world": float(np.linalg.norm(record.root.tangent_world - expected_tangent)),
        "lateral_world": float(np.linalg.norm(record.root.lateral_world - expected_lateral)),
        "ground_origin_m": float(np.linalg.norm(record.root.T_world_from_groundroot[:3, 3] - expected_ground)),
        "armature_origin_m": float(np.linalg.norm(record.root.T_world_from_armature[:3, 3] - expected_pelvis)),
        "ground_frame_rad": rotation_geodesic_angle(expected_rotation, record.root.T_world_from_groundroot[:3, :3]),
        "armature_frame_rad": rotation_geodesic_angle(expected_rotation, record.root.T_world_from_armature[:3, :3]),
        "pelvis_height_m": abs(
            float(
                np.dot(
                    record.root.pelvis_point_world_m - record.root.ground_point_world_m,
                    slope.normal_world,
                )
            )
            - pelvis_height
        ),
        "ground_planarity_m": abs(
            float(
                np.dot(
                    record.root.ground_point_world_m - slope.origin_world_m,
                    slope.normal_world,
                )
            )
        ),
    }


def ski_construction_residuals(
    record: SkierFrameRecord, slope: SlopeFrame, parameters: SkierParameters
) -> dict[str, float]:
    """Independently reconstruct every ski origin/frame/angle/slip diagnostic."""
    state = record.state
    target = record.evaluated_maneuver.targets
    cosine_heading = math.cos(state.heading_rad)
    sine_heading = math.sin(state.heading_rad)
    tangent = cosine_heading * slope.downhill_world + sine_heading * slope.right_world
    lateral = -sine_heading * slope.downhill_world + cosine_heading * slope.right_world
    ground = slope.origin_world_m + state.x_m * slope.downhill_world + state.y_m * slope.right_world
    if record.evaluated_maneuver.maneuver_type is ManeuverType.BRAKE:
        expected_half_width = max(
            parameters.parallel_stance_half_width_m,
            parameters.ski_length_m
            * 0.5
            * max(
                abs(math.sin(target.left_attack_rad)),
                abs(math.sin(target.right_attack_rad)),
            )
            + (parameters.ski_width_m + parameters.minimum_inner_tip_gap_m) * 0.5,
        )
    else:
        expected_half_width = parameters.parallel_stance_half_width_m
    expected_centers = {
        "left": ground - expected_half_width * lateral,
        "right": ground + expected_half_width * lateral,
    }
    expected_attacks = {
        "left": target.left_attack_rad,
        "right": target.right_attack_rad,
    }
    expected_forwards = {
        side: math.cos(attack) * tangent + math.sin(attack) * lateral for side, attack in expected_attacks.items()
    }
    actual_ordering = float(
        np.dot(
            record.skis.right.centerline_origin_world_m - record.skis.left.centerline_origin_world_m,
            lateral,
        )
    )
    left_tip = expected_centers["left"] + parameters.ski_length_m * 0.5 * expected_forwards["left"]
    right_tip = expected_centers["right"] + parameters.ski_length_m * 0.5 * expected_forwards["right"]
    recomputed_tip_gap = float(np.dot(right_tip - left_tip, lateral)) - parameters.ski_width_m
    residuals: dict[str, float] = {
        "centerline_ordering_margin_m": actual_ordering,
        "inner_tip_gap_margin_m": recomputed_tip_gap,
        "centerline_ordering_residual_m": abs(actual_ordering - 2.0 * expected_half_width),
        "stored_centerline_ordering_m": abs(record.skis.centerline_ordering_m - actual_ordering),
        "inner_tip_gap_residual_m": abs(record.skis.inner_tip_gap_m - recomputed_tip_gap),
        "stance_width_m": abs(record.skis.stance_half_width_m - expected_half_width),
    }
    velocity = state.speed_m_s * tangent
    ski_inputs = (
        (record.skis.left, target.left_attack_rad, target.left_edge_rad),
        (record.skis.right, target.right_attack_rad, target.right_edge_rad),
    )
    for ski, expected_attack, expected_edge in ski_inputs:
        prefix = ski.side
        expected_center = expected_centers[prefix]
        expected_forward = math.cos(expected_attack) * tangent + math.sin(expected_attack) * lateral
        expected_zero_edge_right = -math.sin(expected_attack) * tangent + math.cos(expected_attack) * lateral
        expected_edged_right = (
            math.cos(expected_edge) * expected_zero_edge_right - math.sin(expected_edge) * slope.normal_world
        )
        expected_outward_normal = (
            math.cos(expected_edge) * slope.normal_world + math.sin(expected_edge) * expected_zero_edge_right
        )
        expected_frame = np.column_stack((expected_forward, expected_edged_right, -expected_outward_normal))
        sigma = 0.0 if expected_edge == 0.0 else math.copysign(1.0, expected_edge)
        expected_base = (
            expected_center + parameters.ski_width_m * 0.5 * abs(math.sin(expected_edge)) * slope.normal_world
        )
        expected_contact = expected_base + sigma * parameters.ski_width_m * 0.5 * expected_edged_right
        expected_binding = expected_base + parameters.binding_height_m * expected_outward_normal
        realized_frame = ski.realized_F_world_from_ski
        frame_error = rotation_geodesic_angle(expected_frame, realized_frame)
        realized_forward = realized_frame[:, 0]
        realized_edged_right = realized_frame[:, 1]
        realized_attack = math.atan2(
            float(np.dot(realized_forward, lateral)),
            float(np.dot(realized_forward, tangent)),
        )
        realized_edge = math.atan2(
            -float(np.dot(realized_edged_right, slope.normal_world)),
            float(np.dot(realized_edged_right, expected_zero_edge_right)),
        )
        realized_zero_edge_right = (realized_edged_right + math.sin(expected_edge) * slope.normal_world) / math.cos(
            expected_edge
        )
        recomputed_realized_slip = np.array(
            [
                float(np.dot(velocity, realized_forward)),
                float(np.dot(velocity, realized_zero_edge_right)),
            ],
            dtype=np.float64,
        )
        analytic_slip = np.array(
            [
                state.speed_m_s * math.cos(expected_attack),
                -state.speed_m_s * math.sin(expected_attack),
            ],
            dtype=np.float64,
        )
        residuals[f"{prefix}_centerline_origin_m"] = float(
            np.linalg.norm(ski.centerline_origin_world_m - expected_center)
        )
        residuals[f"{prefix}_base_origin_m"] = float(np.linalg.norm(ski.base_origin_world_m - expected_base))
        residuals[f"{prefix}_contact_origin_m"] = float(np.linalg.norm(ski.contact_origin_world_m - expected_contact))
        residuals[f"{prefix}_contact_plane_m"] = abs(
            float(
                np.dot(
                    ski.contact_origin_world_m - expected_center,
                    slope.normal_world,
                )
            )
        )
        residuals[f"{prefix}_binding_origin_m"] = float(np.linalg.norm(ski.binding_origin_world_m - expected_binding))
        residuals[f"{prefix}_target_frame_rad"] = rotation_geodesic_angle(
            expected_frame, ski.commanded_F_world_from_ski
        )
        residuals[f"{prefix}_frame_rad"] = frame_error
        residuals[f"{prefix}_forward_world"] = float(np.linalg.norm(ski.forward_world - expected_forward))
        residuals[f"{prefix}_zero_edge_right_world"] = float(
            np.linalg.norm(ski.zero_edge_right_world - expected_zero_edge_right)
        )
        residuals[f"{prefix}_edged_right_world"] = float(np.linalg.norm(ski.edged_right_world - expected_edged_right))
        residuals[f"{prefix}_outward_normal_world"] = float(
            np.linalg.norm(ski.outward_normal_world - expected_outward_normal)
        )
        residuals[f"{prefix}_attack_rad"] = abs(realized_attack - expected_attack)
        residuals[f"{prefix}_edge_rad"] = abs(realized_edge - expected_edge)
        residuals[f"{prefix}_stored_attack_rad"] = abs(ski.realized_attack_rad - realized_attack)
        residuals[f"{prefix}_stored_edge_rad"] = abs(ski.realized_edge_rad - realized_edge)
        residuals[f"{prefix}_stored_frame_residual_rad"] = abs(ski.frame_orientation_residual_rad - frame_error)
        residuals[f"{prefix}_slip_m_s"] = float(np.linalg.norm(recomputed_realized_slip - analytic_slip))
        residuals[f"{prefix}_stored_analytic_slip_m_s"] = float(
            np.linalg.norm(ski.analytic_slip_longitudinal_lateral_m_s - analytic_slip)
        )
        residuals[f"{prefix}_stored_realized_slip_m_s"] = float(
            np.linalg.norm(ski.realized_slip_longitudinal_lateral_m_s - recomputed_realized_slip)
        )
    return residuals


def residuals_within_contract(residuals: dict[str, float]) -> bool:
    """Check zero-valued construction residuals, excluding positive gate margins."""
    return all(value <= _RESIDUAL_ATOL for name, value in residuals.items() if not name.endswith("margin_m"))


__all__ = [
    "AIR_DENSITY_KG_M3",
    "ANIMATION_PARAMETER_VERSION",
    "AnimationParameters",
    "BINDING_HEIGHT_M",
    "BRAKE_DECELERATION_CAP_M_S2",
    "CURVATURE_RESPONSE_SECONDS",
    "DRAG_AREA_HIGH_M2",
    "DRAG_AREA_MIDDLE_M2",
    "DRAG_AREA_TUCK_M2",
    "EvaluatedManeuver",
    "GRAVITY_M_S2",
    "INTEGRATOR_VERSION",
    "MASS_KG",
    "MIN_INNER_TIP_GAP_M",
    "ManeuverRecord",
    "ManeuverSchedule",
    "ManeuverTargets",
    "ManeuverType",
    "PARALLEL_STANCE_HALF_WIDTH_M",
    "SCHEDULE_SCHEMA_VERSION",
    "SKIER_SCHEMA_VERSION",
    "SKI_CONSTRUCTION_VERSION",
    "SKI_LENGTH_M",
    "SKI_SIDECUT_RADIUS_M",
    "SKI_THICKNESS_M",
    "SKI_WIDTH_M",
    "SLOPE_ANGLE_RAD",
    "SMOKE_SPEED_MAX_M_S",
    "SMOKE_SPEED_MIN_M_S",
    "SNOW_FRICTION",
    "SkiGeometry",
    "SkiPairGeometry",
    "SkierFrameRecord",
    "SkierParameters",
    "SkierState",
    "SlopeFrame",
    "StepDiagnostics",
    "absolute_animation_parameters",
    "advance_skier",
    "bounded_brake_deceleration",
    "construct_root_geometry",
    "construct_skis",
    "default_slope_frame",
    "drag_area_from_crouch",
    "frame_record",
    "ideal_carve_curvature",
    "neutral_targets",
    "quintic_smoothstep",
    "residuals_within_contract",
    "root_construction_residuals",
    "simulate_records",
    "ski_construction_residuals",
    "step_diagnostics",
    "tangent_axes",
    "wrap_pi",
]
