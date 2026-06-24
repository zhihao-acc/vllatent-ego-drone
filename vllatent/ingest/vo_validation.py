"""MegaSaM VO trajectory validation metrics (PURE tier).

Physics-plausibility + smoothness proxies for monocular VO on skiing FPV.
No ground truth needed — uses trajectory shape, speed limits, jerk analysis,
confidence distribution, and scale drift detection.

All numpy/stdlib. No torch, no MegaSaM dependency.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vllatent.ingest.ego_motion import se3_sequence_to_deltas

# ---------------------------------------------------------------------------
# Physics constants for skiing FPV
# ---------------------------------------------------------------------------

MAX_SKIING_SPEED_MS = 40.0       # m/s (world-class downhill ~150 km/h)
MAX_YAW_RATE_DEG_S = 300.0       # deg/s — beyond this is VO failure
ACCEL_DISCONTINUITY_SIGMA = 3.0  # jumps beyond 3σ of acceleration
LOW_CONFIDENCE_THRESHOLD = 0.3


# ---------------------------------------------------------------------------
# Report types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SmoothnessReport:
    mean_jerk: float                 # mean jerk magnitude (m/s³)
    max_jerk: float                  # peak jerk
    n_accel_discontinuities: int     # acceleration jumps > 3σ
    n_angular_spikes: int            # yaw rate spikes > 300°/s
    mean_angular_velocity: float     # mean yaw rate (deg/s)


@dataclass(frozen=True)
class PhysicsReport:
    passes: bool
    max_speed: float                 # m/s
    mean_speed: float                # m/s
    n_speed_violations: int          # frames exceeding MAX_SKIING_SPEED_MS
    max_yaw_rate: float              # deg/s
    net_altitude_change: float       # total dz (negative = descent)
    reasons: list[str]               # failure reasons if not passes


@dataclass(frozen=True)
class ConfidenceReport:
    mean: float
    median: float
    frac_low: float                  # fraction below LOW_CONFIDENCE_THRESHOLD
    longest_low_run: int             # longest contiguous low-confidence run
    n_frames: int


@dataclass(frozen=True)
class DriftReport:
    drift_ratio: float              # speed_last_quarter / speed_first_quarter
    speed_first_quarter: float      # mean displacement in first 25%
    speed_last_quarter: float       # mean displacement in last 25%


@dataclass(frozen=True)
class Verdict:
    decision: str                   # "GO" | "CONDITIONAL-GO" | "NO-GO"
    checks: dict[str, str]          # per-check: "pass" | "warn" | "fail"
    reasons: list[str]              # human-readable explanation


@dataclass(frozen=True)
class ClipValidationReport:
    smoothness: SmoothnessReport
    physics: PhysicsReport
    confidence: ConfidenceReport
    drift: DriftReport
    verdict: Verdict
    n_frames: int
    clip_id: str


# ---------------------------------------------------------------------------
# Smoothness analysis
# ---------------------------------------------------------------------------

def trajectory_smoothness(poses: np.ndarray, *, fps: float = 5.0) -> SmoothnessReport:
    """Analyze trajectory smoothness via jerk and angular velocity."""
    n = poses.shape[0]
    if n < 4:
        return SmoothnessReport(
            mean_jerk=0.0, max_jerk=0.0,
            n_accel_discontinuities=0, n_angular_spikes=0,
            mean_angular_velocity=0.0,
        )

    positions = poses[:, :3, 3]  # (N, 3)
    dt = 1.0 / fps

    velocity = np.diff(positions, axis=0) / dt           # (N-1, 3)
    acceleration = np.diff(velocity, axis=0) / dt         # (N-2, 3)
    jerk = np.diff(acceleration, axis=0) / dt             # (N-3, 3)

    jerk_mag = np.linalg.norm(jerk, axis=1)
    mean_jerk = float(np.mean(jerk_mag))
    max_jerk = float(np.max(jerk_mag)) if len(jerk_mag) > 0 else 0.0

    # Acceleration discontinuities: accel magnitude > median + 3*MAD (robust to outliers).
    # Fallback: if MAD ≈ 0 (near-constant accel), use 5× median as threshold.
    accel_mag = np.linalg.norm(acceleration, axis=1)
    if len(accel_mag) > 0:
        med = float(np.median(accel_mag))
        mad = float(np.median(np.abs(accel_mag - med)))
        if mad > 1e-10:
            threshold = med + ACCEL_DISCONTINUITY_SIGMA * 1.4826 * mad
        elif med > 1e-10:
            threshold = med * 5.0
        else:
            threshold = 1.0
        n_discontinuities = int(np.sum(accel_mag > threshold))
    else:
        n_discontinuities = 0

    # Angular velocity from rotation matrices
    yaw_rates = _yaw_rates_from_poses(poses, fps)
    n_angular_spikes = int(np.sum(np.abs(yaw_rates) > MAX_YAW_RATE_DEG_S))
    mean_angular = float(np.mean(np.abs(yaw_rates))) if len(yaw_rates) > 0 else 0.0

    return SmoothnessReport(
        mean_jerk=mean_jerk,
        max_jerk=max_jerk,
        n_accel_discontinuities=n_discontinuities,
        n_angular_spikes=n_angular_spikes,
        mean_angular_velocity=mean_angular,
    )


def _yaw_rates_from_poses(poses: np.ndarray, fps: float) -> np.ndarray:
    """Extract per-frame yaw rate (deg/s) from SE(3) rotation matrices."""
    n = poses.shape[0]
    if n < 2:
        return np.array([], dtype=np.float64)

    yaws = np.array([
        np.degrees(np.arctan2(poses[i, 1, 0], poses[i, 0, 0]))
        for i in range(n)
    ])

    dyaw = np.diff(yaws)
    # Wrap to [-180, 180]
    dyaw = (dyaw + 180.0) % 360.0 - 180.0
    return dyaw * fps  # deg/s


# ---------------------------------------------------------------------------
# Physics plausibility
# ---------------------------------------------------------------------------

def physics_plausibility(deltas: np.ndarray, *, fps: float = 5.0) -> PhysicsReport:
    """Check physical plausibility of body-frame deltas."""
    if deltas.shape[0] == 0:
        return PhysicsReport(
            passes=True, max_speed=0.0, mean_speed=0.0,
            n_speed_violations=0, max_yaw_rate=0.0,
            net_altitude_change=0.0, reasons=[],
        )

    xyz = deltas[:, :3]
    displacements = np.linalg.norm(xyz, axis=1)
    speeds = displacements * fps  # m/s

    max_speed = float(np.max(speeds))
    mean_speed = float(np.mean(speeds))
    n_violations = int(np.sum(speeds > MAX_SKIING_SPEED_MS))

    yaw_rates = np.abs(deltas[:, 3]) * fps  # deg/s
    max_yaw_rate = float(np.max(yaw_rates)) if len(yaw_rates) > 0 else 0.0

    net_altitude = float(np.sum(deltas[:, 2]))

    reasons: list[str] = []
    if n_violations > len(deltas) * 0.1:
        reasons.append(f"{n_violations}/{len(deltas)} frames exceed {MAX_SKIING_SPEED_MS} m/s")
    if max_yaw_rate > MAX_YAW_RATE_DEG_S:
        reasons.append(f"peak yaw rate {max_yaw_rate:.0f}°/s exceeds {MAX_YAW_RATE_DEG_S}°/s")

    return PhysicsReport(
        passes=len(reasons) == 0,
        max_speed=max_speed,
        mean_speed=mean_speed,
        n_speed_violations=n_violations,
        max_yaw_rate=max_yaw_rate,
        net_altitude_change=net_altitude,
        reasons=reasons,
    )


# ---------------------------------------------------------------------------
# Confidence analysis
# ---------------------------------------------------------------------------

def confidence_analysis(confidences: np.ndarray) -> ConfidenceReport:
    """Analyze VO confidence distribution."""
    n = len(confidences)
    if n == 0:
        return ConfidenceReport(mean=0.0, median=0.0, frac_low=0.0, longest_low_run=0, n_frames=0)

    low_mask = confidences < LOW_CONFIDENCE_THRESHOLD
    frac_low = float(np.mean(low_mask))

    # Longest contiguous low-confidence run
    longest = 0
    current = 0
    for is_low in low_mask:
        if is_low:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return ConfidenceReport(
        mean=float(np.mean(confidences)),
        median=float(np.median(confidences)),
        frac_low=frac_low,
        longest_low_run=longest,
        n_frames=n,
    )


# ---------------------------------------------------------------------------
# Scale drift
# ---------------------------------------------------------------------------

def scale_drift(poses: np.ndarray) -> DriftReport:
    """Detect monocular scale drift by comparing speed in first vs last quarter."""
    n = poses.shape[0]
    if n < 8:
        return DriftReport(drift_ratio=1.0, speed_first_quarter=0.0, speed_last_quarter=0.0)

    positions = poses[:, :3, 3]
    displacements = np.linalg.norm(np.diff(positions, axis=0), axis=1)

    q = max(1, n // 4)
    first_q = displacements[:q]
    last_q = displacements[-q:]

    mean_first = float(np.mean(first_q))
    mean_last = float(np.mean(last_q))

    if mean_first < 1e-10:
        ratio = 1.0 if mean_last < 1e-10 else float("inf")
    else:
        ratio = mean_last / mean_first

    return DriftReport(
        drift_ratio=ratio,
        speed_first_quarter=mean_first,
        speed_last_quarter=mean_last,
    )


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def vo_verdict(
    smoothness: SmoothnessReport,
    physics: PhysicsReport,
    confidence: ConfidenceReport,
    drift: DriftReport,
) -> Verdict:
    """Combine all checks into a GO / CONDITIONAL-GO / NO-GO verdict."""
    checks: dict[str, str] = {}
    reasons: list[str] = []

    # Smoothness
    if smoothness.n_accel_discontinuities == 0 and smoothness.n_angular_spikes == 0:
        checks["smoothness"] = "pass"
    elif smoothness.n_angular_spikes > 5 or smoothness.n_accel_discontinuities > 10:
        checks["smoothness"] = "fail"
        reasons.append(f"smoothness: {smoothness.n_accel_discontinuities} accel jumps, "
                       f"{smoothness.n_angular_spikes} angular spikes")
    else:
        checks["smoothness"] = "warn"
        reasons.append(f"smoothness: minor issues ({smoothness.n_accel_discontinuities} accel, "
                       f"{smoothness.n_angular_spikes} angular)")

    # Physics
    if physics.passes:
        checks["physics"] = "pass"
    else:
        checks["physics"] = "fail"
        reasons.extend(physics.reasons)

    # Confidence
    if confidence.frac_low < 0.1:
        checks["confidence"] = "pass"
    elif confidence.frac_low < 0.3:
        checks["confidence"] = "warn"
        reasons.append(f"confidence: {confidence.frac_low:.0%} low-confidence frames")
    else:
        checks["confidence"] = "fail"
        reasons.append(f"confidence: {confidence.frac_low:.0%} low-confidence frames, "
                       f"longest run = {confidence.longest_low_run}")

    # Drift
    if 0.5 < drift.drift_ratio < 2.0:
        checks["drift"] = "pass"
    elif 0.3 < drift.drift_ratio < 3.0:
        checks["drift"] = "warn"
        reasons.append(f"drift: speed ratio first/last quarter = {drift.drift_ratio:.2f}")
    else:
        checks["drift"] = "fail"
        reasons.append(f"drift: severe scale drift, ratio = {drift.drift_ratio:.2f}")

    # Decision
    n_fail = sum(1 for v in checks.values() if v == "fail")
    n_warn = sum(1 for v in checks.values() if v == "warn")

    if n_fail >= 2:
        decision = "NO-GO"
    elif n_fail == 1:
        decision = "CONDITIONAL-GO"
    elif n_warn >= 2:
        decision = "CONDITIONAL-GO"
    else:
        decision = "GO"

    return Verdict(decision=decision, checks=checks, reasons=reasons)


# ---------------------------------------------------------------------------
# Top-level convenience
# ---------------------------------------------------------------------------

def validate_clip(
    poses: np.ndarray,
    confidences: np.ndarray,
    *,
    fps: float = 5.0,
    clip_id: str = "",
) -> ClipValidationReport:
    """Run full validation on a single clip's MegaSaM output."""
    deltas = se3_sequence_to_deltas(poses)

    sm = trajectory_smoothness(poses, fps=fps)
    ph = physics_plausibility(deltas, fps=fps)
    ca = confidence_analysis(confidences)
    sd = scale_drift(poses)
    vd = vo_verdict(sm, ph, ca, sd)

    return ClipValidationReport(
        smoothness=sm,
        physics=ph,
        confidence=ca,
        drift=sd,
        verdict=vd,
        n_frames=poses.shape[0],
        clip_id=clip_id,
    )


__all__ = [
    "SmoothnessReport",
    "PhysicsReport",
    "ConfidenceReport",
    "DriftReport",
    "Verdict",
    "ClipValidationReport",
    "trajectory_smoothness",
    "physics_plausibility",
    "confidence_analysis",
    "scale_drift",
    "vo_verdict",
    "validate_clip",
]
