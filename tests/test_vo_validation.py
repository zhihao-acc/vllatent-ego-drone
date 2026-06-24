"""Tests for MegaSaM VO validation metrics (PURE tier).

Synthetic trajectories: smooth descent, jerky, stationary, circular,
physically implausible. No MegaSaM dependency.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _smooth_descent(n: int = 100, fps: float = 5.0) -> np.ndarray:
    """Smooth downhill ski trajectory: forward + descending, gentle turns.

    Returns (N, 4, 4) SE(3) poses.
    """
    poses = np.zeros((n, 4, 4), dtype=np.float64)
    dt = 1.0 / fps
    speed = 8.0  # m/s forward
    descent_rate = 2.0  # m/s down
    turn_rate = np.radians(5.0)  # rad/s gentle yaw

    x, y, z, yaw = 0.0, 0.0, 0.0, 0.0
    for i in range(n):
        poses[i, :3, :3] = _yaw_rotation(yaw)
        poses[i, :3, 3] = [x, y, z]
        poses[i, 3, 3] = 1.0
        x += speed * dt * np.cos(yaw)
        y += speed * dt * np.sin(yaw)
        z -= descent_rate * dt
        yaw += turn_rate * dt
    return poses


def _jerky_trajectory(n: int = 100, fps: float = 5.0) -> np.ndarray:
    """Trajectory with sudden jumps — simulates VO failure."""
    poses = _smooth_descent(n, fps)
    # Insert 3 teleportation jumps
    for jump_idx in [20, 50, 75]:
        if jump_idx < n:
            poses[jump_idx, 0, 3] += 10.0  # 10m x-jump
            poses[jump_idx, 2, 3] += 5.0   # 5m z-jump
    return poses


def _stationary(n: int = 50) -> np.ndarray:
    """All poses at origin — degenerate."""
    poses = np.zeros((n, 4, 4), dtype=np.float64)
    for i in range(n):
        poses[i] = np.eye(4)
    return poses


def _physically_implausible(n: int = 100, fps: float = 5.0) -> np.ndarray:
    """Trajectory with impossible speeds (100 m/s at 5 Hz = 20m/frame)."""
    poses = np.zeros((n, 4, 4), dtype=np.float64)
    for i in range(n):
        poses[i] = np.eye(4)
        poses[i, 0, 3] = float(i) * 20.0  # 20m per frame = 100 m/s
    return poses


def _yaw_rotation(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _poses_to_deltas_simple(poses: np.ndarray) -> np.ndarray:
    """Simple pose-to-delta for test fixtures (position diff + yaw diff)."""
    from vllatent.ingest.ego_motion import se3_sequence_to_deltas
    return se3_sequence_to_deltas(poses)


# ---------------------------------------------------------------------------
# Tests — SmoothnessReport
# ---------------------------------------------------------------------------

class TestTrajectorySmoothness:
    def test_smooth_trajectory_low_jerk(self) -> None:
        from vllatent.ingest.vo_validation import trajectory_smoothness
        poses = _smooth_descent(100, fps=5.0)
        report = trajectory_smoothness(poses, fps=5.0)
        assert report.mean_jerk < 5.0, f"Smooth descent should have low jerk, got {report.mean_jerk}"
        assert report.n_accel_discontinuities == 0

    def test_jerky_trajectory_high_jerk(self) -> None:
        from vllatent.ingest.vo_validation import trajectory_smoothness
        poses = _jerky_trajectory(100, fps=5.0)
        report = trajectory_smoothness(poses, fps=5.0)
        assert report.n_accel_discontinuities > 0

    def test_stationary_zero_jerk(self) -> None:
        from vllatent.ingest.vo_validation import trajectory_smoothness
        poses = _stationary(50)
        report = trajectory_smoothness(poses, fps=5.0)
        assert report.mean_jerk == pytest.approx(0.0, abs=1e-6)

    def test_angular_velocity_spikes(self) -> None:
        from vllatent.ingest.vo_validation import trajectory_smoothness
        poses = _smooth_descent(100, fps=5.0)
        # Insert a 180° yaw flip at frame 40
        poses[40, :3, :3] = _yaw_rotation(np.pi)
        report = trajectory_smoothness(poses, fps=5.0)
        assert report.n_angular_spikes > 0


# ---------------------------------------------------------------------------
# Tests — PhysicsReport
# ---------------------------------------------------------------------------

class TestPhysicsPlausibility:
    def test_smooth_descent_passes(self) -> None:
        from vllatent.ingest.vo_validation import physics_plausibility
        poses = _smooth_descent(100, fps=5.0)
        deltas = _poses_to_deltas_simple(poses)
        report = physics_plausibility(deltas, fps=5.0)
        assert report.passes

    def test_implausible_speed_fails(self) -> None:
        from vllatent.ingest.vo_validation import physics_plausibility
        poses = _physically_implausible(100, fps=5.0)
        deltas = _poses_to_deltas_simple(poses)
        report = physics_plausibility(deltas, fps=5.0)
        assert not report.passes
        assert report.max_speed > 50.0

    def test_descent_detected(self) -> None:
        from vllatent.ingest.vo_validation import physics_plausibility
        poses = _smooth_descent(100, fps=5.0)
        deltas = _poses_to_deltas_simple(poses)
        report = physics_plausibility(deltas, fps=5.0)
        assert report.net_altitude_change < 0, "Downhill should have negative altitude change"

    def test_stationary_passes(self) -> None:
        from vllatent.ingest.vo_validation import physics_plausibility
        poses = _stationary(50)
        deltas = _poses_to_deltas_simple(poses)
        report = physics_plausibility(deltas, fps=5.0)
        assert report.passes


# ---------------------------------------------------------------------------
# Tests — ConfidenceReport
# ---------------------------------------------------------------------------

class TestConfidenceAnalysis:
    def test_high_confidence(self) -> None:
        from vllatent.ingest.vo_validation import confidence_analysis
        conf = np.ones(100, dtype=np.float64)
        report = confidence_analysis(conf)
        assert report.frac_low == 0.0
        assert report.longest_low_run == 0

    def test_all_low_confidence(self) -> None:
        from vllatent.ingest.vo_validation import confidence_analysis
        conf = np.full(100, 0.1, dtype=np.float64)
        report = confidence_analysis(conf)
        assert report.frac_low == pytest.approx(1.0)
        assert report.longest_low_run == 100

    def test_mixed_confidence(self) -> None:
        from vllatent.ingest.vo_validation import confidence_analysis
        conf = np.ones(100, dtype=np.float64)
        conf[30:45] = 0.1  # 15-frame low-confidence run
        report = confidence_analysis(conf)
        assert report.longest_low_run == 15
        assert 0.1 < report.frac_low < 0.2


# ---------------------------------------------------------------------------
# Tests — DriftReport
# ---------------------------------------------------------------------------

class TestScaleDrift:
    def test_no_drift(self) -> None:
        from vllatent.ingest.vo_validation import scale_drift
        poses = _smooth_descent(100, fps=5.0)
        report = scale_drift(poses)
        assert abs(report.drift_ratio - 1.0) < 0.3, "Smooth descent should have ~1.0 drift ratio"

    def test_stationary_no_drift(self) -> None:
        from vllatent.ingest.vo_validation import scale_drift
        poses = _stationary(50)
        report = scale_drift(poses)
        assert report.drift_ratio == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Tests — Verdict
# ---------------------------------------------------------------------------

class TestVoVerdict:
    def test_smooth_descent_go(self) -> None:
        from vllatent.ingest.vo_validation import (
            confidence_analysis,
            physics_plausibility,
            scale_drift,
            trajectory_smoothness,
            vo_verdict,
        )
        poses = _smooth_descent(100, fps=5.0)
        deltas = _poses_to_deltas_simple(poses)
        conf = np.ones(100, dtype=np.float64)

        sm = trajectory_smoothness(poses, fps=5.0)
        ph = physics_plausibility(deltas, fps=5.0)
        ca = confidence_analysis(conf)
        sd = scale_drift(poses)
        verdict = vo_verdict(sm, ph, ca, sd)
        assert verdict.decision == "GO"

    def test_implausible_no_go(self) -> None:
        from vllatent.ingest.vo_validation import (
            confidence_analysis,
            physics_plausibility,
            scale_drift,
            trajectory_smoothness,
            vo_verdict,
        )
        poses = _physically_implausible(100, fps=5.0)
        deltas = _poses_to_deltas_simple(poses)
        conf = np.ones(100, dtype=np.float64)

        sm = trajectory_smoothness(poses, fps=5.0)
        ph = physics_plausibility(deltas, fps=5.0)
        ca = confidence_analysis(conf)
        sd = scale_drift(poses)
        verdict = vo_verdict(sm, ph, ca, sd)
        assert verdict.decision in ("CONDITIONAL-GO", "NO-GO")


# ---------------------------------------------------------------------------
# Tests — validate_clip (integration)
# ---------------------------------------------------------------------------

class TestValidateClip:
    def test_returns_full_report(self) -> None:
        from vllatent.ingest.vo_validation import validate_clip
        poses = _smooth_descent(100, fps=5.0)
        conf = np.ones(100, dtype=np.float64)
        report = validate_clip(poses, conf, fps=5.0)
        assert report.verdict.decision == "GO"
        assert report.smoothness is not None
        assert report.physics is not None
        assert report.confidence is not None
        assert report.drift is not None


# ---------------------------------------------------------------------------
# Import purity
# ---------------------------------------------------------------------------

class TestImportPurity:
    def test_no_heavy_imports(self) -> None:
        src = Path(__file__).resolve().parent.parent / "vllatent" / "ingest" / "vo_validation.py"
        tree = ast.parse(src.read_text())
        heavy = {"torch", "transformers", "timm", "ultralytics", "cv2", "PIL", "plotly"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in heavy, f"top-level import of {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in heavy, f"top-level from-import of {node.module}"

    def test_module_importable_without_torch(self) -> None:
        saved = {}
        for mod in ("torch", "transformers", "timm", "ultralytics"):
            if mod in sys.modules:
                saved[mod] = sys.modules.pop(mod)
        try:
            import vllatent.ingest.vo_validation  # noqa: F401
        finally:
            sys.modules.update(saved)
