"""Tests for B3.4 Stage-0 gate decisions."""
from __future__ import annotations

from vllatent.train.person_probes import (
    K1CausalityMetrics,
    K2PredictorMetrics,
    Stage0GateThresholds,
    Stage0ProbeMetrics,
    evaluate_stage0_gates,
)


def _stage0(**kwargs: float) -> Stage0ProbeMetrics:
    values = {
        "n_train": 100,
        "n_val": 20,
        "n_val_visible": 12,
        "presence_auroc": 0.99,
        "center_l2_error": 0.05,
        "center_l1_error": 0.03,
        "log_height_mae": 0.04,
    }
    values.update(kwargs)
    return Stage0ProbeMetrics(**values)  # type: ignore[arg-type]


def _k1(**kwargs: float) -> K1CausalityMetrics:
    values = {
        "n_train": 100,
        "n_val": 20,
        "n_target_values": 300,
        "zero_mse": 0.10,
        "plan_only_mse": 0.098,
        "plan_only_r2": 0.02,
    }
    values.update(kwargs)
    return K1CausalityMetrics(**values)  # type: ignore[arg-type]


def _k2(**kwargs: float) -> K2PredictorMetrics:
    values = {
        "n_train": 100,
        "n_val": 20,
        "n_target_values": 400,
        "persistence_mse": 0.10,
        "conditioned_mse": 0.08,
        "improvement_frac": 0.20,
    }
    values.update(kwargs)
    return K2PredictorMetrics(**values)  # type: ignore[arg-type]


def test_stage0_gate_decision_passes_all_thresholds() -> None:
    decision = evaluate_stage0_gates(_stage0(), _k1(), _k2(), Stage0GateThresholds())
    assert decision.passed
    assert decision.failures == ()


def test_stage0_gate_decision_reports_failed_gates() -> None:
    decision = evaluate_stage0_gates(
        _stage0(presence_auroc=0.80, center_l2_error=0.20),
        _k1(plan_only_r2=0.30),
        _k2(improvement_frac=0.01),
        Stage0GateThresholds(),
    )
    assert not decision.passed
    assert decision.failures == ("G0", "K1", "K2")
