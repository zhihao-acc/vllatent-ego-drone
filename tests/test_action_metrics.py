"""Tests for B2 action-policy metrics and deterministic baselines."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vllatent.scale_free_targets import SCALE_FREE_ACTION_DIM  # noqa: E402
from vllatent.schemas import HORIZON  # noqa: E402
from vllatent.train.action_metrics import (  # noqa: E402
    baseline_action_predictions,
    compute_action_metrics,
    repeat_last_baseline,
    score_action_predictions,
)


def _target(batch_size: int = 2) -> torch.Tensor:
    target = torch.zeros(batch_size, HORIZON, SCALE_FREE_ACTION_DIM)
    target[..., 0] = 1.0
    target[..., 3] = 0.0
    return target


def _mask(batch_size: int = 2) -> torch.Tensor:
    return torch.ones(batch_size, HORIZON, dtype=torch.bool)


@pytest.mark.torch
class TestActionMetrics:
    def test_perfect_prediction_scores_near_zero(self) -> None:
        target = _target()
        metrics = compute_action_metrics(target.clone(), target, _mask())
        assert metrics.direction_cosine == pytest.approx(1.0, abs=1e-6)
        assert metrics.angular_error_deg == pytest.approx(0.0, abs=1e-5)
        assert metrics.speed_ratio_mae == pytest.approx(0.0, abs=1e-6)
        assert metrics.path_ade == pytest.approx(0.0, abs=1e-6)
        assert metrics.path_fde == pytest.approx(0.0, abs=1e-6)
        assert metrics.aggregate_score == pytest.approx(0.0, abs=1e-6)
        assert metrics.n_samples == 2
        assert metrics.n_valid == 2 * HORIZON

    def test_bad_prediction_is_worse(self) -> None:
        target = _target()
        bad = target.clone()
        bad[..., 0] = -1.0
        bad[..., 3] = 1.0
        good_metrics = compute_action_metrics(target, target, _mask())
        bad_metrics = compute_action_metrics(bad, target, _mask())
        assert bad_metrics.aggregate_score > good_metrics.aggregate_score
        assert bad_metrics.angular_error_deg > 100.0
        assert bad_metrics.speed_ratio_mae > 0.9

    def test_repeat_last_baseline_is_deterministic(self) -> None:
        last = torch.randn(3, SCALE_FREE_ACTION_DIM)
        b1 = repeat_last_baseline(last, horizon=HORIZON)
        b2 = repeat_last_baseline(last, horizon=HORIZON)
        assert torch.equal(b1, b2)
        assert b1.shape == (3, HORIZON, SCALE_FREE_ACTION_DIM)
        for k in range(HORIZON):
            assert torch.equal(b1[:, k], last)

    def test_all_baseline_keys_present(self) -> None:
        last = torch.zeros(2, SCALE_FREE_ACTION_DIM)
        last[:, 0] = 1.0
        baselines = baseline_action_predictions(last, horizon=HORIZON)
        assert set(baselines) == {"repeat_last", "no_turn", "zero", "mean", "linear"}
        for pred in baselines.values():
            assert pred.shape == (2, HORIZON, SCALE_FREE_ACTION_DIM)

    def test_margin_positive_only_when_model_beats_best_baseline(self) -> None:
        target = _target(batch_size=1)
        target[..., 3] = 0.5
        last = torch.zeros(1, SCALE_FREE_ACTION_DIM)
        last[:, 1] = 1.0
        perfect = target.clone()
        bad = repeat_last_baseline(last, horizon=HORIZON)

        good_card = score_action_predictions(perfect, target, _mask(batch_size=1), last)
        bad_card = score_action_predictions(bad, target, _mask(batch_size=1), last)

        assert good_card.margin > 0.0
        assert bad_card.margin <= 0.0
        assert good_card.best_baseline in good_card.baselines

    def test_sample_weights_affect_aggregate(self) -> None:
        target = _target(batch_size=2)
        pred = target.clone()
        pred[1, :, 0] = -1.0
        equal = compute_action_metrics(pred, target, _mask(batch_size=2))
        weighted = compute_action_metrics(
            pred,
            target,
            _mask(batch_size=2),
            sample_weight=torch.tensor([10.0, 1.0]),
        )
        assert weighted.aggregate_score < equal.aggregate_score

    def test_masked_steps_are_ignored(self) -> None:
        target = _target(batch_size=1)
        pred = target.clone()
        pred[:, 1:, 0] = -1.0
        mask = torch.tensor([[True, False, False, False]])
        metrics = compute_action_metrics(pred, target, mask)
        assert metrics.aggregate_score == pytest.approx(0.0, abs=1e-6)
        assert metrics.n_valid == 1
        assert metrics.n_speed_valid == 1

    def test_speed_mask_ignores_invalid_speed_mae_only(self) -> None:
        target = _target(batch_size=1)
        pred = target.clone()
        pred[..., 3] = 8.0
        mask = _mask(batch_size=1)
        speed_mask = torch.zeros(1, HORIZON, dtype=torch.bool)

        metrics = compute_action_metrics(pred, target, mask, speed_mask=speed_mask)

        assert metrics.speed_ratio_mae == pytest.approx(0.0, abs=1e-6)
        assert metrics.n_valid == HORIZON
        assert metrics.n_speed_valid == 0
