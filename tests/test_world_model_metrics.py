"""Tests for B3 world-model metrics."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vllatent.plan_tokens import PLAN_TOKEN_DIM  # noqa: E402
from vllatent.schemas import PATCH_TOKENS  # noqa: E402
from vllatent.train.world_model_metrics import (  # noqa: E402
    aggregate_stage1_metrics,
    flipped_plan,
    null_plan,
    per_window_latent_loss,
    persistence_rollout,
    shuffled_plan,
    summarize_stage1_batch,
)


def _state(batch_size: int = 2, horizon: int = 3):
    state = torch.zeros(batch_size, horizon, 4)
    state[..., 0] = 0.5
    state[..., 1] = 0.5
    state[..., 2] = torch.log(torch.full((batch_size, horizon), 0.25))
    state[..., 3] = 1.0
    valid = torch.ones(batch_size, horizon, dtype=torch.bool)
    conf = torch.ones(batch_size, horizon)
    return state, valid, conf


@pytest.mark.torch
def test_persistence_rollout_shape() -> None:
    z_t = torch.randn(2, PATCH_TOKENS, 4)
    out = persistence_rollout(z_t, horizon=5)
    assert out.shape == (2, 5, PATCH_TOKENS, 4)
    assert torch.allclose(out[:, 0], z_t)
    assert torch.allclose(out[:, -1], z_t)


@pytest.mark.torch
def test_plan_ablations_preserve_shape() -> None:
    plan = torch.randn(2, 4, PLAN_TOKEN_DIM)
    plan[..., 5] = 1.0
    assert null_plan(plan).shape == plan.shape
    assert torch.allclose(null_plan(plan), torch.zeros_like(plan))

    flipped = flipped_plan(plan)
    assert flipped.shape == plan.shape
    assert torch.allclose(flipped[..., :3], -plan[..., :3])
    assert torch.allclose(flipped[..., 3], plan[..., 3])
    assert torch.allclose(flipped[..., 4], -plan[..., 4])
    assert torch.allclose(flipped[..., 5], plan[..., 5])

    shuffled = shuffled_plan(plan)
    assert shuffled.shape == plan.shape


@pytest.mark.torch
def test_per_window_latent_loss_shape_and_zero() -> None:
    pred = torch.zeros(2, 3, PATCH_TOKENS, 4)
    target = torch.zeros_like(pred)
    state, valid, conf = _state()
    loss = per_window_latent_loss(pred, target, state, valid, conf)
    assert loss.shape == (2, 3)
    assert torch.allclose(loss, torch.zeros_like(loss))


@pytest.mark.torch
def test_summarize_stage1_batch_prefers_better_prediction() -> None:
    target = torch.zeros(2, 3, PATCH_TOKENS, 4)
    pred = torch.zeros_like(target)
    persistence = torch.ones_like(target)
    null = torch.ones_like(target) * 0.8
    shuffled = torch.ones_like(target) * 0.5
    flipped = torch.ones_like(target) * 0.4
    state, valid, conf = _state()

    metrics = summarize_stage1_batch(
        predicted_latents=pred,
        persistence_latents=persistence,
        null_plan_latents=null,
        shuffled_plan_latents=shuffled,
        flipped_plan_latents=flipped,
        target_latents=target,
        person_state_target=state,
        person_state_valid=valid,
        person_conf=conf,
    )

    assert metrics.improvement_vs_persistence > 0.0
    assert metrics.improvement_vs_null > 0.0
    assert all(metrics.rollout_beats_persistence)
    assert metrics.true_beats_shuffled_rate == pytest.approx(1.0)
    assert metrics.true_beats_flipped_rate == pytest.approx(1.0)


@pytest.mark.torch
def test_aggregate_stage1_metrics_sets_gate_flags() -> None:
    target = torch.zeros(2, 3, PATCH_TOKENS, 4)
    state, valid, conf = _state()
    batch_metrics = summarize_stage1_batch(
        predicted_latents=torch.zeros_like(target),
        persistence_latents=torch.ones_like(target),
        null_plan_latents=torch.ones_like(target),
        shuffled_plan_latents=torch.ones_like(target),
        flipped_plan_latents=torch.ones_like(target),
        target_latents=target,
        person_state_target=state,
        person_state_valid=valid,
        person_conf=conf,
    )

    aggregate = aggregate_stage1_metrics([batch_metrics, batch_metrics])
    assert aggregate["g1a_pass"] is True
    assert aggregate["g1b_pass"] is True
    assert aggregate["g1d_pass"] is True
