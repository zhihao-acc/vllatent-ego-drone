"""Tests for B3 world-model metrics."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vllatent.plan_tokens import PLAN_TOKEN_DIM  # noqa: E402
from vllatent.schemas import PATCH_TOKENS  # noqa: E402
from vllatent.train import world_model_metrics as metrics_module  # noqa: E402
from vllatent.train.world_model_metrics import (  # noqa: E402
    aggregate_stage1_metrics,
    flipped_plan,
    null_plan,
    per_window_latent_loss,
    persistence_rollout,
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


def _cluster_ids(n_windows: int) -> list[str]:
    return [f"source-{index}" for index in range(n_windows)]


def _constant_batch_metrics(
    batch_size: int,
    *,
    model_value: float,
    persistence_value: float,
    null_value: float,
    shuffled_value: float,
    flipped_value: float,
    horizon: int = 2,
):
    target = torch.zeros(batch_size, horizon, PATCH_TOKENS, 1)
    state, valid, conf = _state(batch_size=batch_size, horizon=horizon)
    return summarize_stage1_batch(
        predicted_latents=torch.full_like(target, model_value),
        persistence_latents=torch.full_like(target, persistence_value),
        null_plan_latents=torch.full_like(target, null_value),
        shuffled_plan_latents=torch.full_like(target, shuffled_value),
        flipped_plan_latents=torch.full_like(target, flipped_value),
        target_latents=target,
        person_state_target=state,
        person_state_valid=valid,
        person_conf=conf,
    )


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

    assert not hasattr(metrics_module, "shuffled_plan")


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
def test_multinegative_shuffled_win_requires_a_per_window_majority() -> None:
    target = torch.zeros(1, 2, PATCH_TOKENS, 1)
    state, valid, conf = _state(batch_size=1, horizon=2)
    predicted = torch.full_like(target, 0.5)
    shuffled = torch.stack(
        [
            torch.full_like(target, 0.1),
            torch.full_like(target, 0.2),
            torch.full_like(target, 2.0),
        ],
        dim=0,
    )

    metrics = summarize_stage1_batch(
        predicted_latents=predicted,
        persistence_latents=torch.ones_like(target),
        null_plan_latents=torch.ones_like(target),
        shuffled_plan_latents=shuffled,
        flipped_plan_latents=torch.ones_like(target),
        target_latents=target,
        person_state_target=state,
        person_state_valid=valid,
        person_conf=conf,
    )

    assert metrics.true_beats_shuffled_rate == 0.0


@pytest.mark.torch
def test_aggregate_stage1_metrics_sets_gate_flags() -> None:
    target = torch.zeros(8, 3, PATCH_TOKENS, 4)
    state, valid, conf = _state(batch_size=8)
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

    aggregate = aggregate_stage1_metrics(
        [batch_metrics],
        cluster_ids=_cluster_ids(8),
        yaw_geometry_correct_by_window=[True] * 8,
        yaw_geometry_eligible_by_window=[True] * 8,
    )
    assert aggregate["g1a_pass"] is True
    assert aggregate["g1b_pass"] is True
    assert aggregate["g1d_pass"] is True


@pytest.mark.torch
def test_aggregate_uses_ratio_of_global_losses_with_unequal_batch_sizes() -> None:
    one_window = _constant_batch_metrics(
        1,
        model_value=0.0,
        persistence_value=1.0,
        null_value=1.0,
        shuffled_value=1.0,
        flipped_value=1.0,
    )
    three_windows = _constant_batch_metrics(
        3,
        model_value=0.9,
        persistence_value=1.0,
        null_value=1.0,
        shuffled_value=1.0,
        flipped_value=1.0,
    )

    aggregate = aggregate_stage1_metrics(
        [one_window, three_windows],
        cluster_ids=_cluster_ids(4),
        yaw_geometry_correct_by_window=[True] * 4,
        yaw_geometry_eligible_by_window=[True] * 4,
    )
    expected_model = (one_window.model_loss + 3.0 * three_windows.model_loss) / 4.0
    expected_persistence = (one_window.persistence_loss + 3.0 * three_windows.persistence_loss) / 4.0
    expected_improvement = (expected_persistence - expected_model) / expected_persistence

    assert aggregate["n_windows"] == 4
    assert aggregate["model_loss"] == pytest.approx(expected_model)
    assert aggregate["persistence_loss"] == pytest.approx(expected_persistence)
    assert aggregate["improvement_vs_persistence"] == pytest.approx(expected_improvement)


@pytest.mark.torch
def test_g1d_requires_five_percent_shuffled_and_flipped_margins() -> None:
    metrics = _constant_batch_metrics(
        4,
        model_value=0.90,
        persistence_value=1.20,
        null_value=1.20,
        shuffled_value=0.92,
        flipped_value=0.93,
    )
    aggregate = aggregate_stage1_metrics(
        [metrics],
        cluster_ids=_cluster_ids(4),
        yaw_geometry_correct_by_window=[True] * 4,
        yaw_geometry_eligible_by_window=[True] * 4,
    )

    assert aggregate["true_beats_shuffled_rate"] == pytest.approx(1.0)
    assert aggregate["true_beats_flipped_rate"] == pytest.approx(1.0)
    assert aggregate["improvement_vs_shuffled"] < 0.05
    assert aggregate["improvement_vs_flipped"] < 0.05
    assert aggregate["g1d_pass"] is False


@pytest.mark.torch
def test_g1d_requires_strict_yaw_geometry_majority() -> None:
    metrics = _constant_batch_metrics(
        8,
        model_value=0.0,
        persistence_value=1.0,
        null_value=1.0,
        shuffled_value=1.0,
        flipped_value=1.0,
    )

    no_majority = aggregate_stage1_metrics(
        [metrics],
        cluster_ids=_cluster_ids(8),
        yaw_geometry_correct_by_window=[True] * 4 + [False] * 4,
        yaw_geometry_eligible_by_window=[True] * 8,
    )
    majority = aggregate_stage1_metrics(
        [metrics],
        cluster_ids=_cluster_ids(8),
        yaw_geometry_correct_by_window=[True] * 8,
        yaw_geometry_eligible_by_window=[True] * 8,
    )

    assert no_majority["g1d_pass"] is False
    assert majority["g1d_pass"] is True


@pytest.mark.torch
def test_g1d_confidence_guard_rejects_one_of_one_yaw_coverage() -> None:
    metrics = _constant_batch_metrics(
        4,
        model_value=0.0,
        persistence_value=1.0,
        null_value=1.0,
        shuffled_value=1.0,
        flipped_value=1.0,
    )

    aggregate = aggregate_stage1_metrics(
        [metrics],
        cluster_ids=_cluster_ids(4),
        yaw_geometry_correct_by_window=[True, False, False, False],
        yaw_geometry_eligible_by_window=[True, False, False, False],
    )

    assert aggregate["yaw_geometry_ci95_lower"] < 0.5
    assert aggregate["yaw_geometry_eligible_source_count"] == 1
    assert aggregate["g1d_pass"] is False


@pytest.mark.torch
def test_g1d_zero_yaw_coverage_fails_closed_without_crashing() -> None:
    metrics = _constant_batch_metrics(
        4,
        model_value=0.0,
        persistence_value=1.0,
        null_value=1.0,
        shuffled_value=1.0,
        flipped_value=1.0,
    )

    aggregate = aggregate_stage1_metrics(
        [metrics],
        cluster_ids=_cluster_ids(4),
        yaw_geometry_correct_by_window=[False] * 4,
        yaw_geometry_eligible_by_window=[False] * 4,
    )

    assert aggregate["yaw_geometry_rate"] == 0.0
    assert aggregate["g1d_pass"] is False


@pytest.mark.torch
def test_boundary_aware_source_interval_is_not_degenerate() -> None:
    lower, upper = metrics_module.wilson_interval(2, 2)

    assert lower < 0.5
    assert upper == pytest.approx(1.0)


@pytest.mark.torch
def test_g1d_rejects_many_overlapping_windows_from_one_source() -> None:
    metrics = _constant_batch_metrics(
        16,
        model_value=0.0,
        persistence_value=1.0,
        null_value=1.0,
        shuffled_value=1.0,
        flipped_value=1.0,
    )

    aggregate = aggregate_stage1_metrics(
        [metrics],
        cluster_ids=["same-source"] * 16,
        yaw_geometry_correct_by_window=[True] * 16,
        yaw_geometry_eligible_by_window=[True] * 16,
    )

    assert aggregate["true_beats_shuffled_rate"] == 1.0
    assert aggregate["true_beats_shuffled_ci95"][0] < 0.5
    assert aggregate["confidence_cluster_count"] == 1
    assert aggregate["g1d_pass"] is False


@pytest.mark.torch
def test_plan_derangements_are_deterministic_distinct_and_distance_bounded() -> None:
    plans = torch.zeros(6, 2, PLAN_TOKEN_DIM)
    plans[..., 0] = torch.linspace(-0.75, 0.75, 6).unsqueeze(1)
    plans[..., 5] = 1.0
    clip_ids = ["clip-a", "clip-a", "clip-b", "clip-b", "clip-c", "clip-c"]
    min_distance = 0.1

    first = metrics_module.build_plan_derangements(
        plans,
        clip_ids,
        num_permutations=3,
        seed=17,
        min_plan_distance=min_distance,
    )
    second = metrics_module.build_plan_derangements(
        plans,
        clip_ids,
        num_permutations=3,
        seed=17,
        min_plan_distance=min_distance,
    )

    assert len(first) == len(second) == 3
    assert len({tuple(permutation.tolist()) for permutation in first}) == 3
    identity = torch.arange(len(plans))
    for permutation, repeated in zip(first, second, strict=True):
        assert torch.equal(permutation, repeated)
        assert torch.equal(torch.sort(permutation).values, identity)
        assert torch.all(permutation != identity)
        assert all(clip_ids[row] != clip_ids[int(donor)] for row, donor in enumerate(permutation))
        distance = torch.sqrt(torch.mean((plans - plans[permutation]) ** 2, dim=(1, 2)))
        assert torch.all(distance >= min_distance)


@pytest.mark.torch
def test_plan_derangements_fail_closed_when_constraints_are_impossible() -> None:
    distinct = torch.zeros(3, 2, PLAN_TOKEN_DIM)
    distinct[..., 0] = torch.tensor([-0.5, 0.0, 0.5]).unsqueeze(1)
    distinct[..., 5] = 1.0

    with pytest.raises(ValueError, match="different.clip"):
        metrics_module.build_plan_derangements(
            distinct,
            ["same", "same", "same"],
            num_permutations=1,
            seed=0,
            min_plan_distance=0.0,
        )

    identical = torch.zeros(3, 2, PLAN_TOKEN_DIM)
    identical[..., 5] = 1.0
    with pytest.raises(ValueError, match="distance"):
        metrics_module.build_plan_derangements(
            identical,
            ["a", "b", "c"],
            num_permutations=1,
            seed=0,
            min_plan_distance=0.01,
        )


@pytest.mark.torch
def test_yaw_geometry_counts_expected_center_sign_and_ignores_small_yaw() -> None:
    true_state = torch.zeros(3, 2, 4)
    yaw_flipped_state = torch.zeros_like(true_state)
    plan = torch.zeros(3, 2, PLAN_TOKEN_DIM)
    dt = torch.ones(3, 2)
    valid = torch.ones(3, 2, dtype=torch.bool)

    # Positive NED yaw turns the camera right, so the same subject moves left.
    plan[0, :, 4] = 0.5
    true_state[0, -1, 0] = 0.4
    yaw_flipped_state[0, -1, 0] = 0.6
    # This row deliberately has the wrong response sign.
    plan[1, :, 4] = -0.5
    true_state[1, -1, 0] = 0.4
    yaw_flipped_state[1, -1, 0] = 0.6
    # Near-zero cumulative yaw is not eligible.
    plan[2, :, 4] = 1e-4

    counts = metrics_module.yaw_geometry_counts(
        true_state,
        yaw_flipped_state,
        plan,
        dt,
        valid,
        min_cumulative_yaw=0.05,
    )

    assert counts.correct == 1
    assert counts.eligible == 2
    assert counts.rate == pytest.approx(0.5)
