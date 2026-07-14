"""Tests for the B3 human-conditioned latent world model."""
from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")

from vllatent.model.human_world_model import (  # noqa: E402
    HumanWorldModel,
    HumanWorldModelOutput,
    PlanConditionedLatentPredictor,
    TransitionPlanVerifier,
    apply_whole_plan_dropout,
    count_parameters,
)
from vllatent.plan_tokens import PLAN_TOKEN_DIM  # noqa: E402
from vllatent.schemas import HISTORY, PATCH_TOKENS  # noqa: E402


def _inputs(batch_size: int = 2, dim: int = 32, horizon: int = 8):
    history = torch.randn(batch_size, HISTORY, PATCH_TOKENS, dim)
    z_t = torch.randn(batch_size, PATCH_TOKENS, dim)
    history_mask = torch.ones(batch_size, HISTORY, dtype=torch.bool)
    planned_actions = torch.randn(batch_size, horizon, PLAN_TOKEN_DIM)
    planned_actions[..., 5] = 1.0
    dt = torch.full((batch_size, horizon), 0.2)
    return history, z_t, history_mask, planned_actions, dt


def _enable_condition_film(model: PlanConditionedLatentPredictor | HumanWorldModel) -> None:
    predictor = model.predictor if isinstance(model, HumanWorldModel) else model
    for film in [*predictor.plan_film, *predictor.dt_film]:
        final = film.net[-1]
        final.weight.data.normal_(std=0.05)
        final.bias.data.normal_(std=0.05)


@pytest.mark.torch
class TestTransitionPlanVerifier:
    def test_is_action_blind_and_validates_transition_inputs(self) -> None:
        verifier = TransitionPlanVerifier(dim=32, hidden_dim=64, dropout=0.0)
        previous = torch.randn(2, 8, PATCH_TOKENS, 32)
        next_latents = torch.randn_like(previous)

        recovered = verifier(previous, next_latents)

        assert recovered.shape == (2, 8, PLAN_TOKEN_DIM - 1)
        signature = inspect.signature(TransitionPlanVerifier.forward)
        assert tuple(signature.parameters) == (
            "self",
            "previous_latents",
            "next_latents",
        )
        with pytest.raises(ValueError, match="previous_latents"):
            verifier(previous[:, 0], next_latents[:, 0])
        with pytest.raises(ValueError, match="previous/next"):
            verifier(previous, next_latents[:, :-1])
        with pytest.raises(ValueError, match="last dimension"):
            verifier(previous[..., :-1], next_latents[..., :-1])

    def test_whole_plan_dropout_returns_all_step_masks_and_null_rows(self) -> None:
        plans = torch.ones(16, 8, PLAN_TOKEN_DIM)
        generator = torch.Generator().manual_seed(0)

        conditioned, keep_mask = apply_whole_plan_dropout(
            plans,
            dropout_p=0.5,
            training=True,
            generator=generator,
        )

        assert keep_mask.shape == plans.shape[:2]
        assert keep_mask.dtype == torch.bool
        assert torch.all(keep_mask == keep_mask[:, :1])
        assert torch.any(keep_mask[:, 0])
        assert torch.any(~keep_mask[:, 0])
        assert torch.equal(conditioned[keep_mask], plans[keep_mask])
        assert torch.count_nonzero(conditioned[~keep_mask]) == 0
        assert torch.count_nonzero(conditioned[~keep_mask[:, 0]]) == 0

        eval_conditioned, eval_keep_mask = apply_whole_plan_dropout(
            plans,
            dropout_p=0.5,
            training=False,
            generator=generator,
        )
        assert torch.equal(eval_conditioned, plans)
        assert torch.all(eval_keep_mask)


@pytest.mark.torch
class TestPlanConditionedLatentPredictor:
    def test_horizon8_output_shape(self) -> None:
        model = PlanConditionedLatentPredictor(dim=32, depth=1, heads=4, horizon=8)
        out = model(*_inputs(dim=32, horizon=8))
        assert out.shape == (2, 8, PATCH_TOKENS, 32)

    def test_rejects_bad_plan_shape(self) -> None:
        model = PlanConditionedLatentPredictor(dim=32, depth=1, heads=4, horizon=8)
        history, z_t, mask, _, dt = _inputs(dim=32, horizon=8)
        with pytest.raises(ValueError, match="planned_actions"):
            model(history, z_t, mask, torch.zeros(2, 8, PLAN_TOKEN_DIM - 1), dt)

    def test_plan_causality_blocks_future_plan_leakage(self) -> None:
        model = PlanConditionedLatentPredictor(dim=32, depth=2, heads=4, horizon=8, dropout=0.0)
        _enable_condition_film(model)
        model.eval()
        history, z_t, mask, plan, dt = _inputs(batch_size=1, dim=32, horizon=8)
        future_changed = plan.clone()
        future_changed[:, 4:] += 100.0
        with torch.no_grad():
            out1 = model(history, z_t, mask, plan, dt)
            out2 = model(history, z_t, mask, future_changed, dt)
        assert torch.allclose(out1[:, :4], out2[:, :4], atol=1e-5)

    def test_plan_sensitivity_for_current_and_future_steps(self) -> None:
        model = PlanConditionedLatentPredictor(dim=32, depth=2, heads=4, horizon=8, dropout=0.0)
        _enable_condition_film(model)
        model.eval()
        history, z_t, mask, plan, dt = _inputs(batch_size=1, dim=32, horizon=8)
        changed = plan.clone()
        changed[:, 0, 0] += 5.0
        changed[:, 6, 1] -= 5.0
        with torch.no_grad():
            out1 = model(history, z_t, mask, plan, dt)
            out2 = model(history, z_t, mask, changed, dt)
        assert not torch.allclose(out1[:, 0], out2[:, 0], atol=1e-6)
        assert not torch.allclose(out1[:, 6], out2[:, 6], atol=1e-6)

    def test_action_dropout_only_applies_in_training(self) -> None:
        plan = torch.ones(2, 8, PLAN_TOKEN_DIM)
        eval_plan, eval_keep = apply_whole_plan_dropout(
            plan,
            dropout_p=1.0,
            training=False,
        )
        train_plan, train_keep = apply_whole_plan_dropout(
            plan,
            dropout_p=1.0,
            training=True,
        )
        assert torch.equal(eval_plan, plan)
        assert torch.all(eval_keep)
        assert torch.count_nonzero(train_plan) == 0
        assert not torch.any(train_keep)

    def test_predictor_does_not_hide_dropout_without_a_keep_mask(self) -> None:
        signature = inspect.signature(PlanConditionedLatentPredictor.__init__)
        assert "action_dropout_p" not in signature.parameters

    def test_residual_delta_is_patch_local(self) -> None:
        model = PlanConditionedLatentPredictor(dim=32, depth=1, heads=4, horizon=8, dropout=0.0)
        model.eval()
        history, z_t, mask, plan, dt = _inputs(batch_size=1, dim=32, horizon=8)
        with torch.no_grad():
            out = model(history, z_t, mask, plan, dt)
        delta = out - z_t.unsqueeze(1)
        assert delta.std(dim=2).mean().item() > 1e-7

    def test_zero_inputs_still_have_patch_local_residual_delta(self) -> None:
        model = PlanConditionedLatentPredictor(dim=32, depth=1, heads=4, horizon=8, dropout=0.0)
        model.eval()
        history, z_t, mask, plan, dt = _inputs(batch_size=1, dim=32, horizon=8)
        history.zero_()
        z_t.zero_()
        plan.zero_()
        with torch.no_grad():
            out = model(history, z_t, mask, plan, dt)
        assert out.std(dim=2).mean().item() > 1e-7


@pytest.mark.torch
class TestHumanWorldModel:
    def test_forward_shapes(self) -> None:
        model = HumanWorldModel(dim=32, depth=1, heads=4, horizon=8, hidden_dim=64)
        out = model(*_inputs(dim=32, horizon=8))
        assert isinstance(out, HumanWorldModelOutput)
        assert out.predicted_latents.shape == (2, 8, PATCH_TOKENS, 32)
        assert out.predicted_person_state.shape == (2, 8, 4)
        assert not hasattr(out, "predicted_plan")

        previous = torch.cat(
            [_inputs(dim=32, horizon=8)[1].unsqueeze(1), out.predicted_latents[:, :-1]],
            dim=1,
        )
        recovered = model.recover_plan(previous, out.predicted_latents)
        assert recovered.shape == (2, 8, PLAN_TOKEN_DIM - 1)

    def test_forward_signature_has_no_future_targets(self) -> None:
        sig = inspect.signature(HumanWorldModel.forward)
        forbidden = {
            "target_latents",
            "future_latents",
            "person_state_target",
            "target_person_state_valid",
            "person_conf",
            "labels",
        }
        assert forbidden.isdisjoint(sig.parameters)
        assert "planned_actions" in sig.parameters
        assert "dt_seconds" in sig.parameters

    def test_depth6_parameter_count_is_exact_and_in_expected_range(self) -> None:
        model = HumanWorldModel(dim=768, depth=6, heads=12, horizon=8)
        n_params = count_parameters(model)
        assert 50_000_000 < n_params < 70_000_000
        assert n_params == 59_280_137
