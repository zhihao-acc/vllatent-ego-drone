"""Tests for B3 human world-model losses."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vllatent.model.human_world_model import TransitionPlanVerifier  # noqa: E402
from vllatent.plan_tokens import PLAN_TOKEN_DIM  # noqa: E402
from vllatent.schemas import PATCH_TOKENS  # noqa: E402
from vllatent.train.world_model_losses import (  # noqa: E402
    WorldModelLossOutput,
    human_world_model_loss,
    person_patch_weights,
    person_state_loss,
    person_weighted_latent_loss,
    physical_inverse_plan_loss,
)


def _state(batch_size: int = 2, horizon: int = 8):
    state = torch.zeros(batch_size, horizon, 4)
    state[..., 0] = 0.5
    state[..., 1] = 0.5
    state[..., 2] = torch.log(torch.full((batch_size, horizon), 0.25))
    state[..., 3] = 1.0
    valid = torch.ones(batch_size, horizon, dtype=torch.bool)
    conf = torch.full((batch_size, horizon), 0.75)
    return state, valid, conf


@pytest.mark.torch
class TestPersonPatchWeights:
    def test_weights_shape_and_bounds(self) -> None:
        state, valid, conf = _state()
        weights = person_patch_weights(state, valid, conf)
        assert weights.shape == (2, 8, PATCH_TOKENS)
        assert torch.all(weights >= 0.0)
        assert torch.all(weights <= 1.0)

    def test_invalid_rows_are_zero(self) -> None:
        state, valid, conf = _state(batch_size=1)
        valid[:, 3:] = False
        weights = person_patch_weights(state, valid, conf)
        assert torch.any(weights[:, :3] > 0.0)
        assert torch.allclose(weights[:, 3:], torch.zeros_like(weights[:, 3:]))

    def test_center_weights_exceed_corner_weights(self) -> None:
        state, valid, conf = _state(batch_size=1)
        weights = person_patch_weights(state, valid, conf)
        center_idx = 7 * 14 + 7
        corner_idx = 0
        assert torch.all(weights[..., center_idx] > weights[..., corner_idx])


@pytest.mark.torch
class TestWorldModelLosses:
    def test_person_weighted_latent_loss_uses_background_when_no_valid_person(self) -> None:
        pred = torch.randn(2, 8, PATCH_TOKENS, 16, requires_grad=True)
        target = torch.randn(2, 8, PATCH_TOKENS, 16)
        state, valid, conf = _state()
        valid[:] = False
        loss = person_weighted_latent_loss(pred, target, state, valid, conf)
        assert loss.shape == ()
        assert loss.item() > 0.0
        loss.backward()
        assert pred.grad is not None

    def test_person_weight_changes_latent_loss(self) -> None:
        pred = torch.zeros(1, 8, PATCH_TOKENS, 4)
        target = torch.zeros_like(pred)
        state, valid, conf = _state(batch_size=1)
        center_idx = 7 * 14 + 7
        target[:, :, center_idx] = 10.0
        base = person_weighted_latent_loss(pred, target, state, valid, conf, person_weight=0.0)
        weighted = person_weighted_latent_loss(pred, target, state, valid, conf, person_weight=4.0)
        assert weighted > base

    def test_person_state_loss_masks_center_but_trains_visibility(self) -> None:
        target, valid, conf = _state(batch_size=1)
        pred = target.clone().requires_grad_(True)
        valid[:] = False
        pred.data[..., 3] = -2.0
        loss = person_state_loss(pred, target, valid, conf)
        assert loss.item() > 0.0
        loss.backward()
        assert pred.grad is not None

    def test_person_state_loss_uses_visible_target_when_state_invalid(self) -> None:
        target, valid, conf = _state(batch_size=1)
        valid[:] = False
        low_vis = target.clone()
        high_vis = target.clone()
        low_vis[..., 3] = -2.0
        high_vis[..., 3] = 2.0
        assert person_state_loss(high_vis, target, valid, conf) < person_state_loss(low_vis, target, valid, conf)

        shifted = high_vis.clone()
        shifted[..., :3] = 100.0
        assert person_state_loss(shifted, target, valid, conf).item() == pytest.approx(
            person_state_loss(high_vis, target, valid, conf).item(),
            abs=1e-7,
        )

    def test_physical_inverse_loss_excludes_valid_field_and_reports_components(self) -> None:
        predicted = torch.zeros(1, 2, PLAN_TOKEN_DIM - 1, requires_grad=True)
        target_valid_zero = torch.zeros(1, 2, PLAN_TOKEN_DIM)
        target_valid_one = target_valid_zero.clone()
        target_valid_one[..., -1] = 1.0
        mask = torch.ones(1, 2, dtype=torch.bool)

        valid_zero = physical_inverse_plan_loss(
            predicted,
            target_valid_zero,
            mask,
        )
        valid_one = physical_inverse_plan_loss(
            predicted,
            target_valid_one,
            mask,
        )

        assert valid_zero.total.item() == pytest.approx(0.0, abs=1e-7)
        assert torch.allclose(valid_zero.total, valid_one.total, atol=1e-7)
        assert valid_one.per_field.shape == (PLAN_TOKEN_DIM - 1,)
        assert torch.allclose(valid_one.per_field, torch.zeros_like(valid_one.per_field))

        physical_target = target_valid_one.clone()
        physical_target[..., : PLAN_TOKEN_DIM - 1] = torch.tensor(
            [0.25, 0.5, 0.75, 1.0, 1.25]
        )
        physical = physical_inverse_plan_loss(predicted, physical_target, mask)
        assert torch.all(physical.per_field > 0.0)
        assert physical.total.item() == pytest.approx(physical.per_field.mean().item())

    def test_frozen_verifier_cycle_routes_gradients_and_masks_dropped_rows(self) -> None:
        torch.manual_seed(0)
        verifier = TransitionPlanVerifier(dim=8, hidden_dim=16, dropout=0.0)
        verifier.eval()
        verifier.requires_grad_(False)
        previous = torch.randn(2, 3, PATCH_TOKENS, 8)
        predicted_next = torch.randn(
            2,
            3,
            PATCH_TOKENS,
            8,
            requires_grad=True,
        )
        target_plan = torch.randn(2, 3, PLAN_TOKEN_DIM)
        target_plan[..., -1] = 1.0
        valid_mask = torch.ones(2, 3, dtype=torch.bool)
        whole_plan_keep_mask = torch.tensor(
            [
                [True, True, True],
                [False, False, False],
            ]
        )

        recovered = verifier(previous.detach(), predicted_next)
        loss = physical_inverse_plan_loss(
            recovered,
            target_plan,
            valid_mask,
            plan_keep_mask=whole_plan_keep_mask,
        )
        loss.total.backward()

        assert predicted_next.grad is not None
        assert predicted_next.grad[0].abs().sum() > 0
        assert torch.count_nonzero(predicted_next.grad[1]) == 0
        assert all(parameter.grad is None for parameter in verifier.parameters())

    def test_combined_loss_returns_components(self) -> None:
        pred_lat = torch.randn(2, 8, PATCH_TOKENS, 16, requires_grad=True)
        target_lat = torch.randn(2, 8, PATCH_TOKENS, 16)
        pred_state = torch.randn(2, 8, 4, requires_grad=True)
        state, valid, conf = _state()
        pred_plan = torch.randn(2, 8, PLAN_TOKEN_DIM - 1, requires_grad=True)
        plan = torch.randn(2, 8, PLAN_TOKEN_DIM)
        plan_valid = torch.ones(2, 8, dtype=torch.bool)

        out = human_world_model_loss(
            predicted_latents=pred_lat,
            target_latents=target_lat,
            predicted_person_state=pred_state,
            person_state_target=state,
            person_state_valid=valid,
            predicted_plan=pred_plan,
            planned_actions=plan,
            planned_actions_valid_mask=plan_valid,
            person_conf=conf,
        )

        assert isinstance(out, WorldModelLossOutput)
        assert out.total.shape == ()
        assert out.latent.shape == ()
        assert out.person_state.shape == ()
        assert out.inverse_plan.shape == ()
        assert out.inverse_plan_per_field.shape == (PLAN_TOKEN_DIM - 1,)
        assert -1.0 <= out.latent_cosine.item() <= 1.0
