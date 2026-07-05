"""Tests for loss functions (B1.18)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vllatent.schemas import DOF, EMBED_DIM, HORIZON, PATCH_TOKENS  # noqa: E402
from vllatent.train.losses import (  # noqa: E402
    LossOutput,
    action_policy_loss,
    combined_loss,
    latent_loss,
    waypoint_loss,
)

B = 4
T = HORIZON
P = PATCH_TOKENS
D = EMBED_DIM


@pytest.mark.torch
class TestLatentLoss:
    def test_scalar_output(self) -> None:
        pred = torch.randn(B, T, P, D)
        tgt = torch.randn(B, T, P, D)
        w = torch.ones(B)
        loss = latent_loss(pred, tgt, w)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_beta_01_not_default(self) -> None:
        pred = torch.randn(B, T, P, D)
        tgt = pred + 0.05
        w = torch.ones(B)
        loss_01 = latent_loss(pred, tgt, w, beta=0.1)
        loss_10 = latent_loss(pred, tgt, w, beta=1.0)
        assert not torch.allclose(loss_01, loss_10)

    def test_quality_weighting(self) -> None:
        pred = torch.randn(B, T, P, D)
        tgt = torch.randn(B, T, P, D)
        w_high = torch.ones(B)
        w_low = torch.full((B,), 0.1)
        loss_high = latent_loss(pred, tgt, w_high)
        loss_low = latent_loss(pred, tgt, w_low)
        assert loss_high > loss_low

    def test_zero_loss_on_identical(self) -> None:
        x = torch.randn(B, T, P, D)
        w = torch.ones(B)
        loss = latent_loss(x, x, w)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_differentiable(self) -> None:
        pred = torch.randn(B, T, P, D, requires_grad=True)
        tgt = torch.randn(B, T, P, D)
        w = torch.ones(B)
        loss = latent_loss(pred, tgt, w)
        loss.backward()
        assert pred.grad is not None


@pytest.mark.torch
class TestWaypointLoss:
    def test_scalar_output(self) -> None:
        pred = torch.randn(B, T, DOF)
        tgt = torch.randn(B, T, DOF)
        w = torch.ones(B)
        loss = waypoint_loss(pred, tgt, w)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_confidence_weighting(self) -> None:
        pred = torch.randn(B, T, DOF)
        tgt = torch.randn(B, T, DOF)
        w_high = torch.ones(B)
        w_low = torch.full((B,), 0.05)
        loss_high = waypoint_loss(pred, tgt, w_high)
        loss_low = waypoint_loss(pred, tgt, w_low)
        assert loss_high > loss_low

    def test_zero_loss_on_identical(self) -> None:
        x = torch.randn(B, T, DOF)
        w = torch.ones(B)
        loss = waypoint_loss(x, x, w)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_differentiable(self) -> None:
        pred = torch.randn(B, T, DOF, requires_grad=True)
        tgt = torch.randn(B, T, DOF)
        w = torch.ones(B)
        loss = waypoint_loss(pred, tgt, w)
        loss.backward()
        assert pred.grad is not None


@pytest.mark.torch
class TestActionPolicyLoss:
    def test_zero_loss_on_identical(self) -> None:
        x = torch.zeros(B, T, DOF)
        x[..., 0] = 1.0
        mask = torch.ones(B, T, dtype=torch.bool)
        loss = action_policy_loss(x, x, mask)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_bad_prediction_positive_loss(self) -> None:
        target = torch.zeros(B, T, DOF)
        target[..., 0] = 1.0
        pred = target.clone()
        pred[..., 0] = -1.0
        pred[..., 3] = 1.0
        mask = torch.ones(B, T, dtype=torch.bool)
        loss = action_policy_loss(pred, target, mask)
        assert loss.item() > 1.0

    def test_differentiable(self) -> None:
        target = torch.zeros(B, T, DOF)
        target[..., 0] = 1.0
        pred = torch.randn(B, T, DOF, requires_grad=True)
        mask = torch.ones(B, T, dtype=torch.bool)
        loss = action_policy_loss(pred, target, mask)
        loss.backward()
        assert pred.grad is not None

    def test_masked_steps_ignored(self) -> None:
        target = torch.zeros(1, T, DOF)
        target[..., 0] = 1.0
        pred = target.clone()
        pred[:, 1:, 0] = -1.0
        mask = torch.tensor([[True, False, False, False]])
        loss = action_policy_loss(pred, target, mask)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_sample_weight_changes_loss(self) -> None:
        target = torch.zeros(2, T, DOF)
        target[..., 0] = 1.0
        pred = target.clone()
        pred[1, :, 0] = -1.0
        mask = torch.ones(2, T, dtype=torch.bool)
        equal = action_policy_loss(pred, target, mask)
        weighted = action_policy_loss(pred, target, mask, sample_weight=torch.tensor([10.0, 1.0]))
        assert weighted < equal


@pytest.mark.torch
class TestCombinedLoss:
    def test_returns_loss_output(self) -> None:
        pred_lat = torch.randn(B, T, P, D)
        tgt_lat = torch.randn(B, T, P, D)
        pred_wp = torch.randn(B, T, DOF)
        tgt_wp = torch.randn(B, T, DOF)
        fq = torch.ones(B)
        vo = torch.ones(B, T)
        out = combined_loss(pred_lat, tgt_lat, pred_wp, tgt_wp, fq, vo)
        assert isinstance(out, LossOutput)
        assert out.total.shape == ()
        assert out.latent.shape == ()
        assert out.waypoint.shape == ()
        assert out.cosine_sim.shape == ()

    def test_quality_floor_clamps(self) -> None:
        pred_lat = torch.randn(B, T, P, D)
        tgt_lat = torch.randn(B, T, P, D)
        pred_wp = torch.randn(B, T, DOF)
        tgt_wp = torch.randn(B, T, DOF)
        fq = torch.zeros(B)
        vo = torch.zeros(B, T)
        out = combined_loss(pred_lat, tgt_lat, pred_wp, tgt_wp, fq, vo)
        assert out.total.item() > 0

    def test_lambda_scaling(self) -> None:
        pred_lat = torch.randn(B, T, P, D)
        tgt_lat = torch.randn(B, T, P, D)
        pred_wp = torch.randn(B, T, DOF)
        tgt_wp = torch.randn(B, T, DOF)
        fq = torch.ones(B)
        vo = torch.ones(B, T)
        out_eq = combined_loss(pred_lat, tgt_lat, pred_wp, tgt_wp, fq, vo,
                               lambda_latent=1.0, lambda_waypoint=1.0)
        out_wp0 = combined_loss(pred_lat, tgt_lat, pred_wp, tgt_wp, fq, vo,
                                lambda_latent=1.0, lambda_waypoint=0.0)
        assert torch.allclose(out_wp0.total, out_wp0.latent)
        assert out_eq.total > out_wp0.total

    def test_cosine_sim_range(self) -> None:
        pred_lat = torch.randn(B, T, P, D)
        tgt_lat = torch.randn(B, T, P, D)
        pred_wp = torch.randn(B, T, DOF)
        tgt_wp = torch.randn(B, T, DOF)
        fq = torch.ones(B)
        vo = torch.ones(B, T)
        out = combined_loss(pred_lat, tgt_lat, pred_wp, tgt_wp, fq, vo)
        assert -1.0 <= out.cosine_sim.item() <= 1.0

    def test_cosine_sim_perfect(self) -> None:
        x = torch.randn(B, T, P, D)
        pred_wp = torch.randn(B, T, DOF)
        tgt_wp = torch.randn(B, T, DOF)
        fq = torch.ones(B)
        vo = torch.ones(B, T)
        out = combined_loss(x, x, pred_wp, tgt_wp, fq, vo)
        assert out.cosine_sim.item() == pytest.approx(1.0, abs=1e-4)

    def test_frame_quality_not_applied_to_wp(self) -> None:
        """L_wp must NOT be weighted by frame_quality."""
        pred_lat = torch.zeros(B, T, P, D)
        tgt_lat = torch.zeros(B, T, P, D)
        pred_wp = torch.randn(B, T, DOF)
        tgt_wp = torch.randn(B, T, DOF)
        vo = torch.ones(B, T)

        fq_high = torch.ones(B)
        fq_low = torch.full((B,), 0.1)
        out_high = combined_loss(pred_lat, tgt_lat, pred_wp, tgt_wp, fq_high, vo,
                                 lambda_latent=0.0, lambda_waypoint=1.0)
        out_low = combined_loss(pred_lat, tgt_lat, pred_wp, tgt_wp, fq_low, vo,
                                lambda_latent=0.0, lambda_waypoint=1.0)
        assert torch.allclose(out_high.total, out_low.total, atol=1e-7)
