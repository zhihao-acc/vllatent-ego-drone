"""Tests for the B2.10 control-relevant world-action model."""
from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")

from vllatent.config import PredictorConfig  # noqa: E402
from vllatent.model.predictor import LatentPredictor  # noqa: E402
from vllatent.model.world_action_model import WorldActionModel, WorldActionOutput  # noqa: E402
from vllatent.scale_free_targets import SCALE_FREE_ACTION_DIM  # noqa: E402
from vllatent.schemas import HISTORY, HORIZON, PATCH_TOKENS  # noqa: E402


def _cfg(dropout: float = 0.0) -> PredictorConfig:
    return PredictorConfig(depth=1, heads=4, mlp_ratio=1, dropout=dropout)


def _make_inputs(batch_size: int = 2, dim: int = 16):
    history = torch.randn(batch_size, HISTORY, PATCH_TOKENS, dim)
    z_t = torch.randn(batch_size, PATCH_TOKENS, dim)
    history_mask = torch.ones(batch_size, HISTORY, dtype=torch.bool)
    last_action = torch.randn(batch_size, SCALE_FREE_ACTION_DIM)
    dt = torch.full((batch_size, HORIZON), 0.2)
    action_history = torch.randn(batch_size, HISTORY, SCALE_FREE_ACTION_DIM)
    action_history_mask = torch.ones(batch_size, HISTORY, dtype=torch.bool)
    camera_history_path = torch.randn(batch_size, HISTORY, 3)
    return (
        history,
        z_t,
        history_mask,
        last_action,
        dt,
        action_history,
        action_history_mask,
        camera_history_path,
    )


@pytest.mark.torch
class TestWorldActionModel:
    def test_forward_action_shape_and_rollout_latent_shape(self) -> None:
        model = WorldActionModel(_cfg(), dim=16, action_hidden_dim=16)
        inputs = _make_inputs(dim=16)
        actions = model(*inputs)
        rollout = model.rollout(*inputs)

        assert actions.shape == (2, HORIZON, SCALE_FREE_ACTION_DIM)
        assert actions.dtype == torch.float32
        assert isinstance(rollout, WorldActionOutput)
        assert rollout.predicted_latents.shape == (2, HORIZON, PATCH_TOKENS, 16)
        assert rollout.predicted_actions.shape == (2, HORIZON, SCALE_FREE_ACTION_DIM)

    def test_uses_b1_latent_predictor(self) -> None:
        model = WorldActionModel(_cfg(), dim=16, action_hidden_dim=16)
        assert isinstance(model.predictor, LatentPredictor)

    def test_deterministic_in_eval(self) -> None:
        model = WorldActionModel(_cfg(dropout=0.5), dim=16, action_hidden_dim=16)
        model.eval()
        inputs = _make_inputs(batch_size=1, dim=16)
        with torch.no_grad():
            out1 = model(*inputs)
            out2 = model(*inputs)
        assert torch.allclose(out1, out2, atol=1e-7)

    def test_gradient_flow_through_predictor_and_action_head(self) -> None:
        model = WorldActionModel(_cfg(), dim=16, action_hidden_dim=16)
        out = model(*_make_inputs(dim=16))
        out.square().mean().backward()

        predictor_grad = sum(
            float(p.grad.abs().sum()) for p in model.predictor.parameters() if p.grad is not None
        )
        action_head_grad = sum(
            float(p.grad.abs().sum()) for p in model.action_head.parameters() if p.grad is not None
        )
        assert predictor_grad > 0.0
        assert action_head_grad > 0.0

    def test_past_action_history_changes_world_rollout(self) -> None:
        model = WorldActionModel(_cfg(), dim=16, action_hidden_dim=16, latent_residual_init_std=1e-2)
        torch.nn.init.normal_(model.action_head.net[-1].weight, std=0.1)
        model.eval()
        inputs = list(_make_inputs(batch_size=1, dim=16))
        changed = list(inputs)
        changed[5] = changed[5].clone()
        changed[5][:, -1, 0] += 10.0

        with torch.no_grad():
            out1 = model.rollout(*inputs)
            out2 = model.rollout(*changed)

        assert not torch.allclose(out1.predicted_latents, out2.predicted_latents, atol=1e-6)
        assert not torch.allclose(out1.predicted_actions, out2.predicted_actions, atol=1e-6)

    def test_masked_action_history_does_not_change_rollout(self) -> None:
        model = WorldActionModel(_cfg(), dim=16, action_hidden_dim=16)
        model.eval()
        inputs = list(_make_inputs(batch_size=1, dim=16))
        inputs[6][:, 0] = False
        changed = list(inputs)
        changed[5] = changed[5].clone()
        changed[5][:, 0, :] += 100.0

        with torch.no_grad():
            out1 = model.rollout(*inputs)
            out2 = model.rollout(*changed)

        assert torch.allclose(out1.predicted_latents, out2.predicted_latents, atol=1e-6)
        assert torch.allclose(out1.predicted_actions, out2.predicted_actions, atol=1e-6)

    def test_masked_history_latents_do_not_change_rollout(self) -> None:
        model = WorldActionModel(_cfg(), dim=16, action_hidden_dim=16)
        model.eval()
        inputs = list(_make_inputs(batch_size=1, dim=16))
        inputs[2][:, 0] = False
        changed = list(inputs)
        changed[0] = changed[0].clone()
        changed[0][:, 0] += 100.0

        with torch.no_grad():
            out1 = model.rollout(*inputs)
            out2 = model.rollout(*changed)

        assert torch.allclose(out1.predicted_latents, out2.predicted_latents, atol=1e-6)
        assert torch.allclose(out1.predicted_actions, out2.predicted_actions, atol=1e-6)

    def test_rejects_bad_action_history_shape(self) -> None:
        model = WorldActionModel(_cfg(), dim=16, action_hidden_dim=16)
        inputs = list(_make_inputs(dim=16))
        inputs[5] = torch.zeros(2, HISTORY + 1, SCALE_FREE_ACTION_DIM)
        with pytest.raises(ValueError, match="action_history_scale_free"):
            model(*inputs)

    def test_forward_signature_has_no_future_targets(self) -> None:
        sig = inspect.signature(WorldActionModel.forward)
        forbidden = {
            "future_actions",
            "labels",
            "target_actions",
            "target_actions_scale_free",
            "target_latents",
            "future_latents",
            "odom_reference_speed",
        }
        assert forbidden.isdisjoint(sig.parameters)
        assert "last_action_scale_free" in sig.parameters
        assert "action_history_scale_free" in sig.parameters
