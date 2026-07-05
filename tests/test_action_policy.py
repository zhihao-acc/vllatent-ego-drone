"""Tests for the B2 direct scale-free action policy."""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from vllatent.model.action_policy import ScaleFreeActionPolicy  # noqa: E402
from vllatent.scale_free_targets import SCALE_FREE_ACTION_DIM  # noqa: E402
from vllatent.schemas import HISTORY, HORIZON, PATCH_TOKENS  # noqa: E402


def _make_inputs(batch_size: int = 2, dim: int = 32):
    history = torch.randn(batch_size, HISTORY, PATCH_TOKENS, dim)
    z_t = torch.randn(batch_size, PATCH_TOKENS, dim)
    history_mask = torch.ones(batch_size, HISTORY, dtype=torch.bool)
    last_action = torch.randn(batch_size, SCALE_FREE_ACTION_DIM)
    dt = torch.full((batch_size, HORIZON), 0.2)
    return history, z_t, history_mask, last_action, dt


@pytest.mark.torch
class TestScaleFreeActionPolicy:
    def test_output_shape(self) -> None:
        model = ScaleFreeActionPolicy(dim=32, hidden_dim=64, depth=1, heads=4)
        out = model(*_make_inputs(dim=32))
        assert out.shape == (2, HORIZON, SCALE_FREE_ACTION_DIM)
        assert out.dtype == torch.float32

    def test_accepts_cached_fp16_latents(self) -> None:
        model = ScaleFreeActionPolicy(dim=32, hidden_dim=64, depth=1, heads=4)
        history, z_t, mask, action, dt = _make_inputs(dim=32)
        out = model(history.half(), z_t.half(), mask, action, dt)
        assert out.shape == (2, HORIZON, SCALE_FREE_ACTION_DIM)
        assert out.dtype == torch.float32

    def test_deterministic_in_eval(self) -> None:
        model = ScaleFreeActionPolicy(dim=32, hidden_dim=64, depth=1, heads=4, dropout=0.5)
        model.eval()
        inputs = _make_inputs(batch_size=1, dim=32)
        with torch.no_grad():
            out1 = model(*inputs)
            out2 = model(*inputs)
        assert torch.allclose(out1, out2, atol=1e-7)

    def test_differentiable(self) -> None:
        model = ScaleFreeActionPolicy(dim=32, hidden_dim=64, depth=1, heads=4)
        history, z_t, mask, action, dt = _make_inputs(dim=32)
        history.requires_grad_(True)
        z_t.requires_grad_(True)
        out = model(history, z_t, mask, action, dt)
        loss = out.square().mean()
        loss.backward()
        assert history.grad is not None
        assert z_t.grad is not None
        assert any(p.grad is not None for p in model.parameters())

    def test_previous_action_changes_output(self) -> None:
        model = ScaleFreeActionPolicy(dim=32, hidden_dim=64, depth=1, heads=4)
        model.eval()
        history, z_t, mask, _, dt = _make_inputs(batch_size=1, dim=32)
        with torch.no_grad():
            out1 = model(history, z_t, mask, torch.zeros(1, SCALE_FREE_ACTION_DIM), dt)
            out2 = model(history, z_t, mask, torch.ones(1, SCALE_FREE_ACTION_DIM), dt)
        assert not torch.allclose(out1, out2, atol=1e-6)

    def test_dt_changes_output(self) -> None:
        model = ScaleFreeActionPolicy(dim=32, hidden_dim=64, depth=1, heads=4)
        model.eval()
        history, z_t, mask, action, _ = _make_inputs(batch_size=1, dim=32)
        with torch.no_grad():
            out1 = model(history, z_t, mask, action, torch.full((1, HORIZON), 0.2))
            out2 = model(history, z_t, mask, action, torch.full((1, HORIZON), 0.4))
        assert not torch.allclose(out1, out2, atol=1e-6)

    def test_history_mask_changes_output(self) -> None:
        model = ScaleFreeActionPolicy(dim=32, hidden_dim=64, depth=1, heads=4)
        model.eval()
        history, z_t, mask, action, dt = _make_inputs(batch_size=1, dim=32)
        history[:, 0] = history[:, 0] + 20.0
        masked = mask.clone()
        masked[:, 0] = False
        with torch.no_grad():
            out_all = model(history, z_t, mask, action, dt)
            out_masked = model(history, z_t, masked, action, dt)
        assert not torch.allclose(out_all, out_masked, atol=1e-6)

    def test_rejects_bad_head_divisibility(self) -> None:
        with pytest.raises(ValueError, match="hidden_dim"):
            ScaleFreeActionPolicy(dim=32, hidden_dim=63, heads=4)

    def test_forward_signature_has_no_future_targets(self) -> None:
        sig = inspect.signature(ScaleFreeActionPolicy.forward)
        forbidden = {
            "future_actions",
            "labels",
            "odom_reference_speed",
            "target_actions",
            "target_actions_scale_free",
        }
        assert forbidden.isdisjoint(sig.parameters)
        assert "last_action_scale_free" in sig.parameters


def test_policy_module_does_not_import_b1_predictor() -> None:
    path = Path("vllatent/model/action_policy.py")
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module != "vllatent.model.predictor"
        if isinstance(node, ast.Name):
            assert node.id != "LatentPredictor"
