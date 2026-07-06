"""Tests for waypoint head (B1.16)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vllatent.model.heads import ScaleFreeActionHead, WaypointHead  # noqa: E402
from vllatent.scale_free_targets import SCALE_FREE_ACTION_DIM  # noqa: E402
from vllatent.schemas import DOF, EMBED_DIM, HORIZON  # noqa: E402


@pytest.mark.torch
class TestWaypointHead:
    def test_output_shape(self) -> None:
        head = WaypointHead(dim=EMBED_DIM)
        x = torch.randn(2, HORIZON, EMBED_DIM)
        out = head(x)
        assert out.shape == (2, HORIZON, DOF)

    def test_output_shape_small(self) -> None:
        head = WaypointHead(dim=384)
        x = torch.randn(3, HORIZON, 384)
        out = head(x)
        assert out.shape == (3, HORIZON, DOF)

    def test_output_dtype(self) -> None:
        head = WaypointHead()
        x = torch.randn(1, HORIZON, EMBED_DIM)
        out = head(x)
        assert out.dtype == torch.float32

    def test_differentiable(self) -> None:
        head = WaypointHead()
        x = torch.randn(2, HORIZON, EMBED_DIM, requires_grad=True)
        out = head(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_param_count(self) -> None:
        head = WaypointHead(dim=EMBED_DIM)
        n = sum(p.numel() for p in head.parameters())
        assert 200_000 < n < 300_000


@pytest.mark.torch
class TestScaleFreeActionHead:
    def test_output_shape(self) -> None:
        head = ScaleFreeActionHead(dim=32, hidden_dim=32)
        x = torch.randn(2, HORIZON, 32)
        out = head(x)
        assert out.shape == (2, HORIZON, SCALE_FREE_ACTION_DIM)

    def test_rejects_unlocked_action_dim(self) -> None:
        with pytest.raises(ValueError, match="action_dim"):
            ScaleFreeActionHead(dim=32, hidden_dim=32, action_dim=SCALE_FREE_ACTION_DIM + 1)

    def test_small_final_init_allows_gradient_flow(self) -> None:
        head = ScaleFreeActionHead(dim=32, hidden_dim=32, final_init_std=1e-3)
        x = torch.randn(2, HORIZON, 32, requires_grad=True)
        out = head(x)
        out.square().mean().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_zero_final_init_outputs_zero_residual(self) -> None:
        head = ScaleFreeActionHead(dim=32, hidden_dim=32, final_init_std=0.0)
        x = torch.randn(2, HORIZON, 32)
        out = head(x)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-7)
