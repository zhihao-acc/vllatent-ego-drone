"""Tests for waypoint head (B1.16)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vllatent.model.heads import WaypointHead  # noqa: E402
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
