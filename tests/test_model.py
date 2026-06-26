"""Tests for full model assembly (B1.17)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vllatent.config import PredictorConfig  # noqa: E402
from vllatent.data.collate import TrainingBatch  # noqa: E402
from vllatent.model.sports_model import ModelOutput, SportsFollowingModel  # noqa: E402
from vllatent.schemas import DOF, EMBED_DIM, HISTORY, HORIZON, PATCH_TOKENS  # noqa: E402


def _make_batch(B: int = 2, dim: int = EMBED_DIM) -> TrainingBatch:
    return TrainingBatch(
        z_t=torch.randn(B, PATCH_TOKENS, dim),
        history_latents=torch.randn(B, HISTORY, PATCH_TOKENS, dim),
        history_mask=torch.ones(B, HISTORY, dtype=torch.bool),
        target_latents=torch.randn(B, HORIZON, PATCH_TOKENS, dim),
        target_deltas=torch.randn(B, HORIZON, DOF),
        vo_confidence=torch.ones(B, HORIZON),
        frame_quality=torch.ones(B),
        dt_seconds=torch.full((B, HORIZON), 0.2),
        sample_weight=torch.ones(B),
    )


@pytest.mark.torch
class TestSportsFollowingModel:
    def test_output_shapes(self) -> None:
        cfg = PredictorConfig(depth=2)
        model = SportsFollowingModel(cfg)
        batch = _make_batch(B=2)
        out = model(batch)
        assert out.predicted_latents.shape == (2, HORIZON, PATCH_TOKENS, EMBED_DIM)
        assert out.predicted_deltas.shape == (2, HORIZON, DOF)

    def test_output_shapes_small_dim(self) -> None:
        cfg = PredictorConfig(depth=2, heads=6)
        model = SportsFollowingModel(cfg, dim=384)
        batch = _make_batch(B=3, dim=384)
        out = model(batch)
        assert out.predicted_latents.shape == (3, HORIZON, PATCH_TOKENS, 384)
        assert out.predicted_deltas.shape == (3, HORIZON, DOF)

    def test_model_output_type(self) -> None:
        cfg = PredictorConfig(depth=2)
        model = SportsFollowingModel(cfg)
        batch = _make_batch(B=1)
        out = model(batch)
        assert isinstance(out, ModelOutput)

    def test_from_config(self) -> None:
        cfg = PredictorConfig(depth=2)
        model = SportsFollowingModel.from_config(cfg)
        assert isinstance(model, SportsFollowingModel)
        assert model.predictor.depth == 2

    def test_differentiable(self) -> None:
        cfg = PredictorConfig(depth=2)
        model = SportsFollowingModel(cfg)
        batch = _make_batch(B=2)
        out = model(batch)
        loss = out.predicted_latents.sum() + out.predicted_deltas.sum()
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_param_count_depth6(self) -> None:
        cfg = PredictorConfig(depth=6)
        model = SportsFollowingModel(cfg)
        n = sum(p.numel() for p in model.parameters())
        assert 40_000_000 < n < 70_000_000

    def test_config_driven_depth(self) -> None:
        cfg2 = PredictorConfig(depth=2)
        cfg4 = PredictorConfig(depth=4)
        m2 = SportsFollowingModel(cfg2)
        m4 = SportsFollowingModel(cfg4)
        n2 = sum(p.numel() for p in m2.parameters())
        n4 = sum(p.numel() for p in m4.parameters())
        assert n4 > n2

    def test_eval_deterministic(self) -> None:
        cfg = PredictorConfig(depth=2, dropout=0.5)
        model = SportsFollowingModel(cfg)
        model.eval()
        batch = _make_batch(B=1)
        with torch.no_grad():
            out1 = model(batch)
            out2 = model(batch)
        assert torch.allclose(out1.predicted_latents, out2.predicted_latents, atol=1e-7)
        assert torch.allclose(out1.predicted_deltas, out2.predicted_deltas, atol=1e-7)
