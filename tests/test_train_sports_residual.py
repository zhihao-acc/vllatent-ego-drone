"""Residual latent training path tests for B1.22e."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from scripts.train_sports import _forward_loss  # noqa: E402
from vllatent.config import PredictorConfig, TrainConfig  # noqa: E402
from vllatent.data.collate import TrainingBatch  # noqa: E402
from vllatent.model.sports_model import SportsFollowingModel  # noqa: E402
from vllatent.schemas import DOF, HISTORY, HORIZON, PATCH_TOKENS  # noqa: E402
from vllatent.train.losses import latent_loss  # noqa: E402

pytestmark = pytest.mark.torch


def _make_batch(dim: int = 32) -> TrainingBatch:
    bsz = 1
    z_t = torch.randn(bsz, PATCH_TOKENS, dim)
    delta = torch.full((bsz, HORIZON, PATCH_TOKENS, dim), 0.25)
    return TrainingBatch(
        z_t=z_t,
        history_latents=torch.randn(bsz, HISTORY, PATCH_TOKENS, dim),
        history_mask=torch.ones(bsz, HISTORY, dtype=torch.bool),
        target_latents=z_t.unsqueeze(1).expand(-1, HORIZON, -1, -1) + delta,
        target_deltas=torch.randn(bsz, HORIZON, DOF),
        last_action=torch.randn(bsz, DOF),
        vo_confidence=torch.ones(bsz, HORIZON),
        frame_quality=torch.ones(bsz),
        dt_seconds=torch.full((bsz, HORIZON), 0.2),
        sample_weight=torch.ones(bsz),
    )


def test_residual_absolute_loss_starts_from_persistence() -> None:
    cfg = PredictorConfig(depth=1, heads=4)
    model = SportsFollowingModel(cfg, dim=32, prediction_mode="residual")
    batch = _make_batch()
    tcfg = TrainConfig(latent_only=True, prediction_mode="residual", latent_loss_mode="absolute")

    loss_out, predicted, _ = _forward_loss(model, batch, tcfg, device="cpu")

    persistence = batch.z_t.unsqueeze(1).expand_as(batch.target_latents)
    expected = latent_loss(persistence, batch.target_latents, torch.ones(1), beta=0.1)
    assert torch.allclose(predicted, persistence, atol=1e-6)
    assert loss_out.latent == pytest.approx(expected)


def test_residual_combined_loss_adds_delta_auxiliary() -> None:
    cfg = PredictorConfig(depth=1, heads=4)
    model = SportsFollowingModel(cfg, dim=32, prediction_mode="residual")
    batch = _make_batch()
    tcfg = TrainConfig(
        latent_only=True,
        prediction_mode="residual",
        latent_loss_mode="combined",
        delta_loss_weight=0.5,
    )

    loss_out, predicted, _ = _forward_loss(model, batch, tcfg, device="cpu")

    persistence = batch.z_t.unsqueeze(1).expand_as(batch.target_latents)
    target_delta = batch.target_latents - persistence
    zero_delta = predicted - persistence
    expected_abs = latent_loss(persistence, batch.target_latents, torch.ones(1), beta=0.1)
    expected_delta = latent_loss(zero_delta, target_delta, torch.ones(1), beta=0.1)
    assert loss_out.latent == pytest.approx(expected_abs + 0.5 * expected_delta)
