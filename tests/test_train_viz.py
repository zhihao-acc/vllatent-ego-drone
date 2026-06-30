"""Tests for training visualization logger (B1.21)."""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from vllatent.schemas import DOF, EMBED_DIM, HORIZON, PATCH_TOKENS  # noqa: E402
from vllatent.train.losses import LossOutput  # noqa: E402
from vllatent.train.viz import TrainingLogger  # noqa: E402


def _fake_loss_output() -> LossOutput:
    return LossOutput(
        total=torch.tensor(0.5),
        latent=torch.tensor(0.3),
        waypoint=torch.tensor(0.2),
        cosine_sim=torch.tensor(0.7),
    )


@pytest.mark.torch
class TestTrainingLogger:
    def test_creates_log_dir(self, tmp_path: object) -> None:
        log_dir = tmp_path / "logs"  # type: ignore[operator]
        TrainingLogger(log_dir=log_dir)
        assert log_dir.exists()

    def test_should_log(self) -> None:
        import pathlib
        logger = TrainingLogger(log_dir=pathlib.Path("/tmp/test_viz"), log_every=50)
        assert logger.should_log(0)
        assert not logger.should_log(1)
        assert logger.should_log(50)
        assert logger.should_log(100)

    def test_log_basic_step(self, tmp_path: object) -> None:
        logger = TrainingLogger(log_dir=tmp_path)  # type: ignore[arg-type]
        logger.log_step(step=0, epoch=0, loss_output=_fake_loss_output(), lr=1e-4)
        lines = logger.log_path.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["step"] == 0
        assert entry["loss_total"] == pytest.approx(0.5)
        assert entry["cosine_sim"] == pytest.approx(0.7)
        assert entry["lr"] == pytest.approx(1e-4)

    def test_log_extra_metrics(self, tmp_path: object) -> None:
        logger = TrainingLogger(log_dir=tmp_path)  # type: ignore[arg-type]
        logger.log_step(
            step=0,
            epoch=0,
            loss_output=_fake_loss_output(),
            lr=1e-4,
            extra_metrics={"grad_norm": 1.25},
        )
        entry = json.loads(logger.log_path.read_text().strip())
        assert entry["grad_norm"] == pytest.approx(1.25)

    def test_log_with_per_horizon(self, tmp_path: object) -> None:
        logger = TrainingLogger(log_dir=tmp_path)  # type: ignore[arg-type]
        B, T, P, D = 2, HORIZON, PATCH_TOKENS, EMBED_DIM
        pred_lat = torch.randn(B, T, P, D)
        tgt_lat = torch.randn(B, T, P, D)
        pred_wp = torch.randn(B, T, DOF)
        tgt_wp = torch.randn(B, T, DOF)
        logger.log_step(
            step=50, epoch=1, loss_output=_fake_loss_output(), lr=5e-5,
            predicted_latents=pred_lat, target_latents=tgt_lat,
            predicted_deltas=pred_wp, target_deltas=tgt_wp,
        )
        entry = json.loads(logger.log_path.read_text().strip())
        assert "cosine_per_horizon" in entry
        assert len(entry["cosine_per_horizon"]) == T
        assert "wp_l1_per_horizon" in entry
        assert len(entry["wp_l1_per_horizon"]) == T

    def test_multiple_steps_appended(self, tmp_path: object) -> None:
        logger = TrainingLogger(log_dir=tmp_path)  # type: ignore[arg-type]
        for i in range(3):
            logger.log_step(step=i, epoch=0, loss_output=_fake_loss_output(), lr=1e-4)
        lines = logger.log_path.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_log_every_zero_raises(self) -> None:
        import pathlib
        with pytest.raises(ValueError, match="log_every"):
            TrainingLogger(log_dir=pathlib.Path("/tmp/test_viz"), log_every=0)

    def test_cosine_per_horizon_range(self, tmp_path: object) -> None:
        logger = TrainingLogger(log_dir=tmp_path)  # type: ignore[arg-type]
        B, T, P, D = 2, HORIZON, PATCH_TOKENS, EMBED_DIM
        x = torch.randn(B, T, P, D)
        logger.log_step(
            step=0, epoch=0, loss_output=_fake_loss_output(), lr=1e-4,
            predicted_latents=x, target_latents=x,
        )
        entry = json.loads(logger.log_path.read_text().strip())
        for cos in entry["cosine_per_horizon"]:
            assert -1.0 <= cos <= 1.01
