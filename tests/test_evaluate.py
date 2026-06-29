"""B1.22a tests: per-horizon val cosine + persistence baseline (TORCH tier)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vllatent.config import PredictorConfig  # noqa: E402
from vllatent.data.collate import TrainingBatch, collate_sports_batch  # noqa: E402
from vllatent.data.sports_loader import SportsTrainingDataset, split_clips_by_source  # noqa: E402
from vllatent.model.sports_model import SportsFollowingModel  # noqa: E402
from vllatent.schemas import DOF, EMBED_DIM, HISTORY, HORIZON, PATCH_TOKENS  # noqa: E402
from vllatent.train.evaluate import evaluate, per_horizon_cosine  # noqa: E402

pytestmark = pytest.mark.torch


def _make_batch(B: int = 2) -> TrainingBatch:
    return TrainingBatch(
        z_t=torch.randn(B, PATCH_TOKENS, EMBED_DIM),
        history_latents=torch.randn(B, HISTORY, PATCH_TOKENS, EMBED_DIM),
        history_mask=torch.ones(B, HISTORY, dtype=torch.bool),
        target_latents=torch.randn(B, HORIZON, PATCH_TOKENS, EMBED_DIM),
        target_deltas=torch.randn(B, HORIZON, DOF),
        last_action=torch.randn(B, DOF),
        vo_confidence=torch.ones(B, HORIZON),
        frame_quality=torch.ones(B),
        dt_seconds=torch.full((B, HORIZON), 0.2),
        sample_weight=torch.ones(B),
    )


def _write_clip(path: Path, n_frames: int = 20, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(path),
        latents=rng.standard_normal((n_frames, PATCH_TOKENS, EMBED_DIM)).astype(np.float16),
        deltas=(rng.standard_normal((n_frames - 1, DOF)) * 0.1).astype(np.float32),
        vo_confidence=np.clip(rng.random(n_frames), 0.1, 1.0).astype(np.float32),
        frame_quality=np.clip(rng.random(n_frames), 0.2, 1.0).astype(np.float32),
        timestamps=(np.arange(n_frames) / 5.0).astype(np.float64),
    )


# --- per_horizon_cosine ---


def test_per_horizon_cosine_identical_is_one() -> None:
    x = torch.randn(2, HORIZON, PATCH_TOKENS, EMBED_DIM)
    cos = per_horizon_cosine(x, x)
    assert cos.shape == (HORIZON,)
    assert torch.allclose(cos, torch.ones(HORIZON), atol=1e-5)


def test_per_horizon_cosine_negated_is_minus_one() -> None:
    x = torch.randn(2, HORIZON, PATCH_TOKENS, EMBED_DIM)
    cos = per_horizon_cosine(x, -x)
    assert torch.allclose(cos, -torch.ones(HORIZON), atol=1e-5)


# --- evaluate ---


def test_evaluate_returns_per_horizon_structure() -> None:
    model = SportsFollowingModel(PredictorConfig(depth=2))
    loader = [_make_batch(B=2), _make_batch(B=3)]
    out = evaluate(model, loader, device="cpu")
    for key in ("per_horizon_cos", "per_horizon_persistence", "per_horizon_margin"):
        assert len(out[key]) == HORIZON
    assert out["n_samples"] == 5
    assert all(-1.0 <= c <= 1.0 for c in out["per_horizon_cos"])


def test_evaluate_margin_is_cos_minus_persistence() -> None:
    model = SportsFollowingModel(PredictorConfig(depth=2))
    out = evaluate(model, [_make_batch(B=2)], device="cpu")
    for k in range(HORIZON):
        expected = out["per_horizon_cos"][k] - out["per_horizon_persistence"][k]
        assert abs(out["per_horizon_margin"][k] - expected) < 1e-5


def test_evaluate_persistence_is_one_when_target_equals_zt() -> None:
    """If GT future == z_t, the persistence baseline cosine is 1.0 (next == current)."""
    batch = _make_batch(B=2)
    z_rep = batch.z_t.unsqueeze(1).expand(-1, HORIZON, -1, -1).contiguous()
    batch = batch._replace(target_latents=z_rep)
    model = SportsFollowingModel(PredictorConfig(depth=2))
    out = evaluate(model, [batch], device="cpu")
    assert out["val_persistence"] == pytest.approx(1.0, abs=1e-4)


def test_evaluate_empty_loader_raises() -> None:
    model = SportsFollowingModel(PredictorConfig(depth=2))
    with pytest.raises(ValueError, match="no batches"):
        evaluate(model, [], device="cpu")


def test_evaluate_restores_train_mode() -> None:
    model = SportsFollowingModel(PredictorConfig(depth=2))
    model.train()
    evaluate(model, [_make_batch(B=1)], device="cpu")
    assert model.training is True


def test_evaluate_end_to_end_two_sources(tmp_path: Path) -> None:
    """split → train/val datasets (val on TRAIN norm-stats) → evaluate over val."""
    idx = 0
    for src in ("skiA", "skiB"):
        for c in range(2):
            _write_clip(tmp_path / f"{src}_fpv00_c00{c}.npz", seed=idx)
            idx += 1
    stems = sorted(p.stem for p in tmp_path.glob("*.npz"))
    train_stems, val_stems = split_clips_by_source(stems, val_frac=0.5, seed=1)
    assert val_stems, "expected a held-out val source"

    train_ds = SportsTrainingDataset(tmp_path, clip_ids=train_stems, augment=False)
    val_ds = SportsTrainingDataset(
        tmp_path, clip_ids=val_stems, augment=False, norm_stats=train_ds.norm_stats
    )
    loader = torch.utils.data.DataLoader(val_ds, batch_size=8, collate_fn=collate_sports_batch)
    model = SportsFollowingModel(PredictorConfig(depth=2))
    out = evaluate(model, loader, device="cpu")
    assert out["n_samples"] == len(val_ds)
    assert len(out["per_horizon_margin"]) == HORIZON
    # val dataset must reuse the TRAIN norm stats (no leakage)
    assert np.allclose(val_ds.norm_stats.mean, train_ds.norm_stats.mean)
