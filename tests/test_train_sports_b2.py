"""Tests for B2 action-policy trainer."""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.train_sports_b2 import (  # noqa: E402
    ActionTrainConfig,
    _select_clip_stems,
    evaluate_action_policy,
    train_full,
    train_overfit_tiny,
)
from vllatent.data.collate import ActionPolicyBatch  # noqa: E402
from vllatent.scale_free_targets import SCALE_FREE_ACTION_DIM  # noqa: E402
from vllatent.schemas import (  # noqa: E402
    EMBED_DIM,
    HISTORY,
    HORIZON,
    LATENT_DTYPE,
    MASK_DTYPE,
    PATCH_TOKENS,
)


def _make_clip_npz(
    path: Path,
    *,
    n_frames: int = 12,
    delta: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0),
) -> None:
    rng = np.random.default_rng(abs(hash(path.stem)) % (2**32))
    latents = rng.standard_normal((n_frames, PATCH_TOKENS, EMBED_DIM)).astype(LATENT_DTYPE)
    deltas = np.tile(np.array(delta, dtype=np.float32), (n_frames - 1, 1))
    vo_confidence = np.ones(n_frames, dtype=np.float32)
    frame_quality = np.ones(n_frames, dtype=np.float32)
    timestamps = np.arange(n_frames, dtype=np.float64) / 5.0
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(path),
        latents=latents,
        deltas=deltas,
        vo_confidence=vo_confidence,
        frame_quality=frame_quality,
        timestamps=timestamps,
    )


def _manual_batch(target: torch.Tensor, last_action: torch.Tensor) -> ActionPolicyBatch:
    batch_size = target.shape[0]
    return ActionPolicyBatch(
        z_t=torch.zeros(batch_size, PATCH_TOKENS, EMBED_DIM, dtype=torch.float16),
        history_latents=torch.zeros(batch_size, HISTORY, PATCH_TOKENS, EMBED_DIM, dtype=torch.float16),
        history_mask=torch.ones(batch_size, HISTORY, dtype=torch.bool),
        target_actions_scale_free=target,
        target_actions_moving_mask=torch.ones(batch_size, HORIZON, dtype=torch.bool),
        last_action_scale_free=last_action,
        dt_seconds=torch.full((batch_size, HORIZON), 0.2),
        odom_reference_speed=torch.ones(batch_size),
        vo_confidence=torch.ones(batch_size, HORIZON),
        frame_quality=torch.ones(batch_size),
        sample_weight=torch.ones(batch_size),
    )


class _ConstantPolicy(torch.nn.Module):
    def __init__(self, action: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("action", action)

    def forward(self, history_latents, z_t, history_mask, last_action_scale_free, dt_seconds):  # noqa: ANN001
        return self.action[: z_t.shape[0]].to(z_t.device)


@pytest.mark.torch
def test_evaluate_action_policy_reports_positive_margin_for_perfect_nonbaseline() -> None:
    target = torch.zeros(2, HORIZON, SCALE_FREE_ACTION_DIM)
    target[..., 1] = 1.0
    target[..., 3] = 0.5
    last = torch.zeros(2, SCALE_FREE_ACTION_DIM)
    last[..., 0] = 1.0
    batch = _manual_batch(target, last)
    metrics = evaluate_action_policy(_ConstantPolicy(target), [batch], device="cpu")
    assert metrics["action_score"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["action_margin"] > 0.0
    assert "baseline_repeat_last_score" in metrics


@pytest.mark.torch
def test_overfit_tiny_writes_action_artifacts(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    _make_clip_npz(cache / "srca_fpv00_c000.npz")
    args = Namespace(
        cache_dir=cache,
        run_dir=tmp_path / "run_overfit",
        device="cpu",
        overfit_samples=4,
        max_steps=2,
        log_every=1,
    )
    cfg = ActionTrainConfig(batch_size=2, hidden_dim=16, depth=1, heads=4, epochs=1)
    metrics = train_overfit_tiny(args, cfg)
    run_dir = Path(args.run_dir)
    assert "action_margin" in metrics
    assert (run_dir / "config_snapshot.yaml").exists()
    assert (run_dir / "train_b2_config.json").exists()
    assert (run_dir / "train_action_metrics.jsonl").exists()
    assert (run_dir / "ckpt_best.pt").exists()
    assert (run_dir / "ckpt_final.pt").exists()


@pytest.mark.torch
def test_full_train_writes_val_and_source_metrics(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    _make_clip_npz(cache / "srca_fpv00_c000.npz", delta=(0.0, 1.0, 0.0, 0.0))
    _make_clip_npz(cache / "srcb_fpv00_c000.npz", delta=(1.0, 0.0, 0.0, 0.0))
    args = Namespace(
        cache_dir=cache,
        run_dir=tmp_path / "run_full",
        device="cpu",
        max_clips=0,
        max_clips_per_source=0,
        eval_by_source=True,
    )
    cfg = ActionTrainConfig(
        batch_size=2,
        hidden_dim=16,
        depth=1,
        heads=4,
        epochs=1,
        val_frac=0.5,
        early_stop_patience=1,
    )
    metrics = train_full(args, cfg)
    run_dir = Path(args.run_dir)
    assert "action_margin" in metrics
    assert (run_dir / "val_action_metrics.jsonl").exists()
    assert (run_dir / "source_action_metrics.jsonl").exists()
    assert (run_dir / "ckpt_best.pt").exists()
    assert (run_dir / "norm_stats.npz").exists()


def test_synthetic_cache_helper_uses_expected_dtypes(tmp_path: Path) -> None:
    path = tmp_path / "cache" / "srca_fpv00_c000.npz"
    _make_clip_npz(path)
    with np.load(path) as data:
        assert data["latents"].dtype == LATENT_DTYPE
        assert data["deltas"].dtype == np.float32
        assert data["latents"].shape[1:] == (PATCH_TOKENS, EMBED_DIM)
        assert data["deltas"].shape[1] == SCALE_FREE_ACTION_DIM
        assert np.dtype(MASK_DTYPE) == np.dtype(np.bool_)


def test_select_clip_stems_can_cap_per_source(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    for source in ["srca", "srcb", "srcc"]:
        for i in range(3):
            _make_clip_npz(cache / f"{source}_fpv00_c{i:03d}.npz")

    stems = _select_clip_stems(cache, max_clips_per_source=2)

    assert len(stems) == 6
    assert all(stem.endswith(("c000", "c001")) for stem in stems)
