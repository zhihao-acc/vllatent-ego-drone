"""Tests for pre-train sanity checks (B1.21)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from vllatent.schemas import DOF, EMBED_DIM, HISTORY, HORIZON, PATCH_TOKENS
from vllatent.train.sanity import _check_sample, run_sanity_check


@dataclass(frozen=True)
class FakeSample:
    z_t: np.ndarray
    history_latents: np.ndarray
    history_mask: np.ndarray
    target_latents: np.ndarray
    target_deltas: np.ndarray
    vo_confidence: np.ndarray
    frame_quality: float
    dt_seconds: np.ndarray


def _valid_sample() -> FakeSample:
    return FakeSample(
        z_t=np.zeros((PATCH_TOKENS, EMBED_DIM), dtype=np.float16),
        history_latents=np.zeros((HISTORY, PATCH_TOKENS, EMBED_DIM), dtype=np.float16),
        history_mask=np.array([False, True, True], dtype=np.bool_),
        target_latents=np.zeros((HORIZON, PATCH_TOKENS, EMBED_DIM), dtype=np.float16),
        target_deltas=np.ones((HORIZON, DOF), dtype=np.float32),
        vo_confidence=np.ones(HORIZON, dtype=np.float32),
        frame_quality=0.9,
        dt_seconds=np.full(HORIZON, 0.2, dtype=np.float32),
    )


class TestSanityCheck:
    def test_valid_passes(self) -> None:
        _check_sample(_valid_sample(), 0)

    def test_bad_z_t_shape(self) -> None:
        s = _valid_sample()
        bad = FakeSample(
            z_t=np.zeros((10, EMBED_DIM), dtype=np.float16),
            history_latents=s.history_latents,
            history_mask=s.history_mask,
            target_latents=s.target_latents,
            target_deltas=s.target_deltas,
            vo_confidence=s.vo_confidence,
            frame_quality=s.frame_quality,
            dt_seconds=s.dt_seconds,
        )
        with pytest.raises(ValueError, match="z_t shape"):
            _check_sample(bad, 0)

    def test_bad_z_t_dtype(self) -> None:
        s = _valid_sample()
        bad = FakeSample(
            z_t=np.zeros((PATCH_TOKENS, EMBED_DIM), dtype=np.float32),
            history_latents=s.history_latents,
            history_mask=s.history_mask,
            target_latents=s.target_latents,
            target_deltas=s.target_deltas,
            vo_confidence=s.vo_confidence,
            frame_quality=s.frame_quality,
            dt_seconds=s.dt_seconds,
        )
        with pytest.raises(ValueError, match="z_t dtype"):
            _check_sample(bad, 0)

    def test_bad_mask_last_false(self) -> None:
        s = _valid_sample()
        bad = FakeSample(
            z_t=s.z_t,
            history_latents=s.history_latents,
            history_mask=np.array([False, True, False], dtype=np.bool_),
            target_latents=s.target_latents,
            target_deltas=s.target_deltas,
            vo_confidence=s.vo_confidence,
            frame_quality=s.frame_quality,
            dt_seconds=s.dt_seconds,
        )
        with pytest.raises(ValueError, match="history_mask"):
            _check_sample(bad, 0)

    def test_nonfinite_deltas(self) -> None:
        s = _valid_sample()
        deltas = np.ones((HORIZON, DOF), dtype=np.float32)
        deltas[1, 2] = np.inf
        bad = FakeSample(
            z_t=s.z_t,
            history_latents=s.history_latents,
            history_mask=s.history_mask,
            target_latents=s.target_latents,
            target_deltas=deltas,
            vo_confidence=s.vo_confidence,
            frame_quality=s.frame_quality,
            dt_seconds=s.dt_seconds,
        )
        with pytest.raises(ValueError, match="non-finite"):
            _check_sample(bad, 0)

    def test_negative_dt(self) -> None:
        s = _valid_sample()
        bad = FakeSample(
            z_t=s.z_t,
            history_latents=s.history_latents,
            history_mask=s.history_mask,
            target_latents=s.target_latents,
            target_deltas=s.target_deltas,
            vo_confidence=s.vo_confidence,
            frame_quality=s.frame_quality,
            dt_seconds=np.array([0.2, -0.1, 0.2, 0.2], dtype=np.float32),
        )
        with pytest.raises(ValueError, match="non-positive"):
            _check_sample(bad, 0)

    def test_empty_dataset(self) -> None:
        class EmptyDataset:
            def __len__(self) -> int:
                return 0

        with pytest.raises(ValueError, match="empty"):
            run_sanity_check(EmptyDataset(), n_samples=3)  # type: ignore[arg-type]
