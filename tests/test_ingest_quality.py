"""Tests for vllatent.ingest.quality — frame quality scoring."""
from __future__ import annotations

import numpy as np

from vllatent.ingest.quality import (
    clip_quality_summary,
    composite_quality,
    exposure_score,
    filter_frames,
    motion_blur_score,
    score_frames,
    snow_whiteout_score,
)


def _random_frame(h: int = 64, w: int = 64, seed: int = 42) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _flat_frame(value: int = 128, h: int = 64, w: int = 64) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


class TestMotionBlurScore:
    def test_range(self) -> None:
        score = motion_blur_score(_random_frame())
        assert 0.0 <= score <= 1.0

    def test_flat_is_zero(self) -> None:
        assert motion_blur_score(_flat_frame()) == 0.0

    def test_sharp_higher_than_flat(self) -> None:
        sharp = _random_frame()
        flat = _flat_frame()
        assert motion_blur_score(sharp) > motion_blur_score(flat)

    def test_grayscale_input(self) -> None:
        gray = np.random.default_rng(1).integers(0, 256, (64, 64), dtype=np.uint8)
        score = motion_blur_score(gray)
        assert 0.0 <= score <= 1.0


class TestExposureScore:
    def test_range(self) -> None:
        score = exposure_score(_random_frame())
        assert 0.0 <= score <= 1.0

    def test_flat_is_low(self) -> None:
        score = exposure_score(_flat_frame())
        assert score < 0.1

    def test_random_is_higher(self) -> None:
        assert exposure_score(_random_frame()) > exposure_score(_flat_frame())


class TestSnowWhiteoutScore:
    def test_no_whiteout(self) -> None:
        score = snow_whiteout_score(_flat_frame(value=100))
        assert score == 0.0

    def test_full_whiteout(self) -> None:
        score = snow_whiteout_score(_flat_frame(value=240))
        assert score == 1.0

    def test_range(self) -> None:
        score = snow_whiteout_score(_random_frame())
        assert 0.0 <= score <= 1.0


class TestCompositeQuality:
    def test_range(self) -> None:
        q = composite_quality(_random_frame())
        assert 0.0 <= q <= 1.0

    def test_custom_weights(self) -> None:
        q = composite_quality(_random_frame(), w_blur=1.0, w_exposure=0.0, w_whiteout=0.0)
        assert 0.0 <= q <= 1.0


class TestScoreFrames:
    def test_batch(self) -> None:
        frames = [_random_frame(seed=i) for i in range(5)]
        scores = score_frames(frames)
        assert scores.shape == (5,)
        assert scores.dtype == np.float32

    def test_4d_array(self) -> None:
        batch = np.stack([_random_frame(seed=i) for i in range(3)])
        scores = score_frames(batch)
        assert scores.shape == (3,)


class TestFilterFrames:
    def test_threshold(self) -> None:
        q = np.array([0.1, 0.5, 0.3, 0.8], dtype=np.float32)
        mask = filter_frames(q, threshold=0.3)
        assert mask.tolist() == [False, True, True, True]


class TestClipQualitySummary:
    def test_summary_keys(self) -> None:
        q = np.array([0.2, 0.5, 0.8], dtype=np.float32)
        summary = clip_quality_summary(q, threshold=0.3)
        for key in ("min", "max", "mean", "median", "n_total", "n_rejected", "rejection_rate"):
            assert key in summary

    def test_rejection_count(self) -> None:
        q = np.array([0.1, 0.5, 0.8], dtype=np.float32)
        summary = clip_quality_summary(q, threshold=0.3)
        assert summary["n_rejected"] == 1.0
