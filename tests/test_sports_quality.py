"""Tests for vllatent.sports.quality — frame quality scoring and filtering."""
from __future__ import annotations

import numpy as np
import pytest

from vllatent.sports.quality import (
    clip_quality_summary,
    composite_quality,
    exposure_score,
    filter_frames,
    motion_blur_score,
    score_frames,
    snow_whiteout_score,
)


def _sharp_frame(h: int = 64, w: int = 64) -> np.ndarray:
    """Synthetic frame with high-frequency content (sharp edges)."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[::2, :, :] = 255
    return frame


def _blurry_frame(h: int = 64, w: int = 64) -> np.ndarray:
    """Synthetic uniform-ish frame (simulates heavy motion blur)."""
    return np.full((h, w, 3), 128, dtype=np.uint8)


def _white_frame(h: int = 64, w: int = 64) -> np.ndarray:
    """All-white frame (simulates whiteout)."""
    return np.full((h, w, 3), 250, dtype=np.uint8)


def _well_exposed_frame(h: int = 64, w: int = 64) -> np.ndarray:
    """Frame using full intensity range."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        frame[i, :, :] = int(255 * i / max(h - 1, 1))
    return frame


class TestMotionBlurScore:
    def test_sharp_scores_high(self) -> None:
        score = motion_blur_score(_sharp_frame())
        assert score > 0.5

    def test_blurry_scores_low(self) -> None:
        score = motion_blur_score(_blurry_frame())
        assert score < 0.05

    def test_output_range(self) -> None:
        score = motion_blur_score(_sharp_frame())
        assert 0.0 <= score <= 1.0

    def test_grayscale_input(self) -> None:
        gray = np.zeros((64, 64), dtype=np.uint8)
        gray[::2, :] = 255
        score = motion_blur_score(gray)
        assert score > 0.3


class TestExposureScore:
    def test_well_exposed_scores_high(self) -> None:
        score = exposure_score(_well_exposed_frame())
        assert score > 0.5

    def test_uniform_scores_low(self) -> None:
        score = exposure_score(_blurry_frame())
        assert score < 0.1

    def test_output_range(self) -> None:
        score = exposure_score(_well_exposed_frame())
        assert 0.0 <= score <= 1.0


class TestSnowWhiteoutScore:
    def test_white_frame_high(self) -> None:
        score = snow_whiteout_score(_white_frame())
        assert score > 0.9

    def test_normal_frame_low(self) -> None:
        score = snow_whiteout_score(_well_exposed_frame())
        assert score < 0.2

    def test_output_range(self) -> None:
        score = snow_whiteout_score(_sharp_frame())
        assert 0.0 <= score <= 1.0


class TestCompositeQuality:
    def test_good_frame_scores_high(self) -> None:
        score = composite_quality(_well_exposed_frame())
        assert score > 0.3

    def test_bad_frame_scores_low(self) -> None:
        score = composite_quality(_blurry_frame())
        assert score < 0.5

    def test_output_range(self) -> None:
        for frame in [_sharp_frame(), _blurry_frame(), _white_frame(), _well_exposed_frame()]:
            score = composite_quality(frame)
            assert 0.0 <= score <= 1.0, f"score {score} out of [0,1]"


class TestScoreFrames:
    def test_batch_ndarray(self) -> None:
        batch = np.stack([_sharp_frame(), _blurry_frame(), _well_exposed_frame()])
        scores = score_frames(batch)
        assert scores.shape == (3,)
        assert scores.dtype == np.float32

    def test_list_of_frames(self) -> None:
        frames = [_sharp_frame(), _blurry_frame()]
        scores = score_frames(frames)
        assert scores.shape == (2,)


class TestFilterFrames:
    def test_threshold(self) -> None:
        qualities = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)
        mask = filter_frames(qualities, threshold=0.5)
        assert mask.tolist() == [False, False, True, True, True]

    def test_all_pass(self) -> None:
        qualities = np.array([0.8, 0.9], dtype=np.float32)
        mask = filter_frames(qualities, threshold=0.3)
        assert mask.all()

    def test_all_fail(self) -> None:
        qualities = np.array([0.1, 0.2], dtype=np.float32)
        mask = filter_frames(qualities, threshold=0.5)
        assert not mask.any()


class TestClipQualitySummary:
    def test_summary_keys(self) -> None:
        qualities = np.array([0.1, 0.5, 0.8], dtype=np.float32)
        summary = clip_quality_summary(qualities, threshold=0.3)
        assert set(summary.keys()) == {"min", "max", "mean", "median", "n_total", "n_rejected", "rejection_rate"}

    def test_rejection_count(self) -> None:
        qualities = np.array([0.1, 0.2, 0.5, 0.8], dtype=np.float32)
        summary = clip_quality_summary(qualities, threshold=0.3)
        assert summary["n_total"] == 4
        assert summary["n_rejected"] == 2
        assert summary["rejection_rate"] == pytest.approx(0.5)

    def test_stats_correct(self) -> None:
        qualities = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32)
        summary = clip_quality_summary(qualities)
        assert summary["min"] == pytest.approx(0.2, abs=1e-5)
        assert summary["max"] == pytest.approx(0.8, abs=1e-5)
        assert summary["mean"] == pytest.approx(0.5, abs=1e-5)
        assert summary["median"] == pytest.approx(0.5, abs=1e-5)
