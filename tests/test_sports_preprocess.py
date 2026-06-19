"""Tests for vllatent.sports.preprocess — frame extraction (mocked for CI)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vllatent.sports.preprocess import FrameExtraction, load_frame


class TestFrameExtraction:
    def test_frozen(self) -> None:
        fe = FrameExtraction(frame_dir=Path("/tmp"), n_frames=10, fps=5.0, width=1280, height=720)
        with pytest.raises(AttributeError):
            fe.n_frames = 20  # type: ignore[misc]

    def test_fields(self) -> None:
        fe = FrameExtraction(frame_dir=Path("/tmp"), n_frames=10, fps=5.0, width=1280, height=720)
        assert fe.n_frames == 10
        assert fe.fps == 5.0
        assert fe.width == 1280


class TestLoadFrame:
    def test_load_synthetic_jpeg(self, tmp_path: Path) -> None:
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not available")

        frame = np.full((64, 64, 3), 128, dtype=np.uint8)
        img = Image.fromarray(frame)
        path = tmp_path / "test.jpg"
        img.save(str(path))

        loaded = load_frame(path)
        assert loaded.shape == (64, 64, 3)
        assert loaded.dtype == np.uint8
        assert np.abs(int(loaded.mean()) - 128) < 10
