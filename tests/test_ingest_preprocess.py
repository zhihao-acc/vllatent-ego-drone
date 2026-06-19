"""Tests for vllatent.ingest.preprocess — ffmpeg frame extraction (mocked)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vllatent.ingest.preprocess import FrameExtraction, cut_fixed_clips, load_frame


def test_frame_extraction_fields() -> None:
    fe = FrameExtraction(frame_dir=Path("/tmp/f"), n_frames=10, fps=5.0, width=1280, height=720)
    assert fe.n_frames == 10
    assert fe.fps == 5.0


def test_load_frame_delegates_to_io(tmp_path) -> None:
    from PIL import Image
    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    p = tmp_path / "frame.jpg"
    img.save(str(p))
    frame = load_frame(p)
    assert frame.shape == (64, 64, 3)
    assert frame.dtype == np.uint8


class TestCutFixedClips:
    def _paths(self, n: int) -> list[Path]:
        return [Path(f"/tmp/{i:06d}.jpg") for i in range(n)]

    def test_exact_division(self) -> None:
        segs = cut_fixed_clips(self._paths(50), clip_length_frames=10, min_usable_frames=7)
        assert len(segs) == 5
        assert all(len(s) == 10 for s in segs)

    def test_trailing_short_segment_discarded(self) -> None:
        segs = cut_fixed_clips(self._paths(53), clip_length_frames=10, min_usable_frames=7)
        assert len(segs) == 5  # 3-frame tail discarded

    def test_trailing_usable_segment_kept(self) -> None:
        segs = cut_fixed_clips(self._paths(57), clip_length_frames=10, min_usable_frames=7)
        assert len(segs) == 6  # 7-frame tail kept
        assert len(segs[-1]) == 7

    def test_empty_input(self) -> None:
        segs = cut_fixed_clips([], clip_length_frames=10)
        assert segs == []

    def test_fewer_than_min_usable(self) -> None:
        segs = cut_fixed_clips(self._paths(5), clip_length_frames=10, min_usable_frames=7)
        assert segs == []

    def test_rejects_clip_length_less_than_min(self) -> None:
        with pytest.raises(ValueError, match="clip_length_frames"):
            cut_fixed_clips(self._paths(10), clip_length_frames=3, min_usable_frames=7)

    def test_default_min_usable_is_7(self) -> None:
        segs = cut_fixed_clips(self._paths(50), clip_length_frames=50)
        assert len(segs) == 1
        assert len(segs[0]) == 50
