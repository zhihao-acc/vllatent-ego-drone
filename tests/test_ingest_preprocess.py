"""Tests for vllatent.ingest.preprocess — ffmpeg frame extraction (mocked)."""
from __future__ import annotations

from vllatent.ingest.preprocess import FrameExtraction, load_frame


def test_frame_extraction_fields() -> None:
    from pathlib import Path
    fe = FrameExtraction(frame_dir=Path("/tmp/f"), n_frames=10, fps=5.0, width=1280, height=720)
    assert fe.n_frames == 10
    assert fe.fps == 5.0


def test_load_frame_delegates_to_io(tmp_path) -> None:
    import numpy as np
    from PIL import Image
    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    p = tmp_path / "frame.jpg"
    img.save(str(p))
    frame = load_frame(p)
    assert frame.shape == (64, 64, 3)
    assert frame.dtype == np.uint8
