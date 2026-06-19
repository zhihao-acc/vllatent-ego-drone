"""Tests for vllatent.ingest.acquire — yt-dlp wrapper (mocked subprocess)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from vllatent.ingest.acquire import (
    ClipMetadata,
    download_clip,
    load_clips_yaml,
    probe_clip,
    validate_clip,
)


@pytest.mark.tool
class TestProbeClip:
    def test_parses_json(self) -> None:
        info = {"title": "Test", "duration": 60, "fps": 30, "width": 1920, "height": 1080}
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=json.dumps(info), returncode=0)
            result = probe_clip("https://example.com/video")
        assert result["title"] == "Test"
        assert result["fps"] == 30


@pytest.mark.tool
class TestDownloadClip:
    def test_returns_clip_metadata(self, tmp_path: Path) -> None:
        video = tmp_path / "test_clip.mp4"
        video.write_bytes(b"fake video")
        info = {"title": "T", "duration": 30, "fps": 30, "width": 1280, "height": 720, "id": "test_clip"}

        with patch("subprocess.run") as mock_run, \
             patch("vllatent.ingest.acquire.probe_clip", return_value=info):
            mock_run.return_value = MagicMock(returncode=0)
            meta = download_clip("https://example.com", str(tmp_path), clip_id="test_clip")

        assert isinstance(meta, ClipMetadata)
        assert meta.clip_id == "test_clip"
        assert meta.path == video


@pytest.mark.tool
class TestValidateClip:
    def test_valid_video(self) -> None:
        info = {"streams": [{"codec_type": "video"}]}
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=json.dumps(info), returncode=0)
            assert validate_clip("/fake/path.mp4") is True

    def test_no_video_stream(self) -> None:
        info = {"streams": [{"codec_type": "audio"}]}
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=json.dumps(info), returncode=0)
            assert validate_clip("/fake/path.mp4") is False

    def test_ffprobe_failure(self) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            assert validate_clip("/fake/path.mp4") is False

    def test_timeout(self) -> None:
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("ffprobe", 30)):
            assert validate_clip("/fake/path.mp4") is False


class TestLoadClipsYaml:
    def test_loads_clips(self, tmp_path: Path) -> None:
        p = tmp_path / "clips.yaml"
        p.write_text(yaml.dump({"clips": [{"url": "https://example.com", "clip_id": "c1"}]}))
        clips = load_clips_yaml(p)
        assert len(clips) == 1
        assert clips[0]["clip_id"] == "c1"

    def test_rejects_non_list(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.dump({"clips": "not a list"}))
        with pytest.raises(ValueError, match="must be a list"):
            load_clips_yaml(p)

    def test_empty_yaml(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.yaml"
        p.write_text("")
        clips = load_clips_yaml(p)
        assert clips == []


class TestClipMetadata:
    def test_to_dict(self) -> None:
        m = ClipMetadata(
            path=Path("/a/b.mp4"), url="https://x", title="T",
            duration=30.0, fps=30.0, width=1920, height=1080, clip_id="c1",
        )
        d = m.to_dict()
        assert d["clip_id"] == "c1"
        assert d["path"] == "/a/b.mp4"
