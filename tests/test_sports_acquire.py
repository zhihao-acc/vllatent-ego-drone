"""Tests for vllatent.sports.acquire — yt-dlp wrapper (TOOL tier, mocked for CI)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from vllatent.sports.acquire import (
    ClipMetadata,
    _find_downloaded_file,
    load_clips_yaml,
    validate_clip,
)


class TestClipMetadata:
    def test_to_dict(self) -> None:
        m = ClipMetadata(
            path=Path("/tmp/ski01.mp4"),
            url="https://youtube.com/watch?v=abc",
            title="Ski Run",
            duration=45.0,
            fps=30.0,
            width=1920,
            height=1080,
            clip_id="ski01",
        )
        d = m.to_dict()
        assert d["url"] == "https://youtube.com/watch?v=abc"
        assert d["duration"] == 45.0
        assert d["clip_id"] == "ski01"

    def test_frozen(self) -> None:
        m = ClipMetadata(
            path=Path("/tmp/ski01.mp4"), url="u", title="t",
            duration=1.0, fps=30.0, width=1920, height=1080,
        )
        with pytest.raises(AttributeError):
            m.duration = 99.0  # type: ignore[misc]


class TestFindDownloadedFile:
    def test_finds_mp4(self, tmp_path: Path) -> None:
        (tmp_path / "ski01.mp4").write_bytes(b"fake")
        found = _find_downloaded_file(tmp_path, "ski01")
        assert found.name == "ski01.mp4"

    def test_finds_mkv(self, tmp_path: Path) -> None:
        (tmp_path / "ski01.mkv").write_bytes(b"fake")
        found = _find_downloaded_file(tmp_path, "ski01")
        assert found.name == "ski01.mkv"

    def test_raises_when_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="ski01"):
            _find_downloaded_file(tmp_path, "ski01")

    def test_ignores_non_video(self, tmp_path: Path) -> None:
        (tmp_path / "ski01.json").write_text("{}")
        with pytest.raises(FileNotFoundError):
            _find_downloaded_file(tmp_path, "ski01")


class TestLoadClipsYaml:
    def test_valid_yaml(self, tmp_path: Path) -> None:
        f = tmp_path / "clips.yaml"
        f.write_text('clips:\n  - url: "https://example.com"\n    clip_id: ski01\n')
        clips = load_clips_yaml(f)
        assert len(clips) == 1
        assert clips[0]["url"] == "https://example.com"

    def test_empty_yaml(self, tmp_path: Path) -> None:
        f = tmp_path / "clips.yaml"
        f.write_text("clips: []\n")
        clips = load_clips_yaml(f)
        assert clips == []

    def test_missing_clips_key(self, tmp_path: Path) -> None:
        f = tmp_path / "clips.yaml"
        f.write_text("other: value\n")
        clips = load_clips_yaml(f)
        assert clips == []

    def test_bad_clips_type(self, tmp_path: Path) -> None:
        f = tmp_path / "clips.yaml"
        f.write_text("clips: not_a_list\n")
        with pytest.raises(ValueError, match="clips"):
            load_clips_yaml(f)


class TestValidateClip:
    def test_nonexistent_file(self, tmp_path: Path) -> None:
        assert validate_clip(tmp_path / "nope.mp4") is False

    @patch("vllatent.sports.acquire.subprocess.run")
    def test_valid_video_mock(self, mock_run: object, tmp_path: Path) -> None:
        import subprocess as sp
        from unittest.mock import MagicMock

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '{"streams": [{"codec_type": "video"}]}'
        assert isinstance(mock_run, MagicMock)
        mock_run.return_value = mock_result

        assert validate_clip(tmp_path / "test.mp4") is True

    @patch("vllatent.sports.acquire.subprocess.run")
    def test_invalid_video_mock(self, mock_run: object, tmp_path: Path) -> None:
        from unittest.mock import MagicMock

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        assert isinstance(mock_run, MagicMock)
        mock_run.return_value = mock_result

        assert validate_clip(tmp_path / "test.mp4") is False
