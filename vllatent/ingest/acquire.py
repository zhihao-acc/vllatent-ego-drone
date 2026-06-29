"""Video acquisition via yt-dlp (TOOL tier).

Subprocess wrapper for yt-dlp. No yt-dlp Python API import — subprocess only,
which avoids version coupling and is more robust for long-running downloads.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ClipMetadata:
    """Metadata for a downloaded video clip."""

    path: Path
    url: str
    title: str
    duration: float
    fps: float
    width: int
    height: int
    clip_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "url": self.url,
            "title": self.title,
            "duration": self.duration,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "clip_id": self.clip_id,
        }


def probe_clip(url: str) -> dict[str, Any]:
    """Fetch video metadata without downloading via ``yt-dlp --dump-json``."""
    result = subprocess.run(
        ["yt-dlp", "--dump-json", "--no-download", url],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return json.loads(result.stdout)


def download_clip(
    url: str,
    out_dir: str | Path,
    *,
    clip_id: str = "",
    max_height: int = 1080,
    sponsorblock: bool = False,
) -> ClipMetadata:
    """Download a single video clip via yt-dlp.

    When ``sponsorblock=True``, strips sponsor/intro/outro segments via SponsorBlock
    crowdsourced data (``--sponsorblock-remove all``).
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    filename = clip_id if clip_id else "%(id)s"
    output_template = str(out_path / f"{filename}.%(ext)s")

    format_spec = f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]/best"

    cmd = [
        "yt-dlp",
        "--format", format_spec,
        "--merge-output-format", "mp4",
        "--output", output_template,
        "--no-playlist",
        "--write-info-json",
    ]
    # Proxy: yt-dlp chokes on a socks:// ALL_PROXY (needs socks5://). Pass an explicit --proxy
    # from the http(s) proxy env and pop ALL_PROXY in the subprocess so it can't interfere.
    proxy = (
        os.environ.get("YTDLP_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("HTTPS_PROXY")
    )
    if proxy:
        cmd.extend(["--proxy", proxy])
    if sponsorblock:
        cmd.extend(["--sponsorblock-remove", "all"])
    cmd.append(url)

    env = dict(os.environ)
    env.pop("ALL_PROXY", None)
    env.pop("all_proxy", None)
    subprocess.run(cmd, check=True, timeout=600, env=env)

    info = probe_clip(url)

    video_path = _find_downloaded_file(out_path, clip_id or info.get("id", "unknown"))

    return ClipMetadata(
        path=video_path,
        url=url,
        title=str(info.get("title", "")),
        duration=float(info.get("duration", 0)),
        fps=float(info.get("fps", 30)),
        width=int(info.get("width", 0)),
        height=int(info.get("height", 0)),
        clip_id=clip_id,
    )


def _find_downloaded_file(out_dir: Path, stem: str) -> Path:
    """Find the downloaded video file by stem name."""
    for ext in (".mp4", ".mkv", ".webm", ".avi"):
        candidate = out_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    mp4s = list(out_dir.glob(f"{stem}.*"))
    video_exts = {".mp4", ".mkv", ".webm", ".avi", ".mov"}
    for p in mp4s:
        if p.suffix in video_exts:
            return p
    raise FileNotFoundError(f"No video file found for stem {stem!r} in {out_dir}")


def validate_clip(path: str | Path) -> bool:
    """Check if a file is a valid video via ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_type",
                "-of", "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return False
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        return any(s.get("codec_type") == "video" for s in streams)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        return False


def load_clips_yaml(path: str | Path) -> list[dict[str, Any]]:
    """Load a clips YAML file, returning the list of clip entries."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    clips = raw.get("clips", [])
    if not isinstance(clips, list):
        raise ValueError(f"clips_yaml: 'clips' must be a list, got {type(clips).__name__}")
    return clips


def download_batch(
    clips_yaml: str | Path,
    out_dir: str | Path,
    *,
    max_height: int = 1080,
    skip_existing: bool = True,
) -> list[ClipMetadata]:
    """Download all clips from a YAML clip list, skipping already-downloaded ones."""
    clips = load_clips_yaml(clips_yaml)
    out_path = Path(out_dir)
    results: list[ClipMetadata] = []

    for entry in clips:
        url = entry.get("url", "")
        clip_id = entry.get("clip_id", "")
        if not url:
            continue

        if skip_existing and clip_id:
            existing = list(out_path.glob(f"{clip_id}.*"))
            video_exts = {".mp4", ".mkv", ".webm", ".avi", ".mov"}
            if any(p.suffix in video_exts for p in existing):
                continue

        meta = download_clip(url, out_dir, clip_id=clip_id, max_height=max_height)
        results.append(meta)

    return results


__all__ = [
    "ClipMetadata",
    "probe_clip",
    "download_clip",
    "validate_clip",
    "load_clips_yaml",
    "download_batch",
]
