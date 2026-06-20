#!/usr/bin/env python3
"""Curate FPV drone-skiing clips from YouTube via yt-dlp search.

Searches multiple queries, probes metadata, filters by duration/resolution/title,
dedupes, and writes configs/sports_clips.yaml.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

QUERIES = [
    "FPV drone follow skiing",
    "drone chase skier POV",
    "FPV skiing follow cam",
    "drone tracking skier downhill",
    "skiing drone follow behind",
    "FPV drone ski backcountry",
    "drone following skier powder",
    "ski drone chase cam FPV",
]

REJECT_TITLE_PATTERNS = re.compile(
    r"(compilation|top\s*\d+|best\s*of|react|review|unbox|setup|tutorial|how\s+to"
    r"|indoor|simulator|game|virtual|gopro\s*max|360\s*cam|edit\s*breakdown)",
    re.IGNORECASE,
)

MIN_DURATION_S = 45
MAX_DURATION_S = 600
MIN_HEIGHT = 640
TARGET_COUNT = 15
RESULTS_PER_QUERY = 8


def search_youtube_flat(query: str, n: int) -> list[dict]:
    """Fast flat search — returns ids and titles but no resolution."""
    cmd = [
        "yt-dlp",
        f"ytsearch{n}:{query}",
        "--dump-json",
        "--no-download",
        "--flat-playlist",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=90,
        )
    except subprocess.TimeoutExpired:
        print(f"  [timeout] {query}", file=sys.stderr)
        return []

    entries = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def probe_full(vid_id: str) -> dict | None:
    """Full probe for a single video — gets best-format resolution, fps, duration."""
    url = f"https://www.youtube.com/watch?v={vid_id}"
    cmd = [
        "yt-dlp", "--dump-json", "--no-download",
        "--format", "bestvideo[height>=720]+bestaudio/best[height>=720]/bestvideo+bestaudio/best",
        url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
        if result.returncode != 0:
            return None
        info = json.loads(result.stdout)
        if not info.get("height"):
            best_h = 0
            for fmt in info.get("formats", []):
                h = fmt.get("height") or 0
                if h > best_h:
                    best_h = h
                    info["height"] = h
                    info["fps"] = fmt.get("fps") or info.get("fps", 30)
        return info
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        return None


def passes_flat_filter(info: dict) -> tuple[bool, str]:
    """Quick filter on flat-playlist metadata (title + duration only)."""
    title = info.get("title", "")
    if REJECT_TITLE_PATTERNS.search(title):
        return False, "rejected title pattern"

    duration = info.get("duration") or 0
    if duration and duration < MIN_DURATION_S:
        return False, f"too short ({duration}s)"
    if duration and duration > MAX_DURATION_S:
        return False, f"too long ({duration}s)"

    if info.get("is_live"):
        return False, "live stream"

    return True, "ok"


def passes_filter(info: dict) -> tuple[bool, str]:
    title = info.get("title", "")
    if REJECT_TITLE_PATTERNS.search(title):
        return False, "rejected title pattern"

    duration = info.get("duration") or 0
    if duration < MIN_DURATION_S:
        return False, f"too short ({duration}s)"
    if duration > MAX_DURATION_S:
        return False, f"too long ({duration}s)"

    height = info.get("height") or 0
    if height < MIN_HEIGHT:
        return False, f"low res ({height}p)"

    if info.get("is_live"):
        return False, "live stream"

    return True, "ok"


def make_clip_id(index: int) -> str:
    return f"ski{index:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default="configs/sports_clips.yaml",
        help="Output YAML path (default: configs/sports_clips.yaml)",
    )
    parser.add_argument(
        "--target", type=int, default=TARGET_COUNT,
        help=f"Target number of clips (default: {TARGET_COUNT})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print but don't write")
    args = parser.parse_args()

    seen_ids: set[str] = set()
    to_probe: list[dict] = []

    for query in QUERIES:
        if len(to_probe) >= args.target * 2:
            break
        print(f"[search] '{query}' ...", file=sys.stderr)
        entries = search_youtube_flat(query, RESULTS_PER_QUERY)
        print(f"  got {len(entries)} flat results", file=sys.stderr)

        for entry in entries:
            vid_id = entry.get("id", "")
            if not vid_id or vid_id in seen_ids:
                continue
            seen_ids.add(vid_id)

            ok, reason = passes_flat_filter(entry)
            if not ok:
                print(f"  FLAT-SKIP {vid_id}: {reason}", file=sys.stderr)
                continue

            to_probe.append(entry)
            print(f"  QUEUED {vid_id}: {entry.get('title', '?')}", file=sys.stderr)

    print(f"\n[probe] {len(to_probe)} candidates to full-probe ...", file=sys.stderr)
    candidates: list[dict] = []

    for entry in to_probe:
        if len(candidates) >= args.target:
            break
        vid_id = entry.get("id", "")
        print(f"  probing {vid_id} ...", file=sys.stderr, end=" ")
        info = probe_full(vid_id)
        if info is None:
            print("FAIL (probe error)", file=sys.stderr)
            continue

        ok, reason = passes_filter(info)
        if not ok:
            print(f"SKIP ({reason})", file=sys.stderr)
            continue

        title = info.get("title", "unknown")
        duration = info.get("duration", 0)
        height = info.get("height") or 0
        fps = info.get("fps") or 0
        url = info.get("webpage_url") or f"https://www.youtube.com/watch?v={vid_id}"

        print(f"ACCEPT ({duration:.0f}s, {height}p, {fps}fps)", file=sys.stderr)
        candidates.append({
            "vid_id": vid_id,
            "url": url,
            "title": title,
            "duration": duration,
            "height": height,
            "fps": fps,
        })

    clips = []
    for i, c in enumerate(candidates, start=1):
        clips.append({
            "url": c["url"],
            "clip_id": make_clip_id(i),
            "sport": "skiing",
            "notes": f"{c['title']} ({c['duration']:.0f}s, {c['height']}p, {c['fps']}fps)",
        })

    out_data = {
        "clips": clips,
    }

    yaml_str = yaml.dump(out_data, default_flow_style=False, allow_unicode=True, sort_keys=False)

    if args.dry_run:
        print(yaml_str)
    else:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(yaml_str)
        print(f"\nWrote {len(clips)} clips to {out_path}", file=sys.stderr)

    print(f"\n=== Summary: {len(clips)} clips curated ===", file=sys.stderr)
    for c in clips:
        print(f"  {c['clip_id']}: {c['notes']}", file=sys.stderr)


if __name__ == "__main__":
    main()
