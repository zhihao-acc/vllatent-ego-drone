#!/usr/bin/env python3
"""Create B3 person-label audit montages from cache boxes and frame dirs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vllatent.ingest.person_tracking import PERSON_BBOX_DIM, person_tracks_from_cache


def _frame_paths(frames_root: Path, clip_id: str) -> list[Path]:
    frames_dir = frames_root / clip_id
    if not frames_dir.exists():
        return []
    return sorted(frames_dir.glob("*.jpg")) or sorted(frames_dir.glob("*.png"))


def _crop_box(image_size: tuple[int, int]) -> tuple[int, int, int]:
    width, height = image_size
    crop = min(width, height)
    left = (width - crop) // 2
    top = (height - crop) // 2
    return left, top, crop


def _draw_bbox(draw, bbox: np.ndarray, image_size: tuple[int, int], *, color: str) -> None:
    if bbox.shape != (PERSON_BBOX_DIM,) or bbox[2] <= 0.0 or bbox[3] <= 0.0:
        return
    left, top, crop = _crop_box(image_size)
    cx, cy, bw, bh = [float(v) for v in bbox]
    x1 = left + (cx - 0.5 * bw) * crop
    y1 = top + (cy - 0.5 * bh) * crop
    x2 = left + (cx + 0.5 * bw) * crop
    y2 = top + (cy + 0.5 * bh) * crop
    draw.rectangle([x1, y1, x2, y2], outline=color, width=4)


def _candidate_frame_indices(visible: np.ndarray, state_valid: np.ndarray, n_frames: int, limit: int) -> list[int]:
    valid = np.flatnonzero(state_valid)
    visible_only = np.flatnonzero(visible & ~state_valid)
    empty = np.flatnonzero(~visible)
    ordered = np.concatenate([valid, visible_only, empty])
    if ordered.size == 0:
        return []
    unique = []
    seen = set()
    for i in ordered.tolist():
        if 0 <= i < n_frames and i not in seen:
            unique.append(int(i))
            seen.add(int(i))
        if len(unique) >= limit:
            break
    return unique


def _source_clip_ids(report: dict, source: str, clips_per_source: int) -> list[str]:
    clips = [c for c in report.get("clips", []) if c.get("source") == source]
    clips.sort(
        key=lambda c: (
            -int(c.get("person_valid_windows", 0)),
            -int(c.get("person_trackable_frames", 0)),
            str(c.get("clip_id", "")),
        )
    )
    return [str(c["clip_id"]) for c in clips[:clips_per_source]]


def build_montage(
    *,
    cache_dir: Path,
    frames_root: Path,
    report_path: Path,
    sources: list[str],
    out_dir: Path,
    clips_per_source: int,
    frames_per_clip: int,
    tile_width: int,
) -> list[Path]:
    from PIL import Image, ImageDraw

    report = json.loads(report_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    for source in sources:
        tiles = []
        for clip_id in _source_clip_ids(report, source, clips_per_source):
            cache_path = cache_dir / f"{clip_id}.npz"
            frames = _frame_paths(frames_root, clip_id)
            if not cache_path.exists() or not frames:
                continue
            with np.load(str(cache_path)) as data:
                clip = {k: data[k] for k in data.files}
            tracks = person_tracks_from_cache(clip)
            indices = _candidate_frame_indices(
                tracks.person_visible,
                tracks.person_state_valid,
                min(len(frames), tracks.person_visible.shape[0]),
                frames_per_clip,
            )
            for frame_idx in indices:
                with Image.open(frames[frame_idx]) as img:
                    tile = img.convert("RGB")
                scale = tile_width / float(tile.width)
                tile = tile.resize((tile_width, max(1, int(tile.height * scale))))
                draw = ImageDraw.Draw(tile)
                bbox = tracks.person_bbox[frame_idx]
                color = "lime" if bool(tracks.person_state_valid[frame_idx]) else "red"
                _draw_bbox(draw, bbox, tile.size, color=color)
                label = f"{clip_id} f{frame_idx} vis={int(tracks.person_visible[frame_idx])} state={int(tracks.person_state_valid[frame_idx])}"
                draw.rectangle([0, 0, tile.width, 18], fill="black")
                draw.text((4, 3), label, fill="white")
                tiles.append(tile)
        if not tiles:
            continue
        cols = min(4, len(tiles))
        rows = (len(tiles) + cols - 1) // cols
        tile_h = max(t.height for t in tiles)
        sheet = Image.new("RGB", (cols * tile_width, rows * tile_h), color="white")
        for i, tile in enumerate(tiles):
            x = (i % cols) * tile_width
            y = (i // cols) * tile_h
            sheet.paste(tile, (x, y))
        out_path = out_dir / f"{source}_person_label_montage.jpg"
        sheet.save(out_path, quality=92)
        out_paths.append(out_path)
    return out_paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--clips-per-source", type=int, default=6)
    parser.add_argument("--frames-per-clip", type=int, default=4)
    parser.add_argument("--tile-width", type=int, default=320)
    args = parser.parse_args(argv)

    paths = build_montage(
        cache_dir=Path(args.cache_dir),
        frames_root=Path(args.frames_root),
        report_path=Path(args.report),
        sources=args.sources,
        out_dir=Path(args.out_dir),
        clips_per_source=args.clips_per_source,
        frames_per_clip=args.frames_per_clip,
        tile_width=args.tile_width,
    )
    print(json.dumps({"out_paths": [str(p) for p in paths]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
