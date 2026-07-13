#!/usr/bin/env python3
"""Backfill B3 person-track labels into cached sports .npz files.

Full-cache execution is user-gated. Use ``--dry-run`` locally to inspect which
clips have matching frame directories and which caches already contain labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from vllatent.ingest.person_tracking import (
    PERSON_BBOX_KEY,
    PERSON_BBOX_SPACE_ENCODER_CROP,
    PERSON_BBOX_SPACE_KEY,
    PERSON_CONF_KEY,
    PERSON_SECOND_BEST_TRACK_ID_KEY,
    PERSON_SELECTED_TRACK_ID_KEY,
    PERSON_STATE_VALID_KEY,
    PERSON_SUBJECT_AMBIGUITY_MARGIN_KEY,
    PERSON_SUBJECT_IS_AMBIGUOUS_KEY,
    PERSON_TRACK_CLASSES,
    PERSON_TRACKER_ID,
    PERSON_VISIBLE_KEY,
    track_persons_from_paths,
)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(str(path)) as data:
        return {k: data[k] for k in data.files}


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez(str(path), **arrays)


def _frame_paths(frames_root: Path, clip_id: str) -> list[Path]:
    frames_dir = frames_root / clip_id
    return sorted(frames_dir.glob("*.jpg"))


def _has_person_keys(arrays: dict[str, np.ndarray]) -> bool:
    return (
        PERSON_BBOX_KEY in arrays
        and PERSON_VISIBLE_KEY in arrays
        and PERSON_CONF_KEY in arrays
    )


def backfill_one(
    cache_path: Path,
    *,
    frames_root: Path,
    device: str,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    clip_id = cache_path.stem
    arrays = _load_npz(cache_path)
    frames = _frame_paths(frames_root, clip_id)
    has_person = _has_person_keys(arrays)
    record: dict[str, Any] = {
        "clip_id": clip_id,
        "cache_path": str(cache_path),
        "n_frames_cache": int(arrays["latents"].shape[0]),
        "n_frames_dir": len(frames),
        "had_person_keys": has_person,
        "dry_run": dry_run,
        "status": "pending",
        "person_tracker": {
            "detector": PERSON_TRACKER_ID,
            "tracker": "bytetrack",
            "classes": list(PERSON_TRACK_CLASSES),
        },
    }
    if has_person and not overwrite:
        record["status"] = "skipped_existing"
        return record
    if not frames:
        record["status"] = "missing_frames"
        return record
    if len(frames) != arrays["latents"].shape[0]:
        record["status"] = "frame_count_mismatch"
        return record
    if dry_run:
        record["status"] = "would_backfill"
        return record

    tracks = track_persons_from_paths(frames, device=device)
    arrays[PERSON_BBOX_KEY] = tracks.person_bbox
    arrays[PERSON_BBOX_SPACE_KEY] = np.array(PERSON_BBOX_SPACE_ENCODER_CROP)
    arrays[PERSON_VISIBLE_KEY] = tracks.person_visible
    arrays[PERSON_STATE_VALID_KEY] = tracks.person_state_valid
    arrays[PERSON_CONF_KEY] = tracks.person_conf
    arrays[PERSON_SELECTED_TRACK_ID_KEY] = np.array(tracks.selected_track_id, dtype=np.int64)
    arrays[PERSON_SECOND_BEST_TRACK_ID_KEY] = np.array(tracks.second_best_track_id, dtype=np.int64)
    arrays[PERSON_SUBJECT_AMBIGUITY_MARGIN_KEY] = np.array(
        tracks.subject_ambiguity_margin, dtype=np.float32
    )
    arrays[PERSON_SUBJECT_IS_AMBIGUOUS_KEY] = np.array(tracks.subject_is_ambiguous, dtype=np.bool_)
    _write_npz(cache_path, arrays)
    record["status"] = "backfilled"
    record["person_visible_frames"] = int(np.sum(tracks.person_visible))
    record["person_tracker"] = tracks.provenance
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, help="Directory containing .npz caches")
    parser.add_argument("--frames-root", required=True, help="Directory containing per-clip frame dirs")
    parser.add_argument("--device", default="cuda", help="YOLO/ByteTrack device for real backfill")
    parser.add_argument("--limit", type=int, default=None, help="Maximum cache files to inspect")
    parser.add_argument("--dry-run", action="store_true", help="Do not modify .npz files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing person-track labels")
    parser.add_argument("--log-jsonl", default=None, help="Optional backfill log path")
    args = parser.parse_args(argv)

    cache_dir = Path(args.cache_dir)
    frames_root = Path(args.frames_root)
    paths = sorted(cache_dir.glob("*.npz"))
    if args.limit is not None:
        paths = paths[: args.limit]

    records = [
        backfill_one(
            p,
            frames_root=frames_root,
            device=args.device,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
        for p in paths
    ]

    if args.log_jsonl:
        out = Path(args.log_jsonl)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n")

    counts: dict[str, int] = {}
    for r in records:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(json.dumps({"n_files": len(records), "status_counts": counts}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
