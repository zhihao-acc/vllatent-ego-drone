#!/usr/bin/env python3
"""Convert cached B3 person boxes from raw-frame coords to DINO encoder-crop coords."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vllatent.ingest.person_tracking import (
    PERSON_BBOX_KEY,
    PERSON_BBOX_SPACE_ENCODER_CROP,
    PERSON_BBOX_SPACE_KEY,
    PERSON_BBOX_SPACE_RAW_FRAME,
    PERSON_CONF_KEY,
    PERSON_VISIBLE_KEY,
    raw_frame_cxcywh_to_encoder_crop,
    sanitize_person_track_arrays,
)


def _bbox_space(arrays: dict[str, np.ndarray]) -> str:
    if PERSON_BBOX_SPACE_KEY not in arrays:
        return PERSON_BBOX_SPACE_RAW_FRAME
    value = arrays[PERSON_BBOX_SPACE_KEY]
    if isinstance(value, np.ndarray):
        return str(value.tolist())
    return str(value)


def _first_frame_size(frames_dir: Path, clip_id: str) -> tuple[int, int] | None:
    from PIL import Image

    clip_dir = frames_dir / clip_id
    if not clip_dir.exists():
        return None
    frame = next(iter(sorted(clip_dir.glob("*.jpg"))), None)
    if frame is None:
        frame = next(iter(sorted(clip_dir.glob("*.png"))), None)
    if frame is None:
        return None
    with Image.open(frame) as img:
        w, h = img.size
    return int(h), int(w)


def convert_one(path: Path, frames_dir: Path, *, dry_run: bool = False) -> dict[str, object]:
    with np.load(str(path), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    record: dict[str, object] = {
        "clip_id": path.stem,
        "status": "unknown",
        "previous_space": _bbox_space(arrays),
    }
    if PERSON_BBOX_KEY not in arrays or PERSON_VISIBLE_KEY not in arrays:
        record["status"] = "missing_person_keys"
        return record
    if record["previous_space"] == PERSON_BBOX_SPACE_ENCODER_CROP:
        record["status"] = "skipped_existing"
        return record
    image_hw = _first_frame_size(frames_dir, path.stem)
    if image_hw is None:
        record["status"] = "missing_frames"
        return record

    visible = np.asarray(arrays[PERSON_VISIBLE_KEY]).astype(np.bool_)
    raw_visible = int(np.sum(visible))
    conf = np.asarray(arrays.get(PERSON_CONF_KEY, np.zeros_like(visible, dtype=np.float32)), dtype=np.float32)
    converted = raw_frame_cxcywh_to_encoder_crop(np.asarray(arrays[PERSON_BBOX_KEY], dtype=np.float32), image_hw)
    converted, visible, conf = sanitize_person_track_arrays(
        person_bbox=converted,
        person_visible=visible,
        person_conf=conf,
    )
    if dry_run:
        record["status"] = "would_convert"
    else:
        arrays[PERSON_BBOX_KEY] = converted.astype(np.float32)
        arrays[PERSON_VISIBLE_KEY] = visible
        if PERSON_CONF_KEY in arrays:
            arrays[PERSON_CONF_KEY] = conf
        arrays[PERSON_BBOX_SPACE_KEY] = np.array(PERSON_BBOX_SPACE_ENCODER_CROP)
        arrays["person_bbox_previous_space"] = np.array(record["previous_space"])
        arrays["person_bbox_image_hw"] = np.array(image_hw, dtype=np.int32)
        np.savez(str(path), **arrays)
        record["status"] = "converted"
    record["image_hw"] = list(image_hw)
    record["visible_frames"] = int(visible.sum())
    record["sanitized_invisible_frames"] = raw_visible - int(visible.sum())
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    paths = sorted(Path(args.cache_dir).glob("*.npz"))
    if args.limit is not None:
        paths = paths[:args.limit]
    records = [convert_one(p, Path(args.frames_dir), dry_run=args.dry_run) for p in paths]
    summary: dict[str, int] = {}
    for rec in records:
        status = str(rec["status"])
        summary[status] = summary.get(status, 0) + 1
    payload = {"summary": summary, "records": records}
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
    print(text)
    return 0 if summary.get("missing_frames", 0) == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
