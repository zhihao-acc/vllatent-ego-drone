#!/usr/bin/env python
"""Data quality report for sports FPV cache (B1.9).

Reads a manifest + .npz files from a cache directory and produces JSON +
terminal report: per-clip frame count, quality distribution, VO confidence
distribution, delta magnitude stats, and MegaSaM confidence_source breakdown.

Usage:
    python scripts/data_quality_report.py --cache /path/to/cache_dir
    python scripts/data_quality_report.py --cache /path/to/cache_dir --json report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _percentiles(arr: np.ndarray) -> dict[str, float]:
    if len(arr) == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "p5": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def load_manifest(cache_dir: Path) -> dict | None:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    with open(manifest_path) as f:
        return json.load(f)


def analyze_npz(path: Path) -> dict:
    data = dict(np.load(path, allow_pickle=False))
    result: dict = {"path": str(path.name), "keys": sorted(data.keys())}

    if "latents" in data:
        result["n_frames"] = int(data["latents"].shape[0])
    elif "z_t" in data:
        result["n_frames"] = 1
    else:
        result["n_frames"] = 0

    if "deltas" in data:
        deltas = data["deltas"]
        magnitudes = np.linalg.norm(deltas[:, :3], axis=1).astype(float)
        result["delta_stats"] = _percentiles(magnitudes)
        result["delta_stats"]["total_steps"] = int(len(magnitudes))
    elif "waypoint_4dof" in data:
        wp = data["waypoint_4dof"]
        magnitudes = np.linalg.norm(wp[:, :3], axis=1).astype(float)
        result["delta_stats"] = _percentiles(magnitudes)
        result["delta_stats"]["total_steps"] = int(len(magnitudes))

    if "vo_confidence" in data:
        result["vo_confidence"] = _percentiles(data["vo_confidence"].astype(float))
    if "frame_quality" in data:
        result["frame_quality"] = _percentiles(data["frame_quality"].astype(float))
    if "vjepa_surprise" in data:
        result["vjepa_surprise"] = _percentiles(data["vjepa_surprise"].astype(float))

    return result


def build_report(cache_dir: Path) -> dict:
    manifest = load_manifest(cache_dir)

    npz_files = sorted(cache_dir.glob("*.npz"))
    clips = [analyze_npz(p) for p in npz_files]

    all_frame_counts = np.array([c["n_frames"] for c in clips], dtype=np.int64)

    all_deltas = []
    all_vo_conf = []
    all_quality = []
    for c in clips:
        if "delta_stats" in c:
            npz = np.load(cache_dir / c["path"], allow_pickle=False)
            key = "deltas" if "deltas" in npz else "waypoint_4dof"
            if key in npz:
                all_deltas.append(npz[key])
        if "vo_confidence" in c:
            npz = np.load(cache_dir / c["path"], allow_pickle=False)
            if "vo_confidence" in npz:
                all_vo_conf.append(npz["vo_confidence"])
        if "frame_quality" in c:
            npz = np.load(cache_dir / c["path"], allow_pickle=False)
            if "frame_quality" in npz:
                all_quality.append(npz["frame_quality"])

    delta_magnitudes = np.array([], dtype=np.float64)
    if all_deltas:
        combined = np.concatenate(all_deltas, axis=0)
        delta_magnitudes = np.linalg.norm(combined[:, :3], axis=1).astype(float)

    vo_confs = np.concatenate(all_vo_conf).astype(float) if all_vo_conf else np.array([])
    qualities = np.concatenate(all_quality).astype(float) if all_quality else np.array([])

    report: dict = {
        "cache_dir": str(cache_dir),
        "n_clips": len(clips),
        "total_frames": int(np.sum(all_frame_counts)) if len(all_frame_counts) > 0 else 0,
        "frame_counts": _percentiles(all_frame_counts.astype(float)),
        "delta_magnitude": _percentiles(delta_magnitudes),
        "vo_confidence": _percentiles(vo_confs),
        "frame_quality": _percentiles(qualities),
        "per_clip": clips,
    }

    if manifest is not None:
        report["manifest_present"] = True
        report["encoder_model"] = manifest.get("encoder", {}).get("model_id", "unknown")
        motion = manifest.get("motion_source", {})
        report["motion_method"] = motion.get("method", "unknown")
        entries = manifest.get("entries", [])
        report["manifest_entries"] = len(entries)
    else:
        report["manifest_present"] = False

    return report


def print_report(report: dict) -> None:
    print("=" * 60)
    print("  DATA QUALITY REPORT")
    print("=" * 60)
    print(f"  Cache dir:       {report['cache_dir']}")
    print(f"  Clips:           {report['n_clips']}")
    print(f"  Total frames:    {report['total_frames']}")
    print(f"  Manifest:        {'yes' if report['manifest_present'] else 'NO'}")
    if report["manifest_present"]:
        print(f"  Encoder:         {report.get('encoder_model', '?')}")
        print(f"  Motion method:   {report.get('motion_method', '?')}")
        print(f"  Manifest entries:{report.get('manifest_entries', 0)}")
    print()

    def _show_dist(label: str, d: dict) -> None:
        if d["max"] == 0.0 and d["min"] == 0.0:
            print(f"  {label:20s}  (no data)")
            return
        print(
            f"  {label:20s}  "
            f"min={d['min']:.4f}  p5={d['p5']:.4f}  "
            f"p50={d['p50']:.4f}  mean={d['mean']:.4f}  "
            f"p95={d['p95']:.4f}  max={d['max']:.4f}"
        )

    print("  AGGREGATE DISTRIBUTIONS")
    print("  " + "-" * 56)
    _show_dist("Frame counts", report["frame_counts"])
    _show_dist("Delta magnitude", report["delta_magnitude"])
    _show_dist("VO confidence", report["vo_confidence"])
    _show_dist("Frame quality", report["frame_quality"])
    print()

    if report["per_clip"]:
        print(f"  PER-CLIP SUMMARY ({len(report['per_clip'])} clips)")
        print("  " + "-" * 56)
        for c in report["per_clip"]:
            line = f"  {c['path']:30s}  frames={c['n_frames']}"
            if "delta_stats" in c:
                ds = c["delta_stats"]
                line += f"  delta_mean={ds['mean']:.4f}"
            if "vo_confidence" in c:
                vc = c["vo_confidence"]
                line += f"  vo_conf_mean={vc['mean']:.4f}"
            print(line)

    print()
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Data quality report for sports FPV cache")
    parser.add_argument("--cache", required=True, type=Path, help="Path to cache directory")
    parser.add_argument("--json", type=Path, default=None, help="Write JSON report to file")
    args = parser.parse_args(argv)

    if not args.cache.is_dir():
        print(f"Error: {args.cache} is not a directory", file=sys.stderr)
        return 1

    report = build_report(args.cache)
    print_report(report)

    if args.json is not None:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  JSON report written to {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
