#!/usr/bin/env python
"""Generate per-clip HTML quality report from cached .npz files.

Usage:
    python scripts/clip_report.py --cache data/sports_cache/ --clip ski_01
    python scripts/clip_report.py --npz data/sports_cache/ski_01.npz --out report.html
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Per-clip Plotly HTML quality report (B1.9b)")
    parser.add_argument("--cache", type=str, help="Cache directory containing .npz files")
    parser.add_argument("--clip", type=str, help="Clip ID (stem of .npz file)")
    parser.add_argument("--npz", type=str, help="Direct path to .npz file (overrides --cache/--clip)")
    parser.add_argument("--out", type=str, default=None, help="Output HTML path (default: <clip_id>.html)")
    args = parser.parse_args(argv)

    if args.npz:
        npz_path = Path(args.npz)
    elif args.cache and args.clip:
        npz_path = Path(args.cache) / f"{args.clip}.npz"
    else:
        parser.error("Provide --npz, or both --cache and --clip")
        return 1

    if not npz_path.exists():
        print(f"[clip-report] ERROR: {npz_path} not found", file=sys.stderr)
        return 1

    clip_id = args.clip or npz_path.stem
    out_path = Path(args.out) if args.out else Path(f"{clip_id}_report.html")

    from vllatent.ingest.visualize import generate_clip_report

    html = generate_clip_report(npz_path, clip_id=clip_id, out_path=out_path)
    print(f"[clip-report] wrote {out_path} ({len(html)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
