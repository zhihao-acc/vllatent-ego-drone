#!/usr/bin/env python3
"""Screen B3 person-track caches for clip/window/source counts and artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vllatent.ingest.person_tracking import screen_cache_dir
from vllatent.schemas import HISTORY, HORIZON


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, help="Directory containing .npz caches")
    parser.add_argument("--history", type=int, default=HISTORY)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None, help="Optional JSON report path")
    args = parser.parse_args(argv)

    report = screen_cache_dir(
        args.cache_dir,
        history=args.history,
        horizon=args.horizon,
        limit=args.limit,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
