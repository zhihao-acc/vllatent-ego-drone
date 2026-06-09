"""``python -m vllatent.data`` CLI (TORCH tier). Subcommand ``inspect`` dumps the first N
distillation samples of a latent cache — A5.16 runs it over the real teacher/oracle dump.
"""
from __future__ import annotations

import argparse

from vllatent.data.loader import inspect_cache


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m vllatent.data", description="latent-cache tools")
    sub = parser.add_subparsers(dest="cmd", required=True)
    insp = sub.add_parser("inspect", help="print the first --n distillation samples of a cache")
    insp.add_argument("--cache", required=True, help="cache dir (holds manifest.json + per-episode .npz)")
    insp.add_argument("--n", type=int, default=4, help="number of samples to print (default 4)")
    args = parser.parse_args(argv)
    if args.cmd == "inspect":
        return inspect_cache(args.cache, args.n)
    parser.error(f"unknown command {args.cmd!r}")  # NoReturn


if __name__ == "__main__":
    raise SystemExit(_main())
