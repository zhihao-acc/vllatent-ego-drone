"""CLI for sports-following dataset pipeline.

Usage:
    python -m vllatent.sports process --url URL --clip-id ID [--config PATH] [--device DEVICE]
    python -m vllatent.sports batch --clips PATH [--config PATH] [--device DEVICE]
    python -m vllatent.sports inspect --cache DIR [--n N]
"""
from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m vllatent.sports",
        description="Sports-following dataset pipeline",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # --- process ---
    proc = sub.add_parser("process", help="Process a single clip")
    proc.add_argument("--url", required=True, help="Video URL or local path")
    proc.add_argument("--clip-id", required=True, help="Unique clip identifier")
    proc.add_argument("--config", default="configs/sports.yaml", help="Config YAML")
    proc.add_argument("--device", default="cuda", help="Torch device")
    proc.add_argument("--skip-download", action="store_true")
    proc.add_argument("--skip-megasam", action="store_true")

    # --- batch ---
    bat = sub.add_parser("batch", help="Process all clips in a YAML list")
    bat.add_argument("--clips", required=True, help="Clips YAML path")
    bat.add_argument("--config", default="configs/sports.yaml", help="Config YAML")
    bat.add_argument("--device", default="cuda", help="Torch device")
    bat.add_argument("--no-skip-existing", action="store_true")

    # --- inspect ---
    ins = sub.add_parser("inspect", help="Inspect a sports latent cache")
    ins.add_argument("--cache", required=True, help="Cache directory")
    ins.add_argument("--n", type=int, default=5, help="Number of samples to show")

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "inspect":
        from vllatent.sports.loader import inspect_sports_cache
        return inspect_sports_cache(args.cache, n=args.n)

    from vllatent.config import Config
    from vllatent.sports.pipeline import (
        process_batch,
        process_clip,
        update_manifest_from_results,
    )

    config = Config.from_yaml(args.config)
    if config.sports is None:
        print(f"ERROR: {args.config} has no 'sports' section", file=sys.stderr)
        return 1
    cfg = config.sports

    if args.command == "process":
        result = process_clip(
            url=args.url,
            clip_id=args.clip_id,
            cfg=cfg,
            skip_download=args.skip_download,
            skip_megasam=args.skip_megasam,
            device=args.device,
        )
        if result.errors:
            for e in result.errors:
                print(f"ERROR: {e}", file=sys.stderr)
            return 1
        update_manifest_from_results([result], cfg)
        print(f"OK: {result.clip_id} → {result.n_accepted}/{result.n_frames} frames")
        return 0

    if args.command == "batch":
        results = process_batch(
            clips_yaml=args.clips,
            cfg=cfg,
            skip_existing=not args.no_skip_existing,
            device=args.device,
        )
        update_manifest_from_results(results, cfg)
        ok = sum(1 for r in results if not r.errors)
        fail = sum(1 for r in results if r.errors)
        total_frames = sum(r.n_accepted for r in results if not r.errors)
        print(f"batch complete: {ok} ok, {fail} failed, {total_frames} total accepted frames")
        return 1 if fail > 0 else 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
