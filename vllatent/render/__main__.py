"""``python -m vllatent.render`` — live render smoke (SIM tier, USER-GATED).

Parses an AerialVLN episode, teleports to each ``reference_path`` pose, captures the Scene RGB, and
saves ``<out>/<episode_id>_<t>.npy`` (uint8 RGB). RUNNING needs the fly0-m1 docker + a UE4 scene hot
on port 41451 — the agent emits the command; the user launches the sim. No blobs are committed.

  python -m vllatent.render --episode fixtures/episodes/tiny_episode.json --scene 1 --out /tmp/render_smoke/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _main(argv: list[str] | None = None) -> int:  # pragma: no cover - USER-GATED (needs the live sim)
    import numpy as np

    from vllatent.audit import parse_episode
    from vllatent.render.harness import RenderHarness

    parser = argparse.ArgumentParser(prog="python -m vllatent.render", description="live render smoke")
    parser.add_argument("--episode", required=True, help="AerialVLN episode JSON")
    parser.add_argument("--scene", type=int, default=1, help="scene id (the user launches this UE4 scene)")
    parser.add_argument("--out", required=True, help="output dir for per-pose .npy RGB frames")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41451)
    args = parser.parse_args(argv)

    episode = parse_episode(json.loads(Path(args.episode).read_text()))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    harness = RenderHarness(host=args.host, port=args.port)
    n = 0
    for t, row in enumerate(episode.reference_path):
        rgb = harness.render_reference_row(np.asarray(row))
        np.save(out / f"{episode.episode_id}_{t:04d}.npy", rgb)
        n += 1
    print(f"[render] scene={args.scene} episode={episode.episode_id}: {n} RGB frames {rgb.shape} -> {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - USER-GATED
    raise SystemExit(_main())
