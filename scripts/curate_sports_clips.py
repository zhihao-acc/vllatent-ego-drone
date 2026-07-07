#!/usr/bin/env python
"""Curate real sports-FPV YouTube candidates (B1.22c).

Searches a thoughtful keyword set with yt-dlp, fetches metadata, applies the resolution/fps/
aspect/duration gates + 3-level dedup (vs each other AND the already-curated set), and writes
``configs/sports_clips_candidates.yaml`` for human review → promotion into ``sports_clips.yaml``.

The decision logic lives in ``vllatent.ingest.curate`` (PURE, unit-tested); this script is the
network/orchestration shell. Run on the dev box:

    python scripts/curate_sports_clips.py --max-per-query 15 --out configs/sports_clips_candidates.yaml
    python scripts/curate_sports_clips.py --clip-prefix ski --start-index 16 --out configs/sports_clips_b34a_ski.yaml

Keyword strategy — FOLLOW-CAM ONLY (the deployment view): a drone CHASING/FOLLOWING a skier, so
the followed skier is IN FRAME (from behind/above). This is what the latent world model must learn
to predict — the subject's motion + the environment flowing past. We deliberately EXCLUDE
subject-free egocentric "POV / helmet-cam / first-person" footage (wrong viewpoint: it teaches
terrain-flow with no subject, and the domain-blind predictor would be polluted toward subject-free
predictions) — those terms are in ``CurationGate.reject_title_substrings``. We also avoid
review/tutorial/gear content. Skiing FPV ≈ a drone following a skier, so "FPV"/"cinematic FPV" stay
positive; "POV" is negative.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllatent.ingest.curate import (
    CurationGate,
    candidate_to_entry,
    dedup_candidates,
    gate_candidate,
)

# DRONE-FOLLOW (deployment-matched) + POV/EGO + continuity-biased.
DEFAULT_KEYWORDS = [
    # the deployment view: a drone chasing/following a skier — subject (skier) IN FRAME, from behind
    "FPV drone skiing chase",
    "FPV drone chasing skier",
    "drone chasing skier downhill",
    "drone following skier",
    "cinematic FPV skiing",            # ski FPV films track the athlete (subject in frame)
    "FPV drone ski follow run",
    "chase cam skiing behind skier",
    "freeride skiing drone follow",
    "downhill skier FPV drone",
    "ski follow drone tracking",
]


def _ytdlp_env() -> dict[str, str]:
    """yt-dlp env with the socks:// ALL_PROXY popped (it breaks yt-dlp's requests; use --proxy)."""
    env = dict(os.environ)
    env.pop("ALL_PROXY", None)
    env.pop("all_proxy", None)
    return env


def _run_ytdlp(args: list[str], proxy: str, timeout: int) -> subprocess.CompletedProcess:
    cmd = ["yt-dlp", "--proxy", proxy, "--no-warnings", "--ignore-errors", *args]
    return subprocess.run(cmd, capture_output=True, text=True, env=_ytdlp_env(), timeout=timeout)


def search_keyword(query: str, n: int, proxy: str) -> list[dict]:
    """Flat search → lightweight candidates (id, title, duration, channel)."""
    out = _run_ytdlp(["--flat-playlist", "--dump-json", f"ytsearch{n}:{query}"], proxy, timeout=120)
    cands: list[dict] = []
    for line in out.stdout.splitlines():
        try:
            j = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not j.get("id"):
            continue
        cands.append({
            "id": j["id"],
            "title": j.get("title") or "",
            "duration": j.get("duration"),
            "channel": j.get("channel") or j.get("uploader"),
            "query": query,
        })
    return cands


def fetch_meta(vid: str, proxy: str) -> dict | None:
    """Full metadata for one video → {id,title,duration,height,width,fps,channel,is_live}."""
    out = _run_ytdlp(["--skip-download", "--dump-json", f"https://www.youtube.com/watch?v={vid}"],
                     proxy, timeout=90)
    if out.returncode != 0 or not out.stdout.strip():
        return None
    try:
        j = json.loads(out.stdout)
    except json.JSONDecodeError:
        return None
    formats = j.get("formats") or []
    heights = [f["height"] for f in formats if f.get("height")]
    max_h = max(heights) if heights else int(j.get("height") or 0)
    best = max((f for f in formats if f.get("height") == max_h), key=lambda f: (f.get("fps") or 0),
               default=None)
    fps = (best.get("fps") if best else None) or j.get("fps") or 0
    width = (best.get("width") if best else None) or j.get("width") or round(max_h * 16 / 9)
    return {
        "id": j.get("id"),
        "title": j.get("title") or "",
        "duration": j.get("duration"),
        "height": max_h,
        "width": width,
        "fps": fps,
        "channel": j.get("channel") or j.get("uploader"),
        "is_live": bool(j.get("is_live")),
    }


def _video_id(url: str) -> str | None:
    q = parse_qs(urlparse(url).query)
    return q.get("v", [None])[0]


def load_existing(path: Path) -> tuple[set[str], list[str]]:
    """Existing curated ids + titles (for dedup)."""
    if not path.exists():
        return set(), []
    raw = yaml.safe_load(path.read_text()) or {}
    ids, titles = set(), []
    for c in raw.get("clips", []):
        vid = _video_id(c.get("url", ""))
        if vid:
            ids.add(vid)
        if c.get("notes"):
            titles.append(c["notes"])
    return ids, titles


def main() -> None:
    p = argparse.ArgumentParser(description="Curate ski-FPV YouTube candidates")
    p.add_argument("--out", default="configs/sports_clips_candidates.yaml")
    p.add_argument("--existing", default="configs/sports_clips.yaml")
    p.add_argument("--proxy", default=os.environ.get("YTDLP_PROXY", "http://127.0.0.1:7890"))
    p.add_argument("--max-per-query", type=int, default=15)
    p.add_argument("--max-fetch", type=int, default=90, help="cap full-metadata fetches (time bound)")
    p.add_argument("--sport", default="skiing")
    p.add_argument("--clip-prefix", default="cand", help="prefix for emitted clip IDs; do not include '_'")
    p.add_argument("--start-index", type=int, default=1, help="first numeric suffix for emitted clip IDs")
    p.add_argument("--keywords", nargs="*", default=None, help="override the default keyword set")
    args = p.parse_args()
    if "_" in args.clip_prefix:
        raise SystemExit("--clip-prefix must not contain '_' because source split uses '_' as a separator")
    if args.start_index < 0:
        raise SystemExit("--start-index must be non-negative")

    gate = CurationGate()
    keywords = args.keywords or DEFAULT_KEYWORDS
    existing_ids, existing_titles = load_existing(Path(args.existing))
    print(f"[curate] {len(keywords)} queries, {len(existing_ids)} already-curated ids to exclude")

    raw: list[dict] = []
    for kw in keywords:
        hits = search_keyword(kw, args.max_per_query, args.proxy)
        print(f"[curate]   '{kw}': {len(hits)} hits")
        raw.extend(hits)
    print(f"[curate] {len(raw)} raw hits across queries")

    kept, dropped = dedup_candidates(raw, existing_ids=existing_ids, existing_titles=existing_titles)
    print(f"[curate] after dedup: {len(kept)} unique ({len(dropped)} dropped as dups)")

    accepted: list[dict] = []
    rejected: list[dict] = []
    for c in kept[:args.max_fetch]:
        # cheap title pre-filter — skip the metadata fetch for off-domain / meta titles
        title_l = (c.get("title") or "").lower()
        hit = next((b for b in gate.reject_title_substrings if b in title_l), None)
        if hit:
            rejected.append({**c, "_drop": f"title~{hit.strip()!r}"})
            continue
        meta = fetch_meta(c["id"], args.proxy)
        if meta is None:
            rejected.append({**c, "_drop": "meta-fail"})
            continue
        ok, reasons = gate_candidate(meta, gate)
        meta["query"] = c.get("query")
        if ok:
            accepted.append(meta)
        else:
            rejected.append({**meta, "_drop": "; ".join(reasons)})

    entries = [
        candidate_to_entry(m, f"{args.clip_prefix}{i:02d}", sport=args.sport)
        for i, m in enumerate(accepted, start=args.start_index)
    ]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.dump({"clips": entries}, default_flow_style=False, sort_keys=False))

    print(f"\n[curate] ACCEPTED {len(accepted)} → {out_path}")
    for m in accepted:
        print(f"  + {m['id']}  {m['height']}p/{m['fps']}fps/{int(m.get('duration') or 0)}s  {m['title'][:60]}")
    print(f"\n[curate] REJECTED {len(rejected)} (sample):")
    for r in rejected[:25]:
        print(f"  - {r.get('id')}  {r.get('_drop')}  {(r.get('title') or '')[:50]}")


if __name__ == "__main__":
    main()
