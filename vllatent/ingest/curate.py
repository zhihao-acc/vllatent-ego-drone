"""YouTube clip curation — quality gates + 3-level dedup + candidate emission (PURE tier).

The download/metadata side (yt-dlp) lives in ``scripts/curate_sports_clips.py``; this module is the
pure, testable decision logic: which candidates pass the resolution/fps/aspect/duration gates, and
which are duplicates (of each other or of the already-curated set). stdlib only — no torch, no yt-dlp.

A *candidate* is a plain dict of yt-dlp metadata:
  ``{"id", "title", "duration" (s), "height", "width", "fps", "channel", "is_live"}``.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CurationGate:
    """Source-video acceptance thresholds (the sub-clip cutting happens later in the pipeline)."""

    min_duration_s: float = 30.0     # need room for >=1 FPV sub-clip after filtering
    max_duration_s: float = 1200.0   # 20 min — bound download size; longer = huge files
    min_height: int = 720            # matches the 720p decision (reject sub-HD)
    min_fps: float = 23.0            # reject choppy/timelapse; 24/25/30/50/60 all pass
    min_aspect: float = 1.30         # reject vertical / near-square (Shorts)
    max_aspect: float = 2.10         # reject ultra-wide / letterboxed oddities
    reject_live: bool = True
    # Cheap title pre-filter (case-insensitive substring) — drops off-domain "ski" homographs
    # (jet/water ski, snowmobile) and meta/talking-head content (reviews, tutorials, gear guides)
    # BEFORE we spend download+MegaSaM+encode on them. YOLO-World per-frame filter is the 2nd line.
    reject_title_substrings: tuple[str, ...] = (
        # off-domain "ski" homographs (water, not snow) + wrong subject
        "jet ski", "jetski", "jet-ski", "water ski", "waterski", "wakeboard", "snowmobile",
        # subject-FREE egocentric (helmet/chest/first-person) — wrong viewpoint for a follow drone:
        # the model must see the FOLLOWED skier in frame, not the skier's own POV.
        "pov", "point of view", "first-person", "first person view", "helmet cam", "helmet camera",
        "chest cam", "head cam", "gopro line",
        # meta / talking-head / instructional (no continuous follow footage)
        " vs ", "comparison", "compared", "review", "best drone", "best gopro",
        "best action camera", "which drone", "what's the best", "how to", "tutorial",
        "guide to", "settings", " mounts", "attach", "suitable as a drone", "unboxing",
    )


def gate_candidate(meta: dict, gate: CurationGate) -> tuple[bool, list[str]]:
    """Apply the gate to one candidate. Returns (accepted, list-of-reject-reasons)."""
    reasons: list[str] = []
    dur = float(meta.get("duration") or 0)
    if dur < gate.min_duration_s:
        reasons.append(f"duration {dur:.0f}s < {gate.min_duration_s:.0f}")
    elif dur > gate.max_duration_s:
        reasons.append(f"duration {dur:.0f}s > {gate.max_duration_s:.0f}")

    h = int(meta.get("height") or 0)
    if h < gate.min_height:
        reasons.append(f"height {h} < {gate.min_height}")

    fps = float(meta.get("fps") or 0)
    if fps < gate.min_fps:
        reasons.append(f"fps {fps:.0f} < {gate.min_fps:.0f}")

    w = int(meta.get("width") or 0)
    if h > 0 and w > 0:
        aspect = w / h
        if aspect < gate.min_aspect:
            reasons.append(f"aspect {aspect:.2f} < {gate.min_aspect} (vertical?)")
        elif aspect > gate.max_aspect:
            reasons.append(f"aspect {aspect:.2f} > {gate.max_aspect} (ultra-wide?)")

    if gate.reject_live and meta.get("is_live"):
        reasons.append("is_live")

    title_l = (meta.get("title") or "").lower()
    for bad in gate.reject_title_substrings:
        if bad in title_l:
            reasons.append(f"title~{bad.strip()!r}")
            break

    return (not reasons), reasons


def normalize_title(title: str) -> str:
    """Lowercase, strip non-alnum, collapse whitespace — for fuzzy title dedup."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (title or "").lower())).strip()


def title_similarity(a: str, b: str) -> float:
    """Ratio in [0,1] between two normalized titles (1.0 = identical)."""
    return difflib.SequenceMatcher(None, a, b).ratio()


def dedup_candidates(
    candidates: list[dict],
    *,
    existing_ids: set[str] | None = None,
    existing_titles: list[str] | None = None,
    title_threshold: float = 0.85,
    duration_tol_s: float = 2.0,
) -> tuple[list[dict], list[dict]]:
    """3-level dedup. Returns (kept, dropped); each dropped dict gets a ``_drop`` reason.

    - **L1** exact video id (vs prior candidates AND the already-curated ``existing_ids``).
    - **L2** fuzzy title similarity ``>= title_threshold`` (vs kept + ``existing_titles``).
    - **L3** same channel AND duration within ``duration_tol_s`` (re-upload / mirror).
    """
    existing_ids = set(existing_ids or set())
    existing_norm_titles = [normalize_title(t) for t in (existing_titles or [])]

    kept: list[dict] = []
    dropped: list[dict] = []
    seen_ids: set[str] = set(existing_ids)
    seen: list[tuple[str, str | None, float]] = [(t, None, -1.0) for t in existing_norm_titles]

    for c in candidates:
        vid = c.get("id")
        if not vid or vid in seen_ids:
            dropped.append({**c, "_drop": "dup-id" if vid else "no-id"})
            continue
        nt = normalize_title(c.get("title", ""))
        ch = c.get("channel")
        dur = float(c.get("duration") or 0)
        is_dup = False
        for prev_t, prev_ch, prev_dur in seen:
            if prev_t and title_similarity(nt, prev_t) >= title_threshold:
                is_dup = True
                break
            if prev_ch is not None and ch == prev_ch and prev_dur >= 0 and abs(dur - prev_dur) <= duration_tol_s:
                is_dup = True
                break
        if is_dup:
            dropped.append({**c, "_drop": "dup-title/channel"})
            continue
        seen_ids.add(vid)
        seen.append((nt, ch, dur))
        kept.append(c)

    return kept, dropped


def candidate_to_entry(meta: dict, clip_id: str, sport: str = "skiing") -> dict:
    """One candidate → a clips-YAML entry (matches configs/sports_clips.yaml schema)."""
    dur = int(meta.get("duration") or 0)
    h = meta.get("height") or "?"
    fps = meta.get("fps") or "?"
    title = (meta.get("title") or "").strip()
    return {
        "url": f"https://www.youtube.com/watch?v={meta['id']}",
        "clip_id": clip_id,
        "sport": sport,
        "notes": f"{title} ({dur}s, {h}p, {fps}fps)",
    }


__all__ = [
    "CurationGate",
    "gate_candidate",
    "normalize_title",
    "title_similarity",
    "dedup_candidates",
    "candidate_to_entry",
]
