#!/usr/bin/env python3
"""Data-quality QC report: content-filter accept/reject + MegaSaM trajectory.

Produces self-contained browsable HTML so you can SEE, per clip:
  1. Which frames the content filter kept (FPV) vs dropped, and why
     (static / object / short-run) — a decision filmstrip + thumbnail sheet.
  2. The MegaSaM camera trajectory + VO verdict for each sub-clip.

A top-level index links every clip with headline metrics.

Reads decisions persisted by the ingest run (``frames/<clip>/_filter.json``).
For clips processed before persistence existed, pass --compute-filter to
generate it once (re-runs the YOLO content filter).

Usage:
    # one source clip
    python scripts/qc_report.py --clip cand01 --device cuda
    # all source clips + index
    python scripts/qc_report.py --all --device cuda
    # regenerate the filter decisions if _filter.json is missing
    python scripts/qc_report.py --clip cand01 --compute-filter --device cuda
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

_ROOT = str(Path(__file__).resolve().parent.parent)
_HERE = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import qc_lib  # noqa: E402


def _log(msg: str) -> None:
    print(f"[qc] {msg}", file=sys.stderr)


def discover_source_clips(frames_root: Path) -> list[str]:
    """Source clips = frame dirs with no underscore (sub-clips/megasam have one)."""
    out = []
    for d in sorted(frames_root.iterdir()):
        if d.is_dir() and "_" not in d.name and list(d.glob("*.jpg")):
            out.append(d.name)
    return out


def subclips_for(clip_id: str, frames_root: Path, cache_dir: Path) -> list[dict]:
    """Find a source clip's sub-clips and their megasam/npz artifacts."""
    subs: list[dict] = []
    for d in sorted(frames_root.glob(f"{clip_id}_fpv*_c*")):
        if not d.is_dir() or d.name.endswith("_megasam"):
            continue
        sub_id = d.name
        subs.append({
            "sub_id": sub_id,
            "frames_dir": d,
            "megasam_dir": frames_root / f"{sub_id}_megasam",
            "npz": cache_dir / f"{sub_id}.npz",
        })
    return subs


def _ensure_filter_json(clip_id: str, frames_dir: Path, device: str, compute: bool) -> dict | None:
    from vllatent.ingest.content_filter import (
        filter_video_from_paths,
        load_filter_result,
        save_filter_result,
    )

    data = load_filter_result(frames_dir)
    if data is not None:
        return data
    if not compute:
        _log(f"  {clip_id}: no _filter.json (run ingest, or pass --compute-filter)")
        return None
    _log(f"  {clip_id}: computing content filter (YOLO) — one-time...")
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    result = filter_video_from_paths(frame_paths, device=device)
    save_filter_result(frames_dir, result)
    return load_filter_result(frames_dir)


def build_frame_panel(clip_id: str, frames_dir: Path, fdata: dict) -> tuple[str, dict]:
    """Build the content-filter accept/reject HTML section + a summary dict."""
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    fpv_mask = np.array(fdata["fpv_mask"], dtype=bool)
    motion = np.array(fdata["motion_scores"], dtype=float) if fdata.get("motion_scores") else None
    rejected = np.array(fdata["rejected_objects"], dtype=bool) if fdata.get("rejected_objects") else None

    reasons = qc_lib.classify_frame_reasons(
        fpv_mask, motion, rejected, motion_threshold=fdata.get("motion_threshold", 8.0)
    )
    counts = {r: reasons.count(r) for r in set(reasons)}
    n = len(reasons)
    n_fpv = int(fpv_mask.sum())

    strip = qc_lib.filmstrip_data_uri(reasons)
    sheet_html, stride = qc_lib.contact_sheet_html(frame_paths, reasons, motion)
    present = [r for r in ["fpv", "object", "static", "short", "other"] if r in counts]

    breakdown = " · ".join(
        f"{qc_lib.REASON_LABELS[r]}: {counts.get(r, 0)}" for r in present
    )
    body = [
        f"<h2>Content filter — {clip_id}</h2>",
        f"<p class='small'>verdict <b>{fdata['verdict']}</b> · "
        f"{n_fpv}/{n} frames kept ({100.0 * n_fpv / max(1, n):.1f}%) · "
        f"{len(fdata['shots'])} shots · contact-sheet stride {stride}</p>",
        f"<p class='small'>{breakdown}</p>",
        qc_lib.legend_html(present),
        f"<img class='filmstrip' src='{strip}' title='one cell per frame'>",
        sheet_html,
    ]
    summary = {
        "n_frames": n, "n_fpv": n_fpv,
        "fpv_pct": 100.0 * n_fpv / max(1, n),
        "verdict": fdata["verdict"], "counts": counts,
    }
    return "".join(body), summary


def subclip_source_range(sub_id: str, fdata: dict | None, clip_length_frames: int) -> tuple[int, int] | None:
    """Reconstruct a sub-clip's frame range in its SOURCE clip.

    Mirrors the orchestrator's cut: extract_fpv_ranges(shots, fpv_mask) then
    sequential clip_length_frames chunks. ``fpv{ri}_c{ci}`` selects range ri,
    chunk ci. Exact whenever fdata is the same _filter.json that drove the cut.
    """
    m = re.search(r"_fpv(\d+)_c(\d+)$", sub_id)
    if m is None or not fdata or not fdata.get("shots"):
        return None
    ri, ci = int(m.group(1)), int(m.group(2))
    from vllatent.ingest.content_filter import ShotInfo, extract_fpv_ranges

    shots = [
        ShotInfo(start=int(s["start"]), end=int(s["end"]),
                 is_fpv=bool(s["is_fpv"]), mean_score=float(s["mean_score"]))
        for s in fdata["shots"]
    ]
    fpv_mask = np.array(fdata["fpv_mask"], dtype=bool)
    ranges = extract_fpv_ranges(shots, fpv_mask)
    if ri >= len(ranges):
        return None
    rs, re_ = ranges[ri]
    start = rs + ci * clip_length_frames
    end = min(rs + (ci + 1) * clip_length_frames, re_)
    return (int(start), int(end)) if start < end else None


def _range_label(s: dict, fdata: dict | None, clip_length_frames: int) -> str:
    """Frame-range label for a sub-clip: source range (if derivable) + local count."""
    n_local = len(list(s["frames_dir"].glob("*.jpg")))
    src = subclip_source_range(s["sub_id"], fdata, clip_length_frames)
    if src is not None:
        return f"source frames [{src[0]}–{src[1]}) · {n_local} frames"
    return f"{n_local} frames"


def build_trajectory_section(
    subs: list[dict], fps: float, fdata: dict | None, clip_length_frames: int,
) -> tuple[str, list[dict], bool]:
    """Build the trajectory HTML for each sub-clip with a megasam output."""
    from vllatent.ingest.ego_motion import se3_sequence_to_deltas
    from vllatent.ingest.megasam import parse_megasam_output
    from vllatent.ingest.vo_validation import (
        LOW_CONFIDENCE_THRESHOLD,
        MAX_SKIING_SPEED_MS,
        MAX_YAW_RATE_DEG_S,
        validate_clip,
    )

    parts = ["<h2>MegaSaM trajectories</h2>"]
    rows: list[dict] = []
    first_fig = True
    any_fig = False
    n_with_traj = sum(1 for s in subs if s["megasam_dir"].exists())
    parts.append(f"<p class='small'>{n_with_traj}/{len(subs)} sub-clips have MegaSaM output</p>")

    for s in subs:
        sub_id = s["sub_id"]
        rng = _range_label(s, fdata, clip_length_frames)
        if not s["megasam_dir"].exists():
            parts.append(f"<p class='small'>· {sub_id} — {rng} — no trajectory (no megasam output)</p>")
            rows.append({"sub_id": sub_id, "decision": "—"})
            continue
        try:
            res = parse_megasam_output(s["megasam_dir"])
            rep = validate_clip(res.poses, res.confidences, fps=fps, clip_id=sub_id)
            deltas = se3_sequence_to_deltas(res.poses)
            fig = qc_lib.build_vo_figure(
                poses=res.poses, confidences=res.confidences, deltas=deltas,
                decision=rep.verdict.decision, clip_id=sub_id, fps=fps,
                max_speed_ms=MAX_SKIING_SPEED_MS,
                max_yaw_rate_deg_s=MAX_YAW_RATE_DEG_S,
                low_confidence=LOW_CONFIDENCE_THRESHOLD,
            )
            div = fig.to_html(full_html=False, include_plotlyjs="cdn" if first_fig else False)
            first_fig = False
            any_fig = True
            parts.append(
                f"<h3 style='font-size:14px'>{sub_id} — "
                f"<span class='small' style='font-weight:400'>{rng}</span> — {rep.verdict.decision}</h3>"
            )
            parts.append(div)
            rows.append({"sub_id": sub_id, "decision": rep.verdict.decision})
        except Exception as e:  # noqa: BLE001
            parts.append(f"<p class='small'>· {sub_id} — {rng} — trajectory FAILED — {e}</p>")
            rows.append({"sub_id": sub_id, "decision": "ERR"})
    return "".join(parts), rows, any_fig


def build_clip_report(
    clip_id: str, frames_root: Path, cache_dir: Path, out_root: Path,
    *, fps: float, device: str, compute_filter: bool, clip_length_frames: int,
) -> dict | None:
    frames_dir = frames_root / clip_id
    if not frames_dir.is_dir():
        _log(f"  {clip_id}: no frames dir {frames_dir}")
        return None

    fdata = _ensure_filter_json(clip_id, frames_dir, device, compute_filter)
    body = [f"<h1>QC — {clip_id}</h1>", "<p><a href='../index.html'>&larr; index</a></p>"]
    summary: dict = {"clip_id": clip_id}

    if fdata is not None:
        panel, fsum = build_frame_panel(clip_id, frames_dir, fdata)
        body.append(panel)
        summary.update(fsum)
    else:
        body.append("<p class='small'>(no content-filter data)</p>")

    subs = subclips_for(clip_id, frames_root, cache_dir)
    traj_html, traj_rows, _ = build_trajectory_section(subs, fps, fdata, clip_length_frames)
    body.append(traj_html)
    summary["n_subclips"] = len(subs)
    summary["n_traj_go"] = sum(1 for r in traj_rows if r["decision"] == "GO")
    summary["n_traj_cond"] = sum(1 for r in traj_rows if r["decision"] == "CONDITIONAL-GO")
    summary["n_traj_nogo"] = sum(1 for r in traj_rows if r["decision"] == "NO-GO")

    out_dir = out_root / clip_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(qc_lib.html_page(f"QC {clip_id}", "".join(body)))
    _log(f"  {clip_id}: wrote {out_dir / 'index.html'}")
    return summary


def build_index(rows: list[dict], out_root: Path) -> None:
    head = (
        "<tr><th>clip</th><th>verdict</th><th>frames</th><th>FPV%</th>"
        "<th>sub-clips</th><th>VO GO/COND/NOGO</th><th></th></tr>"
    )
    trs = [head]
    for r in rows:
        cid = r["clip_id"]
        verdict = r.get("verdict", "—")
        frames = r.get("n_frames", "—")
        fpvp = f"{r['fpv_pct']:.1f}%" if "fpv_pct" in r else "—"
        nsub = r.get("n_subclips", 0)
        vo = f"{r.get('n_traj_go', 0)}/{r.get('n_traj_cond', 0)}/{r.get('n_traj_nogo', 0)}"
        trs.append(
            f"<tr><td>{cid}</td><td>{verdict}</td><td>{frames}</td><td>{fpvp}</td>"
            f"<td>{nsub}</td><td>{vo}</td><td><a href='{cid}/index.html'>open</a></td></tr>"
        )
    body = f"<h1>QC index — {len(rows)} clips</h1><table>{''.join(trs)}</table>"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "index.html").write_text(qc_lib.html_page("QC index", body))
    _log(f"index: {out_root / 'index.html'}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Data-quality QC report (frames + trajectory)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--clip", nargs="+", help="One or more source clip ids (e.g. cand01 cand02)")
    g.add_argument("--all", action="store_true", help="All source clips + index")
    p.add_argument("--frames-root", type=Path, default=Path("ingest_data/frames"))
    p.add_argument("--cache", type=Path, default=Path("ingest_data/latent_cache"))
    p.add_argument("--out", type=Path, default=Path("reports/qc"))
    p.add_argument("--fps", type=float, default=5.0)
    p.add_argument("--config", default="configs/sports.yaml",
                   help="Sports config (for clip_length_frames = clip_length_seconds * target_fps)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--compute-filter", action="store_true",
                   help="Run YOLO filter if _filter.json missing (one-time)")
    args = p.parse_args(argv)

    clip_length_frames = 50
    try:
        from vllatent.config import Config

        ing = Config.from_yaml(args.config).ingest
        if ing is not None:
            clip_length_frames = int(ing.clip_length_seconds * ing.target_fps)
    except Exception as e:  # noqa: BLE001
        _log(f"config read failed ({e}); using clip_length_frames={clip_length_frames}")

    if args.clip:
        clip_ids = args.clip
    else:
        clip_ids = discover_source_clips(args.frames_root)
        if not clip_ids:
            _log(f"No source clips under {args.frames_root}")
            return 1
        _log(f"Found {len(clip_ids)} source clips: {clip_ids}")

    rows: list[dict] = []
    for cid in clip_ids:
        s = build_clip_report(
            cid, args.frames_root, args.cache, args.out,
            fps=args.fps, device=args.device, compute_filter=args.compute_filter,
            clip_length_frames=clip_length_frames,
        )
        if s is not None:
            rows.append(s)

    build_index(rows, args.out)
    _log(f"DONE: {len(rows)} clip report(s) under {args.out}/")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
