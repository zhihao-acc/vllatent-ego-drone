#!/usr/bin/env python
"""MegaSaM VO validation CLI (B1.10b).

Parses MegaSaM output, runs physics/smoothness/confidence/drift checks,
generates interactive Plotly HTML report + terminal verdict.

Usage:
    python scripts/validate_megasam.py \\
        --megasam-dir ingest_data/frames/ski01_megasam \\
        --fps 5 --clip-id ski01 \\
        --out reports/ski01_vo.html

    # Batch mode: validate all *_megasam dirs under a parent
    python scripts/validate_megasam.py \\
        --batch-dir ingest_data/frames \\
        --fps 5 --out reports/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _log(msg: str) -> None:
    print(f"[vo-validate] {msg}", file=sys.stderr)


def _parse_and_validate(megasam_dir: Path, fps: float, clip_id: str) -> dict:
    """Run full validation pipeline on one clip's MegaSaM output."""
    from vllatent.ingest.ego_motion import normalize_scale, se3_sequence_to_deltas
    from vllatent.ingest.megasam import parse_megasam_output, validate_megasam_result
    from vllatent.ingest.vo_validation import validate_clip

    result = parse_megasam_output(megasam_dir)

    structural_errors = validate_megasam_result(result)
    if structural_errors:
        _log(f"  structural errors: {structural_errors}")

    report = validate_clip(result.poses, result.confidences, fps=fps, clip_id=clip_id)

    deltas = se3_sequence_to_deltas(result.poses)
    deltas_norm = normalize_scale(deltas, mode="median_speed")

    return {
        "report": report,
        "poses": result.poses,
        "confidences": result.confidences,
        "deltas": deltas,
        "deltas_norm": deltas_norm,
        "structural_errors": structural_errors,
        "fps": fps,
    }


def _print_report(report, clip_id: str) -> None:
    """Print terminal summary for one clip."""
    from vllatent.ingest.vo_validation import ClipValidationReport

    r: ClipValidationReport = report
    v = r.verdict

    print(f"\n{'='*60}")
    print(f"  {clip_id}  ({r.n_frames} frames)")
    print(f"{'='*60}")
    print(f"  Verdict: {v.decision}")
    print("  Checks:")
    for check, status in v.checks.items():
        marker = {"pass": "OK", "warn": "!!", "fail": "XX"}[status]
        print(f"    [{marker}] {check}")
    if v.reasons:
        print("  Reasons:")
        for reason in v.reasons:
            print(f"    - {reason}")

    sm = r.smoothness
    print(f"  Smoothness: mean_jerk={sm.mean_jerk:.1f} m/s³, "
          f"max_jerk={sm.max_jerk:.1f}, "
          f"accel_disc={sm.n_accel_discontinuities}, "
          f"angular_spikes={sm.n_angular_spikes}")

    ph = r.physics
    print(f"  Physics: max_speed={ph.max_speed:.1f} m/s, "
          f"mean_speed={ph.mean_speed:.1f}, "
          f"altitude_change={ph.net_altitude_change:.1f} m, "
          f"max_yaw_rate={ph.max_yaw_rate:.0f}°/s")

    ca = r.confidence
    print(f"  Confidence: mean={ca.mean:.2f}, "
          f"low_frac={ca.frac_low:.1%}, "
          f"longest_low_run={ca.longest_low_run}")

    dr = r.drift
    print(f"  Drift: ratio={dr.drift_ratio:.2f} "
          f"(first_q={dr.speed_first_quarter:.3f}, last_q={dr.speed_last_quarter:.3f})")
    print()


def _generate_html(data: dict, out_path: Path) -> None:
    """Generate interactive Plotly HTML report (figure built in qc_lib)."""
    import sys as _sys

    _here = str(Path(__file__).resolve().parent)
    if _here not in _sys.path:
        _sys.path.insert(0, _here)
    import qc_lib

    report = data["report"]
    fig = qc_lib.build_vo_figure(
        poses=data["poses"],
        confidences=data["confidences"],
        deltas=data["deltas"],
        decision=report.verdict.decision,
        clip_id=report.clip_id,
        fps=float(data.get("fps", 5.0)),
        max_speed_ms=MAX_SKIING_SPEED_MS,
        max_yaw_rate_deg_s=MAX_YAW_RATE_DEG_S,
        low_confidence=LOW_CONFIDENCE_THRESHOLD,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    _log(f"HTML report: {out_path}")


# Import constants for the HTML generator
from vllatent.ingest.vo_validation import (  # noqa: E402
    LOW_CONFIDENCE_THRESHOLD,
    MAX_SKIING_SPEED_MS,
    MAX_YAW_RATE_DEG_S,
)


def _discover_megasam_dirs(parent: Path) -> list[tuple[str, Path]]:
    """Find all *_megasam directories under parent."""
    results = []
    for d in sorted(parent.iterdir()):
        if d.is_dir() and d.name.endswith("_megasam"):
            clip_id = d.name.replace("_megasam", "")
            results.append((clip_id, d))
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="MegaSaM VO validation")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--megasam-dir", type=Path, help="Single MegaSaM output directory")
    g.add_argument("--batch-dir", type=Path, help="Parent dir containing *_megasam subdirs")
    p.add_argument("--fps", type=float, default=5.0, help="Source FPS (default: 5.0)")
    p.add_argument("--clip-id", default="", help="Clip ID (auto-derived if not set)")
    p.add_argument("--out", type=Path, default=None, help="Output HTML path or directory")
    p.add_argument("--json", type=Path, default=None, help="Write JSON summary")
    args = p.parse_args(argv)

    clips: list[tuple[str, Path]] = []
    if args.megasam_dir:
        clip_id = args.clip_id or args.megasam_dir.name.replace("_megasam", "")
        clips = [(clip_id, args.megasam_dir)]
    else:
        clips = _discover_megasam_dirs(args.batch_dir)
        if not clips:
            _log(f"No *_megasam directories found in {args.batch_dir}")
            return 1

    verdicts = []
    for clip_id, megasam_dir in clips:
        _log(f"Validating {clip_id} from {megasam_dir}")
        try:
            data = _parse_and_validate(megasam_dir, args.fps, clip_id)
        except FileNotFoundError as e:
            _log(f"  SKIP: {e}")
            continue

        _print_report(data["report"], clip_id)

        if args.out:
            out_path = args.out
            if out_path.suffix != ".html":
                out_path = out_path / f"{clip_id}_vo.html"
            _generate_html(data, out_path)

        v = data["report"].verdict
        verdicts.append({"clip_id": clip_id, "decision": v.decision, "checks": v.checks})

    # Summary
    if len(verdicts) > 1:
        print(f"\n{'='*60}")
        print(f"  BATCH SUMMARY ({len(verdicts)} clips)")
        print(f"{'='*60}")
        for v in verdicts:
            print(f"  {v['clip_id']}: {v['decision']}")
        n_go = sum(1 for v in verdicts if v["decision"] == "GO")
        n_cond = sum(1 for v in verdicts if v["decision"] == "CONDITIONAL-GO")
        n_nogo = sum(1 for v in verdicts if v["decision"] == "NO-GO")
        print(f"\n  GO={n_go}  CONDITIONAL-GO={n_cond}  NO-GO={n_nogo}")

        if n_go >= len(verdicts) * 0.67:
            overall = "GO"
        elif n_nogo >= len(verdicts) * 0.5:
            overall = "NO-GO"
        else:
            overall = "CONDITIONAL-GO"
        print(f"  Overall: {overall}\n")

    if args.json and verdicts:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(verdicts, indent=2))
        _log(f"JSON summary: {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
