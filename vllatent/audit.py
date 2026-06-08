"""AerialVLN-JSON AUDIT parser (PURE tier) — Phase-A step 5 (DoD item 2).

Reads an AerialVLN episode JSON and confirms it yields the
(RGB obs, 4-DoF action/waypoint, next obs, language) tuples the loader needs.
Pure-numpy / stdlib; NO sim, NO torch. STUB at scaffold time; implemented in step 5.

What the implemented audit does:
  * extract the tuple fields from an episode dict;
  * PIN foot-gun #1: reorder start_rotation [w,x,y,z] vs reference_path [...,qx,qy,qz,qw]
    into one canonical order (xyzw) and assert consistency;
  * assert actions[step] is index-aligned with reference_path[step];
  * derive the continuous 4-DoF delta from consecutive reference_path poses and
    VERIFY it matches the quantized action delta from vllatent.actions;
  * emit an AuditReport (per-action counts, tuple completeness, quaternion-order
    verdict, derived-vs-quantized mismatches, scene_id range, splits present).

CLI:  python -m vllatent.audit --episode <episode.json> [--report <out.json|->]

See plans/phase-a-data-and-io-contract.md step 5 + docs/io-contract.md.
"""
from __future__ import annotations

import argparse
import sys


def audit_episode(episode: dict) -> dict:  # pragma: no cover - implemented in step 5
    """Audit a single AerialVLN episode dict -> AuditReport dict."""
    raise NotImplementedError("vllatent.audit.audit_episode lands in Phase-A step 5")


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="AerialVLN episode audit (Phase-A step 5)")
    p.add_argument("--episode", help="path to an AerialVLN episode JSON")
    p.add_argument("--report", help="write the AuditReport JSON here ('-' = stdout)")
    p.parse_args(argv)
    print("vllatent.audit: not yet implemented (Phase-A step 5).", file=sys.stderr)
    return 2


__all__ = ["audit_episode"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
