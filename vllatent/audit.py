"""AerialVLN-JSON AUDIT parser (PURE tier) — Phase-A step 5 (DoD item 2).

Reads an AerialVLN episode JSON and confirms it yields the
``(RGB obs, 4-DoF action/waypoint, next obs, language)`` tuples the loader needs.
Pure-numpy / stdlib; NO sim, NO torch. Schema confirmed against AirVLN's own loader
(``src/vlnce_src/env.py``):

  * ``episode_id`` / ``trajectory_id`` / ``scene_id``
  * ``instruction.instruction_text``
  * ``start_position`` ``[x,y,z]`` (NED) · ``start_rotation`` ``[w,x,y,z]`` **w-FIRST**
    (env.py builds ``Quaternionr(x=sr[1], y=sr[2], z=sr[3], w=sr[0])``)
  * ``goals[].position`` · ``actions`` list[int] 0..7
  * ``reference_path`` rows ``[x,y,z,pitch,roll,yaw]`` — EULER radians, 6-wide,
    pitch=roll==0 (4-DoF), yaw = row[5]. ``len(reference_path) == len(actions)``;
    ``reference_path[0]`` is the start pose and ``actions[t]`` drives
    ``reference_path[t] -> reference_path[t+1]`` (the terminal STOP has no stored next pose).

What the audit does (foot-gun #1 + tuple/Δ verification):
  * ``parse_episode``: reorder ``start_rotation`` (w-FIRST quaternion) -> canonical ``xyzw``.
  * quaternion verdict: confirm the reordered start yaw matches ``reference_path[0]`` yaw,
    and FLAG that a naïve (no-reorder) read would corrupt the yaw (the foot-gun).
  * assert ``actions[t]`` is index-aligned with the pose pair
    ``reference_path[t] -> reference_path[t+1]``.
  * derive the continuous body-frame Δ from consecutive poses and VERIFY it matches the
    quantized ``vllatent.actions.action_to_delta`` within tolerance.
  * emit an :class:`AuditReport` (per-action counts, tuple completeness, quaternion verdict,
    Δ-mismatch list, scene_id range, splits, license).

CLI:  python -m vllatent.audit --episode <episode.json> [--report <out.json|->]

See docs/io-contract.md + plans/phase-a-data-and-io-contract.md step 5.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vllatent.actions import (
    Pose,
    _wrap_pi,
    _xyzw_from_yaw,
    _yaw_from_xyzw,
    action_to_delta,
    pose_pair_to_body_delta,
)
from vllatent.frames import REFERENCE_PATH_ROW_WIDTH, REFERENCE_PATH_YAW_INDEX
from vllatent.schemas import N_ACTIONS, EpisodeRecord

DELTA_TOL = 1e-3        # body-frame Δ match tolerance (m / deg) vs the quantized action delta
QUAT_TOL_DEG = 1.0      # yaw consistency tolerance (degrees)
LICENSE = "CC BY-NC-SA 4.0"


def parse_episode(episode: dict[str, Any]) -> EpisodeRecord:
    """Parse one AerialVLN episode dict into an :class:`EpisodeRecord`.

    Reorders ``start_rotation`` from w-FIRST ``[w,x,y,z]`` to canonical ``xyzw`` (foot-gun #1).
    """
    sr = episode["start_rotation"]  # [w, x, y, z] — w-FIRST
    start_rotation_xyzw = np.array([sr[1], sr[2], sr[3], sr[0]], dtype=float)
    goals = episode.get("goals", [])
    goal_positions = np.asarray([g["position"] for g in goals], dtype=float) if goals else np.zeros((0, 3))
    return EpisodeRecord(
        episode_id=str(episode["episode_id"]),
        trajectory_id=str(episode.get("trajectory_id", "")),
        scene_id=int(episode["scene_id"]),
        instruction_text=str(episode["instruction"]["instruction_text"]),
        start_position=np.asarray(episode["start_position"], dtype=float),
        start_rotation_xyzw=start_rotation_xyzw,
        goal_positions=goal_positions,
        actions=np.asarray(episode["actions"], dtype=int),
        reference_path=np.asarray(episode["reference_path"], dtype=float),
    )


@dataclass(frozen=True)
class QuaternionVerdict:
    """Foot-gun #1 verdict: does the reorder hold, and would skipping it corrupt yaw?"""

    start_rotation_order: str       # "wxyz"
    reference_path_order: str       # "xyzw"
    canonical_yaw_deg: float        # yaw from the REORDERED start_rotation
    reference0_yaw_deg: float       # yaw from reference_path[0] orientation
    naive_yaw_deg: float            # yaw if start_rotation were read as xyzw (NO reorder)
    reorder_consistent: bool        # reordered start matches reference_path[0]
    naive_would_mismatch: bool      # True = skipping the reorder corrupts the yaw (the foot-gun)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_rotation_order": self.start_rotation_order,
            "reference_path_order": self.reference_path_order,
            "canonical_yaw_deg": self.canonical_yaw_deg,
            "reference0_yaw_deg": self.reference0_yaw_deg,
            "naive_yaw_deg": self.naive_yaw_deg,
            "reorder_consistent": self.reorder_consistent,
            "naive_would_mismatch": self.naive_would_mismatch,
        }


@dataclass(frozen=True)
class DeltaMismatch:
    """One step where the derived pose Δ disagrees with the quantized action delta."""

    step: int
    action: int
    derived: list[float]
    expected: list[float]
    max_abs_err: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "action": self.action,
            "derived": self.derived,
            "expected": self.expected,
            "max_abs_err": self.max_abs_err,
        }


@dataclass(frozen=True)
class AuditReport:
    """The Phase-A audit result for one episode (DoD item 2)."""

    episode_id: str
    scene_id: int
    n_actions: int
    n_poses: int
    action_counts: dict[int, int]
    all_action_classes_present: bool
    tuple_complete: bool
    alignment_ok: bool
    quaternion: QuaternionVerdict
    delta_mismatches: list[DeltaMismatch]
    scene_id_range: tuple[int, int]
    splits_present: list[str]
    license: str
    ok: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "scene_id": self.scene_id,
            "n_actions": self.n_actions,
            "n_poses": self.n_poses,
            "action_counts": {str(k): v for k, v in self.action_counts.items()},
            "all_action_classes_present": self.all_action_classes_present,
            "tuple_complete": self.tuple_complete,
            "alignment_ok": self.alignment_ok,
            "quaternion": self.quaternion.to_dict(),
            "delta_mismatches": [m.to_dict() for m in self.delta_mismatches],
            "scene_id_range": list(self.scene_id_range),
            "splits_present": self.splits_present,
            "license": self.license,
            "ok": self.ok,
        }


def _quaternion_verdict(episode: dict[str, Any], rec: EpisodeRecord) -> QuaternionVerdict:
    sr = episode["start_rotation"]  # raw, w-FIRST quaternion
    canonical_yaw = math.degrees(_yaw_from_xyzw(rec.start_rotation_xyzw))
    naive_yaw = math.degrees(_yaw_from_xyzw(np.asarray(sr, dtype=float)))  # read as xyzw (wrong)
    # reference_path orientation is EULER radians; yaw is a direct field (no quaternion).
    reference0_yaw = math.degrees(float(rec.reference_path[0, REFERENCE_PATH_YAW_INDEX]))
    consistent = abs(math.degrees(_wrap_pi(math.radians(canonical_yaw - reference0_yaw)))) <= QUAT_TOL_DEG
    naive_mismatch = abs(math.degrees(_wrap_pi(math.radians(naive_yaw - reference0_yaw)))) > QUAT_TOL_DEG
    return QuaternionVerdict(
        start_rotation_order="wxyz",
        reference_path_order="euler_pitch_roll_yaw_rad",
        canonical_yaw_deg=canonical_yaw,
        reference0_yaw_deg=reference0_yaw,
        naive_yaw_deg=naive_yaw,
        reorder_consistent=consistent,
        naive_would_mismatch=naive_mismatch,
    )


def _euler_row_to_pose(row: np.ndarray) -> Pose:
    """reference_path row [x,y,z,pitch,roll,yaw] (rad) -> (position, xyzw quaternion).

    pitch == roll == 0 in AerialVLN (4-DoF); only yaw drives the body frame.
    """
    return (np.asarray(row[0:3], dtype=float), _xyzw_from_yaw(float(row[REFERENCE_PATH_YAW_INDEX])))


def _delta_mismatches(rec: EpisodeRecord) -> list[DeltaMismatch]:
    mismatches: list[DeltaMismatch] = []
    # action[t] drives reference_path[t] -> reference_path[t+1]; the terminal STOP (last
    # action, no stored next pose) is unverifiable, so cap at len(reference_path) - 1.
    n_transitions = min(len(rec.actions), len(rec.reference_path) - 1)
    for t in range(n_transitions):
        before = _euler_row_to_pose(rec.reference_path[t])
        after = _euler_row_to_pose(rec.reference_path[t + 1])
        derived = pose_pair_to_body_delta(before, after)
        expected = action_to_delta(int(rec.actions[t]))
        err = float(np.max(np.abs(derived - expected)))
        if err > DELTA_TOL:
            mismatches.append(
                DeltaMismatch(
                    step=t,
                    action=int(rec.actions[t]),
                    derived=[round(float(x), 6) for x in derived],
                    expected=[round(float(x), 6) for x in expected],
                    max_abs_err=err,
                )
            )
    return mismatches


def audit_episode(episode: dict[str, Any]) -> AuditReport:
    """Audit a single AerialVLN episode dict -> :class:`AuditReport`."""
    rec = parse_episode(episode)
    counts = {int(k): int(v) for k, v in Counter(rec.actions.tolist()).items()}
    all_classes = all(counts.get(a, 0) >= 1 for a in range(N_ACTIONS))
    # Real AerialVLN: one action per pose (len ==), reference_path[0] is the start pose.
    starts_at_start = bool(
        rec.reference_path.shape[0] > 0
        and np.allclose(rec.reference_path[0, 0:3], rec.start_position, atol=1e-3)
    )
    alignment_ok = len(rec.reference_path) == len(rec.actions) and starts_at_start
    tuple_complete = bool(
        rec.instruction_text
        and rec.actions.size > 0
        and rec.reference_path.shape[0] > 0
        and rec.start_position.shape == (3,)
        and rec.reference_path.shape[1] == REFERENCE_PATH_ROW_WIDTH  # [x,y,z,pitch,roll,yaw]
    )
    quaternion = _quaternion_verdict(episode, rec)
    mismatches = _delta_mismatches(rec)
    ok = bool(quaternion.reorder_consistent and alignment_ok and tuple_complete and not mismatches)
    return AuditReport(
        episode_id=rec.episode_id,
        scene_id=rec.scene_id,
        n_actions=int(rec.actions.size),
        n_poses=int(rec.reference_path.shape[0]),
        action_counts=counts,
        all_action_classes_present=all_classes,
        tuple_complete=tuple_complete,
        alignment_ok=alignment_ok,
        quaternion=quaternion,
        delta_mismatches=mismatches,
        scene_id_range=(rec.scene_id, rec.scene_id),
        splits_present=[],
        license=LICENSE,
        ok=ok,
    )


def _iter_episodes(data: Any) -> list[dict[str, Any]]:
    """Accept a single episode dict, a list of episodes, or ``{'episodes': [...]}``."""
    if isinstance(data, dict) and "episodes" in data:
        return list(data["episodes"])
    if isinstance(data, list):
        return list(data)
    return [data]


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="AerialVLN episode audit (Phase-A step 5)")
    p.add_argument("--episode", required=True, help="path to an AerialVLN episode JSON")
    p.add_argument("--report", help="write the AuditReport JSON here ('-' = stdout)")
    args = p.parse_args(argv)

    data = json.loads(Path(args.episode).read_text())
    reports = [audit_episode(ep) for ep in _iter_episodes(data)]
    all_ok = all(r.ok for r in reports)

    payload: Any = reports[0].to_dict() if len(reports) == 1 else [r.to_dict() for r in reports]
    if args.report == "-":
        print(json.dumps(payload, indent=2))
    elif args.report:
        Path(args.report).write_text(json.dumps(payload, indent=2))

    for r in reports:
        flag = "OK" if r.ok else "FAIL"
        print(
            f"[{flag}] {r.episode_id} scene={r.scene_id} actions={r.n_actions} poses={r.n_poses} "
            f"quat(reorder_consistent={r.quaternion.reorder_consistent}, "
            f"naive_would_mismatch={r.quaternion.naive_would_mismatch}) "
            f"delta_mismatches={len(r.delta_mismatches)}",
            file=sys.stderr,
        )
    return 0 if all_ok else 1


__all__ = [
    "parse_episode",
    "audit_episode",
    "AuditReport",
    "QuaternionVerdict",
    "DeltaMismatch",
    "DELTA_TOL",
    "QUAT_TOL_DEG",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
