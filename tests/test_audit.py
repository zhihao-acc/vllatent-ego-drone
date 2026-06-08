"""Step-5 tests: AerialVLN-JSON audit parser (PURE tier, DoD item 2).

Runs the audit on two committed fixtures:
  * tiny_episode.json   — clean: all 8 action classes, derived Δ == quantized Δ, quaternion OK.
  * quaternion_trap.json — the audit must FLAG that a no-reorder read corrupts the yaw.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vllatent.actions import Action
from vllatent.audit import _main, audit_episode, parse_episode

FIX = Path(__file__).resolve().parent.parent / "fixtures" / "episodes"


def _load(name: str) -> dict:
    return json.loads((FIX / name).read_text())


# --- tiny_episode: clean report ---

def test_tiny_episode_is_clean() -> None:
    rep = audit_episode(_load("tiny_episode.json"))
    assert rep.ok
    assert rep.alignment_ok
    assert rep.tuple_complete
    assert rep.quaternion.reorder_consistent
    assert rep.delta_mismatches == []


def test_tiny_has_all_action_classes() -> None:
    rep = audit_episode(_load("tiny_episode.json"))
    assert rep.all_action_classes_present
    for a in Action:
        assert rep.action_counts.get(int(a), 0) >= 1


def test_tiny_alignment_poses_eq_actions_plus_one() -> None:
    rep = audit_episode(_load("tiny_episode.json"))
    assert rep.n_poses == rep.n_actions + 1


# --- quaternion_trap: the foot-gun must be flagged ---

def test_quaternion_trap_flagged() -> None:
    rep = audit_episode(_load("quaternion_trap.json"))
    # parse_episode reorders correctly, so the data itself is consistent ...
    assert rep.quaternion.reorder_consistent
    # ... but the audit FLAGS that skipping the reorder would corrupt the yaw.
    assert rep.quaternion.naive_would_mismatch
    assert abs(rep.quaternion.canonical_yaw_deg - 90.0) < 1.0
    assert abs(rep.quaternion.reference0_yaw_deg - 90.0) < 1.0
    assert abs(rep.quaternion.naive_yaw_deg - 0.0) < 1.0


def test_parse_episode_reorders_start_rotation_to_xyzw() -> None:
    rec = parse_episode(_load("quaternion_trap.json"))
    # raw start_rotation is w-FIRST [cos45,0,0,sin45]; canonical xyzw is [0,0,sin45,cos45].
    s = 0.7071067811865476
    np.testing.assert_allclose(rec.start_rotation_xyzw, [0.0, 0.0, s, s], atol=1e-9)


def test_naive_read_without_reorder_is_wrong() -> None:
    # Demonstrates WHY the reorder matters: the raw start_rotation read as xyzw yields yaw 0,
    # which disagrees with the true yaw 90 the audit recovers after reordering.
    rep = audit_episode(_load("quaternion_trap.json"))
    assert abs(rep.quaternion.naive_yaw_deg - rep.quaternion.canonical_yaw_deg) > 45.0


# --- CLI exit code (drives `make audit`) ---

def test_cli_returns_zero_on_clean_episode() -> None:
    rc = _main(["--episode", str(FIX / "tiny_episode.json")])
    assert rc == 0
