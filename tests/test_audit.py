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
from vllatent.audit import (
    _infer_splits,
    _main,
    audit_episode,
    parse_episode,
    summarize_episodes,
)

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


def test_tiny_alignment_poses_eq_actions() -> None:
    # Real AerialVLN layout: one pose per action; reference_path[0] is the start pose.
    rep = audit_episode(_load("tiny_episode.json"))
    assert rep.n_poses == rep.n_actions
    assert rep.alignment_ok


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


# --- AuditSummary slice aggregator (A5.7, M3) ---

def _both_reports() -> list:
    return [audit_episode(_load("tiny_episode.json")), audit_episode(_load("quaternion_trap.json"))]


def test_summary_aggregates_dataset_level_checks_across_slice() -> None:
    reps = _both_reports()  # tiny (scene 1, all 8 classes) + quaternion_trap (scene 3, subset)
    summ = summarize_episodes(reps, splits=["train"])
    assert summ.n_episodes == 2 and summ.n_ok == 2
    # all-8-classes is the UNION over the slice (tiny alone supplies all 8), not per-episode.
    assert summ.all_action_classes_present
    # scene-id range is min..max across the slice (1..3), not the (id,id) the per-episode report uses.
    assert summ.scene_id_range == (1, 3) and summ.scene_ids == [1, 3]
    assert summ.splits_present == ["train"]
    assert summ.n_reorder_consistent == 2          # both fixtures reorder-consistent
    assert summ.n_naive_would_mismatch == 1        # only quaternion_trap trips the foot-gun
    assert summ.total_delta_mismatches == 0
    # action_counts is the SUM across episodes.
    expected: dict[int, int] = {}
    for r in reps:
        for a, c in r.action_counts.items():
            expected[a] = expected.get(a, 0) + c
    assert summ.action_counts == dict(sorted(expected.items()))
    assert summ.ok  # all episodes ok AND the slice has all action classes


def test_summary_not_ok_when_a_class_is_missing() -> None:
    # quaternion_trap alone (scene 3) does NOT contain all 8 classes -> slice not ok.
    summ = summarize_episodes([audit_episode(_load("quaternion_trap.json"))])
    assert not summ.all_action_classes_present
    assert not summ.ok
    assert summ.scene_id_range == (3, 3) and summ.splits_present == []


def test_summary_empty_slice_is_not_ok() -> None:
    summ = summarize_episodes([])
    assert summ.n_episodes == 0 and not summ.ok and summ.scene_id_range == (0, 0)


def test_infer_splits_from_filename() -> None:
    assert _infer_splits("data/aerialvln_json/train.slice.json") == ["train"]
    assert _infer_splits("/x/val_unseen.slice.json") == ["val_unseen"]
    assert _infer_splits("/x/whatever.json") == []


# --- CLI exit codes ---

def test_cli_returns_zero_on_clean_episode() -> None:
    rc = _main(["--episode", str(FIX / "tiny_episode.json")])
    assert rc == 0


def test_cli_slice_summary_writes_aggregate(tmp_path) -> None:
    slice_file = tmp_path / "train.slice.json"
    slice_file.write_text(json.dumps({"episodes": [_load("tiny_episode.json"), _load("quaternion_trap.json")]}))
    out = tmp_path / "summary.json"
    rc = _main(["--slice", str(slice_file), "--summary", str(out)])
    summ = json.loads(out.read_text())
    assert summ["n_episodes"] == 2
    assert summ["all_action_classes_present"] is True
    assert summ["scene_id_range"] == [1, 3]
    assert summ["splits_present"] == ["train"]            # inferred from the filename
    assert rc == (0 if summ["ok"] else 1)
