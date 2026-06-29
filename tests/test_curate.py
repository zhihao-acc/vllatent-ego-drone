"""B1.22c tests: curation gates + 3-level dedup (PURE — no torch, no yt-dlp)."""
from __future__ import annotations

from vllatent.ingest.curate import (
    CurationGate,
    candidate_to_entry,
    dedup_candidates,
    gate_candidate,
    normalize_title,
    title_similarity,
)

GOOD = {"id": "a", "title": "FPV ski run", "duration": 120, "height": 1080,
        "width": 1920, "fps": 30, "channel": "ch", "is_live": False}


def test_gate_accepts_good() -> None:
    ok, reasons = gate_candidate(GOOD, CurationGate())
    assert ok and reasons == []


def test_gate_rejects_low_res() -> None:
    ok, reasons = gate_candidate({**GOOD, "height": 480, "width": 854}, CurationGate())
    assert not ok and any("height" in r for r in reasons)


def test_gate_rejects_low_fps() -> None:
    ok, reasons = gate_candidate({**GOOD, "fps": 12}, CurationGate())
    assert not ok and any("fps" in r for r in reasons)


def test_gate_rejects_vertical() -> None:
    ok, reasons = gate_candidate({**GOOD, "height": 1920, "width": 1080}, CurationGate())
    assert not ok and any("aspect" in r for r in reasons)


def test_gate_rejects_too_short_and_too_long() -> None:
    assert not gate_candidate({**GOOD, "duration": 5}, CurationGate())[0]
    assert not gate_candidate({**GOOD, "duration": 5000}, CurationGate())[0]


def test_gate_rejects_live() -> None:
    ok, reasons = gate_candidate({**GOOD, "is_live": True}, CurationGate())
    assert not ok and "is_live" in reasons


def test_gate_rejects_offdomain_and_meta_titles() -> None:
    gate = CurationGate()
    for bad_title in (
        "FPV Chasing a Racing Jet Ski at Full Speed",
        "FPV drone chasing waterski jumpers on lake",
        "Using an FPV Drone to Chase Snowmobiles!",
        "What's the best drone for skiing? Part 1",
        "DJI NEO 2 vs HoverAir X1 Pro Max: Skiing Test",
        "How to Film Skiing GoPro | Guide to Mounts",
        # subject-free egocentric POV — wrong viewpoint for a follow drone
        "My Top 5 Ski Runs | GoPro POV [4K]",
        "Powerhouse Peak | Full POV Descent",
        "Kitzbuhel Downhill helmet camera",
        "Skiing a New Chute (4k raw POV)",
    ):
        ok, reasons = gate_candidate({**GOOD, "title": bad_title}, gate)
        assert not ok, f"should reject: {bad_title}"
        assert any("title~" in r for r in reasons)


def test_gate_keeps_follow_cam_titles() -> None:
    """Follow-cam (subject in frame) survives; 'first person ever to ski' is NOT a POV title."""
    gate = CurationGate()
    for good_title in (
        "Cinematic FPV Skiing in Hintertux (Drone epic shots)",
        "FPV Drones Chase Snowboarders Showing a New Angle",
        "Chased Skiers with my FPV Drone!",
        "SKI CHASE with DJI FPV Drone in CHILE",
        "First person ever to ski down the Matterhorn",  # achievement phrase, not POV
    ):
        ok, _ = gate_candidate({**GOOD, "title": good_title}, gate)
        assert ok, f"should keep: {good_title}"


def test_normalize_title() -> None:
    assert normalize_title("FPV: Ski-Run!! (4K)") == "fpv ski run 4k"


def test_title_similarity_bounds() -> None:
    assert title_similarity("fpv ski run", "fpv ski run") == 1.0
    assert title_similarity("fpv ski run", "cooking pasta tutorial") < 0.5


def test_dedup_exact_id() -> None:
    cands = [{"id": "x", "title": "A"}, {"id": "x", "title": "B"}]
    kept, dropped = dedup_candidates(cands)
    assert len(kept) == 1 and dropped[0]["_drop"] == "dup-id"


def test_dedup_against_existing_ids() -> None:
    cands = [{"id": "x", "title": "A"}]
    kept, dropped = dedup_candidates(cands, existing_ids={"x"})
    assert kept == [] and len(dropped) == 1


def test_dedup_fuzzy_title() -> None:
    cands = [
        {"id": "1", "title": "FPV Drone Skiing Chase 4K"},
        {"id": "2", "title": "FPV Drone Skiing Chase 4K!!"},  # near-identical
    ]
    kept, dropped = dedup_candidates(cands)
    assert len(kept) == 1 and dropped[0]["_drop"] == "dup-title/channel"


def test_dedup_channel_plus_duration() -> None:
    cands = [
        {"id": "1", "title": "Morning run", "channel": "c", "duration": 100},
        {"id": "2", "title": "Totally different name", "channel": "c", "duration": 101},  # reupload
    ]
    kept, dropped = dedup_candidates(cands, duration_tol_s=2.0)
    assert len(kept) == 1 and dropped[0]["_drop"] == "dup-title/channel"


def test_dedup_keeps_distinct() -> None:
    cands = [
        {"id": "1", "title": "FPV skiing alps", "channel": "a", "duration": 100},
        {"id": "2", "title": "POV snowboard japan", "channel": "b", "duration": 300},
    ]
    kept, _ = dedup_candidates(cands)
    assert len(kept) == 2


def test_candidate_to_entry_schema() -> None:
    e = candidate_to_entry(GOOD, "cand01")
    assert e["url"] == "https://www.youtube.com/watch?v=a"
    assert e["clip_id"] == "cand01"
    assert e["sport"] == "skiing"
    assert "1080p" in e["notes"] and "120s" in e["notes"] and "30fps" in e["notes"]
