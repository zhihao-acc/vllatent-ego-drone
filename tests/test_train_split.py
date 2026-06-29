"""B1.22a tests: scene-split by source video (PURE — no torch).

The #1 leak risk in this pipeline is splitting sub-clips of one source video across
train/val. ``split_clips_by_source`` must hold out WHOLE sources.
"""
from __future__ import annotations

import pytest

from vllatent.data.sports_loader import clip_source, split_clips_by_source

STEMS = [
    "ski03_fpv00_c000", "ski03_fpv00_c001", "ski03_fpv01_c000",  # source ski03 (3 sub-clips)
    "ski05_fpv00_c000", "ski05_fpv00_c001",                      # source ski05 (2)
    "ski07_fpv00_c000",                                          # source ski07 (1)
    "ski09_fpv00_c000", "ski09_fpv00_c002",                      # source ski09 (2)
]


def test_clip_source_parses_prefix() -> None:
    assert clip_source("ski03_fpv00_c000") == "ski03"
    assert clip_source("ski12_fpv05_c017") == "ski12"
    assert clip_source("noUnderscore") == "noUnderscore"


def test_no_source_leak_across_split() -> None:
    train, val = split_clips_by_source(STEMS, val_frac=0.25, seed=1)
    train_sources = {clip_source(s) for s in train}
    val_sources = {clip_source(s) for s in val}
    assert train_sources.isdisjoint(val_sources), "a source leaked across train/val"


def test_split_is_a_partition() -> None:
    train, val = split_clips_by_source(STEMS, val_frac=0.5, seed=3)
    assert set(train) | set(val) == set(STEMS)
    assert set(train).isdisjoint(set(val))


def test_whole_source_held_out() -> None:
    """Every sub-clip of a held-out source goes to val (none stays in train)."""
    train, val = split_clips_by_source(STEMS, val_frac=0.25, seed=1)
    val_sources = {clip_source(s) for s in val}
    for src in val_sources:
        in_train = [s for s in train if clip_source(s) == src]
        assert not in_train, f"source {src} split across train+val"


def test_val_frac_zero_gives_empty_val() -> None:
    train, val = split_clips_by_source(STEMS, val_frac=0.0)
    assert val == []
    assert sorted(train) == sorted(STEMS)


def test_single_source_cannot_split() -> None:
    one = ["ski03_fpv00_c000", "ski03_fpv01_c000"]
    train, val = split_clips_by_source(one, val_frac=0.5)
    assert val == []
    assert sorted(train) == sorted(one)


def test_at_least_one_train_source() -> None:
    """Even a high val_frac leaves >= 1 train source."""
    train, _ = split_clips_by_source(STEMS, val_frac=0.99, seed=7)
    assert train, "train set emptied"
    assert {clip_source(s) for s in train}


def test_outputs_sorted() -> None:
    train, val = split_clips_by_source(STEMS, val_frac=0.5, seed=2)
    assert train == sorted(train)
    assert val == sorted(val)


def test_deterministic_with_seed() -> None:
    a = split_clips_by_source(STEMS, val_frac=0.25, seed=42)
    b = split_clips_by_source(STEMS, val_frac=0.25, seed=42)
    assert a == b


def test_bad_val_frac_raises() -> None:
    with pytest.raises(ValueError, match="val_frac"):
        split_clips_by_source(STEMS, val_frac=1.0)
    with pytest.raises(ValueError, match="val_frac"):
        split_clips_by_source(STEMS, val_frac=-0.1)
