"""Tests for the B3 Stage-1 training harness helpers."""
from __future__ import annotations

import pytest

from scripts.train_sports_b3 import limit_indices, source_split_indices


def test_source_split_indices_splits_by_source() -> None:
    sources = ["a", "a", "b", "b", "c", "c", "d", "d"]
    split = source_split_indices(sources, val_frac=0.25, seed=0)
    train_sources = {sources[i] for i in split.train}
    val_sources = {sources[i] for i in split.val}
    assert train_sources
    assert val_sources
    assert train_sources.isdisjoint(val_sources)
    assert set(split.train_sources) == train_sources
    assert set(split.val_sources) == val_sources


def test_source_split_rejects_one_source() -> None:
    with pytest.raises(ValueError, match="two sources"):
        source_split_indices(["a", "a"], val_frac=0.25)


def test_limit_indices_is_deterministic() -> None:
    indices = list(range(20))
    a = limit_indices(indices, 5, seed=3)
    b = limit_indices(indices, 5, seed=3)
    assert a == b
    assert len(a) == 5
    assert set(a).issubset(indices)
