"""Tests for the B3 Stage-1 training harness helpers."""

from __future__ import annotations

import pytest

from scripts import train_sports_b3 as harness_module
from scripts.train_sports_b3 import (
    limit_indices,
    loss_window_improvement,
    parse_args,
    select_train_val_indices,
    source_split_indices,
)


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


def test_overfit_tiny_validation_reuses_exact_train_indices() -> None:
    split = source_split_indices(["a", "a", "b", "b", "c", "c", "d", "d"], val_frac=0.25, seed=0)
    train_indices, val_indices = select_train_val_indices(
        split,
        train_max_samples=3,
        val_max_samples=3,
        overfit_tiny=True,
        seed=5,
    )
    assert val_indices == train_indices


def test_non_overfit_validation_stays_source_disjoint() -> None:
    sources = ["a", "a", "b", "b", "c", "c", "d", "d"]
    split = source_split_indices(sources, val_frac=0.25, seed=0)
    train_indices, val_indices = select_train_val_indices(
        split,
        train_max_samples=None,
        val_max_samples=None,
        overfit_tiny=False,
        seed=5,
    )
    assert {sources[i] for i in train_indices}.isdisjoint({sources[i] for i in val_indices})


def test_loss_window_improvement_uses_window_means() -> None:
    initial, final, improvement = loss_window_improvement([10.0, 6.0, 4.0, 2.0], window=2)
    assert initial == pytest.approx(8.0)
    assert final == pytest.approx(3.0)
    assert improvement == pytest.approx(0.625)


def test_loss_window_improvement_handles_short_runs() -> None:
    assert loss_window_improvement([]) == (None, None, None)
    assert loss_window_improvement([1.0]) == (None, None, None)


def test_b3_harness_defaults_to_strict_person_windows() -> None:
    args = parse_args([])
    assert args.strict_person_windows


def test_source_balanced_limit_covers_all_sources_when_budget_allows() -> None:
    sample_sources = ["dominant"] * 100 + ["small"] * 2 + ["singleton"]
    indices = list(range(len(sample_sources)))

    first = harness_module.source_balanced_limit_indices(
        indices,
        sample_sources,
        max_samples=9,
        seed=7,
    )
    second = harness_module.source_balanced_limit_indices(
        indices,
        sample_sources,
        max_samples=9,
        seed=7,
    )

    assert first == second
    assert len(first) == 9
    assert set(first).issubset(indices)
    assert {sample_sources[index] for index in first} == {
        "dominant",
        "small",
        "singleton",
    }


def test_represented_sources_for_report_uses_only_selected_indices() -> None:
    sample_sources = ["a", "a", "b", "b", "c", "c"]
    nominal_split_sources = ["a", "b", "c"]
    selected_indices = [0, 1, 4]

    represented = harness_module.represented_sources(
        sample_sources,
        selected_indices,
    )

    assert represented == ["a", "c"]
    assert represented != nominal_split_sources


def test_real_transition_previous_latents_uses_anchor_then_real_targets() -> None:
    torch = pytest.importorskip("torch")
    z_t = torch.full((2, 196, 4), -1.0)
    target = torch.stack(
        [torch.full((2, 196, 4), float(step)) for step in range(1, 4)],
        dim=1,
    )

    previous = harness_module.real_transition_previous_latents(z_t, target)

    assert previous.shape == target.shape
    assert torch.equal(previous[:, 0], z_t)
    assert torch.equal(previous[:, 1:], target[:, :-1])
