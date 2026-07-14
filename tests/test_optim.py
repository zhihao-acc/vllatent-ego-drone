"""B1.22a tests: AdamW decay / no-decay parameter groups (TORCH tier)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vllatent.train.optim import build_param_groups  # noqa: E402

pytestmark = pytest.mark.torch


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(8, 4)          # weight ndim2 -> decay; bias ndim1 -> no_decay
        self.norm = torch.nn.LayerNorm(4)        # weight+bias ndim1 -> no_decay
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, 4))  # "embed" in name -> no_decay


def _ids(group):
    return {id(p) for p in group["params"]}


def test_two_groups_with_correct_wd() -> None:
    m = _Tiny()
    groups = build_param_groups(m, weight_decay=0.05)
    assert len(groups) == 2
    decay = next(g for g in groups if g["weight_decay"] == 0.05)
    no_decay = next(g for g in groups if g["weight_decay"] == 0.0)
    assert id(m.fc.weight) in _ids(decay)
    assert id(m.fc.bias) in _ids(no_decay)
    assert id(m.norm.weight) in _ids(no_decay)
    assert id(m.norm.bias) in _ids(no_decay)
    assert id(m.pos_embed) in _ids(no_decay)


def test_every_param_assigned_once() -> None:
    m = _Tiny()
    groups = build_param_groups(m, weight_decay=0.05)
    assigned = [id(p) for g in groups for p in g["params"]]
    all_params = [id(p) for p in m.parameters()]
    assert sorted(assigned) == sorted(all_params)
    assert len(assigned) == len(set(assigned)), "a param landed in two groups"


def test_requires_grad_false_excluded() -> None:
    m = _Tiny()
    m.fc.weight.requires_grad_(False)
    groups = build_param_groups(m, weight_decay=0.05)
    assigned = {id(p) for g in groups for p in g["params"]}
    assert id(m.fc.weight) not in assigned
