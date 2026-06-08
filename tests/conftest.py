"""pytest config: auto-skip torch/sim/siblings-marked tests when their deps are absent.

  @pytest.mark.torch    -> skipped unless `import torch` succeeds (default CI is torch-free)
  @pytest.mark.sim      -> skipped unless `import airsim` succeeds (runs only in fly0-m1)
  @pytest.mark.siblings -> always skipped in Phases A-C (standalone; fly0 is Phase D)

This keeps the default CI lane green (pure tier only) while the dev box / H20 /
fly0-m1 run the fuller suites.
"""
from __future__ import annotations

import importlib.util

import pytest


def _have(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_torch = pytest.mark.skip(reason="torch extra not installed (run `make test-torch` on a torch box)")
    skip_sim = pytest.mark.skip(reason="airsim not installed (sim tier runs only in fly0-m1)")
    skip_siblings = pytest.mark.skip(reason="Phases A-C are standalone; fly0 sibling is Phase D")
    have_torch = _have("torch")
    have_sim = _have("airsim")
    for item in items:
        if "torch" in item.keywords and not have_torch:
            item.add_marker(skip_torch)
        if "sim" in item.keywords and not have_sim:
            item.add_marker(skip_sim)
        if "siblings" in item.keywords:
            item.add_marker(skip_siblings)
