"""pytest config: auto-skip optional torch/tool tests when deps are absent.

  @pytest.mark.torch    -> skipped unless `import torch` succeeds (default CI is torch-free)
  @pytest.mark.tool     -> skipped unless yt-dlp and ffmpeg are available

This keeps the default CI lane green while optional local suites stay explicit.
"""
from __future__ import annotations

import importlib.util

import pytest


def _have(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _tool_available(name: str) -> bool:
    import shutil
    return shutil.which(name) is not None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_torch = pytest.mark.skip(reason="torch extra not installed (run `make test-torch` on a torch box)")
    skip_tool = pytest.mark.skip(reason="external tools (yt-dlp/ffmpeg/megasam) not available")
    have_torch = _have("torch")
    have_tools = _tool_available("yt-dlp") and _tool_available("ffmpeg")
    for item in items:
        if "torch" in item.keywords and not have_torch:
            item.add_marker(skip_torch)
        if "tool" in item.keywords and not have_tools:
            item.add_marker(skip_tool)
