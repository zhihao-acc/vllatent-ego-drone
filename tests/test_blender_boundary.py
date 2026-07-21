"""Static isolation guard for the B3-CS3 Blender bridge."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    roots.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    return roots


def test_bpy_is_isolated_to_the_single_blender_bridge() -> None:
    bridge = REPO_ROOT / "scripts" / "blender" / "b3_cs3_bridge.py"
    assert "bpy" in _imports(bridge)
    for path in sorted((REPO_ROOT / "vllatent" / "sim").glob("*.py")):
        assert "bpy" not in _imports(path), path
    bpy_importers = [
        path
        for root in (REPO_ROOT / "vllatent", REPO_ROOT / "scripts")
        for path in root.rglob("*.py")
        if "bpy" in _imports(path)
    ]
    assert bpy_importers == [bridge]


def test_bridge_has_no_model_controller_network_or_stateful_time_dependencies() -> None:
    bridge = REPO_ROOT / "scripts" / "blender" / "b3_cs3_bridge.py"
    imported = _imports(bridge)
    assert imported.isdisjoint(
        {
            "airsim",
            "cv2",
            "datetime",
            "requests",
            "socket",
            "subprocess",
            "time",
            "timm",
            "torch",
            "torchvision",
            "transformers",
            "urllib",
        }
    )
    source = bridge.read_text(encoding="utf-8")
    assert source.startswith("# SPDX-License-Identifier: GPL-3.0-or-later")
    assert "exec(" not in source
    assert "eval(" not in source
    assert "python_file_run" not in source
