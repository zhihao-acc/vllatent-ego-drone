"""Scaffold smoke tests (PURE tier) — green from commit 1.

Asserts the pure modules import with numpy/pyyaml only and that the cache-manifest
round-trip + frame-order constants hold. The deep tests (test_schemas, test_actions,
test_frames, test_audit, ...) land with their Phase-A steps.
"""
from __future__ import annotations

import ast
import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize(
    "mod",
    ["vllatent.schemas", "vllatent.actions", "vllatent.frames",
     "vllatent.config", "vllatent.manifest", "vllatent.audit"],
)
def test_pure_modules_import(mod: str) -> None:
    importlib.import_module(mod)


def test_encode_module_imports_torch_free() -> None:
    """The TORCH-tier DINOv3 wrapper must import on a torch-free box (lazy guard, A5.10).

    This runs in the PURE gate (`make test`, no torch marker) — the box where the guarantee
    matters. Importing the module must NOT pull torch/transformers, and the pure-numpy
    BGR->RGB boundary helper must work standalone.
    """
    dinov3 = importlib.import_module("vllatent.encode.dinov3")
    rgb = dinov3.bgr_to_rgb(np.array([[[1, 2, 3]]], dtype=np.uint8))  # B=1,G=2,R=3 -> R,G,B
    assert rgb[0, 0, 0] == 3 and rgb[0, 0, 1] == 2 and rgb[0, 0, 2] == 1
    assert rgb.flags["C_CONTIGUOUS"]

    # Structural guard: NO module-level torch/transformers/timm/cv2 import (must stay lazy).
    heavy = {"torch", "transformers", "timm", "cv2", "einops"}
    tree = ast.parse(Path(dinov3.__file__).read_text())
    for node in tree.body:  # module scope only — function-local imports are allowed (lazy)
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names = [node.module]
        for n in names:
            assert n.split(".")[0] not in heavy, f"module-level heavy import {n!r} breaks tier purity"


def test_manifest_roundtrip_is_valid() -> None:
    from vllatent.manifest import empty_manifest, validate_manifest
    assert validate_manifest(empty_manifest()) == []


def test_manifest_rejects_missing_keys() -> None:
    from vllatent.manifest import validate_manifest
    errors = validate_manifest({"cache_version": "0.1"})
    assert any("missing key" in e for e in errors)


def test_frame_order_constants() -> None:
    from vllatent import frames
    # Foot-gun #1, pinned: start_rotation is a w-FIRST quaternion; reference_path is EULER
    # [x,y,z,pitch,roll,yaw] (radians, 6-wide, yaw=row[5]) — NOT a quaternion (confirmed step 5b).
    assert frames.QUAT_ORDER_START_ROTATION == "wxyz"
    assert frames.QUAT_ORDER_CANONICAL == "xyzw"
    assert frames.REFERENCE_PATH_ORIENTATION == "euler_pitch_roll_yaw_rad"
    assert frames.REFERENCE_PATH_ROW_WIDTH == 6
    assert frames.REFERENCE_PATH_YAW_INDEX == 5
