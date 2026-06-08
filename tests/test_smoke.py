"""Scaffold smoke tests (PURE tier) — green from commit 1.

Asserts the pure modules import with numpy/pyyaml only and that the cache-manifest
round-trip + frame-order constants hold. The deep tests (test_schemas, test_actions,
test_frames, test_audit, ...) land with their Phase-A steps.
"""
from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "mod",
    ["vllatent.schemas", "vllatent.actions", "vllatent.frames",
     "vllatent.config", "vllatent.manifest", "vllatent.audit"],
)
def test_pure_modules_import(mod: str) -> None:
    importlib.import_module(mod)


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
