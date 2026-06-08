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
    # The two data foot-guns, pinned: start_rotation is w-first, reference_path is xyzw.
    assert frames.QUAT_ORDER_START_ROTATION == "wxyz"
    assert frames.QUAT_ORDER_REFERENCE_PATH == "xyzw"
    assert frames.QUAT_ORDER_CANONICAL == "xyzw"
