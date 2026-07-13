"""Tests for the B3 Stage-0 command harness."""
from __future__ import annotations

from scripts.run_stage0_gates import parse_args


def test_stage0_harness_defaults_to_strict_person_windows() -> None:
    args = parse_args(["--cache-dir", "cache"])
    assert args.strict_person_windows


def test_stage0_harness_allows_explicit_legacy_windows() -> None:
    args = parse_args(["--cache-dir", "cache", "--no-strict-person-windows"])
    assert not args.strict_person_windows
