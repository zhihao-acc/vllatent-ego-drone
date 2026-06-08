"""Cached-latent CACHE MANIFEST — serialization + validation (PURE tier).

Cached latents are **render-once**: RGB is rendered from the AirSim sim at each
ground-truth pose, encoded by the frozen DINOv3 ViT-B/16, and written to disk as
fp16 patch tokens. A cache without a manifest is unauditable, so EVERY cache build
writes/updates one. The manifest pins exactly the things that make a cached latent
reproducible — the encoder identity, the dataset slice, and the two data foot-guns
(quaternion order + colour order).

This module imports stdlib only (no numpy needed, no torch, no sibling), so it is
CI-importable and provides the ``--validate`` / ``--emit-empty`` CLI used by CI.
The typed dataclasses live in ``vllatent.schemas`` (Phase-A step 3); this module
validates plain dicts so it works independently of that step.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

CACHE_VERSION = "0.1"

# Required top-level keys of a cache manifest, with their expected JSON types.
_REQUIRED: dict[str, type] = {
    "cache_version": str,
    "encoder": dict,      # {model_id, revision, dtype, patch_tokens, dim}
    "dataset": dict,      # {name, variant, split, license}
    "convention": dict,   # {quaternion_order, color_order, frame}
    "entries": list,      # list of per-episode entries
}

_REQUIRED_ENCODER = {"model_id", "revision", "dtype", "patch_tokens", "dim"}
_REQUIRED_CONVENTION = {"quaternion_order", "color_order", "frame"}


def empty_manifest() -> dict[str, Any]:
    """A minimal VALID cache manifest (empty entry list), for round-trip CLI tests."""
    return {
        "cache_version": CACHE_VERSION,
        "encoder": {
            "model_id": "facebook/dinov3-vitb16",
            "revision": "",
            "dtype": "float16",
            "patch_tokens": 196,
            "dim": 768,
        },
        "dataset": {
            "name": "aerialvln",
            "variant": "",
            "split": "",
            "license": "CC BY-NC-SA 4.0",
        },
        "convention": {
            # The two data foot-guns, pinned in the manifest so a cache is auditable.
            "quaternion_order": "xyzw",  # canonical order latents were rendered under
            "color_order": "RGB",        # BGR->RGB applied before the encoder
            "frame": "airsim_ned_body",
        },
        "entries": [],
    }


def validate_manifest(data: dict[str, Any]) -> list[str]:
    """Return a list of validation errors (empty == valid)."""
    errors: list[str] = []
    for key, typ in _REQUIRED.items():
        if key not in data:
            errors.append(f"missing key: {key}")
        elif not isinstance(data[key], typ):
            errors.append(f"key {key}: expected {typ.__name__}, got {type(data[key]).__name__}")

    enc = data.get("encoder")
    if isinstance(enc, dict):
        missing = _REQUIRED_ENCODER - set(enc)
        if missing:
            errors.append(f"encoder missing keys: {sorted(missing)}")

    conv = data.get("convention")
    if isinstance(conv, dict):
        missing = _REQUIRED_CONVENTION - set(conv)
        if missing:
            errors.append(f"convention missing keys: {sorted(missing)}")
        if conv.get("color_order") not in (None, "RGB", "BGR"):
            errors.append(f"convention.color_order must be RGB|BGR, got {conv.get('color_order')!r}")
        if conv.get("quaternion_order") not in (None, "xyzw", "wxyz"):
            errors.append(f"convention.quaternion_order must be xyzw|wxyz, got {conv.get('quaternion_order')!r}")

    if isinstance(data.get("entries"), list):
        for i, e in enumerate(data["entries"]):
            if not isinstance(e, dict):
                errors.append(f"entries[{i}] must be an object")
                continue
            for k in ("episode_id", "scene_id", "n_frames", "latent_path"):
                if k not in e:
                    errors.append(f"entries[{i}] missing key: {k}")
    return errors


def write_manifest(data: dict[str, Any], out_dir: str | Path) -> Path:
    """Write ``<out_dir>/manifest.json`` (caller owns schema correctness)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "manifest.json"
    path.write_text(json.dumps(data, indent=2, sort_keys=False))
    return path


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="vllatent cache-manifest tool")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--validate", metavar="PATH", help="validate a manifest JSON ('-' = stdin)")
    g.add_argument("--emit-empty", action="store_true", help="print a minimal valid manifest to stdout")
    args = p.parse_args(argv)

    if args.emit_empty:
        print(json.dumps(empty_manifest(), indent=2))
        return 0

    raw = sys.stdin.read() if args.validate == "-" else Path(args.validate).read_text()
    errors = validate_manifest(json.loads(raw))
    if errors:
        for e in errors:
            print(f"INVALID: {e}", file=sys.stderr)
        return 1
    print("manifest: OK")
    return 0


__all__ = ["CACHE_VERSION", "empty_manifest", "validate_manifest", "write_manifest"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
