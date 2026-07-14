"""Cached-latent CACHE MANIFEST — typed builder + validation (PURE tier).

Cached latents are **render-once**: RGB is rendered from the AirSim sim at each
ground-truth pose, encoded by the frozen DINOv3 ViT-B/16, and written to disk as
fp16 patch tokens. A cache without a manifest is unauditable, so EVERY cache build
writes/updates one. The manifest pins exactly the things that make a cached latent
reproducible — the encoder identity, the dataset slice, and the two data foot-guns
(quaternion order + colour order).

This module is PURE tier (numpy/pyyaml/stdlib): it builds the manifest from the
typed ``vllatent.config.Config`` — the single source of truth — so the encoder id,
dtype, cache version, and conventions are NOT re-hardcoded here, and the fixed
DINOv3 shapes (196/768) come from ``vllatent.schemas`` constants (M5 de-dup). The
per-entry required keys are derived from ``CacheManifestEntry`` so the validator is
type-enforced, not hand-kept in sync. Provides the ``--validate`` / ``--emit-empty``
CLI used by CI.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from vllatent.config import CacheConfig, Config
from vllatent.schemas import EMBED_DIM, PATCH_TOKENS, CacheManifestEntry

# Cache-manifest schema version. The single literal lives in CacheConfig.version; this
# derived alias is kept for backward-compatible external reads.
CACHE_VERSION = CacheConfig().version

# Required top-level keys of a cache manifest, with their expected JSON types.
_REQUIRED: dict[str, type] = {
    "cache_version": str,
    "encoder": dict,      # {model_id, revision, dtype, patch_tokens, dim}
    "dataset": dict,      # {name, variant, split, license}
    "convention": dict,   # {quaternion_order, color_order, frame}
    "entries": list,      # list of per-episode entries
}

_REQUIRED_ENCODER = {"model_id", "revision", "dtype", "patch_tokens", "dim"}
_REQUIRED_DATASET = {"name", "variant", "split", "license"}
_REQUIRED_CONVENTION = {"quaternion_order", "color_order", "frame"}


def build_manifest(
    config: Config,
    *,
    split: str = "",
    variant: str = "",
    entries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a cache manifest from the typed ``Config`` (the single source of truth).

    The fixed encoder shapes come from ``vllatent.schemas`` (``PATCH_TOKENS`` / ``EMBED_DIM``)
    — NOT re-hardcoded here. ``split`` / ``variant`` are per-build labels. The retired A5
    teacher/cache fields are no longer emitted by default; historical manifests with extra keys
    still parse because validation ignores unknown fields.
    """
    enc, cache, data = config.encoder, config.cache, config.data
    return {
        "cache_version": cache.version,
        "encoder": {
            "model_id": enc.model_id,
            "revision": "",            # pinned at real-weight load (A5.10/A5.14)
            "dtype": enc.dtype,
            "patch_tokens": PATCH_TOKENS,   # de-dup: the one true 196 (schemas)
            "dim": EMBED_DIM,               # de-dup: the one true 768 (schemas)
        },
        "dataset": {
            "name": data.name,
            "variant": variant,
            "split": split,
            "license": data.license,
        },
        "convention": {
            # The two data foot-guns, pinned in the manifest so a cache is auditable.
            "quaternion_order": cache.quaternion_order,  # canonical order latents rendered under
            "color_order": cache.color_order,            # BGR->RGB applied before the encoder
            "frame": cache.frame,
        },
        "entries": list(entries) if entries is not None else [],
    }


def empty_manifest() -> dict[str, Any]:
    """A minimal VALID cache manifest (default ``Config``, empty entries) for round-trip/CLI."""
    return build_manifest(Config())


_REQUIRED_ENCODER_WILD_VIDEO = {"model_id", "dtype", "patch_tokens", "dim"}


def validate_manifest(data: dict[str, Any]) -> list[str]:
    """Return a list of validation errors (empty == valid)."""
    errors: list[str] = []

    source_type = ""
    if isinstance(data.get("dataset"), dict):
        source_type = data["dataset"].get("source_type", "aerialvln")
    is_wild = source_type == "wild_video"

    required_top = _REQUIRED if not is_wild else {
        k: v for k, v in _REQUIRED.items() if k != "teacher"
    }
    for key, typ in required_top.items():
        if key not in data:
            errors.append(f"missing key: {key}")
        elif not isinstance(data[key], typ):
            errors.append(f"key {key}: expected {typ.__name__}, got {type(data[key]).__name__}")

    enc = data.get("encoder")
    if isinstance(enc, dict):
        req_enc = _REQUIRED_ENCODER_WILD_VIDEO if is_wild else _REQUIRED_ENCODER
        missing = req_enc - set(enc)
        if missing:
            errors.append(f"encoder missing keys: {sorted(missing)}")

    ds = data.get("dataset")
    if isinstance(ds, dict):
        missing = _REQUIRED_DATASET - set(ds)
        if missing:
            errors.append(f"dataset missing keys: {sorted(missing)}")

    conv = data.get("convention")
    if isinstance(conv, dict):
        missing = _REQUIRED_CONVENTION - set(conv)
        if missing:
            errors.append(f"convention missing keys: {sorted(missing)}")
        if conv.get("color_order") not in (None, "RGB", "BGR"):
            errors.append(f"convention.color_order must be RGB|BGR, got {conv.get('color_order')!r}")
        if conv.get("quaternion_order") not in (None, "xyzw", "wxyz"):
            errors.append(f"convention.quaternion_order must be xyzw|wxyz, got {conv.get('quaternion_order')!r}")

    if is_wild:
        ms = data.get("motion_source")
        if not isinstance(ms, dict):
            if "motion_source" not in data:
                errors.append("missing key: motion_source (required for wild_video)")
        else:
            for k in _REQUIRED_MOTION_SOURCE:
                if k not in ms:
                    errors.append(f"motion_source missing key: {k}")
        required_entry_keys: tuple[str, ...] = _REQUIRED_WILD_VIDEO_ENTRY
    else:
        required_entry_keys = CacheManifestEntry.required_keys()

    if isinstance(data.get("entries"), list):
        for i, e in enumerate(data["entries"]):
            if not isinstance(e, dict):
                errors.append(f"entries[{i}] must be an object")
                continue
            for k in required_entry_keys:
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


# ---------------------------------------------------------------------------
# Wild-video ingestion manifest (source_type = "wild_video")
# ---------------------------------------------------------------------------

_REQUIRED_MOTION_SOURCE = {"method", "model", "scale_mode", "source_fps"}
_REQUIRED_WILD_VIDEO_ENTRY = ("clip_id", "n_frames", "latent_path")


def build_manifest_wild_video(
    *,
    encoder_model_id: str,
    encoder_dtype: str = "float16",
    motion_method: str = "megasam",
    motion_model: str = "megasam_base",
    scale_mode: str = "normalized",
    source_fps: float = 5.0,
    person_tracker: dict[str, Any] | None = None,
    entries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a cache manifest for wild-video ingestion (source_type=wild_video)."""
    manifest = {
        "cache_version": CACHE_VERSION,
        "encoder": {
            "model_id": encoder_model_id,
            "dtype": encoder_dtype,
            "patch_tokens": PATCH_TOKENS,
            "dim": EMBED_DIM,
        },
        "dataset": {
            "name": "wild_video",
            "source_type": "wild_video",
            "variant": "",
            "split": "",
            "license": "fair-use-research",
        },
        "convention": {
            "quaternion_order": "xyzw",
            "color_order": "RGB",
            "frame": "camera_body",
        },
        "motion_source": {
            "method": motion_method,
            "model": motion_model,
            "scale_mode": scale_mode,
            "source_fps": source_fps,
        },
        "entries": list(entries) if entries is not None else [],
    }
    if person_tracker is not None:
        manifest["person_tracker"] = dict(person_tracker)
    return manifest


__all__ = [
    "CACHE_VERSION",
    "build_manifest",
    "build_manifest_wild_video",
    "empty_manifest",
    "validate_manifest",
    "write_manifest",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
