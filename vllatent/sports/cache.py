"""Sports cache assembly (ORCH tier) — Phase B1 step 8.

Assembles per-clip ``.npz`` files from encoded latents, MegaSaM deltas,
and quality scores. Writes a sports-specific manifest.

On-disk ``.npz`` format per clip (N frames):

    latents       (N, 196, 768) fp16  — DINOv3 patch tokens
    deltas        (N-1, 4)      f32   — body-frame (dx,dy,dz,dyaw)
    vo_confidence (N,)          f32   — MegaSaM per-frame confidence
    frame_quality (N,)          f32   — composite quality score
    timestamps    (N,)          f64   — frame timestamps in seconds
    quality_mask  (N,)          bool  — True = frame passes quality filter
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from vllatent.schemas import DELTA_DTYPE, LATENT_DTYPE, MASK_DTYPE, PATCH_TOKENS, EMBED_DIM


def build_clip_npz(
    *,
    latents: np.ndarray,
    deltas: np.ndarray,
    vo_confidence: np.ndarray,
    frame_quality: np.ndarray,
    timestamps: np.ndarray,
    quality_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Validate and return the arrays dict for a single clip's .npz.

    Does NOT write to disk — the caller decides the path.
    """
    n = latents.shape[0]
    if latents.shape != (n, PATCH_TOKENS, EMBED_DIM):
        raise ValueError(f"latents: expected (N, {PATCH_TOKENS}, {EMBED_DIM}), got {latents.shape}")
    if deltas.shape != (n - 1, 4):
        raise ValueError(f"deltas: expected ({n - 1}, 4), got {deltas.shape}")
    if vo_confidence.shape != (n,):
        raise ValueError(f"vo_confidence: expected ({n},), got {vo_confidence.shape}")
    if frame_quality.shape != (n,):
        raise ValueError(f"frame_quality: expected ({n},), got {frame_quality.shape}")
    if timestamps.shape != (n,):
        raise ValueError(f"timestamps: expected ({n},), got {timestamps.shape}")
    if quality_mask.shape != (n,):
        raise ValueError(f"quality_mask: expected ({n},), got {quality_mask.shape}")

    return {
        "latents": latents.astype(LATENT_DTYPE),
        "deltas": deltas.astype(DELTA_DTYPE),
        "vo_confidence": vo_confidence.astype(np.float32),
        "frame_quality": frame_quality.astype(np.float32),
        "timestamps": timestamps.astype(np.float64),
        "quality_mask": quality_mask.astype(MASK_DTYPE),
    }


def write_clip_npz(
    arrays: dict[str, np.ndarray],
    out_path: str | Path,
) -> Path:
    """Write a clip's arrays to a .npz file."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(p), **arrays)
    return p


def build_sports_manifest(
    *,
    encoder_model_id: str,
    encoder_dtype: str = "float16",
    sport: str = "skiing",
    megasam_model: str = "megasam_base",
    scale_mode: str = "normalized",
    source_fps: float = 5.0,
    entries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a sports-following cache manifest."""
    return {
        "cache_version": "0.2",
        "encoder": {
            "model_id": encoder_model_id,
            "dtype": encoder_dtype,
            "patch_tokens": PATCH_TOKENS,
            "dim": EMBED_DIM,
        },
        "dataset": {
            "name": "sports_following",
            "sport": sport,
            "license": "fair-use-research",
        },
        "convention": {
            "color_order": "RGB",
            "frame": "camera_body",
        },
        "motion_source": {
            "method": "megasam",
            "model": megasam_model,
            "scale_mode": scale_mode,
            "source_fps": source_fps,
        },
        "entries": list(entries) if entries is not None else [],
    }


_REQUIRED_SPORTS_MANIFEST = {"cache_version", "encoder", "dataset", "convention", "motion_source", "entries"}
_REQUIRED_SPORTS_ENTRY = {"clip_id", "n_frames", "latent_path"}


def validate_sports_manifest(data: dict[str, Any]) -> list[str]:
    """Validate a sports cache manifest. Returns list of errors (empty = valid)."""
    errors: list[str] = []

    missing_top = _REQUIRED_SPORTS_MANIFEST - set(data)
    if missing_top:
        errors.append(f"missing top-level keys: {sorted(missing_top)}")

    enc = data.get("encoder")
    if isinstance(enc, dict):
        for k in ("model_id", "dtype", "patch_tokens", "dim"):
            if k not in enc:
                errors.append(f"encoder missing key: {k}")

    ds = data.get("dataset")
    if isinstance(ds, dict):
        if ds.get("name") != "sports_following":
            errors.append(f"dataset.name: expected 'sports_following', got {ds.get('name')!r}")

    ms = data.get("motion_source")
    if isinstance(ms, dict):
        for k in ("method", "model", "scale_mode", "source_fps"):
            if k not in ms:
                errors.append(f"motion_source missing key: {k}")

    entries = data.get("entries")
    if isinstance(entries, list):
        for i, e in enumerate(entries):
            if not isinstance(e, dict):
                errors.append(f"entries[{i}]: expected dict")
                continue
            for k in _REQUIRED_SPORTS_ENTRY:
                if k not in e:
                    errors.append(f"entries[{i}] missing key: {k}")

    return errors


def write_sports_manifest(data: dict[str, Any], out_dir: str | Path) -> Path:
    """Write ``<out_dir>/manifest.json``."""
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    path = p / "manifest.json"
    path.write_text(json.dumps(data, indent=2, sort_keys=False))
    return path


__all__ = [
    "build_clip_npz",
    "write_clip_npz",
    "build_sports_manifest",
    "validate_sports_manifest",
    "write_sports_manifest",
]
