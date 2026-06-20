"""Batch DINOv3 frame encoding (TORCH tier) — loop over sorted JPEGs in a directory.

torch / timm imports are LAZY (inside the function body) so a torch-free box imports this module
without crashing. The pure tier NEVER imports this module.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from vllatent.schemas import EMBED_DIM, LATENT_DTYPE, PATCH_TOKENS


def encode_frames(frames_dir: str | Path, *, device: str = "cuda") -> np.ndarray:
    """Encode all ``*.jpg`` frames in *frames_dir* -> ``(N, 196, 768)`` fp16 numpy array."""
    from vllatent.encode.dinov3 import DinoV3Encoder
    from vllatent.io import load_rgb

    d = Path(frames_dir)
    if not d.exists():
        raise ValueError(f"frames_dir does not exist: {d}")

    paths = sorted(d.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No .jpg files found in {d}")

    encoder = DinoV3Encoder(device=device)
    latents = np.empty((len(paths), PATCH_TOKENS, EMBED_DIM), dtype=LATENT_DTYPE)
    for i, p in enumerate(paths):
        frame = load_rgb(p)
        latents[i] = encoder.encode_rgb(frame)

    return latents


__all__ = ["encode_frames"]
