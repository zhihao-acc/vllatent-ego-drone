"""Batch DINOv3 encoding for sports frames (TORCH tier) — Phase B1 step 8.

Wraps ``vllatent.encode.dinov3.DinoV3Encoder.encode_rgb()`` for batch
processing of extracted video frames. Lazy torch import.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from vllatent.schemas import EMBED_DIM, LATENT_DTYPE, PATCH_TOKENS


def encode_frames(
    frame_dir: str | Path,
    *,
    model_id: str = "vit_base_patch16_dinov3.lvd1689m",
    device: str = "cuda",
    batch_report_interval: int = 50,
) -> np.ndarray:
    """Encode all JPEG frames in a directory to DINOv3 latents.

    Parameters
    ----------
    frame_dir : directory containing sequential ``%06d.jpg`` frames
    model_id : timm model id for DINOv3
    device : "cuda" or "cpu"
    batch_report_interval : print progress every N frames

    Returns
    -------
    latents : (N, 196, 768) fp16 — DINOv3 patch tokens per frame
    """
    from vllatent.encode.dinov3 import DinoV3Encoder

    fdir = Path(frame_dir)
    paths = sorted(fdir.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No .jpg frames in {fdir}")

    encoder = DinoV3Encoder(model_id=model_id, device=device)

    n = len(paths)
    latents = np.empty((n, PATCH_TOKENS, EMBED_DIM), dtype=LATENT_DTYPE)

    for i, p in enumerate(paths):
        frame_rgb = _load_rgb(p)
        latents[i] = encoder.encode_rgb(frame_rgb)
        if batch_report_interval > 0 and (i + 1) % batch_report_interval == 0:
            import sys
            print(f"  encoded {i + 1}/{n} frames", file=sys.stderr)

    return latents


def _load_rgb(path: Path) -> np.ndarray:
    """Load a JPEG as RGB uint8 (lazy cv2, fallback PIL)."""
    try:
        import cv2
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"cv2 failed to read {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except ImportError:
        pass
    from PIL import Image
    return np.array(Image.open(str(path)).convert("RGB"))


__all__ = ["encode_frames"]
