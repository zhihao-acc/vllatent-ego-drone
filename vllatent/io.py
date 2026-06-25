"""I/O utilities (TORCH tier — lazy cv2 import)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_rgb(path: str | Path) -> np.ndarray:
    """Load an image file as an RGB uint8 numpy array (H, W, 3)."""
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


__all__ = ["load_rgb"]
