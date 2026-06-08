"""Frozen DINOv3 ViT-B/16 encoder wrapper (TORCH tier) — Phase-A step 7.

RGB 224x224 -> last-layer patch tokens (196, 768) fp16, frozen (eval/no_grad/fp16).
Enforces foot-gun #2: input MUST be RGB; the render->encode boundary applies
cv2.cvtColor(BGR2RGB) and records the flag in the cache manifest.

torch/transformers/timm imports are LAZY (inside __init__/forward) so importing
this module on a torch-free box (CI) does not crash. STUB at scaffold time;
implemented in step 7. Weights via HF_ENDPOINT=https://hf-mirror.com.

See plans/phase-a-data-and-io-contract.md step 7.
"""
from __future__ import annotations

PATCH_TOKENS = 196
EMBED_DIM = 768
INPUT_HW = (224, 224)
MODEL_ID = "facebook/dinov3-vitb16"


class DinoV3Encoder:  # pragma: no cover - implemented in step 7
    """Frozen DINOv3 ViT-B/16. Lazy-imports torch on construction."""

    def __init__(self, model_id: str = MODEL_ID, device: str = "cuda") -> None:
        raise NotImplementedError("DinoV3Encoder lands in Phase-A step 7")


__all__ = ["DinoV3Encoder", "PATCH_TOKENS", "EMBED_DIM", "INPUT_HW", "MODEL_ID"]
