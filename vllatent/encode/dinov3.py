"""Frozen DINOv3 ViT-B/16 encoder wrapper (TORCH tier) — Phase-A step A5.10 (was step 7).

RGB 224x224 -> last-layer patch tokens ``(196, 768)`` fp16, frozen (eval / no_grad / fp16).
This is the **student's frozen, cached front-end encoder** (NOT the student — the student is the
latent-prediction transformer distilled from WorldVLN; see the re-plan). Phases B+ train on the
cached fp16 latents this wrapper produces; the encoder itself never trains (no EMA / no VICReg —
a frozen cached encoder is a fixed target and cannot collapse).

**Foot-gun #2 (BGR->RGB) is enforced HERE.** AirSim ``Scene`` frames are BGR; DINOv3 expects RGB.
:meth:`DinoV3Encoder.encode_bgr` flips the channels at this render->encode boundary (and the cache
manifest records ``color_order = "RGB"``). :meth:`encode_rgb` is for already-RGB input.

**torch / timm imports are LAZY** — every heavy import lives inside :func:`_load_backbone`
or a method, so a torch-free box (the pure CI lane) imports this module without crashing. The pure
tier (``vllatent.{schemas,actions,frames,config,manifest,audit}``) NEVER imports this module.

The backbone is timm's **non-gated** re-host of Meta's DINOv3 ViT-B/16 (HF
``timm/vit_base_patch16_dinov3.lvd1689m``) — same LVD-1689M weights, **no HF token / no gate** (Meta's
``facebook/dinov3-vitb16-pretrain-lvd1689m`` is gated and rejected our access 2026-06-09).

Tested two ways (A5.10): a monkeypatched-backbone **contract** test (``tests/test_encode_contract.py``,
``@pytest.mark.torch``) that pins the BGR->RGB flip + the ``(196,768)`` fp16 output without real
weights, and a real-weight smoke ``make encode-smoke`` (downloads ~330 MB; no token needed).

See plans/phase-a5-replan-postpivot.md A5.10.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from vllatent.config import EncoderConfig
from vllatent.schemas import EMBED_DIM, LATENT_DTYPE, PATCH_TOKENS

if TYPE_CHECKING:  # type-only; never imported at runtime on the pure box
    import torch

# Single source of truth = vllatent.config (pure tier); the A5.14 cache build passes
# config.encoder.{model_id,input_hw}. DINOv3 ViT-B/16: 224/16 = 14 -> 14*14 = 196 patch tokens,
# each EMBED_DIM (768)-wide. Weights = timm's NON-GATED re-host (no token; see _load_backbone).
_ENCODER_DEFAULTS = EncoderConfig()
MODEL_ID = _ENCODER_DEFAULTS.model_id
INPUT_HW = (_ENCODER_DEFAULTS.input_hw, _ENCODER_DEFAULTS.input_hw)

# A backbone forward maps one RGB uint8 frame (H,W,3) -> last_hidden_state (1, T, 768) torch.Tensor,
# where T = 1 CLS + R register tokens + 196 patches. Injected via _load_backbone so the contract
# test can monkeypatch it without real weights.
BackboneForward = Callable[[np.ndarray], "torch.Tensor"]


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Reverse the channel axis of an ``(H,W,3)`` image — the BGR->RGB boundary (foot-gun #2).

    Pure numpy (no cv2, no torch) so it stays importable + testable on a torch-free box. Returns a
    C-contiguous copy (downstream HF processors dislike negative strides).
    """
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"frame: expected np.ndarray, got {type(frame).__name__}")
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"frame: expected (H,W,3), got shape {frame.shape}")
    return np.ascontiguousarray(frame[..., ::-1])


def _load_backbone(model_id: str, device: str, dtype: str) -> BackboneForward:
    """Lazy-load the frozen DINOv3 ViT-B/16 backbone via **timm**; return its forward callable.

    Uses timm's NON-GATED re-host of Meta's DINOv3 ViT-B/16 (HF ``timm/<model_id>``) — same
    LVD-1689M weights, no gate/token (Meta's gated repo rejected our access 2026-06-09). ALL heavy
    imports (torch / timm) live in here so the module imports on a torch-free box. Monkeypatched in
    the contract test (so no real weights are downloaded in CI). The returned closure maps an RGB
    uint8 ``(H,W,3)`` frame -> token sequence ``(1,T,768)`` (T = 1 CLS + 4 registers + 196 patches).
    """
    import os
    import timm
    import torch
    from timm.data import resolve_model_data_config

    for k in ("ALL_PROXY", "all_proxy"):
        v = os.environ.get(k)
        if v and v.startswith("socks://"):
            os.environ[k] = v.replace("socks://", "socks5://", 1)

    try:
        model = timm.create_model(model_id, pretrained=True, num_classes=0)
    except Exception as exc:  # network / offline — make it actionable
        raise RuntimeError(
            f"Failed to load DINOv3 weights {model_id!r} via timm (HF 'timm/{model_id}'). These are "
            f"NON-GATED — no HF_TOKEN needed. From CN, set HF_ENDPOINT=https://hf-mirror.com if the "
            f"direct download is slow. Original error: {exc}"
        ) from exc
    model = model.to(device).eval()
    if dtype == "float16":
        model = model.half()
    for param in model.parameters():  # frozen — no grads, never trains
        param.requires_grad_(False)

    # timm's canonical normalization for THIS checkpoint (mean/std/input size). Frames are rendered
    # at INPUT_HW already, so the fast path is a pure normalize (resize only if a stray size slips
    # in). Pure torch — no PIL/torchvision dependency.
    data_cfg = resolve_model_data_config(model)
    mean = torch.tensor(data_cfg["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(data_cfg["std"]).view(1, 3, 1, 1)
    target_hw = (int(data_cfg["input_size"][1]), int(data_cfg["input_size"][2]))
    compute_dtype = torch.float16 if dtype == "float16" else torch.float32

    def _forward(frame_rgb: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(np.ascontiguousarray(frame_rgb)).permute(2, 0, 1)[None].float().div_(255.0)
        if t.shape[-2:] != target_hw:
            t = torch.nn.functional.interpolate(t, size=target_hw, mode="bicubic", align_corners=False)
        t = ((t - mean) / std).to(device=device, dtype=compute_dtype)
        with torch.no_grad():
            tokens = model.forward_features(t)  # (1, T, 768)
        return tokens

    return _forward


class DinoV3Encoder:
    """Frozen DINOv3 ViT-B/16. Lazy-loads torch on construction; emits ``(196,768)`` fp16 latents.

    The constructor does the heavy load (real weights). Tests monkeypatch
    :func:`_load_backbone` to inject a fake forward, so the encode contract is exercised without
    any download. ``device`` defaults to ``"cuda"`` (the dev box / H20); pass ``"cpu"`` for a
    weightless contract test.
    """

    def __init__(self, model_id: str = MODEL_ID, device: str = "cuda", dtype: str = "float16") -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self._forward: BackboneForward = _load_backbone(model_id, device, dtype)

    def encode_bgr(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Encode a BGR frame (the AirSim ``Scene`` convention) — flips BGR->RGB then encodes."""
        # Foot-gun #2: AirSim Scene is BGR; DINOv3 expects RGB. Convert at THIS boundary.
        return self.encode_rgb(bgr_to_rgb(frame_bgr))

    def encode_rgb(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Encode an already-RGB ``(H,W,3)`` uint8 frame -> ``(196,768)`` fp16 patch tokens."""
        if not isinstance(frame_rgb, np.ndarray):
            raise TypeError(f"frame_rgb: expected np.ndarray, got {type(frame_rgb).__name__}")
        if frame_rgb.ndim != 3 or frame_rgb.shape[-1] != 3:
            raise ValueError(f"frame_rgb: expected (H,W,3), got shape {frame_rgb.shape}")

        hidden = self._forward(frame_rgb)  # (1, T, 768); T = CLS + registers + 196 patches
        # DINOv3 orders tokens [CLS, register..., patches]; take the TRAILING PATCH_TOKENS so the
        # wrapper is robust to however many CLS/register prefix tokens the checkpoint carries.
        patches = hidden[:, -PATCH_TOKENS:, :].squeeze(0)  # (196, 768)
        # Cached latents are ALWAYS fp16 (== LATENT_DTYPE) — halves disk + matches the z_next target —
        # regardless of the constructor's `dtype` (which only sets the model's COMPUTE precision).
        arr = patches.half().cpu().numpy()
        _validate_latent(arr)
        return arr


def _validate_latent(arr: np.ndarray) -> None:
    """Enforce the cache contract: exactly ``(PATCH_TOKENS, EMBED_DIM)`` fp16."""
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"latent: expected np.ndarray, got {type(arr).__name__}")
    if arr.shape != (PATCH_TOKENS, EMBED_DIM):
        raise ValueError(f"latent: expected shape ({PATCH_TOKENS}, {EMBED_DIM}), got {arr.shape}")
    if arr.dtype != np.dtype(LATENT_DTYPE):
        raise ValueError(f"latent: expected dtype {np.dtype(LATENT_DTYPE)}, got {arr.dtype}")


def _smoke(model_id: str, device: str) -> int:  # pragma: no cover - downloads real weights
    """Real-weight smoke: build the encoder, encode one random BGR frame, assert the contract.

    Run via ``make encode-smoke`` (downloads ~330 MB non-gated timm DINOv3 weights; no HF token).
    From CN, ``HF_ENDPOINT=https://hf-mirror.com`` speeds the download.
    """
    rng = np.random.default_rng(0)
    frame_bgr = rng.integers(0, 256, size=(INPUT_HW[0], INPUT_HW[1], 3), dtype=np.uint8)
    enc = DinoV3Encoder(model_id=model_id, device=device)
    latent = enc.encode_bgr(frame_bgr)
    print(f"[encode-smoke] model={model_id} device={device} -> latent {latent.shape} {latent.dtype}")
    assert latent.shape == (PATCH_TOKENS, EMBED_DIM), latent.shape
    assert latent.dtype == np.dtype(LATENT_DTYPE), latent.dtype
    print("[encode-smoke] OK — (196,768) fp16 frozen DINOv3 forward")
    return 0


def _main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI / USER-GATED
    parser = argparse.ArgumentParser(description="DINOv3 ViT-B/16 encoder (A5.10).")
    parser.add_argument("--smoke", action="store_true", help="real-weight smoke (downloads weights)")
    parser.add_argument("--model-id", default=MODEL_ID, help=f"HF model id (default {MODEL_ID})")
    parser.add_argument("--device", default="cuda", help="cuda | cpu")
    args = parser.parse_args(argv)
    if args.smoke:
        return _smoke(args.model_id, args.device)
    parser.error("nothing to do: pass --smoke (real-weight) or run the contract test")  # NoReturn


__all__ = ["DinoV3Encoder", "bgr_to_rgb", "INPUT_HW", "MODEL_ID"]


if __name__ == "__main__":  # pragma: no cover - USER-GATED
    raise SystemExit(_main())
