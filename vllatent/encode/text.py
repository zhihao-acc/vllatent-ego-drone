"""Frozen CLIP text-tower encoder (TORCH tier) — Phase-A step A5.13b.

Produces the per-episode cached ``lang_tokens`` the student cross-attends to: an instruction string ->
``(M, 768)`` fp16 per-token language embeddings (``M`` = real token count, BOS..EOS, no padding). This
is the **frozen text half** of the student's input encoding (the vision half is the DINOv3 wrapper in
``dinov3.py``); like DINOv3 it never trains (no EMA / no VICReg).

**Weights — NON-GATED.** ``openai/clip-vit-base-patch32`` (15M downloads, `gated:false`; probed
2026-06-14) — no HF token, no re-host fallback (contrast the DINOv3 gate lesson). Id single-sourced from
``Config.encoder.text_model_id``; the cache manifest (``build_manifest``) records it for provenance.

**512 → 768 lift.** CLIP ViT-B/32's text transformer is **512**-wide (``last_hidden_state`` = ``(M, 512)``),
but the cache contract (``StepSample.lang_tokens``) and the predictor residual stream are ``EMBED_DIM`` =
768. The *meaningful* 512→768 mapping is the student's **learned cross-attention K/V projection (Phase B)**,
so the frozen cache lift here is a deterministic **zero-pad** (first 512 dims = CLIP, rest 0) — reproducible,
information-preserving, no training, and trivially swappable. ``_lift_to_embed_dim`` owns it.

**torch / transformers imports are LAZY** (inside :func:`_load_backbone`) so a torch-free box (the pure CI
lane) imports this module without crashing. The pure tier never imports it. Tested two ways (A5.13b): a
monkeypatched-backbone **contract** test (``tests/test_text_contract.py``, PURE — the seam returns numpy,
so no torch/transformers needed, mirroring the WorldVLN/V-JEPA-2 client tests) and a real-weight smoke
``make text-smoke`` (USER-GATED; downloads CLIP weights; no token).

See plans/phase-a5-replan-postpivot.md (A5.13b — text tower for the lang_tokens cache).
"""
from __future__ import annotations

import argparse
from collections.abc import Callable

import numpy as np

from vllatent.config import EncoderConfig
from vllatent.schemas import EMBED_DIM, LATENT_DTYPE

# Single source of truth = vllatent.config (pure tier). build_manifest records the same id for provenance.
MODEL_ID = EncoderConfig().text_model_id

# A backbone forward maps an instruction string -> per-token frozen embeddings (M, Dn) fp32 (Dn = the
# text tower's native width, 512 for CLIP ViT-B/32). Injected via _load_backbone so the contract test can
# monkeypatch it without real weights (numpy out -> the test needs neither torch nor transformers).
TextForward = Callable[[str], "np.ndarray"]


def _lift_to_embed_dim(tokens: np.ndarray) -> np.ndarray:
    """Lift per-token features ``(M, Dn)`` -> ``(M, EMBED_DIM)`` by zero-pad (or exact pass-through).

    Pure numpy. ``Dn <= EMBED_DIM`` (CLIP-B = 512 ≤ 768): the first ``Dn`` dims carry CLIP, the rest are 0.
    The real 512→768 map is the student's learned cross-attention (Phase B); this only satisfies the cache
    contract reproducibly. ``Dn > EMBED_DIM`` is rejected (a wider tower needs a real projection decision).
    """
    if tokens.ndim != 2:
        raise ValueError(f"tokens: expected 2-D (M,Dn), got shape {tokens.shape}")
    m, native = tokens.shape
    if native > EMBED_DIM:
        raise ValueError(
            f"tokens native width {native} > EMBED_DIM {EMBED_DIM}: a wider text tower needs a real "
            f"projection, not a zero-pad lift"
        )
    if native == EMBED_DIM:
        return np.ascontiguousarray(tokens)
    lifted = np.zeros((m, EMBED_DIM), dtype=tokens.dtype)
    lifted[:, :native] = tokens
    return lifted


def _load_backbone(model_id: str, device: str, dtype: str) -> TextForward:
    """Lazy-load the frozen CLIP text tower; return a string -> ``(M, Dn)`` fp32 callable.

    ALL heavy imports (torch / transformers) live in here so the module imports on a torch-free box.
    Monkeypatched in the contract test (no real weights in CI). Tokenizes WITHOUT padding so ``M`` is the
    real token count (BOS..EOS); returns the encoder's per-token ``last_hidden_state`` as fp32 numpy.
    """
    import torch
    from transformers import AutoTokenizer, CLIPTextModel

    try:
        model = CLIPTextModel.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as exc:  # network / offline — make it actionable
        raise RuntimeError(
            f"Failed to load CLIP text tower {model_id!r}. These are NON-GATED — no HF_TOKEN needed. "
            f"From CN, set HF_ENDPOINT=https://hf-mirror.com if the direct download is slow. "
            f"Original error: {exc}"
        ) from exc
    model = model.to(device).eval()
    if dtype == "float16":
        model = model.half()
    for param in model.parameters():  # frozen — no grads, never trains
        param.requires_grad_(False)

    def _forward(text: str) -> np.ndarray:
        # No padding -> M = the real token count (BOS..EOS); the cache stores only real tokens (the
        # loader sets lang_mask all-True), so padding would poison the per-episode token count.
        enc = tokenizer(text, return_tensors="pt", padding=False, truncation=True)
        input_ids = enc["input_ids"].to(device)
        with torch.no_grad():
            hidden = model(input_ids=input_ids).last_hidden_state  # (1, M, Dn)
        return hidden[0].float().cpu().numpy()

    return _forward


class ClipTextEncoder:
    """Frozen CLIP ViT-B/32 text tower. Lazy-loads torch on construction; emits ``(M,768)`` fp16 tokens.

    Tests monkeypatch :func:`_load_backbone` to inject a fake forward, so the lang-token contract is
    exercised without any download. ``device`` defaults to ``"cuda"``; pass ``"cpu"`` for a weightless
    contract test.
    """

    def __init__(self, model_id: str = MODEL_ID, device: str = "cuda", dtype: str = "float16") -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self._forward: TextForward = _load_backbone(model_id, device, dtype)

    def encode(self, text: str) -> np.ndarray:
        """Encode an instruction string -> ``(M, 768)`` fp16 per-token tokens (M real tokens)."""
        if not isinstance(text, str):
            raise TypeError(f"text: expected str, got {type(text).__name__}")
        if not text.strip():
            raise ValueError("text: expected a non-empty instruction")
        tokens = self._forward(text)              # (M, Dn) fp32
        lifted = _lift_to_embed_dim(tokens)       # (M, 768)
        arr = lifted.astype(LATENT_DTYPE)         # cache dtype == fp16, matches StepSample.lang_tokens
        _validate_lang(arr)
        return arr


def _validate_lang(arr: np.ndarray) -> None:
    """Enforce the lang-token cache contract: ``(M, EMBED_DIM)`` fp16, ``M >= 1``."""
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"lang_tokens: expected np.ndarray, got {type(arr).__name__}")
    if arr.ndim != 2 or arr.shape[1] != EMBED_DIM:
        raise ValueError(f"lang_tokens: expected (M,{EMBED_DIM}), got shape {arr.shape}")
    if arr.shape[0] < 1:
        raise ValueError(f"lang_tokens: expected M>=1 tokens, got {arr.shape[0]}")
    if arr.dtype != np.dtype(LATENT_DTYPE):
        raise ValueError(f"lang_tokens: expected dtype {np.dtype(LATENT_DTYPE)}, got {arr.dtype}")


def _smoke(model_id: str, device: str, text: str) -> int:  # pragma: no cover - downloads real weights
    """Real-weight smoke: build the encoder, encode one instruction, assert the (M,768) fp16 contract.

    Run via ``make text-smoke`` (downloads CLIP weights; no HF token). From CN,
    ``HF_ENDPOINT=https://hf-mirror.com`` speeds the download. USER-GATED: the user runs it.
    """
    enc = ClipTextEncoder(model_id=model_id, device=device)
    tokens = enc.encode(text)
    print(f"[text-smoke] model={model_id} device={device} text={text!r} -> lang_tokens {tokens.shape} {tokens.dtype}")
    assert tokens.ndim == 2 and tokens.shape[1] == EMBED_DIM, tokens.shape
    assert tokens.dtype == np.dtype(LATENT_DTYPE), tokens.dtype
    print("[text-smoke] OK — (M,768) fp16 frozen CLIP text tokens")
    return 0


def _main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI / USER-GATED
    parser = argparse.ArgumentParser(description="CLIP text tower encoder (A5.13b).")
    parser.add_argument("--smoke", action="store_true", help="real-weight smoke (downloads CLIP weights)")
    parser.add_argument("--model-id", default=MODEL_ID, help=f"HF model id (default {MODEL_ID})")
    parser.add_argument("--device", default="cuda", help="cuda | cpu")
    parser.add_argument("--text", default="fly forward and turn right at the building", help="instruction")
    args = parser.parse_args(argv)
    if args.smoke:
        return _smoke(args.model_id, args.device, args.text)
    parser.error("nothing to do: pass --smoke (real-weight) or run the contract test")  # NoReturn


__all__ = ["ClipTextEncoder", "MODEL_ID"]


if __name__ == "__main__":  # pragma: no cover - USER-GATED
    raise SystemExit(_main())
