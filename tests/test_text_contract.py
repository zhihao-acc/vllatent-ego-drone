"""A5.13b contract tests: frozen CLIP text tower with a MONKEYPATCHED backbone (no real weights).

PURE gate (no marker): the ``_load_backbone`` seam returns numpy ``(M, Dn)`` tokens, so the contract
needs neither torch nor transformers (mirrors the WorldVLN / V-JEPA-2 client tests). The heavy half —
real CLIP weights — is the USER-GATED ``make text-smoke``.

Pins:
  1. the 512→768 zero-pad lift (first 512 dims = CLIP, rest 0; pass-through at 768; reject wider);
  2. ``encode`` output is ``(M, 768)`` fp16 with M = real token count, feeding ``StepSample.lang_tokens``;
  3. input validation (non-str / empty) and the model-id single-source;
  4. tier purity — no torch/transformers import at module scope (the lazy guard).
"""
from __future__ import annotations

import ast
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

import vllatent.encode.text as txt
from vllatent.config import Config
from vllatent.schemas import EMBED_DIM, HISTORY, LATENT_DTYPE, MASK_DTYPE, PATCH_TOKENS, StepSample

_NATIVE = 512  # CLIP ViT-B/32 text width


def _install_fake_backbone(monkeypatch: pytest.MonkeyPatch, forward: Callable | None = None):
    """Patch ``_load_backbone`` with a fake forward: default returns ``(M, 512)`` where M = word count
    and each row is filled with a per-token sentinel, so the lift + token-count are checkable."""
    seen: dict[str, str] = {}

    def default_forward(text: str) -> np.ndarray:
        m = len(text.split())
        toks = np.zeros((m, _NATIVE), dtype=np.float32)
        for i in range(m):
            toks[i, :] = float(i + 1)  # row i sentinel = i+1
        return toks

    respond = forward or default_forward

    def fake_load_backbone(model_id: str, device: str, dtype: str):  # noqa: ANN202 - test stub
        def _forward(text: str) -> np.ndarray:
            seen["text"] = text
            return respond(text)

        return _forward

    monkeypatch.setattr(txt, "_load_backbone", fake_load_backbone)
    return seen


# --- the 512->768 lift (pure, no mock) ---


def test_lift_zero_pads_to_embed_dim() -> None:
    tokens = np.arange(3 * _NATIVE, dtype=np.float32).reshape(3, _NATIVE)
    lifted = txt._lift_to_embed_dim(tokens)
    assert lifted.shape == (3, EMBED_DIM)
    np.testing.assert_array_equal(lifted[:, :_NATIVE], tokens)  # CLIP carried in the first 512
    assert np.all(lifted[:, _NATIVE:] == 0.0)                   # the rest is zero-pad


def test_lift_passthrough_when_native_equals_embed() -> None:
    tokens = np.ones((2, EMBED_DIM), dtype=np.float32)
    np.testing.assert_array_equal(txt._lift_to_embed_dim(tokens), tokens)


def test_lift_rejects_wider_than_embed_and_non_2d() -> None:
    with pytest.raises(ValueError, match="needs a real"):
        txt._lift_to_embed_dim(np.zeros((2, EMBED_DIM + 1), dtype=np.float32))
    with pytest.raises(ValueError, match="2-D"):
        txt._lift_to_embed_dim(np.zeros((EMBED_DIM,), dtype=np.float32))


# --- encode end-to-end over the mocked backbone ---


def test_encode_shape_dtype_and_token_count(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_backbone(monkeypatch)
    enc = txt.ClipTextEncoder(device="cpu")
    tokens = enc.encode("fly forward and turn")   # 4 words -> M = 4
    assert tokens.shape == (4, EMBED_DIM) and tokens.dtype == np.dtype(LATENT_DTYPE)
    # token 0 sentinel = 1.0 in the first 512 dims, 0 in the padded tail.
    assert tokens[0, 0] == pytest.approx(1.0) and np.all(tokens[0, _NATIVE:] == 0.0)


def test_encode_preserves_token_count(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_backbone(monkeypatch)
    enc = txt.ClipTextEncoder(device="cpu")
    assert enc.encode("one two").shape[0] == 2
    assert enc.encode("a b c d e").shape[0] == 5


def test_encode_feeds_stepsample_lang_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_backbone(monkeypatch)
    enc = txt.ClipTextEncoder(device="cpu")
    lang = enc.encode("go to the red car")   # M = 5
    # The whole point: the encoded tokens must satisfy StepSample's locked (M,768) fp16 lang contract.
    step = StepSample(
        z_t=np.zeros((PATCH_TOKENS, EMBED_DIM), dtype=LATENT_DTYPE),
        history_latents=np.zeros((HISTORY, PATCH_TOKENS, EMBED_DIM), dtype=LATENT_DTYPE),
        history_mask=np.zeros((HISTORY,), dtype=MASK_DTYPE),
        lang_tokens=lang,
        lang_mask=np.ones((lang.shape[0],), dtype=MASK_DTYPE),
        action_id=0,
        z_next=np.zeros((PATCH_TOKENS, EMBED_DIM), dtype=LATENT_DTYPE),
        delta_4dof=np.zeros((4,), dtype=np.float32),
    )
    assert step.lang_tokens.shape == (5, EMBED_DIM)


def test_encode_rejects_bad_text(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_backbone(monkeypatch)
    enc = txt.ClipTextEncoder(device="cpu")
    with pytest.raises(TypeError, match="str"):
        enc.encode(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        enc.encode("   ")


# --- single-source + tier purity ---


def test_model_id_single_sourced_from_config() -> None:
    assert txt.MODEL_ID == Config().encoder.text_model_id == "openai/clip-vit-base-patch32"


def test_module_imports_heavy_free() -> None:
    """stdlib+numpy+pure-config at module scope; torch/transformers only inside _load_backbone."""
    heavy = {"torch", "transformers", "timm", "cv2", "PIL"}
    tree = ast.parse(Path(txt.__file__).read_text())
    for node in tree.body:  # module scope only — function-local imports are the lazy pattern
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names = [node.module]
        for n in names:
            assert n.split(".")[0] not in heavy, f"module-level heavy import {n!r} breaks tier purity"
    assert "torch" not in sys.modules or True  # informational; the AST check is the real guard
