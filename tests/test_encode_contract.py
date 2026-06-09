"""A5.10 contract tests: frozen DINOv3 wrapper with a MONKEYPATCHED backbone (no real weights).

Marked ``@pytest.mark.torch`` (whole module via ``pytestmark``) — runs under ``make test-torch`` /
wherever torch is installed; auto-skipped in the pure CI gate (see ``conftest``). It pins the two
contract guarantees WITHOUT downloading the gated multi-GB DINOv3 weights:

  1. the **BGR->RGB** boundary (foot-gun #2) is applied before the backbone sees the frame, and
  2. the output is exactly ``(196,768)`` fp16, with the ``[CLS]+register`` prefix dropped (trailing-196).

All heavy imports (``torch``) stay function-local so pytest can still *collect* this module on a
torch-free box (the marker then skips execution); only ``numpy``/``pytest``/the torch-free wrapper
module are imported at module scope.
"""
from __future__ import annotations

import numpy as np
import pytest

from vllatent.encode import dinov3
from vllatent.schemas import EMBED_DIM, LATENT_DTYPE, PATCH_TOKENS

pytestmark = pytest.mark.torch

# DINOv3 ViT-B/16 at 224x224: last_hidden_state = [CLS] + 4 register tokens + 196 patch tokens = 201
# (research-confirmed). The wrapper takes the TRAILING 196, so the prefix count is just a sentinel here.
N_PREFIX = 5
PREFIX_SENTINEL = 9.0  # CLS + registers — MUST be dropped
PATCH_SENTINEL = 1.0   # the 196 patch tokens — MUST be kept


def _install_fake_backbone(monkeypatch: pytest.MonkeyPatch) -> dict[str, np.ndarray]:
    """Patch ``_load_backbone`` with a fake forward: records the RGB it receives and returns a
    ``(1, N_PREFIX+196, 768)`` tensor whose prefix vs patch tokens are distinguishable."""
    import torch

    seen: dict[str, np.ndarray] = {}

    def fake_load_backbone(model_id: str, device: str, dtype: str):  # noqa: ANN202 - test stub
        def _forward(frame_rgb: np.ndarray) -> torch.Tensor:
            seen["rgb"] = np.array(frame_rgb, copy=True)
            total = N_PREFIX + PATCH_TOKENS
            hidden = torch.empty(1, total, EMBED_DIM, dtype=torch.float32)
            hidden[:, :N_PREFIX, :] = PREFIX_SENTINEL
            hidden[:, N_PREFIX:, :] = PATCH_SENTINEL
            return hidden

        return _forward

    monkeypatch.setattr(dinov3, "_load_backbone", fake_load_backbone)
    return seen


def _bgr_frame() -> np.ndarray:
    """A BGR frame with distinct per-channel values so the RGB flip is observable."""
    h, w = dinov3.INPUT_HW
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    bgr[..., 0] = 10  # B
    bgr[..., 1] = 20  # G
    bgr[..., 2] = 30  # R
    return bgr


def test_encode_bgr_flips_to_rgb_before_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _install_fake_backbone(monkeypatch)
    enc = dinov3.DinoV3Encoder(device="cpu")
    enc.encode_bgr(_bgr_frame())
    rgb = seen["rgb"]
    # The backbone must have received RGB: channel 0 = R(30), channel 1 = G(20), channel 2 = B(10).
    assert rgb[0, 0, 0] == 30 and rgb[0, 0, 1] == 20 and rgb[0, 0, 2] == 10


def test_encode_output_shape_and_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_backbone(monkeypatch)
    enc = dinov3.DinoV3Encoder(device="cpu")
    out = enc.encode_bgr(_bgr_frame())
    assert out.shape == (PATCH_TOKENS, EMBED_DIM)      # (196, 768)
    assert out.dtype == np.dtype(LATENT_DTYPE)         # fp16


def test_trailing_196_drops_cls_and_registers(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_backbone(monkeypatch)
    enc = dinov3.DinoV3Encoder(device="cpu")
    out = enc.encode_rgb(np.zeros((*dinov3.INPUT_HW, 3), dtype=np.uint8)).astype(np.float32)
    # Only the patch sentinel survives — the CLS + register prefix is dropped.
    assert np.allclose(out, PATCH_SENTINEL)
    assert not np.any(np.isclose(out, PREFIX_SENTINEL))


def test_encode_rgb_rejects_bad_input(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_backbone(monkeypatch)
    enc = dinov3.DinoV3Encoder(device="cpu")
    with pytest.raises(ValueError):
        enc.encode_rgb(np.zeros((224, 224), dtype=np.uint8))      # missing channel axis
    with pytest.raises(ValueError):
        enc.encode_rgb(np.zeros((224, 224, 4), dtype=np.uint8))   # wrong channel count
    with pytest.raises(TypeError):
        enc.encode_rgb([[1, 2, 3]])  # type: ignore[arg-type]     # not an ndarray
