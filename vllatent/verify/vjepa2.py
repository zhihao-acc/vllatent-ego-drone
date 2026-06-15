"""Frozen V-JEPA-2 surprise verifier (TORCH tier) — Phase-A step A5.12.

The **independent second trust gate** (the first is the WorldVLN K-rollout disagreement, A5.11).
V-JEPA-2 is Meta's video joint-embedding predictive model: given observed *context* frames its
predictor forecasts the latent of *target* frames, and the encoder also produces the actual latent
of those frames. The **surprise**

    s_j = 1 - cos(ẑ_j, z_j)        per GT future frame j   (∈ [0, 2], 0 = perfectly predicted)

between the predictor forecast ``ẑ_j`` and the encoder's actual latent ``z_j`` is how unexpected
the world's evolution was — high surprise = low trust. Both vectors live in the SAME encoder space
(the predictor projects back to ``hidden_size``), so the cosine compares like with like. This scalar
feeds :class:`vllatent.schemas.OracleTarget`'s ``vjepa_surprise`` field (``>= 0``, finite).

**Weights — NON-GATED (unlike DINOv3).** ``facebook/vjepa2-vitl-fpc64-256`` (ViT-L, ~1.30 GB
safetensors) is fully public: ``gated: false``, MIT license (probed on hf-mirror 2026-06-11). No HF
token and **no re-host fallback** are needed (contrast the DINOv3 lesson, where Meta's gated repo
rejected access and we fell back to timm's re-host). The model id is single-sourced from
``Config.trust.vjepa2_model_id``.

**torch / transformers imports are LAZY** — every heavy import lives inside :func:`_load_backbone`,
so a torch-free box (the pure CI lane) imports this module without crashing. The pure tier
(``vllatent.{schemas,actions,frames,config,manifest,audit}``) NEVER imports this module.

**Color convention.** Frames are **RGB** ``uint8`` ``(n,H,W,3)`` — the render harness (A5.13) owns the
AirSim BGRA→BGR→RGB flip, so this verifier (like ``encode_rgb``) must NOT flip again. The V-JEPA-2
video processor resizes/center-crops to the model's ``crop_size`` (256²) and normalizes.

**Frame→token layout (real loader).** V-JEPA-2 patchifies with a 3D conv of temporal stride
``tubelet_size`` (=2), so two consecutive frames fuse into one temporal token slot. To get a clean
*per-future-frame* surprise we duplicate each logical frame ``tubelet_size`` times before the model,
so frame ``f`` maps to exactly one temporal slot = one block of ``grid²`` (=256) patch tokens. This
is V-JEPA-2's sanctioned degenerate mode (the model itself repeats frames when ``num_frames <
tubelet_size``); it trades 2× tokens for per-frame granularity. The recipe was validated end-to-end
on a shrunk random-weight model before this was written.

Tested two ways (A5.12): a monkeypatched-backbone **contract** test (``tests/test_verify_contract.py``,
PURE — the seam returns numpy so no torch/transformers is needed, mirroring the teacher client) that
pins the cosine-surprise math + the ``OracleTarget`` feed without real weights, and a real-weight
smoke ``make vjepa-smoke`` (USER-GATED; downloads ~1.30 GB non-gated weights; no token).

See plans/phase-a5-replan-postpivot.md A5.12.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable

import numpy as np

from vllatent.config import TrustConfig

# Single source of truth = vllatent.config (pure tier). The cache manifest (`build_manifest`) records
# the same id from Config.trust.vjepa2_model_id, so the verifier and the cache provenance never drift.
MODEL_ID = TrustConfig().vjepa2_model_id

# The cosine-surprise lower clip absorbs float epsilon only (cos can drift to 1+1e-7 -> s = -1e-7).
SURPRISE_MIN, SURPRISE_MAX = 0.0, 2.0

# A backbone forward maps (context_rgb, future_rgb) uint8 (n,H,W,3) frames -> a pair of per-future-frame
# pooled embeddings (ẑ, z), each (J, D) fp32: ẑ = predictor forecast, z = encoder's actual latent.
# Injected via _load_backbone so the contract test can monkeypatch it without real weights (numpy in,
# numpy out -> the contract test needs neither torch nor transformers, like the WorldVLN client test).
PairForward = Callable[[np.ndarray, np.ndarray], "tuple[np.ndarray, np.ndarray]"]


def _validate_frames(name: str, frames: object) -> None:
    """Validate an RGB frame stack: ``(n,H,W,3)`` uint8, ``n >= 1``."""
    if not isinstance(frames, np.ndarray):
        raise TypeError(f"{name}: expected np.ndarray, got {type(frames).__name__}")
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"{name}: expected (n,H,W,3), got shape {frames.shape}")
    if frames.shape[0] < 1:
        raise ValueError(f"{name}: expected n>=1 frames, got {frames.shape[0]}")
    if frames.dtype != np.uint8:
        raise ValueError(f"{name}: expected dtype uint8, got {frames.dtype}")


def cosine_surprise(zhat: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Per-row surprise ``s_j = 1 - cos(ẑ_j, z_j)`` for paired ``(J,D)`` embeddings -> ``(J,)`` fp32.

    Pure numpy (no torch) so it is directly testable and the contract test stays heavy-free. Result is
    clipped to ``[0, 2]`` (cosine ∈ [-1, 1]); a zero-norm or non-finite row maps to ``cos = 0`` (=> s = 1,
    a neutral value) so the output is always finite and ``>= 0`` — exactly what ``OracleTarget`` requires.
    """
    zhat = np.asarray(zhat, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if zhat.ndim != 2 or z.ndim != 2:
        raise ValueError(f"cosine_surprise: expected 2-D (J,D) arrays, got {zhat.shape} and {z.shape}")
    if zhat.shape != z.shape:
        raise ValueError(f"cosine_surprise: ẑ {zhat.shape} and z {z.shape} must match")

    dot = np.einsum("jd,jd->j", zhat, z)
    norm_h = np.sqrt(np.einsum("jd,jd->j", zhat, zhat))
    norm_z = np.sqrt(np.einsum("jd,jd->j", z, z))
    denom = norm_h * norm_z
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = np.where(denom > 0.0, dot / denom, 0.0)
    cos = np.where(np.isfinite(cos), cos, 0.0)  # guard NaN/inf -> neutral surprise (s=1)
    surprise = np.clip(1.0 - cos, SURPRISE_MIN, SURPRISE_MAX)
    return surprise.astype(np.float32)


def _load_backbone(model_id: str, device: str, dtype: str) -> PairForward:
    """Lazy-load the frozen V-JEPA-2 (ViT-L) encoder+predictor; return its (ẑ, z) pair callable.

    ALL heavy imports (torch / transformers) live in here so the module imports on a torch-free box.
    Monkeypatched in the contract test (so no real weights download in CI). The returned closure maps
    ``(context_rgb (C,H,W,3), future_rgb (J,H,W,3))`` uint8 RGB -> ``(ẑ (J,D), z (J,D))`` fp32, where
    each row is mean-pooled over the ``grid²`` patch tokens of one future frame.
    """
    import torch
    from transformers import AutoVideoProcessor, VJEPA2Model

    try:
        model = VJEPA2Model.from_pretrained(model_id)
        processor = AutoVideoProcessor.from_pretrained(model_id)
    except Exception as exc:  # network / offline — make it actionable
        raise RuntimeError(
            f"Failed to load V-JEPA-2 weights {model_id!r}. These are NON-GATED (gated:false, MIT) — no "
            f"HF_TOKEN needed. From CN, set HF_ENDPOINT=https://hf-mirror.com if the direct download is "
            f"slow. Original error: {exc}"
        ) from exc
    model = model.to(device).eval()
    if dtype == "float16":
        model = model.half()
    for param in model.parameters():  # frozen — no grads, never trains
        param.requires_grad_(False)

    cfg = model.config
    patch = cfg.patch_size if isinstance(cfg.patch_size, int) else cfg.patch_size[0]
    grid = int(cfg.crop_size) // int(patch)
    tokens_per_frame = grid * grid                 # patch tokens in one temporal slot (=one frame here)
    tubelet = int(cfg.tubelet_size)                # temporal conv stride: frames fused per slot
    compute_dtype = torch.float16 if dtype == "float16" else torch.float32

    def _forward(context_rgb: np.ndarray, future_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_ctx, n_fut = int(context_rgb.shape[0]), int(future_rgb.shape[0])
        frames = np.concatenate([context_rgb, future_rgb], axis=0)  # (C+J, H, W, 3) RGB
        # Processor: resize -> center-crop to crop_size² -> rescale -> normalize. Key = pixel_values_videos.
        pixel = processor(list(frames), return_tensors="pt")["pixel_values_videos"]  # (1, C+J, 3, S, S)
        # Duplicate each frame tubelet_size times so the temporal conv groups identical frames -> one
        # token slot per logical frame (clean per-frame surprise; see module docstring).
        pixel = pixel.repeat_interleave(tubelet, dim=1).to(device=device, dtype=compute_dtype)

        n_ctx_tok = n_ctx * tokens_per_frame
        n_tok = (n_ctx + n_fut) * tokens_per_frame
        # Mask = list of LongTensor index tensors into the patch dim (V-JEPA-2 convention). Context =
        # the first C frames' tokens; target = the J future frames' tokens (the prediction targets).
        ctx_idx = torch.arange(0, n_ctx_tok, device=device).unsqueeze(0)
        tgt_idx = torch.arange(n_ctx_tok, n_tok, device=device).unsqueeze(0)
        with torch.no_grad():
            out = model(pixel, context_mask=[ctx_idx], target_mask=[tgt_idx])
        zhat = out.predictor_output.last_hidden_state[0]    # (J*tpf, D) — predictor forecast at targets
        z = out.predictor_output.target_hidden_state[0]     # (J*tpf, D) — encoder's actual target latent
        dim = zhat.shape[-1]
        zhat_j = zhat.reshape(n_fut, tokens_per_frame, dim).mean(dim=1)  # (J, D) mean-pool per frame
        z_j = z.reshape(n_fut, tokens_per_frame, dim).mean(dim=1)        # (J, D)
        return zhat_j.float().cpu().numpy(), z_j.float().cpu().numpy()

    return _forward


class VJEPA2SurpriseVerifier:
    """Frozen V-JEPA-2 ViT-L surprise verifier. Lazy-loads torch on construction; emits per-frame surprise.

    The constructor does the heavy load (real weights). Tests monkeypatch :func:`_load_backbone` to
    inject a fake (ẑ, z) callable, so the surprise contract is exercised without any download.
    ``device`` defaults to ``"cuda"`` (the dev box / H20); pass ``"cpu"`` for a weightless contract test.
    """

    def __init__(self, model_id: str = MODEL_ID, device: str = "cuda", dtype: str = "float16") -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self._forward: PairForward = _load_backbone(model_id, device, dtype)

    def surprise(self, context_rgb: np.ndarray, future_rgb: np.ndarray) -> np.ndarray:
        """Per-future-frame surprise ``(J,)`` fp32 from RGB context + GT future frames.

        ``context_rgb`` ``(C,H,W,3)`` uint8 RGB (observed) and ``future_rgb`` ``(J,H,W,3)`` uint8 RGB
        (ground-truth future). Returns ``s`` ``(J,)`` ∈ [0,2], one scalar per future frame.
        """
        _validate_frames("context_rgb", context_rgb)
        _validate_frames("future_rgb", future_rgb)
        if context_rgb.shape[1:3] != future_rgb.shape[1:3]:
            raise ValueError(
                f"context_rgb H,W {context_rgb.shape[1:3]} must match future_rgb {future_rgb.shape[1:3]}"
            )
        zhat, z = self._forward(context_rgb, future_rgb)
        n_fut = int(future_rgb.shape[0])
        if zhat.shape[0] != n_fut or z.shape[0] != n_fut:
            raise RuntimeError(
                f"backbone returned {zhat.shape[0]}/{z.shape[0]} rows for {n_fut} future frames"
            )
        surprise = cosine_surprise(zhat, z)
        if surprise.shape != (n_fut,) or not np.all(np.isfinite(surprise)):
            raise RuntimeError(f"surprise must be finite {(n_fut,)}, got {surprise.shape} {surprise}")
        return surprise

    def scalar_surprise(self, context_rgb: np.ndarray, future_rgb: np.ndarray) -> float:
        """Mean per-frame surprise -> one ``float >= 0`` for ``OracleTarget.vjepa_surprise``.

        A5.14 may choose a different reduction over the future horizon (mean here; the per-frame array
        from :meth:`surprise` is the granular signal). Mean keeps it horizon-length-invariant.
        """
        return float(np.mean(self.surprise(context_rgb, future_rgb)))


def _smoke(model_id: str, device: str, n_context: int, n_future: int) -> int:  # pragma: no cover - downloads weights
    """Real-weight smoke: build the verifier, run one surprise on random RGB frames, assert the contract.

    Run via ``make vjepa-smoke`` (downloads ~1.30 GB non-gated V-JEPA-2 ViT-L weights; no HF token).
    From CN, ``HF_ENDPOINT=https://hf-mirror.com`` speeds the download. USER-GATED: the user runs it.
    """
    rng = np.random.default_rng(0)
    context = rng.integers(0, 256, size=(n_context, 224, 224, 3), dtype=np.uint8)
    future = rng.integers(0, 256, size=(n_future, 224, 224, 3), dtype=np.uint8)
    verifier = VJEPA2SurpriseVerifier(model_id=model_id, device=device)
    surprise = verifier.surprise(context, future)
    scalar = verifier.scalar_surprise(context, future)
    print(f"[vjepa-smoke] model={model_id} device={device} -> surprise {surprise} (mean {scalar:.4f})")
    assert surprise.shape == (n_future,), surprise.shape
    assert np.all(np.isfinite(surprise)) and np.all(surprise >= 0.0), surprise
    print("[vjepa-smoke] OK — per-future-frame V-JEPA-2 surprise, finite >= 0")
    return 0


def _main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI / USER-GATED
    parser = argparse.ArgumentParser(description="V-JEPA-2 surprise verifier (A5.12).")
    parser.add_argument("--smoke", action="store_true", help="real-weight smoke (downloads ~1.30 GB weights)")
    parser.add_argument("--model-id", default=MODEL_ID, help=f"HF model id (default {MODEL_ID})")
    parser.add_argument("--device", default="cuda", help="cuda | cpu")
    parser.add_argument("--context-frames", type=int, default=2, help="number of context frames (smoke)")
    parser.add_argument("--future-frames", type=int, default=2, help="number of GT future frames (smoke)")
    args = parser.parse_args(argv)
    if args.smoke:
        return _smoke(args.model_id, args.device, args.context_frames, args.future_frames)
    parser.error("nothing to do: pass --smoke (real-weight) or run the contract test")  # NoReturn


__all__ = ["VJEPA2SurpriseVerifier", "cosine_surprise", "MODEL_ID"]


if __name__ == "__main__":  # pragma: no cover - USER-GATED
    raise SystemExit(_main())
