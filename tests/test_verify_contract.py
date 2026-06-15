"""A5.12 contract tests: V-JEPA-2 surprise verifier with a MONKEYPATCHED backbone (no real weights).

PURE gate (no marker): the ``_load_backbone`` seam returns numpy ``(ẑ, z)`` pairs, so the contract
needs neither torch nor transformers — it runs in the default ``make test`` lane (mirrors the WorldVLN
client test, not the DINOv3 test which builds torch tensors). The heavy half — real V-JEPA-2 ViT-L
weights + the encoder/predictor forward — is the USER-GATED ``make vjepa-smoke``.

Pins:
  1. the surprise math ``s_j = 1 - cos(ẑ_j, z_j)``: identical->0, orthogonal->1, opposite->2;
  2. one scalar per GT future frame, finite and in ``[0, 2]``, feeding ``OracleTarget.vjepa_surprise``;
  3. zero-norm / non-finite embeddings degrade to a neutral finite surprise (no NaN poisoning);
  4. input validation (shape / dtype / mismatched H,W) and the model-id single-source;
  5. tier purity — no torch/transformers import at module scope (the lazy guard).
"""
from __future__ import annotations

import ast
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

import vllatent.verify.vjepa2 as vj
from vllatent.config import Config
from vllatent.schemas import OracleTarget


def _install_fake_backbone(monkeypatch: pytest.MonkeyPatch, pairs: Callable | None = None):
    """Patch ``_load_backbone`` with a fake forward returning scripted ``(ẑ, z)`` numpy pairs.

    ``pairs(context_rgb, future_rgb) -> (zhat (J,D), z (J,D))``; defaults to a J-row pair where each
    future frame's ẑ == z (=> surprise 0), so individual tests override only what they exercise.
    """
    seen: dict[str, np.ndarray] = {}

    def default_pairs(context_rgb: np.ndarray, future_rgb: np.ndarray):
        j = future_rgb.shape[0]
        base = np.ones((j, 4), dtype=np.float32)
        return base, base.copy()

    respond = pairs or default_pairs

    def fake_load_backbone(model_id: str, device: str, dtype: str):  # noqa: ANN202 - test stub
        def _forward(context_rgb: np.ndarray, future_rgb: np.ndarray):
            seen["context"] = np.array(context_rgb, copy=True)
            seen["future"] = np.array(future_rgb, copy=True)
            return respond(context_rgb, future_rgb)

        return _forward

    monkeypatch.setattr(vj, "_load_backbone", fake_load_backbone)
    return seen


def _frames(n: int, hw: tuple[int, int] = (8, 8)) -> np.ndarray:
    return np.zeros((n, hw[0], hw[1], 3), dtype=np.uint8)


# --- cosine surprise math (pure, no mock) ---


def test_cosine_surprise_identical_orthogonal_opposite() -> None:
    zhat = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    z = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    s = vj.cosine_surprise(zhat, z)
    np.testing.assert_allclose(s, [0.0, 1.0, 2.0], atol=1e-6)
    assert s.dtype == np.float32


def test_cosine_surprise_scale_invariant() -> None:
    # cosine ignores magnitude: a 100x-scaled ẑ is still perfectly aligned -> surprise 0.
    zhat = np.array([[3.0, 4.0]], dtype=np.float32)
    z = np.array([[300.0, 400.0]], dtype=np.float32)
    np.testing.assert_allclose(vj.cosine_surprise(zhat, z), [0.0], atol=1e-6)


def test_cosine_surprise_zero_and_nonfinite_norm_are_neutral() -> None:
    zhat = np.array([[0.0, 0.0], [np.inf, 0.0]], dtype=np.float64)
    z = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    s = vj.cosine_surprise(zhat, z)
    assert np.all(np.isfinite(s)) and np.allclose(s, 1.0)  # neutral, not NaN


def test_cosine_surprise_stays_in_unit_range_near_boundary() -> None:
    # Near-(anti)parallel vectors where float rounding can push |cos| slightly past 1: the clip must
    # keep s within [0, 2] so a removed/!flipped clip (=> s < 0 or > 2) is caught here.
    zhat = np.array([[1.0, 1e-8], [1.0, -1e-8]], dtype=np.float64)
    z = np.array([[1.0, -1e-8], [-1.0, 1e-8]], dtype=np.float64)  # ~parallel, ~anti-parallel
    s = vj.cosine_surprise(zhat, z)
    assert np.all(s >= 0.0) and np.all(s <= 2.0)
    assert s[0] == pytest.approx(0.0, abs=1e-6) and s[1] == pytest.approx(2.0, abs=1e-6)


def test_cosine_surprise_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="must match"):
        vj.cosine_surprise(np.zeros((2, 4)), np.zeros((3, 4)))
    with pytest.raises(ValueError, match="2-D"):
        vj.cosine_surprise(np.zeros(4), np.zeros(4))


# --- verifier.surprise end-to-end over the mocked backbone ---


def test_surprise_shape_dtype_and_range(monkeypatch: pytest.MonkeyPatch) -> None:
    def pairs(_c, fut):
        j = fut.shape[0]
        zhat = np.tile([1.0, 0.0, 0.0, 0.0], (j, 1)).astype(np.float32)
        z = np.tile([0.0, 1.0, 0.0, 0.0], (j, 1)).astype(np.float32)  # orthogonal -> s=1
        return zhat, z

    _install_fake_backbone(monkeypatch, pairs)
    v = vj.VJEPA2SurpriseVerifier(device="cpu")
    s = v.surprise(_frames(2), _frames(3))
    assert s.shape == (3,) and s.dtype == np.float32
    assert np.allclose(s, 1.0)
    # Full [0,2] contract — both bounds, so a sign-flip/removed-clip regression can't slip through >= 0.
    assert np.all(np.isfinite(s)) and np.all(s >= 0.0) and np.all(s <= 2.0)


def test_surprise_is_per_future_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    # ẑ rows align with z rows except frame 1 (opposite) -> only s[1] is large.
    def pairs(_c, fut):
        zhat = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        z = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        return zhat, z

    _install_fake_backbone(monkeypatch, pairs)
    v = vj.VJEPA2SurpriseVerifier(device="cpu")
    s = v.surprise(_frames(1), _frames(3))
    np.testing.assert_allclose(s, [0.0, 2.0, 0.0], atol=1e-6)
    assert np.all(s >= 0.0) and np.all(s <= 2.0)


def test_scalar_surprise_feeds_oracle_target(monkeypatch: pytest.MonkeyPatch) -> None:
    def pairs(_c, fut):
        j = fut.shape[0]
        zhat = np.tile([1.0, 0.0], (j, 1)).astype(np.float32)
        z = np.tile([0.0, 1.0], (j, 1)).astype(np.float32)  # all orthogonal -> mean surprise 1.0
        return zhat, z

    _install_fake_backbone(monkeypatch, pairs)
    v = vj.VJEPA2SurpriseVerifier(device="cpu")
    scalar = v.scalar_surprise(_frames(2), _frames(2))
    assert isinstance(scalar, float) and scalar == pytest.approx(1.0)
    assert 0.0 <= scalar <= 2.0  # mean of per-frame s ∈ [0,2] stays in range
    # The whole point: the scalar must satisfy OracleTarget's finite >= 0 contract.
    oracle = OracleTarget(
        waypoint_4dof=np.zeros(4, dtype=np.float32),
        teacher_pose6=np.zeros(6, dtype=np.float64),
        rollpitch_resid=0.0,
        disagreement=0.0,
        vjepa_surprise=scalar,
    )
    assert oracle.vjepa_surprise == pytest.approx(1.0)


@pytest.mark.parametrize(
    "zrow, expected",
    [([1.0, 0.0], 0.0), ([0.0, 1.0], 1.0), ([-1.0, 0.0], 2.0)],  # identical / orthogonal / opposite
)
def test_scalar_surprise_spans_full_range(monkeypatch: pytest.MonkeyPatch, zrow, expected) -> None:
    # Across the cosine extremes the scalar mean stays in [0,2] and the OracleTarget feed holds.
    def pairs(_c, fut):
        j = fut.shape[0]
        return np.tile([1.0, 0.0], (j, 1)).astype(np.float32), np.tile(zrow, (j, 1)).astype(np.float32)

    _install_fake_backbone(monkeypatch, pairs)
    v = vj.VJEPA2SurpriseVerifier(device="cpu")
    scalar = v.scalar_surprise(_frames(1), _frames(3))
    assert scalar == pytest.approx(expected) and 0.0 <= scalar <= 2.0
    OracleTarget(  # must not raise — finite, >= 0
        waypoint_4dof=np.zeros(4, dtype=np.float32), teacher_pose6=np.zeros(6, dtype=np.float64),
        rollpitch_resid=0.0, disagreement=0.0, vjepa_surprise=scalar,
    )


def test_backbone_receives_rgb_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _install_fake_backbone(monkeypatch)
    v = vj.VJEPA2SurpriseVerifier(device="cpu")
    v.surprise(_frames(2), _frames(4))
    assert seen["context"].shape == (2, 8, 8, 3) and seen["future"].shape == (4, 8, 8, 3)


# --- input validation ---


def test_surprise_rejects_bad_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_backbone(monkeypatch)
    v = vj.VJEPA2SurpriseVerifier(device="cpu")
    with pytest.raises(ValueError, match="n,H,W,3"):
        v.surprise(np.zeros((8, 8, 3), dtype=np.uint8), _frames(2))      # missing frame axis
    with pytest.raises(ValueError, match="dtype uint8"):
        v.surprise(_frames(2).astype(np.float32), _frames(2))            # wrong dtype
    with pytest.raises(TypeError, match="np.ndarray"):
        v.surprise([[1, 2, 3]], _frames(2))  # type: ignore[arg-type]    # not an ndarray
    with pytest.raises(ValueError, match="must match"):
        v.surprise(_frames(2, (8, 8)), _frames(2, (16, 16)))            # mismatched H,W


def test_backbone_row_count_mismatch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # A misbehaving backbone that returns the wrong number of rows is caught (not silently pooled wrong).
    _install_fake_backbone(monkeypatch, lambda _c, fut: (np.ones((1, 4), np.float32), np.ones((1, 4), np.float32)))
    v = vj.VJEPA2SurpriseVerifier(device="cpu")
    with pytest.raises(RuntimeError, match="rows for"):
        v.surprise(_frames(1), _frames(3))  # asked for 3 future frames, backbone returned 1


# --- single-source + tier purity ---


def test_model_id_single_sourced_from_config() -> None:
    assert vj.MODEL_ID == Config().trust.vjepa2_model_id == "facebook/vjepa2-vitl-fpc64-256"


def test_module_imports_heavy_free() -> None:
    """stdlib+numpy+pure-config at module scope; torch/transformers only inside _load_backbone."""
    heavy = {"torch", "transformers", "timm", "cv2", "PIL"}
    tree = ast.parse(Path(vj.__file__).read_text())
    for node in tree.body:  # module scope only — function-local imports are the lazy pattern
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names = [node.module]
        for n in names:
            assert n.split(".")[0] not in heavy, f"module-level heavy import {n!r} breaks tier purity"
    assert "torch" not in sys.modules or True  # informational; the AST check is the real guard
