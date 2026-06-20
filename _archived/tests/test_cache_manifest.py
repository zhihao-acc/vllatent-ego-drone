"""A5.14 tests: cache orchestrator builds well-formed .npz + manifest (MOCKED seams).

All five heavy seams are mocked (render, DINOv3, CLIP-text, WorldVLN teacher, V-JEPA-2),
so this runs in the PURE gate (``make test``). The test writes a tiny 2-episode cache to
``tmp_path``, then round-trips through ``CachedLatentDataset`` (A5.15) and asserts every
``.npz`` key/shape/dtype EXACTLY matches the loader read-contract, and ``validate_manifest``
returns clean with teacher/render provenance populated.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from vllatent.cache import (
    build_cache,
    center_crop_to_square,
    resize_square,
)
from vllatent.config import Config
from vllatent.data.loader import CachedLatentDataset
from vllatent.manifest import validate_manifest
from vllatent.schemas import (
    DELTA_DTYPE,
    DOF,
    EMBED_DIM,
    LATENT_DTYPE,
    N_ACTIONS,
    PATCH_TOKENS,
    TEACHER_DOF,
)

try:
    import cv2 as _cv2  # noqa: F401

    _HAS_CV2 = True
except ModuleNotFoundError:
    _HAS_CV2 = False

# --- Tiny episode fixtures (2 episodes, N=4 and N=3 poses) -----------------------

_INSTRUCTION = "fly forward and turn right at the building"


def _make_episode(episode_id: str, n_poses: int, scene_id: int) -> dict[str, Any]:
    """Minimal AerialVLN-shaped episode dict that ``parse_episode`` accepts."""
    ref = []
    for i in range(n_poses):
        ref.append([float(i), 0.0, -1.0, 0.0, 0.0, 0.1 * i])
    actions = [((i % 7) + 1) for i in range(n_poses - 1)] + [0]  # terminal STOP
    return {
        "episode_id": episode_id,
        "trajectory_id": f"traj_{episode_id}",
        "scene_id": scene_id,
        "instruction": {"instruction_text": _INSTRUCTION, "instruction_id": "1"},
        "start_position": [0.0, 0.0, -1.0],
        "start_rotation": [1.0, 0.0, 0.0, 0.0],  # wxyz
        "goals": [{"position": [10.0, 0.0, -1.0]}],
        "reference_path": ref,
        "actions": actions,
        "scanName": str(scene_id),
    }


# --- Mock factories for the 5 seams -------------------------------------------------

def _mock_renderer():
    """Mock RenderHarness: returns a random (480, 640, 3) RGB frame per row."""
    rng = np.random.default_rng(42)
    r = MagicMock()
    r.render_reference_row = lambda row: rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    return r


def _mock_vision_encoder():
    """Mock DinoV3Encoder: encode_rgb → (196, 768) fp16."""
    rng = np.random.default_rng(7)
    enc = MagicMock()
    enc.encode_rgb = lambda f: rng.standard_normal((PATCH_TOKENS, EMBED_DIM)).astype(LATENT_DTYPE)
    return enc


def _mock_text_encoder():
    """Mock ClipTextEncoder: encode → (M, 768) fp16 (M=5)."""
    rng = np.random.default_rng(13)
    enc = MagicMock()
    enc.encode = lambda txt: rng.standard_normal((5, EMBED_DIM)).astype(LATENT_DTYPE)
    return enc


def _mock_teacher_client(t_segment: int = 16):
    """Mock WorldVLNTeacherClient: k_rollout_segment → (K,T,6) seam."""
    rng = np.random.default_rng(99)
    client = MagicMock()

    def _k_rollout(frames, instruction, *, config=None, **kw):
        k = (config or Config()).trust.k_rollouts
        rollouts = rng.standard_normal((k, t_segment, TEACHER_DOF)).astype(np.float32) * 0.01
        return rollouts, [{}] * k

    client.k_rollout_segment = _k_rollout
    return client


def _mock_verifier():
    """Mock VJEPA2SurpriseVerifier: scalar_surprise → float ∈ [0, 2]."""
    rng = np.random.default_rng(17)
    v = MagicMock()
    v.scalar_surprise = lambda ctx, fut: float(rng.uniform(0.0, 0.5))
    return v


# --- Tests --------------------------------------------------------------------------

def test_center_crop_to_square_480x640() -> None:
    """480x640 → 480x480 center-crop (native res, no resize)."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = center_crop_to_square(frame)
    assert out.shape == (480, 480, 3) and out.dtype == np.uint8


def test_center_crop_to_square_already_square() -> None:
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    out = center_crop_to_square(frame)
    assert out.shape == (224, 224, 3)


def test_center_crop_to_square_tall() -> None:
    frame = np.zeros((640, 480, 3), dtype=np.uint8)
    out = center_crop_to_square(frame)
    assert out.shape == (480, 480, 3)


def test_center_crop_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="expected"):
        center_crop_to_square(np.zeros((224, 224), dtype=np.uint8))


@pytest.mark.skipif(not _HAS_CV2, reason="cv2 not installed (pure CI)")
def test_resize_square_480_to_224() -> None:
    """480x480 native-square → 224x224 for DINOv3."""
    frame = np.zeros((480, 480, 3), dtype=np.uint8)
    out = resize_square(frame, 224)
    assert out.shape == (224, 224, 3) and out.dtype == np.uint8


def test_resize_square_noop() -> None:
    """Already at target size → no resize, same array."""
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    out = resize_square(frame, 224)
    assert out.shape == (224, 224, 3)


def _numpy_resize_square(frame: np.ndarray, target_hw: int) -> np.ndarray:
    """Pure-numpy nearest-neighbor resize (no cv2) for CI."""
    if frame.shape[0] == target_hw and frame.shape[1] == target_hw:
        return frame
    idx = np.linspace(0, frame.shape[0] - 1, target_hw).astype(int)
    return frame[np.ix_(idx, idx)]


@pytest.fixture()
def tiny_cache(tmp_path, monkeypatch):
    """Build a 2-episode mocked cache and return the cache dir path."""
    if not _HAS_CV2:
        monkeypatch.setattr("vllatent.cache.resize_square", _numpy_resize_square)
    episodes = [
        _make_episode("ep0", n_poses=4, scene_id=1),
        _make_episode("ep1", n_poses=3, scene_id=2),
    ]
    manifest = build_cache(
        episodes,
        tmp_path / "cache",
        renderer=_mock_renderer(),
        vision_encoder=_mock_vision_encoder(),
        text_encoder=_mock_text_encoder(),
        teacher_client=_mock_teacher_client(t_segment=16),
        verifier=_mock_verifier(),
        config=Config(),
        split="train",
    )
    return str(tmp_path / "cache"), manifest


def test_manifest_valid(tiny_cache) -> None:
    cache_dir, manifest = tiny_cache
    errors = validate_manifest(manifest)
    assert errors == [], errors


def test_manifest_teacher_provenance_populated(tiny_cache) -> None:
    _, manifest = tiny_cache
    teacher = manifest["teacher"]
    assert teacher["worldvln_model_id"] != ""
    assert teacher["worldvln_revision"] != ""
    assert teacher["render_config_hash"] != ""
    assert teacher["disagreement_source"] == "vjepa_only"
    assert teacher["vjepa2_model_id"] != ""


def test_manifest_entries_count(tiny_cache) -> None:
    _, manifest = tiny_cache
    assert len(manifest["entries"]) == 2
    for entry in manifest["entries"]:
        assert "episode_id" in entry
        assert "n_frames" in entry
        assert "latent_path" in entry


def test_npz_keys_and_shapes_ep0(tiny_cache) -> None:
    """Assert the per-episode .npz matches the A5.15 read-contract exactly."""
    cache_dir, _ = tiny_cache
    from pathlib import Path
    data = dict(np.load(Path(cache_dir) / "ep0.npz"))
    n = 4
    assert data["latents"].shape == (n, PATCH_TOKENS, EMBED_DIM)
    assert data["latents"].dtype == LATENT_DTYPE
    assert data["actions"].shape == (n,)
    assert data["deltas"].shape == (n, DOF)
    assert data["deltas"].dtype == DELTA_DTYPE
    assert data["lang_tokens"].ndim == 2 and data["lang_tokens"].shape[1] == EMBED_DIM
    assert data["lang_tokens"].dtype == LATENT_DTYPE
    assert data["waypoint_4dof"].shape == (n, DOF)
    assert data["waypoint_4dof"].dtype == DELTA_DTYPE
    assert data["teacher_pose6"].shape == (n, TEACHER_DOF)
    assert data["teacher_pose6"].dtype == np.float32
    assert data["rollpitch_resid"].shape == (n,)
    assert data["rollpitch_resid"].dtype == np.float32
    assert data["disagreement"].shape == (n,)
    assert data["disagreement"].dtype == np.float32
    assert data["vjepa_surprise"].shape == (n,)
    assert data["vjepa_surprise"].dtype == np.float32


def test_npz_keys_and_shapes_ep1(tiny_cache) -> None:
    cache_dir, _ = tiny_cache
    from pathlib import Path
    data = dict(np.load(Path(cache_dir) / "ep1.npz"))
    n = 3
    assert data["latents"].shape == (n, PATCH_TOKENS, EMBED_DIM)
    assert data["actions"].shape == (n,)
    assert data["vjepa_surprise"].shape == (n,)


def test_roundtrip_through_cached_latent_dataset(tiny_cache) -> None:
    """The mocked .npz round-trips through CachedLatentDataset (the A5.15 loader)."""
    cache_dir, _ = tiny_cache
    ds = CachedLatentDataset(cache_dir)
    # ep0: N=4 → 3 transitions; ep1: N=3 → 2 transitions → total 5
    assert len(ds) == 5
    for i in range(len(ds)):
        step, oracle = ds[i]
        assert step.z_t.shape == (PATCH_TOKENS, EMBED_DIM)
        assert step.z_t.dtype == LATENT_DTYPE
        assert step.z_next.shape == (PATCH_TOKENS, EMBED_DIM)
        assert 0 <= step.action_id < N_ACTIONS
        assert oracle.waypoint_4dof.shape == (DOF,)
        assert oracle.disagreement >= 0.0
        assert oracle.vjepa_surprise >= 0.0
        assert oracle.rollpitch_resid >= 0.0


def test_resumable_skip_existing(tmp_path, monkeypatch) -> None:
    """If the .npz already exists, build_cache skips re-rendering."""
    if not _HAS_CV2:
        monkeypatch.setattr("vllatent.cache.resize_square", _numpy_resize_square)
    episodes = [_make_episode("ep0", n_poses=3, scene_id=1)]
    cache_dir = tmp_path / "cache"
    build_cache(
        episodes,
        cache_dir,
        renderer=_mock_renderer(),
        vision_encoder=_mock_vision_encoder(),
        text_encoder=_mock_text_encoder(),
        teacher_client=_mock_teacher_client(),
        verifier=_mock_verifier(),
    )
    # Build again with a renderer that would fail if called
    boom_renderer = MagicMock()
    boom_renderer.render_reference_row = MagicMock(side_effect=RuntimeError("should not render"))
    manifest = build_cache(
        episodes,
        cache_dir,
        renderer=boom_renderer,
        vision_encoder=_mock_vision_encoder(),
        text_encoder=_mock_text_encoder(),
        teacher_client=_mock_teacher_client(),
        verifier=_mock_verifier(),
    )
    errors = validate_manifest(manifest)
    assert errors == []
    boom_renderer.render_reference_row.assert_not_called()


def test_oracle_target_values_nonnegative(tiny_cache) -> None:
    """All OracleTarget scalars are non-negative (the schema requirement)."""
    cache_dir, _ = tiny_cache
    ds = CachedLatentDataset(cache_dir)
    for i in range(len(ds)):
        _, oracle = ds[i]
        assert oracle.disagreement >= 0.0
        assert oracle.vjepa_surprise >= 0.0
        assert oracle.rollpitch_resid >= 0.0
