"""Step-3 tests: loader-tuple SCHEMAS (PURE tier).

Constructs each schema with synthetic numpy arrays; asserts shapes/dtypes,
boundary validation, immutability, and the manifest-entry JSON round-trip.
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from vllatent.config import Config
from vllatent.manifest import build_manifest, empty_manifest, validate_manifest
from vllatent.schemas import (
    DELTA_DTYPE,
    DOF,
    EMBED_DIM,
    HORIZON,
    LATENT_DTYPE,
    PATCH_TOKENS,
    CacheManifestEntry,
    PredictorOutput,
    SportsTarget,
    Target,
)

# --- Student output seams (A5.5, H3) ---

def test_predictor_output_valid_and_immutable() -> None:
    po = PredictorOutput(predicted_latents=np.zeros((HORIZON, PATCH_TOKENS, EMBED_DIM), LATENT_DTYPE))
    assert po.predicted_latents.shape == (HORIZON, PATCH_TOKENS, EMBED_DIM)
    assert po.predicted_latents.dtype == LATENT_DTYPE
    with pytest.raises(dataclasses.FrozenInstanceError):
        po.predicted_latents = po.predicted_latents  # type: ignore[misc]


@pytest.mark.parametrize(
    "arr",
    [
        np.zeros((HORIZON, PATCH_TOKENS, EMBED_DIM), np.float32),        # wrong dtype (must be fp16)
        np.zeros((HORIZON + 1, PATCH_TOKENS, EMBED_DIM), LATENT_DTYPE),  # wrong horizon T
        np.zeros((HORIZON, 10, EMBED_DIM), LATENT_DTYPE),               # wrong token count
        np.zeros((PATCH_TOKENS, EMBED_DIM), LATENT_DTYPE),             # wrong ndim
    ],
)
def test_predictor_output_rejects_bad(arr: np.ndarray) -> None:
    with pytest.raises((ValueError, TypeError)):
        PredictorOutput(predicted_latents=arr)


# --- SportsTarget (B1.6) ---

def _sports_target(**over: object) -> SportsTarget:
    kw: dict[str, object] = dict(
        waypoint_4dof=np.zeros((DOF,), DELTA_DTYPE),
    )
    kw.update(over)
    return SportsTarget(**kw)  # type: ignore[arg-type]


def test_sports_target_valid_and_immutable() -> None:
    st = _sports_target()
    assert st.waypoint_4dof.shape == (DOF,) and st.waypoint_4dof.dtype == DELTA_DTYPE
    with pytest.raises(dataclasses.FrozenInstanceError):
        st.waypoint_4dof = np.zeros((DOF,), DELTA_DTYPE)  # type: ignore[misc]


@pytest.mark.parametrize(
    "bad",
    [
        dict(waypoint_4dof=np.zeros((DOF,), np.float16)),    # wrong dtype (must be f32)
        dict(waypoint_4dof=np.zeros((3,), DELTA_DTYPE)),     # wrong DoF
        dict(waypoint_4dof=np.zeros((DOF,), np.int32)),      # wrong dtype
    ],
)
def test_sports_target_rejects_bad(bad: dict[str, object]) -> None:
    with pytest.raises((ValueError, TypeError)):
        _sports_target(**bad)


def test_target_type_alias_is_sports_target() -> None:
    sports: Target = _sports_target()
    assert isinstance(sports, SportsTarget)


# --- CacheManifestEntry ---

def test_manifest_entry_json_roundtrip() -> None:
    e = CacheManifestEntry(
        episode_id="ep0", scene_id=2, n_frames=42, latent_path="ep0/latents.npy", trajectory_id="traj0"
    )
    e2 = CacheManifestEntry.from_dict(json.loads(json.dumps(e.to_dict())))
    assert e == e2


def test_manifest_entry_satisfies_manifest_validator() -> None:
    m = empty_manifest()
    m["entries"].append(CacheManifestEntry("ep0", 2, 42, "ep0/latents.npy").to_dict())
    assert validate_manifest(m) == []


def test_manifest_entry_required_keys_are_the_no_default_fields() -> None:
    # The validator's per-entry required keys come from the type, not a hand-kept literal (M5).
    assert CacheManifestEntry.required_keys() == ("episode_id", "scene_id", "n_frames", "latent_path")


# --- build_manifest: typed builder fed from Config (A5.4, M5) ---

def test_build_manifest_from_config_is_valid_and_dedups_shapes() -> None:
    cfg = Config()
    m = build_manifest(cfg, split="train", variant="v1")
    assert validate_manifest(m) == []
    # De-dup: encoder identity + shapes come from Config / schemas constants, not re-hardcoded.
    assert m["encoder"]["model_id"] == cfg.encoder.model_id
    assert m["encoder"]["dtype"] == cfg.encoder.dtype
    assert m["encoder"]["patch_tokens"] == PATCH_TOKENS
    assert m["encoder"]["dim"] == EMBED_DIM
    assert m["cache_version"] == cfg.cache.version
    assert m["dataset"]["name"] == cfg.data.name
    assert m["dataset"]["license"] == cfg.data.license
    assert m["dataset"]["split"] == "train" and m["dataset"]["variant"] == "v1"
    assert m["convention"]["quaternion_order"] == cfg.cache.quaternion_order
    assert m["convention"]["color_order"] == cfg.cache.color_order


def test_build_manifest_with_typed_entries_roundtrips() -> None:
    entry = CacheManifestEntry("ep0", 2, 42, "ep0/latents.npy", trajectory_id="traj0").to_dict()
    m = build_manifest(Config(), entries=[entry])
    assert validate_manifest(m) == []
    assert m["entries"][0]["episode_id"] == "ep0"


def test_validate_manifest_entry_keys_enforced_from_type() -> None:
    m = empty_manifest()
    m["entries"].append({"episode_id": "ep0"})  # missing scene_id / n_frames / latent_path
    errs = validate_manifest(m)
    for k in ("scene_id", "n_frames", "latent_path"):
        assert any(f"missing key: {k}" in e for e in errs)


def test_build_manifest_reads_shapes_from_schemas_not_literals(monkeypatch: pytest.MonkeyPatch) -> None:
    # True de-dup proof: build_manifest must READ PATCH_TOKENS/EMBED_DIM from schemas, not re-hardcode
    # 196/768. Patch the constants in the manifest module's namespace and require the output to follow
    # (a re-hardcoded literal would ignore the patch and this test would fail).
    import vllatent.manifest as manifest_mod
    monkeypatch.setattr(manifest_mod, "PATCH_TOKENS", 4242)
    monkeypatch.setattr(manifest_mod, "EMBED_DIM", 1717)
    m = manifest_mod.build_manifest(Config())
    assert m["encoder"]["patch_tokens"] == 4242
    assert m["encoder"]["dim"] == 1717


def test_validate_manifest_rejects_missing_dataset_key() -> None:
    m = empty_manifest()
    del m["dataset"]["license"]
    assert any("dataset missing keys" in e for e in validate_manifest(m))


# --- Wild-video manifest ---

def test_validate_manifest_wild_video_valid() -> None:
    from vllatent.manifest import build_manifest_wild_video
    m = build_manifest_wild_video(
        encoder_model_id="vit_base_patch16_dinov3.lvd1689m",
        entries=[{"clip_id": "c1", "n_frames": 10, "latent_path": "c1.npz"}],
    )
    assert validate_manifest(m) == []


def test_validate_manifest_wild_video_missing_motion_source() -> None:
    from vllatent.manifest import build_manifest_wild_video
    m = build_manifest_wild_video(encoder_model_id="test")
    del m["motion_source"]
    errors = validate_manifest(m)
    assert any("motion_source" in e for e in errors)


def test_validate_manifest_wild_video_entry_keys() -> None:
    from vllatent.manifest import build_manifest_wild_video
    m = build_manifest_wild_video(
        encoder_model_id="test",
        entries=[{"clip_id": "c1"}],
    )
    errors = validate_manifest(m)
    assert any("n_frames" in e for e in errors)


def test_validate_manifest_wild_video_person_tracker_provenance() -> None:
    from vllatent.manifest import build_manifest_wild_video
    m = build_manifest_wild_video(
        encoder_model_id="test",
        person_tracker={
            "detector": "yolov8s-worldv2.pt",
            "tracker": "bytetrack",
            "classes": ["person", "skier"],
        },
    )
    assert m["person_tracker"]["tracker"] == "bytetrack"
    assert validate_manifest(m) == []
