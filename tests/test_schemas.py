"""Step-3 tests: loader-tuple SCHEMAS (PURE tier).

Constructs each schema with synthetic numpy arrays; asserts shapes/dtypes,
boundary validation, immutability, and the manifest-entry JSON round-trip.
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from vllatent.config import DISAGREEMENT_SOURCES, Config
from vllatent.manifest import build_manifest, empty_manifest, validate_manifest
from vllatent.schemas import (
    DELTA_DTYPE,
    DOF,
    EMBED_DIM,
    HISTORY,
    LATENT_DTYPE,
    N_ACTIONS,
    PATCH_TOKENS,
    CacheManifestEntry,
    EpisodeRecord,
    StepSample,
)


def _step_sample(**over: object) -> StepSample:
    kw: dict[str, object] = dict(
        z_t=np.zeros((PATCH_TOKENS, EMBED_DIM), LATENT_DTYPE),
        history_latents=np.zeros((HISTORY, PATCH_TOKENS, EMBED_DIM), LATENT_DTYPE),
        lang_tokens=np.zeros((5, EMBED_DIM), LATENT_DTYPE),
        action_id=1,
        z_next=np.zeros((PATCH_TOKENS, EMBED_DIM), LATENT_DTYPE),
        delta_4dof=np.zeros((DOF,), DELTA_DTYPE),
    )
    kw.update(over)
    return StepSample(**kw)  # type: ignore[arg-type]


def _episode(**over: object) -> EpisodeRecord:
    n = 4
    kw: dict[str, object] = dict(
        episode_id="ep0",
        trajectory_id="traj0",
        scene_id=1,
        instruction_text="go to the fountain",
        start_position=np.zeros(3),
        start_rotation_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        goal_positions=np.zeros((1, 3)),
        actions=np.array([1, 6, 7, 0], dtype=int),
        reference_path=np.zeros((n + 1, 6)),  # [x,y,z,pitch,roll,yaw] Euler
    )
    kw.update(over)
    return EpisodeRecord(**kw)  # type: ignore[arg-type]


# --- StepSample ---

def test_stepsample_shapes_and_dtypes() -> None:
    s = _step_sample()
    assert s.z_t.shape == (PATCH_TOKENS, EMBED_DIM)
    assert s.z_t.dtype == LATENT_DTYPE
    assert s.z_next.dtype == LATENT_DTYPE
    assert s.history_latents.shape == (HISTORY, PATCH_TOKENS, EMBED_DIM)
    assert s.lang_tokens.shape[1] == EMBED_DIM
    assert s.delta_4dof.shape == (DOF,)
    assert s.delta_4dof.dtype == DELTA_DTYPE
    assert 0 <= s.action_id < N_ACTIONS
    assert s.future_frame_rgb is None


def test_stepsample_optional_future_frame() -> None:
    s = _step_sample(future_frame_rgb=np.zeros((224, 224, 3), np.uint8))
    assert s.future_frame_rgb is not None
    assert s.future_frame_rgb.shape == (224, 224, 3)
    assert s.future_frame_rgb.dtype == np.uint8


def test_stepsample_is_immutable() -> None:
    s = _step_sample()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.action_id = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    "bad",
    [
        dict(z_t=np.zeros((PATCH_TOKENS, EMBED_DIM), np.float32)),               # wrong dtype
        dict(z_t=np.zeros((10, EMBED_DIM), LATENT_DTYPE)),                       # wrong token count
        dict(delta_4dof=np.zeros((3,), DELTA_DTYPE)),                           # wrong DoF
        dict(delta_4dof=np.zeros((DOF,), np.float16)),                          # wrong delta dtype
        dict(history_latents=np.zeros((2, PATCH_TOKENS, EMBED_DIM), LATENT_DTYPE)),  # wrong H
        dict(lang_tokens=np.zeros((5, 512), LATENT_DTYPE)),                     # wrong embed dim
        dict(action_id=N_ACTIONS),                                             # out of range high
        dict(action_id=-1),                                                    # out of range low
        dict(action_id=True),                                                  # bool is not a valid id
        dict(z_t=[[0.0] * EMBED_DIM] * PATCH_TOKENS),                          # not an ndarray
    ],
)
def test_stepsample_rejects_bad_inputs(bad: dict[str, object]) -> None:
    with pytest.raises((ValueError, TypeError)):
        _step_sample(**bad)


# --- EpisodeRecord ---

def test_episoderecord_construct_and_immutable() -> None:
    e = _episode()
    assert e.scene_id == 1
    assert e.start_rotation_xyzw.shape == (4,)
    assert e.reference_path.shape[1] == 6
    assert e.actions.dtype.kind == "i"
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.scene_id = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    "bad",
    [
        dict(start_rotation_xyzw=np.zeros(3)),          # quaternion must be (4,)
        dict(start_position=np.zeros(2)),               # position must be (3,)
        dict(reference_path=np.zeros((5, 7))),          # pose row must be 6-wide [x,y,z,pitch,roll,yaw]
        dict(goal_positions=np.zeros((1, 2))),          # goal must be (G,3)
        dict(actions=np.zeros(4, dtype=float)),         # actions must be integer-kind
        dict(scene_id="1"),                             # scene_id must be int
    ],
)
def test_episoderecord_rejects_bad_inputs(bad: dict[str, object]) -> None:
    with pytest.raises((ValueError, TypeError)):
        _episode(**bad)


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


def test_build_manifest_teacher_provenance_is_stubbed() -> None:
    m = build_manifest(Config())
    t = m["teacher"]
    assert set(t) == {
        "worldvln_model_id",
        "worldvln_revision",
        "disagreement_source",
        "vjepa2_model_id",
        "render_config_hash",
    }
    # Stubs now (populated by the cache build in A5.14), except disagreement_source from Config.
    assert t["worldvln_model_id"] == "" and t["worldvln_revision"] == ""
    assert t["vjepa2_model_id"] == "" and t["render_config_hash"] == ""
    assert t["disagreement_source"] in DISAGREEMENT_SOURCES


def test_build_manifest_reflects_config_sweep_no_code_surgery() -> None:
    # Flipping a trust knob in Config flows into the manifest — the whole point of the SoT.
    cfg = dataclasses.replace(
        Config(), trust=dataclasses.replace(Config().trust, disagreement_source="airscape_multiseed")
    )
    m = build_manifest(cfg)
    assert m["teacher"]["disagreement_source"] == "airscape_multiseed"
    assert validate_manifest(m) == []


def test_build_manifest_with_typed_entries_roundtrips() -> None:
    entry = CacheManifestEntry("ep0", 2, 42, "ep0/latents.npy", trajectory_id="traj0").to_dict()
    m = build_manifest(Config(), entries=[entry])
    assert validate_manifest(m) == []
    assert m["entries"][0]["episode_id"] == "ep0"


def test_validate_manifest_rejects_missing_teacher_section() -> None:
    m = empty_manifest()
    del m["teacher"]
    assert any("missing key: teacher" in e for e in validate_manifest(m))


def test_validate_manifest_entry_keys_enforced_from_type() -> None:
    m = empty_manifest()
    m["entries"].append({"episode_id": "ep0"})  # missing scene_id / n_frames / latent_path
    errs = validate_manifest(m)
    for k in ("scene_id", "n_frames", "latent_path"):
        assert any(f"missing key: {k}" in e for e in errs)
