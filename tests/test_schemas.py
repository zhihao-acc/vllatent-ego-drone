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
    HISTORY,
    HORIZON,
    LATENT_DTYPE,
    N_ACTIONS,
    PATCH_TOKENS,
    CacheManifestEntry,
    EpisodeRecord,
    PredictorOutput,
    SportsTarget,
    StepSample,
    Target,
    Waypoint,
)


def _step_sample(**over: object) -> StepSample:
    kw: dict[str, object] = dict(
        z_t=np.zeros((PATCH_TOKENS, EMBED_DIM), LATENT_DTYPE),
        history_latents=np.zeros((HISTORY, PATCH_TOKENS, EMBED_DIM), LATENT_DTYPE),
        history_mask=np.ones((HISTORY,), bool),
        lang_tokens=np.zeros((5, EMBED_DIM), LATENT_DTYPE),
        lang_mask=np.ones((5,), bool),
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


def test_stepsample_masks_are_real_fields() -> None:
    # M4: history_mask + language padding-mask are explicit boolean fields the loader honors.
    s = _step_sample(
        history_mask=np.array([False, True, True]),
        lang_tokens=np.zeros((3, EMBED_DIM), LATENT_DTYPE),
        lang_mask=np.array([True, True, False]),
    )
    assert s.history_mask.shape == (HISTORY,) and s.history_mask.dtype == np.bool_
    assert s.lang_mask.shape == (3,) and s.lang_mask.dtype == np.bool_
    assert not bool(s.history_mask[0]) and bool(s.history_mask[1])


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
        dict(history_mask=np.ones((HISTORY,), np.float32)),                     # mask wrong dtype
        dict(history_mask=np.ones((HISTORY + 1,), bool)),                       # mask wrong length
        dict(lang_tokens=np.zeros((5, 512), LATENT_DTYPE)),                     # wrong embed dim
        dict(lang_mask=np.ones((4,), bool)),                                   # mask length != M
        dict(lang_mask=np.ones((5,), np.int8)),                               # mask wrong dtype
        dict(action_id=N_ACTIONS),                                             # out of range high
        dict(action_id=-1),                                                    # out of range low
        dict(action_id=True),                                                  # bool is not a valid id
        dict(z_t=[[0.0] * EMBED_DIM] * PATCH_TOKENS),                          # not an ndarray
    ],
)
def test_stepsample_rejects_bad_inputs(bad: dict[str, object]) -> None:
    with pytest.raises((ValueError, TypeError)):
        _step_sample(**bad)


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


def test_waypoint_valid_and_rejects_bad() -> None:
    wp = Waypoint(delta_4dof=np.zeros((DOF,), DELTA_DTYPE))
    assert wp.delta_4dof.shape == (DOF,) and wp.delta_4dof.dtype == DELTA_DTYPE
    with pytest.raises((ValueError, TypeError)):
        Waypoint(delta_4dof=np.zeros((DOF,), np.float16))   # wrong dtype (must be f32)
    with pytest.raises((ValueError, TypeError)):
        Waypoint(delta_4dof=np.zeros((3,), DELTA_DTYPE))    # wrong DoF


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


# --- StepSample optional ingest metadata ---

def test_stepsample_optional_ingest_metadata_none_by_default() -> None:
    s = _step_sample()
    assert s.vo_confidence is None
    assert s.frame_quality is None
    assert s.dt_seconds is None


def test_stepsample_optional_ingest_metadata_valid() -> None:
    s = _step_sample(vo_confidence=0.95, frame_quality=0.7, dt_seconds=0.2)
    assert s.vo_confidence == 0.95
    assert s.frame_quality == 0.7
    assert s.dt_seconds == 0.2


@pytest.mark.parametrize("field,bad_val,err_type", [
    ("vo_confidence", 1.5, ValueError),
    ("vo_confidence", -0.1, ValueError),
    ("vo_confidence", "nope", TypeError),
    ("frame_quality", 2.0, ValueError),
    ("frame_quality", -0.5, ValueError),
    ("dt_seconds", 0.0, ValueError),
    ("dt_seconds", -1.0, ValueError),
])
def test_stepsample_ingest_metadata_rejects_bad(field, bad_val, err_type) -> None:
    with pytest.raises(err_type):
        _step_sample(**{field: bad_val})


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
