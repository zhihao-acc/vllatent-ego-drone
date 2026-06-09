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
    HORIZON,
    LATENT_DTYPE,
    N_ACTIONS,
    PATCH_TOKENS,
    CacheManifestEntry,
    EpisodeRecord,
    PredictorOutput,
    StepSample,
    TrustReadout,
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


def test_trust_readout_valid() -> None:
    tr = TrustReadout(p_commit=np.full((HORIZON,), 0.5), k_star=2.0, sigma=0.1)
    assert tr.p_commit.shape == (HORIZON,)
    assert 0.0 <= tr.k_star <= HORIZON and tr.sigma >= 0.0


def test_trust_readout_accepts_int_scalars() -> None:
    # k_star/sigma may arrive as ints; both must validate (int is a valid float here).
    tr = TrustReadout(p_commit=np.zeros((HORIZON,)), k_star=0, sigma=0)
    assert tr.k_star == 0 and tr.sigma == 0


@pytest.mark.parametrize(
    "kw",
    [
        dict(p_commit=np.full((HORIZON,), 1.5)),          # probability above 1
        dict(p_commit=np.full((HORIZON,), -0.1)),         # probability below 0
        dict(p_commit=np.zeros((HORIZON + 1,))),          # wrong horizon
        dict(p_commit=np.zeros((HORIZON,), dtype=int)),   # not float-kind
        dict(k_star=-1.0),                                # k* below 0
        dict(k_star=float(HORIZON) + 1.0),                # k* above T
        dict(k_star=True),                                # bool is not a valid k*
        dict(sigma=-0.01),                                # sigma negative
        dict(sigma=float("nan")),                         # sigma must be finite
        dict(sigma=float("inf")),                         # sigma must be finite
    ],
)
def test_trust_readout_rejects_bad(kw: dict[str, object]) -> None:
    base: dict[str, object] = dict(p_commit=np.full((HORIZON,), 0.5), k_star=1.0, sigma=0.1)
    base.update(kw)
    with pytest.raises((ValueError, TypeError)):
        TrustReadout(**base)  # type: ignore[arg-type]


def test_waypoint_valid_and_rejects_bad() -> None:
    wp = Waypoint(delta_4dof=np.zeros((DOF,), DELTA_DTYPE))
    assert wp.delta_4dof.shape == (DOF,) and wp.delta_4dof.dtype == DELTA_DTYPE
    with pytest.raises((ValueError, TypeError)):
        Waypoint(delta_4dof=np.zeros((DOF,), np.float16))   # wrong dtype (must be f32)
    with pytest.raises((ValueError, TypeError)):
        Waypoint(delta_4dof=np.zeros((3,), DELTA_DTYPE))    # wrong DoF


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


def test_validate_manifest_rejects_bad_disagreement_source() -> None:
    m = empty_manifest()
    m["teacher"]["disagreement_source"] = "bogus_source"
    assert any("disagreement_source" in e for e in validate_manifest(m))


def test_validate_manifest_accepts_empty_disagreement_source_stub() -> None:
    m = empty_manifest()
    m["teacher"]["disagreement_source"] = ""  # the allowed stub (provenance populated in A5.14)
    assert validate_manifest(m) == []


def test_validate_manifest_rejects_missing_dataset_key() -> None:
    m = empty_manifest()
    del m["dataset"]["license"]
    assert any("dataset missing keys" in e for e in validate_manifest(m))


def test_validate_manifest_rejects_missing_teacher_key() -> None:
    m = empty_manifest()
    del m["teacher"]["vjepa2_model_id"]
    assert any("teacher missing keys" in e for e in validate_manifest(m))
