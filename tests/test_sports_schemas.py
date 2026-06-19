"""Tests for vllatent.sports.schemas — SportsSample / SportsClipRecord / SportsClipManifestEntry."""
from __future__ import annotations

import numpy as np
import pytest

from vllatent.schemas import DELTA_DTYPE, EMBED_DIM, HISTORY, LATENT_DTYPE, MASK_DTYPE, PATCH_TOKENS
from vllatent.sports.schemas import (
    CAMERA_MODELS,
    SCALE_MODES,
    SportsClipManifestEntry,
    SportsClipRecord,
    SportsSample,
)


def _make_sample(**overrides: object) -> SportsSample:
    defaults: dict[str, object] = {
        "z_t": np.zeros((PATCH_TOKENS, EMBED_DIM), dtype=LATENT_DTYPE),
        "history_latents": np.zeros((HISTORY, PATCH_TOKENS, EMBED_DIM), dtype=LATENT_DTYPE),
        "history_mask": np.array([False, False, True], dtype=MASK_DTYPE),
        "z_next": np.zeros((PATCH_TOKENS, EMBED_DIM), dtype=LATENT_DTYPE),
        "delta_4dof": np.array([1.0, 0.0, 0.0, 15.0], dtype=DELTA_DTYPE),
        "vo_confidence": 0.95,
        "frame_quality": 0.8,
        "dt_seconds": 0.2,
    }
    defaults.update(overrides)
    return SportsSample(**defaults)  # type: ignore[arg-type]


class TestSportsSample:
    def test_valid_construction(self) -> None:
        s = _make_sample()
        assert s.z_t.shape == (PATCH_TOKENS, EMBED_DIM)
        assert s.z_t.dtype == LATENT_DTYPE
        assert s.vo_confidence == 0.95
        assert s.frame_quality == 0.8
        assert s.dt_seconds == 0.2

    def test_frozen(self) -> None:
        s = _make_sample()
        with pytest.raises(AttributeError):
            s.vo_confidence = 0.5  # type: ignore[misc]

    def test_bad_z_t_shape(self) -> None:
        with pytest.raises(ValueError, match="z_t"):
            _make_sample(z_t=np.zeros((100, EMBED_DIM), dtype=LATENT_DTYPE))

    def test_bad_z_t_dtype(self) -> None:
        with pytest.raises(ValueError, match="z_t"):
            _make_sample(z_t=np.zeros((PATCH_TOKENS, EMBED_DIM), dtype=np.float32))

    def test_bad_history_shape(self) -> None:
        with pytest.raises(ValueError, match="history_latents"):
            _make_sample(history_latents=np.zeros((2, PATCH_TOKENS, EMBED_DIM), dtype=LATENT_DTYPE))

    def test_bad_delta_shape(self) -> None:
        with pytest.raises(ValueError, match="delta_4dof"):
            _make_sample(delta_4dof=np.array([1.0, 0.0, 0.0], dtype=DELTA_DTYPE))

    def test_vo_confidence_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="vo_confidence"):
            _make_sample(vo_confidence=1.5)

    def test_vo_confidence_negative(self) -> None:
        with pytest.raises(ValueError, match="vo_confidence"):
            _make_sample(vo_confidence=-0.1)

    def test_frame_quality_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="frame_quality"):
            _make_sample(frame_quality=2.0)

    def test_dt_seconds_zero(self) -> None:
        with pytest.raises(ValueError, match="dt_seconds"):
            _make_sample(dt_seconds=0.0)

    def test_dt_seconds_negative(self) -> None:
        with pytest.raises(ValueError, match="dt_seconds"):
            _make_sample(dt_seconds=-1.0)

    def test_vo_confidence_bool_rejected(self) -> None:
        with pytest.raises(TypeError, match="vo_confidence"):
            _make_sample(vo_confidence=True)

    def test_boundary_values(self) -> None:
        s = _make_sample(vo_confidence=0.0, frame_quality=0.0, dt_seconds=0.001)
        assert s.vo_confidence == 0.0
        s2 = _make_sample(vo_confidence=1.0, frame_quality=1.0)
        assert s2.vo_confidence == 1.0


class TestSportsClipRecord:
    def test_valid_construction(self) -> None:
        r = SportsClipRecord(
            clip_id="ski01",
            source_url="https://youtube.com/watch?v=abc",
            sport="skiing",
            n_frames=150,
            fps_original=30.0,
            fps_sampled=5.0,
            duration_seconds=30.0,
            scale_mode="normalized",
            camera_model="pinhole_undistorted",
        )
        assert r.clip_id == "ski01"
        assert r.n_frames == 150

    def test_frozen(self) -> None:
        r = SportsClipRecord(
            clip_id="ski01", source_url="url", sport="skiing", n_frames=100,
            fps_original=30.0, fps_sampled=5.0, duration_seconds=20.0,
            scale_mode="normalized", camera_model="pinhole_undistorted",
        )
        with pytest.raises(AttributeError):
            r.clip_id = "other"  # type: ignore[misc]

    def test_empty_clip_id(self) -> None:
        with pytest.raises(ValueError, match="clip_id"):
            SportsClipRecord(
                clip_id="", source_url="url", sport="skiing", n_frames=100,
                fps_original=30.0, fps_sampled=5.0, duration_seconds=20.0,
                scale_mode="normalized", camera_model="pinhole_undistorted",
            )

    def test_bad_n_frames(self) -> None:
        with pytest.raises(ValueError, match="n_frames"):
            SportsClipRecord(
                clip_id="ski01", source_url="url", sport="skiing", n_frames=0,
                fps_original=30.0, fps_sampled=5.0, duration_seconds=20.0,
                scale_mode="normalized", camera_model="pinhole_undistorted",
            )

    def test_bad_scale_mode(self) -> None:
        with pytest.raises(ValueError, match="scale_mode"):
            SportsClipRecord(
                clip_id="ski01", source_url="url", sport="skiing", n_frames=100,
                fps_original=30.0, fps_sampled=5.0, duration_seconds=20.0,
                scale_mode="unknown", camera_model="pinhole_undistorted",
            )

    def test_bad_camera_model(self) -> None:
        with pytest.raises(ValueError, match="camera_model"):
            SportsClipRecord(
                clip_id="ski01", source_url="url", sport="skiing", n_frames=100,
                fps_original=30.0, fps_sampled=5.0, duration_seconds=20.0,
                scale_mode="normalized", camera_model="equirectangular",
            )

    def test_all_scale_modes(self) -> None:
        for mode in SCALE_MODES:
            r = SportsClipRecord(
                clip_id="t", source_url="u", sport="s", n_frames=1,
                fps_original=1.0, fps_sampled=1.0, duration_seconds=1.0,
                scale_mode=mode, camera_model="pinhole_undistorted",
            )
            assert r.scale_mode == mode

    def test_all_camera_models(self) -> None:
        for model in CAMERA_MODELS:
            r = SportsClipRecord(
                clip_id="t", source_url="u", sport="s", n_frames=1,
                fps_original=1.0, fps_sampled=1.0, duration_seconds=1.0,
                scale_mode="normalized", camera_model=model,
            )
            assert r.camera_model == model


class TestSportsClipManifestEntry:
    def _make_entry(self) -> SportsClipManifestEntry:
        return SportsClipManifestEntry(
            clip_id="ski01",
            n_frames=150,
            latent_path="ski01.npz",
            source_url="https://youtube.com/watch?v=abc",
            sport="skiing",
            scale_mode="normalized",
            fps_sampled=5.0,
            duration_seconds=30.0,
        )

    def test_to_dict(self) -> None:
        e = self._make_entry()
        d = e.to_dict()
        assert d["clip_id"] == "ski01"
        assert d["n_frames"] == 150
        assert d["latent_path"] == "ski01.npz"
        assert d["fps_sampled"] == 5.0

    def test_from_dict_roundtrip(self) -> None:
        e = self._make_entry()
        d = e.to_dict()
        e2 = SportsClipManifestEntry.from_dict(d)
        assert e == e2

    def test_required_keys(self) -> None:
        keys = SportsClipManifestEntry.required_keys()
        assert "clip_id" in keys
        assert "n_frames" in keys
        assert "latent_path" in keys
        assert len(keys) == 8
