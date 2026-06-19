"""Tests for vllatent.sports.cache — cache assembly + manifest."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vllatent.schemas import EMBED_DIM, PATCH_TOKENS
from vllatent.sports.cache import (
    build_clip_npz,
    build_sports_manifest,
    validate_sports_manifest,
    write_clip_npz,
    write_sports_manifest,
)


def _make_arrays(n: int = 20) -> dict[str, np.ndarray]:
    """Minimal valid arrays for N frames."""
    return {
        "latents": np.random.randn(n, PATCH_TOKENS, EMBED_DIM).astype(np.float16),
        "deltas": np.random.randn(n - 1, 4).astype(np.float32),
        "vo_confidence": np.random.rand(n).astype(np.float32),
        "frame_quality": np.random.rand(n).astype(np.float32),
        "timestamps": np.linspace(0.0, (n - 1) * 0.2, n),
        "quality_mask": np.ones(n, dtype=np.bool_),
    }


class TestBuildClipNpz:
    def test_valid(self) -> None:
        arrs = _make_arrays(10)
        result = build_clip_npz(**arrs)
        assert result["latents"].shape == (10, PATCH_TOKENS, EMBED_DIM)
        assert result["latents"].dtype == np.float16
        assert result["deltas"].shape == (9, 4)
        assert result["deltas"].dtype == np.float32
        assert result["vo_confidence"].dtype == np.float32
        assert result["timestamps"].dtype == np.float64
        assert result["quality_mask"].dtype == np.bool_

    def test_bad_latent_shape(self) -> None:
        arrs = _make_arrays(10)
        arrs["latents"] = np.zeros((10, 100, EMBED_DIM), dtype=np.float16)
        with pytest.raises(ValueError, match="latents"):
            build_clip_npz(**arrs)

    def test_bad_delta_shape(self) -> None:
        arrs = _make_arrays(10)
        arrs["deltas"] = np.zeros((10, 4), dtype=np.float32)  # should be (9, 4)
        with pytest.raises(ValueError, match="deltas"):
            build_clip_npz(**arrs)

    def test_bad_confidence_shape(self) -> None:
        arrs = _make_arrays(10)
        arrs["vo_confidence"] = np.zeros(5, dtype=np.float32)
        with pytest.raises(ValueError, match="vo_confidence"):
            build_clip_npz(**arrs)

    def test_bad_quality_shape(self) -> None:
        arrs = _make_arrays(10)
        arrs["frame_quality"] = np.zeros(5, dtype=np.float32)
        with pytest.raises(ValueError, match="frame_quality"):
            build_clip_npz(**arrs)

    def test_bad_timestamps_shape(self) -> None:
        arrs = _make_arrays(10)
        arrs["timestamps"] = np.zeros(5)
        with pytest.raises(ValueError, match="timestamps"):
            build_clip_npz(**arrs)

    def test_bad_mask_shape(self) -> None:
        arrs = _make_arrays(10)
        arrs["quality_mask"] = np.zeros(5, dtype=np.bool_)
        with pytest.raises(ValueError, match="quality_mask"):
            build_clip_npz(**arrs)

    def test_dtype_coercion(self) -> None:
        arrs = _make_arrays(5)
        arrs["latents"] = arrs["latents"].astype(np.float32)
        arrs["deltas"] = arrs["deltas"].astype(np.float64)
        result = build_clip_npz(**arrs)
        assert result["latents"].dtype == np.float16
        assert result["deltas"].dtype == np.float32


class TestWriteClipNpz:
    def test_round_trip(self, tmp_path: Path) -> None:
        arrs = _make_arrays(8)
        arrays = build_clip_npz(**arrs)
        out = write_clip_npz(arrays, tmp_path / "clip01.npz")
        assert out.exists()

        loaded = dict(np.load(str(out)))
        assert loaded["latents"].shape == (8, PATCH_TOKENS, EMBED_DIM)
        assert loaded["deltas"].shape == (7, 4)
        assert loaded["vo_confidence"].shape == (8,)
        assert loaded["quality_mask"].shape == (8,)

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        arrs = _make_arrays(5)
        arrays = build_clip_npz(**arrs)
        out = write_clip_npz(arrays, tmp_path / "sub" / "dir" / "clip.npz")
        assert out.exists()


class TestBuildSportsManifest:
    def test_defaults(self) -> None:
        m = build_sports_manifest(encoder_model_id="vit_base_patch16_dinov3.lvd1689m")
        assert m["cache_version"] == "0.2"
        assert m["encoder"]["model_id"] == "vit_base_patch16_dinov3.lvd1689m"
        assert m["encoder"]["patch_tokens"] == PATCH_TOKENS
        assert m["encoder"]["dim"] == EMBED_DIM
        assert m["dataset"]["name"] == "sports_following"
        assert m["dataset"]["sport"] == "skiing"
        assert m["motion_source"]["method"] == "megasam"
        assert m["motion_source"]["scale_mode"] == "normalized"
        assert m["entries"] == []

    def test_with_entries(self) -> None:
        entries = [
            {"clip_id": "ski01", "n_frames": 100, "latent_path": "ski01.npz"},
            {"clip_id": "ski02", "n_frames": 80, "latent_path": "ski02.npz"},
        ]
        m = build_sports_manifest(
            encoder_model_id="test",
            entries=entries,
            sport="snowboarding",
        )
        assert len(m["entries"]) == 2
        assert m["dataset"]["sport"] == "snowboarding"

    def test_no_teacher_section(self) -> None:
        m = build_sports_manifest(encoder_model_id="test")
        assert "teacher" not in m


class TestValidateSportsManifest:
    def test_valid(self) -> None:
        m = build_sports_manifest(
            encoder_model_id="test",
            entries=[{"clip_id": "ski01", "n_frames": 50, "latent_path": "ski01.npz"}],
        )
        assert validate_sports_manifest(m) == []

    def test_missing_top_level(self) -> None:
        errs = validate_sports_manifest({"cache_version": "0.2"})
        assert any("missing top-level" in e for e in errs)

    def test_missing_encoder_keys(self) -> None:
        m = build_sports_manifest(encoder_model_id="test")
        del m["encoder"]["dtype"]
        errs = validate_sports_manifest(m)
        assert any("encoder missing key: dtype" in e for e in errs)

    def test_wrong_dataset_name(self) -> None:
        m = build_sports_manifest(encoder_model_id="test")
        m["dataset"]["name"] = "aerialvln"
        errs = validate_sports_manifest(m)
        assert any("sports_following" in e for e in errs)

    def test_missing_motion_source_keys(self) -> None:
        m = build_sports_manifest(encoder_model_id="test")
        del m["motion_source"]["scale_mode"]
        errs = validate_sports_manifest(m)
        assert any("motion_source missing key: scale_mode" in e for e in errs)

    def test_missing_entry_keys(self) -> None:
        m = build_sports_manifest(
            encoder_model_id="test",
            entries=[{"clip_id": "ski01"}],
        )
        errs = validate_sports_manifest(m)
        assert any("n_frames" in e for e in errs)
        assert any("latent_path" in e for e in errs)

    def test_bad_entry_type(self) -> None:
        m = build_sports_manifest(encoder_model_id="test", entries=["not_a_dict"])
        errs = validate_sports_manifest(m)
        assert any("expected dict" in e for e in errs)


class TestWriteSportsManifest:
    def test_write_and_read(self, tmp_path: Path) -> None:
        m = build_sports_manifest(
            encoder_model_id="test",
            entries=[{"clip_id": "ski01", "n_frames": 50, "latent_path": "ski01.npz"}],
        )
        path = write_sports_manifest(m, tmp_path)
        assert path.name == "manifest.json"

        loaded = json.loads(path.read_text())
        assert loaded["cache_version"] == "0.2"
        assert loaded["entries"][0]["clip_id"] == "ski01"

    def test_creates_dirs(self, tmp_path: Path) -> None:
        m = build_sports_manifest(encoder_model_id="test")
        path = write_sports_manifest(m, tmp_path / "deep" / "nested")
        assert path.exists()
