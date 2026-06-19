"""Tests for vllatent.ingest.pipeline — end-to-end per-clip processing."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vllatent.config import IngestConfig
from vllatent.ingest.pipeline import (
    ClipPipelineResult,
    _build_clip_npz,
    _write_clip_npz,
    update_manifest_from_results,
)
from vllatent.schemas import EMBED_DIM, LATENT_DTYPE, MASK_DTYPE, PATCH_TOKENS


class TestBuildClipNpz:
    def _arrays(self, n: int = 10):
        return dict(
            latents=np.zeros((n, PATCH_TOKENS, EMBED_DIM), dtype=LATENT_DTYPE),
            deltas=np.zeros((n - 1, 4), dtype=np.float32),
            vo_confidence=np.ones(n, dtype=np.float32),
            frame_quality=np.ones(n, dtype=np.float32),
            timestamps=np.arange(n, dtype=np.float64),
            quality_mask=np.ones(n, dtype=MASK_DTYPE),
        )

    def test_valid(self) -> None:
        result = _build_clip_npz(**self._arrays(5))
        assert result["latents"].shape == (5, PATCH_TOKENS, EMBED_DIM)
        assert result["deltas"].shape == (4, 4)

    def test_bad_latent_shape(self) -> None:
        a = self._arrays(5)
        a["latents"] = np.zeros((5, 10, 10), dtype=LATENT_DTYPE)
        with pytest.raises(ValueError, match="latents"):
            _build_clip_npz(**a)

    def test_bad_delta_shape(self) -> None:
        a = self._arrays(5)
        a["deltas"] = np.zeros((5, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="deltas"):
            _build_clip_npz(**a)

    def test_bad_quality_shape(self) -> None:
        a = self._arrays(5)
        a["frame_quality"] = np.ones(3, dtype=np.float32)
        with pytest.raises(ValueError, match="frame_quality"):
            _build_clip_npz(**a)


class TestWriteClipNpz:
    def test_writes_file(self, tmp_path: Path) -> None:
        arrays = {
            "latents": np.zeros((3, PATCH_TOKENS, EMBED_DIM), dtype=LATENT_DTYPE),
            "deltas": np.zeros((2, 4), dtype=np.float32),
        }
        out = tmp_path / "sub" / "clip.npz"
        result = _write_clip_npz(arrays, out)
        assert result.exists()
        loaded = np.load(str(result))
        assert "latents" in loaded
        assert "deltas" in loaded


class TestClipPipelineResult:
    def test_fields(self) -> None:
        r = ClipPipelineResult(
            clip_id="test", n_frames=10, n_accepted=8,
            latent_path="test.npz", stages_skipped=["download"], errors=[],
        )
        assert r.clip_id == "test"
        assert r.n_accepted == 8

    def test_as_dict(self) -> None:
        r = ClipPipelineResult(
            clip_id="x", n_frames=5, n_accepted=5,
            latent_path="x.npz", stages_skipped=[], errors=[],
        )
        d = asdict(r)
        assert d["clip_id"] == "x"


class TestUpdateManifestFromResults:
    def test_creates_manifest(self, tmp_path: Path) -> None:
        cfg = IngestConfig(cache_dir=str(tmp_path))
        results = [
            ClipPipelineResult(
                clip_id="c1", n_frames=10, n_accepted=8,
                latent_path="c1.npz", stages_skipped=[], errors=[],
            ),
        ]
        with patch("vllatent.manifest.write_manifest") as mock_write:
            mock_write.return_value = tmp_path / "manifest.json"
            path = update_manifest_from_results(results, cfg)
        assert path.name == "manifest.json"

    def test_skips_errored_results(self, tmp_path: Path) -> None:
        cfg = IngestConfig(cache_dir=str(tmp_path))
        results = [
            ClipPipelineResult(
                clip_id="bad", n_frames=0, n_accepted=0,
                latent_path="", stages_skipped=[], errors=["download failed"],
            ),
        ]
        with patch("vllatent.manifest.write_manifest") as mock_write, \
             patch("vllatent.manifest.build_manifest_wild_video") as mock_build, \
             patch("vllatent.manifest.validate_manifest", return_value=[]):
            mock_build.return_value = {"entries": []}
            mock_write.return_value = tmp_path / "manifest.json"
            update_manifest_from_results(results, cfg)
        call_kwargs = mock_build.call_args
        assert len(call_kwargs.kwargs.get("entries", [])) == 0
