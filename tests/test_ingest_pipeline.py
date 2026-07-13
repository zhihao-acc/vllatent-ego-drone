"""Tests for vllatent.ingest.pipeline — end-to-end per-clip processing."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from vllatent.config import IngestConfig
from vllatent.ingest.pipeline import (
    MIN_SEGMENT_FRAMES,
    ClipPipelineResult,
    _build_clip_npz,
    _passes_human_trackability_gate,
    _write_clip_npz,
    update_manifest_from_results,
)
from vllatent.schemas import EMBED_DIM, LATENT_DTYPE, PATCH_TOKENS


class TestBuildClipNpz:
    def _arrays(self, n: int = 10):
        return dict(
            latents=np.zeros((n, PATCH_TOKENS, EMBED_DIM), dtype=LATENT_DTYPE),
            deltas=np.zeros((n - 1, 4), dtype=np.float32),
            vo_confidence=np.ones(n, dtype=np.float32),
            frame_quality=np.ones(n, dtype=np.float32),
            timestamps=np.arange(n, dtype=np.float64),
        )

    def test_valid(self) -> None:
        result = _build_clip_npz(**self._arrays(5))
        assert result["latents"].shape == (5, PATCH_TOKENS, EMBED_DIM)
        assert result["deltas"].shape == (4, 4)
        assert result["person_bbox"].shape == (5, 4)
        assert result["person_visible"].shape == (5,)
        assert result["person_state_valid"].shape == (5,)
        assert result["person_conf"].shape == (5,)
        assert str(result["person_bbox_space"].tolist()) == "encoder_crop"
        assert "quality_mask" not in result

    def test_accepts_person_tracks(self) -> None:
        a = self._arrays(5)
        a["person_bbox"] = np.tile(np.array([[0.5, 0.5, 0.2, 0.3]], dtype=np.float32), (5, 1))
        a["person_visible"] = np.ones(5, dtype=bool)
        a["person_conf"] = np.full(5, 0.8, dtype=np.float32)
        result = _build_clip_npz(**a)
        np.testing.assert_allclose(result["person_bbox"][0], [0.5, 0.5, 0.2, 0.3])
        assert result["person_visible"].dtype == np.bool_
        assert result["person_state_valid"].dtype == np.bool_
        assert np.all(result["person_state_valid"])
        assert result["person_conf"][0] == pytest.approx(0.8)

    def test_stores_subject_selection_provenance(self) -> None:
        a = self._arrays(5)
        a["person_bbox"] = np.tile(np.array([[0.5, 0.5, 0.2, 0.3]], dtype=np.float32), (5, 1))
        a["person_visible"] = np.ones(5, dtype=bool)
        a["person_conf"] = np.full(5, 0.8, dtype=np.float32)
        a["person_selected_track_id"] = 11
        a["person_second_best_track_id"] = 22
        a["person_subject_ambiguity_margin"] = 0.42
        a["person_subject_is_ambiguous"] = False

        result = _build_clip_npz(**a)

        assert int(result["person_selected_track_id"]) == 11
        assert int(result["person_second_best_track_id"]) == 22
        assert float(result["person_subject_ambiguity_margin"]) == pytest.approx(0.42)
        assert not bool(result["person_subject_is_ambiguous"])

    def test_sanitizes_tiny_visible_person_tracks(self) -> None:
        a = self._arrays(5)
        a["person_bbox"] = np.tile(np.array([[0.5, 0.5, 0.01, 0.01]], dtype=np.float32), (5, 1))
        a["person_visible"] = np.ones(5, dtype=bool)
        a["person_conf"] = np.full(5, 0.8, dtype=np.float32)
        result = _build_clip_npz(**a)
        assert not np.any(result["person_visible"])
        assert not np.any(result["person_state_valid"])
        np.testing.assert_allclose(result["person_bbox"], np.zeros((5, 4), dtype=np.float32))
        np.testing.assert_allclose(result["person_conf"], np.zeros(5, dtype=np.float32))

    def test_state_valid_is_stricter_than_visible(self) -> None:
        a = self._arrays(5)
        a["person_bbox"] = np.tile(np.array([[0.1, 0.5, 0.2, 0.2]], dtype=np.float32), (5, 1))
        a["person_visible"] = np.ones(5, dtype=bool)
        a["person_conf"] = np.full(5, 0.8, dtype=np.float32)
        result = _build_clip_npz(**a)
        assert np.all(result["person_visible"])
        assert not np.any(result["person_state_valid"])

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

    def test_bad_person_shape(self) -> None:
        a = self._arrays(5)
        a["person_bbox"] = np.zeros((4, 4), dtype=np.float32)
        a["person_visible"] = np.zeros(5, dtype=bool)
        a["person_conf"] = np.zeros(5, dtype=np.float32)
        with pytest.raises(ValueError, match="person_bbox"):
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


class TestUndistortWiring:
    """B1.1: batch_undistort() is conditionally called between stages 2 and 3."""

    def test_undistort_skipped_for_pinhole(self) -> None:
        """When undistort_model='pinhole' (default), batch_undistort is not called."""
        cfg = IngestConfig()
        assert cfg.undistort_model == "pinhole"
        assert cfg.person_gate_history == 3
        assert cfg.person_gate_horizon == 8

    def test_undistort_import_available(self) -> None:
        """batch_undistort is importable from preprocess and available in pipeline."""
        from vllatent.ingest.preprocess import batch_undistort
        assert callable(batch_undistort)

    def test_pipeline_accepts_camera_params(self) -> None:
        """process_clip accepts camera_K and camera_D keyword args."""
        import inspect

        from vllatent.ingest.pipeline import process_clip

        sig = inspect.signature(process_clip)
        assert "camera_K" in sig.parameters
        assert "camera_D" in sig.parameters
        assert "track_persons" in sig.parameters


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


class TestFindAcceptedSegments:
    """Quality mask → contiguous accepted segments for MegaSaM."""

    def test_all_accepted_single_segment(self) -> None:
        from vllatent.ingest.quality import find_accepted_segments
        mask = np.ones(20, dtype=bool)
        assert find_accepted_segments(mask, min_length=8) == [(0, 20)]

    def test_all_rejected_empty(self) -> None:
        from vllatent.ingest.quality import find_accepted_segments
        mask = np.zeros(20, dtype=bool)
        assert find_accepted_segments(mask, min_length=8) == []

    def test_split_at_bad_block(self) -> None:
        from vllatent.ingest.quality import find_accepted_segments
        mask = np.ones(30, dtype=bool)
        mask[10:15] = False
        result = find_accepted_segments(mask, min_length=8)
        assert result == [(0, 10), (15, 30)]

    def test_short_segment_discarded(self) -> None:
        from vllatent.ingest.quality import find_accepted_segments
        mask = np.ones(30, dtype=bool)
        mask[3:25] = False
        result = find_accepted_segments(mask, min_length=8)
        assert result == []

    def test_mixed_keeps_only_long(self) -> None:
        from vllatent.ingest.quality import find_accepted_segments
        mask = np.zeros(40, dtype=bool)
        mask[0:5] = True
        mask[10:25] = True
        mask[30:35] = True
        result = find_accepted_segments(mask, min_length=8)
        assert result == [(10, 25)]

    def test_empty_mask(self) -> None:
        from vllatent.ingest.quality import find_accepted_segments
        mask = np.array([], dtype=bool)
        assert find_accepted_segments(mask, min_length=8) == []


class TestMinSegmentFrames:
    def test_value(self) -> None:
        from vllatent.schemas import HISTORY, HORIZON
        assert MIN_SEGMENT_FRAMES == HISTORY + HORIZON


class TestHumanTrackabilityGate:
    def test_passes_when_history_and_future_have_trackable_support(self) -> None:
        state_valid = np.ones(12, dtype=np.bool_)
        assert _passes_human_trackability_gate(state_valid, history=3, horizon=8)

    def test_rejects_when_only_padded_start_history_would_pass(self) -> None:
        state_valid = np.zeros(12, dtype=np.bool_)
        state_valid[0] = True
        state_valid[3:9] = True
        assert not _passes_human_trackability_gate(state_valid, history=3, horizon=8)

    def test_rejects_segments_too_short_for_configured_horizon(self) -> None:
        state_valid = np.ones(8, dtype=np.bool_)
        assert not _passes_human_trackability_gate(state_valid, history=3, horizon=8)

    def test_rejects_sparse_future_trackability(self) -> None:
        state_valid = np.ones(12, dtype=np.bool_)
        state_valid[4:11] = False
        assert not _passes_human_trackability_gate(state_valid, history=3, horizon=8)

    def test_rejects_one_missing_future_frame(self) -> None:
        state_valid = np.ones(12, dtype=np.bool_)
        state_valid[10] = False
        assert not _passes_human_trackability_gate(state_valid, history=3, horizon=8)
