"""Tests for vllatent.ingest.content_filter — CLIP+PySceneDetect content filter (B1.7b).

Pure-tier contract tests: all CLIP and scenedetect calls are mocked.
The real-weight path (CLIP ViT-B/32) is exercised by the existing text-smoke.

TDD: written BEFORE the implementation.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures — synthetic frames + mock CLIP scores
# ---------------------------------------------------------------------------

def _make_frames(n: int, h: int = 224, w: int = 224) -> list[np.ndarray]:
    """Create N synthetic RGB frames."""
    rng = np.random.RandomState(42)
    return [rng.randint(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(n)]


def _make_video_file(tmp_path: Path, n_frames: int = 50, fps: int = 5) -> Path:
    """Create a tiny dummy video file (just needs to exist for path validation)."""
    video = tmp_path / "test_video.mp4"
    video.write_bytes(b"\x00" * 1024)
    return video


# ---------------------------------------------------------------------------
# Shot boundary detection
# ---------------------------------------------------------------------------

class TestShotBoundaryDetection:
    """Shot boundary detection wrapper around PySceneDetect."""

    def test_detect_shots_returns_list_of_boundaries(self) -> None:
        from vllatent.ingest.content_filter import detect_shot_boundaries
        boundaries = detect_shot_boundaries(
            _make_frames(30),
            adaptive_threshold=3.0,
        )
        assert isinstance(boundaries, list)
        for b in boundaries:
            assert isinstance(b, int)

    def test_detect_shots_single_scene_returns_empty(self) -> None:
        """Identical frames => no shot boundary."""
        from vllatent.ingest.content_filter import detect_shot_boundaries
        frame = np.full((64, 64, 3), 128, dtype=np.uint8)
        frames = [frame.copy() for _ in range(20)]
        boundaries = detect_shot_boundaries(frames, adaptive_threshold=3.0)
        assert boundaries == []

    def test_detect_shots_sharp_cut(self) -> None:
        """Abrupt dark-to-bright transition (with noise) should be detected."""
        from vllatent.ingest.content_filter import detect_shot_boundaries
        rng = np.random.RandomState(0)
        dark = [rng.randint(0, 30, (64, 64, 3), dtype=np.uint8) for _ in range(10)]
        bright = [rng.randint(200, 255, (64, 64, 3), dtype=np.uint8) for _ in range(10)]
        frames = dark + bright
        boundaries = detect_shot_boundaries(frames, adaptive_threshold=3.0)
        assert len(boundaries) >= 1
        assert 8 <= boundaries[0] <= 14

    def test_detect_shots_empty_frames_raises(self) -> None:
        from vllatent.ingest.content_filter import detect_shot_boundaries
        with pytest.raises(ValueError, match="frames"):
            detect_shot_boundaries([], adaptive_threshold=3.0)


# ---------------------------------------------------------------------------
# CLIP zero-shot FPV scoring
# ---------------------------------------------------------------------------

class TestClipFpvScoring:
    """CLIP zero-shot classification for FPV content."""

    def test_score_frames_shape(self) -> None:
        """score_frames_fpv returns per-frame float scores."""
        from vllatent.ingest.content_filter import score_frames_fpv

        frames = _make_frames(5)
        with patch("vllatent.ingest.content_filter._get_clip_scorer") as mock_scorer:
            mock_fn = MagicMock(return_value=np.array([0.7, 0.6, 0.8, 0.5, 0.9], dtype=np.float32))
            mock_scorer.return_value = mock_fn
            scores = score_frames_fpv(frames, device="cpu")

        assert isinstance(scores, np.ndarray)
        assert scores.shape == (5,)
        assert scores.dtype == np.float32

    def test_score_frames_range(self) -> None:
        """Scores should be in [0, 1]."""
        from vllatent.ingest.content_filter import score_frames_fpv

        frames = _make_frames(3)
        with patch("vllatent.ingest.content_filter._get_clip_scorer") as mock_scorer:
            mock_fn = MagicMock(return_value=np.array([0.3, 0.8, 0.5], dtype=np.float32))
            mock_scorer.return_value = mock_fn
            scores = score_frames_fpv(frames, device="cpu")

        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_score_frames_empty_raises(self) -> None:
        from vllatent.ingest.content_filter import score_frames_fpv
        with pytest.raises(ValueError, match="frames"):
            score_frames_fpv([], device="cpu")


# ---------------------------------------------------------------------------
# Per-shot majority vote
# ---------------------------------------------------------------------------

class TestShotVoting:
    """Per-shot accept/reject via FPV score majority vote."""

    def test_classify_shots_all_fpv(self) -> None:
        from vllatent.ingest.content_filter import classify_shots

        scores = np.array([0.8, 0.9, 0.7, 0.85, 0.6], dtype=np.float32)
        boundaries: list[int] = []
        result = classify_shots(scores, boundaries, threshold=0.65)
        assert result.n_shots == 1
        assert result.shots[0].is_fpv is True

    def test_classify_shots_mixed(self) -> None:
        from vllatent.ingest.content_filter import classify_shots

        # Shot 1: frames 0-4 (high FPV), Shot 2: frames 5-9 (low FPV)
        scores = np.array(
            [0.8, 0.9, 0.7, 0.85, 0.75,  # shot 1: FPV
             0.1, 0.05, 0.15, 0.1, 0.08],  # shot 2: not FPV
            dtype=np.float32,
        )
        boundaries = [5]
        result = classify_shots(scores, boundaries, threshold=0.65)
        assert result.n_shots == 2
        assert result.shots[0].is_fpv is True
        assert result.shots[1].is_fpv is False

    def test_classify_shots_threshold_boundary(self) -> None:
        """Frames exactly at threshold: majority must be strictly above."""
        from vllatent.ingest.content_filter import classify_shots

        scores = np.array([0.65, 0.65, 0.65], dtype=np.float32)
        result = classify_shots(scores, [], threshold=0.65)
        # >= threshold counts as FPV for the per-frame check
        assert result.shots[0].is_fpv is True


# ---------------------------------------------------------------------------
# Whole-video verdict
# ---------------------------------------------------------------------------

class TestVideoVerdict:
    """ACCEPT / PARTIAL / REJECT whole-video classification."""

    def test_verdict_all_fpv(self) -> None:
        from vllatent.ingest.content_filter import VideoVerdict, classify_shots, video_verdict

        scores = np.ones(20, dtype=np.float32) * 0.8
        classification = classify_shots(scores, [], threshold=0.65)
        verdict = video_verdict(classification)
        assert verdict == VideoVerdict.ACCEPT

    def test_verdict_all_non_fpv(self) -> None:
        from vllatent.ingest.content_filter import VideoVerdict, classify_shots, video_verdict

        scores = np.ones(20, dtype=np.float32) * 0.05
        classification = classify_shots(scores, [], threshold=0.65)
        verdict = video_verdict(classification)
        assert verdict == VideoVerdict.REJECT

    def test_verdict_partial(self) -> None:
        from vllatent.ingest.content_filter import VideoVerdict, classify_shots, video_verdict

        # 50% FPV shots => PARTIAL (between 30% and 60%)
        scores = np.array(
            [0.8] * 10 + [0.05] * 10,
            dtype=np.float32,
        )
        boundaries = [10]
        classification = classify_shots(scores, boundaries, threshold=0.65)
        verdict = video_verdict(classification)
        assert verdict == VideoVerdict.PARTIAL

    def test_verdict_thresholds(self) -> None:
        """Verify the 60% ACCEPT and 30% REJECT thresholds."""
        from vllatent.ingest.content_filter import VideoVerdict, classify_shots, video_verdict

        # 7 FPV shots out of 10 = 70% => ACCEPT
        all_fpv_scores = np.ones(5, dtype=np.float32) * 0.8
        non_fpv_scores = np.ones(5, dtype=np.float32) * 0.05
        boundaries_10 = list(range(5, 50, 5))  # 10 shots of 5 frames each
        scores_70 = np.concatenate([all_fpv_scores] * 7 + [non_fpv_scores] * 3)
        cls_70 = classify_shots(scores_70, boundaries_10[:9], threshold=0.65)
        assert video_verdict(cls_70) == VideoVerdict.ACCEPT

        # 2 FPV shots out of 10 = 20% => REJECT
        scores_20 = np.concatenate([all_fpv_scores] * 2 + [non_fpv_scores] * 8)
        cls_20 = classify_shots(scores_20, boundaries_10[:9], threshold=0.65)
        assert video_verdict(cls_20) == VideoVerdict.REJECT


# ---------------------------------------------------------------------------
# FPV frame mask (for pipeline integration)
# ---------------------------------------------------------------------------

class TestFpvFrameMask:
    """Per-frame is_fpv boolean mask."""

    def test_fpv_mask_shape(self) -> None:
        from vllatent.ingest.content_filter import classify_shots, fpv_frame_mask

        scores = np.ones(20, dtype=np.float32) * 0.8
        cls = classify_shots(scores, [10], threshold=0.65)
        mask = fpv_frame_mask(cls)
        assert mask.shape == (20,)
        assert mask.dtype == np.bool_

    def test_fpv_mask_values(self) -> None:
        from vllatent.ingest.content_filter import classify_shots, fpv_frame_mask

        scores = np.array([0.8] * 5 + [0.05] * 5, dtype=np.float32)
        cls = classify_shots(scores, [5], threshold=0.65)
        mask = fpv_frame_mask(cls)
        assert np.all(mask[:5])      # FPV shot
        assert not np.any(mask[5:])  # non-FPV shot


# ---------------------------------------------------------------------------
# Thumbnail grid (structure only — no PIL rendering in pure test)
# ---------------------------------------------------------------------------

class TestThumbnailGrid:
    """Thumbnail grid data generation (not rendering)."""

    def test_grid_data_structure(self) -> None:
        from vllatent.ingest.content_filter import classify_shots, thumbnail_grid_data

        frames = _make_frames(20)
        scores = np.array([0.8] * 10 + [0.1] * 10, dtype=np.float32)
        cls = classify_shots(scores, [10], threshold=0.65)
        grid = thumbnail_grid_data(frames, cls, max_thumbs=6)
        assert isinstance(grid, list)
        assert len(grid) <= 6
        for entry in grid:
            assert "frame_idx" in entry
            assert "is_fpv" in entry
            assert "score" in entry
            assert "frame" in entry
            assert isinstance(entry["frame"], np.ndarray)


# ---------------------------------------------------------------------------
# FPV range extraction
# ---------------------------------------------------------------------------

class TestExtractFpvRanges:
    """Merging consecutive FPV shots into contiguous frame ranges."""

    def test_all_fpv_merge_single_range(self) -> None:
        from vllatent.ingest.content_filter import ShotInfo, extract_fpv_ranges

        shots = [
            ShotInfo(start=0, end=10, is_fpv=True, mean_score=0.8),
            ShotInfo(start=10, end=20, is_fpv=True, mean_score=0.7),
            ShotInfo(start=20, end=30, is_fpv=True, mean_score=0.9),
        ]
        assert extract_fpv_ranges(shots) == [(0, 30)]

    def test_mixed_fpv_produces_separate_ranges(self) -> None:
        from vllatent.ingest.content_filter import ShotInfo, extract_fpv_ranges

        shots = [
            ShotInfo(start=0, end=10, is_fpv=True, mean_score=0.8),
            ShotInfo(start=10, end=20, is_fpv=True, mean_score=0.7),
            ShotInfo(start=20, end=30, is_fpv=False, mean_score=0.1),
            ShotInfo(start=30, end=40, is_fpv=True, mean_score=0.9),
        ]
        assert extract_fpv_ranges(shots) == [(0, 20), (30, 40)]

    def test_no_fpv_returns_empty(self) -> None:
        from vllatent.ingest.content_filter import ShotInfo, extract_fpv_ranges

        shots = [
            ShotInfo(start=0, end=10, is_fpv=False, mean_score=0.1),
            ShotInfo(start=10, end=20, is_fpv=False, mean_score=0.05),
        ]
        assert extract_fpv_ranges(shots) == []

    def test_empty_shots_returns_empty(self) -> None:
        from vllatent.ingest.content_filter import extract_fpv_ranges

        assert extract_fpv_ranges([]) == []

    def test_single_fpv_shot(self) -> None:
        from vllatent.ingest.content_filter import ShotInfo, extract_fpv_ranges

        shots = [ShotInfo(start=5, end=15, is_fpv=True, mean_score=0.75)]
        assert extract_fpv_ranges(shots) == [(5, 15)]

    def test_non_consecutive_fpv_with_gap(self) -> None:
        from vllatent.ingest.content_filter import ShotInfo, extract_fpv_ranges

        shots = [
            ShotInfo(start=0, end=10, is_fpv=True, mean_score=0.8),
            ShotInfo(start=10, end=20, is_fpv=False, mean_score=0.1),
            ShotInfo(start=20, end=30, is_fpv=False, mean_score=0.15),
            ShotInfo(start=30, end=40, is_fpv=True, mean_score=0.9),
        ]
        assert extract_fpv_ranges(shots) == [(0, 10), (30, 40)]


# ---------------------------------------------------------------------------
# Path-based scoring (memory-efficient, no stride sampling)
# ---------------------------------------------------------------------------

class TestScoreFramesFromPaths:
    """score_frames_from_paths: batched CLIP scoring from file paths."""

    def test_shape_matches_input_count(self, tmp_path: Path) -> None:
        from vllatent.ingest.content_filter import score_frames_from_paths

        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()
        for i in range(10):
            _save_dummy_frame(frame_dir / f"{i:06d}.jpg")

        paths = sorted(frame_dir.glob("*.jpg"))
        with patch("vllatent.ingest.content_filter._get_clip_scorer") as mock_scorer:
            mock_fn = MagicMock(side_effect=lambda batch: np.ones(len(batch), dtype=np.float32) * 0.7)
            mock_scorer.return_value = mock_fn
            scores = score_frames_from_paths(paths, device="cpu", batch_size=4)

        assert isinstance(scores, np.ndarray)
        assert scores.shape == (10,)
        assert scores.dtype == np.float32

    def test_batching_respects_batch_size(self, tmp_path: Path) -> None:
        from vllatent.ingest.content_filter import score_frames_from_paths

        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()
        for i in range(7):
            _save_dummy_frame(frame_dir / f"{i:06d}.jpg")

        paths = sorted(frame_dir.glob("*.jpg"))
        batch_sizes_seen: list[int] = []
        with patch("vllatent.ingest.content_filter._get_clip_scorer") as mock_scorer:
            def _track_batch(batch: list) -> np.ndarray:
                batch_sizes_seen.append(len(batch))
                return np.ones(len(batch), dtype=np.float32) * 0.5
            mock_scorer.return_value = _track_batch
            score_frames_from_paths(paths, device="cpu", batch_size=3)

        assert batch_sizes_seen == [3, 3, 1]

    def test_empty_paths_raises(self) -> None:
        from vllatent.ingest.content_filter import score_frames_from_paths
        with pytest.raises(ValueError, match="frame_paths"):
            score_frames_from_paths([], device="cpu")


class TestDetectShotBoundariesFromPaths:
    """detect_shot_boundaries_from_paths: one-at-a-time loading from paths."""

    def test_processes_all_frames(self, tmp_path: Path) -> None:
        from vllatent.ingest.content_filter import detect_shot_boundaries_from_paths

        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()
        for i in range(15):
            _save_dummy_frame(frame_dir / f"{i:06d}.jpg")

        paths = sorted(frame_dir.glob("*.jpg"))
        boundaries = detect_shot_boundaries_from_paths(paths)
        assert isinstance(boundaries, list)
        for b in boundaries:
            assert isinstance(b, int)

    def test_empty_paths_raises(self) -> None:
        from vllatent.ingest.content_filter import detect_shot_boundaries_from_paths
        with pytest.raises(ValueError, match="frame_paths"):
            detect_shot_boundaries_from_paths([])


class TestFilterVideoFromPaths:
    """filter_video_from_paths: full pipeline on file paths, every frame scored."""

    def test_returns_filter_result_covering_all_frames(self, tmp_path: Path) -> None:
        from vllatent.ingest.content_filter import FilterResult, VideoVerdict, filter_video_from_paths

        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()
        for i in range(20):
            _save_dummy_frame(frame_dir / f"{i:06d}.jpg")

        paths = sorted(frame_dir.glob("*.jpg"))
        high_motion = np.ones(20, dtype=np.float32) * 30.0
        with patch("vllatent.ingest.content_filter._get_clip_scorer") as mock_scorer, \
             patch("vllatent.ingest.content_filter.compute_motion_scores", return_value=high_motion):
            mock_fn = MagicMock(side_effect=lambda batch: np.ones(len(batch), dtype=np.float32) * 0.8)
            mock_scorer.return_value = mock_fn
            result = filter_video_from_paths(paths, device="cpu")

        assert isinstance(result, FilterResult)
        assert result.verdict == VideoVerdict.ACCEPT
        assert result.n_frames == 20
        assert result.fpv_mask.shape == (20,)
        assert result.n_fpv_frames == 20

    def test_motion_rejects_static_frames(self, tmp_path: Path) -> None:
        """Frames with high CLIP but zero motion should be rejected."""
        from vllatent.ingest.content_filter import filter_video_from_paths

        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()
        for i in range(10):
            _save_dummy_frame(frame_dir / f"{i:06d}.jpg")

        paths = sorted(frame_dir.glob("*.jpg"))
        zero_motion = np.zeros(10, dtype=np.float32)
        with patch("vllatent.ingest.content_filter._get_clip_scorer") as mock_scorer, \
             patch("vllatent.ingest.content_filter.compute_motion_scores", return_value=zero_motion):
            mock_fn = MagicMock(side_effect=lambda batch: np.ones(len(batch), dtype=np.float32) * 0.9)
            mock_scorer.return_value = mock_fn
            result = filter_video_from_paths(paths, device="cpu")

        assert result.n_fpv_frames == 0

    def test_mask_covers_every_frame(self, tmp_path: Path) -> None:
        """The mask length must equal the number of input paths — no stride gaps."""
        from vllatent.ingest.content_filter import filter_video_from_paths

        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()
        n = 37
        for i in range(n):
            _save_dummy_frame(frame_dir / f"{i:06d}.jpg")

        paths = sorted(frame_dir.glob("*.jpg"))
        clip_scores = np.concatenate([
            np.ones(20, dtype=np.float32) * 0.8,
            np.ones(17, dtype=np.float32) * 0.1,
        ])
        high_motion = np.ones(n, dtype=np.float32) * 30.0
        call_count = [0]
        with patch("vllatent.ingest.content_filter._get_clip_scorer") as mock_scorer, \
             patch("vllatent.ingest.content_filter.compute_motion_scores", return_value=high_motion):
            def _score(batch: list) -> np.ndarray:
                start = call_count[0]
                call_count[0] += len(batch)
                return clip_scores[start:start + len(batch)]
            mock_scorer.return_value = _score
            result = filter_video_from_paths(paths, device="cpu")

        assert result.fpv_mask.shape == (n,)
        assert result.per_frame_scores.shape == (n,)
        assert np.all(result.fpv_mask[:20])

    def test_empty_paths_raises(self) -> None:
        from vllatent.ingest.content_filter import filter_video_from_paths
        with pytest.raises(ValueError, match="frame_paths"):
            filter_video_from_paths([], device="cpu")


def _save_dummy_frame(path: Path, h: int = 64, w: int = 64) -> None:
    """Write a tiny JPEG to disk for path-based tests."""
    from PIL import Image
    img = Image.fromarray(np.random.RandomState(42).randint(0, 256, (h, w, 3), dtype=np.uint8))
    img.save(path)


# ---------------------------------------------------------------------------
# Module purity (no torch at import time)
# ---------------------------------------------------------------------------

class TestImportPurity:
    """content_filter module imports without torch/transformers."""

    def test_no_heavy_imports(self) -> None:
        import ast
        import importlib

        spec = importlib.util.find_spec("vllatent.ingest.content_filter")
        assert spec is not None and spec.origin is not None
        source = Path(spec.origin).read_text()
        tree = ast.parse(source)
        top_imports: set[str] = set()
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_imports.add(node.module.split(".")[0])
        heavy = {"torch", "transformers", "timm", "scenedetect"}
        leaked = top_imports & heavy
        assert not leaked, f"Module-level heavy imports found: {leaked}"

    def test_module_imports_cleanly(self) -> None:
        import sys
        mods_before = set(sys.modules.keys())
        import vllatent.ingest.content_filter  # noqa: F401
        mods_after = set(sys.modules.keys())
        new = mods_after - mods_before
        heavy_loaded = {m for m in new if m.startswith(("torch", "transformers", "timm", "scenedetect"))}
        assert not heavy_loaded, f"Heavy modules loaded at import: {heavy_loaded}"


# ---------------------------------------------------------------------------
# Integration: full filter_video pipeline (all internals mocked)
# ---------------------------------------------------------------------------

class TestFilterVideoIntegration:
    """End-to-end filter_video orchestration."""

    def test_filter_video_returns_result(self) -> None:
        from vllatent.ingest.content_filter import FilterResult, VideoVerdict, filter_video

        frames = _make_frames(20)
        with patch("vllatent.ingest.content_filter._get_clip_scorer") as mock_scorer:
            mock_fn = MagicMock(return_value=np.ones(20, dtype=np.float32) * 0.8)
            mock_scorer.return_value = mock_fn
            result = filter_video(frames, device="cpu")

        assert isinstance(result, FilterResult)
        assert result.verdict == VideoVerdict.ACCEPT
        assert result.n_frames == 20
        assert isinstance(result.fpv_mask, np.ndarray)
        assert result.fpv_mask.shape == (20,)
        assert isinstance(result.shot_boundaries, list)

    def test_filter_video_reject(self) -> None:
        from vllatent.ingest.content_filter import VideoVerdict, filter_video

        frames = _make_frames(20)
        with patch("vllatent.ingest.content_filter._get_clip_scorer") as mock_scorer:
            mock_fn = MagicMock(return_value=np.ones(20, dtype=np.float32) * 0.05)
            mock_scorer.return_value = mock_fn
            result = filter_video(frames, device="cpu")

        assert result.verdict == VideoVerdict.REJECT
        assert not np.any(result.fpv_mask)
