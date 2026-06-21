"""Tests for vllatent.ingest.content_filter — motion + YOLO-World content filter (B1.7c).

Pure-tier contract tests: all YOLO and scenedetect calls are mocked.
The real-weight path is exercised by separate integration/smoke tests.

TDD: written BEFORE the implementation.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures — synthetic frames + mock helpers
# ---------------------------------------------------------------------------

def _make_frames(n: int, h: int = 224, w: int = 224) -> list[np.ndarray]:
    """Create N synthetic RGB frames."""
    rng = np.random.RandomState(42)
    return [rng.randint(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(n)]


def _save_dummy_frame(path: Path, h: int = 64, w: int = 64) -> None:
    """Write a tiny JPEG to disk for path-based tests."""
    from PIL import Image
    img = Image.fromarray(np.random.RandomState(42).randint(0, 256, (h, w, 3), dtype=np.uint8))
    img.save(path)


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
# YOLO-World object detection
# ---------------------------------------------------------------------------

class TestYoloObjectDetection:
    """YOLO-World open-vocabulary object detection for rejected objects."""

    def test_detect_objects_returns_per_frame_bool(self) -> None:
        """detect_rejected_objects returns (N,) bool array."""
        from vllatent.ingest.content_filter import detect_rejected_objects

        frames = _make_frames(5)
        with patch("vllatent.ingest.content_filter._get_yolo_detector") as mock_det:
            mock_fn = MagicMock(return_value=np.array([False, False, True, False, False]))
            mock_det.return_value = mock_fn
            rejected = detect_rejected_objects(frames, device="cpu")

        assert isinstance(rejected, np.ndarray)
        assert rejected.shape == (5,)
        assert rejected.dtype == np.bool_
        assert rejected[2] is np.True_

    def test_detect_objects_empty_raises(self) -> None:
        from vllatent.ingest.content_filter import detect_rejected_objects
        with pytest.raises(ValueError, match="frames"):
            detect_rejected_objects([], device="cpu")

    def test_detect_objects_from_paths(self, tmp_path: Path) -> None:
        """detect_rejected_objects_from_paths works on file paths."""
        from vllatent.ingest.content_filter import detect_rejected_objects_from_paths

        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()
        for i in range(6):
            _save_dummy_frame(frame_dir / f"{i:06d}.jpg")

        paths = sorted(frame_dir.glob("*.jpg"))
        full_result = np.array([False, True, False, False, False, False])
        call_count = [0]
        with patch("vllatent.ingest.content_filter._get_yolo_detector") as mock_det:
            def _batch_detect(batch: list) -> np.ndarray:
                start = call_count[0]
                call_count[0] += len(batch)
                return full_result[start:start + len(batch)]
            mock_det.return_value = _batch_detect
            rejected = detect_rejected_objects_from_paths(paths, device="cpu", batch_size=4)

        assert rejected.shape == (6,)
        assert rejected[1] is np.True_

    def test_detect_objects_from_paths_empty_raises(self) -> None:
        from vllatent.ingest.content_filter import detect_rejected_objects_from_paths
        with pytest.raises(ValueError, match="frame_paths"):
            detect_rejected_objects_from_paths([], device="cpu")

    def test_rejected_classes_are_configurable(self) -> None:
        """The REJECTED_CLASSES list should be accessible and non-empty."""
        from vllatent.ingest.content_filter import REJECTED_CLASSES
        assert isinstance(REJECTED_CLASSES, (list, tuple))
        assert len(REJECTED_CLASSES) > 0
        assert "drone" in REJECTED_CLASSES

    def test_rejected_classes_include_drone_parts(self) -> None:
        """REJECTED_CLASSES must cover drone body parts, not just the whole drone."""
        from vllatent.ingest.content_filter import REJECTED_CLASSES
        drone_parts = {"rotor", "propeller", "gimbal", "landing gear"}
        found = drone_parts & set(REJECTED_CLASSES)
        assert found == drone_parts, f"Missing drone parts: {drone_parts - found}"


# ---------------------------------------------------------------------------
# Minimum segment filter (continuity)
# ---------------------------------------------------------------------------

class TestFilterShortSegments:
    """filter_short_segments: discard accepted runs shorter than min_length."""

    def test_short_run_removed(self) -> None:
        from vllatent.ingest.content_filter import filter_short_segments
        mask = np.array([True, True, True, False, True, True, False, True, True, True], dtype=np.bool_)
        # min_length=3: the 2-frame run at [4,5] is too short
        result = filter_short_segments(mask, min_length=3)
        assert result.tolist() == [True, True, True, False, False, False, False, True, True, True]

    def test_all_long_runs_kept(self) -> None:
        from vllatent.ingest.content_filter import filter_short_segments
        mask = np.array([True] * 5 + [False] * 2 + [True] * 5, dtype=np.bool_)
        result = filter_short_segments(mask, min_length=3)
        np.testing.assert_array_equal(result, mask)

    def test_all_short_runs_removed(self) -> None:
        from vllatent.ingest.content_filter import filter_short_segments
        # alternating: T,F,T,F,T,F — each True run is length 1
        mask = np.array([True, False, True, False, True, False], dtype=np.bool_)
        result = filter_short_segments(mask, min_length=2)
        assert not np.any(result)

    def test_min_length_1_keeps_everything(self) -> None:
        from vllatent.ingest.content_filter import filter_short_segments
        mask = np.array([True, False, True], dtype=np.bool_)
        result = filter_short_segments(mask, min_length=1)
        np.testing.assert_array_equal(result, mask)

    def test_empty_mask(self) -> None:
        from vllatent.ingest.content_filter import filter_short_segments
        mask = np.array([], dtype=np.bool_)
        result = filter_short_segments(mask, min_length=5)
        assert len(result) == 0

    def test_immutable(self) -> None:
        """Original mask must not be mutated."""
        from vllatent.ingest.content_filter import filter_short_segments
        mask = np.array([True, False, True], dtype=np.bool_)
        original = mask.copy()
        filter_short_segments(mask, min_length=5)
        np.testing.assert_array_equal(mask, original)


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

        all_fpv_scores = np.ones(5, dtype=np.float32) * 0.8
        non_fpv_scores = np.ones(5, dtype=np.float32) * 0.05
        boundaries_10 = list(range(5, 50, 5))

        scores_70 = np.concatenate([all_fpv_scores] * 7 + [non_fpv_scores] * 3)
        cls_70 = classify_shots(scores_70, boundaries_10[:9], threshold=0.65)
        assert video_verdict(cls_70) == VideoVerdict.ACCEPT

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
        assert np.all(mask[:5])
        assert not np.any(mask[5:])


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
# Path-based detection
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Full pipeline: filter_video_from_paths (motion + YOLO)
# ---------------------------------------------------------------------------

class TestFilterVideoFromPaths:
    """filter_video_from_paths: motion + YOLO object detection pipeline."""

    def test_returns_filter_result_all_fpv(self, tmp_path: Path) -> None:
        """High motion + no rejected objects → all FPV."""
        from vllatent.ingest.content_filter import FilterResult, VideoVerdict, filter_video_from_paths

        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()
        for i in range(20):
            _save_dummy_frame(frame_dir / f"{i:06d}.jpg")

        paths = sorted(frame_dir.glob("*.jpg"))
        high_motion = np.ones(20, dtype=np.float32) * 30.0
        no_rejected = np.zeros(20, dtype=np.bool_)
        with patch("vllatent.ingest.content_filter.compute_motion_scores", return_value=high_motion), \
             patch("vllatent.ingest.content_filter.detect_rejected_objects_from_paths", return_value=no_rejected):
            result = filter_video_from_paths(paths, device="cpu")

        assert isinstance(result, FilterResult)
        assert result.verdict == VideoVerdict.ACCEPT
        assert result.n_frames == 20
        assert result.fpv_mask.shape == (20,)
        assert result.n_fpv_frames == 20

    def test_motion_rejects_static_frames(self, tmp_path: Path) -> None:
        """Frames with zero motion should be rejected even with no objects detected."""
        from vllatent.ingest.content_filter import filter_video_from_paths

        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()
        for i in range(10):
            _save_dummy_frame(frame_dir / f"{i:06d}.jpg")

        paths = sorted(frame_dir.glob("*.jpg"))
        zero_motion = np.zeros(10, dtype=np.float32)
        no_rejected = np.zeros(10, dtype=np.bool_)
        with patch("vllatent.ingest.content_filter.compute_motion_scores", return_value=zero_motion), \
             patch("vllatent.ingest.content_filter.detect_rejected_objects_from_paths", return_value=no_rejected):
            result = filter_video_from_paths(paths, device="cpu")

        assert result.n_fpv_frames == 0

    def test_yolo_rejects_frames_with_objects(self, tmp_path: Path) -> None:
        """Frames with rejected objects should be rejected even with high motion."""
        from vllatent.ingest.content_filter import filter_video_from_paths

        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()
        for i in range(10):
            _save_dummy_frame(frame_dir / f"{i:06d}.jpg")

        paths = sorted(frame_dir.glob("*.jpg"))
        high_motion = np.ones(10, dtype=np.float32) * 30.0
        # Frames 3,4,5 have a drone visible; disable min segment to test YOLO in isolation
        rejected = np.array([False, False, False, True, True, True, False, False, False, False])
        with patch("vllatent.ingest.content_filter.compute_motion_scores", return_value=high_motion), \
             patch("vllatent.ingest.content_filter.detect_rejected_objects_from_paths", return_value=rejected):
            result = filter_video_from_paths(paths, device="cpu", min_segment_frames=1)

        assert result.n_fpv_frames == 7
        assert not result.fpv_mask[3]
        assert not result.fpv_mask[4]
        assert not result.fpv_mask[5]
        assert result.fpv_mask[0]
        assert result.fpv_mask[6]

    def test_short_segments_discarded(self, tmp_path: Path) -> None:
        """Accepted runs shorter than min_segment_frames are discarded."""
        from vllatent.ingest.content_filter import filter_video_from_paths

        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()
        for i in range(20):
            _save_dummy_frame(frame_dir / f"{i:06d}.jpg")

        paths = sorted(frame_dir.glob("*.jpg"))
        high_motion = np.ones(20, dtype=np.float32) * 30.0
        # Drone at frames 3-5 splits into [0-2] (3 frames) and [6-19] (14 frames)
        rejected = np.zeros(20, dtype=np.bool_)
        rejected[3:6] = True
        with patch("vllatent.ingest.content_filter.compute_motion_scores", return_value=high_motion), \
             patch("vllatent.ingest.content_filter.detect_rejected_objects_from_paths", return_value=rejected):
            result = filter_video_from_paths(paths, device="cpu", min_segment_frames=5)

        # [0-2] is only 3 frames < 5 minimum → discarded
        assert not result.fpv_mask[0]
        assert not result.fpv_mask[2]
        # [6-19] is 14 frames >= 5 → kept
        assert result.fpv_mask[6]
        assert result.fpv_mask[19]
        assert result.n_fpv_frames == 14

    def test_mask_covers_every_frame(self, tmp_path: Path) -> None:
        """The mask length must equal the number of input paths — no stride gaps."""
        from vllatent.ingest.content_filter import filter_video_from_paths

        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()
        n = 37
        for i in range(n):
            _save_dummy_frame(frame_dir / f"{i:06d}.jpg")

        paths = sorted(frame_dir.glob("*.jpg"))
        high_motion = np.ones(n, dtype=np.float32) * 30.0
        no_rejected = np.zeros(n, dtype=np.bool_)
        with patch("vllatent.ingest.content_filter.compute_motion_scores", return_value=high_motion), \
             patch("vllatent.ingest.content_filter.detect_rejected_objects_from_paths", return_value=no_rejected):
            result = filter_video_from_paths(paths, device="cpu")

        assert result.fpv_mask.shape == (n,)
        assert result.per_frame_scores.shape == (n,)

    def test_empty_paths_raises(self) -> None:
        from vllatent.ingest.content_filter import filter_video_from_paths
        with pytest.raises(ValueError, match="frame_paths"):
            filter_video_from_paths([], device="cpu")


# ---------------------------------------------------------------------------
# Module purity (no torch at import time)
# ---------------------------------------------------------------------------

class TestImportPurity:
    """content_filter module imports without torch/transformers/ultralytics."""

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
        heavy = {"torch", "transformers", "timm", "scenedetect", "ultralytics"}
        leaked = top_imports & heavy
        assert not leaked, f"Module-level heavy imports found: {leaked}"

    def test_module_imports_cleanly(self) -> None:
        import sys
        mods_before = set(sys.modules.keys())
        import vllatent.ingest.content_filter  # noqa: F401
        mods_after = set(sys.modules.keys())
        new = mods_after - mods_before
        heavy_loaded = {m for m in new if m.startswith(("torch", "transformers", "timm", "scenedetect", "ultralytics"))}
        assert not heavy_loaded, f"Heavy modules loaded at import: {heavy_loaded}"


# ---------------------------------------------------------------------------
# Integration: full filter_video pipeline (all internals mocked)
# ---------------------------------------------------------------------------

class TestFilterVideoIntegration:
    """End-to-end filter_video orchestration."""

    def test_filter_video_returns_result(self) -> None:
        from vllatent.ingest.content_filter import FilterResult, VideoVerdict, filter_video

        frames = _make_frames(20)
        high_motion = np.ones(20, dtype=np.float32) * 30.0
        no_rejected = np.zeros(20, dtype=np.bool_)
        with patch("vllatent.ingest.content_filter._compute_motion_from_arrays", return_value=high_motion), \
             patch("vllatent.ingest.content_filter.detect_rejected_objects", return_value=no_rejected):
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
        zero_motion = np.zeros(20, dtype=np.float32)
        no_rejected = np.zeros(20, dtype=np.bool_)
        with patch("vllatent.ingest.content_filter._compute_motion_from_arrays", return_value=zero_motion), \
             patch("vllatent.ingest.content_filter.detect_rejected_objects", return_value=no_rejected):
            result = filter_video(frames, device="cpu")

        assert result.verdict == VideoVerdict.REJECT
        assert not np.any(result.fpv_mask)
