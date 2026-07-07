"""Tests for B3 person tracking contracts and data screens."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vllatent.ingest.person_tracking import (
    PERSON_BBOX_DIM,
    TrackedDetection,
    accel_outlier_flags_from_deltas,
    duplicate_frame_runs_from_latents,
    empty_person_tracks,
    person_state_from_bbox,
    raw_frame_cxcywh_to_encoder_crop,
    screen_cache_dir,
    screen_clip_arrays,
    select_subject_track,
    time_remap_flags_from_deltas,
    validate_person_track_arrays,
    xyxy_to_encoder_crop_cxcywh,
)


def test_empty_person_tracks_shapes() -> None:
    tracks = empty_person_tracks(5)
    assert tracks.person_bbox.shape == (5, PERSON_BBOX_DIM)
    assert tracks.person_visible.shape == (5,)
    assert tracks.person_conf.shape == (5,)
    assert not np.any(tracks.person_visible)


def test_validate_person_track_arrays_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="person_bbox"):
        validate_person_track_arrays(
            n_frames=5,
            person_bbox=np.zeros((4, 4), dtype=np.float32),
            person_visible=np.zeros(5, dtype=bool),
            person_conf=np.zeros(5, dtype=np.float32),
        )


def test_select_subject_prefers_longest_track() -> None:
    detections = [
        TrackedDetection(0, 1, np.array([0, 0, 20, 20], dtype=np.float32), 0.5),
        TrackedDetection(1, 1, np.array([0, 0, 20, 20], dtype=np.float32), 0.5),
        TrackedDetection(0, 2, np.array([40, 40, 60, 60], dtype=np.float32), 0.9),
    ]
    result = select_subject_track(detections, n_frames=3, image_hw=(100, 100))
    assert result.person_visible.tolist() == [True, True, False]
    assert result.person_conf[0] == pytest.approx(0.5)


def test_select_subject_tie_breaks_by_centrality() -> None:
    detections = [
        TrackedDetection(0, 1, np.array([0, 0, 20, 20], dtype=np.float32), 0.5),
        TrackedDetection(1, 1, np.array([0, 0, 20, 20], dtype=np.float32), 0.5),
        TrackedDetection(0, 2, np.array([40, 40, 60, 60], dtype=np.float32), 0.9),
        TrackedDetection(1, 2, np.array([40, 40, 60, 60], dtype=np.float32), 0.9),
    ]
    result = select_subject_track(detections, n_frames=3, image_hw=(100, 100))
    np.testing.assert_allclose(result.person_bbox[0], [0.5, 0.5, 0.2, 0.2], atol=1e-6)
    assert result.person_conf[0] == pytest.approx(0.9)


def test_encoder_crop_bbox_conversion_for_wide_frame() -> None:
    # 1280x720 crops to x=[280,1000], y=[0,720] before DINO resize.
    bbox = xyxy_to_encoder_crop_cxcywh(np.array([280, 180, 1000, 540], dtype=np.float32), (720, 1280))
    np.testing.assert_allclose(bbox, [0.5, 0.5, 1.0, 0.5], atol=1e-6)
    raw = np.array([[0.5, 0.5, 720 / 1280, 0.5]], dtype=np.float32)
    converted = raw_frame_cxcywh_to_encoder_crop(raw, (720, 1280))
    np.testing.assert_allclose(converted[0], [0.5, 0.5, 1.0, 0.5], atol=1e-6)


def test_person_state_masks_invisible_rows() -> None:
    bbox = np.array([[0.5, 0.4, 0.2, 0.25], [0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    visible = np.array([True, False])
    state = person_state_from_bbox(bbox, visible)
    np.testing.assert_allclose(state[0, :2], [0.5, 0.4], atol=1e-6)
    assert state[0, 2] == pytest.approx(np.log(0.25))
    np.testing.assert_allclose(state[1], [0.0, 0.0, 0.0, 0.0], atol=1e-6)


def test_duplicate_frame_runs_from_latents_flags_repeats() -> None:
    latents = np.array([
        [[1.0, 0.0]],
        [[1.0, 0.0]],
        [[0.0, 1.0]],
    ], dtype=np.float32)
    flags = duplicate_frame_runs_from_latents(latents)
    assert flags.tolist() == [False, True, False]


def test_time_remap_flags_from_deltas_flags_speed_jump() -> None:
    deltas = np.ones((8, 4), dtype=np.float32)
    deltas[:, 1:] = 0.0
    deltas[4:, 0] = 4.0
    flags = time_remap_flags_from_deltas(deltas, window=4, jump_ratio=1.5)
    assert np.any(flags[4:])


def test_accel_outlier_flags_from_deltas_flags_spike() -> None:
    deltas = np.zeros((8, 4), dtype=np.float32)
    deltas[:, 0] = 1.0
    deltas[4, 0] = 20.0
    flags = accel_outlier_flags_from_deltas(deltas)
    assert np.any(flags)


def test_screen_clip_arrays_counts_person_valid_windows() -> None:
    latents = np.ones((10, 2, 2), dtype=np.float32)
    deltas = np.ones((9, 4), dtype=np.float32)
    visible = np.array([False, True, True, True, True, True, False, False, False, False])
    report = screen_clip_arrays(
        latents=latents,
        deltas=deltas,
        person_visible=visible,
        history=3,
        horizon=4,
    )
    assert report.n_frames == 10
    assert report.n_windows == 6
    assert report.person_visible_frames == 5
    assert report.person_valid_windows >= 1


def test_screen_cache_dir_reports_clip_window_source_counts(tmp_path: Path) -> None:
    latents = np.ones((10, 2, 2), dtype=np.float32)
    deltas = np.ones((9, 4), dtype=np.float32)
    visible = np.ones(10, dtype=bool)
    bbox = np.tile(np.array([[0.5, 0.5, 0.2, 0.3]], dtype=np.float32), (10, 1))
    conf = np.full(10, 0.8, dtype=np.float32)
    np.savez(
        tmp_path / "ski03_fpv00_c000.npz",
        latents=latents,
        deltas=deltas,
        vo_confidence=np.ones(10, dtype=np.float32),
        frame_quality=np.ones(10, dtype=np.float32),
        timestamps=np.arange(10, dtype=np.float64),
        person_bbox=bbox,
        person_visible=visible,
        person_conf=conf,
    )
    report = screen_cache_dir(tmp_path, history=3, horizon=4)
    assert report["totals"]["clips"] == 1
    assert report["totals"]["windows"] == 6
    assert report["totals"]["sources"] == 1
    assert report["sources"]["ski03"]["clips"] == 1
    assert report["clips"][0]["clip_id"] == "ski03_fpv00_c000"
