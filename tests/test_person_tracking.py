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
    person_trackable_mask,
    person_tracks_from_cache,
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
    assert tracks.person_state_valid.shape == (5,)
    assert tracks.person_conf.shape == (5,)
    assert not np.any(tracks.person_visible)
    assert not np.any(tracks.person_state_valid)


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


def test_select_subject_prefers_strict_windows_over_longer_fragmented_track() -> None:
    fragmented_frames = (0, 1, 2, 4, 5, 6, 8, 9, 10)
    strict_window_frames = range(11, 18)
    detections = [
        TrackedDetection(
            frame_idx,
            11,
            np.array([35, 35, 55, 65], dtype=np.float32),
            0.9,
        )
        for frame_idx in fragmented_frames
    ]
    detections.extend(
        TrackedDetection(
            frame_idx,
            22,
            np.array([40, 35, 60, 65], dtype=np.float32),
            0.8,
        )
        for frame_idx in strict_window_frames
    )

    result = select_subject_track(
        detections,
        n_frames=18,
        image_hw=(100, 100),
        history=3,
        horizon=4,
    )

    assert result.selected_track_id == 22
    assert np.all(result.person_state_valid[11:18])


def test_select_subject_marks_zero_area_crop_box_invisible() -> None:
    detections = [
        # Wide 200x100 frame center-crops x=[50,150], so this clips to zero width.
        TrackedDetection(0, 1, np.array([0, 20, 40, 60], dtype=np.float32), 0.9),
    ]
    result = select_subject_track(detections, n_frames=2, image_hw=(100, 200))
    assert result.person_visible.tolist() == [False, False]
    np.testing.assert_allclose(result.person_bbox, np.zeros((2, 4), dtype=np.float32))


def test_select_subject_rejects_concurrently_plausible_second_track() -> None:
    detections = []
    for frame_idx in range(5):
        detections.extend(
            [
                TrackedDetection(
                    frame_idx, 11, np.array([35, 35, 55, 65], dtype=np.float32), 0.85
                ),
                TrackedDetection(
                    frame_idx, 22, np.array([45, 35, 65, 65], dtype=np.float32), 0.84
                ),
            ]
        )

    result = select_subject_track(detections, n_frames=5, image_hw=(100, 100))

    assert result.selected_track_id in {11, 22}
    assert result.second_best_track_id in {11, 22}
    assert result.selected_track_id != result.second_best_track_id
    assert result.subject_is_ambiguous
    assert result.subject_ambiguity_margin < 0.2
    assert not np.any(result.person_state_valid)


def test_select_subject_keeps_clear_track_when_runner_up_is_short() -> None:
    detections = [
        TrackedDetection(i, 11, np.array([35, 35, 55, 65], dtype=np.float32), 0.85)
        for i in range(5)
    ]
    detections.append(TrackedDetection(0, 22, np.array([45, 35, 65, 65], dtype=np.float32), 0.84))

    result = select_subject_track(detections, n_frames=5, image_hw=(100, 100))

    assert result.selected_track_id == 11
    assert result.second_best_track_id == 22
    assert not result.subject_is_ambiguous
    assert result.subject_ambiguity_margin > 0.2
    assert np.all(result.person_state_valid)


def test_select_subject_does_not_confuse_non_overlapping_track_fragments_with_two_subjects() -> None:
    detections = [
        TrackedDetection(i, 11, np.array([35, 35, 55, 65], dtype=np.float32), 0.85)
        for i in range(3)
    ]
    detections.extend(
        TrackedDetection(i, 22, np.array([45, 35, 65, 65], dtype=np.float32), 0.84)
        for i in range(3, 6)
    )

    result = select_subject_track(detections, n_frames=6, image_hw=(100, 100))

    assert not result.subject_is_ambiguous
    assert result.provenance["second_best_covisible_frames"] == 0


def test_select_subject_checks_all_covisible_runners_up() -> None:
    detections = [
        TrackedDetection(i, 11, np.array([35, 35, 55, 65], dtype=np.float32), 0.85)
        for i in range(6)
    ]
    detections.extend(
        TrackedDetection(i, 22, np.array([40, 35, 60, 65], dtype=np.float32), 0.84)
        for i in range(6, 11)
    )
    detections.extend(
        TrackedDetection(i, 33, np.array([45, 35, 65, 65], dtype=np.float32), 0.84)
        for i in range(5)
    )

    result = select_subject_track(detections, n_frames=11, image_hw=(100, 100))

    assert result.selected_track_id == 11
    assert result.second_best_track_id == 33
    assert result.subject_is_ambiguous
    assert not np.any(result.person_state_valid)


def test_encoder_crop_bbox_conversion_for_wide_frame() -> None:
    # 1280x720 crops to x=[280,1000], y=[0,720] before DINO resize.
    bbox = xyxy_to_encoder_crop_cxcywh(np.array([280, 180, 1000, 540], dtype=np.float32), (720, 1280))
    np.testing.assert_allclose(bbox, [0.5, 0.5, 1.0, 0.5], atol=1e-6)
    raw = np.array([[0.5, 0.5, 720 / 1280, 0.5]], dtype=np.float32)
    converted = raw_frame_cxcywh_to_encoder_crop(raw, (720, 1280))
    np.testing.assert_allclose(converted[0], [0.5, 0.5, 1.0, 0.5], atol=1e-6)


def test_person_tracks_from_cache_sanitizes_tiny_visible_boxes() -> None:
    clip = {
        "latents": np.ones((2, 1, 1), dtype=np.float32),
        "person_bbox": np.array(
            [
                [0.5, 0.5, 0.01, 0.01],
                [0.5, 0.5, 0.2, 0.2],
            ],
            dtype=np.float32,
        ),
        "person_visible": np.array([True, True]),
        "person_conf": np.array([0.9, 0.8], dtype=np.float32),
    }
    result = person_tracks_from_cache(clip)
    assert result.person_visible.tolist() == [False, True]
    assert result.person_state_valid.tolist() == [False, False]
    np.testing.assert_allclose(result.person_bbox[0], np.zeros(4, dtype=np.float32))
    assert result.person_conf[0] == pytest.approx(0.0)
    assert result.provenance["sanitized_invisible_frames"] == 1


def test_person_trackable_mask_requires_patch_scale_non_edge_run() -> None:
    bbox = np.array(
        [
            [0.5, 0.5, 0.20, 0.20],
            [0.5, 0.5, 0.20, 0.20],
            [0.5, 0.5, 0.20, 0.20],
            [0.1, 0.5, 0.20, 0.20],  # touches left edge.
            [0.5, 0.5, 0.04, 0.04],  # smaller than 4 DINO patches.
        ],
        dtype=np.float32,
    )
    visible = np.ones(5, dtype=np.bool_)
    mask = person_trackable_mask(bbox, visible)
    assert mask.tolist() == [True, True, True, False, False]


def test_person_tracks_from_cache_uses_state_valid_key_conservatively() -> None:
    bbox = np.tile(np.array([[0.5, 0.5, 0.20, 0.20]], dtype=np.float32), (4, 1))
    clip = {
        "latents": np.ones((4, 1, 1), dtype=np.float32),
        "person_bbox": bbox,
        "person_visible": np.ones(4, dtype=np.bool_),
        "person_state_valid": np.array([True, False, True, True]),
        "person_conf": np.ones(4, dtype=np.float32),
    }
    result = person_tracks_from_cache(clip)
    assert result.person_visible.tolist() == [True, True, True, True]
    assert result.person_state_valid.tolist() == [True, False, True, True]


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


def test_screen_clip_arrays_counts_only_full_valid_windows() -> None:
    latents = np.ones((7, 2, 2), dtype=np.float32)
    deltas = np.ones((6, 4), dtype=np.float32)
    visible = np.ones(7, dtype=np.bool_)
    report = screen_clip_arrays(
        latents=latents,
        deltas=deltas,
        person_visible=visible,
        history=3,
        horizon=4,
    )
    assert report.n_frames == 7
    assert report.n_windows == 3
    assert report.person_visible_frames == 7
    assert report.person_valid_windows == 1


def test_screen_clip_arrays_rejects_partial_future_validity() -> None:
    latents = np.ones((7, 2, 2), dtype=np.float32)
    deltas = np.ones((6, 4), dtype=np.float32)
    state_valid = np.ones(7, dtype=np.bool_)
    state_valid[-1] = False
    report = screen_clip_arrays(
        latents=latents,
        deltas=deltas,
        person_visible=np.ones(7, dtype=np.bool_),
        person_state_valid=state_valid,
        history=3,
        horizon=4,
    )
    assert report.person_valid_windows == 0


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
        person_selected_track_id=np.array(11, dtype=np.int64),
        person_second_best_track_id=np.array(22, dtype=np.int64),
        person_subject_ambiguity_margin=np.array(0.4, dtype=np.float32),
        person_subject_is_ambiguous=np.array(False, dtype=np.bool_),
    )
    report = screen_cache_dir(tmp_path, history=3, horizon=4)
    assert report["totals"]["clips"] == 1
    assert report["totals"]["windows"] == 6
    assert report["totals"]["sources"] == 1
    assert report["totals"]["subject_ambiguity_known_clips"] == 1
    assert report["totals"]["subject_ambiguity_unknown_clips"] == 0
    assert report["totals"]["subject_ambiguous_clips"] == 0
    assert report["sources"]["ski03"]["clips"] == 1
    assert report["clips"][0]["clip_id"] == "ski03_fpv00_c000"
    assert report["clips"][0]["subject_ambiguity_known"]
    assert report["clips"][0]["selected_track_id"] == 11
    assert report["clips"][0]["second_best_track_id"] == 22


def test_screen_cache_dir_reports_person_label_qc(tmp_path: Path) -> None:
    latents = np.ones((10, 2, 2), dtype=np.float32)
    deltas = np.ones((9, 4), dtype=np.float32)
    visible = np.array([True, True, True, False, False, False, False, False, False, False])
    bbox = np.zeros((10, 4), dtype=np.float32)
    bbox[0] = [0.5, 0.5, 0.01, 0.01]  # tiny -> invalid.
    bbox[1] = [0.1, 0.5, 0.2, 0.2]  # touches crop edge.
    bbox[2] = [0.5, 0.5, 0.2, 0.2]
    np.savez(
        tmp_path / "cand01_fpv00_c000.npz",
        latents=latents,
        deltas=deltas,
        vo_confidence=np.ones(10, dtype=np.float32),
        frame_quality=np.ones(10, dtype=np.float32),
        timestamps=np.arange(10, dtype=np.float64),
        person_bbox=bbox,
        person_visible=visible,
        person_conf=np.ones(10, dtype=np.float32),
    )
    report = screen_cache_dir(tmp_path, history=3, horizon=4)
    assert report["totals"]["person_invalid_visible_frames"] == 1
    assert report["totals"]["person_tiny_visible_frames"] == 1
    assert report["totals"]["person_edge_visible_frames"] == 1
    assert report["totals"]["subject_ambiguity_unknown_clips"] == 1
    assert "person_label_qc" in report["clips"][0]
    assert "person_invalid_labels" in report["clips"][0]["flags"]
    assert "subject_ambiguity_unknown" in report["clips"][0]["flags"]
