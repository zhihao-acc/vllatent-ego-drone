"""Tests for sports sliding-window loader (B1.13)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vllatent.data.sports_loader import (
    MAX_DISPLACEMENT_5HZ,
    MAX_DYAW_DEG_5HZ,
    NormStats,
    SportsSample,
    SportsTrainingDataset,
    compute_norm_stats,
    median_filter_deltas,
    physics_clip,
    velocity_normalize,
)
from vllatent.plan_tokens import PLAN_TOKEN_DIM
from vllatent.scale_free_targets import SCALE_FREE_ACTION_DIM
from vllatent.schemas import DOF, EMBED_DIM, HISTORY, HORIZON, LATENT_DTYPE, MASK_DTYPE, PATCH_TOKENS


def _make_clip_npz(
    path: Path,
    n_frames: int = 20,
    *,
    fps: float = 5.0,
    constant_delta: bool = False,
    spike_frame: int | None = None,
    domain: str | None = None,
    deltas_override: np.ndarray | None = None,
    include_person: bool = False,
) -> None:
    """Write a synthetic sports .npz clip."""
    rng = np.random.default_rng(0)
    latents = rng.standard_normal((n_frames, PATCH_TOKENS, EMBED_DIM)).astype(LATENT_DTYPE)
    deltas = rng.standard_normal((n_frames - 1, DOF)).astype(np.float32) * 0.1
    if deltas_override is not None:
        deltas = deltas_override.astype(np.float32)
    if constant_delta:
        deltas[:] = [0.1, 0.05, -0.02, 1.0]
    if spike_frame is not None and spike_frame < n_frames - 1:
        deltas[spike_frame] = [100.0, 100.0, 100.0, 500.0]
    vo_confidence = np.clip(rng.random(n_frames).astype(np.float32), 0.1, 1.0)
    frame_quality = np.clip(rng.random(n_frames).astype(np.float32), 0.2, 1.0)
    timestamps = np.arange(n_frames, dtype=np.float64) / fps
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {"domain": np.array(domain)} if domain is not None else {}
    if include_person:
        extra["person_bbox"] = np.tile(
            np.array([[0.5, 0.4, 0.2, 0.25]], dtype=np.float32),
            (n_frames, 1),
        )
        extra["person_visible"] = np.ones(n_frames, dtype=bool)
        extra["person_conf"] = np.full(n_frames, 0.75, dtype=np.float32)
    np.savez(
        str(path),
        latents=latents,
        deltas=deltas,
        vo_confidence=vo_confidence,
        frame_quality=frame_quality,
        timestamps=timestamps,
        **extra,
    )


# --- Preprocessing unit tests ---


class TestPhysicsClip:
    def test_clips_displacement(self) -> None:
        deltas = np.array([[10.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        dt = np.array([0.2], dtype=np.float32)
        result = physics_clip(deltas, dt)
        assert result[0, 0] == pytest.approx(MAX_DISPLACEMENT_5HZ)

    def test_clips_dyaw(self) -> None:
        deltas = np.array([[0.0, 0.0, 0.0, 100.0]], dtype=np.float32)
        dt = np.array([0.2], dtype=np.float32)
        result = physics_clip(deltas, dt)
        assert result[0, 3] == pytest.approx(MAX_DYAW_DEG_5HZ)

    def test_scales_with_dt(self) -> None:
        deltas = np.array([[10.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        dt_slow = np.array([0.5], dtype=np.float32)
        result = physics_clip(deltas, dt_slow)
        assert result[0, 0] == pytest.approx(MAX_DISPLACEMENT_5HZ * 2.5)

    def test_passthrough_within_limits(self) -> None:
        deltas = np.array([[0.1, 0.05, -0.02, 1.0]], dtype=np.float32)
        dt = np.array([0.2], dtype=np.float32)
        result = physics_clip(deltas, dt)
        np.testing.assert_array_almost_equal(result, deltas)


class TestMedianFilter:
    def test_removes_spike(self) -> None:
        deltas = np.array([
            [0.1, 0.1, 0.1, 0.1],
            [100.0, 100.0, 100.0, 100.0],
            [0.1, 0.1, 0.1, 0.1],
        ], dtype=np.float32)
        result = median_filter_deltas(deltas, k=3)
        np.testing.assert_array_almost_equal(result[1], [0.1, 0.1, 0.1, 0.1])

    def test_preserves_two_consecutive(self) -> None:
        deltas = np.array([
            [0.1, 0.1, 0.1, 0.1],
            [5.0, 5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0, 5.0],
            [0.1, 0.1, 0.1, 0.1],
        ], dtype=np.float32)
        result = median_filter_deltas(deltas, k=3)
        assert result[1, 0] == pytest.approx(5.0)
        assert result[2, 0] == pytest.approx(5.0)

    def test_short_input(self) -> None:
        deltas = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        result = median_filter_deltas(deltas, k=3)
        np.testing.assert_array_almost_equal(result, deltas)


class TestVelocityNormalize:
    def test_divides_by_dt(self) -> None:
        deltas = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        dt = np.array([0.5], dtype=np.float32)
        result = velocity_normalize(deltas, dt)
        np.testing.assert_array_almost_equal(result, [[2.0, 4.0, 6.0, 8.0]])

    def test_handles_zero_dt(self) -> None:
        deltas = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        dt = np.array([0.0], dtype=np.float32)
        result = velocity_normalize(deltas, dt)
        assert np.all(np.isfinite(result))


class TestNormStats:
    def test_normalize_denormalize_roundtrip(self) -> None:
        stats = NormStats(
            mean=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            std=np.array([0.5, 1.0, 2.0, 0.1], dtype=np.float32),
        )
        v = np.array([1.5, 3.0, 5.0, 4.2], dtype=np.float32)
        normalized = stats.normalize(v)
        recovered = stats.denormalize(normalized)
        np.testing.assert_array_almost_equal(recovered, v)

    def test_zero_std_safe(self) -> None:
        stats = NormStats(
            mean=np.zeros(DOF, dtype=np.float32),
            std=np.zeros(DOF, dtype=np.float32),
        )
        v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = stats.normalize(v)
        assert np.all(np.isfinite(result))

    def test_compute_from_list(self) -> None:
        rng = np.random.default_rng(0)
        vels = [rng.standard_normal((10, DOF)).astype(np.float32) for _ in range(3)]
        stats = compute_norm_stats(vels)
        assert stats.mean.shape == (DOF,)
        assert stats.std.shape == (DOF,)
        assert stats.mean.dtype == np.float32

    def test_compute_empty(self) -> None:
        stats = compute_norm_stats([])
        np.testing.assert_array_equal(stats.mean, np.zeros(DOF))
        np.testing.assert_array_equal(stats.std, np.ones(DOF))


# --- Dataset tests ---


class TestSportsTrainingDataset:
    def test_sample_shapes(self, tmp_path: Path) -> None:
        _make_clip_npz(tmp_path / "clip01.npz", n_frames=20)
        ds = SportsTrainingDataset(tmp_path)
        sample = ds[0]
        assert isinstance(sample, SportsSample)
        assert sample.z_t.shape == (PATCH_TOKENS, EMBED_DIM)
        assert sample.z_t.dtype == LATENT_DTYPE
        assert sample.history_latents.shape == (HISTORY, PATCH_TOKENS, EMBED_DIM)
        assert sample.history_latents.dtype == LATENT_DTYPE
        assert sample.history_mask.shape == (HISTORY,)
        assert sample.history_mask.dtype == MASK_DTYPE
        assert sample.target_latents.shape == (HORIZON, PATCH_TOKENS, EMBED_DIM)
        assert sample.target_latents.dtype == LATENT_DTYPE
        assert sample.history_person_bbox.shape == (HISTORY, 4)
        assert sample.history_person_bbox.dtype == np.float32
        assert sample.history_person_visible.shape == (HISTORY,)
        assert sample.history_person_visible.dtype == MASK_DTYPE
        assert sample.history_person_conf.shape == (HISTORY,)
        assert sample.history_person_conf.dtype == np.float32
        assert sample.target_person_bbox.shape == (HORIZON, 4)
        assert sample.target_person_bbox.dtype == np.float32
        assert sample.target_person_visible.shape == (HORIZON,)
        assert sample.target_person_visible.dtype == MASK_DTYPE
        assert sample.target_person_conf.shape == (HORIZON,)
        assert sample.target_person_conf.dtype == np.float32
        assert sample.person_state_target.shape == (HORIZON, 4)
        assert sample.person_state_target.dtype == np.float32
        assert sample.target_deltas.shape == (HORIZON, DOF)
        assert sample.target_deltas.dtype == np.float32
        assert sample.last_action.shape == (DOF,)
        assert sample.last_action.dtype == np.float32
        assert sample.planned_actions.shape == (HORIZON, PLAN_TOKEN_DIM)
        assert sample.planned_actions.dtype == np.float32
        assert sample.planned_actions_valid_mask.shape == (HORIZON,)
        assert sample.planned_actions_valid_mask.dtype == MASK_DTYPE
        assert sample.target_actions_scale_free.shape == (HORIZON, SCALE_FREE_ACTION_DIM)
        assert sample.target_actions_scale_free.dtype == np.float32
        assert sample.target_actions_moving_mask.shape == (HORIZON,)
        assert sample.target_actions_moving_mask.dtype == MASK_DTYPE
        assert sample.target_actions_speed_mask.shape == (HORIZON,)
        assert sample.target_actions_speed_mask.dtype == MASK_DTYPE
        assert sample.last_action_scale_free.shape == (SCALE_FREE_ACTION_DIM,)
        assert sample.last_action_scale_free.dtype == np.float32
        assert sample.action_history_scale_free.shape == (HISTORY, SCALE_FREE_ACTION_DIM)
        assert sample.action_history_scale_free.dtype == np.float32
        assert sample.action_history_mask.shape == (HISTORY,)
        assert sample.action_history_mask.dtype == MASK_DTYPE
        assert sample.camera_history_path_scale_free.shape == (HISTORY, 3)
        assert sample.camera_history_path_scale_free.dtype == np.float32
        assert isinstance(sample.odom_reference_speed, float)
        assert sample.vo_confidence.shape == (HORIZON,)
        assert sample.dt_seconds.shape == (HORIZON,)
        assert isinstance(sample.frame_quality, float)

    def test_old_cache_person_fallback_is_invisible(self, tmp_path: Path) -> None:
        _make_clip_npz(tmp_path / "clip01.npz", n_frames=20)
        sample = SportsTrainingDataset(tmp_path)[0]
        assert not np.any(sample.history_person_visible)
        assert not np.any(sample.target_person_visible)
        np.testing.assert_allclose(sample.person_state_target, 0.0)

    def test_person_labels_read_from_cache(self, tmp_path: Path) -> None:
        _make_clip_npz(tmp_path / "clip01.npz", n_frames=20, include_person=True)
        sample = SportsTrainingDataset(tmp_path)[3]
        assert np.all(sample.history_person_visible)
        assert np.all(sample.target_person_visible)
        np.testing.assert_allclose(sample.target_person_bbox[0], [0.5, 0.4, 0.2, 0.25], atol=1e-6)
        assert sample.person_state_target[0, 2] == pytest.approx(np.log(0.25))

    def test_block_causal_mask_at_start(self, tmp_path: Path) -> None:
        _make_clip_npz(tmp_path / "clip01.npz", n_frames=20)
        ds = SportsTrainingDataset(tmp_path)
        s0 = ds[0]
        assert s0.history_mask.tolist() == [False, False, True]
        s1 = ds[1]
        assert s1.history_mask.tolist() == [False, True, True]
        s2 = ds[2]
        assert s2.history_mask.tolist() == [True, True, True]

    def test_last_action_zero_at_clip_start(self, tmp_path: Path) -> None:
        _make_clip_npz(tmp_path / "clip01.npz", n_frames=20)
        ds = SportsTrainingDataset(tmp_path)
        s0 = ds[0]
        np.testing.assert_array_equal(s0.last_action, np.zeros(DOF, dtype=np.float32))
        np.testing.assert_allclose(s0.last_action_scale_free, [1.0, 0.0, 0.0, 0.0], atol=1e-6)
        assert not np.any(s0.action_history_mask)
        np.testing.assert_allclose(s0.camera_history_path_scale_free, 0.0, atol=1e-6)

    def test_last_action_nonzero_after_start(self, tmp_path: Path) -> None:
        _make_clip_npz(tmp_path / "clip01.npz", n_frames=20, constant_delta=True)
        ds = SportsTrainingDataset(tmp_path)
        s5 = ds[5]
        assert not np.allclose(s5.last_action, np.zeros(DOF))
        assert s5.action_history_mask.tolist() == [True, True, True]
        np.testing.assert_allclose(s5.action_history_scale_free[-1], s5.last_action_scale_free, atol=1e-6)
        assert np.all(np.isfinite(s5.camera_history_path_scale_free))

    def test_gt_history_not_predicted(self, tmp_path: Path) -> None:
        """History latents must be the actual cached latents, not zeros/predicted."""
        _make_clip_npz(tmp_path / "clip01.npz", n_frames=20)
        ds = SportsTrainingDataset(tmp_path)
        sample = ds[5]
        clip = ds._clips[0]
        for h in range(HISTORY):
            src = 5 - HISTORY + 1 + h
            np.testing.assert_array_equal(
                sample.history_latents[h],
                clip["latents"][src],
            )

    def test_dataset_length(self, tmp_path: Path) -> None:
        _make_clip_npz(tmp_path / "clip01.npz", n_frames=20)
        ds = SportsTrainingDataset(tmp_path)
        assert len(ds) == 20 - HORIZON

    def test_horizon8_configurable(self, tmp_path: Path) -> None:
        horizon = 8
        _make_clip_npz(tmp_path / "clip01.npz", n_frames=20, include_person=True)
        ds = SportsTrainingDataset(tmp_path, horizon=horizon)
        sample = ds[0]
        assert ds.horizon == horizon
        assert len(ds) == 20 - horizon
        assert sample.target_latents.shape == (horizon, PATCH_TOKENS, EMBED_DIM)
        assert sample.target_person_bbox.shape == (horizon, 4)
        assert sample.person_state_target.shape == (horizon, 4)
        assert sample.target_deltas.shape == (horizon, DOF)
        assert sample.planned_actions.shape == (horizon, PLAN_TOKEN_DIM)
        assert sample.planned_actions_valid_mask.shape == (horizon,)
        assert sample.vo_confidence.shape == (horizon,)
        assert sample.dt_seconds.shape == (horizon,)

    def test_multi_clip(self, tmp_path: Path) -> None:
        _make_clip_npz(tmp_path / "clip01.npz", n_frames=15)
        _make_clip_npz(tmp_path / "clip02.npz", n_frames=20)
        ds = SportsTrainingDataset(tmp_path)
        assert len(ds) == (15 - HORIZON) + (20 - HORIZON)

    def test_skips_short_clips(self, tmp_path: Path) -> None:
        _make_clip_npz(tmp_path / "short.npz", n_frames=5)
        _make_clip_npz(tmp_path / "long.npz", n_frames=20)
        ds = SportsTrainingDataset(tmp_path)
        assert len(ds) == 20 - HORIZON

    def test_clip_ids_filter(self, tmp_path: Path) -> None:
        _make_clip_npz(tmp_path / "clip01.npz", n_frames=20)
        _make_clip_npz(tmp_path / "clip02.npz", n_frames=20)
        ds = SportsTrainingDataset(tmp_path, clip_ids=["clip01"])
        assert len(ds) == 20 - HORIZON

    def test_augmentation_changes_output(self, tmp_path: Path) -> None:
        _make_clip_npz(tmp_path / "clip01.npz", n_frames=20, constant_delta=True)
        ds_no = SportsTrainingDataset(tmp_path, augment=False)
        ds_aug = SportsTrainingDataset(tmp_path, augment=True)
        s_no = ds_no[5]
        s_aug = ds_aug[5]
        assert not np.array_equal(s_no.target_deltas, s_aug.target_deltas)

    def test_physics_clip_applied(self, tmp_path: Path) -> None:
        _make_clip_npz(tmp_path / "clip01.npz", n_frames=20, spike_frame=5)
        ds = SportsTrainingDataset(tmp_path)
        for i in range(len(ds)):
            sample = ds[i]
            assert np.all(np.isfinite(sample.target_deltas))

    def test_dt_seconds_from_timestamps(self, tmp_path: Path) -> None:
        _make_clip_npz(tmp_path / "clip01.npz", n_frames=20, fps=5.0)
        ds = SportsTrainingDataset(tmp_path)
        sample = ds[0]
        np.testing.assert_array_almost_equal(
            sample.dt_seconds,
            np.full(HORIZON, 0.2, dtype=np.float32),
            decimal=5,
        )

    def test_norm_stats_saved_loaded(self, tmp_path: Path) -> None:
        _make_clip_npz(tmp_path / "clip01.npz", n_frames=20)
        ds = SportsTrainingDataset(tmp_path)
        stats_path = tmp_path / "norm_stats.npz"
        ds.save_norm_stats(stats_path)
        loaded = SportsTrainingDataset.load_norm_stats(stats_path)
        np.testing.assert_array_almost_equal(loaded.mean, ds.norm_stats.mean)
        np.testing.assert_array_almost_equal(loaded.std, ds.norm_stats.std)

    def test_external_norm_stats(self, tmp_path: Path) -> None:
        _make_clip_npz(tmp_path / "clip01.npz", n_frames=20)
        custom = NormStats(
            mean=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            std=np.array([0.5, 1.0, 2.0, 0.1], dtype=np.float32),
        )
        ds = SportsTrainingDataset(tmp_path, norm_stats=custom)
        np.testing.assert_array_equal(ds.norm_stats.mean, custom.mean)

    def test_empty_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="No .npz clips"):
            SportsTrainingDataset(tmp_path)

    def test_all_short_raises(self, tmp_path: Path) -> None:
        _make_clip_npz(tmp_path / "tiny.npz", n_frames=5)
        with pytest.raises(ValueError, match="No clips with enough"):
            SportsTrainingDataset(tmp_path)

    def test_vo_confidence_range(self, tmp_path: Path) -> None:
        _make_clip_npz(tmp_path / "clip01.npz", n_frames=20)
        ds = SportsTrainingDataset(tmp_path)
        for i in range(min(5, len(ds))):
            s = ds[i]
            assert np.all(s.vo_confidence >= 0)
            assert np.all(s.vo_confidence <= 1)

    def test_velocity_normalized_not_raw(self, tmp_path: Path) -> None:
        """target_deltas should be z-score normalized velocities, not raw deltas."""
        _make_clip_npz(tmp_path / "clip01.npz", n_frames=20, constant_delta=True)
        ds = SportsTrainingDataset(tmp_path, augment=False)
        sample = ds[5]
        assert sample.target_deltas.shape == (HORIZON, DOF)
        raw_delta = np.array([0.1, 0.05, -0.02, 1.0], dtype=np.float32)
        assert not np.allclose(sample.target_deltas[0], raw_delta, atol=0.01)

    def test_scale_free_targets_are_finite(self, tmp_path: Path) -> None:
        _make_clip_npz(tmp_path / "clip01.npz", n_frames=20)
        ds = SportsTrainingDataset(tmp_path)
        sample = ds[5]
        assert np.all(np.isfinite(sample.target_actions_scale_free))
        assert np.all(np.isfinite(sample.last_action_scale_free))
        assert np.isfinite(sample.odom_reference_speed)
        assert not np.any(np.abs(sample.target_actions_scale_free[sample.target_actions_speed_mask, 3]) > 8.0)
        moving = sample.target_actions_moving_mask
        unit_norms = np.linalg.norm(sample.target_actions_scale_free[moving, :3], axis=1)
        np.testing.assert_allclose(unit_norms, np.ones_like(unit_norms), atol=1e-6)

    def test_tiny_past_reference_speed_masks_clipped_future_speed(self, tmp_path: Path) -> None:
        n_frames = 20
        deltas = np.tile(np.array([1e-7, 0.0, 0.0, 0.0], dtype=np.float32), (n_frames - 1, 1))
        deltas[5:5 + HORIZON] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        _make_clip_npz(tmp_path / "clip01.npz", n_frames=n_frames, deltas_override=deltas)
        sample = SportsTrainingDataset(tmp_path, augment=False)[5]
        assert np.any(sample.target_actions_moving_mask)
        assert not np.any(sample.target_actions_speed_mask)
        assert np.all(np.abs(sample.target_actions_scale_free[:, 3]) <= 8.0)

    def test_future_delta_changes_do_not_change_b2_past_inputs(self, tmp_path: Path) -> None:
        """B2 previous-action inputs must be computed from observed past motion only."""
        n_frames = 20
        sample_t = 6
        base = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n_frames - 1, 1))
        changed = base.copy()
        changed[sample_t:sample_t + HORIZON] = np.array([0.0, 2.0, 0.0, 0.0], dtype=np.float32)

        _make_clip_npz(tmp_path / "a" / "clip01.npz", n_frames=n_frames, deltas_override=base)
        _make_clip_npz(tmp_path / "b" / "clip01.npz", n_frames=n_frames, deltas_override=changed)
        a = SportsTrainingDataset(tmp_path / "a")[sample_t]
        b = SportsTrainingDataset(tmp_path / "b")[sample_t]

        np.testing.assert_allclose(a.last_action_scale_free, b.last_action_scale_free, atol=1e-6)
        np.testing.assert_allclose(a.action_history_scale_free, b.action_history_scale_free, atol=1e-6)
        np.testing.assert_array_equal(a.action_history_mask, b.action_history_mask)
        np.testing.assert_allclose(
            a.camera_history_path_scale_free,
            b.camera_history_path_scale_free,
            atol=1e-6,
        )
        assert a.odom_reference_speed == pytest.approx(b.odom_reference_speed)
        assert not np.allclose(a.target_actions_scale_free, b.target_actions_scale_free)
        assert not np.allclose(a.planned_actions, b.planned_actions)


class TestDomainPlumbing:
    """B1.22a: per-clip domain tag flows to sample_domains (default 'real')."""

    def test_default_domain_is_real(self, tmp_path: Path) -> None:
        _make_clip_npz(tmp_path / "clip01.npz", n_frames=20)
        ds = SportsTrainingDataset(tmp_path)
        assert set(ds.sample_domains) == {"real"}
        assert len(ds.sample_domains) == len(ds)

    def test_game_domain_read_from_npz(self, tmp_path: Path) -> None:
        _make_clip_npz(tmp_path / "game01.npz", n_frames=20, domain="game")
        ds = SportsTrainingDataset(tmp_path)
        assert set(ds.sample_domains) == {"game"}

    def test_mixed_domains_parallel_to_samples(self, tmp_path: Path) -> None:
        _make_clip_npz(tmp_path / "real01.npz", n_frames=15)
        _make_clip_npz(tmp_path / "game01.npz", n_frames=20, domain="game")
        ds = SportsTrainingDataset(tmp_path, clip_ids=["real01", "game01"])
        assert len(ds.sample_domains) == len(ds)
        assert ds.sample_domains.count("game") == 20 - HORIZON
        assert ds.sample_domains.count("real") == 15 - HORIZON

    def test_sample_sources_parallel_to_samples(self, tmp_path: Path) -> None:
        _make_clip_npz(tmp_path / "ski03_fpv00_c000.npz", n_frames=15)
        _make_clip_npz(tmp_path / "cand05_fpv00_c000.npz", n_frames=20)
        ds = SportsTrainingDataset(tmp_path, clip_ids=["ski03_fpv00_c000", "cand05_fpv00_c000"])
        assert len(ds.sample_sources) == len(ds)
        assert ds.sample_sources.count("ski03") == 15 - HORIZON
        assert ds.sample_sources.count("cand05") == 20 - HORIZON


class TestImportPurity:
    def test_no_torch_at_module_level(self) -> None:
        import ast
        import importlib

        src = importlib.util.find_spec("vllatent.data.sports_loader")
        assert src is not None and src.origin is not None
        tree = ast.parse(Path(src.origin).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "torch", "torch imported at module level"
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("torch"):
                    raise AssertionError(f"torch imported at module level: {node.module}")
