"""Tests for vllatent.sports.loader — SportsDataset map-style loader."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vllatent.schemas import EMBED_DIM, HISTORY, PATCH_TOKENS
from vllatent.sports.cache import build_clip_npz, write_clip_npz
from vllatent.sports.loader import SportsDataset, inspect_sports_cache


def _write_test_cache(cache_dir: Path, n_clips: int = 2, n_frames: int = 10) -> None:
    """Write a minimal test cache with manifest."""
    entries = []
    for c in range(n_clips):
        clip_id = f"clip{c:02d}"
        arrs = build_clip_npz(
            latents=np.random.randn(n_frames, PATCH_TOKENS, EMBED_DIM).astype(np.float16),
            deltas=np.random.randn(n_frames - 1, 4).astype(np.float32),
            vo_confidence=np.ones(n_frames, dtype=np.float32) * 0.9,
            frame_quality=np.ones(n_frames, dtype=np.float32) * 0.8,
            timestamps=np.linspace(0.0, (n_frames - 1) * 0.2, n_frames),
            quality_mask=np.ones(n_frames, dtype=np.bool_),
        )
        write_clip_npz(arrs, cache_dir / f"{clip_id}.npz")
        entries.append({
            "clip_id": clip_id,
            "n_frames": n_frames,
            "latent_path": f"{clip_id}.npz",
        })

    manifest = {
        "cache_version": "0.2",
        "encoder": {"model_id": "test", "dtype": "float16", "patch_tokens": PATCH_TOKENS, "dim": EMBED_DIM},
        "dataset": {"name": "sports_following", "sport": "skiing"},
        "convention": {"color_order": "RGB", "frame": "camera_body"},
        "motion_source": {"method": "megasam", "model": "base", "scale_mode": "normalized", "source_fps": 5.0},
        "entries": entries,
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest))


class TestSportsDataset:
    def test_length(self, tmp_path: Path) -> None:
        _write_test_cache(tmp_path, n_clips=2, n_frames=10)
        ds = SportsDataset(tmp_path, require_quality=False)
        assert len(ds) == 2 * 9  # 2 clips, 9 transitions each

    def test_n_clips(self, tmp_path: Path) -> None:
        _write_test_cache(tmp_path, n_clips=3, n_frames=5)
        ds = SportsDataset(tmp_path, require_quality=False)
        assert ds.n_clips == 3

    def test_getitem_shapes(self, tmp_path: Path) -> None:
        _write_test_cache(tmp_path, n_clips=1, n_frames=8)
        ds = SportsDataset(tmp_path, require_quality=False)
        s = ds[0]
        assert s.z_t.shape == (PATCH_TOKENS, EMBED_DIM)
        assert s.z_t.dtype == np.float16
        assert s.z_next.shape == (PATCH_TOKENS, EMBED_DIM)
        assert s.history_latents.shape == (HISTORY, PATCH_TOKENS, EMBED_DIM)
        assert s.history_mask.shape == (HISTORY,)
        assert s.delta_4dof.shape == (4,)
        assert s.delta_4dof.dtype == np.float32

    def test_history_padding_at_start(self, tmp_path: Path) -> None:
        _write_test_cache(tmp_path, n_clips=1, n_frames=5)
        ds = SportsDataset(tmp_path, require_quality=False)
        s = ds[0]  # t=0 → only 1 real history frame (the current one)
        assert s.history_mask.tolist() == [False, False, True]

    def test_history_fully_filled(self, tmp_path: Path) -> None:
        _write_test_cache(tmp_path, n_clips=1, n_frames=10)
        ds = SportsDataset(tmp_path, require_quality=False)
        s = ds[HISTORY]  # t=HISTORY → all history slots filled
        assert all(s.history_mask)

    def test_dt_seconds(self, tmp_path: Path) -> None:
        _write_test_cache(tmp_path, n_clips=1, n_frames=10)
        ds = SportsDataset(tmp_path, require_quality=False)
        s = ds[0]
        assert abs(s.dt_seconds - 0.2) < 0.01

    def test_quality_filtering(self, tmp_path: Path) -> None:
        n = 10
        cache_dir = tmp_path
        mask = np.ones(n, dtype=np.bool_)
        mask[3] = False
        mask[7] = False

        arrs = build_clip_npz(
            latents=np.random.randn(n, PATCH_TOKENS, EMBED_DIM).astype(np.float16),
            deltas=np.random.randn(n - 1, 4).astype(np.float32),
            vo_confidence=np.ones(n, dtype=np.float32),
            frame_quality=np.ones(n, dtype=np.float32),
            timestamps=np.linspace(0.0, (n - 1) * 0.2, n),
            quality_mask=mask,
        )
        write_clip_npz(arrs, cache_dir / "clip.npz")
        manifest = {
            "cache_version": "0.2",
            "encoder": {"model_id": "t", "dtype": "float16", "patch_tokens": PATCH_TOKENS, "dim": EMBED_DIM},
            "dataset": {"name": "sports_following", "sport": "skiing"},
            "convention": {"color_order": "RGB", "frame": "camera_body"},
            "motion_source": {"method": "megasam", "model": "b", "scale_mode": "normalized", "source_fps": 5.0},
            "entries": [{"clip_id": "clip", "n_frames": n, "latent_path": "clip.npz"}],
        }
        (cache_dir / "manifest.json").write_text(json.dumps(manifest))

        ds_filtered = SportsDataset(cache_dir, require_quality=True)
        ds_all = SportsDataset(cache_dir, require_quality=False)

        assert len(ds_filtered) < len(ds_all)
        # mask[3]=False → transitions (2,3) and (3,4) both excluded
        # mask[7]=False → transitions (6,7) and (7,8) both excluded
        # 9 total transitions - 4 excluded = 5
        assert len(ds_filtered) == 5


class TestInspectSportsCache:
    def test_prints_output(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        _write_test_cache(tmp_path, n_clips=1, n_frames=5)
        ret = inspect_sports_cache(tmp_path, n=2)
        assert ret == 0
        captured = capsys.readouterr()
        assert "1 clips" in captured.out
        assert "[0]" in captured.out
        assert "[1]" in captured.out
