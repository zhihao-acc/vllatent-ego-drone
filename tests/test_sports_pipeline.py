"""Tests for vllatent.sports.pipeline and __main__ CLI."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vllatent.schemas import EMBED_DIM, PATCH_TOKENS
from vllatent.sports.cache import build_clip_npz, write_clip_npz


class TestCLIInspect:
    """Test the inspect subcommand (no torch/tool deps)."""

    def _make_cache(self, cache_dir: Path, n_clips: int = 1, n_frames: int = 8) -> None:
        entries = []
        for c in range(n_clips):
            cid = f"clip{c:02d}"
            arrs = build_clip_npz(
                latents=np.random.randn(n_frames, PATCH_TOKENS, EMBED_DIM).astype(np.float16),
                deltas=np.random.randn(n_frames - 1, 4).astype(np.float32),
                vo_confidence=np.ones(n_frames, dtype=np.float32),
                frame_quality=np.ones(n_frames, dtype=np.float32),
                timestamps=np.linspace(0.0, (n_frames - 1) * 0.2, n_frames),
                quality_mask=np.ones(n_frames, dtype=np.bool_),
            )
            write_clip_npz(arrs, cache_dir / f"{cid}.npz")
            entries.append({"clip_id": cid, "n_frames": n_frames, "latent_path": f"{cid}.npz"})

        manifest = {
            "cache_version": "0.2",
            "encoder": {"model_id": "test", "dtype": "float16", "patch_tokens": PATCH_TOKENS, "dim": EMBED_DIM},
            "dataset": {"name": "sports_following", "sport": "skiing"},
            "convention": {"color_order": "RGB", "frame": "camera_body"},
            "motion_source": {"method": "megasam", "model": "base", "scale_mode": "normalized", "source_fps": 5.0},
            "entries": entries,
        }
        (cache_dir / "manifest.json").write_text(json.dumps(manifest))

    def test_inspect_runs(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        self._make_cache(tmp_path)
        from vllatent.sports.__main__ import main
        ret = main(["inspect", "--cache", str(tmp_path), "--n", "3"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "1 clips" in out

    def test_inspect_multiple_clips(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        self._make_cache(tmp_path, n_clips=3)
        from vllatent.sports.__main__ import main
        ret = main(["inspect", "--cache", str(tmp_path)])
        assert ret == 0
        out = capsys.readouterr().out
        assert "3 clips" in out


class TestClipPipelineResult:
    def test_frozen(self) -> None:
        from vllatent.sports.pipeline import ClipPipelineResult
        r = ClipPipelineResult(
            clip_id="test", n_frames=10, n_accepted=8,
            latent_path="test.npz", stages_skipped=[], errors=[],
        )
        with pytest.raises(AttributeError):
            r.clip_id = "other"  # type: ignore[misc]

    def test_with_errors(self) -> None:
        from vllatent.sports.pipeline import ClipPipelineResult
        r = ClipPipelineResult(
            clip_id="bad", n_frames=0, n_accepted=0,
            latent_path="", stages_skipped=["download"],
            errors=["video not found"],
        )
        assert r.errors == ["video not found"]


class TestUpdateManifest:
    def test_builds_manifest_from_results(self, tmp_path: Path) -> None:
        from vllatent.sports.pipeline import ClipPipelineResult, update_manifest_from_results
        from vllatent.sports.config import SportsDataConfig

        cfg = SportsDataConfig(
            name="test",
            sport="skiing",
            raw_dir=str(tmp_path / "raw"),
            frames_dir=str(tmp_path / "frames"),
            cache_dir=str(tmp_path / "cache"),
            clips_yaml=str(tmp_path / "clips.yaml"),
        )

        results = [
            ClipPipelineResult(
                clip_id="ski01", n_frames=50, n_accepted=45,
                latent_path="ski01.npz", stages_skipped=[], errors=[],
            ),
            ClipPipelineResult(
                clip_id="bad", n_frames=0, n_accepted=0,
                latent_path="", stages_skipped=[], errors=["failed"],
            ),
        ]

        path = update_manifest_from_results(results, cfg)
        assert path.exists()

        loaded = json.loads(path.read_text())
        assert len(loaded["entries"]) == 1  # only the successful one
        assert loaded["entries"][0]["clip_id"] == "ski01"

    def test_appends_to_existing(self, tmp_path: Path) -> None:
        from vllatent.sports.pipeline import ClipPipelineResult, update_manifest_from_results
        from vllatent.sports.config import SportsDataConfig

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        existing = {
            "cache_version": "0.2",
            "encoder": {"model_id": "test", "dtype": "float16", "patch_tokens": 196, "dim": 768},
            "dataset": {"name": "sports_following", "sport": "skiing"},
            "convention": {"color_order": "RGB", "frame": "camera_body"},
            "motion_source": {"method": "megasam", "model": "base", "scale_mode": "normalized", "source_fps": 5.0},
            "entries": [{"clip_id": "ski01", "n_frames": 50, "latent_path": "ski01.npz"}],
        }
        (cache_dir / "manifest.json").write_text(json.dumps(existing))

        cfg = SportsDataConfig(
            name="test", sport="skiing",
            raw_dir=str(tmp_path / "raw"), frames_dir=str(tmp_path / "frames"),
            cache_dir=str(cache_dir), clips_yaml=str(tmp_path / "clips.yaml"),
        )

        results = [
            ClipPipelineResult(
                clip_id="ski02", n_frames=30, n_accepted=28,
                latent_path="ski02.npz", stages_skipped=[], errors=[],
            ),
        ]

        path = update_manifest_from_results(results, cfg)
        loaded = json.loads(path.read_text())
        assert len(loaded["entries"]) == 2
        ids = {e["clip_id"] for e in loaded["entries"]}
        assert ids == {"ski01", "ski02"}
