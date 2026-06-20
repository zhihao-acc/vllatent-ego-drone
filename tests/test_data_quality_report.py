"""Tests for scripts/data_quality_report.py (PURE tier)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# Import the script as a module
import importlib.util

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "data_quality_report.py"
_spec = importlib.util.spec_from_file_location("data_quality_report", _SCRIPT)
assert _spec is not None and _spec.loader is not None
dqr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dqr)


@pytest.fixture()
def synthetic_cache(tmp_path: Path) -> Path:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    for i, n in enumerate([10, 8]):
        np.savez(
            cache_dir / f"clip{i}.npz",
            latents=np.zeros((n, 196, 768), dtype=np.float16),
            deltas=np.ones((n, 4), dtype=np.float32) * 0.1 * (i + 1),
            vo_confidence=np.full(n, 0.9, dtype=np.float32),
            frame_quality=np.full(n, 0.7, dtype=np.float32),
            vjepa_surprise=np.full(n, 0.1, dtype=np.float32),
        )
    manifest = {
        "encoder": {"model_id": "test_encoder", "dtype": "float16", "patch_tokens": 196, "dim": 768},
        "dataset": {"name": "wild_video", "source_type": "wild_video", "variant": "", "split": "", "license": "fair-use-research"},
        "motion_source": {"method": "megasam", "model": "megasam_base", "scale_mode": "normalized", "source_fps": 5.0},
        "entries": [{"clip_id": f"clip{i}", "n_frames": n, "latent_path": f"clip{i}.npz"} for i, n in enumerate([10, 8])],
    }
    with open(cache_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)
    return cache_dir


def test_build_report_structure(synthetic_cache: Path) -> None:
    report = dqr.build_report(synthetic_cache)
    assert report["n_clips"] == 2
    assert report["total_frames"] == 18
    assert report["manifest_present"] is True
    assert report["encoder_model"] == "test_encoder"
    assert report["motion_method"] == "megasam"
    assert len(report["per_clip"]) == 2


def test_build_report_percentiles(synthetic_cache: Path) -> None:
    report = dqr.build_report(synthetic_cache)
    assert report["vo_confidence"]["mean"] == pytest.approx(0.9, abs=0.01)
    assert report["frame_quality"]["mean"] == pytest.approx(0.7, abs=0.01)
    assert report["delta_magnitude"]["min"] > 0.0


def test_build_report_per_clip_frames(synthetic_cache: Path) -> None:
    report = dqr.build_report(synthetic_cache)
    frames = [c["n_frames"] for c in report["per_clip"]]
    assert frames == [10, 8]  # sorted by filename: clip0 (10 frames) before clip1 (8 frames)


def test_build_report_no_manifest(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    np.savez(cache_dir / "c.npz", latents=np.zeros((3, 196, 768), dtype=np.float16))
    report = dqr.build_report(cache_dir)
    assert report["manifest_present"] is False
    assert report["n_clips"] == 1


def test_build_report_empty_dir(tmp_path: Path) -> None:
    cache_dir = tmp_path / "empty"
    cache_dir.mkdir()
    report = dqr.build_report(cache_dir)
    assert report["n_clips"] == 0
    assert report["total_frames"] == 0


def test_json_output(synthetic_cache: Path, tmp_path: Path) -> None:
    json_path = tmp_path / "report.json"
    exit_code = dqr.main(["--cache", str(synthetic_cache), "--json", str(json_path)])
    assert exit_code == 0
    assert json_path.exists()
    loaded = json.loads(json_path.read_text())
    assert loaded["n_clips"] == 2


def test_missing_dir_returns_error(tmp_path: Path) -> None:
    exit_code = dqr.main(["--cache", str(tmp_path / "nonexistent")])
    assert exit_code == 1
