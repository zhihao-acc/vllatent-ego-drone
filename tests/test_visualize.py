"""Tests for vllatent.ingest.visualize — per-clip Plotly HTML quality report (B1.9b).

Pure-tier contract tests over synthetic fixtures. All Plotly rendering is tested
structurally (HTML contains expected sections, valid tags, correct data).

TDD: written BEFORE the implementation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from vllatent.schemas import DELTA_DTYPE, EMBED_DIM, LATENT_DTYPE, MASK_DTYPE, PATCH_TOKENS

# ---------------------------------------------------------------------------
# Fixtures — synthetic .npz cache data
# ---------------------------------------------------------------------------

def _make_clip_npz(tmp_path: Path, clip_id: str = "test_clip", n_frames: int = 25) -> Path:
    """Create a synthetic .npz matching the ingest cache contract."""
    rng = np.random.RandomState(42)
    arrays = {
        "latents": rng.randn(n_frames, PATCH_TOKENS, EMBED_DIM).astype(LATENT_DTYPE),
        "deltas": rng.randn(n_frames - 1, 4).astype(DELTA_DTYPE),
        "vo_confidence": rng.uniform(0.3, 1.0, n_frames).astype(np.float32),
        "frame_quality": rng.uniform(0.2, 1.0, n_frames).astype(np.float32),
        "timestamps": np.arange(n_frames, dtype=np.float64) / 5.0,
        "quality_mask": np.ones(n_frames, dtype=MASK_DTYPE),
    }
    p = tmp_path / f"{clip_id}.npz"
    np.savez(str(p), **arrays)
    return p


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

class TestGenerateClipReport:
    """Structural tests for the HTML quality report."""

    def test_returns_html_string(self, tmp_path: Path) -> None:
        from vllatent.ingest.visualize import generate_clip_report

        npz_path = _make_clip_npz(tmp_path)
        html = generate_clip_report(npz_path, clip_id="test_clip")
        assert isinstance(html, str)
        assert len(html) > 100
        assert "<html" in html.lower()

    def test_contains_quality_section(self, tmp_path: Path) -> None:
        from vllatent.ingest.visualize import generate_clip_report

        html = generate_clip_report(_make_clip_npz(tmp_path), clip_id="test_clip")
        assert "quality" in html.lower()

    def test_contains_delta_section(self, tmp_path: Path) -> None:
        from vllatent.ingest.visualize import generate_clip_report

        html = generate_clip_report(_make_clip_npz(tmp_path), clip_id="test_clip")
        assert "delta" in html.lower() or "body" in html.lower()

    def test_contains_vo_confidence_section(self, tmp_path: Path) -> None:
        from vllatent.ingest.visualize import generate_clip_report

        html = generate_clip_report(_make_clip_npz(tmp_path), clip_id="test_clip")
        assert "confidence" in html.lower()

    def test_contains_trajectory_section(self, tmp_path: Path) -> None:
        from vllatent.ingest.visualize import generate_clip_report

        html = generate_clip_report(_make_clip_npz(tmp_path), clip_id="test_clip")
        assert "trajectory" in html.lower() or "3d" in html.lower().replace(" ", "")

    def test_contains_latent_coherence_section(self, tmp_path: Path) -> None:
        from vllatent.ingest.visualize import generate_clip_report

        html = generate_clip_report(_make_clip_npz(tmp_path), clip_id="test_clip")
        assert "coherence" in html.lower() or "cosine" in html.lower()

    def test_contains_summary(self, tmp_path: Path) -> None:
        from vllatent.ingest.visualize import generate_clip_report

        html = generate_clip_report(_make_clip_npz(tmp_path), clip_id="test_clip")
        assert "summary" in html.lower() or "test_clip" in html

    def test_self_contained_html(self, tmp_path: Path) -> None:
        """HTML should be self-contained (plotly.js bundled or CDN link)."""
        from vllatent.ingest.visualize import generate_clip_report

        html = generate_clip_report(_make_clip_npz(tmp_path), clip_id="test_clip")
        assert "plotly" in html.lower()

    def test_write_to_file(self, tmp_path: Path) -> None:
        from vllatent.ingest.visualize import generate_clip_report

        npz_path = _make_clip_npz(tmp_path)
        out = tmp_path / "report.html"
        html = generate_clip_report(npz_path, clip_id="test_clip", out_path=out)
        assert out.exists()
        content = out.read_text()
        assert content == html

    def test_minimal_frames(self, tmp_path: Path) -> None:
        """Report should handle minimum viable frame count (H+T = 7)."""
        from vllatent.ingest.visualize import generate_clip_report

        npz_path = _make_clip_npz(tmp_path, n_frames=7)
        html = generate_clip_report(npz_path, clip_id="tiny_clip")
        assert isinstance(html, str)
        assert "<html" in html.lower()


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

class TestDataExtraction:
    """Helper functions for extracting report data from .npz."""

    def test_latent_coherence_shape(self, tmp_path: Path) -> None:
        from vllatent.ingest.visualize import compute_latent_coherence

        npz_path = _make_clip_npz(tmp_path, n_frames=10)
        with np.load(str(npz_path)) as data:
            latents = data["latents"]
        coherence = compute_latent_coherence(latents)
        assert coherence.shape == (9,)
        assert coherence.dtype == np.float32

    def test_latent_coherence_range(self, tmp_path: Path) -> None:
        from vllatent.ingest.visualize import compute_latent_coherence

        npz_path = _make_clip_npz(tmp_path)
        with np.load(str(npz_path)) as data:
            latents = data["latents"]
        coherence = compute_latent_coherence(latents)
        assert np.all(coherence >= -1.0)
        assert np.all(coherence <= 1.0)

    def test_cumulative_trajectory(self, tmp_path: Path) -> None:
        from vllatent.ingest.visualize import compute_cumulative_trajectory

        deltas = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=np.float32)
        traj = compute_cumulative_trajectory(deltas)
        assert traj.shape == (4, 3)  # n_deltas + 1 positions, xyz
        np.testing.assert_array_almost_equal(traj[0], [0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(traj[1], [1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(traj[2], [1.0, 1.0, 0.0])
        np.testing.assert_array_almost_equal(traj[3], [1.0, 1.0, 1.0])

    def test_speed_magnitudes(self, tmp_path: Path) -> None:
        from vllatent.ingest.visualize import compute_speed_magnitudes

        deltas = np.array([
            [3.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)
        speeds = compute_speed_magnitudes(deltas)
        assert speeds.shape == (2,)
        np.testing.assert_almost_equal(speeds[0], 5.0)  # 3-4-5 triangle


# ---------------------------------------------------------------------------
# Import purity
# ---------------------------------------------------------------------------

class TestImportPurity:
    """visualize module imports without plotly/torch at module level."""

    def test_no_heavy_imports(self) -> None:
        import ast
        import importlib

        spec = importlib.util.find_spec("vllatent.ingest.visualize")
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
        heavy = {"plotly", "torch", "PIL", "jinja2"}
        leaked = top_imports & heavy
        assert not leaked, f"Module-level heavy imports: {leaked}"
