"""Tests for sports batch collation (B1.14)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vllatent.schemas import DOF, EMBED_DIM, HISTORY, HORIZON, LATENT_DTYPE, PATCH_TOKENS

torch = pytest.importorskip("torch")

from vllatent.data.collate import (  # noqa: E402
    TrainingBatch,
    collate_sports_batch,
)
from vllatent.data.sports_loader import SportsTrainingDataset  # noqa: E402
from vllatent.plan_tokens import PLAN_TOKEN_DIM  # noqa: E402


def _make_clip_npz(path: Path, n_frames: int = 20, fps: float = 5.0) -> None:
    rng = np.random.default_rng(0)
    latents = rng.standard_normal((n_frames, PATCH_TOKENS, EMBED_DIM)).astype(LATENT_DTYPE)
    deltas = rng.standard_normal((n_frames - 1, DOF)).astype(np.float32) * 0.1
    vo_confidence = np.clip(rng.random(n_frames).astype(np.float32), 0.1, 1.0)
    frame_quality = np.clip(rng.random(n_frames).astype(np.float32), 0.2, 1.0)
    timestamps = np.arange(n_frames, dtype=np.float64) / fps
    person_bbox = np.tile(np.array([[0.5, 0.4, 0.2, 0.25]], dtype=np.float32), (n_frames, 1))
    person_visible = np.ones(n_frames, dtype=bool)
    person_conf = np.full(n_frames, 0.75, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), latents=latents, deltas=deltas,
             vo_confidence=vo_confidence, frame_quality=frame_quality,
             timestamps=timestamps, person_bbox=person_bbox,
             person_visible=person_visible, person_conf=person_conf)


@pytest.mark.torch
class TestCollate:
    def test_batch_shapes(self, tmp_path: Path) -> None:
        _make_clip_npz(tmp_path / "c1.npz")
        ds = SportsTrainingDataset(tmp_path)
        samples = [ds[i] for i in range(4)]
        batch = collate_sports_batch(samples)

        assert isinstance(batch, TrainingBatch)
        B = 4
        assert batch.z_t.shape == (B, PATCH_TOKENS, EMBED_DIM)
        assert batch.history_latents.shape == (B, HISTORY, PATCH_TOKENS, EMBED_DIM)
        assert batch.history_mask.shape == (B, HISTORY)
        assert batch.target_latents.shape == (B, HORIZON, PATCH_TOKENS, EMBED_DIM)
        assert batch.history_person_bbox.shape == (B, HISTORY, 4)
        assert batch.history_person_visible.shape == (B, HISTORY)
        assert batch.history_person_state_valid.shape == (B, HISTORY)
        assert batch.history_person_conf.shape == (B, HISTORY)
        assert batch.target_person_bbox.shape == (B, HORIZON, 4)
        assert batch.target_person_visible.shape == (B, HORIZON)
        assert batch.target_person_state_valid.shape == (B, HORIZON)
        assert batch.target_person_conf.shape == (B, HORIZON)
        assert batch.person_state_target.shape == (B, HORIZON, 4)
        assert batch.target_deltas.shape == (B, HORIZON, DOF)
        assert batch.last_action.shape == (B, DOF)
        assert batch.planned_actions.shape == (B, HORIZON, PLAN_TOKEN_DIM)
        assert batch.planned_actions_valid_mask.shape == (B, HORIZON)
        assert batch.vo_confidence.shape == (B, HORIZON)
        assert batch.frame_quality.shape == (B,)
        assert batch.dt_seconds.shape == (B, HORIZON)
        assert batch.sample_weight.shape == (B,)
        assert not hasattr(batch, "target_actions_scale_free")
        assert not hasattr(batch, "last_action_scale_free")

    def test_dtypes(self, tmp_path: Path) -> None:
        _make_clip_npz(tmp_path / "c1.npz")
        ds = SportsTrainingDataset(tmp_path)
        batch = collate_sports_batch([ds[0], ds[1]])

        assert batch.z_t.dtype == torch.float16
        assert batch.history_latents.dtype == torch.float16
        assert batch.history_mask.dtype == torch.bool
        assert batch.history_person_bbox.dtype == torch.float32
        assert batch.history_person_visible.dtype == torch.bool
        assert batch.history_person_state_valid.dtype == torch.bool
        assert batch.history_person_conf.dtype == torch.float32
        assert batch.target_person_bbox.dtype == torch.float32
        assert batch.target_person_visible.dtype == torch.bool
        assert batch.target_person_state_valid.dtype == torch.bool
        assert batch.target_person_conf.dtype == torch.float32
        assert batch.person_state_target.dtype == torch.float32
        assert batch.target_deltas.dtype == torch.float32
        assert batch.last_action.dtype == torch.float32
        assert batch.planned_actions.dtype == torch.float32
        assert batch.planned_actions_valid_mask.dtype == torch.bool
        assert batch.vo_confidence.dtype == torch.float32
        assert batch.frame_quality.dtype == torch.float32
        assert batch.dt_seconds.dtype == torch.float32
        assert batch.sample_weight.dtype == torch.float32

    def test_sample_weight_floors(self, tmp_path: Path) -> None:
        _make_clip_npz(tmp_path / "c1.npz")
        ds = SportsTrainingDataset(tmp_path)
        batch = collate_sports_batch([ds[0], ds[1]])
        assert torch.all(batch.sample_weight > 0)
        min_weight = 0.1 * 0.05
        assert torch.all(batch.sample_weight >= min_weight - 1e-6)

    def test_sample_weight_formula(self, tmp_path: Path) -> None:
        _make_clip_npz(tmp_path / "c1.npz")
        ds = SportsTrainingDataset(tmp_path)
        samples = [ds[5], ds[6]]
        batch = collate_sports_batch(samples)
        for i, s in enumerate(samples):
            fq = max(s.frame_quality, 0.1)
            vo_mean = max(float(np.mean(s.vo_confidence)), 0.05)
            expected = fq * vo_mean
            assert batch.sample_weight[i].item() == pytest.approx(expected, rel=1e-5)

    def test_works_with_dataloader(self, tmp_path: Path) -> None:
        _make_clip_npz(tmp_path / "c1.npz", n_frames=20)
        ds = SportsTrainingDataset(tmp_path)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=4, collate_fn=collate_sports_batch,
        )
        batch = next(iter(loader))
        assert isinstance(batch, TrainingBatch)
        assert batch.z_t.shape[0] == 4

    def test_horizon8_batch_shapes(self, tmp_path: Path) -> None:
        horizon = 8
        _make_clip_npz(tmp_path / "c1.npz", n_frames=20)
        ds = SportsTrainingDataset(tmp_path, horizon=horizon)
        batch = collate_sports_batch([ds[0], ds[1]])
        assert batch.target_latents.shape == (2, horizon, PATCH_TOKENS, EMBED_DIM)
        assert batch.person_state_target.shape == (2, horizon, 4)
        assert batch.planned_actions.shape == (2, horizon, PLAN_TOKEN_DIM)
        assert batch.planned_actions_valid_mask.shape == (2, horizon)
        assert batch.dt_seconds.shape == (2, horizon)

class TestImportPurity:
    def test_no_torch_at_module_level(self) -> None:
        import ast
        import importlib

        src = importlib.util.find_spec("vllatent.data.collate")
        assert src is not None and src.origin is not None
        tree = ast.parse(Path(src.origin).read_text())
        module_imports: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                module_imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                module_imports.append(node.module)
        assert all(name.split(".")[0] != "torch" for name in module_imports)
