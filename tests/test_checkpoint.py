"""B1.19 tests: checkpoint save/load + config snapshot round-trip.

TORCH-tier tests (``@pytest.mark.torch``). Uses a tiny ``nn.Linear`` model to
verify save/load round-trips, optimizer state restoration, and config snapshot fidelity.
"""
from __future__ import annotations

import pytest
import yaml

from vllatent.config import Config

torch = pytest.importorskip("torch")

from vllatent.train.checkpoint import (  # noqa: E402
    _config_to_dict,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    snapshot_config,
)

pytestmark = pytest.mark.torch


def _tiny_model() -> torch.nn.Module:
    return torch.nn.Linear(8, 4, bias=False)


# --- config snapshot ---


def test_snapshot_config_writes_yaml(tmp_path):
    cfg = Config()
    path = snapshot_config(cfg, tmp_path)
    assert path.exists()
    loaded = yaml.safe_load(path.read_text())
    assert isinstance(loaded, dict)
    assert loaded["encoder"]["model_id"] == cfg.encoder.model_id
    assert loaded["predictor"]["depth"] == cfg.predictor.depth
    assert loaded["distill"]["lambda_latent"] == cfg.distill.lambda_latent


def test_snapshot_config_round_trips_all_sections(tmp_path):
    cfg = Config()
    path = snapshot_config(cfg, tmp_path)
    loaded = yaml.safe_load(path.read_text())
    assert set(loaded.keys()) == {"encoder", "predictor", "distill", "data", "cache"}


def test_snapshot_config_with_ingest(tmp_path):
    from vllatent.config import IngestConfig

    cfg = Config(ingest=IngestConfig())
    path = snapshot_config(cfg, tmp_path)
    loaded = yaml.safe_load(path.read_text())
    assert "ingest" in loaded
    assert loaded["ingest"]["target_fps"] == 5.0


def test_config_to_dict_converts_tuples_to_lists():
    cfg = Config()
    d = _config_to_dict(cfg)
    assert isinstance(d["data"]["splits"], list)


# --- save/load checkpoint ---


def test_save_load_round_trip(tmp_path):
    model = _tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cfg = Config()
    metrics = {"loss": 0.5, "cosine_sim": 0.8}

    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(model, opt, epoch=3, global_step=150, config=cfg, metrics=metrics, path=ckpt_path)
    assert ckpt_path.exists()

    model2 = _tiny_model()
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    ckpt = load_checkpoint(ckpt_path, model2, opt2)

    assert ckpt["epoch"] == 3
    assert ckpt["global_step"] == 150
    assert ckpt["metrics"]["loss"] == 0.5
    assert ckpt["config"]["encoder"]["model_id"] == cfg.encoder.model_id

    for p1, p2 in zip(model.parameters(), model2.parameters(), strict=True):
        assert torch.equal(p1, p2)


def test_save_creates_parent_dirs(tmp_path):
    model = _tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt_path = tmp_path / "deep" / "nested" / "ckpt.pt"
    save_checkpoint(model, opt, epoch=0, global_step=0, config=Config(), metrics={}, path=ckpt_path)
    assert ckpt_path.exists()


def test_load_without_optimizer(tmp_path):
    model = _tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(model, opt, epoch=1, global_step=10, config=Config(), metrics={}, path=ckpt_path)

    model2 = _tiny_model()
    ckpt = load_checkpoint(ckpt_path, model2)
    assert ckpt["epoch"] == 1
    for p1, p2 in zip(model.parameters(), model2.parameters(), strict=True):
        assert torch.equal(p1, p2)


def test_resume_produces_identical_gradients(tmp_path):
    """Save at step N, reload, verify step N+1 produces identical gradients."""
    seed_everything(42)
    model = _tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randn(2, 8)
    target = torch.randn(2, 4)

    loss = torch.nn.functional.mse_loss(model(x), target)
    loss.backward()
    opt.step()
    opt.zero_grad()

    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(model, opt, epoch=0, global_step=1, config=Config(), metrics={}, path=ckpt_path)

    loss2 = torch.nn.functional.mse_loss(model(x), target)
    loss2.backward()
    grad_original = model.weight.grad.clone()

    model_resumed = _tiny_model()
    opt_resumed = torch.optim.AdamW(model_resumed.parameters(), lr=1e-3)
    load_checkpoint(ckpt_path, model_resumed, opt_resumed)

    loss3 = torch.nn.functional.mse_loss(model_resumed(x), target)
    loss3.backward()
    grad_resumed = model_resumed.weight.grad

    assert torch.allclose(grad_original, grad_resumed, atol=1e-7)


# --- seed_everything ---


def test_seed_everything_deterministic():
    seed_everything(123)
    a = torch.randn(10)
    seed_everything(123)
    b = torch.randn(10)
    assert torch.equal(a, b)


# --- import purity ---


def test_checkpoint_module_importable_without_torch(monkeypatch):
    """The module can be imported on a pure box (torch lazy)."""
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "vllatent" / "train" / "checkpoint.py"
    tree = ast.parse(src.read_text())
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.If):
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "torch", "module-level `import torch` found"
        if isinstance(node, ast.ImportFrom):
            assert node.module != "torch", "module-level `from torch import ...` found"
