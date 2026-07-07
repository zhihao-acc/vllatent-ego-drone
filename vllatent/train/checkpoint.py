"""Checkpoint save/load + config snapshot (TORCH tier) — B1.19.

Saves model + optimizer state, epoch, global step, metrics, and a full YAML config
snapshot to the run directory so every checkpoint is reproducibly traceable. The config
snapshot is written ONCE at training start (not per checkpoint) — it's the same Config
that drove the run. Deterministic seeding helper included.

torch is lazy-imported so the module can be imported on a pure box without crashing.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from vllatent.config import Config

if TYPE_CHECKING:
    import torch


def seed_everything(seed: int) -> None:
    """Set deterministic seeds for torch + numpy + python random."""
    import random

    import numpy as np
    import torch as _torch

    random.seed(seed)
    np.random.seed(seed)
    _torch.manual_seed(seed)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(seed)


def _config_to_dict(config: Config) -> dict[str, Any]:
    """Recursively convert a frozen Config tree to a plain dict (YAML-safe)."""
    result: dict[str, Any] = {}
    for f in dataclasses.fields(config):
        val = getattr(config, f.name)
        if val is None:
            continue
        if dataclasses.is_dataclass(val):
            result[f.name] = {
                sf.name: getattr(val, sf.name)
                for sf in dataclasses.fields(val)
            }
            for k, v in result[f.name].items():
                if isinstance(v, tuple):
                    result[f.name][k] = list(v)
        else:
            result[f.name] = val
    return result


def snapshot_config(config: Config, run_dir: str | Path) -> Path:
    """Write a YAML config snapshot to ``<run_dir>/config_snapshot.yaml``.

    Called once at training start. Returns the path written.
    """
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "config_snapshot.yaml"
    path.write_text(yaml.dump(_config_to_dict(config), default_flow_style=False, sort_keys=False))
    return path


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    config: Config,
    metrics: dict[str, float],
    path: str | Path,
    scheduler: Any | None = None,
    val_metrics: dict[str, Any] | None = None,
) -> Path:
    """Save a training checkpoint.

    The checkpoint dict contains: ``model_state_dict``, ``optimizer_state_dict``,
    ``epoch``, ``global_step``, ``config`` (plain dict), ``metrics``, optionally
    ``scheduler_state_dict``, and optionally ``val_metrics`` from the active
    trainer/evaluator.
    """
    import torch as _torch

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dict: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "config": _config_to_dict(config),
        "metrics": dict(metrics),
    }
    if scheduler is not None:
        ckpt_dict["scheduler_state_dict"] = scheduler.state_dict()
    if val_metrics is not None:
        ckpt_dict["val_metrics"] = dict(val_metrics)
    _torch.save(ckpt_dict, out)
    return out


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    map_location: str | None = None,
) -> dict[str, Any]:
    """Load a checkpoint into a model (and optionally optimizer).

    Returns the full checkpoint dict (epoch, global_step, config, metrics)
    so the caller can resume training state.
    """
    import torch as _torch

    ckpt = _torch.load(Path(path), map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


__all__ = [
    "seed_everything",
    "snapshot_config",
    "save_checkpoint",
    "load_checkpoint",
]
