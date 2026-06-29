"""AdamW parameter-group construction (TORCH tier) — B1.22a.

Standard transformer weight-decay recipe: matrices (``ndim >= 2``) get decoupled
weight decay; biases, norm scales, and positional/temporal embeddings do not. Used
by ``train_sports.py`` for both full-model and ``--latent-only`` (predictor-only) runs.

torch is imported lazily so this module imports on a torch-free box.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def build_param_groups(
    module: torch.nn.Module, weight_decay: float
) -> list[dict[str, Any]]:
    """Split ``module``'s trainable params into AdamW decay / no-decay groups.

    - **decay** (``weight_decay``): weight matrices (``p.ndim >= 2``).
    - **no-decay** (``0.0``): biases, LayerNorm scales/shifts (``p.ndim < 2``), and any
      parameter whose name contains ``embed`` (the temporal/positional embedding).

    Empty groups are omitted. Only ``requires_grad`` params are included, so passing
    ``model.predictor`` for ``--latent-only`` naturally yields predictor-only groups.
    """
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "embed" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    groups: list[dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


__all__ = ["build_param_groups"]
