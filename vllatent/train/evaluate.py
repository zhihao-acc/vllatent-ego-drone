"""Validation evaluation for the B-1 latent world model (TORCH tier) — B1.22a.

The B-1 DoD is "the predictor beats a persistence baseline". ``evaluate`` runs the
**predictor only** (latent path; head is Phase B-2) over a held-out val loader and reports,
per horizon step k:

  - ``cos[k]``         : mean cosine(predicted z_{t+k}, GT z_{t+k})   — the model
  - ``persistence[k]`` : mean cosine(z_t,           GT z_{t+k})        — "next ≈ current"
  - ``margin[k]``      : cos[k] − persistence[k]                       — the thing that must be > 0

Cosine is computed in fp32 over the flattened (P·D) patch grid, averaged over the batch and
sample-count-weighted across batches. torch is imported lazily.
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from vllatent.model.sports_model import SportsFollowingModel

_AMP_DTYPE = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


def _autocast_ctx(device: str, amp_dtype: str):  # noqa: ANN202 — context manager
    """CUDA autocast for bf16/fp16; a no-op everywhere else."""
    import torch

    if device == "cuda" and amp_dtype != "fp32":
        return torch.autocast("cuda", dtype=getattr(torch, _AMP_DTYPE[amp_dtype]))
    return nullcontext()


def per_horizon_cosine(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean (over batch) cosine per horizon step. ``(B,T,P,D)`` -> ``(T,)`` fp32."""
    import torch.nn.functional as F

    b, t = predicted.shape[0], predicted.shape[1]
    pred = predicted.reshape(b, t, -1).float()
    tgt = target.reshape(b, t, -1).float()
    return F.cosine_similarity(pred, tgt, dim=2).mean(dim=0)


def evaluate(
    model: SportsFollowingModel,
    loader: Any,
    device: str,
    amp_dtype: str = "fp32",
) -> dict[str, Any]:
    """Per-horizon val cosine + persistence baseline + margin (predictor only).

    Returns a dict with scalar means (``val_cos``, ``val_persistence``, ``val_margin``) and
    per-horizon lists. Raises ``ValueError`` if the loader yields no batches.
    """
    import torch

    was_training = model.training
    model.eval()
    sum_cos: torch.Tensor | None = None
    sum_pers: torch.Tensor | None = None
    n_total = 0

    try:
        with torch.no_grad():
            for batch in loader:
                z_t = batch.z_t.to(device)
                history_latents = batch.history_latents.to(device)
                history_mask = batch.history_mask.to(device)
                target_latents = batch.target_latents.to(device)
                last_action = batch.last_action.to(device)
                dt_seconds = batch.dt_seconds.to(device)

                with _autocast_ctx(device, amp_dtype):
                    predicted = model.predictor(
                        history_latents=history_latents,
                        z_t=z_t,
                        action_4dof=last_action,
                        dt_seconds=dt_seconds,
                        history_mask=history_mask,
                    )

                horizon = target_latents.shape[1]
                z_rep = z_t.unsqueeze(1).expand(-1, horizon, -1, -1)
                cos = per_horizon_cosine(predicted, target_latents)
                pers = per_horizon_cosine(z_rep, target_latents)

                bsz = z_t.shape[0]
                sum_cos = cos * bsz if sum_cos is None else sum_cos + cos * bsz
                sum_pers = pers * bsz if sum_pers is None else sum_pers + pers * bsz
                n_total += bsz
    finally:
        if was_training:
            model.train()

    if n_total == 0 or sum_cos is None or sum_pers is None:
        raise ValueError("evaluate(): val loader yielded no batches")

    mean_cos = sum_cos / n_total
    mean_pers = sum_pers / n_total
    margin = mean_cos - mean_pers
    return {
        "val_cos": float(mean_cos.mean()),
        "val_persistence": float(mean_pers.mean()),
        "val_margin": float(margin.mean()),
        "val_min_margin": float(margin.min()),
        "per_horizon_cos": [float(x) for x in mean_cos.tolist()],
        "per_horizon_persistence": [float(x) for x in mean_pers.tolist()],
        "per_horizon_margin": [float(x) for x in margin.tolist()],
        "n_samples": int(n_total),
    }


__all__ = ["evaluate", "per_horizon_cosine"]
