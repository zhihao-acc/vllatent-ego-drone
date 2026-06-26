"""Training visualization callback (TORCH tier) — B1.21.

Logs GT vs predicted metrics per horizon step every N training steps.
Writes JSON logs; TensorBoard integration optional (Phase B-2).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from vllatent.train.losses import LossOutput


class TrainingLogger:
    """Append-only JSON-lines logger for training metrics."""

    def __init__(self, log_dir: Path, log_every: int = 50) -> None:
        self.log_dir = log_dir
        self.log_every = log_every
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self.log_dir / "train_metrics.jsonl"

    def should_log(self, step: int) -> bool:
        return step % self.log_every == 0

    def log_step(
        self,
        step: int,
        epoch: int,
        loss_output: LossOutput,
        lr: float,
        predicted_latents: torch.Tensor | None = None,
        target_latents: torch.Tensor | None = None,
        predicted_deltas: torch.Tensor | None = None,
        target_deltas: torch.Tensor | None = None,
    ) -> None:
        """Log one training step's metrics to JSONL."""
        import torch

        entry: dict = {
            "step": step,
            "epoch": epoch,
            "loss_total": loss_output.total.item(),
            "loss_latent": loss_output.latent.item(),
            "loss_waypoint": loss_output.waypoint.item(),
            "cosine_sim": loss_output.cosine_sim.item(),
            "lr": lr,
        }

        if predicted_latents is not None and target_latents is not None:
            with torch.no_grad():
                per_step_cos = _per_horizon_cosine(predicted_latents, target_latents)
                entry["cosine_per_horizon"] = [round(v, 4) for v in per_step_cos]

        if predicted_deltas is not None and target_deltas is not None:
            with torch.no_grad():
                per_step_l1 = _per_horizon_l1(predicted_deltas, target_deltas)
                entry["wp_l1_per_horizon"] = [round(v, 4) for v in per_step_l1]

        with open(self._log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    @property
    def log_path(self) -> Path:
        return self._log_path


def _per_horizon_cosine(
    predicted: torch.Tensor, target: torch.Tensor
) -> list[float]:
    """Per-horizon-step mean cosine similarity, (B,T,P,D) → list of T floats."""
    import torch.nn.functional as F

    T = predicted.shape[1]
    result = []
    for t in range(T):
        p = predicted[:, t].reshape(predicted.shape[0], -1)
        g = target[:, t].reshape(target.shape[0], -1)
        cos = F.cosine_similarity(p, g, dim=1).mean().item()
        result.append(cos)
    return result


def _per_horizon_l1(
    predicted: torch.Tensor, target: torch.Tensor
) -> list[float]:
    """Per-horizon-step mean L1 error on waypoint deltas, (B,T,4) → list of T floats."""
    T = predicted.shape[1]
    result = []
    for t in range(T):
        l1 = (predicted[:, t] - target[:, t]).abs().mean().item()
        result.append(l1)
    return result
