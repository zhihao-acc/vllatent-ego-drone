"""Pre-train sanity checks (PURE tier) — B1.21.

Runs at training start: reads N random samples, verifies shapes/dtypes/masks.
Raises on any inconsistency so bad data never enters a training run silently.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vllatent.schemas import DOF, EMBED_DIM, HISTORY, HORIZON, PATCH_TOKENS

if TYPE_CHECKING:
    from vllatent.data.sports_loader import SportsTrainingDataset


def run_sanity_check(dataset: SportsTrainingDataset, n_samples: int = 5) -> None:
    """Validate N random samples from a SportsTrainingDataset.

    Raises ValueError on any contract breach.
    """
    if len(dataset) == 0:
        raise ValueError("Dataset is empty — cannot run sanity check")

    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)

    for idx in indices:
        sample = dataset[int(idx)]
        _check_sample(sample, int(idx))


def _check_sample(sample: object, idx: int) -> None:
    """Validate one SportsSample's shapes/dtypes/masks."""
    prefix = f"sample[{idx}]"

    z_t = getattr(sample, "z_t", None)
    if z_t is None or z_t.shape != (PATCH_TOKENS, EMBED_DIM):
        raise ValueError(f"{prefix}.z_t shape {getattr(z_t, 'shape', None)}, expected ({PATCH_TOKENS}, {EMBED_DIM})")
    if z_t.dtype != np.float16:
        raise ValueError(f"{prefix}.z_t dtype {z_t.dtype}, expected float16")

    hist = getattr(sample, "history_latents", None)
    if hist is None or hist.shape != (HISTORY, PATCH_TOKENS, EMBED_DIM):
        raise ValueError(f"{prefix}.history_latents shape {getattr(hist, 'shape', None)}, expected ({HISTORY}, {PATCH_TOKENS}, {EMBED_DIM})")

    mask = getattr(sample, "history_mask", None)
    if mask is None or mask.shape != (HISTORY,):
        raise ValueError(f"{prefix}.history_mask shape {getattr(mask, 'shape', None)}, expected ({HISTORY},)")
    if mask.dtype != np.bool_:
        raise ValueError(f"{prefix}.history_mask dtype {mask.dtype}, expected bool")
    if not mask[-1]:
        raise ValueError(f"{prefix}.history_mask[-1] is False — z_t slot must always be real")

    tgt = getattr(sample, "target_latents", None)
    if tgt is None or tgt.shape != (HORIZON, PATCH_TOKENS, EMBED_DIM):
        raise ValueError(f"{prefix}.target_latents shape {getattr(tgt, 'shape', None)}")

    deltas = getattr(sample, "target_deltas", None)
    if deltas is None or deltas.shape != (HORIZON, DOF):
        raise ValueError(f"{prefix}.target_deltas shape {getattr(deltas, 'shape', None)}")
    if deltas.dtype != np.float32:
        raise ValueError(f"{prefix}.target_deltas dtype {deltas.dtype}, expected float32")
    if not np.all(np.isfinite(deltas)):
        raise ValueError(f"{prefix}.target_deltas contains non-finite values")

    last_act = getattr(sample, "last_action", None)
    if last_act is None or last_act.shape != (DOF,):
        raise ValueError(f"{prefix}.last_action shape {getattr(last_act, 'shape', None)}, expected ({DOF},)")
    if not np.all(np.isfinite(last_act)):
        raise ValueError(f"{prefix}.last_action contains non-finite values")

    dt = getattr(sample, "dt_seconds", None)
    if dt is None or dt.shape != (HORIZON,):
        raise ValueError(f"{prefix}.dt_seconds shape {getattr(dt, 'shape', None)}")
    if not np.all(np.isfinite(dt)) or np.any(dt <= 0):
        raise ValueError(f"{prefix}.dt_seconds contains non-finite or non-positive values")

    vo = getattr(sample, "vo_confidence", None)
    if vo is not None:
        if not np.all(np.isfinite(vo)):
            raise ValueError(f"{prefix}.vo_confidence contains non-finite values")
        if np.any(vo < 0):
            raise ValueError(f"{prefix}.vo_confidence contains negative values")

    fq = getattr(sample, "frame_quality", None)
    if fq is not None:
        if not np.isfinite(fq):
            raise ValueError(f"{prefix}.frame_quality is non-finite: {fq}")
        if fq < 0 or fq > 1:
            raise ValueError(f"{prefix}.frame_quality out of [0,1]: {fq}")
