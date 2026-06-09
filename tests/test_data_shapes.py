"""A5.15 tests: the distillation loader emits well-formed ``(StepSample, OracleTarget)`` tuples.

The Dataset is numpy-only (it emits the typed numpy contract objects), so these shape/dtype/mask
tests are PURE (run in ``make test``); one ``@pytest.mark.torch`` test proves the Dataset plugs into
a real ``torch.utils.data.DataLoader``. The cache is a synthetic tiny_dump written to ``tmp_path`` —
no blobs committed; it mirrors the on-disk format A5.14 will produce.

Value convention so history/z_next/episode-routing are checkable: episode ``e`` stores
``latents[k] == base_e + k`` (ep0 base 1, ep1 base 10), distinct from the zero-pad.
"""
from __future__ import annotations

import numpy as np
import pytest

from vllatent.config import Config
from vllatent.data.loader import CachedLatentDataset, inspect_cache
from vllatent.manifest import build_manifest, validate_manifest, write_manifest
from vllatent.schemas import (
    DELTA_DTYPE,
    DOF,
    EMBED_DIM,
    HISTORY,
    LATENT_DTYPE,
    MASK_DTYPE,
    N_ACTIONS,
    PATCH_TOKENS,
    TEACHER_DOF,
)

_LANG_M = 5
_EPISODES = [(4, 1), (3, 10)]  # (n_frames, latent base value) per episode


def _episode_arrays(n: int, base: int) -> dict[str, np.ndarray]:
    latents = np.zeros((n, PATCH_TOKENS, EMBED_DIM), dtype=LATENT_DTYPE)
    for k in range(n):
        latents[k] = base + k  # broadcast-fill so latents[k] is identifiable
    actions = np.array([((k % 7) + 1) if k < n - 1 else 0 for k in range(n)], dtype=np.int64)
    deltas = np.stack([np.array([k, k + 0.5, -k, 0.0], dtype=np.float32) for k in range(n)])
    lang = np.full((_LANG_M, EMBED_DIM), 0.25, dtype=LATENT_DTYPE)
    waypoint = np.stack([np.array([k * 0.1, 0.0, 0.0, 0.0], dtype=np.float32) for k in range(n)])
    pose6 = np.zeros((n, 6), dtype=np.float32)
    rollpitch = np.zeros((n,), dtype=np.float32)                       # >= 0
    disagree = np.array([0.1 * (k + 1) for k in range(n)], dtype=np.float32)  # >= 0
    surprise = np.array([0.2 * (k + 1) for k in range(n)], dtype=np.float32)  # >= 0
    return {
        "latents": latents, "actions": actions, "deltas": deltas, "lang_tokens": lang,
        "waypoint_4dof": waypoint, "teacher_pose6": pose6, "rollpitch_resid": rollpitch,
        "disagreement": disagree, "vjepa_surprise": surprise,
    }


def _write_tiny_cache(tmp_path) -> str:
    cache = tmp_path / "cache"
    cache.mkdir()
    entries = []
    for ep_i, (n, base) in enumerate(_EPISODES):
        np.savez(cache / f"ep{ep_i}.npz", **_episode_arrays(n, base))
        entries.append({
            "episode_id": f"ep{ep_i}", "trajectory_id": f"traj{ep_i}",
            "scene_id": ep_i + 1, "n_frames": n, "latent_path": f"ep{ep_i}.npz",
        })
    write_manifest(build_manifest(Config(), split="train", entries=entries), cache)
    return str(cache)


def test_fixture_manifest_is_valid(tmp_path) -> None:
    cache = _write_tiny_cache(tmp_path)
    import json
    from pathlib import Path
    manifest = json.loads((Path(cache) / "manifest.json").read_text())
    assert validate_manifest(manifest) == []


def test_len_and_episode_count(tmp_path) -> None:
    ds = CachedLatentDataset(_write_tiny_cache(tmp_path))
    # transitions = sum(N_e - 1) = (4-1) + (3-1) = 5; terminal STOP excluded.
    assert len(ds) == 5
    assert ds.n_episodes == 2
    assert ds.history == HISTORY and ds.horizon == Config().predictor.horizon


def test_sample_shapes_and_dtypes(tmp_path) -> None:
    ds = CachedLatentDataset(_write_tiny_cache(tmp_path))
    step, oracle = ds[0]
    assert step.z_t.shape == (PATCH_TOKENS, EMBED_DIM) and step.z_t.dtype == LATENT_DTYPE
    assert step.history_latents.shape == (HISTORY, PATCH_TOKENS, EMBED_DIM)
    assert step.history_mask.shape == (HISTORY,) and step.history_mask.dtype == MASK_DTYPE
    assert step.lang_tokens.shape == (_LANG_M, EMBED_DIM) and step.lang_tokens.dtype == LATENT_DTYPE
    assert step.lang_mask.shape == (_LANG_M,) and bool(step.lang_mask.all())
    assert step.z_next.shape == (PATCH_TOKENS, EMBED_DIM) and step.z_next.dtype == LATENT_DTYPE
    assert step.delta_4dof.shape == (DOF,) and step.delta_4dof.dtype == DELTA_DTYPE
    assert isinstance(step.action_id, int) and 0 <= step.action_id < N_ACTIONS
    assert oracle.waypoint_4dof.shape == (DOF,) and oracle.waypoint_4dof.dtype == DELTA_DTYPE
    assert oracle.teacher_pose6.shape == (TEACHER_DOF,)
    assert oracle.disagreement >= 0.0 and oracle.vjepa_surprise >= 0.0 and oracle.rollpitch_resid >= 0.0


def test_block_causal_history_padding_at_episode_start(tmp_path) -> None:
    ds = CachedLatentDataset(_write_tiny_cache(tmp_path))
    step, _ = ds[0]  # (ep0, t=0): only the current frame is real
    assert step.history_mask.tolist() == [False, False, True]
    assert np.all(step.history_latents[0] == 0) and np.all(step.history_latents[1] == 0)
    assert np.all(step.history_latents[HISTORY - 1] == step.z_t)  # current = last history slot
    assert np.all(step.z_t == 1) and np.all(step.z_next == 2)     # latents[0]=all-1, latents[1]=all-2


def test_full_history_window_mid_episode(tmp_path) -> None:
    ds = CachedLatentDataset(_write_tiny_cache(tmp_path))
    step, _ = ds[2]  # (ep0, t=2): all H frames real
    assert step.history_mask.tolist() == [True, True, True]
    assert np.all(step.history_latents[0] == 1)  # latents[0]
    assert np.all(step.history_latents[1] == 2)  # latents[1]
    assert np.all(step.history_latents[2] == 3)  # latents[2] == z_t
    assert np.all(step.z_t == 3) and np.all(step.z_next == 4)
    assert step.action_id == 3  # ep0 actions = [1,2,3,0]; actions[2] = 3


def test_index_routes_to_correct_episode(tmp_path) -> None:
    ds = CachedLatentDataset(_write_tiny_cache(tmp_path))
    step, _ = ds[3]  # (ep1, t=0): ep1 latents base 10
    assert step.history_mask.tolist() == [False, False, True]
    assert np.all(step.z_t == 10) and np.all(step.z_next == 11)
    step4, _ = ds[4]  # (ep1, t=1)
    assert step4.history_mask.tolist() == [False, True, True]
    assert np.all(step4.z_t == 11) and np.all(step4.z_next == 12)


def test_history_must_match_arch_locked_contract(tmp_path) -> None:
    # StepSample fixes the history window at HISTORY; a divergent override fails fast at construction
    # (clear message) instead of erroring deep in __getitem__.
    with pytest.raises(ValueError, match="HISTORY"):
        CachedLatentDataset(_write_tiny_cache(tmp_path), history=HISTORY - 1)


def test_inspect_cache_runs(tmp_path, capsys) -> None:
    rc = inspect_cache(_write_tiny_cache(tmp_path), n=3)
    out = capsys.readouterr().out
    assert rc == 0
    assert "2 episodes, 5 transitions" in out


@pytest.mark.torch
def test_plugs_into_torch_dataloader(tmp_path) -> None:
    import torch
    from torch.utils.data import DataLoader

    ds = CachedLatentDataset(_write_tiny_cache(tmp_path))

    def collate(batch):
        z = torch.stack([torch.from_numpy(s.z_t.astype(np.float32)) for s, _ in batch])
        wp = torch.stack([torch.from_numpy(o.waypoint_4dof.astype(np.float32)) for _, o in batch])
        return z, wp

    loader = DataLoader(ds, batch_size=2, collate_fn=collate)
    z, wp = next(iter(loader))
    assert z.shape == (2, PATCH_TOKENS, EMBED_DIM)
    assert wp.shape == (2, DOF)
