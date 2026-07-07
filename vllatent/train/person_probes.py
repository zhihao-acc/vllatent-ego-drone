"""B3 Stage-0 person probes and local gate metrics.

The helpers are numpy-only so they can run as cheap fixture tests and as a
local cache diagnostic before the B3.5 world-model architecture is built.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vllatent.data.sports_loader import SportsTrainingDataset, clip_source
from vllatent.ingest.person_tracking import person_state_from_bbox, person_tracks_from_cache

SPATIAL_PROJECTION_DIM = 8
SPATIAL_PROJECTION_SEED = 1729


@dataclass(frozen=True, eq=False)
class RidgeModel:
    weights: np.ndarray
    bias: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float32)
        x_std = (x - self.feature_mean) / self.feature_std
        return x_std @ self.weights + self.bias


@dataclass(frozen=True, eq=False)
class FrameProbeExamples:
    features: np.ndarray
    visible: np.ndarray
    person_state: np.ndarray
    sources: np.ndarray
    state_visible: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = self.features.shape[0]
        if self.features.ndim != 2:
            raise ValueError(f"features: expected 2D, got {self.features.shape}")
        if self.visible.shape != (n,):
            raise ValueError(f"visible: expected {(n,)}, got {self.visible.shape}")
        state_visible = self.visible if self.state_visible is None else np.asarray(self.state_visible).astype(np.bool_)
        if state_visible.shape != (n,):
            raise ValueError(f"state_visible: expected {(n,)}, got {state_visible.shape}")
        if self.person_state.shape != (n, 4):
            raise ValueError(f"person_state: expected {(n, 4)}, got {self.person_state.shape}")
        if self.sources.shape != (n,):
            raise ValueError(f"sources: expected {(n,)}, got {self.sources.shape}")
        object.__setattr__(self, "visible", np.asarray(self.visible).astype(np.bool_, copy=False))
        object.__setattr__(self, "state_visible", state_visible.astype(np.bool_, copy=False))


@dataclass(frozen=True, eq=False)
class TokenProbeExamples:
    token_features: np.ndarray
    visible: np.ndarray
    person_state: np.ndarray
    sources: np.ndarray
    state_visible: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = self.token_features.shape[0]
        if self.token_features.ndim != 3:
            raise ValueError(f"token_features: expected 3D, got {self.token_features.shape}")
        if self.visible.shape != (n,):
            raise ValueError(f"visible: expected {(n,)}, got {self.visible.shape}")
        state_visible = self.visible if self.state_visible is None else np.asarray(self.state_visible).astype(np.bool_)
        if state_visible.shape != (n,):
            raise ValueError(f"state_visible: expected {(n,)}, got {state_visible.shape}")
        if self.person_state.shape != (n, 4):
            raise ValueError(f"person_state: expected {(n, 4)}, got {self.person_state.shape}")
        if self.sources.shape != (n,):
            raise ValueError(f"sources: expected {(n,)}, got {self.sources.shape}")
        object.__setattr__(self, "visible", np.asarray(self.visible).astype(np.bool_, copy=False))
        object.__setattr__(self, "state_visible", state_visible.astype(np.bool_, copy=False))


@dataclass(frozen=True, eq=False)
class WindowProbeExamples:
    latent_features: np.ndarray
    current_state: np.ndarray
    planned_actions: np.ndarray
    target_state: np.ndarray
    target_visible: np.ndarray
    sources: np.ndarray

    def __post_init__(self) -> None:
        n = self.latent_features.shape[0]
        if self.latent_features.ndim != 2:
            raise ValueError(f"latent_features: expected 2D, got {self.latent_features.shape}")
        if self.current_state.shape != (n, 4):
            raise ValueError(f"current_state: expected {(n, 4)}, got {self.current_state.shape}")
        if self.planned_actions.ndim != 3 or self.planned_actions.shape[0] != n:
            raise ValueError(f"planned_actions: expected (N,T,6), got {self.planned_actions.shape}")
        if self.target_state.shape[:2] != self.planned_actions.shape[:2] or self.target_state.shape[2] != 4:
            raise ValueError(f"target_state: expected (N,T,4), got {self.target_state.shape}")
        if self.target_visible.shape != self.target_state.shape[:2]:
            raise ValueError(f"target_visible: expected {self.target_state.shape[:2]}, got {self.target_visible.shape}")
        if self.sources.shape != (n,):
            raise ValueError(f"sources: expected {(n,)}, got {self.sources.shape}")


@dataclass(frozen=True)
class Stage0ProbeMetrics:
    n_train: int
    n_val: int
    n_val_visible: int
    presence_auroc: float
    center_l2_error: float
    center_l1_error: float
    log_height_mae: float
    train_presence_auroc: float = float("nan")
    train_center_l2_error: float = float("nan")
    train_log_height_mae: float = float("nan")
    per_source: dict[str, dict[str, float]] | None = None
    n_train_presence_visible: int = 0
    n_val_presence_visible: int = 0
    presence_label: str = "person_visible"
    state_label: str = "person_state_valid"


@dataclass(frozen=True)
class K1CausalityMetrics:
    n_train: int
    n_val: int
    n_target_values: int
    zero_mse: float
    plan_only_mse: float
    plan_only_r2: float


@dataclass(frozen=True)
class K2PredictorMetrics:
    n_train: int
    n_val: int
    n_target_values: int
    persistence_mse: float
    conditioned_mse: float
    improvement_frac: float
    persistence_delta_mse: float = float("nan")
    conditioned_delta_mse: float = float("nan")
    delta_improvement_frac: float = float("nan")
    n_delta_target_values: int = 0


@dataclass(frozen=True)
class Stage0GateThresholds:
    presence_auroc_min: float = 0.60
    center_l2_error_max: float = 0.14
    center_l1_error_max: float = 0.10
    log_height_mae_max: float = 0.25
    k1_plan_only_r2_max: float = 0.05
    k2_delta_improvement_min: float = 0.0


@dataclass(frozen=True)
class Stage0GateDecision:
    g0_pass: bool
    k1_pass: bool
    k2_pass: bool
    passed: bool
    failures: tuple[str, ...]


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Return AUROC with average ranks for ties."""
    y = np.asarray(labels).astype(np.bool_)
    s = np.asarray(scores, dtype=np.float64)
    if y.shape != s.shape:
        raise ValueError(f"labels/scores shape mismatch: {y.shape} vs {s.shape}")
    pos = int(y.sum())
    neg = int((~y).sum())
    if pos == 0 or neg == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_rank
        i = j

    rank_sum_pos = float(ranks[y].sum())
    return (rank_sum_pos - pos * (pos + 1) / 2.0) / float(pos * neg)


def source_split_masks(
    sources: np.ndarray,
    *,
    val_frac: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split examples by source id, never by subclip/window."""
    src = np.asarray(sources)
    unique = np.array(sorted({str(s) for s in src.tolist()}))
    if len(unique) < 2:
        raise ValueError("source split requires at least two source ids")
    if not (0.0 < val_frac < 1.0):
        raise ValueError(f"val_frac must be in (0,1), got {val_frac}")
    n_val = min(max(1, round(float(val_frac) * len(unique))), len(unique) - 1)
    order = np.random.default_rng(seed).permutation(len(unique))
    val_sources = set(unique[order[:n_val]].tolist())
    val_mask = np.array([str(s) in val_sources for s in src], dtype=np.bool_)
    train_mask = ~val_mask
    if not np.any(train_mask) or not np.any(val_mask):
        raise ValueError("source split produced an empty train or val set")
    return train_mask, val_mask


def source_split_masks_with_label_support(
    sources: np.ndarray,
    train_labels: np.ndarray,
    val_labels: np.ndarray | None = None,
    *,
    val_frac: float = 0.2,
    seed: int = 42,
    max_retries: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Find a source split with at least one labeled train and val example."""
    train_ok = np.asarray(train_labels).astype(np.bool_)
    val_ok = train_ok if val_labels is None else np.asarray(val_labels).astype(np.bool_)
    if train_ok.shape != np.asarray(sources).shape or val_ok.shape != np.asarray(sources).shape:
        raise ValueError("label masks must match sources shape")
    for offset in range(max_retries):
        train_mask, val_mask = source_split_masks(sources, val_frac=val_frac, seed=seed + offset)
        if np.any(train_mask & train_ok) and np.any(val_mask & val_ok):
            return train_mask, val_mask
    raise ValueError("source split could not produce labeled train and val sets")


def fit_ridge(features: np.ndarray, targets: np.ndarray, *, l2: float = 1e-3) -> RidgeModel:
    """Fit a standardized ridge regressor with an unregularized intercept."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"features: expected 2D, got {x.shape}")
    if y.ndim == 1:
        y = y[:, None]
    if y.ndim != 2 or y.shape[0] != x.shape[0]:
        raise ValueError(f"targets: expected (N,K), got {y.shape} for features {x.shape}")
    if x.shape[0] == 0:
        raise ValueError("features: expected at least one row")
    if l2 < 0.0:
        raise ValueError(f"l2 must be >= 0, got {l2}")

    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    x_std = (x - mean) / std
    x_aug = np.concatenate([x_std, np.ones((x_std.shape[0], 1), dtype=np.float32)], axis=1)
    reg = np.eye(x_aug.shape[1], dtype=np.float64) * float(l2)
    reg[-1, -1] = 0.0
    xtx = x_aug.T.astype(np.float64) @ x_aug.astype(np.float64) + reg
    xty = x_aug.T.astype(np.float64) @ y.astype(np.float64)
    try:
        coef = np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(xtx) @ xty
    return RidgeModel(
        weights=coef[:-1].astype(np.float32),
        bias=coef[-1].astype(np.float32),
        feature_mean=mean,
        feature_std=std,
    )


def latent_spatial_features(
    latents: np.ndarray,
    *,
    n_projections: int = SPATIAL_PROJECTION_DIM,
    seed: int = SPATIAL_PROJECTION_SEED,
) -> np.ndarray:
    """Convert patch tokens to compact spatial-moment features.

    Mean pooling destroys location. These deterministic random-channel maps keep
    only low-order moments over the 14x14 patch grid, which is enough for cheap
    Stage-0 probes while avoiding a huge flattened-token ridge system.
    """
    arr = np.asarray(latents, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"latents: expected (N,P,D), got {arr.shape}")
    n, patches, dim = arr.shape
    grid = int(round(patches ** 0.5))
    if grid * grid != patches:
        raise ValueError(f"latents: expected square patch grid, got P={patches}")
    if n_projections <= 0:
        raise ValueError(f"n_projections must be > 0, got {n_projections}")

    rng = np.random.default_rng(seed)
    proj = rng.normal(0.0, 1.0 / np.sqrt(dim), size=(dim, n_projections)).astype(np.float32)
    maps = arr.reshape(n, patches, dim) @ proj
    abs_maps = np.abs(maps)

    coords = np.linspace(0.0, 1.0, grid, dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    x = xx.reshape(patches, 1)
    y = yy.reshape(patches, 1)
    denom = abs_maps.sum(axis=1) + 1e-6
    x_mean = (abs_maps * x[None, :, :]).sum(axis=1) / denom
    y_mean = (abs_maps * y[None, :, :]).sum(axis=1) / denom
    x_var = (abs_maps * (x[None, :, :] - x_mean[:, None, :]) ** 2).sum(axis=1) / denom
    y_var = (abs_maps * (y[None, :, :] - y_mean[:, None, :]) ** 2).sum(axis=1) / denom
    features = np.concatenate(
        [
            maps.mean(axis=1),
            maps.std(axis=1),
            abs_maps.mean(axis=1),
            x_mean,
            y_mean,
            np.sqrt(np.maximum(x_var, 0.0)),
            np.sqrt(np.maximum(y_var, 0.0)),
        ],
        axis=1,
    )
    return features.astype(np.float32, copy=False)


def latent_token_features(
    latents: np.ndarray,
    *,
    projection_dim: int = 64,
    seed: int = SPATIAL_PROJECTION_SEED,
) -> np.ndarray:
    """Project patch tokens and append explicit 14x14 patch coordinates."""
    arr = np.asarray(latents, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"latents: expected (N,P,D), got {arr.shape}")
    n, patches, dim = arr.shape
    grid = int(round(patches ** 0.5))
    if grid * grid != patches:
        raise ValueError(f"latents: expected square patch grid, got P={patches}")
    if projection_dim <= 0:
        raise ValueError(f"projection_dim must be > 0, got {projection_dim}")
    rng = np.random.default_rng(seed)
    proj = rng.normal(0.0, 1.0 / np.sqrt(dim), size=(dim, projection_dim)).astype(np.float32)
    tokens = arr @ proj
    coords = np.linspace(0.0, 1.0, grid, dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    xy = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.float32)
    xy = np.broadcast_to(xy[None, :, :], (n, patches, 2))
    return np.concatenate([tokens, xy], axis=2).astype(np.float32, copy=False)


def collect_frame_probe_examples(
    cache_dir: str | Path,
    *,
    clip_ids: list[str] | None = None,
    limit_clips: int | None = None,
    max_frames_per_clip: int | None = None,
    n_spatial_projections: int = SPATIAL_PROJECTION_DIM,
) -> FrameProbeExamples:
    """Collect pooled-frame DINO features and current-frame person labels."""
    root = Path(cache_dir)
    paths = [root / f"{cid}.npz" for cid in clip_ids] if clip_ids is not None else sorted(root.glob("*.npz"))
    paths = [p for p in paths if p.exists()]
    if limit_clips is not None:
        paths = paths[:limit_clips]
    if not paths:
        raise ValueError(f"No .npz clips found in {cache_dir}")

    features: list[np.ndarray] = []
    visible: list[np.ndarray] = []
    state_visible: list[np.ndarray] = []
    states: list[np.ndarray] = []
    sources: list[np.ndarray] = []
    for path in paths:
        with np.load(str(path)) as data:
            clip = {k: data[k] for k in data.files}
        latents = np.asarray(clip["latents"])
        n = latents.shape[0]
        if max_frames_per_clip is not None and n > max_frames_per_clip:
            idx = np.linspace(0, n - 1, max_frames_per_clip, dtype=np.int64)
        else:
            idx = np.arange(n, dtype=np.int64)
        tracks = person_tracks_from_cache(clip)
        state = person_state_from_bbox(tracks.person_bbox, tracks.person_state_valid)
        features.append(latent_spatial_features(latents[idx], n_projections=n_spatial_projections))
        visible.append(tracks.person_visible[idx].astype(np.bool_))
        state_visible.append(tracks.person_state_valid[idx].astype(np.bool_))
        states.append(state[idx].astype(np.float32, copy=False))
        sources.append(np.full(len(idx), clip_source(path.stem), dtype=object))

    return FrameProbeExamples(
        features=np.concatenate(features, axis=0).astype(np.float32, copy=False),
        visible=np.concatenate(visible, axis=0).astype(np.bool_, copy=False),
        person_state=np.concatenate(states, axis=0).astype(np.float32, copy=False),
        sources=np.concatenate(sources, axis=0),
        state_visible=np.concatenate(state_visible, axis=0).astype(np.bool_, copy=False),
    )


def collect_token_probe_examples(
    cache_dir: str | Path,
    *,
    clip_ids: list[str] | None = None,
    limit_clips: int | None = None,
    max_frames_per_clip: int | None = None,
    projection_dim: int = 64,
) -> TokenProbeExamples:
    """Collect token-level projected DINO features and current-frame person labels."""
    root = Path(cache_dir)
    paths = [root / f"{cid}.npz" for cid in clip_ids] if clip_ids is not None else sorted(root.glob("*.npz"))
    paths = [p for p in paths if p.exists()]
    if limit_clips is not None:
        paths = paths[:limit_clips]
    if not paths:
        raise ValueError(f"No .npz clips found in {cache_dir}")

    token_features: list[np.ndarray] = []
    visible: list[np.ndarray] = []
    state_visible: list[np.ndarray] = []
    states: list[np.ndarray] = []
    sources: list[np.ndarray] = []
    for path in paths:
        with np.load(str(path)) as data:
            clip = {k: data[k] for k in data.files}
        latents = np.asarray(clip["latents"])
        n = latents.shape[0]
        if max_frames_per_clip is not None and n > max_frames_per_clip:
            idx = np.linspace(0, n - 1, max_frames_per_clip, dtype=np.int64)
        else:
            idx = np.arange(n, dtype=np.int64)
        tracks = person_tracks_from_cache(clip)
        state = person_state_from_bbox(tracks.person_bbox, tracks.person_state_valid)
        token_features.append(latent_token_features(latents[idx], projection_dim=projection_dim))
        visible.append(tracks.person_visible[idx].astype(np.bool_))
        state_visible.append(tracks.person_state_valid[idx].astype(np.bool_))
        states.append(state[idx].astype(np.float32, copy=False))
        sources.append(np.full(len(idx), clip_source(path.stem), dtype=object))

    return TokenProbeExamples(
        token_features=np.concatenate(token_features, axis=0).astype(np.float32, copy=False),
        visible=np.concatenate(visible, axis=0).astype(np.bool_, copy=False),
        person_state=np.concatenate(states, axis=0).astype(np.float32, copy=False),
        sources=np.concatenate(sources, axis=0),
        state_visible=np.concatenate(state_visible, axis=0).astype(np.bool_, copy=False),
    )


def collect_window_probe_examples(
    dataset: SportsTrainingDataset,
    *,
    limit_samples: int | None = None,
    n_spatial_projections: int = SPATIAL_PROJECTION_DIM,
) -> WindowProbeExamples:
    """Collect B3 window examples from ``SportsTrainingDataset``."""
    n = len(dataset) if limit_samples is None else min(len(dataset), limit_samples)
    if n <= 0:
        raise ValueError("dataset produced no samples")
    latent_features = []
    current_states = []
    planned_actions = []
    target_states = []
    target_visible = []
    sources = []
    for idx in range(n):
        sample = dataset[idx]
        latent_features.append(latent_spatial_features(sample.z_t[None], n_projections=n_spatial_projections)[0])
        current = person_state_from_bbox(
            sample.history_person_bbox[-1:],
            sample.history_person_state_valid[-1:],
        )[0]
        current_states.append(current)
        planned_actions.append(sample.planned_actions)
        target_states.append(sample.person_state_target)
        target_visible.append(sample.target_person_state_valid)
        sources.append(dataset.sample_sources[idx])

    return WindowProbeExamples(
        latent_features=np.stack(latent_features).astype(np.float32, copy=False),
        current_state=np.stack(current_states).astype(np.float32, copy=False),
        planned_actions=np.stack(planned_actions).astype(np.float32, copy=False),
        target_state=np.stack(target_states).astype(np.float32, copy=False),
        target_visible=np.stack(target_visible).astype(np.bool_, copy=False),
        sources=np.asarray(sources, dtype=object),
    )


def _stage0_metrics_from_predictions(
    *,
    visible: np.ndarray,
    state_visible: np.ndarray,
    true_state: np.ndarray,
    presence_scores: np.ndarray,
    pred_state: np.ndarray,
) -> tuple[float, float, float, float, int, int]:
    auc = binary_auroc(visible, presence_scores)
    state_mask = np.asarray(state_visible).astype(np.bool_)
    if not np.any(state_mask):
        return float(auc), float("nan"), float("nan"), float("nan"), int(np.sum(visible)), 0
    center_delta = pred_state[state_mask, :2] - true_state[state_mask, :2]
    center_l2 = float(np.mean(np.linalg.norm(center_delta.astype(np.float64), axis=1)))
    center_l1 = float(np.mean(np.abs(center_delta.astype(np.float64))))
    log_h = float(np.mean(np.abs((pred_state[state_mask, 2] - true_state[state_mask, 2]).astype(np.float64))))
    return float(auc), center_l2, center_l1, log_h, int(np.sum(visible)), int(state_mask.sum())


def _stage0_per_source_metrics(
    sources: np.ndarray,
    visible: np.ndarray,
    state_visible: np.ndarray,
    true_state: np.ndarray,
    presence_scores: np.ndarray,
    pred_state: np.ndarray,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for src in sorted({str(s) for s in sources.tolist()}):
        mask = np.array([str(s) == src for s in sources], dtype=np.bool_)
        auc, center_l2, center_l1, log_h, n_presence_visible, n_state_visible = _stage0_metrics_from_predictions(
            visible=visible[mask],
            state_visible=state_visible[mask],
            true_state=true_state[mask],
            presence_scores=presence_scores[mask],
            pred_state=pred_state[mask],
        )
        out[src] = {
            "n": float(mask.sum()),
            "n_visible": float(n_state_visible),
            "n_presence_visible": float(n_presence_visible),
            "presence_auroc": auc,
            "center_l2_error": center_l2,
            "center_l1_error": center_l1,
            "log_height_mae": log_h,
        }
    return out


def fit_stage0_probes(
    examples: FrameProbeExamples,
    *,
    val_frac: float = 0.2,
    seed: int = 42,
    l2: float = 1e-3,
    split_retries: int = 50,
) -> Stage0ProbeMetrics:
    """Train held-out-source linear probes for presence, center, and log-height."""
    train_mask, val_mask = source_split_masks_with_label_support(
        examples.sources,
        examples.state_visible,
        val_frac=val_frac,
        seed=seed,
        max_retries=split_retries,
    )
    presence_model = fit_ridge(examples.features[train_mask], examples.visible[train_mask].astype(np.float32), l2=l2)
    presence_scores = presence_model.predict(examples.features[val_mask]).reshape(-1)
    auc = binary_auroc(examples.visible[val_mask], presence_scores)

    train_state_visible = train_mask & examples.state_visible
    val_state_visible = val_mask & examples.state_visible
    if not np.any(train_state_visible):
        raise ValueError("no visible person labels in train split")
    if not np.any(val_state_visible):
        raise ValueError("no visible person labels in val split")
    state_model = fit_ridge(examples.features[train_state_visible], examples.person_state[train_state_visible, :3], l2=l2)
    pred_state = state_model.predict(examples.features[val_state_visible])
    true_state = examples.person_state[val_state_visible, :3]
    center_delta = pred_state[:, :2] - true_state[:, :2]
    center_l2 = float(np.mean(np.linalg.norm(center_delta.astype(np.float64), axis=1)))
    center_l1 = float(np.mean(np.abs(center_delta.astype(np.float64))))
    log_h = float(np.mean(np.abs((pred_state[:, 2] - true_state[:, 2]).astype(np.float64))))
    train_presence_scores = presence_model.predict(examples.features[train_mask]).reshape(-1)
    train_pred_state = state_model.predict(examples.features[train_state_visible])
    train_true_state = examples.person_state[train_state_visible, :3]
    train_center_delta = train_pred_state[:, :2] - train_true_state[:, :2]
    train_auc = binary_auroc(examples.visible[train_mask], train_presence_scores)
    train_center_l2 = float(np.mean(np.linalg.norm(train_center_delta.astype(np.float64), axis=1)))
    train_log_h = float(np.mean(np.abs((train_pred_state[:, 2] - train_true_state[:, 2]).astype(np.float64))))
    full_presence_scores = presence_model.predict(examples.features).reshape(-1)
    full_pred_state = state_model.predict(examples.features).astype(np.float32, copy=False)

    return Stage0ProbeMetrics(
        n_train=int(train_mask.sum()),
        n_val=int(val_mask.sum()),
        n_val_visible=int(val_state_visible.sum()),
        presence_auroc=float(auc),
        center_l2_error=center_l2,
        center_l1_error=center_l1,
        log_height_mae=log_h,
        train_presence_auroc=float(train_auc),
        train_center_l2_error=train_center_l2,
        train_log_height_mae=train_log_h,
        per_source=_stage0_per_source_metrics(
            examples.sources[val_mask],
            examples.visible[val_mask],
            examples.state_visible[val_mask],
            examples.person_state[val_mask, :3],
            full_presence_scores[val_mask],
            full_pred_state[val_mask],
        ),
        n_train_presence_visible=int(np.sum(examples.visible[train_mask])),
        n_val_presence_visible=int(np.sum(examples.visible[val_mask])),
    )


def fit_stage0_token_probe(
    examples: TokenProbeExamples,
    *,
    val_frac: float = 0.2,
    seed: int = 42,
    split_retries: int = 50,
    hidden_dim: int = 128,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    state_loss_weight: float = 5.0,
    device: str = "auto",
) -> Stage0ProbeMetrics:
    """Train a bounded token-level G0 probe over patch tokens plus coordinates."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    if hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")
    if epochs <= 0:
        raise ValueError(f"epochs must be > 0, got {epochs}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    train_mask, val_mask = source_split_masks_with_label_support(
        examples.sources,
        examples.state_visible,
        val_frac=val_frac,
        seed=seed,
        max_retries=split_retries,
    )

    class _TokenPersonProbe(nn.Module):
        def __init__(self, in_dim: int, hidden: int) -> None:
            super().__init__()
            self.token_net = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
            )
            self.score = nn.Linear(hidden, 1)
            self.presence = nn.Linear(hidden, 1)
            self.state = nn.Linear(hidden, 3)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            h = self.token_net(x)
            attn = torch.softmax(self.score(h).squeeze(-1), dim=1)
            pooled = (h * attn.unsqueeze(-1)).sum(dim=1)
            logits = self.presence(pooled).squeeze(-1)
            raw_state = self.state(pooled)
            attn_center = (x[..., -2:] * attn.unsqueeze(-1)).sum(dim=1)
            center = torch.clamp(attn_center + 0.25 * torch.tanh(raw_state[:, :2]), 0.0, 1.0)
            state = torch.cat([center, raw_state[:, 2:3]], dim=1)
            return logits, state

    torch.manual_seed(seed)
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
    x = torch.from_numpy(examples.token_features.astype(np.float32, copy=False))
    y_visible = torch.from_numpy(examples.visible.astype(np.float32))
    y_state_visible = torch.from_numpy(examples.state_visible.astype(np.float32))
    y_state = torch.from_numpy(examples.person_state[:, :3].astype(np.float32, copy=False))
    train_idx = torch.from_numpy(np.flatnonzero(train_mask).astype(np.int64))
    val_idx = torch.from_numpy(np.flatnonzero(val_mask).astype(np.int64))

    train_dataset = TensorDataset(x[train_idx], y_visible[train_idx], y_state_visible[train_idx], y_state[train_idx])
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    model = _TokenPersonProbe(x.shape[-1], hidden_dim).to(dev)
    nn.init.zeros_(model.state.weight)
    nn.init.zeros_(model.state.bias)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos = float(examples.visible[train_mask].sum())
    neg = float(train_mask.sum() - pos)
    pos_weight = torch.tensor([max(neg / max(pos, 1.0), 1.0)], device=dev)

    model.train()
    for _ in range(epochs):
        for xb, vb, svb, sb in loader:
            xb = xb.to(dev)
            vb = vb.to(dev)
            svb = svb.to(dev)
            sb = sb.to(dev)
            logits, pred = model(xb)
            loss_presence = F.binary_cross_entropy_with_logits(logits, vb, pos_weight=pos_weight)
            state_visible = svb > 0.5
            if torch.any(state_visible):
                loss_state = F.smooth_l1_loss(pred[state_visible], sb[state_visible])
            else:
                loss_state = pred.sum() * 0.0
            loss = loss_presence + state_loss_weight * loss_state
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    def _predict(indices: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        scores: list[np.ndarray] = []
        states: list[np.ndarray] = []
        eval_dataset = TensorDataset(x[indices])
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
        model.eval()
        with torch.no_grad():
            for (xb,) in eval_loader:
                logits, pred = model(xb.to(dev))
                scores.append(logits.detach().cpu().numpy())
                states.append(pred.detach().cpu().numpy())
        return np.concatenate(scores, axis=0), np.concatenate(states, axis=0)

    train_scores, train_pred = _predict(train_idx)
    val_scores, val_pred = _predict(val_idx)
    train_auc, train_center_l2, _, train_log_h, _, _ = _stage0_metrics_from_predictions(
        visible=examples.visible[train_mask],
        state_visible=examples.state_visible[train_mask],
        true_state=examples.person_state[train_mask, :3],
        presence_scores=train_scores,
        pred_state=train_pred,
    )
    val_auc, val_center_l2, val_center_l1, val_log_h, n_val_presence_visible, n_val_visible = (
        _stage0_metrics_from_predictions(
            visible=examples.visible[val_mask],
            state_visible=examples.state_visible[val_mask],
            true_state=examples.person_state[val_mask, :3],
            presence_scores=val_scores,
            pred_state=val_pred,
        )
    )
    return Stage0ProbeMetrics(
        n_train=int(train_mask.sum()),
        n_val=int(val_mask.sum()),
        n_val_visible=n_val_visible,
        presence_auroc=val_auc,
        center_l2_error=val_center_l2,
        center_l1_error=val_center_l1,
        log_height_mae=val_log_h,
        train_presence_auroc=train_auc,
        train_center_l2_error=train_center_l2,
        train_log_height_mae=train_log_h,
        per_source=_stage0_per_source_metrics(
            examples.sources[val_mask],
            examples.visible[val_mask],
            examples.state_visible[val_mask],
            examples.person_state[val_mask, :3],
            val_scores,
            val_pred,
        ),
        n_train_presence_visible=int(np.sum(examples.visible[train_mask])),
        n_val_presence_visible=n_val_presence_visible,
    )


def _masked_delta_mse(pred_delta: np.ndarray, true_delta: np.ndarray, visible: np.ndarray) -> tuple[float, int]:
    mask = np.repeat(visible[..., None], 3, axis=2).reshape(-1)
    diff = (pred_delta - true_delta).reshape(-1)
    n = int(mask.sum())
    if n == 0:
        return float("nan"), 0
    return float(np.mean((diff[mask] ** 2).astype(np.float64))), n


def _person_state_mse(pred: np.ndarray, target: np.ndarray, visible: np.ndarray) -> tuple[float, int]:
    spatial_mask = np.repeat(visible[..., None], 3, axis=2)
    spatial_sq = (pred[..., :3] - target[..., :3]) ** 2
    vis_sq = (pred[..., 3] - target[..., 3]) ** 2
    numerator = float(spatial_sq[spatial_mask].astype(np.float64).sum() + vis_sq.astype(np.float64).sum())
    denom = int(spatial_mask.sum() + vis_sq.size)
    if denom == 0:
        return float("nan"), 0
    return numerator / float(denom), denom


def run_k1_plan_only_causality(
    examples: WindowProbeExamples,
    *,
    val_frac: float = 0.2,
    seed: int = 42,
    l2: float = 1e-3,
    split_retries: int = 50,
) -> K1CausalityMetrics:
    """Measure how much plan-only input predicts person motion over persistence."""
    row_has_target = examples.target_visible.any(axis=1)
    train_mask, val_mask = source_split_masks_with_label_support(
        examples.sources,
        row_has_target,
        val_frac=val_frac,
        seed=seed,
        max_retries=split_retries,
    )
    x = examples.planned_actions.reshape(examples.planned_actions.shape[0], -1)
    y = (examples.target_state[..., :3] - examples.current_state[:, None, :3]).reshape(x.shape[0], -1)
    if not np.any(train_mask & row_has_target):
        raise ValueError("K1 has no train windows with visible future person labels")
    model = fit_ridge(x[train_mask & row_has_target], y[train_mask & row_has_target], l2=l2)

    pred = model.predict(x[val_mask]).reshape(examples.target_state[val_mask, :, :3].shape)
    true = examples.target_state[val_mask, :, :3] - examples.current_state[val_mask, None, :3]
    visible = examples.target_visible[val_mask]
    zero = np.zeros_like(true)
    zero_mse, n_values = _masked_delta_mse(zero, true, visible)
    model_mse, _ = _masked_delta_mse(pred, true, visible)
    r2 = 1.0 - model_mse / zero_mse if np.isfinite(zero_mse) and zero_mse > 0.0 else float("nan")
    return K1CausalityMetrics(
        n_train=int((train_mask & row_has_target).sum()),
        n_val=int(val_mask.sum()),
        n_target_values=n_values,
        zero_mse=float(zero_mse),
        plan_only_mse=float(model_mse),
        plan_only_r2=float(r2),
    )


def run_k2_conditioned_predictor(
    examples: WindowProbeExamples,
    *,
    val_frac: float = 0.2,
    seed: int = 42,
    l2: float = 1e-3,
    split_retries: int = 50,
) -> K2PredictorMetrics:
    """Tiny linear conditioned person-state predictor versus persistence."""
    row_has_target = examples.target_visible.any(axis=1)
    train_mask, val_mask = source_split_masks_with_label_support(
        examples.sources,
        row_has_target,
        val_frac=val_frac,
        seed=seed,
        max_retries=split_retries,
    )
    x = np.concatenate(
        [
            examples.latent_features,
            examples.current_state,
            examples.planned_actions.reshape(examples.planned_actions.shape[0], -1),
        ],
        axis=1,
    )
    y = examples.target_state.reshape(examples.target_state.shape[0], -1)
    if not np.any(train_mask & row_has_target):
        raise ValueError("K2 has no train windows with visible future person labels")
    model = fit_ridge(x[train_mask & row_has_target], y[train_mask & row_has_target], l2=l2)

    pred = model.predict(x[val_mask]).reshape(examples.target_state[val_mask].shape)
    target = examples.target_state[val_mask]
    visible = examples.target_visible[val_mask]
    persistence = np.repeat(
        examples.current_state[val_mask, None, :],
        examples.target_state.shape[1],
        axis=1,
    )
    persistence_mse, n_values = _person_state_mse(persistence, target, visible)
    conditioned_mse, _ = _person_state_mse(pred, target, visible)
    improvement = (
        (persistence_mse - conditioned_mse) / persistence_mse
        if np.isfinite(persistence_mse) and persistence_mse > 0.0
        else float("nan")
    )
    true_delta = target[..., :3] - examples.current_state[val_mask, None, :3]
    conditioned_delta = pred[..., :3] - examples.current_state[val_mask, None, :3]
    zero_delta = np.zeros_like(true_delta, dtype=np.float32)
    persistence_delta_mse, n_delta_values = _masked_delta_mse(zero_delta, true_delta, visible)
    conditioned_delta_mse, _ = _masked_delta_mse(conditioned_delta, true_delta, visible)
    delta_improvement = (
        (persistence_delta_mse - conditioned_delta_mse) / persistence_delta_mse
        if np.isfinite(persistence_delta_mse) and persistence_delta_mse > 0.0
        else float("nan")
    )
    return K2PredictorMetrics(
        n_train=int((train_mask & row_has_target).sum()),
        n_val=int(val_mask.sum()),
        n_target_values=n_values,
        persistence_mse=float(persistence_mse),
        conditioned_mse=float(conditioned_mse),
        improvement_frac=float(improvement),
        persistence_delta_mse=float(persistence_delta_mse),
        conditioned_delta_mse=float(conditioned_delta_mse),
        delta_improvement_frac=float(delta_improvement),
        n_delta_target_values=n_delta_values,
    )


def evaluate_stage0_gates(
    stage0: Stage0ProbeMetrics,
    k1: K1CausalityMetrics,
    k2: K2PredictorMetrics,
    thresholds: Stage0GateThresholds | None = None,
) -> Stage0GateDecision:
    """Evaluate B3.4 local gates from measured metrics."""
    th = thresholds or Stage0GateThresholds()
    failures: list[str] = []
    g0_pass = bool(
        np.isfinite(stage0.presence_auroc)
        and stage0.presence_auroc >= th.presence_auroc_min
        and np.isfinite(stage0.center_l2_error)
        and stage0.center_l2_error <= th.center_l2_error_max
        and np.isfinite(stage0.center_l1_error)
        and stage0.center_l1_error <= th.center_l1_error_max
        and np.isfinite(stage0.log_height_mae)
        and stage0.log_height_mae <= th.log_height_mae_max
    )
    if not g0_pass:
        failures.append("G0")
    k1_pass = bool(np.isfinite(k1.plan_only_r2) and k1.plan_only_r2 <= th.k1_plan_only_r2_max)
    if not k1_pass:
        failures.append("K1")
    k2_pass = bool(
        np.isfinite(k2.delta_improvement_frac)
        and k2.delta_improvement_frac >= th.k2_delta_improvement_min
    )
    if not k2_pass:
        failures.append("K2")
    return Stage0GateDecision(
        g0_pass=g0_pass,
        k1_pass=k1_pass,
        k2_pass=k2_pass,
        passed=g0_pass and k1_pass and k2_pass,
        failures=tuple(failures),
    )


__all__ = [
    "FrameProbeExamples",
    "K1CausalityMetrics",
    "K2PredictorMetrics",
    "RidgeModel",
    "Stage0GateDecision",
    "Stage0GateThresholds",
    "Stage0ProbeMetrics",
    "TokenProbeExamples",
    "WindowProbeExamples",
    "binary_auroc",
    "collect_frame_probe_examples",
    "collect_token_probe_examples",
    "collect_window_probe_examples",
    "evaluate_stage0_gates",
    "fit_ridge",
    "fit_stage0_probes",
    "fit_stage0_token_probe",
    "latent_spatial_features",
    "latent_token_features",
    "run_k1_plan_only_causality",
    "run_k2_conditioned_predictor",
    "source_split_masks",
    "source_split_masks_with_label_support",
]
