"""Tests for B3 Stage-0 person probe helpers."""
from __future__ import annotations

import numpy as np
import pytest

from vllatent.train.person_probes import (
    FrameProbeExamples,
    TokenProbeExamples,
    WindowProbeExamples,
    binary_auroc,
    fit_ridge,
    fit_stage0_probes,
    fit_stage0_token_probe,
    latent_spatial_features,
    latent_token_features,
    run_k1_plan_only_causality,
    run_k2_conditioned_predictor,
    source_split_masks,
    source_split_masks_with_label_support,
)


def test_binary_auroc_perfect_reversed_and_tied() -> None:
    labels = np.array([False, False, True, True])
    assert binary_auroc(labels, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert binary_auroc(labels, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)
    assert binary_auroc(labels, np.ones(4)) == pytest.approx(0.5)


def test_source_split_masks_hold_out_whole_sources() -> None:
    sources = np.array(["a", "a", "b", "b", "c", "c"], dtype=object)
    train, val = source_split_masks(sources, val_frac=0.34, seed=0)
    assert train.shape == val.shape == (6,)
    assert not np.any(train & val)
    assert set(sources[val]).isdisjoint(set(sources[train]))
    assert np.any(train)
    assert np.any(val)


def test_source_split_with_label_support_retries_unlabeled_val_source() -> None:
    sources = np.array(["a", "a", "b", "b", "c", "c"], dtype=object)
    labels = np.array([True, True, False, False, True, True])
    train, val = source_split_masks_with_label_support(sources, labels, val_frac=0.34, seed=0)
    assert np.any(train & labels)
    assert np.any(val & labels)
    assert set(sources[val]).isdisjoint(set(sources[train]))


def test_fit_ridge_recovers_linear_map() -> None:
    x = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 4.0], [3.0, 8.0]], dtype=np.float32)
    y = 2.0 * x[:, :1] - 0.5 * x[:, 1:2] + 3.0
    model = fit_ridge(x, y, l2=0.0)
    np.testing.assert_allclose(model.predict(x), y, atol=1e-5)


def test_latent_spatial_features_are_deterministic() -> None:
    rng = np.random.default_rng(0)
    latents = rng.normal(size=(3, 196, 8)).astype(np.float32)
    a = latent_spatial_features(latents, n_projections=4, seed=7)
    b = latent_spatial_features(latents, n_projections=4, seed=7)
    assert a.shape == (3, 28)
    np.testing.assert_allclose(a, b, atol=1e-7)
    with pytest.raises(ValueError, match="square patch grid"):
        latent_spatial_features(rng.normal(size=(1, 10, 8)).astype(np.float32))


def test_latent_token_features_append_coordinates() -> None:
    rng = np.random.default_rng(0)
    latents = rng.normal(size=(2, 196, 8)).astype(np.float32)
    feats = latent_token_features(latents, projection_dim=4, seed=7)
    assert feats.shape == (2, 196, 6)
    np.testing.assert_allclose(feats[:, 0, -2:], np.tile([0.0, 0.0], (2, 1)), atol=1e-6)
    np.testing.assert_allclose(feats[:, -1, -2:], np.tile([1.0, 1.0], (2, 1)), atol=1e-6)


def _frame_examples() -> FrameProbeExamples:
    features = []
    visible = []
    states = []
    sources = []
    grid = np.linspace(-1.0, 1.0, 20, dtype=np.float32)
    for src in ("a", "b", "c", "d"):
        x0 = grid
        x1 = np.sin(grid * 2.0)
        x2 = np.cos(grid * 3.0)
        x3 = grid**2
        feat = np.stack([x0, x1, x2, x3], axis=1).astype(np.float32)
        vis = x0 > -0.2
        state = np.zeros((len(grid), 4), dtype=np.float32)
        state[:, 0] = 0.45 + 0.10 * x1
        state[:, 1] = 0.55 + 0.10 * x2
        state[:, 2] = -1.2 + 0.20 * x3
        state[:, 3] = vis.astype(np.float32)
        state[~vis, :3] = 0.0
        features.append(feat)
        visible.append(vis)
        states.append(state)
        sources.append(np.full(len(grid), src, dtype=object))
    return FrameProbeExamples(
        features=np.concatenate(features, axis=0),
        visible=np.concatenate(visible, axis=0),
        person_state=np.concatenate(states, axis=0),
        sources=np.concatenate(sources, axis=0),
    )


def test_fit_stage0_probes_source_held_out() -> None:
    metrics = fit_stage0_probes(_frame_examples(), val_frac=0.25, seed=1, l2=1e-4)
    assert metrics.n_train > 0
    assert metrics.n_val > 0
    assert metrics.n_val_visible > 0
    assert metrics.presence_auroc > 0.98
    assert metrics.center_l2_error < 0.02
    assert metrics.log_height_mae < 0.02


@pytest.mark.torch
def test_fit_stage0_token_probe_decodes_synthetic_tokens() -> None:
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    n_per_source = 24
    patches = 16
    tokens = []
    visible = []
    states = []
    sources = []
    coords = np.array(
        [[0.0, 0.0], [1.0 / 3.0, 0.0], [2.0 / 3.0, 0.0], [1.0, 0.0],
         [0.0, 1.0 / 3.0], [1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 1.0 / 3.0], [1.0, 1.0 / 3.0],
         [0.0, 2.0 / 3.0], [1.0 / 3.0, 2.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0], [1.0, 2.0 / 3.0],
         [0.0, 1.0], [1.0 / 3.0, 1.0], [2.0 / 3.0, 1.0], [1.0, 1.0]],
        dtype=np.float32,
    )
    for src in ("a", "b", "c", "d"):
        for i in range(n_per_source):
            is_visible = i % 3 != 0
            center = coords[(i * 5) % patches].copy()
            feat = rng.normal(0.0, 0.05, size=(patches, 6)).astype(np.float32)
            feat[:, -2:] = coords
            if is_visible:
                nearest = int((i * 5) % patches)
                feat[nearest, 0] = 4.0
                feat[nearest, 1:3] = center
            tokens.append(feat)
            visible.append(is_visible)
            states.append([center[0], center[1], -1.2, float(is_visible)] if is_visible else [0.0, 0.0, 0.0, 0.0])
            sources.append(src)
    metrics = fit_stage0_token_probe(
        TokenProbeExamples(
            token_features=np.stack(tokens).astype(np.float32),
            visible=np.asarray(visible, dtype=np.bool_),
            person_state=np.asarray(states, dtype=np.float32),
            sources=np.asarray(sources, dtype=object),
        ),
        val_frac=0.25,
        seed=1,
        hidden_dim=32,
        epochs=30,
        batch_size=16,
        lr=3e-3,
        device="cpu",
    )
    assert metrics.presence_auroc > 0.95
    assert metrics.center_l2_error < 0.12


def _window_examples() -> WindowProbeExamples:
    rng = np.random.default_rng(0)
    latent_features = []
    current_state = []
    planned_actions = []
    target_state = []
    target_visible = []
    sources = []
    horizon = 3
    for src in ("a", "b", "c", "d"):
        z = rng.normal(size=(40, 5)).astype(np.float32)
        plan = rng.normal(size=(40, horizon, 6)).astype(np.float32)
        current = np.tile(np.array([0.5, 0.5, -1.0, 1.0], dtype=np.float32), (40, 1))
        target = np.repeat(current[:, None, :], horizon, axis=1)
        for k in range(horizon):
            target[:, k, 0] += 0.08 * (k + 1) * z[:, 0]
            target[:, k, 1] += 0.05 * (k + 1) * z[:, 1]
            target[:, k, 2] += 0.03 * (k + 1) * z[:, 2]
            target[:, k, 3] = 1.0
        visible = np.ones((40, horizon), dtype=np.bool_)
        latent_features.append(z)
        current_state.append(current)
        planned_actions.append(plan)
        target_state.append(target.astype(np.float32))
        target_visible.append(visible)
        sources.append(np.full(40, src, dtype=object))
    return WindowProbeExamples(
        latent_features=np.concatenate(latent_features, axis=0),
        current_state=np.concatenate(current_state, axis=0),
        planned_actions=np.concatenate(planned_actions, axis=0),
        target_state=np.concatenate(target_state, axis=0),
        target_visible=np.concatenate(target_visible, axis=0),
        sources=np.concatenate(sources, axis=0),
    )


def test_k1_plan_only_is_near_chance_when_plan_is_independent() -> None:
    metrics = run_k1_plan_only_causality(_window_examples(), val_frac=0.25, seed=1, l2=10.0)
    assert metrics.n_target_values > 0
    assert metrics.zero_mse > 0.0
    assert metrics.plan_only_r2 < 0.05


def test_k2_conditioned_predictor_beats_persistence() -> None:
    metrics = run_k2_conditioned_predictor(_window_examples(), val_frac=0.25, seed=1, l2=1e-4)
    assert metrics.n_target_values > 0
    assert metrics.persistence_mse > 0.0
    assert metrics.conditioned_mse < metrics.persistence_mse
    assert metrics.improvement_frac > 0.9
    assert metrics.n_delta_target_values > 0
    assert metrics.conditioned_delta_mse < metrics.persistence_delta_mse
    assert metrics.delta_improvement_frac > 0.9
