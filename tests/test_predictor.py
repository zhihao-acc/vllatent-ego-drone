"""Tests for block-causal ViT latent predictor (B1.15)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vllatent.model.predictor import LatentPredictor  # noqa: E402
from vllatent.schemas import DOF, EMBED_DIM, HISTORY, HORIZON, PATCH_TOKENS  # noqa: E402


def _make_inputs(B: int = 2, device: str = "cpu"):
    history = torch.randn(B, HISTORY, PATCH_TOKENS, EMBED_DIM, device=device)
    z_t = torch.randn(B, PATCH_TOKENS, EMBED_DIM, device=device)
    action = torch.randn(B, DOF, device=device)
    dt = torch.full((B, HORIZON), 0.2, device=device)
    return history, z_t, action, dt


@pytest.mark.torch
class TestLatentPredictor:
    def test_output_shape(self) -> None:
        model = LatentPredictor(dim=EMBED_DIM, depth=2, heads=12)
        history, z_t, action, dt = _make_inputs(B=2)
        out = model(history, z_t, action, dt)
        assert out.shape == (2, HORIZON, PATCH_TOKENS, EMBED_DIM)

    def test_output_shape_small(self) -> None:
        model = LatentPredictor(dim=384, depth=2, heads=6)
        B = 2
        history = torch.randn(B, HISTORY, PATCH_TOKENS, 384)
        z_t = torch.randn(B, PATCH_TOKENS, 384)
        action = torch.randn(B, DOF)
        dt = torch.full((B, HORIZON), 0.2)
        out = model(history, z_t, action, dt)
        assert out.shape == (2, HORIZON, PATCH_TOKENS, 384)

    def test_param_count_depth6(self) -> None:
        model = LatentPredictor(dim=EMBED_DIM, depth=6, heads=12)
        n_params = sum(p.numel() for p in model.parameters())
        assert 40_000_000 < n_params < 70_000_000

    def test_block_causal_mask_shape(self) -> None:
        model = LatentPredictor(dim=EMBED_DIM, depth=2, heads=12)
        n_frames = HISTORY + 1 + HORIZON
        mask = model._build_block_causal_mask(n_frames, torch.device("cpu"))
        total_tokens = n_frames * PATCH_TOKENS
        assert mask.shape == (total_tokens, total_tokens)

    def test_block_causal_mask_history_visible(self) -> None:
        """SDPA convention: True = CAN attend."""
        model = LatentPredictor(dim=EMBED_DIM, depth=2, heads=12)
        n_frames = HISTORY + 1 + HORIZON
        mask = model._build_block_causal_mask(n_frames, torch.device("cpu"))
        n_visible = HISTORY + 1
        for f in range(n_frames):
            for j in range(n_visible):
                r = f * PATCH_TOKENS
                c = j * PATCH_TOKENS
                assert mask[r, c], (
                    f"Frame {f} should see history frame {j}"
                )

    def test_block_causal_mask_future_blocked(self) -> None:
        """SDPA convention: False = blocked."""
        model = LatentPredictor(dim=EMBED_DIM, depth=2, heads=12)
        n_frames = HISTORY + 1 + HORIZON
        mask = model._build_block_causal_mask(n_frames, torch.device("cpu"))
        n_visible = HISTORY + 1
        for f in range(n_visible, n_frames):
            for j in range(f + 1, n_frames):
                r = f * PATCH_TOKENS
                c = j * PATCH_TOKENS
                assert not mask[r, c], (
                    f"Frame {f} should NOT see future frame {j}"
                )

    def test_action_film_changes_output(self) -> None:
        model = LatentPredictor(dim=EMBED_DIM, depth=2, heads=12)
        for film in model.action_film:
            nn = film.net[-1]
            nn.weight.data.normal_(std=0.5)
            nn.bias.data.normal_(std=0.5)
        model.eval()
        history, z_t, _, dt = _make_inputs(B=1)
        with torch.no_grad():
            action1 = torch.zeros(1, DOF)
            out1 = model(history, z_t, action1, dt)
            action2 = torch.ones(1, DOF) * 5.0
            out2 = model(history, z_t, action2, dt)
        assert not torch.allclose(out1, out2, atol=1e-6)

    def test_no_action_film_ignores_action(self) -> None:
        """use_action_film=False: action is ignored even with non-zero action_film weights."""
        model = LatentPredictor(dim=EMBED_DIM, depth=2, heads=12, use_action_film=False)
        for film in model.action_film:
            film.net[-1].weight.data.normal_(std=0.5)
            film.net[-1].bias.data.normal_(std=0.5)
        model.eval()
        history, z_t, _, dt = _make_inputs(B=1)
        with torch.no_grad():
            out1 = model(history, z_t, torch.zeros(1, DOF), dt)
            out2 = model(history, z_t, torch.ones(1, DOF) * 5.0, dt)
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_no_action_film_still_uses_dt(self) -> None:
        """dt-FiLM stays active in the action-free ablation."""
        model = LatentPredictor(dim=EMBED_DIM, depth=2, heads=12, use_action_film=False)
        for film in model.dt_film:
            film.net[-1].weight.data.normal_(std=0.5)
            film.net[-1].bias.data.normal_(std=0.5)
        model.eval()
        history, z_t, action, _ = _make_inputs(B=1)
        with torch.no_grad():
            out1 = model(history, z_t, action, torch.full((1, HORIZON), 0.2))
            out2 = model(history, z_t, action, torch.full((1, HORIZON), 2.0))
        assert not torch.allclose(out1, out2, atol=1e-6)

    def test_dt_film_changes_output(self) -> None:
        model = LatentPredictor(dim=EMBED_DIM, depth=2, heads=12)
        for film in model.dt_film:
            nn = film.net[-1]
            nn.weight.data.normal_(std=0.5)
            nn.bias.data.normal_(std=0.5)
        model.eval()
        history, z_t, action, _ = _make_inputs(B=1)
        with torch.no_grad():
            dt1 = torch.full((1, HORIZON), 0.2)
            out1 = model(history, z_t, action, dt1)
            dt2 = torch.full((1, HORIZON), 2.0)
            out2 = model(history, z_t, action, dt2)
        assert not torch.allclose(out1, out2, atol=1e-6)

    def test_dropout_active_in_training(self) -> None:
        model = LatentPredictor(dim=EMBED_DIM, depth=2, heads=12, dropout=0.5)
        model.train()
        history, z_t, action, dt = _make_inputs(B=1)
        out1 = model(history, z_t, action, dt)
        out2 = model(history, z_t, action, dt)
        assert not torch.allclose(out1, out2, atol=1e-6)

    def test_deterministic_in_eval(self) -> None:
        model = LatentPredictor(dim=EMBED_DIM, depth=2, heads=12, dropout=0.5)
        model.eval()
        history, z_t, action, dt = _make_inputs(B=1)
        with torch.no_grad():
            out1 = model(history, z_t, action, dt)
            out2 = model(history, z_t, action, dt)
        assert torch.allclose(out1, out2, atol=1e-7)
