"""Typed Config tests (PURE tier) — Phase-A step A5.3 (review H1/H2/L2/L3).

Pins the single-source-of-truth config: frozen defaults, from_yaml override + env expansion,
strict unknown-key rejection (so an ablation yaml typo fails fast, not silently), boundary
validation, and immutability. Mirrors how StepSample guards its boundary.
"""
from __future__ import annotations

import dataclasses

import pytest

from vllatent.config import (
    Config,
    DistillConfig,
    EncoderConfig,
    IngestConfig,
    PredictorConfig,
    TrustConfig,
)
from vllatent.schemas import HISTORY, HORIZON


def test_defaults_construct_and_match_schemas_constants() -> None:
    cfg = Config()
    # Swept defaults reference the single schemas literals (no duplication).
    assert cfg.predictor.history == HISTORY
    assert cfg.predictor.horizon == HORIZON
    assert cfg.predictor.depth == 12 and cfg.predictor.heads == 12
    # timm's NON-GATED DINOv3 ViT-B/16 re-host (Meta's gated repo rejected our access 2026-06-09).
    assert cfg.encoder.model_id == "vit_base_patch16_dinov3.lvd1689m"
    # Frozen CLIP ViT-B/32 text tower for lang_tokens (A5.13b) — NON-GATED.
    assert cfg.encoder.text_model_id == "openai/clip-vit-base-patch32"
    # Trust: sports-following pivot default (WorldVLN retired 2026-06-19).
    assert cfg.trust.disagreement_source == "vjepa_only"
    assert cfg.trust.k_rollouts == 5
    assert 0.0 <= cfg.trust.vjepa_surprise_threshold <= 1.0
    # V-JEPA-2 verifier checkpoint (A5.12) — Meta's NON-GATED ViT-L (gated:false, MIT; unlike DINOv3).
    assert cfg.trust.vjepa2_model_id == "facebook/vjepa2-vitl-fpc64-256"


def test_from_yaml_default_builds() -> None:
    # The committed configs/default.yaml must parse into a valid Config.
    cfg = Config.from_yaml()
    assert isinstance(cfg, Config)
    assert cfg.data.splits == ("train", "val_seen", "val_unseen")  # list -> tuple coercion


def test_from_yaml_env_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: default.yaml has encoder.hf_endpoint = ${HF_ENDPOINT:-https://hf-mirror.com}.
    monkeypatch.setenv("HF_ENDPOINT", "https://example.test")
    # Act
    cfg = Config.from_yaml()
    # Assert
    assert cfg.encoder.hf_endpoint == "https://example.test"


def test_from_yaml_env_expansion_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    cfg = Config.from_yaml()
    assert cfg.encoder.hf_endpoint == "https://hf-mirror.com"


def test_from_yaml_override(tmp_path) -> None:
    p = tmp_path / "sweep.yaml"
    p.write_text("predictor:\n  horizon: 6\n  depth: 8\n")
    cfg = Config.from_yaml(p)
    assert cfg.predictor.horizon == 6 and cfg.predictor.depth == 8
    assert cfg.predictor.history == HISTORY  # untouched knobs keep defaults


def test_from_yaml_rejects_unknown_section(tmp_path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("bogus_section:\n  x: 1\n")
    with pytest.raises(ValueError, match="unknown config section"):
        Config.from_yaml(p)


def test_from_yaml_rejects_unknown_key(tmp_path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("predictor:\n  hoziron: 6\n")  # typo
    with pytest.raises(ValueError, match="unknown key"):
        Config.from_yaml(p)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PredictorConfig(horizon=0),
        lambda: PredictorConfig(history=-1),
        lambda: PredictorConfig(heads=7),  # 768 % 7 != 0
        lambda: DistillConfig(temperature=0.0),
        lambda: DistillConfig(lambda_latent=-1.0),
        lambda: TrustConfig(disagreement_source="nope"),
        lambda: TrustConfig(k_rollouts=0),
        lambda: TrustConfig(vjepa_surprise_threshold=1.5),
        lambda: TrustConfig(vjepa2_model_id=""),  # must be a non-empty model id
        lambda: EncoderConfig(dtype="float64"),
        lambda: EncoderConfig(input_hw=0),
        lambda: EncoderConfig(text_model_id=""),  # must be a non-empty model id
    ],
)
def test_validation_rejects_bad_values(factory) -> None:
    with pytest.raises(ValueError):
        factory()


def test_config_is_frozen_immutable() -> None:
    cfg = Config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.predictor.depth = 99  # type: ignore[misc]


# --- IngestConfig ---

def test_ingest_config_defaults() -> None:
    c = IngestConfig()
    assert c.name == "wild_video"
    assert c.target_fps == 5.0
    assert c.quality_threshold == 0.3


def test_ingest_config_from_yaml(tmp_path) -> None:
    p = tmp_path / "ingest.yaml"
    p.write_text("ingest:\n  sport: snowboard\n  target_fps: 10.0\n")
    cfg = Config.from_yaml(p)
    assert cfg.ingest is not None
    assert cfg.ingest.sport == "snowboard"
    assert cfg.ingest.target_fps == 10.0


def test_ingest_config_absent_by_default() -> None:
    cfg = Config()
    assert cfg.ingest is None


@pytest.mark.parametrize("factory", [
    lambda: IngestConfig(target_fps=0.0),
    lambda: IngestConfig(min_clip_seconds=-1.0),
    lambda: IngestConfig(min_clip_seconds=30.0, max_clip_seconds=10.0),
    lambda: IngestConfig(resolution_h=0),
    lambda: IngestConfig(quality_threshold=1.5),
    lambda: IngestConfig(name=""),
])
def test_ingest_config_rejects_bad(factory) -> None:
    with pytest.raises(ValueError):
        factory()
