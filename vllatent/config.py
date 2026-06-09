"""Typed, frozen Config — the single source of truth for swept ablation knobs (PURE tier).

H1/H2/L2/L3: replaces the old untyped ``load_config`` dict. The repo's whole deliverable is
"flip an ablation via config, not code surgery", so the SWEPT knobs (T/H, predictor
depth/heads, distillation weights+temperature, trust disagreement-source/K/threshold) live
here in a frozen, validated dataclass tree. The LOCKED-fixed shapes (DINOv3 PATCH_TOKENS=196 /
EMBED_DIM=768; N_ACTIONS / DOF) stay constants in ``vllatent.schemas``; the AirVLN action
step sizes stay constants in ``vllatent.actions`` — neither is duplicated here.

Dataclass defaults are the source of truth; ``configs/*.yaml`` provide per-experiment
OVERRIDES (env-expanded, strict unknown-key rejection). The spike-dependent trust knobs
(``disagreement_source`` / ``k_rollouts`` / ``vjepa_surprise_threshold``) are typed
PLACEHOLDERS, finalized in A5.9 after the A5.8 disagreement-source investigation. Config
snapshot / resume is a Phase-B SOP — NOT built here.

pyyaml + stdlib + the pure ``schemas`` constants only — CI imports it.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from vllatent.schemas import EMBED_DIM, HISTORY, HORIZON

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "default.yaml"
_ENV_RE = re.compile(r"\$\{([A-Z0-9_]+)(?::-(.*?))?\}")

# Allowed enum values (validated at the boundary).
DISAGREEMENT_SOURCES = ("worldvln_rollout", "airscape_multiseed", "mc_dropout", "vjepa_only")
ENCODER_DTYPES = ("float16", "float32")


def _expand_env(value: Any) -> Any:
    """Expand ``${VAR}`` / ``${VAR:-default}`` inside string values, recursively."""
    if isinstance(value, str):
        def repl(m: re.Match[str]) -> str:
            return os.environ.get(m.group(1), m.group(2) if m.group(2) is not None else "")
        return _ENV_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


@dataclass(frozen=True)
class EncoderConfig:
    """Frozen DINOv3 student-encoder settings (the 196/768 shapes are schemas constants)."""

    model_id: str = "facebook/dinov3-vitb16"
    input_hw: int = 224
    dtype: str = "float16"
    hf_endpoint: str = "https://hf-mirror.com"

    def __post_init__(self) -> None:
        if self.input_hw <= 0:
            raise ValueError(f"encoder.input_hw must be > 0, got {self.input_hw}")
        if self.dtype not in ENCODER_DTYPES:
            raise ValueError(f"encoder.dtype must be one of {ENCODER_DTYPES}, got {self.dtype!r}")


@dataclass(frozen=True)
class PredictorConfig:
    """SWEPT student-transformer structure. ``EMBED_DIM`` (768) is fixed (matches the encoder)."""

    history: int = HISTORY      # H — history frames (single literal in schemas)
    horizon: int = HORIZON      # T — prediction horizon (single literal in schemas)
    depth: int = 12             # swept (8-vs-12 in Phase B)
    heads: int = 12             # swept
    mlp_ratio: int = 4          # FFN ratio (4 * 768 = 3072)

    def __post_init__(self) -> None:
        for name in ("history", "horizon", "depth", "heads", "mlp_ratio"):
            v = getattr(self, name)
            if not isinstance(v, int) or v < 1:
                raise ValueError(f"predictor.{name} must be a positive int, got {v!r}")
        if EMBED_DIM % self.heads != 0:
            raise ValueError(f"predictor.heads ({self.heads}) must divide EMBED_DIM ({EMBED_DIM})")


@dataclass(frozen=True)
class DistillConfig:
    """SWEPT distillation losses (student <- frozen WorldVLN teacher)."""

    lambda_latent: float = 1.0
    lambda_waypoint: float = 1.0
    lambda_horizon: float = 0.0   # Phase C (horizon-distillation)
    temperature: float = 1.0

    def __post_init__(self) -> None:
        for name in ("lambda_latent", "lambda_waypoint", "lambda_horizon"):
            if getattr(self, name) < 0:
                raise ValueError(f"distill.{name} must be >= 0, got {getattr(self, name)}")
        if self.temperature <= 0:
            raise ValueError(f"distill.temperature must be > 0, got {self.temperature}")


@dataclass(frozen=True)
class TrustConfig:
    """SWEPT trust-oracle knobs. PLACEHOLDERS — finalized in A5.9 after the A5.8 investigation."""

    disagreement_source: str = "worldvln_rollout"
    k_rollouts: int = 5
    vjepa_surprise_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.disagreement_source not in DISAGREEMENT_SOURCES:
            raise ValueError(
                f"trust.disagreement_source must be one of {DISAGREEMENT_SOURCES}, "
                f"got {self.disagreement_source!r}"
            )
        if not isinstance(self.k_rollouts, int) or self.k_rollouts < 1:
            raise ValueError(f"trust.k_rollouts must be a positive int, got {self.k_rollouts!r}")
        if not 0.0 <= self.vjepa_surprise_threshold <= 1.0:
            raise ValueError(
                f"trust.vjepa_surprise_threshold must be in [0, 1], got {self.vjepa_surprise_threshold}"
            )


@dataclass(frozen=True)
class DataConfig:
    """Environment data paths + the dataset splits."""

    root: str = "data"
    json_dir: str = "data/aerialvln_json"
    cache_dir: str = "data/latent_cache"
    scenes_root: str = "/opt/aerialvln"
    splits: tuple[str, ...] = ("train", "val_seen", "val_unseen")

    def __post_init__(self) -> None:
        if not self.splits:
            raise ValueError("data.splits must be non-empty")


@dataclass(frozen=True)
class CacheConfig:
    """Cache provenance defaults (the typed manifest builder in A5.4 reads these)."""

    version: str = "0.1"
    color_order: str = "RGB"
    quaternion_order: str = "xyzw"
    frame: str = "airsim_ned_body"


# YAML section name -> its frozen dataclass. Keep in sync with Config's fields.
_SECTIONS: dict[str, type] = {
    "encoder": EncoderConfig,
    "predictor": PredictorConfig,
    "distill": DistillConfig,
    "trust": TrustConfig,
    "data": DataConfig,
    "cache": CacheConfig,
}


@dataclass(frozen=True)
class Config:
    """The frozen, validated config tree — the single source of truth for swept knobs."""

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)
    trust: TrustConfig = field(default_factory=TrustConfig)
    data: DataConfig = field(default_factory=DataConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> Config:
        """Build a Config from a YAML override file (env-expanded). Unknown keys are rejected."""
        cfg_path = Path(path) if path is not None else _DEFAULT_CONFIG
        raw = yaml.safe_load(cfg_path.read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"config root must be a mapping, got {type(raw).__name__}")
        raw = _expand_env(raw)
        unknown = set(raw) - set(_SECTIONS)
        if unknown:
            raise ValueError(f"unknown config section(s): {sorted(unknown)}")
        built = {name: _build_section(scls, raw.get(name, {})) for name, scls in _SECTIONS.items()}
        return cls(**built)


def _build_section(scls: type, data: Any) -> Any:
    """Construct a frozen section dataclass from a dict, rejecting unknown keys."""
    if not isinstance(data, dict):
        raise ValueError(f"config section {scls.__name__} must be a mapping, got {type(data).__name__}")
    valid = {f.name for f in fields(scls)}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(f"unknown key(s) in {scls.__name__}: {sorted(unknown)}")
    coerced = {k: (tuple(v) if isinstance(v, list) else v) for k, v in data.items()}
    return scls(**coerced)


__all__ = [
    "Config",
    "EncoderConfig",
    "PredictorConfig",
    "DistillConfig",
    "TrustConfig",
    "DataConfig",
    "CacheConfig",
    "DISAGREEMENT_SOURCES",
    "ENCODER_DTYPES",
]
