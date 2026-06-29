"""Typed, frozen Config — the single source of truth for swept ablation knobs (PURE tier).

H1/H2/L2/L3: replaces the old untyped ``load_config`` dict. The repo's whole deliverable is
"flip an ablation via config, not code surgery", so the SWEPT knobs (T/H, predictor
depth/heads, distillation weights+temperature) live here in a frozen, validated dataclass
tree. The LOCKED-fixed shapes (DINOv3 PATCH_TOKENS=196 / EMBED_DIM=768; N_ACTIONS / DOF)
stay constants in ``vllatent.schemas``; the AirVLN action step sizes stay constants in
``vllatent.actions`` — neither is duplicated here.

Dataclass defaults are the source of truth; ``configs/*.yaml`` provide per-experiment
OVERRIDES (env-expanded, strict unknown-key rejection).
Config snapshot / resume is a Phase-B SOP — NOT built here.

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
ENCODER_DTYPES = ("float16", "float32")
AMP_DTYPES = ("bf16", "fp16", "fp32")
EARLY_STOP_METRICS = ("val_cos", "val_margin")


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
    """Frozen student-encoder settings (the 196/768 shapes are schemas constants).

    Two frozen towers feed the student: the DINOv3 vision encoder (``model_id``) and the CLIP text
    tower (``text_model_id``) that produces the cached ``lang_tokens``. Both are NON-GATED HF re-hosts
    and single-sourced here (the verifier/cache/manifest read them from Config, never re-hardcode).
    """

    # timm's NON-GATED re-host of Meta's DINOv3 ViT-B/16 (LVD-1689M) — same weights, no gate/token,
    # loaded via timm.create_model (HF repo 'timm/vit_base_patch16_dinov3.lvd1689m'). Meta's own
    # 'facebook/dinov3-vitb16-pretrain-lvd1689m' is gated and rejected our access (2026-06-09).
    model_id: str = "vit_base_patch16_dinov3.lvd1689m"
    input_hw: int = 224
    dtype: str = "float16"
    hf_endpoint: str = "https://hf-mirror.com"
    # Frozen CLIP ViT-B/32 text tower (A5.13b) — NON-GATED (`gated:false`, 15M downloads; probed
    # 2026-06-14). Native text width 512; the wrapper lifts → EMBED_DIM (768) for the lang_tokens cache
    # (the meaningful 512→768 mapping is the student's learned cross-attention, Phase B).
    text_model_id: str = "openai/clip-vit-base-patch32"

    def __post_init__(self) -> None:
        if self.input_hw <= 0:
            raise ValueError(f"encoder.input_hw must be > 0, got {self.input_hw}")
        if self.dtype not in ENCODER_DTYPES:
            raise ValueError(f"encoder.dtype must be one of {ENCODER_DTYPES}, got {self.dtype!r}")
        if not isinstance(self.text_model_id, str) or not self.text_model_id:
            raise ValueError(f"encoder.text_model_id must be a non-empty str, got {self.text_model_id!r}")


@dataclass(frozen=True)
class PredictorConfig:
    """SWEPT student-transformer structure. D = ``EMBED_DIM`` from schemas (768 default, 384 if ViT-S/16)."""

    history: int = HISTORY      # H — history frames (single literal in schemas)
    horizon: int = HORIZON      # T — prediction horizon (single literal in schemas)
    depth: int = 6              # swept (6-vs-8 in Phase B; DINO-WM precedent)
    heads: int = 12             # swept (must divide EMBED_DIM; 6 for D=384)
    mlp_ratio: int = 4          # FFN ratio (4 * D)
    dropout: float = 0.1        # predictor dropout (DINO-WM precedent; helps with small pilot)

    def __post_init__(self) -> None:
        for name in ("history", "horizon", "depth", "heads", "mlp_ratio"):
            v = getattr(self, name)
            if not isinstance(v, int) or v < 1:
                raise ValueError(f"predictor.{name} must be a positive int, got {v!r}")
        if EMBED_DIM % self.heads != 0:
            raise ValueError(f"predictor.heads ({self.heads}) must divide EMBED_DIM ({EMBED_DIM})")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"predictor.dropout must be in [0, 1), got {self.dropout}")


@dataclass(frozen=True)
class DistillConfig:
    """SWEPT training losses."""

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
class TrainConfig:
    """SWEPT training-run knobs for the B-1 latent run (PURE tier — validated, no torch).

    The single source of truth for the training hyper-parameters the B-1 run sweeps:
    optimizer, schedule, AMP precision, scene-split, eval/early-stop, and the game-domain
    down-weight. ``train_sports.py`` builds this from CLI args (fail-fast on a bad knob) and
    snapshots it next to the run. The model *structure* lives in ``PredictorConfig``; this is
    the run recipe. **No Stage-2/3 / head / freeze knobs — that is Phase B-2.**
    """

    latent_only: bool = False         # B-1: train the predictor only on L_latent (skip head/L_wp)
    lr: float = 2e-4
    weight_decay: float = 0.05        # decoupled AdamW WD (decay group only)
    warmup_frac: float = 0.05         # fraction of total steps for linear LR warmup → cosine
    batch_size: int = 64
    epochs: int = 100                 # upper bound; early-stop usually triggers first
    amp_dtype: str = "bf16"           # one of AMP_DTYPES; bf16 ⇒ no GradScaler
    val_frac: float = 0.2             # fraction of SOURCES (not windows) held out for val
    eval_every_epochs: int = 1
    early_stop_patience: int = 8      # evals without val improvement before stopping
    early_stop_metric: str = "val_cos"  # one of EARLY_STOP_METRICS
    domain_weight: float = 1.0        # sampling weight for domain=game clips (1.0 = no game mix)
    use_action_film: bool = True      # False ⇒ action-free predictor ablation (dt-FiLM only)
    grad_clip: float = 1.0            # 0 disables clipping
    seed: int = 42
    num_workers: int = 4

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError(f"train.lr must be > 0, got {self.lr}")
        if self.weight_decay < 0:
            raise ValueError(f"train.weight_decay must be >= 0, got {self.weight_decay}")
        if not (0.0 <= self.warmup_frac < 1.0):
            raise ValueError(f"train.warmup_frac must be in [0, 1), got {self.warmup_frac}")
        if not (0.0 <= self.val_frac < 1.0):
            raise ValueError(f"train.val_frac must be in [0, 1), got {self.val_frac}")
        if self.domain_weight < 0:
            raise ValueError(f"train.domain_weight must be >= 0, got {self.domain_weight}")
        if self.grad_clip < 0:
            raise ValueError(f"train.grad_clip must be >= 0, got {self.grad_clip}")
        for name in ("batch_size", "epochs", "eval_every_epochs", "early_stop_patience"):
            v = getattr(self, name)
            if not isinstance(v, int) or v < 1:
                raise ValueError(f"train.{name} must be a positive int, got {v!r}")
        if self.num_workers < 0:
            raise ValueError(f"train.num_workers must be >= 0, got {self.num_workers}")
        if self.amp_dtype not in AMP_DTYPES:
            raise ValueError(f"train.amp_dtype must be one of {AMP_DTYPES}, got {self.amp_dtype!r}")
        if self.early_stop_metric not in EARLY_STOP_METRICS:
            raise ValueError(
                f"train.early_stop_metric must be one of {EARLY_STOP_METRICS}, got {self.early_stop_metric!r}"
            )


@dataclass(frozen=True)
class DataConfig:
    """Dataset identity + environment data paths + the splits."""

    name: str = "aerialvln"
    license: str = "CC BY-NC-SA 4.0"        # non-commercial, share-alike (flag at publication)
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


@dataclass(frozen=True)
class IngestConfig:
    """Wild-video ingestion pipeline settings (optional Config section)."""

    name: str = "wild_video"
    sport: str = "skiing"
    license: str = "fair-use-research"
    raw_dir: str = "ingest_data/raw"
    frames_dir: str = "ingest_data/frames"
    cache_dir: str = "ingest_data/latent_cache"
    clips_yaml: str = "configs/ingest_clips.yaml"
    target_fps: float = 5.0
    clip_length_seconds: float = 10.0
    min_clip_seconds: float = 10.0
    max_clip_seconds: float = 120.0
    resolution_h: int = 720
    resolution_w: int = 1280
    megasam_model: str = "megasam_base"
    undistort_model: str = "pinhole"
    quality_threshold: float = 0.3

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"ingest.name must be a non-empty str, got {self.name!r}")
        if self.target_fps <= 0:
            raise ValueError(f"ingest.target_fps must be > 0, got {self.target_fps}")
        if self.clip_length_seconds <= 0:
            raise ValueError(f"ingest.clip_length_seconds must be > 0, got {self.clip_length_seconds}")
        if self.min_clip_seconds <= 0:
            raise ValueError(f"ingest.min_clip_seconds must be > 0, got {self.min_clip_seconds}")
        if self.max_clip_seconds <= self.min_clip_seconds:
            raise ValueError(
                f"ingest.max_clip_seconds ({self.max_clip_seconds}) must be > "
                f"min_clip_seconds ({self.min_clip_seconds})"
            )
        if self.resolution_h <= 0 or self.resolution_w <= 0:
            raise ValueError(
                f"ingest.resolution must be positive, got ({self.resolution_h}, {self.resolution_w})"
            )
        if not (0.0 <= self.quality_threshold <= 1.0):
            raise ValueError(f"ingest.quality_threshold must be in [0,1], got {self.quality_threshold}")


# YAML section name -> its frozen dataclass. Keep in sync with Config's fields.
_SECTIONS: dict[str, type] = {
    "encoder": EncoderConfig,
    "predictor": PredictorConfig,
    "distill": DistillConfig,
    "data": DataConfig,
    "cache": CacheConfig,
}
_OPTIONAL_SECTIONS: dict[str, type] = {
    "ingest": IngestConfig,
}


@dataclass(frozen=True)
class Config:
    """The frozen, validated config tree — the single source of truth for swept knobs."""

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)
    data: DataConfig = field(default_factory=DataConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    ingest: IngestConfig | None = None

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> Config:
        """Build a Config from a YAML override file (env-expanded). Unknown keys are rejected."""
        cfg_path = Path(path) if path is not None else _DEFAULT_CONFIG
        raw = yaml.safe_load(cfg_path.read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"config root must be a mapping, got {type(raw).__name__}")
        raw = _expand_env(raw)
        unknown = set(raw) - set(_SECTIONS) - set(_OPTIONAL_SECTIONS)
        if unknown:
            raise ValueError(f"unknown config section(s): {sorted(unknown)}")
        built: dict[str, Any] = {
            name: _build_section(scls, raw.get(name, {})) for name, scls in _SECTIONS.items()
        }
        for name, scls in _OPTIONAL_SECTIONS.items():
            if name in raw:
                built[name] = _build_section(scls, raw[name])
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
    "TrainConfig",
    "DataConfig",
    "CacheConfig",
    "IngestConfig",
    "ENCODER_DTYPES",
    "AMP_DTYPES",
    "EARLY_STOP_METRICS",
]
