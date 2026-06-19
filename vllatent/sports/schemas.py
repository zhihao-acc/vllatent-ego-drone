"""Sports-following data schemas (PURE tier) — Phase B1 step 1.

Frozen dataclasses for the sports-following training tuple, clip metadata, and cache
manifest entry. Parallel to ``vllatent.schemas`` (AerialVLN) but adapted for
video-sourced data: no discrete actions, no language tokens, continuous body-frame
deltas from MegaSaM ego-motion extraction.

Shares locked constants (``PATCH_TOKENS``, ``EMBED_DIM``, ``HISTORY``, dtypes) from
``vllatent.schemas`` — no duplication.
"""
from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from typing import Any

import numpy as np

from vllatent.schemas import (
    DELTA_DTYPE,
    DOF,
    EMBED_DIM,
    HISTORY,
    LATENT_DTYPE,
    MASK_DTYPE,
    PATCH_TOKENS,
    _check_array,
)


@dataclass(frozen=True, eq=False)
class SportsSample:
    """One training tuple from the sports-following latent cache.

    Structurally similar to ``StepSample`` but without ``action_id`` (continuous-only),
    ``lang_tokens`` / ``lang_mask`` (visual tracking, not language-conditioned), and with
    additional per-frame quality metadata from the extraction pipeline.
    """

    z_t: np.ndarray              # (196,768) fp16 — DINOv3 patch tokens, obs @ t
    history_latents: np.ndarray  # (H,196,768) fp16 — z_{t-H+1..t}; zero-padded at clip start
    history_mask: np.ndarray     # (H,) bool — True = real history frame, False = padding
    z_next: np.ndarray           # (196,768) fp16 — DINOv3 latent of next obs = prediction target
    delta_4dof: np.ndarray       # (4,) f32 — (dx,dy,dz,dyaw) body-frame, from MegaSaM
    vo_confidence: float         # MegaSaM per-frame VO confidence in [0,1]
    frame_quality: float         # composite quality score in [0,1]
    dt_seconds: float            # time delta to next frame (> 0)

    def __post_init__(self) -> None:
        _check_array("z_t", self.z_t, (PATCH_TOKENS, EMBED_DIM), dtype=LATENT_DTYPE)
        _check_array("history_latents", self.history_latents, (HISTORY, PATCH_TOKENS, EMBED_DIM), dtype=LATENT_DTYPE)
        _check_array("history_mask", self.history_mask, (HISTORY,), dtype=MASK_DTYPE)
        _check_array("z_next", self.z_next, (PATCH_TOKENS, EMBED_DIM), dtype=LATENT_DTYPE)
        _check_array("delta_4dof", self.delta_4dof, (DOF,), dtype=DELTA_DTYPE)
        for name in ("vo_confidence", "frame_quality", "dt_seconds"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
                raise TypeError(f"{name}: expected float, got {type(v).__name__}")
        if not (0.0 <= float(self.vo_confidence) <= 1.0):
            raise ValueError(f"vo_confidence: expected [0,1], got {self.vo_confidence}")
        if not (0.0 <= float(self.frame_quality) <= 1.0):
            raise ValueError(f"frame_quality: expected [0,1], got {self.frame_quality}")
        if float(self.dt_seconds) <= 0.0:
            raise ValueError(f"dt_seconds: expected > 0, got {self.dt_seconds}")


SCALE_MODES = ("normalized", "gps_aligned", "imu_aligned")
CAMERA_MODELS = ("pinhole_undistorted", "raw_fisheye", "pinhole_native")


@dataclass(frozen=True)
class SportsClipRecord:
    """Parsed metadata for one video clip (the sports analog of ``EpisodeRecord``)."""

    clip_id: str
    source_url: str
    sport: str
    n_frames: int
    fps_original: float
    fps_sampled: float
    duration_seconds: float
    scale_mode: str
    camera_model: str

    def __post_init__(self) -> None:
        for name in ("clip_id", "source_url", "sport"):
            v = getattr(self, name)
            if not isinstance(v, str) or not v:
                raise ValueError(f"{name}: expected non-empty str, got {v!r}")
        if not isinstance(self.n_frames, int) or self.n_frames < 1:
            raise ValueError(f"n_frames: expected positive int, got {self.n_frames!r}")
        for name in ("fps_original", "fps_sampled", "duration_seconds"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
                raise TypeError(f"{name}: expected float, got {type(v).__name__}")
            if float(v) <= 0.0:
                raise ValueError(f"{name}: expected > 0, got {v}")
        if self.scale_mode not in SCALE_MODES:
            raise ValueError(f"scale_mode: expected one of {SCALE_MODES}, got {self.scale_mode!r}")
        if self.camera_model not in CAMERA_MODELS:
            raise ValueError(f"camera_model: expected one of {CAMERA_MODELS}, got {self.camera_model!r}")


@dataclass(frozen=True)
class SportsClipManifestEntry:
    """One per-clip cache manifest entry (the sports analog of ``CacheManifestEntry``)."""

    clip_id: str
    n_frames: int
    latent_path: str
    source_url: str
    sport: str
    scale_mode: str
    fps_sampled: float
    duration_seconds: float

    @classmethod
    def required_keys(cls) -> tuple[str, ...]:
        return tuple(
            f.name for f in fields(cls) if f.default is MISSING and f.default_factory is MISSING
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "n_frames": self.n_frames,
            "latent_path": self.latent_path,
            "source_url": self.source_url,
            "sport": self.sport,
            "scale_mode": self.scale_mode,
            "fps_sampled": self.fps_sampled,
            "duration_seconds": self.duration_seconds,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SportsClipManifestEntry:
        return cls(
            clip_id=str(d["clip_id"]),
            n_frames=int(d["n_frames"]),
            latent_path=str(d["latent_path"]),
            source_url=str(d["source_url"]),
            sport=str(d["sport"]),
            scale_mode=str(d["scale_mode"]),
            fps_sampled=float(d["fps_sampled"]),
            duration_seconds=float(d["duration_seconds"]),
        )


__all__ = [
    "SportsSample",
    "SportsClipRecord",
    "SportsClipManifestEntry",
    "SCALE_MODES",
    "CAMERA_MODELS",
]
