"""Sports-following data config (PURE tier) — Phase B1 step 2.

``SportsDataConfig`` frozen dataclass for the sports-following pipeline settings.
Integrated into ``vllatent.config.Config`` as an optional ``sports`` section —
absent in AerialVLN YAML, populated when ``sports:`` is present.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SportsDataConfig:
    """Sports-following pipeline settings (the sports analog of ``DataConfig``)."""

    name: str = "sports_following"
    sport: str = "skiing"
    license: str = "fair-use-research"
    raw_dir: str = "sports_data/raw"
    frames_dir: str = "sports_data/frames"
    cache_dir: str = "sports_data/latent_cache"
    clips_yaml: str = "configs/sports_clips.yaml"
    target_fps: float = 5.0
    min_clip_seconds: float = 10.0
    max_clip_seconds: float = 120.0
    resolution_h: int = 720
    resolution_w: int = 1280
    megasam_model: str = "megasam_base"
    undistort_model: str = "pinhole"
    quality_threshold: float = 0.3

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"sports.name must be a non-empty str, got {self.name!r}")
        if self.target_fps <= 0:
            raise ValueError(f"sports.target_fps must be > 0, got {self.target_fps}")
        if self.min_clip_seconds <= 0:
            raise ValueError(f"sports.min_clip_seconds must be > 0, got {self.min_clip_seconds}")
        if self.max_clip_seconds <= self.min_clip_seconds:
            raise ValueError(
                f"sports.max_clip_seconds ({self.max_clip_seconds}) must be > "
                f"min_clip_seconds ({self.min_clip_seconds})"
            )
        if self.resolution_h <= 0 or self.resolution_w <= 0:
            raise ValueError(
                f"sports.resolution must be positive, got ({self.resolution_h}, {self.resolution_w})"
            )
        if not (0.0 <= self.quality_threshold <= 1.0):
            raise ValueError(f"sports.quality_threshold must be in [0,1], got {self.quality_threshold}")


__all__ = ["SportsDataConfig"]
