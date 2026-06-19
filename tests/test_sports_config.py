"""Tests for vllatent.sports.config and its integration into Config."""
from __future__ import annotations

from pathlib import Path

import pytest

from vllatent.config import Config
from vllatent.sports.config import SportsDataConfig

_REPO = Path(__file__).resolve().parent.parent
_SPORTS_YAML = _REPO / "configs" / "sports.yaml"
_DEFAULT_YAML = _REPO / "configs" / "default.yaml"


class TestSportsDataConfig:
    def test_defaults(self) -> None:
        c = SportsDataConfig()
        assert c.target_fps == 5.0
        assert c.sport == "skiing"
        assert c.quality_threshold == 0.3

    def test_frozen(self) -> None:
        c = SportsDataConfig()
        with pytest.raises(AttributeError):
            c.target_fps = 10.0  # type: ignore[misc]

    def test_bad_target_fps(self) -> None:
        with pytest.raises(ValueError, match="target_fps"):
            SportsDataConfig(target_fps=0.0)

    def test_bad_min_clip(self) -> None:
        with pytest.raises(ValueError, match="min_clip_seconds"):
            SportsDataConfig(min_clip_seconds=-1.0)

    def test_max_less_than_min(self) -> None:
        with pytest.raises(ValueError, match="max_clip_seconds"):
            SportsDataConfig(min_clip_seconds=30.0, max_clip_seconds=10.0)

    def test_bad_resolution(self) -> None:
        with pytest.raises(ValueError, match="resolution"):
            SportsDataConfig(resolution_h=0)

    def test_bad_quality_threshold(self) -> None:
        with pytest.raises(ValueError, match="quality_threshold"):
            SportsDataConfig(quality_threshold=1.5)

    def test_empty_name(self) -> None:
        with pytest.raises(ValueError, match="name"):
            SportsDataConfig(name="")


class TestConfigSportsIntegration:
    def test_default_config_has_no_sports(self) -> None:
        cfg = Config()
        assert cfg.sports is None

    def test_default_yaml_has_no_sports(self) -> None:
        cfg = Config.from_yaml(_DEFAULT_YAML)
        assert cfg.sports is None

    def test_sports_yaml_populates_sports(self) -> None:
        cfg = Config.from_yaml(_SPORTS_YAML)
        assert cfg.sports is not None
        assert cfg.sports.sport == "skiing"
        assert cfg.sports.target_fps == 5.0

    def test_sports_yaml_preserves_aerialvln_defaults(self) -> None:
        cfg = Config.from_yaml(_SPORTS_YAML)
        assert cfg.data.name == "aerialvln"
        assert cfg.encoder.model_id == "vit_base_patch16_dinov3.lvd1689m"

    def test_unknown_key_under_sports_rejected(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("sports:\n  nonexistent_key: true\n")
        with pytest.raises(ValueError, match="unknown key"):
            Config.from_yaml(bad)

    def test_unknown_top_level_still_rejected(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("bogus_section:\n  key: value\n")
        with pytest.raises(ValueError, match="unknown config section"):
            Config.from_yaml(bad)
