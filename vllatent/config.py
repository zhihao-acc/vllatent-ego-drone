"""Config loader for ``configs/*.yaml`` (PURE tier).

Loads a YAML config, applies ``${ENV}`` / ``${ENV:-default}``-style expansion on
string values (recursively), and returns a plain dict. Imports pyyaml + stdlib
only — keep it sibling/torch-free (CI imports it).
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "default.yaml"
_ENV_RE = re.compile(r"\$\{([A-Z0-9_]+)(?::-(.*?))?\}")


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


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load + env-expand a config dict. Defaults to ``configs/default.yaml``."""
    cfg_path = Path(path) if path is not None else _DEFAULT_CONFIG
    with cfg_path.open() as f:
        raw = yaml.safe_load(f) or {}
    return _expand_env(raw)


__all__ = ["load_config"]
