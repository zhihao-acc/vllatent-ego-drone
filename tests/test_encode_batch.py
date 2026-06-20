"""Tests for vllatent.encode.batch — monkeypatched encoder, no real weights, no torch at test time."""
from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vllatent.schemas import EMBED_DIM, LATENT_DTYPE, PATCH_TOKENS

_BATCH_FILE = Path(__file__).resolve().parent.parent / "vllatent" / "encode" / "batch.py"


def _fake_latent() -> np.ndarray:
    """A deterministic (196, 768) fp16 array standing in for one encoded frame."""
    return np.ones((PATCH_TOKENS, EMBED_DIM), dtype=LATENT_DTYPE)


def _write_fake_jpgs(directory: Path, names: list[str]) -> None:
    """Write zero-byte .jpg files — content is irrelevant when the encoder is mocked."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"\xff")


def test_encode_frames_shape_and_dtype(tmp_path: Path) -> None:
    _write_fake_jpgs(tmp_path, ["a.jpg", "b.jpg", "c.jpg"])

    mock_encoder = MagicMock()
    mock_encoder.encode_rgb.return_value = _fake_latent()

    with (
        patch("vllatent.encode.dinov3.DinoV3Encoder", return_value=mock_encoder),
        patch("vllatent.io.load_rgb", return_value=np.zeros((224, 224, 3), dtype=np.uint8)),
    ):
        from vllatent.encode.batch import encode_frames

        result = encode_frames(tmp_path, device="cpu")

    assert result.shape == (3, PATCH_TOKENS, EMBED_DIM)
    assert result.dtype == np.dtype(LATENT_DTYPE)


def test_encode_frames_empty_directory_raises(tmp_path: Path) -> None:
    from vllatent.encode.batch import encode_frames

    with pytest.raises(FileNotFoundError, match="No .jpg files"):
        encode_frames(tmp_path, device="cpu")


def test_encode_frames_nonexistent_directory_raises() -> None:
    from vllatent.encode.batch import encode_frames

    with pytest.raises(ValueError, match="does not exist"):
        encode_frames("/no/such/directory", device="cpu")


def test_encode_frames_sorted_order(tmp_path: Path) -> None:
    _write_fake_jpgs(tmp_path, ["c.jpg", "a.jpg", "b.jpg"])

    loaded_paths: list[str] = []

    def tracking_load_rgb(path):  # noqa: ANN001, ANN202
        loaded_paths.append(Path(path).name)
        return np.zeros((224, 224, 3), dtype=np.uint8)

    mock_encoder = MagicMock()
    mock_encoder.encode_rgb.return_value = _fake_latent()

    with (
        patch("vllatent.encode.dinov3.DinoV3Encoder", return_value=mock_encoder),
        patch("vllatent.io.load_rgb", side_effect=tracking_load_rgb),
    ):
        from vllatent.encode.batch import encode_frames

        encode_frames(tmp_path, device="cpu")

    assert loaded_paths == ["a.jpg", "b.jpg", "c.jpg"]


def test_no_module_level_torch_or_timm_import() -> None:
    """AST check: batch.py must not have module-level ``import torch`` or ``import timm``."""
    source = _BATCH_FILE.read_text()
    tree = ast.parse(source)

    forbidden = {"torch", "timm"}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in forbidden, (
                    f"module-level 'import {alias.name}' found at line {node.lineno}"
                )
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                top = node.module.split(".")[0]
                assert top not in forbidden, (
                    f"module-level 'from {node.module} import ...' found at line {node.lineno}"
                )
