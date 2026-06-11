"""A5.11 contract tests: WorldVLN teacher client with a MOCKED HTTP transport (no server, no GPU).

Pins the wire protocol facts verified against the upstream clone (commit ``3409b82``):
  - wire action rows are ``[dx_cm,dy_cm,dz_cm,droll_deg,dyaw_deg,dpitch_deg]`` — position-FIRST
    (cm, deg) — converted to the A5.9 seam ``[roll,yaw,pitch,x,y,z]`` in (m, rad);
  - K stochastic rollouts = K sessions with distinct ``session_id`` + ``seed`` spaced by the
    upstream candidate stride (65537), instruction sent to each (first call of its session);
  - ``segment_index == -1`` / empty actions (warmup) raises an actionable error.

The transport + frame encoder are injected/monkeypatched, so these run in the PURE gate (the
heavy halves — server, GPU, weights — are the USER-GATED live smoke).
"""
from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import numpy as np
import pytest

import vllatent.teacher.worldvln as wv
from vllatent.schemas import TEACHER_DOF, TeacherOutput

# One wire row with distinct values per channel: dx=100cm dy=200cm dz=-50cm droll=90° dyaw=45° dpitch=-30°.
_WIRE_ROW = [100.0, 200.0, -50.0, 90.0, 45.0, -30.0]
# Its seam image: [roll,yaw,pitch, x,y,z] = [pi/2, pi/4, -pi/6, 1.0, 2.0, -0.5] (m, rad).
_SEAM_ROW = [math.pi / 2, math.pi / 4, -math.pi / 6, 1.0, 2.0, -0.5]


def _response(actions: list[list[float]], segment_index: int = 0) -> dict:
    return {
        "actions": actions,
        "segment_index": segment_index,
        "num_received_frames": 1,
        "prefix_latents": 1,
        "done": False,
    }


class _FakeTransport:
    """Records every (url, payload) and returns scripted responses (seed-dependent by default)."""

    def __init__(self, respond=None):
        self.calls: list[tuple[str, dict | None]] = []
        self._respond = respond

    def __call__(self, url: str, payload: dict | None, timeout_s: float) -> dict:
        self.calls.append((url, payload))
        if self._respond is not None:
            return self._respond(url, payload)
        if payload is None:  # GET /health
            return {"status": "ok", "infinity_loaded": True, "ts_ckpt_loaded": True, "points": [1, 17, 33, 49]}
        # Default: one segment of 2 actions whose dx encodes the seed -> rollouts differ per seed.
        seed = int(payload.get("seed", 0))
        row = list(_WIRE_ROW)
        row[0] += seed % 7  # vary dx_cm with the seed
        return _response([row, list(_WIRE_ROW)])


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> tuple[wv.WorldVLNTeacherClient, _FakeTransport]:
    monkeypatch.setattr(wv, "_encode_frame_b64", lambda f: "<b64>")
    transport = _FakeTransport()
    return wv.WorldVLNTeacherClient("http://fake:8001", transport=transport), transport


def _frame() -> np.ndarray:
    return np.zeros((8, 8, 3), dtype=np.uint8)


# --- wire -> seam conversion (pure math) ---


def test_wire_to_seam_order_and_units() -> None:
    seam = wv.wire_actions_to_pose6(np.array([_WIRE_ROW], dtype=np.float32))
    assert seam.shape == (1, TEACHER_DOF) and seam.dtype == np.float32
    np.testing.assert_allclose(seam[0], _SEAM_ROW, rtol=1e-6)


def test_wire_to_seam_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="expected"):
        wv.wire_actions_to_pose6(np.zeros((3, 4)))


# --- predict_segment ---


def test_predict_segment_payload_and_route(client) -> None:
    cl, transport = client
    cl.predict_segment("sess-0", [_frame()], instruction="fly forward", seed=42)
    url, payload = transport.calls[0]
    assert url == "http://fake:8001" + wv.PREDICT_ROUTE
    assert payload["session_id"] == "sess-0"
    assert payload["instruction"] == "fly forward"
    assert payload["seed"] == 42
    assert payload["images_base64"] == ["<b64>"]
    assert payload["allow_future_segments"] is True  # strict closed loop by default


def test_predict_segment_returns_wire_and_seam(client) -> None:
    cl, _ = client
    out = cl.predict_segment("s", [_frame()], instruction="i", seed=0)
    assert out["actions_wire"].shape == (2, TEACHER_DOF)
    np.testing.assert_allclose(out["actions_pose6"][1], _SEAM_ROW, rtol=1e-6)


def test_warmup_minus_one_segment_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wv, "_encode_frame_b64", lambda f: "<b64>")
    transport = _FakeTransport(respond=lambda url, p: _response([], segment_index=-1))
    cl = wv.WorldVLNTeacherClient("http://fake:8001", transport=transport)
    with pytest.raises(RuntimeError, match="no segment"):
        cl.predict_segment("s", [_frame()], instruction="i")


def test_malformed_response_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wv, "_encode_frame_b64", lambda f: "<b64>")
    transport = _FakeTransport(respond=lambda url, p: {"oops": True})
    cl = wv.WorldVLNTeacherClient("http://fake:8001", transport=transport)
    with pytest.raises(RuntimeError, match="malformed"):
        cl.predict_segment("s", [_frame()], instruction="i")


# --- K-rollout disagreement machinery ---


def test_k_rollout_distinct_sessions_and_candidate_seed_stride(client) -> None:
    cl, transport = client
    cl.k_rollout_segment([_frame()], "go", k=3, seed_base=10)
    payloads = [p for _, p in transport.calls]
    assert [p["session_id"] for p in payloads] == ["vllatent-k0", "vllatent-k1", "vllatent-k2"]
    assert [p["seed"] for p in payloads] == [10, 10 + wv.CANDIDATE_SEED_STRIDE, 10 + 2 * wv.CANDIDATE_SEED_STRIDE]
    assert all(p["instruction"] == "go" for p in payloads)       # first call of EACH session
    assert all(p["reset_session"] is True for p in payloads)     # forces a fresh run per rollout


def test_k_rollout_stacks_and_teacher_outputs(client) -> None:
    cl, _ = client
    rollouts, responses = cl.k_rollout_segment([_frame()], "go", k=3, seed_base=0)
    assert rollouts.shape == (3, 2, TEACHER_DOF) and rollouts.dtype == np.float32
    outs = wv.teacher_outputs_from_rollouts(rollouts)
    assert len(outs) == 2 and all(isinstance(o, TeacherOutput) for o in outs)
    assert outs[0].rollouts_pose6.shape == (3, TEACHER_DOF)
    # Step 0 differs per seed (dx varies) -> spread > 0 on the x channel (seam index 3), 0 on yaw.
    spread0 = outs[0].rollout_spread()
    assert spread0[3] > 0.0 and spread0[1] == 0.0
    # Step 1 is identical across seeds -> zero spread everywhere.
    np.testing.assert_allclose(outs[1].rollout_spread(), np.zeros(TEACHER_DOF), atol=1e-12)


def test_k_default_comes_from_config(client) -> None:
    cl, transport = client
    from vllatent.config import Config

    cl.k_rollout_segment([_frame()], "go")  # no k -> Config().trust.k_rollouts
    assert len(transport.calls) == Config().trust.k_rollouts


def test_k_rollout_inconsistent_shapes_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wv, "_encode_frame_b64", lambda f: "<b64>")

    def respond(url, p):
        n = 2 if p["seed"] == 0 else 3  # second session returns a different T
        return _response([list(_WIRE_ROW)] * n)

    cl = wv.WorldVLNTeacherClient("http://fake:8001", transport=_FakeTransport(respond=respond))
    with pytest.raises(RuntimeError, match="inconsistent"):
        cl.k_rollout_segment([_frame()], "go", k=2, seed_base=0)


# --- health + tier purity ---


def test_health_uses_get(client) -> None:
    cl, transport = client
    h = cl.health()
    url, payload = transport.calls[0]
    assert url == "http://fake:8001" + wv.HEALTH_ROUTE and payload is None
    assert h["infinity_loaded"] is True


def test_frame_validation_unmocked() -> None:
    """Bad frame inputs raise in the REAL _encode_frame_b64 (validation precedes the lazy cv2/PIL
    import, so this is hermetic on a cv2-free box). Closes the review gap: the client fixture mocks
    the encoder, so without this test the shape check could be deleted unnoticed."""
    with pytest.raises(ValueError, match="expected \\(H,W,3\\)"):
        wv._encode_frame_b64(np.zeros((8, 8), dtype=np.uint8))       # missing channel axis
    with pytest.raises(ValueError, match="expected \\(H,W,3\\)"):
        wv._encode_frame_b64(np.zeros((8, 8, 4), dtype=np.uint8))    # wrong channel count
    with pytest.raises(ValueError, match="expected \\(H,W,3\\)"):
        wv._encode_frame_b64([[1, 2, 3]])                            # not an ndarray


def test_module_imports_heavy_free() -> None:
    """stdlib+numpy at module scope; cv2/PIL only inside _encode_frame_b64; never torch/upstream."""
    assert "torch" not in sys.modules or True  # informational; the AST check is the real guard
    heavy = {"torch", "transformers", "timm", "cv2", "PIL", "requests", "httpx", "fastapi", "uvicorn"}
    tree = ast.parse(Path(wv.__file__).read_text())
    for node in tree.body:  # module scope only — function-local imports are the lazy pattern
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names = [node.module]
        for n in names:
            assert n.split(".")[0] not in heavy, f"module-level heavy import {n!r} breaks tier purity"
