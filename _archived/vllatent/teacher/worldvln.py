"""Frozen WorldVLN teacher wrapper (TORCH tier) — Phase-A step A5.11.

HTTP client for the UPSTREAM WorldVLN inference server (``EmbodiedCity/WorldVLN``
``infer/server.py``, FastAPI via ``infer/run_server.sh`` -> uvicorn :8001). The server is
USER-GATED (GPU + ~36.9 GB weights); this client never imports, launches, or modifies the
upstream clone — it only speaks the wire protocol, verified first-hand against clone commit
``3409b82`` (2026-06-10 re-probe):

  - ``POST /v1/predict_delta_actions`` — per-session autoregressive: first call = 1 frame +
    ``instruction`` (+ ``seed``); each call emits at most ONE segment of ``step`` (=16) actions;
    with ``allow_future_segments=true`` the strict closed loop is 1 frame -> segment 0, then 16
    real frames -> segment 1, ... (released config: num_frames=49, step=16 -> points [1,17,33,49]
    -> 3 segments). ``segment_index == -1`` means nothing was emitted (warmup).
  - **Wire action format** (CORRECTS the A5.8 note): each action row is
    ``[dx_cm, dy_cm, dz_cm, droll_deg, dyaw_deg, dpitch_deg]`` — position-FIRST, (cm, deg)
    DELTAS (``_to_cm_deg`` converts FROM the model-native (m, rad)). The
    ``[roll,yaw,pitch,x,y,z]`` order in the A5.8/A5.9 notes is the training-stats/seam order,
    NOT the wire.
  - **Seeds / K-rollout disagreement:** ``local_seed = seed + segment_index`` UNLESS
    ``lock_seed_across_steps`` — and the released ``infer/config.json`` sets it ``true``, so one
    session is seed-stable across segments. K stochastic rollouts therefore = K **sessions** with
    distinct ``session_id`` + distinct ``seed`` (upstream's own candidate convention:
    ``--candidate_seed_stride 65537``, GRPO/generate_candidate_rollouts.py).

**Three unit systems — do not mix (foot-gun):** wire = (cm, deg); the teacher seam
(``TeacherOutput.rollouts_pose6``, order ``[roll,yaw,pitch,x,y,z]``) = model-native **(m, rad)**
per-step DELTAS (:func:`wire_actions_to_pose6` converts); the student ``delta_4dof`` = (m, **deg**
yaw). The 6->4 projection at cache-build (A5.14) owns the rad->deg yaw conversion.

Frame-PNG base64 encoding lazily imports cv2 (fallback PIL); everything else is stdlib + numpy,
so the module imports on a torch-free box. The contract test mocks the HTTP transport (no
server); the live smoke (``python -m vllatent.teacher.worldvln``) is USER-GATED.

See plans/phase-a5-replan-postpivot.md A5.11.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any

import numpy as np

from vllatent.config import Config
from vllatent.schemas import TEACHER_DOF, TeacherOutput

if TYPE_CHECKING:  # runtime-free: the union inside the alias would not parse on the Py3.9 pure box
    from collections.abc import Callable

    # transport(url, payload_or_None, timeout_s) -> decoded JSON dict. POST when payload is a
    # dict, GET when payload is None. Injected in tests so no server is needed.
    Transport = Callable[[str, "dict[str, Any] | None", float], dict[str, Any]]

DEFAULT_SERVER = "http://127.0.0.1:8001"
PREDICT_ROUTE = "/v1/predict_delta_actions"
HEALTH_ROUTE = "/health"

# Wire action row (server-side _to_cm_deg + response Field doc): position-FIRST, (cm, deg).
WIRE_ORDER = ("dx_cm", "dy_cm", "dz_cm", "droll_deg", "dyaw_deg", "dpitch_deg")
# Seam order (A5.9 TeacherOutput): [roll, yaw, pitch, x, y, z] -> wire indices [3, 4, 5, 0, 1, 2].
_SEAM_FROM_WIRE = (3, 4, 5, 0, 1, 2)
CM_PER_M = 100.0

# Upstream's own K-candidate seed spacing (GRPO/generate_candidate_rollouts.py --candidate_seed_stride).
CANDIDATE_SEED_STRIDE = 65537


def wire_actions_to_pose6(actions_wire: np.ndarray) -> np.ndarray:
    """Convert wire action rows ``(T,6)`` ``[dx,dy,dz,droll,dyaw,dpitch]`` (cm, deg) to the
    A5.9 seam: ``(T,6)`` ``[roll,yaw,pitch,x,y,z]`` per-step deltas in model-native **(m, rad)**.
    """
    arr = np.asarray(actions_wire, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != TEACHER_DOF:
        raise ValueError(f"actions_wire: expected (T, {TEACHER_DOF}), got {arr.shape}")
    seam = arr[:, _SEAM_FROM_WIRE].copy()
    seam[:, 0:3] *= math.pi / 180.0   # droll, dyaw, dpitch: deg -> rad
    seam[:, 3:6] /= CM_PER_M          # dx, dy, dz: cm -> m
    return seam.astype(np.float32)


def teacher_outputs_from_rollouts(rollouts_seam: np.ndarray) -> list[TeacherOutput]:
    """Slice a ``(K, T, 6)`` seam-order rollout stack into T per-step :class:`TeacherOutput`."""
    arr = np.asarray(rollouts_seam, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != TEACHER_DOF:
        raise ValueError(f"rollouts_seam: expected (K, T, {TEACHER_DOF}), got {arr.shape}")
    return [TeacherOutput(rollouts_pose6=arr[:, t, :].copy()) for t in range(arr.shape[1])]


def _http_transport(url: str, payload: dict[str, Any] | None, timeout_s: float) -> dict[str, Any]:
    """Default stdlib transport: JSON POST (payload dict) or GET (payload None)."""
    if payload is None:
        req = urllib.request.Request(url, method="GET")
    else:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:  # connection refused / timeout — make it actionable
        raise RuntimeError(
            f"WorldVLN server unreachable at {url}: {exc}. Is the USER-GATED server up? "
            f"(upstream infer/run_server.sh -> uvicorn :8001; check GET {HEALTH_ROUTE})"
        ) from exc


def _encode_frame_b64(frame_rgb: np.ndarray) -> str:
    """Encode an ``(H,W,3)`` uint8 RGB frame as base64 PNG (server PIL-decodes; it resizes itself).

    Lazily imports cv2 (fallback PIL) so the module stays importable on a minimal box; the
    contract test monkeypatches this.
    """
    if not isinstance(frame_rgb, np.ndarray) or frame_rgb.ndim != 3 or frame_rgb.shape[-1] != 3:
        raise ValueError(f"frame_rgb: expected (H,W,3) ndarray, got {getattr(frame_rgb, 'shape', type(frame_rgb))}")
    try:
        import cv2  # lazy by design

        ok, buf = cv2.imencode(".png", frame_rgb[:, :, ::-1])  # cv2 wants BGR; input is RGB
        if not ok:
            raise RuntimeError("cv2.imencode('.png', ...) failed")
        png = buf.tobytes()
    except ImportError:
        import io

        from PIL import Image  # lazy fallback

        bio = io.BytesIO()
        Image.fromarray(frame_rgb, mode="RGB").save(bio, format="PNG")
        png = bio.getvalue()
    return base64.b64encode(png).decode("ascii")


class WorldVLNTeacherClient:
    """HTTP client for the frozen WorldVLN teacher server (one segment per call, K-session rollouts)."""

    def __init__(
        self,
        server_url: str = DEFAULT_SERVER,
        *,
        timeout_s: float = 600.0,  # 8B AR sampling per segment is slow; be generous
        transport: Transport | None = None,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.timeout_s = timeout_s
        self._transport: Transport = transport if transport is not None else _http_transport

    def health(self) -> dict[str, Any]:
        """``GET /health`` — model-loaded flags + num_frames/step/points/tgt_hw."""
        return self._transport(self.server_url + HEALTH_ROUTE, None, self.timeout_s)

    def predict_segment(
        self,
        session_id: str,
        frames_rgb: list[np.ndarray],
        *,
        instruction: str | None = None,
        seed: int | None = None,
        allow_future_segments: bool = True,
        reset_session: bool = False,
    ) -> dict[str, Any]:
        """One autoregressive call: send frames (+ instruction on a session's first call), get ONE
        segment of wire actions. Returns the raw response dict augmented with ``actions_wire``
        ``(T,6)`` f32 (cm, deg) and ``actions_pose6`` ``(T,6)`` f32 seam (m, rad).
        """
        payload: dict[str, Any] = {
            "session_id": session_id,
            "images_base64": [_encode_frame_b64(f) for f in frames_rgb],
            "allow_future_segments": allow_future_segments,
            "reset_session": reset_session,
        }
        if instruction is not None:
            payload["instruction"] = instruction
        if seed is not None:
            payload["seed"] = int(seed)
        resp = self._transport(self.server_url + PREDICT_ROUTE, payload, self.timeout_s)

        if "actions" not in resp or "segment_index" not in resp:
            raise RuntimeError(f"malformed WorldVLN response (no actions/segment_index): {list(resp)}")
        if int(resp["segment_index"]) < 0 or not resp["actions"]:
            raise RuntimeError(
                f"WorldVLN emitted no segment (segment_index={resp['segment_index']}, "
                f"n_actions={len(resp.get('actions', []))}). Warmup call? Pass "
                f"allow_future_segments=True (strict closed loop) and check num_received_frames="
                f"{resp.get('num_received_frames')} against the server's points."
            )
        wire = np.asarray(resp["actions"], dtype=np.float32)
        if wire.ndim != 2 or wire.shape[1] != TEACHER_DOF:
            raise RuntimeError(f"WorldVLN actions: expected (T, {TEACHER_DOF}), got {wire.shape}")
        return {**resp, "actions_wire": wire, "actions_pose6": wire_actions_to_pose6(wire)}

    def k_rollout_segment(
        self,
        frames_rgb: list[np.ndarray],
        instruction: str,
        *,
        k: int | None = None,
        seed_base: int = 0,
        session_prefix: str = "vllatent",
        config: Config | None = None,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """K stochastic rollouts of the SAME input = K sessions with distinct id + seed (the
        released config locks the seed across a session's segments, so disagreement comes from
        cross-session seeds: ``seed_base + k * CANDIDATE_SEED_STRIDE``).

        Returns ``(rollouts_seam (K,T,6) f32 (m, rad), raw_responses)``. Per-step
        :class:`TeacherOutput` via :func:`teacher_outputs_from_rollouts`.
        """
        k_n = (config if config is not None else Config()).trust.k_rollouts if k is None else k
        if k_n < 1:
            raise ValueError(f"k must be >= 1, got {k_n}")
        responses = [
            self.predict_segment(
                f"{session_prefix}-k{i}",
                frames_rgb,
                instruction=instruction,
                seed=seed_base + i * CANDIDATE_SEED_STRIDE,
                reset_session=True,  # force a fresh run even if the session id was used before
            )
            for i in range(k_n)
        ]
        shapes = {r["actions_pose6"].shape for r in responses}
        if len(shapes) != 1:
            raise RuntimeError(f"rollouts returned inconsistent action shapes: {sorted(shapes)}")
        return np.stack([r["actions_pose6"] for r in responses]), responses


def _synthetic_frame(hw: int = 224) -> np.ndarray:  # pragma: no cover - live-smoke helper
    """A deterministic RGB gradient frame (connectivity/stochasticity smoke, not semantics)."""
    g = np.linspace(0, 255, hw, dtype=np.uint8)
    frame = np.zeros((hw, hw, 3), dtype=np.uint8)
    frame[:, :, 0] = g[None, :]
    frame[:, :, 1] = g[:, None]
    frame[:, :, 2] = 128
    return frame


def _main(argv: list[str] | None = None) -> int:  # pragma: no cover - USER-GATED (live server)
    """Live K-rollout smoke against a running WorldVLN server. USER-GATED — the user stands up
    the server (GPU + weights) and runs this; the agent only emits the command block.
    """
    parser = argparse.ArgumentParser(
        prog="python -m vllatent.teacher.worldvln", description="WorldVLN teacher K-rollout smoke (A5.11)."
    )
    parser.add_argument("--episode", required=True, help="AerialVLN episode JSON (instruction source)")
    parser.add_argument("--rollouts", type=int, default=Config().trust.k_rollouts, help="K (default Config)")
    parser.add_argument("--server", default=DEFAULT_SERVER, help=f"server base URL (default {DEFAULT_SERVER})")
    parser.add_argument("--frame", default=None, help="optional image file for the first frame (else synthetic)")
    parser.add_argument("--seed-base", type=int, default=0)
    args = parser.parse_args(argv)

    from pathlib import Path

    from vllatent.audit import parse_episode

    episode = parse_episode(json.loads(Path(args.episode).read_text()))
    if args.frame is not None:
        import cv2

        bgr = cv2.imread(args.frame, cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"could not read --frame {args.frame}")
        frame = np.ascontiguousarray(bgr[:, :, ::-1])
    else:
        frame = _synthetic_frame()

    client = WorldVLNTeacherClient(args.server)
    health = client.health()
    print(f"[teacher-smoke] health: status={health.get('status')} infinity_loaded={health.get('infinity_loaded')} "
          f"ts_ckpt_loaded={health.get('ts_ckpt_loaded')} points={health.get('points')}")

    rollouts, responses = client.k_rollout_segment(
        [frame], episode.instruction_text, k=args.rollouts, seed_base=args.seed_base
    )
    k_n, t_n, _ = rollouts.shape
    print(f"[teacher-smoke] K={k_n} rollouts x T={t_n} actions (segment "
          f"{responses[0]['segment_index']}), seam order [roll,yaw,pitch,x,y,z] (m, rad)")
    for i in range(k_n):
        print(f"  k={i} seed={args.seed_base + i * CANDIDATE_SEED_STRIDE} "
              f"step0={np.round(rollouts[i, 0], 4).tolist()}")
    step0 = teacher_outputs_from_rollouts(rollouts)[0]
    spread = step0.rollout_spread()
    print(f"[teacher-smoke] step-0 rollout_spread (6,) = {np.round(spread, 5).tolist()}")
    identical = bool(np.all(rollouts == rollouts[0:1]))
    print(f"[teacher-smoke] rollouts identical across K: {identical} "
          f"(expect False — stochastic sampling, distinct seeds)")
    if identical and k_n > 1:
        print("[teacher-smoke] FAIL: K rollouts are identical — disagreement signal is dead; "
              "check seeds reach the server (response.used_prompt/debug) before building A5.14.")
        return 1
    print("[teacher-smoke] OK")
    return 0


__all__ = [
    "WorldVLNTeacherClient",
    "wire_actions_to_pose6",
    "teacher_outputs_from_rollouts",
    "WIRE_ORDER",
    "CANDIDATE_SEED_STRIDE",
    "DEFAULT_SERVER",
    "PREDICT_ROUTE",
    "HEALTH_ROUTE",
]


if __name__ == "__main__":  # pragma: no cover - USER-GATED
    raise SystemExit(_main())
