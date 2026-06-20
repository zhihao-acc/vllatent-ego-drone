"""A5.13 render-harness UNIT tests with a MOCKED airsim client (no real sim, no cv2).

These inject a fake ``airsim`` module + a fake client, so the test runs in the PURE CI gate (not
``@pytest.mark.sim``) and pins the three render foot-guns without a UE4 scene:
  #1 quaternion order — the teleport orientation is canonical xyzw (``airsim.Quaternionr(x,y,z,w)``).
  #2 BGR→RGB — the BGRA Scene buffer is alpha-dropped + channel-reversed to RGB.
  #3 single-threaded RPC — EVERY client call happens while the harness lock is held.
The live render is exercised separately (USER-GATED, in fly0-m1).
"""
from __future__ import annotations

import ast
import math
import types
from pathlib import Path

import numpy as np
import pytest

from vllatent.render.harness import (
    CAMERA_NAME,
    VEHICLE_NAME,
    RenderHarness,
    decode_scene_to_rgb,
)

# --- fake airsim namespace (only what the harness touches) ---


class _Vec3:
    def __init__(self, x, y, z):
        self.x_val, self.y_val, self.z_val = x, y, z


class _Quat:
    def __init__(self, x, y, z, w):
        self.x_val, self.y_val, self.z_val, self.w_val = x, y, z, w


class _Pose:
    def __init__(self, position_val, orientation_val):
        self.position, self.orientation = position_val, orientation_val


class _ImageRequest:
    def __init__(self, camera, image_type, pixels_as_float, compress):
        self.camera, self.image_type = camera, image_type
        self.pixels_as_float, self.compress = pixels_as_float, compress


class _Resp:
    def __init__(self, data: bytes, h: int, w: int):
        self.image_data_uint8, self.height, self.width = data, h, w


_FAKE_AIRSIM = types.SimpleNamespace(
    Pose=_Pose,
    Vector3r=_Vec3,
    Quaternionr=_Quat,
    ImageRequest=_ImageRequest,
    ImageType=types.SimpleNamespace(Scene="Scene"),
)


class _FakeClient:
    """Records calls; asserts the harness lock is HELD on every RPC (foot-gun #3)."""

    def __init__(self, response: _Resp):
        self.response = response
        self.lock = None  # wired to harness._lock after construction
        self.pose_calls: list = []
        self.image_calls: list = []

    def simSetVehiclePose(self, pose, ignore_collision, vehicle_name):
        assert self.lock is not None and self.lock.locked(), "simSetVehiclePose ran outside the lock"
        self.pose_calls.append((pose, ignore_collision, vehicle_name))

    def simGetImages(self, requests, vehicle_name):
        assert self.lock is not None and self.lock.locked(), "simGetImages ran outside the lock"
        self.image_calls.append((requests, vehicle_name))
        return [self.response]


def _bgra_response(h: int = 4, w: int = 4, b: int = 10, g: int = 20, r: int = 30) -> _Resp:
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3] = b, g, r, 255  # BGRA
    return _Resp(arr.tobytes(), h, w)


def _make_harness(response: _Resp) -> tuple[RenderHarness, _FakeClient]:
    client = _FakeClient(response)
    harness = RenderHarness(_client=client, _airsim=_FAKE_AIRSIM)
    client.lock = harness._lock
    return harness, client


# --- decode_scene_to_rgb (pure) ---


def test_decode_bgra_to_rgb() -> None:
    rgb = decode_scene_to_rgb(_bgra_response(b=10, g=20, r=30))
    assert rgb.shape == (4, 4, 3) and rgb.dtype == np.uint8
    assert rgb[0, 0, 0] == 30 and rgb[0, 0, 1] == 20 and rgb[0, 0, 2] == 10  # R,G,B
    assert rgb.flags["C_CONTIGUOUS"]


def test_decode_three_channel_bgr() -> None:
    arr = np.zeros((2, 3, 3), dtype=np.uint8)
    arr[..., 0], arr[..., 1], arr[..., 2] = 1, 2, 3  # BGR
    rgb = decode_scene_to_rgb(_Resp(arr.tobytes(), 2, 3))
    assert rgb.shape == (2, 3, 3)
    assert rgb[0, 0, 0] == 3 and rgb[0, 0, 2] == 1


def test_decode_size_mismatch_raises() -> None:
    with pytest.raises(RuntimeError, match="buffer size"):
        decode_scene_to_rgb(_Resp(np.zeros(7, dtype=np.uint8).tobytes(), 4, 4))


# --- teleport + capture via the harness (mocked client) ---


def test_render_reference_row_returns_rgb() -> None:
    harness, _ = _make_harness(_bgra_response(b=10, g=20, r=30))
    rgb = harness.render_reference_row(np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.5]))
    assert rgb.shape == (4, 4, 3) and rgb.dtype == np.uint8
    assert rgb[0, 0, 0] == 30 and rgb[0, 0, 2] == 10  # foot-gun #2: BGR->RGB


def test_teleport_builds_yaw_only_xyzw_quaternion() -> None:
    harness, client = _make_harness(_bgra_response())
    harness.render_reference_row(np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.5]))
    pose, ignore_collision, vehicle = client.pose_calls[0]
    # foot-gun #1: canonical xyzw, yaw-only (x=y=0, z=sin(yaw/2), w=cos(yaw/2)).
    assert pose.orientation.x_val == 0.0 and pose.orientation.y_val == 0.0
    assert math.isclose(pose.orientation.z_val, math.sin(0.25), rel_tol=1e-6)
    assert math.isclose(pose.orientation.w_val, math.cos(0.25), rel_tol=1e-6)
    assert (pose.position.x_val, pose.position.y_val, pose.position.z_val) == (1.0, 2.0, 3.0)
    assert ignore_collision is True and vehicle == VEHICLE_NAME


def test_capture_requests_scene_camera() -> None:
    harness, client = _make_harness(_bgra_response())
    harness.render_reference_row(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    requests, vehicle = client.image_calls[0]
    assert requests[0].camera == CAMERA_NAME
    assert requests[0].image_type == _FAKE_AIRSIM.ImageType.Scene
    assert requests[0].pixels_as_float is False and requests[0].compress is False
    assert vehicle == VEHICLE_NAME


def test_every_client_call_is_lock_wrapped() -> None:
    # The fake client asserts self.lock.locked() inside each RPC; a render exercises both calls.
    harness, client = _make_harness(_bgra_response())
    harness.render_reference_row(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert len(client.pose_calls) == 1 and len(client.image_calls) == 1
    assert not harness._lock.locked()  # released after each call


def test_reference_row_width_validated() -> None:
    harness, _ = _make_harness(_bgra_response())
    with pytest.raises(ValueError, match="6-wide"):
        harness.render_reference_row(np.zeros(5))


def test_harness_module_imports_airsim_free() -> None:
    """SIM-tier harness must import on an airsim-free box (lazy guard): no module-level airsim/cv2."""
    import vllatent.render.harness as h

    heavy = {"airsim", "cv2", "msgpackrpc", "msgpack_rpc_python"}
    tree = ast.parse(Path(h.__file__).read_text())
    for node in tree.body:  # module scope only — function-local imports stay lazy
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names = [node.module]
        for n in names:
            assert n.split(".")[0] not in heavy, f"module-level heavy import {n!r} breaks SIM tier purity"
