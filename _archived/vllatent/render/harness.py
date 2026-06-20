"""AirSim teleport-to-pose + capture render harness (SIM tier) — Phase-A step A5.13 (was step 8).

For each AerialVLN ``reference_path`` pose (Euler ``[x,y,z,pitch,roll,yaw]``, NED): build an
``airsim.Pose`` with the yaw-only orientation (pitch=roll≡0; the quaternion is built via
``frames.xyzw_from_yaw`` = ``airsim.to_quaternion(0,0,yaw)``, canonical **xyzw** — foot-gun #1),
``simSetVehiclePose(ignore_collision=True)``, then ``simGetImages`` the ``Scene`` and decode it to an
RGB ``uint8`` ``(H,W,3)`` frame.

**Three foot-guns, all handled here:**
  #1 quaternion order — orientation built in canonical xyzw (``airsim.Quaternionr(x,y,z,w)``).
  #2 BGR→RGB — AirSim ``Scene`` is **BGRA**; we drop alpha (``[:,:,:3]`` → BGR) then reverse to RGB.
  #3 single-threaded msgpack-RPC (tornado IOLoop not re-entrant) — **every** ``client.X()`` call is
     wrapped in one ``threading.Lock``.

Output is the sim's native capture resolution; configure the AirSim ``settings.json`` CaptureSettings
to ``RGB_HW`` (224²), or let the DINOv3 processor resize at encode (A5.14 calls ``encode_rgb`` on this
RGB — the renderer owns the BGR→RGB flip, so the encoder must NOT flip again).

``airsim`` import is LAZY (inside ``_connect``); the module imports on an airsim-free box. The unit
test injects a fake client + fake airsim module (no real sim), so it runs in CI; RUNNING the live
render needs the fly0-m1 docker + a UE4 scene hot on port 41451 — USER-GATED.

CLI (live, USER-GATED):  python -m vllatent.render --episode <json> --scene 1 --out <dir>

See plans/phase-a5-replan-postpivot.md A5.13.
"""
from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from vllatent.frames import REFERENCE_PATH_ROW_WIDTH, REFERENCE_PATH_YAW_INDEX, xyzw_from_yaw

# airsim is imported LAZILY inside _connect (SIM tier); access at runtime is via self._airsim (Any).
CAMERA_NAME = "front_center"
VEHICLE_NAME = "drone_1"
_SETTLE_S = 0.2
RGB_HW = (224, 224)              # the DINOv3 encoder target (sim settings.json CaptureSettings)
DEPTH_HW = (256, 256)
DEPTH_VIS_SCENES = (1, 7)        # use DepthVis; all others use DepthPerspective (Phase C, if needed)


def _connect(host: str, port: int, lock: threading.Lock) -> tuple[Any, Any]:
    """Lazy-import airsim, open a client, confirm the connection (under the lock). USER-GATED path."""
    import airsim  # lazy by design (SIM tier) — never imported on an airsim-free box

    client = airsim.MultirotorClient(ip=host, port=port)
    with lock:
        client.confirmConnection()
    return airsim, client


def decode_scene_to_rgb(response: Any) -> np.ndarray:
    """AirSim ``Scene`` ``ImageResponse`` -> ``(H,W,3)`` uint8 **RGB** (pure numpy; no cv2).

    Scene is packed **BGRA** (4 channels); we drop alpha then reverse BGR→RGB (foot-gun #2).
    Handles a 3-channel buffer too (some AirSim builds). Raises on a size mismatch.
    """
    buf = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    h, w = int(response.height), int(response.width)
    if buf.size == h * w * 4:
        bgr = buf.reshape(h, w, 4)[:, :, :3]   # BGRA -> BGR (drop alpha)
    elif buf.size == h * w * 3:
        bgr = buf.reshape(h, w, 3)
    else:
        raise RuntimeError(f"Scene buffer size {buf.size} != {h}x{w}x(3|4)")
    return np.ascontiguousarray(bgr[:, :, ::-1])  # foot-gun #2: BGR -> RGB


class RenderHarness:
    """Teleport-to-pose + capture over an AirSim msgpack-RPC client (every call Lock-serialized)."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 41451,
        vehicle_name: str = VEHICLE_NAME,
        camera_name: str = CAMERA_NAME,
        *,
        _client: Any = None,
        _airsim: Any = None,
    ) -> None:
        self._lock = threading.Lock()
        self.vehicle_name = vehicle_name
        self.camera_name = camera_name
        self._armed = False
        if _client is not None:
            self._airsim = _airsim
            self._client = _client
            self._armed = True
        else:
            self._airsim, self._client = _connect(host, port, self._lock)

    def _ensure_armed(self) -> None:
        """Arm + takeoff once (fly0 pattern). Required before simSetVehiclePose / simGetImages."""
        if self._armed:
            return
        with self._lock:
            self._client.enableApiControl(True, vehicle_name=self.vehicle_name)
            self._client.armDisarm(True, vehicle_name=self.vehicle_name)
            self._client.takeoffAsync(vehicle_name=self.vehicle_name).join()
        time.sleep(0.5)
        self._armed = True

    def teleport(self, position_ned: np.ndarray, yaw: float) -> None:
        """Set the vehicle pose to ``position_ned`` (NED x,y,z) + yaw-only orientation (xyzw)."""
        self._ensure_armed()
        x, y, z, w = (float(v) for v in xyzw_from_yaw(float(yaw)))  # canonical xyzw (foot-gun #1)
        pose = self._airsim.Pose(
            self._airsim.Vector3r(float(position_ned[0]), float(position_ned[1]), float(position_ned[2])),
            self._airsim.Quaternionr(x, y, z, w),
        )
        with self._lock:  # foot-gun #3
            self._client.simSetVehiclePose(pose, True, self.vehicle_name)
        time.sleep(_SETTLE_S)

    def capture_rgb(self) -> np.ndarray:
        """Capture the ``Scene`` camera and return an ``(H,W,3)`` uint8 RGB frame."""
        request = self._airsim.ImageRequest(
            self.camera_name, self._airsim.ImageType.Scene, False, False  # pixels_as_float, compress
        )
        with self._lock:  # foot-gun #3
            responses = self._client.simGetImages([request], vehicle_name=self.vehicle_name)
        return decode_scene_to_rgb(responses[0])

    def render_reference_row(self, row: np.ndarray) -> np.ndarray:
        """Teleport to one ``reference_path`` Euler row ``[x,y,z,pitch,roll,yaw]`` and capture RGB."""
        if row.shape[-1] != REFERENCE_PATH_ROW_WIDTH:
            raise ValueError(f"reference_path row must be {REFERENCE_PATH_ROW_WIDTH}-wide, got {row.shape}")
        self.teleport(row[:3], float(row[REFERENCE_PATH_YAW_INDEX]))
        return self.capture_rgb()


__all__ = [
    "RenderHarness",
    "decode_scene_to_rgb",
    "CAMERA_NAME",
    "VEHICLE_NAME",
    "RGB_HW",
    "DEPTH_HW",
    "DEPTH_VIS_SCENES",
]
