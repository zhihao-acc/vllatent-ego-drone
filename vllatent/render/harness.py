"""AirSim render harness — teleport-to-pose + capture (SIM tier) — Phase-A step 8.

Wraps the AerialVLN-native teleport+capture (third_party/AirVLN
AirVLNSimulatorClientTool.py — reuse/replay only, do NOT modify). For each
reference_path pose: build airsim.Pose with the CORRECTLY REORDERED quaternion
(xyzw; foot-gun #1), simSetVehiclePose(ignore_collision=True, vehicle_name='Drone_1'),
simGetImages on camera "front_0" (ImageType.Scene) -> uint8 -> reshape HxWx3 ->
cv2.cvtColor(BGR2RGB) (foot-gun #2) -> 224x224.

LOAD-BEARING: AirSim msgpack-RPC is single-threaded (tornado IOLoop not re-entrant)
-> wrap EVERY client.X() call in a threading.Lock (foot-gun #3).
Depth is scene-dependent (scenes 1 & 7 = DepthVis; others = DepthPerspective,
clip[0,100]/100); the encoder is RGB-only so depth is captured raw only if needed
for Phase C.

airsim imports are LAZY. STUB at scaffold time; CODE is writable autonomously but
RUNNING needs docker + manual UE4 launch (wait port 41451) — USER-GATED.

See plans/phase-a-data-and-io-contract.md step 8.
"""
from __future__ import annotations

import threading

CAMERA_NAME = "front_0"
VEHICLE_NAME = "Drone_1"
RGB_HW = (224, 224)
DEPTH_HW = (256, 256)
DEPTH_VIS_SCENES = (1, 7)  # use DepthVis; all others use DepthPerspective


class RenderHarness:  # pragma: no cover - implemented in step 8
    """Teleport-to-pose + capture over an AirSim msgpack-RPC client (Lock-serialized)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 41451) -> None:
        self._lock = threading.Lock()
        raise NotImplementedError("RenderHarness lands in Phase-A step 8")


__all__ = ["RenderHarness", "CAMERA_NAME", "VEHICLE_NAME", "RGB_HW", "DEPTH_HW", "DEPTH_VIS_SCENES"]
