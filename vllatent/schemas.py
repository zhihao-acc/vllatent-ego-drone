"""Loader output-tuple SCHEMAS (PURE tier) — Phase-A step 3.

Frozen dataclasses for the tuple the cached-latent loader emits (arch-design §6
item 5). numpy-typed, no torch. STUB at scaffold time; implemented in step 3.

Target tuple (per step):
    (z_t, history_latents, lang_tokens, action_id, z_next, delta_4dof, future_frame_rgb)
where
    z_t / z_next        : (196, 768) fp16   DINOv3 patch tokens (cached)
    history_latents     : (H=3, 196, 768) fp16
    lang_tokens         : per the frozen text-tower contract
    action_id           : int in [0, 7]     AerialVLN discrete action
    delta_4dof          : (4,) float32       (dx, dy, dz, dyaw) AirSim-NED body, yaw-only
    future_frame_rgb    : uint8 RGB          Phase-C V-JEPA-2 target (optional)

See plans/phase-a-data-and-io-contract.md step 3 and docs/io-contract.md.
"""
from __future__ import annotations

# Implemented in Phase-A step 3 (StepSample, EpisodeRecord, CacheManifestEntry).
__all__: list[str] = []
