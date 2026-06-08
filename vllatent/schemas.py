"""Loader output-tuple SCHEMAS (PURE tier) — Phase-A step 3.

Frozen dataclasses for the tuple the cached-latent loader emits (arch-design §6
item 5), the parsed AerialVLN episode (output of ``vllatent.audit.parse_episode``),
and one cache-manifest entry (the typed view of ``vllatent.manifest``'s per-episode
entry). numpy-typed; **stdlib + numpy only** (no torch / no airsim / no sibling) so
CI imports this module. Each field documents its frame / dtype / order.

Locked shapes/dtypes come from ``docs/io-contract.md`` / vault
``[[arch-design-2026-06-08-latent-pred]]`` §4. Construction validates at the boundary
and raises ``TypeError`` / ``ValueError`` with a specific message on a contract breach.

The array-bearing records use ``eq=False`` on purpose: numpy ``__eq__`` returns an
array, which would make a dataclass-generated ``__eq__`` ambiguous. They are still
``frozen`` (immutable). ``CacheManifestEntry`` is plain scalars, so it keeps value
equality + a JSON round-trip.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# --- Locked shape / dtype constants (arch-design-2026-06-08 §4) ---
PATCH_TOKENS = 196          # DINOv3 ViT-B/16 patch tokens per frame (CLS/register dropped)
EMBED_DIM = 768             # DINOv3 / predictor latent dim
HISTORY = 3                 # H — history frames fed to the predictor (DINO-WM default)
HORIZON = 4                 # T — prediction horizon (documented; not a StepSample field)
N_ACTIONS = 8               # AerialVLN discrete action set, ids 0..7
DOF = 4                     # continuous waypoint DoF: (dx, dy, dz, dyaw)

LATENT_DTYPE = np.float16   # cached DINOv3 latents on disk
DELTA_DTYPE = np.float32    # continuous 4-DoF delta, AirSim-NED body frame
RGB_DTYPE = np.uint8        # rendered frame (RGB, after BGR->RGB)


def _check_array(
    name: str,
    arr: object,
    shape: tuple[int | None, ...],
    *,
    dtype: Any = None,
    kind: str | None = None,
) -> None:
    """Validate an array's ndim/shape/dtype. ``None`` axes are wildcards.

    Pass ``dtype`` for an exact dtype, or ``kind`` ('f', 'i', 'u', ...) for a looser
    dtype-kind check (used where parsed-JSON floats may be f32 or f64).
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name}: expected np.ndarray, got {type(arr).__name__}")
    if arr.ndim != len(shape):
        raise ValueError(f"{name}: expected {len(shape)} dims, got {arr.ndim} (shape {arr.shape})")
    for axis, exp in enumerate(shape):
        if exp is not None and arr.shape[axis] != exp:
            raise ValueError(f"{name}: axis {axis} expected {exp}, got {arr.shape[axis]} (shape {arr.shape})")
    if dtype is not None and arr.dtype != np.dtype(dtype):
        raise ValueError(f"{name}: expected dtype {np.dtype(dtype)}, got {arr.dtype}")
    if kind is not None and arr.dtype.kind != kind:
        raise ValueError(f"{name}: expected dtype kind {kind!r}, got {arr.dtype} (kind {arr.dtype.kind!r})")


@dataclass(frozen=True, eq=False)
class StepSample:
    """One training tuple emitted by the cached-latent loader (arch-design §6 item 5).

    All latents are the FROZEN DINOv3 patch space; ``delta_4dof`` is AirSim-NED body
    (the model's native convention — the NED->FLU->ENU remap is Phase D, see
    docs/io-contract.md). ``future_frame_rgb`` is the optional Phase-C V-JEPA-2 target.
    """

    z_t: np.ndarray              # (196,768) fp16   — DINOv3 patch tokens, obs @ t (cached)
    history_latents: np.ndarray  # (3,196,768) fp16 — z_{t-2..t}; padded+masked at episode start
    lang_tokens: np.ndarray      # (M,768) fp16     — frozen text-tower tokens (cached per episode)
    action_id: int               # int in [0,7]     — AerialVLN discrete actions[t]
    z_next: np.ndarray           # (196,768) fp16   — DINOv3 latent of next obs = prediction target
    delta_4dof: np.ndarray       # (4,) f32         — (dx,dy,dz,dyaw) AirSim-NED body, yaw-only
    future_frame_rgb: np.ndarray | None = None  # (H,W,3) uint8 RGB — Phase-C V-JEPA-2 target (optional)

    def __post_init__(self) -> None:
        _check_array("z_t", self.z_t, (PATCH_TOKENS, EMBED_DIM), dtype=LATENT_DTYPE)
        _check_array("history_latents", self.history_latents, (HISTORY, PATCH_TOKENS, EMBED_DIM), dtype=LATENT_DTYPE)
        _check_array("lang_tokens", self.lang_tokens, (None, EMBED_DIM), dtype=LATENT_DTYPE)
        _check_array("z_next", self.z_next, (PATCH_TOKENS, EMBED_DIM), dtype=LATENT_DTYPE)
        _check_array("delta_4dof", self.delta_4dof, (DOF,), dtype=DELTA_DTYPE)
        if isinstance(self.action_id, bool) or not isinstance(self.action_id, (int, np.integer)):
            raise TypeError(f"action_id: expected int, got {type(self.action_id).__name__}")
        if not (0 <= int(self.action_id) < N_ACTIONS):
            raise ValueError(f"action_id: expected 0..{N_ACTIONS - 1}, got {self.action_id}")
        if self.future_frame_rgb is not None:
            _check_array("future_frame_rgb", self.future_frame_rgb, (None, None, 3), dtype=RGB_DTYPE)


@dataclass(frozen=True, eq=False)
class EpisodeRecord:
    """A parsed AerialVLN episode (output of ``vllatent.audit.parse_episode``).

    ``start_rotation`` is a QUATERNION, w-FIRST in the raw JSON, reordered to canonical
    ``xyzw`` on parse. ``reference_path`` rows are EULER ``[x,y,z,pitch,roll,yaw]`` (radians,
    6-wide, pitch=roll==0; yaw = row[5]) — NOT quaternions (confirmed step 5b). ``actions``
    index-aligns with the transitions ``reference_path[t] -> reference_path[t+1]``.
    """

    episode_id: str
    trajectory_id: str
    scene_id: int
    instruction_text: str
    start_position: np.ndarray       # (3,) float — NED x,y,z
    start_rotation_xyzw: np.ndarray  # (4,) float — canonical xyzw quaternion (reordered from w-FIRST)
    goal_positions: np.ndarray       # (G,3) float — goals[].position (NED)
    actions: np.ndarray              # (N,) int — discrete ids 0..7
    reference_path: np.ndarray       # (P,6) float — [x,y,z,pitch,roll,yaw] per pose (Euler radians)

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str):
            raise TypeError(f"episode_id: expected str, got {type(self.episode_id).__name__}")
        if not isinstance(self.trajectory_id, str):
            raise TypeError(f"trajectory_id: expected str, got {type(self.trajectory_id).__name__}")
        if not isinstance(self.instruction_text, str):
            raise TypeError(f"instruction_text: expected str, got {type(self.instruction_text).__name__}")
        if isinstance(self.scene_id, bool) or not isinstance(self.scene_id, (int, np.integer)):
            raise TypeError(f"scene_id: expected int, got {type(self.scene_id).__name__}")
        _check_array("start_position", self.start_position, (3,), kind="f")
        _check_array("start_rotation_xyzw", self.start_rotation_xyzw, (4,), kind="f")
        _check_array("goal_positions", self.goal_positions, (None, 3), kind="f")
        _check_array("actions", self.actions, (None,), kind="i")
        _check_array("reference_path", self.reference_path, (None, 6), kind="f")


@dataclass(frozen=True)
class CacheManifestEntry:
    """One per-episode entry of the cache manifest (typed view of ``vllatent.manifest``).

    ``to_dict()`` emits exactly the keys ``vllatent.manifest.validate_manifest`` requires
    of an entry (``episode_id, scene_id, n_frames, latent_path``) plus ``trajectory_id``,
    so an entry round-trips through the manifest JSON.
    """

    episode_id: str
    scene_id: int
    n_frames: int
    latent_path: str          # path to the per-episode latent dump, relative to the cache dir
    trajectory_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "trajectory_id": self.trajectory_id,
            "scene_id": self.scene_id,
            "n_frames": self.n_frames,
            "latent_path": self.latent_path,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CacheManifestEntry:
        return cls(
            episode_id=str(d["episode_id"]),
            scene_id=int(d["scene_id"]),
            n_frames=int(d["n_frames"]),
            latent_path=str(d["latent_path"]),
            trajectory_id=str(d.get("trajectory_id", "")),
        )


__all__ = [
    "PATCH_TOKENS",
    "EMBED_DIM",
    "HISTORY",
    "HORIZON",
    "N_ACTIONS",
    "DOF",
    "LATENT_DTYPE",
    "DELTA_DTYPE",
    "RGB_DTYPE",
    "StepSample",
    "EpisodeRecord",
    "CacheManifestEntry",
]
