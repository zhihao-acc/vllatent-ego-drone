"""Historical latent/cache schemas retained for compatibility (PURE tier).

This module contains shared DINO shape/dtype constants, the historical passive-
video predictor/target seams, and the legacy cache-manifest entry. Current B3-CS
simulator contracts live under :mod:`vllatent.sim`; no four-channel simulator
record is represented by the passive-video types below. The module is stdlib +
NumPy only so CI can enforce the PURE import boundary.

Locked shapes/dtypes come from ``docs/io-contract.md`` / vault
``[[arch-design-2026-06-08-latent-pred]]`` §4. Construction validates at the boundary
and raises ``TypeError`` / ``ValueError`` with a specific message on a contract breach.

The array-bearing records use ``eq=False`` on purpose: numpy ``__eq__`` returns an
array, which would make a dataclass-generated ``__eq__`` ambiguous. They are still
``frozen`` (immutable). ``CacheManifestEntry`` is plain scalars, so it keeps value
equality + a JSON round-trip.
"""
from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from typing import Any

import numpy as np

# --- Locked shape / dtype constants (arch-design-2026-06-08 §4) ---
PATCH_TOKENS = 196          # DINOv3 ViT-B/16 patch tokens per frame (CLS/register dropped)
EMBED_DIM = 768             # DINOv3 / predictor latent dim
HISTORY = 3                 # H — observed history frames
HORIZON = 8                 # T — prediction horizon
DOF = 4                     # historical passive-video VO delta width

LATENT_DTYPE = np.float16   # cached DINOv3 latents on disk
DELTA_DTYPE = np.float32    # historical passive-video VO delta dtype
RGB_DTYPE = np.uint8        # RGB image dtype retained for compatibility
MASK_DTYPE = np.bool_       # validity masks (history padding / language padding): True = real token


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


# --- Student output seams (H3) ---

@dataclass(frozen=True, eq=False)
class PredictorOutput:
    """The latent predictor's rollout ẑ_{t+1..t+T} (arch-design §4; io-contract §0).

    ``predicted_latents`` lives in the FROZEN DINOv3 patch space and is compared against the
    cached ``z_next`` targets, so it carries the cache dtype (fp16); the live predictor computes
    in fp32 and casts at this seam. The leading axis is the horizon T = ``HORIZON``.
    """

    predicted_latents: np.ndarray  # (T,196,768) fp16 — ẑ_{t+1..t+T}, DINOv3 patch space

    def __post_init__(self) -> None:
        _check_array(
            "predicted_latents",
            self.predicted_latents,
            (HORIZON, PATCH_TOKENS, EMBED_DIM),
            dtype=LATENT_DTYPE,
        )


# --- Sports-following target (B1.6) — slim target for sports FPV data ---

@dataclass(frozen=True, eq=False)
class SportsTarget:
    """Per-step target for sports-following training (B1.6, Phase B pivot).

    The sports pipeline target is deliberately small: ``waypoint_4dof`` from MegaSaM ego-motion.
    Scale-free action training derives its B2 targets separately from this motion signal.
    """

    waypoint_4dof: np.ndarray   # (4,) f32  — (dx,dy,dz,dyaw) from MegaSaM VO

    def __post_init__(self) -> None:
        _check_array("waypoint_4dof", self.waypoint_4dof, (DOF,), dtype=DELTA_DTYPE)


Target = SportsTarget


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

    @classmethod
    def required_keys(cls) -> tuple[str, ...]:
        """Entry keys with NO default — exactly what ``manifest.validate_manifest`` requires per
        entry. Derived from the dataclass fields so the manifest validator is type-enforced, not
        hand-kept in sync (M5)."""
        return tuple(
            f.name for f in fields(cls) if f.default is MISSING and f.default_factory is MISSING
        )

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
    "DOF",
    "LATENT_DTYPE",
    "DELTA_DTYPE",
    "RGB_DTYPE",
    "MASK_DTYPE",
    "PredictorOutput",
    "SportsTarget",
    "Target",
    "CacheManifestEntry",
]
