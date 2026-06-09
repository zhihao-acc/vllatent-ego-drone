"""Loader output-tuple SCHEMAS (PURE tier) — Phase-A step 3.

Frozen dataclasses for the tuple the cached-latent loader emits (arch-design §6
item 5), the **student output seams** (predictor rollout / trust readout / waypoint
— H3, typed so an ablation is a config flag not code surgery), the **teacher
distillation seam** (``TeacherOutput`` / ``OracleTarget`` — A5.9, the per-step target the
student distills against), the parsed AerialVLN episode (output of
``vllatent.audit.parse_episode``), and one cache-manifest entry (the typed view of
``vllatent.manifest``'s per-episode entry). numpy-typed; **stdlib + numpy only** (no torch /
no airsim / no sibling) so CI imports this module. Each field documents its frame / dtype / order.

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
HISTORY = 3                 # H — history frames fed to the predictor (DINO-WM default)
HORIZON = 4                 # T — prediction horizon (documented; not a StepSample field)
N_ACTIONS = 8               # AerialVLN discrete action set, ids 0..7
DOF = 4                     # continuous waypoint DoF: (dx, dy, dz, dyaw)
TEACHER_DOF = 6             # WorldVLN teacher action head: absolute [roll,yaw,pitch,x,y,z] (A5.8)

LATENT_DTYPE = np.float16   # cached DINOv3 latents on disk
DELTA_DTYPE = np.float32    # continuous 4-DoF delta, AirSim-NED body frame
RGB_DTYPE = np.uint8        # rendered frame (RGB, after BGR->RGB)
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


@dataclass(frozen=True, eq=False)
class StepSample:
    """One training tuple emitted by the cached-latent loader (arch-design §6 item 5).

    All latents are the FROZEN DINOv3 patch space; ``delta_4dof`` is AirSim-NED body
    (the model's native convention — the NED->FLU->ENU remap is Phase D, see
    docs/io-contract.md). ``future_frame_rgb`` is the optional Phase-C V-JEPA-2 target.

    Two padding masks make the variable-validity inputs explicit (M4): ``history_mask``
    marks which of the H history slots are real vs zero-padded (block-causal at an episode
    start), and ``lang_mask`` marks real language tokens vs padding (so attention can ignore
    the pad). Both are ``True`` = real, ``False`` = padding; the loader (A5.15) consumes them.
    """

    z_t: np.ndarray              # (196,768) fp16   — DINOv3 patch tokens, obs @ t (cached)
    history_latents: np.ndarray  # (3,196,768) fp16 — z_{t-2..t}; zero-padded at episode start
    history_mask: np.ndarray     # (3,) bool        — True = real history frame, False = padding
    lang_tokens: np.ndarray      # (M,768) fp16     — frozen text-tower tokens (cached per episode)
    lang_mask: np.ndarray        # (M,) bool        — True = real language token, False = padding
    action_id: int               # int in [0,7]     — AerialVLN discrete actions[t]
    z_next: np.ndarray           # (196,768) fp16   — DINOv3 latent of next obs = prediction target
    delta_4dof: np.ndarray       # (4,) f32         — (dx,dy,dz,dyaw) AirSim-NED body, yaw-only
    future_frame_rgb: np.ndarray | None = None  # (H,W,3) uint8 RGB — Phase-C V-JEPA-2 target (optional)

    def __post_init__(self) -> None:
        _check_array("z_t", self.z_t, (PATCH_TOKENS, EMBED_DIM), dtype=LATENT_DTYPE)
        _check_array("history_latents", self.history_latents, (HISTORY, PATCH_TOKENS, EMBED_DIM), dtype=LATENT_DTYPE)
        _check_array("history_mask", self.history_mask, (HISTORY,), dtype=MASK_DTYPE)
        _check_array("lang_tokens", self.lang_tokens, (None, EMBED_DIM), dtype=LATENT_DTYPE)
        _check_array("lang_mask", self.lang_mask, (None,), dtype=MASK_DTYPE)
        if self.lang_mask.shape[0] != self.lang_tokens.shape[0]:
            raise ValueError(
                f"lang_mask: length {self.lang_mask.shape[0]} must match lang_tokens "
                f"M={self.lang_tokens.shape[0]}"
            )
        _check_array("z_next", self.z_next, (PATCH_TOKENS, EMBED_DIM), dtype=LATENT_DTYPE)
        _check_array("delta_4dof", self.delta_4dof, (DOF,), dtype=DELTA_DTYPE)
        if isinstance(self.action_id, bool) or not isinstance(self.action_id, (int, np.integer)):
            raise TypeError(f"action_id: expected int, got {type(self.action_id).__name__}")
        if not (0 <= int(self.action_id) < N_ACTIONS):
            raise ValueError(f"action_id: expected 0..{N_ACTIONS - 1}, got {self.action_id}")
        if self.future_frame_rgb is not None:
            _check_array("future_frame_rgb", self.future_frame_rgb, (None, None, 3), dtype=RGB_DTYPE)


# --- Student output seams (H3) — typed so "−trust" / swap-predictor is a config flag, not surgery ---

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


@dataclass(frozen=True, eq=False)
class TrustReadout:
    """The deployed single-pass trust/horizon head readout (arch-design §9.6; io-contract §1c).

    ``p_commit`` = per-step "still-trusted" probabilities p_j ∈ [0,1]^T; ``k_star`` = the soft
    expected horizon k* = Σ_j Π_{i≤j} p_i ∈ [0,T] (how far to commit); ``sigma`` = a scalar
    predictive uncertainty σ ≥ 0. ONE forward pass — the K=5 ensemble teacher is Phase C.
    """

    p_commit: np.ndarray  # (T,) float — per-step commit probabilities in [0,1]
    k_star: float         # soft expected horizon in [0, T]
    sigma: float          # scalar predictive uncertainty >= 0

    def __post_init__(self) -> None:
        _check_array("p_commit", self.p_commit, (HORIZON,), kind="f")
        if not bool(np.all((self.p_commit >= 0.0) & (self.p_commit <= 1.0))):
            raise ValueError(f"p_commit: expected all values in [0,1], got {self.p_commit}")
        for name in ("k_star", "sigma"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
                raise TypeError(f"{name}: expected float, got {type(v).__name__}")
        if not (0.0 <= float(self.k_star) <= HORIZON):
            raise ValueError(f"k_star: expected 0..{HORIZON}, got {self.k_star}")
        # Two-sided/finite, symmetric with k_star + p_commit (a one-sided check would let NaN/inf pass).
        if not np.isfinite(float(self.sigma)) or float(self.sigma) < 0.0:
            raise ValueError(f"sigma: expected a finite value >= 0, got {self.sigma}")


@dataclass(frozen=True, eq=False)
class Waypoint:
    """The waypoint head output: continuous 4-DoF (Δx,Δy,Δz,Δψ), AirSim-NED body, yaw-only.

    Native model convention (arch-design §9.7). The NED→FLU→world-ENU remap is Phase D — this
    seam carries the raw body-frame delta, NOT a world pose. Same quantity as ``StepSample``'s
    ``delta_4dof`` GT target, but here it is the model's PREDICTED waypoint.
    """

    delta_4dof: np.ndarray  # (4,) f32 — (dx,dy,dz,dyaw) AirSim-NED body, yaw-only

    def __post_init__(self) -> None:
        _check_array("delta_4dof", self.delta_4dof, (DOF,), dtype=DELTA_DTYPE)


# --- Teacher distillation seam (A5.9) — the (StepSample, OracleTarget) the student trains against ---

@dataclass(frozen=True, eq=False)
class TeacherOutput:
    """Raw frozen-WorldVLN inference for one step: K stochastic rollouts of the 6-DoF pose (A5.9).

    A5.8 confirmed WorldVLN inference is STOCHASTIC by default (top_k/top_p, per-segment seed), so
    K rollouts of the same input DIFFER — that spread is the trust-oracle disagreement signal, FREE
    (no MC-dropout). Pose order is the teacher-native ``[roll,yaw,pitch,x,y,z]`` (SE(3)-integrated,
    A5.8). The 6->4 projection to the student waypoint + the disagreement scalarization happen at
    cache-build (A5.14); this seam pins the raw rollouts + how the per-DoF spread is read.
    """

    rollouts_pose6: np.ndarray  # (K,6) float — K stochastic teacher poses [roll,yaw,pitch,x,y,z]

    def __post_init__(self) -> None:
        _check_array("rollouts_pose6", self.rollouts_pose6, (None, TEACHER_DOF), kind="f")
        if self.rollouts_pose6.shape[0] < 1:
            raise ValueError(f"rollouts_pose6: expected K>=1 rollouts, got {self.rollouts_pose6.shape[0]}")

    def rollout_spread(self) -> np.ndarray:
        """Per-DoF standard deviation across the K rollouts, shape ``(6,)`` — the raw disagreement.

        A5.14 scalarizes this (over the 4 student-relevant channels yaw,x,y,z) into
        ``OracleTarget.disagreement``; Phase-C calibrates the gate threshold.
        """
        return np.std(self.rollouts_pose6, axis=0)


@dataclass(frozen=True, eq=False)
class OracleTarget:
    """Per-step distillation TARGET (frozen seam, A5.9) the student trains against.

    Pairs 1:1 with a :class:`StepSample` (the loader emits ``(StepSample, OracleTarget)``; A5.15).
    Combines the frozen WorldVLN teacher (waypoint + rollout disagreement) with the independent
    V-JEPA-2 surprise gate. ``waypoint_4dof`` is the 6->4-projected (drop roll/pitch + abs->body-delta)
    student target; ``teacher_pose6`` + ``rollpitch_resid`` are kept as provenance + a
    lossless-projection audit (roll/pitch are ≈0 on AerialVLN by construction — verify, don't assume).
    """

    waypoint_4dof: np.ndarray   # (4,) f32  — student target (dx,dy,dz,dyaw), AirSim-NED body
    teacher_pose6: np.ndarray   # (6,) float — raw teacher pose [roll,yaw,pitch,x,y,z] (provenance)
    rollpitch_resid: float      # |roll| + |pitch| of teacher_pose6 (>= 0; ≈0 audit on AerialVLN)
    disagreement: float         # K-rollout spread scalar (>= 0) — trust-horizon distillation target
    vjepa_surprise: float       # V-JEPA-2 surprise (>= 0) — independent second gate

    def __post_init__(self) -> None:
        _check_array("waypoint_4dof", self.waypoint_4dof, (DOF,), dtype=DELTA_DTYPE)
        _check_array("teacher_pose6", self.teacher_pose6, (TEACHER_DOF,), kind="f")
        for name in ("rollpitch_resid", "disagreement", "vjepa_surprise"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
                raise TypeError(f"{name}: expected float, got {type(v).__name__}")
            if not np.isfinite(float(v)) or float(v) < 0.0:
                raise ValueError(f"{name}: expected a finite value >= 0, got {v}")


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
    "N_ACTIONS",
    "DOF",
    "TEACHER_DOF",
    "LATENT_DTYPE",
    "DELTA_DTYPE",
    "RGB_DTYPE",
    "MASK_DTYPE",
    "StepSample",
    "PredictorOutput",
    "TrustReadout",
    "Waypoint",
    "TeacherOutput",
    "OracleTarget",
    "EpisodeRecord",
    "CacheManifestEntry",
]
