"""Render -> encode -> CACHE orchestrator + provenance manifest (SIM+TORCH tier) — A5.14.

For each episode: render every ``reference_path`` pose via the AirSim harness
(``vllatent.render``), **center-crop to square then resize to 224²** (normalizing the
sim-native ``(480,640)`` so DINOv3 and V-JEPA-2 see identical pixels), encode with
DINOv3 + CLIP-text + run the WorldVLN K-rollout teacher + V-JEPA-2 surprise verifier,
and write a per-episode ``.npz`` EXACTLY matching the ``CachedLatentDataset`` read-contract
(A5.15) plus the provenance manifest (M5).

**Five lazy seams (heavy imports live inside ``_load_*`` helpers only):**
  render     — ``vllatent.render.harness.RenderHarness``
  vision     — ``vllatent.encode.dinov3.DinoV3Encoder``
  text       — ``vllatent.encode.text.ClipTextEncoder``
  teacher    — ``vllatent.teacher.worldvln.WorldVLNTeacherClient``
  verifier   — ``vllatent.verify.vjepa2.VJEPA2SurpriseVerifier``

Cache writes are torch-free (numpy ``.npz``); the heavy stacks enter only via the lazy
seams above. ``vllatent/cache.py`` itself imports on a pure box (import-smoke safe).

The manifest records teacher/render provenance (worldvln model id, render transform hash);
``build_manifest`` from ``vllatent.manifest`` is the single builder.

CLI (USER-GATED):
  python -m vllatent.cache build \\
    --slice data/aerialvln_json/train.slice.json --limit 5 \\
    --scenes-root /opt/aerialvln --out data/latent_cache/

See plans/phase-a5-replan-postpivot.md A5.14.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from vllatent.actions import pose_pair_to_body_delta
from vllatent.config import Config
from vllatent.frames import xyzw_from_yaw
from vllatent.manifest import build_manifest, validate_manifest, write_manifest
from vllatent.schemas import (
    DELTA_DTYPE,
    DOF,
    TEACHER_DOF,
    CacheManifestEntry,
)

if TYPE_CHECKING:
    from vllatent.encode.dinov3 import DinoV3Encoder
    from vllatent.encode.text import ClipTextEncoder
    from vllatent.render.harness import RenderHarness
    from vllatent.teacher.worldvln import WorldVLNTeacherClient
    from vllatent.verify.vjepa2 import VJEPA2SurpriseVerifier

TARGET_HW = 224


def center_crop_and_resize(frame: np.ndarray, target_hw: int = TARGET_HW) -> np.ndarray:
    """Center-crop an ``(H,W,3)`` uint8 frame to square, then resize to ``target_hw²``.

    Ensures DINOv3 and V-JEPA-2 see identical pixels from the same render (avoids the
    aspect-distortion foot-gun when encoders silently resize differently).
    Pure numpy + lazy cv2 (the resize needs interpolation; cv2 is already in the [torch]
    extra via opencv-python).
    """
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"frame: expected (H,W,3), got {getattr(frame, 'shape', type(frame))}")
    h, w = frame.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    cropped = frame[y0 : y0 + side, x0 : x0 + side]
    if cropped.shape[0] == target_hw and cropped.shape[1] == target_hw:
        return np.ascontiguousarray(cropped)
    import cv2  # lazy (SIM tier only; the [torch] extra ships opencv-python)

    resized = cv2.resize(cropped, (target_hw, target_hw), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(resized)


def _teacher_6to4(pose6: np.ndarray) -> tuple[np.ndarray, float]:
    """Project a teacher ``(6,)`` seam row ``[roll,yaw,pitch,x,y,z]`` (m, rad) to the student's
    ``(4,)`` ``[dx,dy,dz,dyaw_deg]`` (m, deg) + the roll/pitch residual.

    The 6→4 projection: drop roll (idx 0) and pitch (idx 2); keep x,y,z (idx 3,4,5) as-is
    (already body-frame deltas in metres); convert yaw (idx 1) rad → deg to match
    ``delta_4dof`` (m, deg-yaw).
    """
    arr = np.asarray(pose6, dtype=np.float64)
    if arr.shape != (TEACHER_DOF,):
        raise ValueError(f"pose6: expected ({TEACHER_DOF},), got {arr.shape}")
    roll, yaw_rad, pitch = float(arr[0]), float(arr[1]), float(arr[2])
    dx, dy, dz = float(arr[3]), float(arr[4]), float(arr[5])
    dyaw_deg = math.degrees(yaw_rad)
    waypoint = np.array([dx, dy, dz, dyaw_deg], dtype=DELTA_DTYPE)
    residual = abs(roll) + abs(pitch)
    return waypoint, float(residual)


def _disagreement_scalar(spread: np.ndarray) -> float:
    """Scalarize the ``(6,)`` rollout spread over the 4 student-relevant channels.

    Channels: yaw(1), x(3), y(4), z(5) in seam order ``[roll,yaw,pitch,x,y,z]``.
    """
    return float(np.mean(spread[[1, 3, 4, 5]]))


def _render_transform_hash(target_hw: int) -> str:
    """A short deterministic hash of the render→encode normalization, for the manifest."""
    desc = f"center_crop_to_square+resize_area_{target_hw}x{target_hw}"
    return hashlib.sha256(desc.encode()).hexdigest()[:16]


def build_episode_cache(
    episode: dict[str, Any],
    *,
    renderer: RenderHarness,
    vision_encoder: DinoV3Encoder,
    text_encoder: ClipTextEncoder,
    teacher_client: WorldVLNTeacherClient,
    verifier: VJEPA2SurpriseVerifier,
    config: Config,
    target_hw: int = TARGET_HW,
) -> dict[str, np.ndarray]:
    """Build all cache arrays for one episode. Returns the dict ready for ``np.savez``.

    Callers own parsing the episode JSON (``vllatent.audit.parse_episode``) and writing the
    ``.npz``; this function owns the render→encode→teacher→verifier pipeline.
    """
    from vllatent.audit import parse_episode
    from vllatent.teacher.worldvln import teacher_outputs_from_rollouts

    ep = parse_episode(episode)
    ref = ep.reference_path  # (N, 6) Euler [x,y,z,pitch,roll,yaw]
    n = ref.shape[0]

    # --- 1. Render + normalize --------------------------------------------------
    frames_224: list[np.ndarray] = []
    for i in range(n):
        raw_rgb = renderer.render_reference_row(ref[i])
        frames_224.append(center_crop_and_resize(raw_rgb, target_hw))

    # --- 2. DINOv3 encode --------------------------------------------------------
    latents = np.stack([vision_encoder.encode_rgb(f) for f in frames_224])  # (N, 196, 768) fp16

    # --- 3. CLIP text encode (once per episode) ----------------------------------
    lang_tokens = text_encoder.encode(ep.instruction_text)  # (M, 768) fp16

    # --- 4. WorldVLN K-rollout teacher -------------------------------------------
    rollouts_seam, _ = teacher_client.k_rollout_segment(
        [frames_224[0]], ep.instruction_text, config=config,
    )
    per_step_teachers = teacher_outputs_from_rollouts(rollouts_seam)  # list of T TeacherOutput
    t_segment = len(per_step_teachers)

    # --- 5. V-JEPA-2 surprise (per transition, context=current, future=next) -----
    surprises: list[float] = []
    for i in range(n - 1):
        ctx = frames_224[i][np.newaxis]   # (1, 224, 224, 3)
        fut = frames_224[i + 1][np.newaxis]
        surprises.append(verifier.scalar_surprise(ctx, fut))
    surprises.append(0.0)  # terminal STOP slot (t=N-1): unused by the loader

    # --- 6. Assemble per-step arrays ---------------------------------------------
    actions = ep.actions.astype(np.int64)
    deltas = np.zeros((n, DOF), dtype=DELTA_DTYPE)
    waypoint_4dof = np.zeros((n, DOF), dtype=DELTA_DTYPE)
    teacher_pose6 = np.zeros((n, TEACHER_DOF), dtype=np.float32)
    rollpitch_resid = np.zeros((n,), dtype=np.float32)
    disagreement = np.zeros((n,), dtype=np.float32)
    vjepa_surprise = np.array(surprises, dtype=np.float32)

    for i in range(n):
        # GT body delta from consecutive reference_path poses
        if i < n - 1:
            pos_a, yaw_a = ref[i, :3], float(ref[i, 5])
            pos_b, yaw_b = ref[i + 1, :3], float(ref[i + 1, 5])
            pose_a = (pos_a, xyzw_from_yaw(yaw_a))
            pose_b = (pos_b, xyzw_from_yaw(yaw_b))
            deltas[i] = pose_pair_to_body_delta(pose_a, pose_b)

        # Teacher projection (only for steps within the first segment)
        if i < t_segment:
            t_out = per_step_teachers[i]
            mean_pose6 = np.mean(t_out.rollouts_pose6, axis=0)  # (6,) mean across K
            waypoint_4dof[i], rollpitch_resid[i] = _teacher_6to4(mean_pose6)
            teacher_pose6[i] = mean_pose6.astype(np.float32)
            disagreement[i] = _disagreement_scalar(t_out.rollout_spread())
        # Steps beyond the first segment stay zero (the teacher covers 16 steps per
        # segment; a full episode may be longer — A5.17 handles multi-segment).

    return {
        "latents": latents,
        "actions": actions,
        "deltas": deltas,
        "lang_tokens": lang_tokens,
        "waypoint_4dof": waypoint_4dof,
        "teacher_pose6": teacher_pose6,
        "rollpitch_resid": rollpitch_resid,
        "disagreement": disagreement,
        "vjepa_surprise": vjepa_surprise,
    }


def build_cache(
    episodes: list[dict[str, Any]],
    out_dir: str | Path,
    *,
    renderer: RenderHarness,
    vision_encoder: DinoV3Encoder,
    text_encoder: ClipTextEncoder,
    teacher_client: WorldVLNTeacherClient,
    verifier: VJEPA2SurpriseVerifier,
    config: Config | None = None,
    target_hw: int = TARGET_HW,
    split: str = "",
    limit: int | None = None,
) -> dict[str, Any]:
    """Build the full episode cache: render → encode → teacher → verifier → ``.npz`` + manifest.

    Returns the final manifest dict. Resumable: skips episodes whose ``.npz`` already exists.
    """
    cfg = config if config is not None else Config()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    todo = episodes[:limit] if limit is not None else episodes

    for ep_raw in todo:
        ep_id = str(ep_raw.get("episode_id", ep_raw.get("episodeId", "")))
        traj_id = str(ep_raw.get("trajectory_id", ep_raw.get("trajectoryId", "")))
        scene_id = int(ep_raw.get("scene_id", ep_raw.get("scanName", ep_raw.get("sceneId", 0))))
        latent_path = f"{ep_id}.npz"

        if (out / latent_path).exists():
            with np.load(out / latent_path) as npz_data:
                n_frames = npz_data["latents"].shape[0]
        else:
            arrays = build_episode_cache(
                ep_raw,
                renderer=renderer,
                vision_encoder=vision_encoder,
                text_encoder=text_encoder,
                teacher_client=teacher_client,
                verifier=verifier,
                config=cfg,
                target_hw=target_hw,
            )
            np.savez(out / latent_path, **arrays)
            n_frames = arrays["latents"].shape[0]

        entries.append(
            CacheManifestEntry(
                episode_id=ep_id,
                scene_id=scene_id,
                n_frames=n_frames,
                latent_path=latent_path,
                trajectory_id=traj_id,
            ).to_dict()
        )

    manifest = build_manifest(cfg, split=split, entries=entries)
    manifest["teacher"]["worldvln_model_id"] = "EmbodiedCity/WorldVLN"
    manifest["teacher"]["worldvln_revision"] = "main"
    manifest["teacher"]["render_config_hash"] = _render_transform_hash(target_hw)
    write_manifest(manifest, out)
    return manifest


__all__ = [
    "build_cache",
    "build_episode_cache",
    "center_crop_and_resize",
    "TARGET_HW",
]


if __name__ == "__main__":  # pragma: no cover - USER-GATED
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m vllatent.cache",
        description="vllatent cache builder (A5.14). USER-GATED: needs sim + GPU + WorldVLN server.",
    )
    sub = parser.add_subparsers(dest="cmd")
    build_p = sub.add_parser("build", help="render → encode → teacher → cache .npz + manifest")
    build_p.add_argument("--slice", required=True, help="AerialVLN episode-list JSON")
    build_p.add_argument("--limit", type=int, default=None, help="max episodes to process")
    build_p.add_argument("--out", required=True, help="output cache directory")
    build_p.add_argument("--scenes-root", default="/opt/aerialvln", help="UE4 scene binaries root")
    build_p.add_argument("--host", default="127.0.0.1", help="AirSim host")
    build_p.add_argument("--port", type=int, default=41451, help="AirSim port")
    build_p.add_argument("--teacher-server", default="http://127.0.0.1:8001", help="WorldVLN server")
    build_p.add_argument("--device", default="cuda", help="torch device for encoders/verifier")
    build_p.add_argument("--split", default="train", help="split label for the manifest")
    args = parser.parse_args()

    if args.cmd != "build":
        parser.print_help()
        sys.exit(1)

    from vllatent.encode.dinov3 import DinoV3Encoder
    from vllatent.encode.text import ClipTextEncoder
    from vllatent.render.harness import RenderHarness
    from vllatent.teacher.worldvln import WorldVLNTeacherClient
    from vllatent.verify.vjepa2 import VJEPA2SurpriseVerifier

    episodes = json.loads(Path(args.slice).read_text())
    if isinstance(episodes, dict) and "episodes" in episodes:
        episodes = episodes["episodes"]

    cfg = Config()
    manifest = build_cache(
        episodes,
        args.out,
        renderer=RenderHarness(host=args.host, port=args.port),
        vision_encoder=DinoV3Encoder(device=args.device),
        text_encoder=ClipTextEncoder(device=args.device),
        teacher_client=WorldVLNTeacherClient(args.teacher_server),
        verifier=VJEPA2SurpriseVerifier(device=args.device),
        config=cfg,
        split=args.split,
        limit=args.limit,
    )
    errs = validate_manifest(manifest)
    if errs:
        for e in errs:
            print(f"MANIFEST ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"[cache] wrote {len(manifest['entries'])} episodes to {args.out}")
    print("[cache] manifest OK (teacher provenance populated)")
