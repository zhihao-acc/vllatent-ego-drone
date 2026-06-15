# Cold-start handoff — resume the ralph loop at A5.16 (loader inspect over real cache)

> **Created 2026-06-15.** Paste the one-liner at the very bottom into a NEW session, or just say
> "continue the ralph loop." This is the operator brief; the authoritative state lives in the repo files
> it points to. **A5.14 landed and is live-verified since the last handoff** — the full cache pipeline
> (render → DINOv3 + CLIP-text + WorldVLN teacher K=5 + V-JEPA-2 → .npz + manifest) ran end-to-end on
> 5 episodes in fly0-m1 docker. Lowest pending = **A5.16**.

## 0. What this is

Continue the `vllatent-ego-drone` **Phase-A.5 ralph loop**. Lowest pending = **A5.16 — loader over
real teacher/oracle dump** (USER-GATED inspect over the 5-episode real cache from A5.14). Then
A5.17 (size the full job) → A5.18 (Phase-A DoD sign-off). **All three remaining steps are
USER-GATED** — they need the fly0-m1 docker / H20 / real .npz artifacts.

## 1. Read FIRST (authoritative, cheap → expensive)

1. `DEV_LOG.md` — newest on top; the step-status table is the source of truth. Expect: 1–6+5b done,
   A5.1–A5.15 done, **A5.14 done**, **A5.16 / A5.17 / A5.18 pending**.
2. `.claude/ralph-rules.md` — per-iteration protocol, quality gates, the USER-GATED rule, Test Command Index.
3. `plans/phase-a5-replan-postpivot.md` — each step's tier / gate / DoD / exact test command / deps.
4. `CLAUDE.md` — repo invariants, PURE/TORCH/SIM tier split, foot-guns.

## 2. Current position (verified; pushed @ `origin/main`)

- **DONE & real-weight-verified:** A5.1–A5.9 (pure seams), **A5.10** DINOv3 (`(196,768)` fp16 smoke),
  **A5.11** WorldVLN teacher (live K-rollout on H20), **A5.12** V-JEPA-2 verifier (`surprise [0.174,0.208]`
  cuda smoke), **A5.13** AirSim render (live: 8 frames from `tiny-0001`; user fixes `7e31bf3`),
  **A5.13b** CLIP text tower (`lang_tokens (10,768)` cuda smoke), **A5.14** cache orchestration
  (5-episode small-slice build GREEN, manifest OK, K=5 rollouts verified), **A5.15** distillation loader
  (defines the .npz read-contract A5.14 writes).
- **`[torch]` extra PINNED** in `pyproject.toml`: `torch>=2.8,<2.13`, `transformers>=4.56,<6`,
  `timm>=1.0.20,<2`.
- **Gates:** 251 pure / 5 torch / ruff / mypy(pure) / import-smoke / blob — all green.
- **Latest commit:** `805c650` (A5.14 orchestration + mocked test).

## 3. What A5.14 produced (the real .npz cache)

The small-slice build wrote 5 episodes to `data/latent_cache/` inside fly0-m1 docker at
`/workspace/vllatent-ego-drone/data/latent_cache/`. Each `.npz` contains the full read-contract:
- `latents` (N,196,768) fp16 — DINOv3 from center-crop+resize 224²
- `actions` (N,) int — discrete action indices
- `deltas` (N,4) f32 — body-frame [dx,dy,dz,dyaw]
- `lang_tokens` (M,768) fp16 — frozen CLIP text embeddings
- `waypoint_4dof` (N,4) f32 — teacher 4-DoF (from 6→4 projection)
- `teacher_pose6` (N,6) f32 — raw WorldVLN 6-DoF [roll,yaw,pitch,x,y,z]
- `rollpitch_resid` (N,) f32 — dropped roll/pitch residual
- `disagreement` (N,) f32 — scalarized K-rollout spread (4 student channels)
- `vjepa_surprise` (N,) f32 — V-JEPA-2 cosine surprise per transition

Plus `manifest.json` with teacher provenance (worldvln_model_id, worldvln_revision,
disagreement_source, vjepa2_model_id, render_config_hash).

## 4. Remaining steps (all USER-GATED)

| Step | What | Gate |
|------|------|------|
| **A5.16** | Loader inspect over real cache: `python -m vllatent.data inspect --cache data/latent_cache/ --n 4` | USER-GATED (needs real .npz from A5.14, inside docker) |
| **A5.17** | Size the full render→teacher→cache job (timing AUTO estimate; bulk build USER-GATED) | USER-GATED |
| **A5.18** | Phase-A DoD verification — final sign-off | USER-GATED |

## 5. Foot-guns & constraints (carry forward)

- **SSH HANDS-OFF** — give commands, don't drive remote auth from the harness.
- **USER-GATED rule** — sim/docker/GPU/H20 steps stay `in_progress` until user pastes verification.
- **`data/latent_cache/` is gitignored** (`.npz` in `.gitignore`; no blobs committed).
- **Center-crop+resize (480×640→224²)** is the normalization at the render→encode boundary (NOT sim
  CaptureSettings). Recorded in manifest `render_config_hash`.
- **6→4 projection**: WorldVLN emits `[roll,yaw,pitch,x,y,z]` (m,rad) → student gets
  `[dx,dy,dz,dyaw]` (m,deg); drop roll/pitch, convert yaw rad→deg.
- **China network**: `HF_ENDPOINT=https://hf-mirror.com`; `HF_HUB_OFFLINE=1` inside docker.
- **Pure tier must stay pure** — 251 tests with numpy/pyyaml only; no torch at import time.

## 6. Resume one-liner

```
Continue the vllatent-ego-drone Phase-A.5 ralph loop. Read DEV_LOG.md + .claude/ralph-rules.md + plans/phase-a5-replan-postpivot.md, then resume at the lowest pending step (A5.16 — loader inspect over real cache). All remaining steps (A5.16, A5.17, A5.18) are USER-GATED. Emit command blocks, do not drive. Keep the pure gate green (251 tests; pure env is Py3.9). STOP CHECK at started_step+3 or any user-gated step. Run iterations INLINE (no ralph-loop.local.md).
```
