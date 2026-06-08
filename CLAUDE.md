# vllatent-ego-drone (`vllatent`) — Agent Context

This file is loaded automatically by Claude Code. Read it before doing any work in this repo.

## Project at a glance

A **compact latent world-action model for aerial VLN**; the contribution is **trust-aware
commitment**. One frozen perception backbone, a small latent predictor, and a trust layer that
decides how far ahead to commit.

```
RGB 224² ─► [DINOv3 ViT-B/16, FROZEN, CACHED] ─► z_t (196×768 fp16)
                                                   │  + discrete action (FiLM) + language (cross-attn)
                                          [latent predictor ~120M, block-causal, D=768 d12]
                                                   │  rollout (H=3 history, T=4 horizon)
                          ┌────────────────────────┼─────────────────────────┐
                          ▼                         ▼                         ▼
              4-DoF waypoint head        single-pass horizon head     (Phase C) K=5 ensemble
              [768→512→256→4]            (trust: how far to commit)    + V-JEPA-2 surprise gate
```

Action is **discrete-in** (AerialVLN 8-way) / **continuous-4DoF-out** (Δx,Δy,Δz,Δψ). Frozen+cached
encoder ⇒ **no EMA / no VICReg**. **Phase A is plumbing + data, not research.**

## Architecture — LOCKED vs OPEN

**LOCKED — do NOT relitigate. See vault `[[arch-design-2026-06-08-latent-pred]]` (authoritative).**
Encoder DINOv3 ViT-B/16 frozen+cached (RGB-only, 224², 196×768 fp16) · predictor block-causal ViT
D=768 depth 12 heads 12 MLP 3072 · discrete-codebook→per-step FiLM action · frozen-text (SigLIP/CLIP
text tower 512→768)→cross-attention language · H=3 / T=4 · trust = deployed single-pass horizon head;
offline K=5 ensemble teacher + V-JEPA-2 surprise = **Phase C** · continuous 4-DoF waypoint head ·
**no EMA / no VICReg** (frozen cached encoder = fixed target).

**OPEN — keep the lean, do not resolve in Phase A:** predictor depth/FiLM-vs-interleave (Phase B);
ensemble K + ε/δ + horizon sweep + calibration (Phase C); the EGO-Planner `WaypointHandoff` yaw-field
extension + closed-loop seam (Phase D).

## The reused repo (reuse, do NOT fork) — Phase D only

fly0 at `/home/zh/CODE/vln-ego-drone/fly0-style-pipeline`: `schemas/io_contract.py` `WaypointHandoff`,
`geometry/frames.py` (NED↔ENU / FRD↔FLU), `geometry/pose_utils.py`, `sim/airsim_client.py`.

**Phases A–C do NOT import fly0** — they are standalone (dataset + network + GPU; no sim loop). The
AirSim-NED-body → fly0-FLU → world-ENU remap is **re-derived and unit-tested against fly0's
`frames.py` semantics** in `vllatent/frames.py`, but fly0 is never imported in A–C. The dataset's
native loader (`…/fly0-style-pipeline/third_party/AirVLN`) is **reuse/replay only — never modify it**.

## Tier split (load-bearing)

| Tier | Modules | Imports | Runs |
|---|---|---|---|
| **PURE** | `schemas, actions, frames, config, manifest, audit` | numpy/pyyaml/stdlib | CI hard-gates |
| **TORCH** | `encode/, data/` | + torch/transformers/timm (LAZY) | `make test-torch`, dev box / H20 |
| **SIM** | `render/, cache` | + airsim (LAZY) | `fly0-m1` docker only |

Guard rule: every `import torch` / `import airsim` lives **inside a function/method** (or behind a lazy
guard) so a torch-free box imports any module without crashing. The pure tier must NEVER gain a
torch/airsim/transformers/timm import — CI imports it with numpy/pyyaml only.

## Hardware / topology

Dev box RTX 5060 Ti 16 GB · AutoDL **H20 ~96 GB** (training + full encode→cache; **SSH HANDS-OFF**) ·
lab 5×4090 (K=5, Phase C) · `fly0-m1` docker (UE4+AirSim render; scenes `/opt/aerialvln/…`; launched
MANUALLY, wait port 41451) · Jetson Orin NX 16 GB (deploy = the binding size constraint). Conda env
`vllatent-ego-drone` (Py3.10 / torch 2.8 / CUDA 12.x / transformers≥4.56 / timm≥1.0.20). China network:
GitHub mirror chain, `HF_ENDPOINT=https://hf-mirror.com`. Full map: `docs/TOPOLOGY.md`.

## Wiki Knowledge Base — FETCH CONTEXT before any work

Vault root for this project: `/home/zh/Documents/Obsidian Vault/projects/vln-ego-drone/latent-pred-pipeline/`.
Read in cheap→expensive order:

1. `[[dev-decision-2026-07-latent-pred-pipeline]]` (file `dev-decision-2026-06-07-latent-pred-pipeline.md`)
   — phases A–E, DoDs, locked implementation decisions, §8 repo + EGO z/yaw-unfixed. **Read first.**
2. `[[arch-design-2026-06-08-latent-pred]]` — the LOCKED network spec + I/O contract + data-audit spec
   (§6) + render finding. **Authoritative; do NOT relitigate the locked architecture.**
3. `[[environment-and-equipment]]` — hardware, docker, conda, build/sim, network, the MANUAL-ops list.
4. `[[training-playbook]]` — first-training SOPs (overfit-tiny-batch, scene-split, frame-logging) —
   apply in Phase B.

Vault = authoritative for **INTENT** (why); this repo = authoritative for **CODE state** (what). File
project-level facts (decisions, gotchas, surprising results) in the vault via the `claude-obsidian`
skills, not in source comments.

## The plan & dev log

- `plans/phase-a-data-and-io-contract.md` (repo-root `plans/`, **not** `.claude/plans/`) — the Phase-A
  steps with DoDs + exact test commands + ralph status.
- `DEV_LOG.md` at repo root tracks which step is in progress. **Read it first** to find the position,
  then re-read the relevant plan step.

## Workflow

This project runs as a **ralph loop** — see `.claude/ralph-rules.md` for the iteration protocol and
quality gates. Each iteration: READ `DEV_LOG.md` → ralph-rules → plan; EXECUTE the lowest pending step
(**pure-tier / fixtures-first**); TEST with the step command; RECORD in `DEV_LOG.md`; COMMIT
`feat(phaseA): step N — …` (specific `git add`, never `-A`); STOP CHECK at `started_step + 3`.
**User-gated steps** (render / cache / H20 / docker / network) stay `in_progress` until the user pastes
verification — **never auto-mark them done**.

## Load-bearing invariants (do not break)

- **#1 foot-gun — coordinate frames.** Waypoint output is **AirSim-NED body, yaw-only**; remapped
  NED→FLU→**world ENU** before any `WaypointHandoff`/`PoseStamped` (Phase D). **Never hand-roll a
  parallel frame conversion** — re-derive against fly0's `geometry/frames.py` semantics and keep
  `tests/test_frames.py` (no-flip: up→up, down→down, right→right-of-forward, forward→forward) green.
- **Cached latents are render-once.** RGB is NOT in the AerialVLN JSON — it renders from the sim at
  each GT pose, then DINOv3-encodes → fp16 latents on disk. Phases B+ train on cached latents, no sim.
  Every cache build writes/updates the **provenance manifest** (encoder id+revision, dataset slice,
  quaternion order, BGR→RGB flag, render config hash).
- **AirSim msgpack-RPC is single-threaded → wrap every `client.X()` in a Lock** (tornado IOLoop is not
  re-entrant).
- **BGR→RGB + quaternion-order data foot-guns.** AirSim `Scene` is BGR; DINOv3 expects RGB — convert at
  the render→encode boundary, record the flag. `start_rotation` is `[w,x,y,z]`; `reference_path` poses
  are `[…,qx,qy,qz,qw]`; `airsim.Quaternionr` is xyzw — reorder to canonical xyzw and name the order at
  every pose seam.
- **World-frame ENU seam / no-flip is Phase D.** A–C produce continuous 4-DoF + the remap math + unit
  tests; the live closed-loop seam (and the yaw-field extension) is Phase D.
- **Do NOT modify the sibling repos or their `third_party/`** — workarounds live in THIS repo. fly0 is
  reused (Phase D), never forked.
- **Phases A–C are standalone — no sibling import** (no `_deps.py`; no fly0/navdreamer on `sys.path`).
- **Pure tier stays pure** — `vllatent.{schemas,actions,frames,config,manifest,audit}` import with
  numpy/pyyaml only. CI imports them.
- **No EMA / no VICReg** (frozen+cached encoder = fixed target ⇒ cannot collapse). Reject any
  target-EMA / anti-collapse machinery.
- **No blobs** — never commit weights / `runs/` / cached latents / downloaded JSON / videos / large
  `.npy`. `scripts/check_no_blobs.sh` rejects them; only tiny fixtures under `fixtures/` are allowed.
