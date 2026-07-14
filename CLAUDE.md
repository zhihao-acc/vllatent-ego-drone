# vllatent-ego-drone (`vllatent`) — Agent Context

This file is loaded automatically by Claude Code. Read it before doing any work in this repo.

## Project at a glance

> **⚠ PIVOT 2026-06-19 — SPORTS-FOLLOWING.** The project has pivoted from indoor AerialVLN VLN to
> **autonomous sports-following drone** (skiing primary). AerialVLN is retired as primary data source
> (historical Phase A work). WorldVLN teacher is retired (wrong task domain). Training data = sports FPV
> video only (YouTube + custom GoPro+IMU). See vault
> `[[advisory-sports-following-drift-2026-06-18]]` for the authoritative direction.

A **human-conditioned, action-conditioned latent world model for sports-following drones**. The
active B3 path predicts future DINO patch latents and person state from observed history plus a
candidate future camera/drone plan.

```
observed DINOv3 latents (H=3, 196×768 fp16) + candidate 6-D plan (T=8) + dt
                                      │
                         [depth-6 patch-local predictor]
                                      │
                                      ▼
               future DINO latents + (cx, cy, log_h, visibility)
```

Plans use scale-free translation direction/speed-ratio plus normalized yaw rate and a validity bit.
MegaSaM translation magnitude is diagnostic rather than metric truth. Frozen+cached encoder =>
**no EMA / no VICReg**. Training data = sports FPV video (YouTube + custom GoPro+IMU).

## Active architecture and gate

The current contract is defined by `plans/phase-b3-human-conditioned-world-model.md`, with verified
state in `DEV_LOG.md`. Locked B3 facts: frozen DINOv3 ViT-B/16 (`D=768`), depth 6, `H=3`, `T=8`,
patch-local residual prediction, and six plan fields
`[unit_dir_x, unit_dir_y, unit_dir_z, log_speed_ratio, yaw_rate_norm, valid]`. Future labels never
enter model `forward`; language and metric waypoints are not active B3 inputs/outputs.

B3.6 is blocked: corrected tiny evaluation passes G1b but fails G1a's null-plan margin and G1d's
counterfactual-margin/yaw-geometry requirements. Source-held-out scaling and B3.7/H20 remain
ineligible. B1/B2 action-FiLM, language, waypoint-head, WorldVLN, and TrackVLA material is historical
or deferred context, not the active model contract.

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
| **PURE** | `schemas, actions, frames, config, manifest, audit, ingest/quality, ingest/ego_motion` | numpy/pyyaml/stdlib | CI hard-gates |
| **TORCH** | `encode/, data/, model/, train/`, remaining ingest tools | + torch/timm and ingest extras; optional tools stay lazy where absence is supported | `make test-torch`, dev box; H20 only after B3.6 |
| **SIM** | historical reproduction only | + airsim | `fly0-m1` docker, user-gated |

Guard rule: the pure tier must NEVER gain a torch/airsim/transformers/timm import; CI imports it with
numpy/pyyaml only. Torch-tier modules may import torch normally and are tested in the torch-enabled
environment. Optional model/tool dependencies remain lazy where their absence must not break a
supported entrypoint; AirSim is confined to historical, user-gated reproduction paths.

## Hardware / topology

Dev box RTX 5060 Ti 16 GB · AutoDL **H20 ~96 GB** (training + full encode→cache; **SSH HANDS-OFF**) ·
lab 5×4090 (K=5, Phase C) · `fly0-m1` docker (UE4+AirSim render; scenes `/opt/aerialvln/…`; launched
MANUALLY, wait port 41451) · Jetson Orin NX 16 GB (deploy = the binding size constraint). Conda env
`vllatent-ego-drone` (Py3.10 / torch 2.8 / CUDA 12.x / timm≥1.0.20). China network:
GitHub mirror chain, `HF_ENDPOINT=https://hf-mirror.com`. Full map: `docs/TOPOLOGY.md`.

## Wiki Knowledge Base — FETCH CONTEXT before any work

Vault root for this project: `/home/zh/Documents/Obsidian Vault/projects/vln-ego-drone/latent-pred-pipeline/`.
Read in cheap→expensive order:

0. `[[advisory-sports-following-drift-2026-06-18]]` — **the authoritative direction document** for the
   sports-following pivot. Read this FIRST. Everything else is historical or partially superseded.
1. `[[dev-decision-2026-07-latent-pred-pipeline]]` (file `dev-decision-2026-06-07-latent-pred-pipeline.md`)
   — phases A–E, DoDs, locked implementation decisions. **Has a PIVOT 2026-06-19 banner.**
2. `[[arch-design-2026-06-08-latent-pred]]` — the network spec (partially superseded by the advisory).
   **Has a PIVOT 2026-06-19 banner.**
3. `[[environment-and-equipment]]` — hardware, docker, conda, build/sim, network, the MANUAL-ops list.
4. `[[training-playbook]]` — first-training SOPs (overfit-tiny-batch, scene-split, frame-logging) —
   apply in Phase B.

Vault = authoritative for **INTENT** (why); this repo = authoritative for **CODE state** (what). File
project-level facts (decisions, gotchas, surprising results) in the vault via the `claude-obsidian`
skills, not in source comments.

## The plan & dev log

- **`plans/phase-b3-human-conditioned-world-model.md`** — authoritative active B3 plan and gates.
- `plans/phase-b-sports-training.md` — historical B1/B2 evidence.
- `plans/phase-a5-replan-postpivot.md` — historical Phase-A5 re-plan.
- `DEV_LOG.md` tracks current verified state. Read it first, then the active B3 step.

## Workflow

This project runs as a **Ralph loop** — `.codex/ralph-rules.md` is canonical and
`.claude/ralph-rules.md` is a compatibility summary. Each iteration reads `DEV_LOG.md`, the rules,
and the active B3 plan; makes one bounded evidence-supported change; tests; and records exact results.
Commit only when the user asks, using explicit paths and never `git add -A` or `git add .`.
**User-gated steps** (render / cache / H20 / docker / network) stay `in_progress` until the user pastes
verification — **never auto-mark them done**.

## Load-bearing invariants (do not break)

- **Historical/deferred coordinate-frame seam.** B3 emits no metric waypoint. If the later Phase-D
  controller revives the historical AirSim-NED body waypoint seam, remap NED→FLU→world ENU before
  `WaypointHandoff`/`PoseStamped`; never hand-roll a parallel conversion. Keep `tests/test_frames.py`
  (no-flip: up→up, down→down, right→right-of-forward, forward→forward) green.
- **Cached latents are render-once.** For sports FPV data, frames are extracted from video and
  DINOv3-encoded → fp16 latents on disk. (Historical: AerialVLN rendered from sim at GT poses.)
  Phases B+ train on cached latents. Every cache build writes/updates the **provenance manifest**
  (encoder id+revision, dataset slice, BGR→RGB flag, render config hash).
- **AirSim msgpack-RPC is single-threaded → wrap every `client.X()` in a Lock** (tornado IOLoop is not
  re-entrant).
- **BGR→RGB + orientation-format data foot-guns.** AirSim `Scene` is BGR; DINOv3 expects RGB — convert
  at the render→encode boundary and record the flag. Historical AerialVLN `start_rotation` is
  quaternion `[w,x,y,z]`, while `reference_path` rows are six-wide
  `[x,y,z,pitch,roll,yaw]`; `airsim.Quaternionr` is xyzw.
- **World-frame ENU seam / no-flip is deferred to Phase D.** Historical pure modules retain the
  remap math and tests; the active B3 model does not produce this 4-DoF command.
- **Do NOT modify the sibling repos or their `third_party/`** — workarounds live in THIS repo. fly0 is
  reused (Phase D), never forked.
- **Phases A–C are standalone — no sibling import** (no `_deps.py`; no fly0/navdreamer on `sys.path`).
- **Pure tier stays pure** — `vllatent.{schemas,actions,frames,config,manifest,audit}` import with
  numpy/pyyaml only. CI imports them.
- **No EMA / no VICReg** (frozen+cached encoder = fixed target ⇒ cannot collapse). Reject any
  target-EMA / anti-collapse machinery.
- **No blobs** — never commit weights / `runs/` / cached latents / downloaded JSON / videos / large
  `.npy`. `scripts/check_no_blobs.sh` rejects them; only tiny fixtures under `fixtures/` are allowed.
