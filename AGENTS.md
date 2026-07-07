# vllatent-ego-drone (`vllatent`) - Codex Project Guidance

Read this before working in this repo. It is the Codex-facing companion to
`CLAUDE.md`; when they differ, prefer `DEV_LOG.md` and
`plans/phase-b3-human-conditioned-world-model.md` for current execution state.

## Project Snapshot

`vllatent` is a compact latent world model for a sports-following drone
(skiing primary). Phase B-1 is closed as diagnostic-complete / model-incomplete:
raw future-DINO latent prediction did not beat the persistence baseline on the
real held-out action-video cache. Phase B-2 proved useful scale-free camera-motion
signal but drifted toward imitating the YouTube camera operator. The active
Phase B-3 deliverable is now a **human-conditioned, action-conditioned latent
world model**:

```text
observed human/camera history + candidate future 6-D camera/drone plan
    -> future person/world latents + person-state trajectory
```

B2.11c remains evidence and a partial translation/speed proposal prior. B2.12/H20
is inactive. Youtube/MegaSaM translation scale is not trusted; B3 uses scale-free
translation plan tokens and leaves metric speed to onboard odometry/controller
logic at inference. Commanded speed must be clamped strictly below `7.5 m/s`.

## Required Read Order

1. `DEV_LOG.md` - current step status and latest user-verified facts.
2. `.codex/ralph-rules.md` - Codex Ralph-loop protocol and stop gates.
3. `plans/phase-b3-human-conditioned-world-model.md` - authoritative active Phase B-3 plan.
4. `plans/phase-b-sports-training.md` - historical Phase B/B2 evidence, especially B2.9-B2.11c.
5. `CLAUDE.md` - broader project invariants and historical context.

For project-memory context, use the vault in the order named by `CLAUDE.md`.
Repo state is authoritative for code; vault notes are authoritative for why.

## Current B-3 Guardrails

- Frozen DINOv3 ViT-B/16, D=768, cached fp16 latents.
- Depth-6 predictor, H=3, T=8 first. Do not call this model `~28M`; log exact
  parameter counts after B3.5, with expected order about 57M predictor params.
- Candidate future 6-D plan is an input:
  `[unit_dir_x, unit_dir_y, unit_dir_z, log_speed_ratio, yaw_rate_norm, valid]`.
- Future person/world targets are labels only: target latents, person state
  `(cx, cy, log_h, visibility)`, masks, and confidences must not enter model
  `forward`.
- Add optional cache keys `person_bbox (N,4)`, `person_visible (N,)`, and
  `person_conf (N,)` with detector/tracker provenance. Old caches must still
  load with invisible-person defaults.
- B3.1 cleanup is done. The next AUTO step is B3.2: person-track cache backfill
  implementation and data screens. Full cache backfill remains user-gated.
- Do not continue to B2.12/H20. H20 becomes eligible only at B3.7 after B3.6
  local gates justify one serious depth-6 run.
- Do not add diffusion, language, game data, SAM2, PI-Prober, metric waypoint
  training, or EGO-Planner integration before deterministic B3 gates pass.
- Scene split by **source video**, not sub-clip (`stem.split("_")[0]`).
- Youtube/MegaSaM translation magnitude is diagnostics only, not metric truth.
- Real metric scale is supplied by onboard odometry/controller at inference.
- Controller conversion must clamp commanded speed below `7.5 m/s`.

## Tier Rules

| Tier | Modules | Import rule |
|---|---|---|
| PURE | `schemas`, `actions`, `frames`, `config`, `manifest`, `audit` | stdlib + numpy/pyyaml only |
| TORCH | `encode/`, `data/`, `model/`, `train/` | torch/transformers/timm imports lazy or inside functions |
| SIM | render/cache paths | airsim imports lazy; AirSim RPC calls are locked |

Never add torch, transformers, timm, or airsim imports to the pure tier.

## Data And Artifact Rules

- Do not commit weights, `.npz`, videos, frames, `runs/`, caches, or generated
  QC report artifacts.
- Use `git add <specific files>` only; never `git add -A` or `git add .`.
- Cached-latent provenance must preserve encoder id/revision, BGR-to-RGB flag,
  dtype, domain, and manifest validity.
- Center-square-crop preprocessing is load-bearing; do not reintroduce stretch
  encoding.
- H20, Orin NX, SSH, docker, video download, and multi-GB model/data operations
  are user-gated. Provide paste-ready commands; do not operate those systems.

## Verification Preferences

- After pure changes: `python -m pytest -q --ignore=tests/test_data_shapes.py -x`
  or the narrow affected pytest target.
- After torch/training changes: use the narrow torch tests first, then
  `make test-torch` when feasible.
- Run `make lint` and `make typecheck` before committing broad code changes.
- Before B1.22e training, run local QC over the generated latent cache and
  inspect counts, dtype, shapes, manifest fields, BGR-to-RGB/crop provenance,
  source split health, and per-clip failures.

## Ralph Loop

This repo uses a Ralph-loop style workflow. Follow `.codex/ralph-rules.md`.
User-gated steps stay `in_progress` until the user pastes verification numbers;
do not infer completion from local files alone.
