# vllatent-ego-drone (`vllatent`) - Codex Project Guidance

Read this before working in this repo. It is the Codex-facing companion to
`CLAUDE.md`; when they differ, prefer `DEV_LOG.md` and
`plans/phase-b-sports-training.md` for current execution state.

## Project Snapshot

`vllatent` is a compact world-action model for a sports-following drone
(skiing primary). Phase B-1 is closed as diagnostic-complete / model-incomplete:
raw future-DINO latent prediction did not beat the persistence baseline on the
real held-out action-video cache. The active Phase B-2 deliverable is a
**scale-free future-action policy**:

```text
RGB 224^2 -> frozen cached DINOv3 ViT-B/16 -> z_t / history latents
                                      + previous observed motion + dt
                            [direct scale-free action policy]
                                      -> future action sequence
```

The future action sequence is the **target**, never an input. Youtube/MegaSaM
translation scale is not trusted; B-2 trains on scale-free path shape and leaves
metric scale to onboard odometry/controller logic at inference. Commanded speed
must be clamped below `7.5 m/s`.

## Required Read Order

1. `DEV_LOG.md` - current step status and latest user-verified facts.
2. `.codex/ralph-rules.md` - Codex Ralph-loop protocol and stop gates.
3. `plans/phase-b-sports-training.md` - authoritative Phase B plan, especially Phase B-2.
4. `CLAUDE.md` - broader project invariants and historical context.

For project-memory context, use the vault in the order named by `CLAUDE.md`.
Repo state is authoritative for code; vault notes are authoritative for why.

## Current B-2 Guardrails

- Frozen DINOv3 ViT-B/16, D=768, cached fp16 latents.
- Predict future scale-free action sequence from observation/history and previous
  observed motion. Do not condition on future actions.
- B-2a starts with a direct action head/policy. Do not add PI-Prober, diffusion,
  language cross-attention, game data, or auxiliary latent/world losses before
  the direct-policy local gate passes or fails with a recorded diagnosis.
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
