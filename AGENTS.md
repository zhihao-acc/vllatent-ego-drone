# vllatent-ego-drone (`vllatent`) - Codex Project Guidance

Read this before working in this repo. It is the Codex-facing companion to
`CLAUDE.md`; when they differ, prefer `DEV_LOG.md` and
`plans/phase-b-sports-training.md` for current execution state.

## Project Snapshot

`vllatent` is a compact latent world-action model for a sports-following drone
(skiing primary). The current Phase B-1 deliverable is a **latent predictor**:

```text
RGB 224^2 -> frozen cached DINOv3 ViT-B/16 -> z_t (196x768 fp16)
         + action(4-DoF, FiLM) + dt(FiLM)
         -> LatentPredictor, block-causal ViT, depth=6
         -> predicted future DINOv3 latents
```

B-1 trains the latent predictor only. Waypoint-head training, `L_wp`, MegaSaM
scale fixes, and the MLP vs PI-Prober vs attentive-pool decision are Phase B-2a.
Do not add Stage 2/3 head training to B-1.

## Required Read Order

1. `DEV_LOG.md` - current step status and latest user-verified facts.
2. `.codex/ralph-rules.md` - Codex Ralph-loop protocol and stop gates.
3. `plans/phase-b-sports-training.md` - authoritative Phase B plan, especially Group 8.
4. `CLAUDE.md` - broader project invariants and historical context.

For project-memory context, use the vault in the order named by `CLAUDE.md`.
Repo state is authoritative for code; vault notes are authoritative for why.

## Current B-1 Guardrails

- Frozen DINOv3 ViT-B/16, D=768, cached fp16 latents.
- Latent predictor depth=6 unless the plan or user explicitly changes it.
- No EMA, no VICReg, no anti-collapse machinery.
- Scene split by **source video**, not sub-clip (`stem.split("_")[0]`).
- Train-only `NormStats`; validation must not recompute stats on val clips.
- B-1 success means per-horizon validation cosine beats the persistence
  baseline `cos(z_t, z_{t+k})` by a clear margin.
- The waypoint head is deferred. Do not train it or add head-specific command
  flags while working B-1.

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
