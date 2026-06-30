# Ralph Rules - vllatent-ego-drone Phase B-1

These rules are the Codex-local Ralph-loop protocol. They supersede the stale
Phase-A wording in `.claude/ralph-rules.md` for Codex work.

## Completion Promise

Default promise: `B-1 LATENT WORLD MODEL READY FOR USER-GATED TRAINING`.
Default backstop: `--max-iterations 10`.

Stop when the promise is satisfied, the backstop is reached, a repeated blocker
requires user action, or the next step is user-gated.

## Read Phase

Every iteration starts by reading:

1. `DEV_LOG.md` - find the lowest pending/in-progress B1.x step and latest user
   facts.
2. This file - protocol and gates.
3. `plans/phase-b-sports-training.md`, Group 8 - DoD and runbook.
4. Relevant code/tests only after the step is identified.

The filename `plans/handoff-2026-06-24-b1.10c-onwards.md` is historical. Treat
the latest `DEV_LOG.md` entries and the Group 8 plan as authoritative.

## Current Queue Discipline

- B1.21b and B1.22a are done.
- B1.22b/B1.22c data generation and curation are user-gated. The user may paste
  updated verification showing more encoded clips than `DEV_LOG.md` currently
  records. When that happens, update `DEV_LOG.md` with the pasted numbers before
  marking any user-gated step done.
- Before B1.22e training, perform a local QC pass over `ingest_data/latent_cache`
  if the files are present. QC is an AUTO preflight, but it does not by itself
  close B1.22b/B1.22c; only user-pasted verification closes user-gated steps.
- B1.22e is user-gated H20 training. Stop and hand the user a paste-ready command
  block; do not SSH, rent, operate H20, or run docker.
- B1.23 and B1.24 are also user-gated.

## Iteration Protocol

1. Identify the lowest actionable step from `DEV_LOG.md`.
2. Review its DoD and test command in the plan.
3. Make one bounded improvement or run one bounded verification.
4. Run the narrowest useful check.
5. Record facts in `DEV_LOG.md` only when verified. User-gated steps remain
   `in_progress` until the user provides pasted output.
6. Commit with specific paths only, using `feat(phaseB): B1.x - ...` for real
   step work or `chore(phaseB): ...` for loop/rule maintenance.
7. Stop-check: stop at user gates, at `started_step + 3`, after an unfixable
   test failure, or when the completion promise is satisfied.

## Quality Gates

- Pure tier stays pure: no torch/transformers/timm/airsim in
  `vllatent.{schemas,actions,frames,config,manifest,audit}`.
- Torch imports stay lazy in torch-tier modules.
- No EMA, VICReg, target-EMA, anti-collapse losses, PI-Prober implementation, or
  visual bottleneck in B-1.
- Do not train the waypoint head in B-1. No `--stage 2/3`, predictor-freeze
  head path, `L_wp` training run, or head-input experiments.
- Scene split by source video, not sub-clip.
- Validation uses train-only `NormStats`.
- Latent metrics must include per-horizon cosine, persistence cosine, and margin.
- No blobs: never commit `runs/`, weights, `.npz`, videos, raw frames, or QC
  artifact directories.

## Pre-B1.22e QC Checklist

Use the repo's QC tooling when available (`scripts/qc_report.py` /
`scripts/qc_lib.py`) and inspect the generated cache before training. A healthy
QC handoff should report:

- number of `.npz` clips and total frames/windows;
- latent shape `(N, 196, 768)` and dtype `float16`;
- finite latents and no all-zero/constant clips;
- required action/quality arrays present and aligned to frame count;
- manifest validates and records BGR-to-RGB plus center-square-crop provenance;
- domain field is present (`real` unless explicitly game);
- source-video grouping has enough train/val sources for a non-leaking split;
- per-clip errors or rejected clips are listed explicitly.

If QC finds corrupt caches, stale stretch-encoded outputs, missing manifests, or
source leakage risk, stop before B1.22e and report the exact files.

## User-Gated Paste Blocks

For B1.22e, provide a block derived from the plan runbook, currently:

```bash
$PY scripts/train_sports.py --cache-dir ingest_data/latent_cache --run-dir runs/b1_latent \
  --latent-only --amp-dtype bf16 --depth 6 --batch-size 64 --lr 2e-4 --warmup-frac 0.05 \
  --weight-decay 0.05 --val-frac 0.2 --eval-every-epochs 1 --early-stop-patience 8 \
  --early-stop-metric val_cos --device cuda
```

Ask the user to paste:

- tail of `runs/b1_latent/val_metrics.jsonl`;
- per-horizon cosine, persistence cosine, and margin;
- steps/sec and GPU memory;
- confirmation that `ckpt_best.pt` and `norm_stats.npz` exist.

Artifacts are pulled out-of-band with rsync and never committed.

## Deterministic Stop

Codex does not need Claude's stop hook. If local loop state is ever introduced,
keep it under `.codex/ralph-loop.local.md`; it is ignored by git.
