# Ralph Rules - vllatent-ego-drone Phase B-3

These are the Codex-local Ralph-loop rules for the active Phase B-3 queue. They
supersede the B2 scale-free action-imitation loop and make B2.12/H20 inactive.

## Completion Promise

Default promise: `B3 HUMAN-CONDITIONED WORLD MODEL READY FOR USER-GATED H20 TRAINING`.
Default backstop: `--max-iterations 10`.

Stop when the promise is satisfied, the backstop is reached, a repeated blocker
requires user action, or the next step is user-gated.

## Read Phase

Every iteration starts by reading:

1. `DEV_LOG.md` - current step status and latest user-verified facts.
2. This file - protocol and gates.
3. `plans/phase-b3-human-conditioned-world-model.md` - authoritative active B3 plan.
4. `plans/phase-b-sports-training.md` only for historical B1/B2 evidence when needed.
5. Relevant code/tests only after the active B3 step is identified.

## Current Queue Discipline

- B1 latent-only world modeling is closed as diagnostic-complete / model-incomplete.
- B2 action imitation is closed as evidence. B2.11c is useful as a partial
  translation/speed prior, but it is not the B3 endpoint.
- Do not continue to B2.12 or provide a B2b H20 command.
- The active queue is:
  - `B3.0` write/approve Phase B-3 plan and align guidance (done 2026-07-07),
  - `B3.1` reviewed cleanup of obsolete B1/B2 runnable paths (done 2026-07-07),
  - `B3.2` person-track cache backfill and data screens (done 2026-07-07),
  - `B3.3` 6-D plan-token contract and T configurability,
  - `B3.4` Stage-0 probes plus K1/K2,
  - `B3.5` depth-6 per-step 6-D conditioned world model,
  - `B3.6` Stage-1 local gates G1a-G1d,
  - `B3.7` USER-GATED H20 depth-6 run,
  - `B3.8` planner-facing CEM/MPPI hindsight-replay evaluation.

## Iteration Protocol

1. Identify the lowest actionable B3.x step from `DEV_LOG.md`.
2. Review its DoD and test command in the B3 plan.
3. Make one bounded improvement or run one bounded verification.
4. Run the narrowest useful check listed for that step.
5. Record verified facts in `DEV_LOG.md`.
6. Commit only when the user asks, using specific paths only. Never `git add -A`
   or `git add .`.
7. Stop at USER gates, after an unfixable test failure, or when the completion
   promise is satisfied.

## B3 Quality Gates

- `PLAN_TOKEN_DIM = 6`; fields are
  `[unit_dir_x, unit_dir_y, unit_dir_z, log_speed_ratio, yaw_rate_norm, valid]`.
- Candidate future camera/drone plan is an input. Future person/world labels are
  never inputs.
- Target labels include future latents, person state `(cx, cy, log_h, visibility)`,
  masks, confidences, and optional inverse-dynamics labels.
- Person-track cache keys are optional and backward-compatible:
  `person_bbox (N,4)`, `person_visible (N,)`, `person_conf (N,)`, plus detector
  provenance in manifests/backfill logs.
- Translation conditioning remains scale-free and invariant to positive rescaling.
- Yaw-rate normalization is finite and clipped. Metric speed is controller-side,
  clamped strictly below `7.5 m/s`.
- Source split remains mandatory by source video, not subclip.
- The B3 predictor uses depth 6, D=768, H=3, T=8 first. Do not call it `~28M`;
  log exact counts after B3.5.
- No diffusion, language, game data, SAM2, PI-Prober, metric waypoint training,
  or EGO-Planner integration before deterministic B3 gates pass.
- H20/SSH/docker/long jobs remain user-gated. Codex prepares one command only at
  B3.7 if B3.6 passes.

## B3 Verification Checklist

A healthy B3 local handoff should report:

- plan-token shape/dtype/field semantics and scale-invariance results;
- yaw-rate finite/clipped behavior and `valid` mask composition;
- old-cache fallback plus new person-label cache loading;
- G0 real-latent probe center error and presence AUROC;
- K1 camera/person causality readout;
- K2 tiny conditioned predictor versus person-state persistence;
- depth-6 model parameter count and forward shapes at T=8;
- plan-causality and plan-sensitivity tests;
- G1a/G1b/G1c/G1d/K3/K4/K5-lite/K6 gate readouts before any H20 command;
- exact next USER gate if H20 is justified.

## Stop/Pass Gates

| Gate | Pass | On Fail |
|---|---|---|
| G0 | real-latent probes work on held-out sources | fix labels/probes |
| K1 | plan-only camera-compensated person motion is near chance | rework causal separation or abort |
| K2 | tiny conditioned predictor beats persistence by `>=10%` | abort dense WM path |
| G1a | conditioned predictor beats person-weighted latent persistence `>=10%` and null-plan `>=5%` | objective/conditioning bug hunt |
| G1b | rollout beats persistence at every k<=8 | shorten/reweight before scaling |
| G1c/K4 | probe transfer passes and gameability check passes | calibrate probes or state-head primary |
| G1d/K3/K5-lite | true 6-D plan beats shuffled/flipped plans on `>=70%` windows | strengthen conditioning before capacity/data |
| K6 | source-count ablation improves with more sources | expand data before more H20/model scaling |

## User-Gated Paste Blocks

Do not provide an H20 paste block until B3.6 passes and B3.7 is reached.

At B3.7, provide exactly one command and ask the user to paste:

- tail of metrics JSONL;
- source metrics;
- G1 gate readouts;
- steps/sec and GPU memory;
- confirmation that checkpoint, config snapshot, and metrics files exist.

Artifacts are pulled out-of-band and never committed.

## Deterministic Stop

Codex does not need Claude's stop hook. If local loop state is introduced, keep
it under `.codex/ralph-loop.local.md`; it is ignored by git.
