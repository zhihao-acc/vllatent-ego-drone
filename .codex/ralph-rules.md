# Ralph Rules - vllatent-ego-drone Phase B2

These are the Codex-local Ralph-loop rules for the active Phase B queue. They
supersede the older B1 latent-world-model loop.

## Completion Promise

Default promise: `B-2 SCALE-FREE B1/WAM CHECKPOINT READY FOR USER-GATED H20 TRAINING`.
Default backstop: `--max-iterations 10`.

Stop when the promise is satisfied, the backstop is reached, a repeated blocker
requires user action, or the next step is user-gated.

## Read Phase

Every iteration starts by reading:

1. `DEV_LOG.md` - current step status and latest user-verified facts.
2. This file - protocol and gates.
3. `plans/phase-b-sports-training.md`, especially the Phase B-2 section.
4. Relevant code/tests only after the active B2 step is identified.

## Current Queue Discipline

- B1.22e is closed as diagnostic-complete / model-incomplete.
- B1.23 and B1.24 are superseded until a useful B2 B1/WAM checkpoint exists.
- Do not launch another B1 latent H20 run.
- Do not activate game pretraining to chase the B1 DINO-latent persistence metric.
- B2.1-B2.5 direct-policy diagnostics are complete; B2.5 failed the local
  source-split gate and must not produce an H20 command.
- The next active queue is:
  - `B2.6` diagnosis/replan docs activation,
  - `B2.7` repair scale-free supervision and align loss with metrics,
  - `B2.8` add past-only action/camera-history conditioning,
  - `B2.9` rerun repaired direct-policy diagnostic,
  - `B2.10` implement control-relevant B1/WAM predictor + action head,
  - `B2.11` local B1-arch training-policy verification,
  - `B2.11a` controlled no-cand06 source-balanced WAM diagnostic,
  - `B2.11b` stale WorldVLN cleanup pass with a reviewed path list,
  - `B2.12` USER gate before any H20 command.

## Iteration Protocol

1. Identify the lowest actionable B2.x step from `DEV_LOG.md`.
2. Review its DoD and test command in the plan.
3. Make one bounded improvement or run one bounded verification.
4. Run the narrowest useful check listed for that step.
5. Record verified facts in `DEV_LOG.md`.
6. Commit with specific paths only, using `feat(phaseB): B2.x - ...` for code
   steps, `test(phaseB): B2.x - ...` for RED tests, or `docs(phaseB): ...` for
   plan/rule updates.
7. Stop at USER gates, after an unfixable test failure, or when the completion
   promise is satisfied.

## B2 Quality Gates

- Future action sequence is the target, never an input.
- Action-like model inputs must be past-observed only. B2.8 may add past
  scale-free action/camera trajectory history; future labels must not affect it.
- Youtube/MegaSaM translation scale is not metric truth. B2a targets must be
  invariant to positive rescaling of all translation deltas.
- B2.5 diagnosed speed-ratio outliers and a loss/metric mismatch. Fix the
  supervision/loss contract before model-capacity or H20 escalation.
- Metric speed is supplied at inference by onboard odometry/controller scale.
- Controller conversion must clamp speed strictly below `7.5 m/s`.
- The H20 target artifact is a stronger B1-architecture WAM checkpoint:
  latent/world predictor plus action head, accepted by action-margin improvement.
- Do not add PI-Prober, NoMaD-style diffusion, language cross-attention, game
  data, or real metric waypoint training until the corrected B1/WAM local gate
  passes or fails with a recorded diagnosis.
- After the B2.11 blocker, run B2.11a before model changes: same no-cand06,
  source-balanced recipe as the repaired B2.9 direct diagnostic, but
  `--model-kind world_action`. Treat any cleanup of stale WorldVLN artifacts as
  a separate B2.11b step with an explicit reviewed path list.
- Scene/source split remains mandatory. Do not split sub-clips from one source
  video across train/val.
- Pure tier stays pure: no torch/transformers/timm/airsim in
  `vllatent.{schemas,actions,frames,config,manifest,audit}` or any new pure
  target-transform module.
- No blobs: never commit `runs/`, weights, `.npz`, videos, raw frames, or QC
  artifact directories.

## B2 Verification Checklist

A healthy B2 local handoff should report:

- scale-free target shape/dtype and finite behavior;
- translation-scale invariance test results;
- explicit no-future-action-input/leakage tests;
- target outlier diagnostics, especially speed-ratio/reference-speed masks;
- loss/metric alignment on normalized path-shape geometry;
- overfit-tiny direct diagnostic and B1/WAM result vs best dumb baseline;
- local source-split aggregate action score, direct-policy comparison, and
  per-source metrics;
- whether the corrected B1/WAM gate beat the best baseline by at least 10%;
- exact next USER gate if H20 is justified.

## User-Gated Paste Blocks

Do not provide a B2b H20 paste block until B2.11 passes and B2.12 is reached.

At B2.12, provide one command only, derived from the verified local recipe, and
ask the user to paste:

- tail of `val_action_metrics.jsonl`;
- tail of train metrics if enabled;
- `source_action_metrics.jsonl`;
- steps/sec and GPU memory;
- confirmation that predictor/action-head `ckpt_best.pt`, config snapshot, and
  metrics files exist.

Artifacts are pulled out-of-band with rsync and never committed.

## Deterministic Stop

Codex does not need Claude's stop hook. If local loop state is introduced, keep
it under `.codex/ralph-loop.local.md`; it is ignored by git.
