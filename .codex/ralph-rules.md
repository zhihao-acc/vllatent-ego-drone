# Ralph Rules - vllatent-ego-drone Phase B2

These are the Codex-local Ralph-loop rules for the active Phase B queue. They
supersede the older B1 latent-world-model loop.

## Completion Promise

Default promise: `B-2 SCALE-FREE FUTURE-ACTION POLICY READY FOR USER-GATED H20 TRAINING`.
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
- B1.23 and B1.24 are superseded until a useful B2 action-policy checkpoint exists.
- Do not launch another B1 latent H20 run.
- Do not activate game pretraining to chase the B1 DINO-latent persistence metric.
- The next active queue is:
  - `B2.0` docs/rules activation, then
  - `B2.1` pure scale-free action target contract,
  - `B2.2` additive loader/collate target fields,
  - `B2.3` direct scale-free action policy,
  - `B2.4` action losses/metrics/baselines,
  - `B2.5` local B2a training-policy verification,
  - `B2.6` USER gate before any H20 command.

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
- The only action-like model input in B2a is previous observed motion.
- Youtube/MegaSaM translation scale is not metric truth. B2a targets must be
  invariant to positive rescaling of all translation deltas.
- Metric speed is supplied at inference by onboard odometry/controller scale.
- Controller conversion must clamp speed strictly below `7.5 m/s`.
- Keep B2a direct and small: no PI-Prober, no NoMaD-style diffusion, no language
  cross-attention, no game data, and no auxiliary latent/world loss until the
  direct scale-free policy gate passes or fails with a recorded diagnosis.
- Scene/source split remains mandatory. Do not split sub-clips from one source
  video across train/val.
- Pure tier stays pure: no torch/transformers/timm/airsim in
  `vllatent.{schemas,actions,frames,config,manifest,audit}` or any new pure
  target-transform module.
- No blobs: never commit `runs/`, weights, `.npz`, videos, raw frames, or QC
  artifact directories.

## B2a Verification Checklist

A healthy B2a local handoff should report:

- scale-free target shape/dtype and finite behavior;
- translation-scale invariance test results;
- explicit no-future-action-input/leakage tests;
- overfit-tiny action-policy result vs best dumb baseline;
- local source-split aggregate action score and per-source metrics;
- whether B2a beat the best baseline by at least 10%;
- exact next USER gate if H20 is justified.

## User-Gated Paste Blocks

Do not provide a B2b H20 paste block until B2.5 passes and B2.6 is reached.

At B2.6, provide one command only, derived from the verified local recipe, and
ask the user to paste:

- tail of `val_action_metrics.jsonl`;
- tail of train metrics if enabled;
- `source_action_metrics.jsonl`;
- steps/sec and GPU memory;
- confirmation that `ckpt_best.pt`, config snapshot, and metrics files exist.

Artifacts are pulled out-of-band with rsync and never committed.

## Deterministic Stop

Codex does not need Claude's stop hook. If local loop state is introduced, keep
it under `.codex/ralph-loop.local.md`; it is ignored by git.
