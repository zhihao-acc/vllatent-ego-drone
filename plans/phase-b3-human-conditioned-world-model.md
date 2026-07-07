# Phase B-3: Human-Conditioned Latent World Model Plan

> Created 2026-07-07. This is the active Phase B plan. It supersedes the B2.12/H20
> action-imitation handoff and the active execution portions of
> `plans/phase-b-sports-training.md`. B1/B2 plan history, reports, and `DEV_LOG.md`
> remain historical evidence unless the user explicitly requests removal.

## Objective

Train an action-conditioned, human-centric latent world model for FPV sports following:

```text
observed human/camera history + candidate future 6-D camera/drone plan
    -> future person/world latents + person-state trajectory
```

B2.11c proved that the current latents and scale-free motion labels contain useful
camera-motion signal, but it optimizes imitation of the YouTube camera operator. B3
uses B2.11c as evidence and as a partial translation/speed proposal prior only. It is
not the endpoint, and B2.12/H20 is inactive.

## Locked Defaults And Invariants

- Frozen cached DINOv3 ViT-B/16 latents remain the default, with `D=768`.
- Predictor default is depth 6, `H=3`, and `T=8` first. `T=12` is gated later.
- Do not describe the depth-6 B3 model as `~28M`. Count exact parameters after
  B3.5; expected order is about 57M predictor parameters.
- Future candidate camera/drone plan is an input. Future person/world labels are
  never inputs.
- Training may teacher-force the logged future plan as `planned_actions`; planner
  evaluation passes sampled candidate plans through the same interface.
- Source split remains by source video, not subclip.
- YouTube/MegaSaM translation magnitude is not metric truth. Translation stays
  scale-free and invariant to positive global scale changes.
- Yaw is required in the plan token because monocular rotation is materially more
  trustworthy than translation scale and is central to framing.
- Real metric speed is supplied on the controller side and must be clamped strictly
  below `7.5 m/s`.
- H20, SSH, docker, Orin, full cache backfills, video download, and multi-GB jobs are
  user-gated. Codex prepares commands; it does not operate those systems.
- No diffusion, language, game data, SAM2, PI-Prober, metric waypoint training, or
  EGO-Planner integration before deterministic B3 gates pass.

## Public Interfaces

### B3 6-D Plan Token

The B3 plan-token contract is:

```python
PLAN_TOKEN_DIM = 6
PLAN_TOKEN_FIELDS = (
    "unit_dir_x",
    "unit_dir_y",
    "unit_dir_z",
    "log_speed_ratio",
    "yaw_rate_norm",
    "valid",
)
```

Semantics:

- `unit_dir_*`: body-frame translation direction from the future camera/drone
  delta. Invalid or near-stationary steps use zeros.
- `log_speed_ratio`: `log(speed / history_only_reference_speed)`, clipped to a
  finite configured range. The reference speed must be computed from observed
  history only.
- `yaw_rate_norm`: `(dyaw / dt) / yaw_rate_cap`, clipped to `[-1, 1]`. The cap is a
  physics/config constant, not learned from future labels.
- `valid`: true/1 when moving, speed-valid, and VO confidence is sufficient. It is
  both a plan-token feature and a mask source; it is not a permission to leak future
  person/world labels.
- Scale-free invariance must hold: positive rescaling of all translation deltas
  leaves direction and relative-speed labels unchanged up to numerical tolerance.
- Metric-speed conversion is outside the model. Controller-side conversion clamps
  commanded speed strictly below `7.5 m/s`.

### Cache Contract Extension

Sports cache `.npz` files gain optional person-track labels:

- `person_bbox (N,4)`: normalized `cx, cy, w, h` in encoder crop coordinates. Absent
  frames store zeros.
- `person_visible (N,)`: boolean or 0/1 visibility flag.
- `person_conf (N,)`: detector/tracker confidence in `[0,1]`.

The loader must keep old caches working by filling missing keys with invisible
person labels. Manifest/backfill logs record detector family, model id/version,
tracking method, prompt/classes, crop convention, thresholds, and backfill command.

### B3 Training Batch

Replace B1/B2 runnable training batches with a B3 world-model batch shaped around:

- inputs: `z_t`, `history_latents`, `history_mask`, observed history motion if used,
  `planned_actions (B,T,6)`, `dt_seconds (B,T)`, and source/window metadata;
- labels only: `target_latents`, `person_state_target (cx, cy, log_h, visibility)`,
  `person_visible`, `person_conf`, person-derived patch weights, VO/quality masks,
  and optional inverse-dynamics labels.

`target_latents`, future person labels, and future masks must not be accepted by
model `forward`.

### Model Output

The B3 wrapper returns:

- `predicted_latents (B,T,196,768)`;
- `predicted_person_state (B,T,4)` as `(cx, cy, log_h, visibility_logit)`;
- optional inverse-dynamics auxiliary `predicted_plan (B,T,6)`.

`LatentPredictor` changes from global 4-D action FiLM to per-step 6-D plan
conditioning plus per-step `dt`. The B3 path keeps residual latent output.

## Step Status

| Step | Status | Gate | Summary |
|---|---|---|---|
| B3.0 | done | USER approval received 2026-07-07 | Write/approve B3 plan and align active guidance |
| B3.1 | done | AUTO, verified 2026-07-07 | Cleanup obsolete B1/B2 runnable paths |
| B3.2 | done | AUTO + user pasteback, verified 2026-07-07 | Person-track cache backfill and data screens |
| B3.3 | pending | AUTO | 6-D plan-token contract and T configurability |
| B3.4 | pending | code AUTO, full gates USER-gated if long | Stage-0 probes plus K1/K2 |
| B3.5 | pending | AUTO | Depth-6 per-step conditioned world model |
| B3.6 | pending | AUTO/local, stop on OOM/blocker | Stage-1 local gates G1a-G1d |
| B3.7 | pending | USER-GATED H20 | One serious depth-6 H20 run |
| B3.8 | pending | AUTO local, Orin later USER-gated | CEM/MPPI hindsight-replay planner eval |

## B3 Queue

### B3.0 - Write/Approve Phase B-3 Plan

- DoD: create this file; update `AGENTS.md`, `.codex/ralph-rules.md`, `README.md`,
  and `DEV_LOG.md` so B3 is active and B2.12/H20 is inactive.
- Test:
  `$PY -m pytest` is not required for docs only. Required grep:
  `rg -n "Phase B-3|B3.1|candidate future|PLAN_TOKEN_DIM|G0|K1|K2|B2.12" plans/phase-b3-human-conditioned-world-model.md AGENTS.md .codex/ralph-rules.md README.md DEV_LOG.md`
- Note: do not commit existing deletions of historical plan/report files unless the
  user explicitly confirms.

### B3.1 - Reviewed Cleanup Of Irrelevant B1/B2 Runnable Code

Preserve reusable infrastructure:

- source splitting, source-balanced data selection, sports cache loading, DINO
  encoding, ingest/VO/yaw extraction, checkpointing, optimizer/loss helpers,
  scale-free utilities, and tests that still guard source split, scale invariance,
  no target leakage, and tier purity;
- `vllatent/model/action_policy.py`, `vllatent/model/heads.py`,
  `vllatent/train/action_metrics.py`, `tests/test_action_policy.py`, and
  `tests/test_action_metrics.py` until B3 equivalents exist, because they seed the
  6-D prior/inverse-dynamics diagnostics.

Removed in B3.1 as obsolete runnable B1/B2 paths:

- `scripts/train_sports.py`
- `vllatent/model/sports_model.py`
- `vllatent/train/evaluate.py`
- `tests/test_model.py`
- `tests/test_evaluate.py`
- `tests/test_train_sports_residual.py`
- `scripts/train_sports_b2.py`
- `vllatent/model/world_action_model.py`
- `tests/test_train_sports_b2.py`
- `tests/test_world_action_model.py`

Also fix stale entry points such as Makefile help/targets that point at removed or
already-missing verifier paths. Do not delete historical plans, reports, or log
entries without explicit user approval.

- DoD: reviewed path list is recorded here or in `DEV_LOG.md`; no active imports
  reference removed paths; historical plans/reports/logs are preserved unless
  separately approved.
- Test:
  `$PY -m pytest -q tests/test_smoke.py tests/test_config.py tests/test_sports_loader.py tests/test_collate.py tests/test_scale_free_targets.py tests/test_predictor.py tests/test_losses.py tests/test_checkpoint.py && bash scripts/check_no_blobs.sh`
- Verified 2026-07-07: active-reference scan clean; pytest subset passed
  `157 passed`; `bash scripts/check_no_blobs.sh` passed.
- Deps: B3.0. Blocks B3.2/B3.3.

### B3.2 - Person-Track Cache Backfill And Data Screens

Add YOLO-World plus ByteTrack person tracking, longest/central subject selection,
cache keys, manifest provenance, loader fallback for old `.npz`, and screen reports
for time remap, duplicate frames, accel outliers, and person presence. Calibrate the
artifact screens against cand06-like failures.

Full 908-clip cache backfill is user-gated. Implementation may run fixture and
dry-run tests only.

- DoD: old caches still load; new caches expose person labels; screen report gives
  clip/window/source counts and flags cand06-like artifacts.
- Test:
  `$PY -m pytest -q tests/test_person_tracking.py tests/test_ingest_pipeline.py tests/test_sports_loader.py tests/test_collate.py`
- Verified 2026-07-07: person-track contract, optional cache arrays,
  loader/collate fallback, dry-run-capable backfill script, and cache-screen CLI
  are implemented. Fixture tests passed.
- User pasteback: dry run `20/20 would_backfill`; full backfill `796 backfilled`,
  `102 skipped_existing`, `9 frame_count_mismatch`, `1 missing_frames`.
  The tracker path worked; user manually reviewed low-person sources and judged
  them dataset-side no-goal/true-FPV shots.
- Exclusion decision: delete cache `.npz` files for low/no-person sources
  `cand04`, `cand18`, `cand20`, `cand30`, `ski03`, plus remaining failed
  frame-mismatch rows outside those sources: `cand03_fpv00_c000`,
  `cand03_fpv07_c000`, `cand11_fpv18_c000`, `cand15_fpv00_c000`,
  `cand19_fpv21_c000`, `cand22_fpv00_c000`, `cand36_fpv00_c000`,
  `cand39_fpv02_c001`.
- Post-exclusion T=8 screen: `820` clips, `33` sources, `15,698` windows,
  `8,077` person-valid windows (`51.5%`), `duplicate_frame_runs=0`,
  `time_remap_flags=14,724`, `accel_outlier_frames=1,698`.
- Deps: B3.1. Blocks B3.4/B3.5.

### B3.3 - 6-D Plan Tokens And T Configurability

Add pure 6-D plan token helpers from body-frame deltas and yaw. Make horizon
config-driven so `T=8` works through loader/collate/model tests.

- DoD: scale-free translation invariance holds; yaw-rate is finite/clipped;
  `valid` combines moving, speed-valid, and VO confidence; future person/world
  labels never appear in inputs.
- Test:
  `$PY -m pytest -q tests/test_plan_tokens.py tests/test_sports_loader.py tests/test_collate.py tests/test_predictor.py`
- Deps: B3.1. Blocks B3.4/B3.5.

### B3.4 - Stage-0 Probes Plus K1/K2

Train/evaluate real-latent probes for person presence, center, and log-height.
Run K1 causality and K2 tiny conditioned person-state predictor versus persistence.

- DoD: G0 center error `<~0.1` normalized and presence AUROC `>0.95`; K1 is
  quantified; K2 beats person-state persistence by `>=10%` or B3 stops for replan.
- Test:
  `$PY -m pytest -q tests/test_person_probes.py tests/test_stage0_gates.py`
- Deps: B3.2/B3.3. Blocks B3.5/B3.6.

### B3.5 - Per-Step 6-D Conditioning In Depth-6 Predictor

Implement the B3 human world model with per-step plan FiLM/additive embeddings,
action dropout `p=0.2`, per-step `dt`, residual latent output, person-state head,
person-weighted latent loss, and inverse-dynamics 6-D auxiliary head.

- DoD: forward shapes pass for `T=8`; plan causality and plan sensitivity tests
  pass; target latents/person labels are not accepted by `forward`; exact parameter
  count is logged.
- Test:
  `$PY -m pytest -q tests/test_human_world_model.py tests/test_predictor.py tests/test_world_model_losses.py`
- Deps: B3.3/B3.4. Blocks B3.6.

### B3.6 - Stage-1 Local Depth-6 World-Model Gate

Run depth-6 with the smallest viable local batch. If batch=1 OOMs, stop and
request an H20 gate instead of reducing depth as the serious B3 model.

- DoD: tiny overfit works; G1a/G1b/G1c/G1d pass locally or a precise blocker is
  recorded; K6 source-count trend is reported before paid scaling.
- Test:
  `$PY -m pytest -q tests/test_train_sports_b3.py tests/test_human_world_model.py tests/test_world_model_metrics.py`
- Deps: B3.5. Blocks B3.7.

### B3.7 - USER-GATED H20 Depth-6 Run

Provide one paste-ready H20 command only after B3.6 gates justify it.

- DoD: user pasteback includes metrics JSONL tails, source metrics, G1 gate
  readouts, steps/sec, memory, checkpoint/config existence; G1a tightened to
  `>=15%`.
- Rule: no Codex SSH/docker/H20 operation.
- Deps: B3.6. Blocks B3.8 scale-up decisions.

### B3.8 - Planner-Facing CEM/MPPI Hindsight Replay

Implement offline planner scoring over 6-D candidate plans, seeded by B2.11c
translation/speed plus yaw statistics, then replace with a trained 6-D proposal
prior when available.

- DoD: selected plans beat random, inertia, and B2-derived priors under real
  future labels; true logged plan ranks well among 64 candidates; jerk/speed/yaw
  trust-region rejects off-distribution candidates.
- Test:
  `$PY -m pytest -q tests/test_planner_replay.py tests/test_planner_scoring.py tests/test_world_model_metrics.py`
- Deps: B3.5 for local scoring, B3.7 for serious checkpoint scoring.

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

## Ralph Execution Rules

1. Execute the lowest active B3.x step.
2. Read that step's DoD and test command before touching code.
3. Make one bounded change or run one bounded verification.
4. Run the narrowest useful test listed for the step.
5. Append verified facts to `DEV_LOG.md`.
6. Stop at user gates, repeated blockers, H20/SSH/docker/long-job boundaries, or
   when the active completion promise is satisfied.

## Assumptions

- Research memo recommendations 5.1-5.6 are adopted as approved defaults.
- B2.11c is evidence and a partial proposal prior, not the B3 endpoint.
- The first serious B3 path is deterministic and depth-6; broad ablation grids wait
  until deterministic gates identify a need.
- Full-frame latent prediction is a support loss, not the primary success signal.
  Person-state trajectory and person-weighted latent metrics drive planner-facing
  decisions.
