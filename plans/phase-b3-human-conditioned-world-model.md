# Phase B-3: Human-Conditioned Latent World Model Plan

> **Status update 2026-07-14:** B3.6 remains blocked and B3.7/H20 remains
> ineligible. `plans/phase-b3-causal-ski-sim-latent-decoder.md` is the active
> research successor. The former untracked controller-heavy B3-Sim/B4 draft is
> rejected and has no executable card. This file and all of its results remain
> preserved historical evidence; do not continue B3.7/H20 or B3.8/CEM from it. Several runnable paths
> and the former B1/B2 plan named below were removed in the 2026-07-21 P0
> cleanup; recover them from Git history only if historical inspection is needed.

> **Historical creation note:** Created 2026-07-07. This was the active Phase B plan
> when written. It superseded the B2.12/H20
> action-imitation handoff and the active execution portions of
> the former Phase-B sports-training plan (now retained in Git history only).
> B1/B2 plan history, reports, and `DEV_LOG.md`
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
- `person_visible (N,)`: boolean or 0/1 detector visibility after geometry
  sanitization.
- `person_state_valid (N,)`: optional stricter B3 supervision mask for trackable
  followed-subject labels. It is computed from valid non-edge crop boxes, DINO
  patch-scale area, bounded center jumps, and a minimum visible run; old caches
  without the key fall back to a computed mask.
- `person_conf (N,)`: detector/tracker confidence in `[0,1]`.

The loader must keep old caches working by filling missing keys with invisible
person labels. Manifest/backfill logs record detector family, model id/version,
tracking method, prompt/classes, crop convention, thresholds, and backfill command.

### B3 Training Batch

Replace B1/B2 runnable training batches with a B3 world-model batch shaped around:

- inputs: `z_t`, `history_latents`, `history_mask`, observed history motion if used,
  `planned_actions (B,T,6)`, `dt_seconds (B,T)`, and source/window metadata;
- labels only: `target_latents`, `person_state_target (cx, cy, log_h, visibility)`,
  `person_visible`, `person_state_valid`, `person_conf`, person-derived patch
  weights, VO/quality masks, and optional inverse-dynamics labels.

`target_latents`, future person labels, and future masks must not be accepted by
model `forward`.

### Model Output

The B3 wrapper returns:

- `predicted_latents (B,T,196,768)`;
- `predicted_person_state (B,T,4)` as `(cx, cy, log_h, visibility_logit)`.

Cycle supervision is explicit rather than a `forward` output:
`recover_plan(real_previous_latents, next_latents) -> (B,T,5)` recovers only
the physical plan fields. This prevents accidental predicted-to-predicted plan
reconstruction and keeps the semantic `valid` bit out of regression.

`LatentPredictor` changes from global 4-D action FiLM to per-step 6-D plan
conditioning plus per-step `dt`. The B3 path keeps residual latent output.

## Step Status

| Step | Status | Gate | Summary |
|---|---|---|---|
| B3.0 | done | USER approval received 2026-07-07 | Write/approve B3 plan and align active guidance |
| B3.1 | done | AUTO, verified 2026-07-07 | Cleanup obsolete B1/B2 runnable paths |
| B3.2 | done | AUTO + user pasteback, verified 2026-07-07 | Person-track cache backfill, bad-source deletion, and data screens |
| B3.3 | done | AUTO, verified 2026-07-07 | 6-D plan-token contract and T configurability |
| B3.4 | done | AUTO/local, verified 2026-07-12 | Strict-window G0/K1/K2 diagnostic passed |
| B3.4a | done | AUTO local + USER-gated future regeneration, verified 2026-07-12 | YOLO-standard cache/data path prepared; legacy ambiguity provenance is unrecoverable from retained NPZs |
| B3.5 | done | AUTO, verified 2026-07-08 | Patch-local future queries and detector-visible person-state target semantics |
| B3.6 | blocked | AUTO/local, verified 2026-07-13 | Corrected tiny G1b passes, but G1a/G1d fail; held-out rerun is ineligible |
| B3.7 | superseded/ineligible | none | B3.6 remains blocked; no H20 run |
| B3.8 | superseded | none | Controller/CEM work rejected from current causal-transition scope |

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
- `vllatent/model/action_policy.py`,
  `vllatent/train/action_metrics.py`, `tests/test_action_policy.py`, and
  `tests/test_action_metrics.py` until B3 equivalents exist, because they seed the
  6-D prior/inverse-dynamics diagnostics.

The test-only 4-D heads were retired after B3.5/B3.6 supplied the active
person-state head and five-field transition verifier.

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
  `$PY -m pytest -q tests/test_smoke.py tests/test_config.py tests/test_sports_loader.py tests/test_collate.py tests/test_scale_free_targets.py tests/test_human_world_model.py tests/test_losses.py tests/test_checkpoint.py && bash scripts/check_no_blobs.sh`
- Reverified after the 2026-07-14 predictor retirement: active-reference scan
  clean; current pytest subset passed `167 passed`; blob guard passed.
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
- Drift cleanup 2026-07-14: the one-shot backfill script was retired after the
  retained frame tree was removed; supported ingest writes person tracks
  directly and legacy caches continue to use the conservative loader fallback.
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
- Bad-label source deletion 2026-07-07: after visual label audit, user directed
  deletion of `cand11`, `cand28`, and previously identified `cand04`, `cand18`,
  `cand20`, `cand30`. Local cache already had no `cand04`, `cand18`, `cand20`,
  or `cand30`; deleted remaining `cand11` (`18` clips) and `cand28` (`1` clip).
  Frames/reports remain intact.
- Post-bad-source-delete T=8 screen: `801` clips, `31` sources, `15,359`
  windows, `7,880` person-valid windows (`51.3%`), `duplicate_frame_runs=0`,
  `time_remap_flags=14,501`, `accel_outlier_frames=1,656`, `731` flagged clips.
- Label-geometry hygiene 2026-07-07: tracker selection, cache reads, cache-builder
  writes, and raw-frame conversion now mask zero-area/tiny encoder-crop boxes
  invisible (`area < 0.0025`). Backfill writes `person_bbox_space="encoder_crop"`.
  Screen reports include invalid/tiny/edge label, flicker, center-jump, and area
  QC counters.
- Post-sanitize T=8 screen: `801` clips, `31` sources, `15,359` windows,
  `6,638` person-valid windows, `11,396` sanitized person-visible frames,
  `1,857` invalid/tiny visible labels masked, `528` degenerate visible labels,
  `2,781` edge-touching visible labels, `2,599` flicker transitions,
  `duplicate_frame_runs=0`, `time_remap_flags=14,501`,
  `accel_outlier_frames=1,656`.
- Trackable-source deletion 2026-07-07: after per-source montage review, deleted
  `cand38`, `cand40`, and `cand45` from the active cache. Latest T=8 screen:
  `778` clips, `28` sources, `14,900` windows, `2,927` `person_state_valid`
  windows, `4,987` trackable frames, `duplicate_frame_runs=0`,
  `time_remap_flags=13,992`, `accel_outlier_frames=1,559`.
- Subject selection repair 2026-07-07: future YOLO/ByteTrack ingest now scores
  candidate subject tracks by strict B3 trackability-window count first, then
  trackable-frame count, then the old valid-count/centrality/area tie-breakers.
  The ingest pipeline passes configured person-gate history/horizon into tracking.
- Deps: B3.1. Blocks B3.4/B3.5.

### B3.3 - 6-D Plan Tokens And T Configurability

Add pure 6-D plan token helpers from body-frame deltas and yaw. Make horizon
config-driven so `T=8` works through loader/collate/model tests.

- DoD: scale-free translation invariance holds; yaw-rate is finite/clipped;
  `valid` combines moving, speed-valid, and VO confidence; future person/world
  labels never appear in inputs.
- Test:
  `$PY -m pytest -q tests/test_plan_tokens.py tests/test_sports_loader.py tests/test_collate.py tests/test_human_world_model.py`
- Verified 2026-07-07: added pure `vllatent.plan_tokens` with
  `PLAN_TOKEN_DIM=6`, `PLAN_TOKEN_FIELDS`, clipped yaw-rate normalization, and
  VO/motion/speed-composed validity. `SportsTrainingDataset(..., horizon=8)`
  emits `planned_actions (T,6)` and `planned_actions_valid_mask (T,)`;
  `collate_sports_batch` emits `planned_actions (B,T,6)` as the B3
  world-model input. `LatentPredictor(horizon=8)` shape test passes. Future
  person/world labels remain labels only and are not threaded into model
  `forward`. The original `LatentPredictor` shape test was retired with B1;
  current `HumanWorldModel(horizon=8)` coverage preserves the contract.
- Verified command:
  `$PY -m pytest -q tests/test_plan_tokens.py tests/test_sports_loader.py tests/test_collate.py tests/test_human_world_model.py`
  passed (`77 passed`, reverified 2026-07-14). Ruff and `git diff --check`
  passed.
- Deps: B3.1. Blocks B3.4/B3.5.

### B3.4 - Stage-0 Probes Plus K1/K2

Train/evaluate real-latent probes for person presence, center, and log-height.
Run K1 causality and K2 tiny conditioned person-state predictor versus persistence.

- DoD: G0a detector-visible presence from `person_visible` clears a weak
  held-out sanity floor; G0b center/log-height decode only on
  `person_state_valid` clears bounded error thresholds; K1 is quantified; K2
  improves raw person-state MSE over persistence on valid-current, full-future,
  moving person-state rows. Delta improvement is diagnostic only.
- Test:
  `$PY -m pytest -q tests/test_person_probes.py tests/test_stage0_gates.py`
- Verified 2026-07-07: added `vllatent.train.person_probes` and
  `scripts/run_stage0_gates.py`. Synthetic B3.4 tests passed (`10 passed`), ruff
  passed, and full local T=8 gate ran over the post-exclusion cache.
- Full local gate result with 8 spatial projections: B3.4 does **not** pass.
  G0 failed with presence AUROC `0.621`, center L2 error `0.159`, and log-height
  MAE `0.570`. K1 passed/quantified with plan-only R2 `0.0397`. K2 passed with
  `41.35%` improvement over person-state persistence.
- Bounded probe-capacity check with 32 spatial projections still failed G0:
  presence AUROC `0.654`, center L2 error `0.153`, log-height MAE `0.462`; K1
  remained pass and K2 remained pass at `29.21%` improvement.
- Rework/refire 2026-07-07: fixed the person bbox coordinate contract so new
  tracks are selected/stored in DINO encoder-crop coordinates, converted the
  local 820-cache post-exclusion set in place, replaced G0 with a bounded
  token-level torch probe over patch features plus explicit patch coordinates,
  and added train/per-source G0 diagnostics.
- Refired token G0/K1/K2 after conversion: B3.4 still does **not** pass. Default
  token probe (`projection_dim=64`, hidden `128`, `30` epochs) got G0 presence
  AUROC `0.681`, center L2 `0.196`, train AUROC `0.918`, train center L2
  `0.172`; K1 passed with plan-only R2 `0.0387`; K2 passed with `40.97%`
  improvement.
- Strong token-capacity check (`projection_dim=128`, hidden `256`, `60` epochs)
  still failed G0: presence AUROC `0.658`, center L2 `0.219`, train AUROC
  `0.999`, train center L2 `0.155`; K1/K2 remained pass. This points to label
  calibration/source drift/criterion mismatch rather than the previous moment
  feature bottleneck.
- Bad-source-delete refire on the reduced 801-clip cache still failed G0:
  presence AUROC `0.690`, center L2 `0.212`, center L1 `0.132`, log-height MAE
  `0.408`; train AUROC `0.923`, train center L2 `0.163`; K1 passed with
  plan-only R2 `0.0459`; K2 passed with `45.19%` improvement.
- Label-sanitize refire still failed G0: presence AUROC `0.688`, center L2
  `0.209`, center L1 `0.132`, log-height MAE `0.349`; train AUROC `0.950`,
  train center L2 `0.142`; K1 passed with plan-only R2 `0.0270`; K2 passed
  with `37.05%` improvement.
- Human-label gate rework/refire 2026-07-07: `person_visible` now means detector
  visibility, while `person_state_valid` is the stricter B3 followed-subject
  supervision mask. The mask requires valid non-edge encoder-crop boxes, area at
  least 4 DINO patches, bounded consecutive center jumps, and a minimum run of
  3 frames. New ingest runs with `track_persons=True` apply the configured
  `ingest.person_gate_history`/`ingest.person_gate_horizon` before MegaSaM/DINO,
  so segments without at least one usable B3 person-state window are rejected
  early.
- Per-source audit montages were generated under `/tmp/b3_trackable_audit/`.
  After visual/source review, `cand38`, `cand40`, and `cand45` were deleted from
  the active cache; `cand27`, `cand41`, and `cand13` remain because their montage
  samples include plausible follow-subject footage even though many labels are
  weak. The active cache is now 778 clips / 28 sources.
- Active T=8 screen after trackable-source deletion:
  `14,900` windows, `2,927` `person_state_valid` windows, `4,987` trackable
  frames, `11,167` sanitized detector-visible frames, `duplicate_frame_runs=0`,
  `time_remap_flags=13,992`, and `accel_outlier_frames=1,559`.
- Active token G0/K1/K2 refire after trackable-source deletion still failed
  under the old gate:
  G0 presence AUROC `0.752`, center L2 `0.157`, center L1 `0.097`, log-height
  MAE `0.196`; train AUROC `0.993`, train center L2 `0.124`. K1 passed with
  plan-only R2 `0.0199`; K2 failed with conditioned MSE worse than persistence
  (`-5.21%` improvement).
- Review-resolution replan 2026-07-07: G0 now separates detector presence from
  followed-subject state supervision. The token probe trains presence from
  `person_visible`, trains center/log-height only where `person_state_valid`, and
  uses a small residual correction over attention-derived patch coordinates. G0
  thresholds are now presence AUROC `>=0.60`, center L2 `<=0.14`, center L1
  `<=0.10`, and log-height MAE `<=0.25`. K2 now gates on motion-delta
  improvement over persistence (`>=0.0`) while retaining raw state-MSE reporting.
- Active T=8 refire after the replan passed G0/K1/K2:
  `reports/stage0_gates_T8_token_g0_relabel_replanned.json`. G0 presence AUROC
  `0.658`, center L2 `0.134`, center L1 `0.084`, log-height MAE `0.230`; K1
  plan-only R2 `0.0199`; K2 delta improvement `54.9%`; raw state-MSE improvement
  remained `-5.2%`. No active-cache source deletion was performed in this replan;
  visual audit montages for candidate weak sources were generated under
  `/tmp/b3_trackable_audit_g0_relabel/`.
- Review follow-up 2026-07-07: the weak G0/K2 replan is not accepted as a true
  B3.4 pass. Added a reusable upstream person-label quality prefilter so stage0
  metrics and future training can use only clips with full-history usable
  person-state windows. Screen counts now use the same full-history window
  definition as ingest (`2,676` strict valid windows on the active 778-clip
  cache, not the older padded-start `2,927` count).
- K2 now gates on raw person-state MSE improvement again, evaluated only on rows
  with valid current person state, full future `person_state_valid`, and non-static
  future person motion. With the default prefilter, K1/K2 pass locally:
  K1 R2 `0.04795`; K2 raw improvement `18.64%`.
- User replan 2026-07-07: the old `0.95` held-out AUROC probe is no longer a
  hard B3 blocker. It remains useful as a bug detector, but DINO's role is the
  frozen spatial/object latent space; the decisive proof moves to trained-model
  gates: person-state prediction, person-weighted latent improvement, plan
  sensitivity, and source-held-out generalization.
- The best accepted prefiltered refire in
  `reports/stage0_gates_T8_token_prefilter_k2fixed.json` failed the old G0:
  `person_visible` AUROC `0.659`, center L2 `0.120`, center L1 `0.073`, log-height
  MAE `0.183` against old thresholds AUROC `>=0.95`, center L2 `<=0.10`. Stronger
  token probes and stricter label-quality/source-volume filters did not produce
  an honest `0.95` held-out AUROC on the current cache, so the next work is
  upstream label/data replacement rather than AUROC tuning.
- Strict-window refire 2026-07-12: `scripts/run_stage0_gates.py` now defaults
  K1/K2 to full `H=3,T=8` `person_state_valid` windows and records that setting.
  On the retained 1,100-clip / 100-source cache, G0 passed with presence AUROC
  `0.6912`, center L2 `0.1161`, center L1 `0.0731`, and log-height MAE `0.1671`;
  K1 passed with plan-only R2 `-0.00423`; K2 passed with `9.46%` delta-MSE
  improvement over persistence. Combined decision passed. This completes B3.4
  as a bug/data diagnostic; it does not claim that legacy multi-subject
  ambiguity was audited.
- Deps: B3.2/B3.3. Blocks B3.6.

### B3.4a - YOLO-Standard Data Cleanup And Expansion Prep

Use YOLO/ByteTrack label quality, not AUROC source-picking, to prepare the next B3
cache before starting B3.5.

- DoD: active-cache `.npz` files are cleared or excluded only when YOLO/person
  evidence shows bad supervision: no followed human, tiny/edge/self/body-only
  labels, bystanders instead of the followed subject, severe domain mismatch, or
  object-filter violations. Do not delete clips solely because they hurt AUROC.
- DoD: the ingest content filter keeps the existing YOLO object-negative signal,
  then adds a YOLO human-positive signal before FPV range extraction and
  `cut_fixed_clips()`. Accepted auto-clipped ranges must satisfy motion,
  no rejected objects, and visible human evidence.
- DoD: future cache generation keeps `track_persons=True`, so the downstream
  segment gate still verifies strict followed-subject windows before MegaSaM/DINO.
- DoD: prepare ski-first YouTube expansion inputs and paste-ready commands. If
  ski footage is exhausted, extend to adjacent follow-sport footage that matches
  the current depth-6 sports-following domain. Actual video download/full cache
  expansion remains USER-gated; Codex stops before operating it.
- Test:
  `$PY -m pytest -q tests/test_content_filter.py tests/test_ingest_pipeline.py tests/test_person_tracking.py`
- Deps: B3.4 replan. Blocks B3.5.
- 2026-07-08 follow-up replan: after the user-gated 300-clip expansion lands,
  merge the current ~40 B3.4a ski clips into the expanded candidate set, rerun
  ingest/YOLO filtering over the old ~40 clips as well as new clips, then patch
  training/evaluation to consume strict person-valid windows first. Multi-subject
  ambiguity guards (`selected_track_id`, second-best track evidence, ambiguity
  margin/window rejection) are deferred until after this strict-window training
  filter is in place.
- 2026-07-12 local closeout: the retained cache has 1,100 clips / 100 sources /
  12,499 exact strict `H=3,T=8` windows, and the strict data path is active.
  Future ingest supports ambiguity rejection and exact configured `H+T` clips,
  but second-track provenance cannot be reconstructed for the retained legacy
  NPZs; that limitation requires user-gated frame regeneration rather than
  further cache-only filtering.

### B3.5 - Per-Step 6-D Conditioning In Depth-6 Predictor

Implement the B3 human world model with per-step plan FiLM/additive embeddings,
action dropout `p=0.2`, per-step `dt`, residual latent output, person-state head,
person-weighted latent loss, and inverse-dynamics 6-D auxiliary head.

2026-07-08 initial status: implemented as the B3 torch-tier `HumanWorldModel` and
`PlanConditionedLatentPredictor` path. The model `forward` accepts only observed
history/current latents, `history_mask`, `planned_actions (B,T,6)`, and
`dt_seconds (B,T)`; future latents, person labels, confidences, and masks enter
only the loss functions.

YOLO/ByteTrack person labels are converted into bounded soft weights over the
DINO `14x14` patch grid in `vllatent.train.world_model_losses`, with a background
term retained for world tokens. They are not new DINO class tokens.

2026-07-08 rework: fixed future-token patch symmetry. Future tokens are now
patch-local queries initialized from current patch latents plus a learned
`(1,1,196,D)` patch query embedding and per-step plan/dt embeddings. Residual
output remains `z_t + delta`, and future labels remain absent from `forward`.
Plan dropout now zeros dropped plan tokens instead of inverted-scaling the
semantic `valid` field above `1`. Person-state target semantics are split:
visibility target comes from detector-visible `person_visible`, while
center/log-height regression and foreground patch weighting remain masked by
strict `person_state_valid`. The depth-6, D=768, H=3, T=8 B3 wrapper now has
exactly `59,082,250` parameters.

- DoD: forward shapes pass for `T=8`; plan causality and plan sensitivity tests
  pass; residual deltas are patch-local rather than symmetric; target latents/person
  labels are not accepted by `forward`; exact parameter count is logged.
- Test:
  `$PY -m pytest -q tests/test_human_world_model.py tests/test_world_model_losses.py`
- Deps: B3.3/B3.4a. Blocks B3.6.

### B3.6 - Stage-1 Local Depth-6 World-Model Gate

Run depth-6 with the smallest viable local batch. If batch=1 OOMs, stop and
request an H20 gate instead of reducing depth as the serious B3 model.

2026-07-08 initial status on existing active cache: local OOM is not the blocker. BF16
batch=1 and batch=4 depth-6 training steps fit on the RTX 5060 Ti; batch=8 OOMed
under current GPU load. B3.6 is blocked because the trained model does not beat
person-weighted DINO latent persistence: source-split latent-focused 160-step
run had `model_loss=0.163196` versus `persistence_loss=0.162633`, all per-step
rollout comparisons lost to persistence, and true-plan preference stayed below
chance. A 400-step tiny-overfit latent-focused run also remained worse than
persistence on its own 16-window slice. This is an objective/data/conditioning
blocker; do not proceed to B3.7/H20 from these results.

2026-07-08 rework: the initial B3.6 conclusion is retained as pre-fix evidence,
but it is not the final B3.6 gate because B3.5 had patch-symmetric future tokens.
The B3.6 harness now evaluates `--overfit-tiny` on the exact same limited
training indices and reports early/late loss-window means instead of only first
versus last minibatch. Rerun B3.6 after the user-gated B3.4a expansion lands and
strict person-valid-window training/evaluation filtering is implemented.

2026-07-12 strict rerun: repaired the `H+T+1` loader/ingest off-by-one so all
1,100 clips and 12,499 strict windows enter training. Tiny overfit now beats
persistence by `14.34%` and G1b passes at every step, but it remains only `1.83%`
better than null and misses shuffled-plan preference at `68.75%`. On a fixed
source-disjoint 1,024-train/512-validation split, the 800-step model beats
persistence by only `3.38%`, is `0.44%` worse than null, and beats shuffled/
flipped plans on `42.58%`/`54.10%`. Raising inverse-plan weight from `0.01` to
the research-recipe `0.5` worsens all gates. Extending the better objective to
2,400 steps leaves plan separation flat (`3.61%` over persistence, `0.83%`
worse than null, `42.58%`/`59.77%` shuffled/flipped), while G1b still passes at
all eight steps. B3.6 is blocked on objective/conditioning; B3.7/H20 is not
eligible.

2026-07-13 review-backed repair: replace the circular plan decoder with an
action-blind verifier trained on real consecutive DINO transitions, freeze it,
and apply cycle consistency to a real previous latent plus the predicted next
latent. Regress only the five physical action fields. Whole-window plan dropout
now returns its keep mask and dropped plans are excluded from the auxiliary.
Model `forward` exposes no predicted-plan tensor or hidden plan dropout; the
transition pair and keep mask must be supplied explicitly.
The exact wrapper has `59,280,137` parameters, including `397,829` verifier
parameters. Component/per-field losses and shared-predictor gradient
norms/cosine are logged.

G1d now uses three fixed global different-clip derangements, no fixed points,
minimum physical-plan RMS distance `0.10`, a strict per-window majority across
negatives, window-weighted aggregate losses, required `5%` shuffled/flipped
margins, source-cluster bootstrap intervals for paired margins, boundary-aware
Wilson intervals over source-majority outcomes, a minimum five-source coverage
floor, and a yaw-only person-center geometry check. Overlapping windows are
clustered by source rather than treated as independent. Serious capped runs use source-balanced
selection and report represented rather than nominal sources. Future ingest
accepts exact configured `H+T` clips and ranks subject tracks by strict-window
count first. The old K1/K2 results remain diagnostics rather than proof of
incremental causal plan signal; legacy second-track ambiguity provenance cannot
be recovered from the retained caches.

The corrected same-seed 16-window tiny run (`400` verifier + `400` predictor
steps, batch `2`, `lambda_inverse_plan=0.01`) passed the verifier prerequisite
at `+89.70%` versus a train-mean-plan baseline and improved training loss by
`41.24%`. G1a still failed: `+15.82%` versus persistence but only `+1.38%`
versus null. G1b passed every `k<=8`. G1d failed: strict-majority shuffled and
flipped wins were each `16/16` with `9/9` source-majority wins (Wilson lower
`70.09%`), but aggregate margins were only `+3.04%` (source-cluster paired 95%
CI `1.93–4.32%`) and `+3.24%` (`2.51–3.92%`), and yaw geometry was `0/8`
across six eligible sources (Wilson upper `39.03%`). A single paired
`lambda_inverse_plan=0` diagnostic produced `+15.20%` persistence, `+1.57%`
null, `14/16` shuffled and `15/16` flipped wins, `+2.44%`/`+1.56%` margins,
and `0/8` yaw geometry. The final pair suggests a small conditioning effect,
but repeated BF16 CUDA executions were not bit-deterministic, so sub-point
deltas are directional single-run evidence rather than a multi-seed causal
estimate. Both objectives remain far below G1a/G1d. The source-held-out rerun
was skipped because corrected tiny health failed; B3.7/H20 remains ineligible.

- DoD: tiny overfit works; G1a/G1b/G1c/G1d pass locally or a precise blocker is
  recorded; K6 source-count trend is reported before paid scaling.
- Test:
  `$PY -m pytest -q tests/test_train_sports_b3.py tests/test_human_world_model.py tests/test_world_model_metrics.py`
- Deps: B3.5. Blocks B3.7.

### B3.7 - SUPERSEDED/INELIGIBLE H20 Depth-6 Run

Historical card only. B3.6 did not pass and the B3-CS successor does not
reactivate H20. Do not provide or run this command.

- DoD: user pasteback includes metrics JSONL tails, source metrics, G1 gate
  readouts, steps/sec, memory, checkpoint/config existence; G1a tightened to
  `>=15%`.
- Rule: no Codex SSH/docker/H20 operation.
- Deps: B3.6. Blocks B3.8 scale-up decisions.

### B3.8 - SUPERSEDED Planner-Facing CEM/MPPI Hindsight Replay

Historical rejected card only. Controller-side planning is outside the active
B3-CS causal-transition and decoder scope.

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
| G0 | detector-visible presence clears a weak held-out sanity floor, and center/log-height decode on `person_state_valid` clears bounded errors | fix labels/probes or replan |
| K1 | plan-only camera-compensated person motion is near chance | rework causal separation or abort |
| K2 | tiny conditioned predictor improves person-state motion deltas over persistence | abort or replan dense WM path |
| G1a | conditioned predictor beats person-weighted latent persistence `>=10%` and null-plan `>=5%` | objective/conditioning bug hunt |
| G1b | rollout beats persistence at every k<=8 | shorten/reweight before scaling |
| G1c/K4 | probe transfer passes and gameability check passes | calibrate probes or state-head primary |
| G1d/K3/K5-lite | true 6-D plan wins on `>=70%` windows, source-majority Wilson confidence is above chance with at least five sources, aggregate shuffled/flipped margins reach `>=5%` with positive source-cluster paired confidence, and yaw-only flips move predicted person center correctly across at least five eligible sources | strengthen conditioning before capacity/data |
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
