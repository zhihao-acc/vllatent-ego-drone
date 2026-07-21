# Phase B3-CS: Causal Ski Simulator and Latent Decoder

**Date:** 2026-07-14; execution status updated 2026-07-21

**Status:** `B3-CS1`–`B3-CS3` complete; `B3-CS4` blocked at USER and
normative-spec gates

**Current completed step:** `B3-CS3`

**Lowest pending step:** `B3-CS4` (`USER-DATA/CPU-RENDER`, not started)

**Satisfied planning completion promise:**
`CAUSAL SKI-SIM AND LATENT-DECODER REPLAN COMPLETE`

This is the active successor plan for the B3 causal-transition question. It
supersedes the former untracked B3-Sim/B4 draft and the old B3.8 controller/CEM
queue. Historical B3 evidence remains intact.

CS1/CS2 completed on 2026-07-15. CS3 completed on 2026-07-20 under explicit
Blender/asset/CPU-render authority. Completion does not authorize CS4 data
generation, encoding, training, GPU/H20, or controller operation. B3.6 remains
blocked and B3.7/H20 remains ineligible.

## 1. Authority and locked scope

Read, in order:

1. `AGENTS.md`;
2. `.codex/ralph-rules.md`;
3. `DEV_LOG.md`;
4. this plan;
5. only the code/tests named by the active step.

Do not consult Obsidian for this queue.

### Normative-spec availability gate

The original untracked draft delegated several exact contracts to
`reports/action-conditioned-world-model-research-2026-07-06.md` and
`reports/simulator-and-world-model-policy-decision-2026-07-14.md`. Neither file
exists in the workspace or any Git branch/tag/history/stash, and neither was ever
tracked. They therefore cannot appear in the required-read list or be treated as
available authority.

The implemented CS1–CS3 contracts are frozen by reviewed source, tests,
manifests, hashes, and `DEV_LOG.md` evidence. CS4 and later additionally depend
on delegated material that is not fully reproduced here, including the complete
versioned 32-row CS4 root manifest, the complete simulator gate table and label
formula set, continuation cue thresholds, CS7 sibling-delta algebra, and D0/D1
masks/reductions/geometry estimator. Before CS4 mutation, either restore the
normative source documents or migrate all missing clauses into this plan through
a separate complete documentation decision and independent review. Never infer,
simplify, or weaken them.

The only learned transition in scope is:

```text
observed latent history + future camera/drone action sequence
    -> eight future visual latent frames
```

Controller-side IMU, VINS, radar, depth, terrain safety, trajectory optimization,
obstacle avoidance, and low-level flight control stay outside the world model and
decoder. No actor, critic, reward, continuation, CEM/MPPI, PX4, flight dynamics,
online RL, or full autonomy stack may enter this plan.

## 2. Locked research decisions

| Concern | Decision |
|---|---|
| simulator | Blender 4.5.11 LTS, bundled Python API, Cycles CPU, no plugin |
| scene | one 15-degree textured slope with authored primitive obstacles |
| camera | rigid fixed-intrinsics rig, deterministic body-FRD SE(3) integration |
| skier root | fixed-step slope-plane dynamics driven by absolute-time maneuver records |
| skier body | root-free in-place ski clips + deterministic blending + stateless contact IK |
| external asset | Quaternius August 2025 Universal Base Characters Source archive; selected `Regular_Male_FullBody.blend`, embedded CC0-1.0; everything else authored |
| paired branches | zero; +/- yaw, forward, lateral, vertical; later bounded combinations |
| latent model | frozen `timm` `vit_base_patch16_dinov3.lvd1689m` + existing eight-step dense latent predictor |
| simulator action | four command channels; `dt_seconds` remains separate |
| decoder | standalone <=1.5M learned-query spatial pooler + one temporal block |
| decoder output | eight rows of `(cx, cy, log_h, p_visible)` only |
| decoder training | D0 ground-truth tokens, then D1 50:50 ground-truth/predicted tokens; DINO and world model frozen |
| final boundary | offline adapter into the existing controller only after every gate |

Implemented CS1–CS3 facts are pinned by their versioned modules and tracked CS3
manifests. Any shorthand that still says “the report” is deliberately
non-executable until the normative-spec availability gate above is cleared.

## 3. Dependency DAG and queue

```text
B3-CS0 research/documentation replan (DONE)
  |
  v
B3-CS1 renderer-neutral pure contracts (DONE 2026-07-15)
  |
  v
B3-CS2 deterministic skier root/digest proof (DONE 2026-07-15)
  |
  v
B3-CS3 Blender scene, rig, animation, and eight-root feasibility
        (DONE 2026-07-20)
  |
  v
B3-CS4 32-root x 9-branch paired smoke and simulator proof
        (USER data-generation + normative-spec gates)
  |
  v
B3-CS5 256-root paired corpus, manifest/loader, and action adapter
        (USER render/data/encoding authority)
  |
  v
B3-CS6 eight-step latent-transition training (USER training/resource authority)
  |
  v
B3-CS7 causal action, sibling-latent, and held-out evaluation
  |
  v
B3-CS8 standalone decoder + D0 ground-truth-token gate
        (USER training/resource authority)
  |
  v
B3-CS9 D1 predicted-token adaptation and decoder held-out gate
        (USER training/resource authority)
  |
  v
B3-CS10 offline controller-interface fixture test (USER interface authority)
```

| Step | Status | Resource class | Current eligibility |
|---|---|---|---|
| B3-CS0 | done 2026-07-14 | DOC/WEB | complete |
| B3-CS1 | done 2026-07-15 | USER-AUTH-LOCAL-PURE | complete |
| B3-CS2 | done 2026-07-15 | LOCAL-PURE | complete |
| B3-CS3 | done 2026-07-20 | USER-ASSET/CPU-RENDER | complete |
| B3-CS4 | pending | USER-DATA/CPU-RENDER | blocked on explicit USER authority and normative-spec restoration/migration |
| B3-CS5 | pending | USER-DATA/CPU-RENDER/ENCODE + LOCAL-CODE | blocked |
| B3-CS6 | pending | USER-TRAIN | blocked; H20 prohibited |
| B3-CS7 | pending | EVAL | blocked |
| B3-CS8 | pending | USER-TRAIN | blocked |
| B3-CS9 | pending | USER-TRAIN | blocked |
| B3-CS10 | pending | USER-INTERFACE/OFFLINE | blocked |

## 4. Load-bearing data contracts

### 4.1 Camera action

```text
requested_command[t] = [v_forward, v_right, v_down, yaw_rate]
dt_seconds[t] = 0.2
record_valid[t] = true for zero, pure yaw, and translation records
```

Storage shapes/dtypes are `(8,4) float64`, `(8,) float64`, and `(8,) bool`; model
inputs are matching float32 action/dt arrays. Camera pose fields are named,
row-major `(4,4) float64` transforms: authoritative `T_world_from_rig[t]` plus
distinct requested/achieved `T_rig0_from_rig_t[t]`.

Requested action, achieved relative SE(3), achieved world SE(3), and camera pose
are distinct fields. Blender camera axes are `+X` right, `+Y` up, `-Z` optical
forward. Semantic FRD maps to camera-local translation as
`[x,y,z]=[v_right,-v_down,-v_forward]`. Name that rotation
`R_cam_from_rig`, its transpose `R_rig_from_cam`, and both translations zero;
positive yaw is a right turn about camera-local `-Y`. Export/integrate canonical
little-endian float64 SI values (`m/s`, `rad/s`, seconds); the yaw branch is
`+/-pi/15 rad/s`. The loader casts actions and separate `dt` to float32 with no
normalization. Hash the action arrays, both named transforms, `K`, and crop/render
camera settings into `camera_contract_sha256`.

### 4.2 Root and sibling identity

`root_id` identifies one initial camera/skier/scene state and maneuver schedule.
`split_group_id` equals the root group and contains the shared three-frame history
plus every camera sibling. No frame or sibling can be split independently.

History ticks `-2,-1,0` hold the camera at its serialized pose under zero command.
Future row `k` is captured after integrating its branch action for one fixed step;
there is no target-following/look-at camera constraint.

The canonical skier digest covers root, skis, contacts, maneuver/animation phase,
bone-local transforms, and randomness. It excludes camera, branch, visibility,
and pixels. Every sibling must have the same digest at the same absolute tick.

Each `H=3,T=8` forecast window has one identifiable continuation law. Its active
target/ramp starts no later than history tick `-2`; no target, maneuver, ramp, or
trigger begins in future ticks `1..8`. A non-steady continuation must be visible
in at least two history frames and clear the report's state-cue threshold. A
pre-render terminal-state-key audit rejects equal observed-state keys with
different continuation-law IDs or target parameters. Held-out compositions expose
their cue and phase in history; hidden post-history sequences are invalid.

### 4.3 Observation labels

The target object set is person/clothing/helmet/boots only. Deterministic projected
mesh rasterization yields the full amodal mask/box; its crop intersection yields
the in-crop amodal mask; the normal depth-tested target-ID pass yields the visible
mask. Export integer areas and the exact report formulas for frame fraction,
visible fraction, occlusion fraction, `(cx,cy,log_h)`, amodal-regression validity,
and `p_visible_target`. Skis, poles, obstacles, and terrain never enter target
geometry. Label thresholds and object membership are manifest-versioned.

### 4.4 Decoder interface

```text
target_state.shape = (8, 4), dtype = float32
target_state[k] = [cx, cy, log_h, p_visible]
t_offset_seconds.shape = (8,), dtype = float32
t_offset_seconds[k] = (k + 1) * 0.2
coords = 224x224 encoder crop, x right, y down
```

Only this trajectory, timestamps, and a fixed intrinsics identifier may cross the
future controller adapter. Simulator root state, depth, range, masks, bone state,
maneuver labels, and privileged camera truth may not cross it.

## 5. Step cards

### B3-CS0 — Research and documentation replan

- **Objective:** replace the rejected controller-heavy plan with one
  source-grounded causal ski-simulator and latent-decoder plan.
- **Dependencies:** locked user direction dated 2026-07-14.
- **Work:** inspect the required repository files and scoped B3 code; research
  current primary engine, license, DINO/decoder, and skiing-dynamics sources;
  draft the decision material, this plan, and active guidance.
- **Outputs:** this plan, aligned Ralph guidance, historical plan status notes,
  and a `DEV_LOG.md` correction. The draft's referenced decision reports were not
  retained or tracked; Section 1 now makes that an explicit CS4+ blocker.
- **DoD:** facts/decisions/user choices are separate; one engine, skier mechanism,
  decoder, interface, dataset, and gate set are selected; B3.6/H20 status is
  unchanged; no prohibited operation occurs.
- **Verification:** scoped link/reference scan, stale-authority scan,
  `git diff --check`, and independent read-only review.
- **Failure action:** correct documentation only; do not activate implementation.
- **Authority:** documentation only.

### B3-CS1 — Renderer-neutral pure contracts

- **Objective:** create the smallest inspectable causal-record schema before any
  renderer code.
- **Dependencies:** explicit user authorization after CS0.
- **Work:**
  - define four command channels plus separate `dt_seconds` and validity;
  - define the nine branch IDs/programs and body-FRD sign convention;
  - lock SI units, float64 canonical storage, float32 model inputs without
    normalization, and separate `dt`;
  - lock the exact named `R_cam_from_rig`/`R_rig_from_cam` matrices, zero
    translations, yaw axis, sign-eligibility rule, expected image-sign table, and
    camera-contract hash;
  - define root/sibling/split IDs;
  - define requested versus achieved SE(3);
  - define a minimal canonical skier-state/pose digest contract;
  - add focused pure tests.
- **Expected surfaces:** a clean, reviewed PURE contract module and focused test
  file. Existing untracked `vllatent/sim/` and simulator tests are not adopted
  automatically; audit or replace them file by file.
- **DoD:**
  - zero and pure yaw are valid records;
  - command shape is `(T,4)` and `dt` is not a fifth optimizable action;
  - `yaw_rate` is radians/second and the pilot yaw value is exactly `pi/15`;
  - serialized camera/action arrays and the camera-contract hash are stable;
  - branch enumeration is exactly nine;
  - `+yaw` and `+right` move an eligible target left, `+down` moves it up, and
    `+forward` increases `log_h`, with opposite effects for negative branches;
  - camera/branch fields cannot enter the skier digest;
  - split validation rejects separated siblings;
  - requested/achieved transforms cannot alias.
- **Verification:** narrow pure pytest target plus import-boundary/AST check.
- **Failure action:** stop at the contract; do not install Blender or touch model
  code.
- **Authority:** completed under explicit `USER-AUTH-LOCAL-PURE` authority on
  2026-07-15.

### B3-CS2 — Deterministic skier root and digest proof

- **Objective:** prove the root law and branch-independent skier future without a
  renderer.
- **Dependencies:** passing CS1.
- **Work:** implement the versioned float64 fixed-step slope-plane update,
  quintic maneuver schedule, steady-carve constraint, bounded braking term,
  deterministic ground-root/armature construction, ski dimensions, stance and
  centerline/base/binding/contact origins, left/right commanded ski-frame and
  longitudinal/lateral slip law,
  forecast-continuation eligibility/audit, absolute-tick animation parameters,
  canonical serialization, and mechanical residual diagnostics.
- **DoD:**
  - repeat runs produce byte-identical canonical records;
  - straight, accelerate/tuck, brake, left/right carve, crouch/transition, and
    occlusion-path fixtures meet the frozen CS2 mechanical gates;
  - straight/carve parallel-ski and brake opposing-wedge frame/slip fixtures meet
    every realized attack/edge/orientation/slip residual gate;
  - root/armature construction residuals, strict ski ordering, inner-tip gap, and
    every stance/origin residual meet the frozen fixture gates;
  - changing a camera branch produces zero digest difference;
  - no future maneuver/target/ramp starts after history and equal quantized
    terminal-state keys cannot select different continuation laws;
  - every non-steady forecast has a visible history cue meeting the frozen CS2
    audit threshold;
  - ideal carving is checked only in steady high-edge holds;
  - no camera, renderer, wall clock, or stateful random input enters advancement.
- **Verification:** pure deterministic/replay/property tests over the eight
  canonical root fixtures.
- **Failure action:** repair or simplify the root law once; otherwise replan
  before rendering.
- **Authority:** completed as local pure work on 2026-07-15.

### B3-CS3 — Blender feasibility, rig, and meaningful body dynamics

- **Objective:** demonstrate one legal rigged skier and one deterministic
  fixed-intrinsics scene before dataset generation.
- **Dependencies:** passing CS2; explicit user authorization for Blender/asset
  acquisition and CPU rendering.
- **Work:**
  - acquire and hash the official Blender 4.5.11 build and selected CC0 character;
  - save license/provenance evidence;
  - before pose authoring/evaluation, freeze exact evaluated bone names, rest
    frames, root-local metric frame, left/right mapping, and boot exclusions;
  - author slope, texture, obstacles, equipment, and root-free ski clips;
  - attach the fixed camera rig;
  - implement absolute-tick pose blending and stateless boot-binding IK;
  - bind dimensioned ski meshes, bindings, and boot targets to the authoritative
    stance/origin/frame construction; animation cannot override them;
  - implement the versioned amodal-mesh and depth-tested visible-mask label
    construction frozen by the CS3 label schemas;
  - render eight canonical roots under zero camera motion twice.
- **DoD:**
  - same-host fresh-process state, pose, RGB, and mask hashes are exact;
  - every numeric animation-amplitude, edge/lean timing, transition-flexion,
    mirrored-pose, crouch, stance/tip-gap, binding-slip, and contact gate passes;
  - realized ski-mesh attack/edge/frame and longitudinal/lateral slip residual
    gates pass for straight, carve, and braking fixtures;
  - temporary occlusion alters only observation labels;
  - camera intrinsics and target mask projection are independently checked;
  - asset manifest contains only approved CC0 and authored content.
- **Verification:** manifest audit, deterministic replay report, contact/body
  metrics, and visual contact-strip review.
- **Failure action:** one bounded rig/render diagnosis; if exact replay or
  meaningful ski motion still fails, stop and re-decide the build. Do not switch
  engines automatically.
- **Authority:** completed under explicit USER external acquisition and CPU
  rendering authority on 2026-07-20.

### B3-CS4 — Paired 32-root causal smoke

- **Objective:** prove exact-root paired interventions and image geometry at the
  smallest useful scale.
- **Dependencies:** passing CS3; explicit data-generation authority; and the
  normative-spec availability gate in Section 1.
- **Work:** render 32 roots x nine branches x eight future frames plus shared
  histories, but first restore or fully migrate, review, freeze, and hash the
  complete versioned 32-row root manifest; independently replay one complete
  branch per root; validate all records and split groups. A changed row restarts
  the smoke under a new version.
- **DoD:** every restored/migrated simulator/skier gate passes, including
  byte-identical replay, identical skier futures, `1e-6` achieved SE(3), one-pixel
  pinhole geometry, correct signed axis effects, exact amodal/visible label
  identities and formulas, complete labels, and indivisible split groups.
- **Verification:** one machine-readable smoke audit and one concise proof table;
  no model training.
- **Failure action:** simulator/exporter NO-GO. Do not encode DINO or expand data.
- **Authority:** USER-gated render/data generation; currently unauthorized and
  specification-blocked.

### B3-CS5 — 256-root paired corpus, manifest, loader, and action adapter

- **Objective:** collect and audit the first training corpus, then make it
  consumable without corrupting historical passive-video semantics.
- **Dependencies:** passing CS4 and explicit user authority for the exact
  256-root CPU render, storage, frozen-DINO encoding resource, and wall-time.
- **Work:**
  - freeze the root table and split before rendering: 160 ordinary train roots,
    32 ordinary validation roots, 32 ordinary test roots, 16 held-out maneuver
    composition roots, and 16 held-out starting-state roots;
  - render exactly `256 x 9 x 8 = 18,432` future RGB/mask pairs and
    `256 x 3 = 768` shared history pairs;
  - audit every root/branch/digest/label and independently replay a preregistered
    10% root sample before encoding;
  - reject hidden post-history events, missing history cues, or terminal-state-key
    collisions with different continuation laws before rendering/encoding;
  - after the RGB audit passes, encode with the frozen pinned DINOv3 encoder under
    the separately approved resource; no H20;
  - record model ID/upstream revision where available, exact weight SHA-256,
    `timm`/PyTorch versions, preprocessing/crop hash, dtype/device, and a
    deterministic repeat-encoding check in each latent manifest;
  - encode the preregistered CS4 same-branch fresh-process repeat pairs, compute
    float32 latent-delta norm/MSE distributions, and freeze their 99th-percentile
    noise floors before CS6;
  - write immutable root-group manifests and held-out factor tags;
  - load shared history and nine sibling futures without leakage;
  - add the four-channel simulator action adapter with separate `dt`;
  - carry full/in-crop/visible areas, amodal/visible labels and masks, regression
    validity, and binary visibility exactly as defined in the report;
  - audit the fixed-obstacle, frame-fraction-qualified visibility slice and fail
    collection unless each required split has at least 50 positive pre/post and 50
    negative during-obstacle steps;
  - preserve the old six-field passive token under its existing path;
  - generate or encode nothing beyond the approved root table.
- **DoD:** exact RGB/mask/history/latent counts and hashes are present; the full
  corpus passes the simulator structural audit; the replay sample is exact;
  latent shape/provenance is reproducible; repeat-noise floors are frozen;
  shape/dtype/frame/sign tests pass;
  zero/pure-yaw stay valid; siblings never cross splits; future labels never enter
  model forward; held-out maneuver compositions and start-state tuples are absent
  from training; every forecast continuation is history-identifiable and the
  terminal-state-key collision audit is empty.
- **Verification:** machine-readable collection/encoding audit, root/split
  manifest audit, replay report, and focused loader/collate/action tests.
- **Failure action:** stop before latent training. Do not partially train on an
  unaudited corpus or silently render/encode replacement roots.
- **Authority:** USER-DATA/CPU-RENDER/ENCODE plus local code; no part is currently
  authorized and H20 is prohibited.

### B3-CS6 — Eight-step latent-transition training

- **Objective:** train only the causal visual transition on the approved paired
  corpus.
- **Dependencies:** passing CS5; explicit training/machine/time authority.
- **Work:** freeze DINOv3; train the existing block-causal `H=3,T=8` predictor with
  four-channel future actions and separate `dt`; use float32 Smooth-L1 beta `0.1`,
  mean over 768 dimensions, patch weights `0.25 + 2.0*person_patch_weights`, and
  normalize separately over 196 patches for every root/branch/horizon exactly as
  specified in the report; equally average those per-window values over roots,
  branches, and horizons; do not reuse the current batch/time-global
  `person_weighted_latent_loss` unchanged; all patches keep background weight and
  only valid track-visible target patches receive the foreground increment;
  balance roots/branches; omit/freeze
  the current mean-pool person head and
  transition verifier and set their state/inverse-plan weights to zero; save exact
  config, split, and checkpoint provenance.
- **Forbidden:** embedded person-state/inverse-plan auxiliary gradients, decoder
  gradients, controller losses, reward/continuation, actor/critic, CEM, H20,
  capacity sweep, or silent paired-difference loss.
- **DoD:** training completes within the separately approved bounded budget and
  produces one evaluation-eligible checkpoint; no causal GO is claimed here.
- **Verification:** training log/config/checkpoint audit and narrow model-forward
  tests.
- **Failure action:** stop. Longer/larger/H20 runs need a new evidence-backed user
  decision.
- **Authority:** USER-TRAIN; B3.7/H20 remains ineligible.

### B3-CS7 — Causal action, sibling-latent, and held-out evaluation

- **Objective:** falsify action use and sibling-difference geometry before decoder
  work.
- **Dependencies:** one CS6 checkpoint and untouched test splits.
- **Work:** using identical targets/masks/primary weights, compute
  true/null/within-root-deranged/axis-flipped and repeat-last-persistence losses,
  per-horizon/root-cluster uncertainty, the fixed plus-zero/minus-zero/plus-minus
  real-versus-predicted sibling deltas, and ordinary/held-out-factor slices.
  Exclude the true zero branch from action-ablation margins and gate its prediction
  against persistence, exact camera-identity metadata, and repeat consistency
  instead; the skier still moves.
- **DoD:**
  - use one preregistered no-fixed-point derangement of all eight nonzero branch
    IDs within each root; temporal shuffling is forbidden for constant commands;
  - on every nonzero branch, true action beats null, that guaranteed-different
    derangement, and the same-axis sign flip by at least 5% overall and at every
    horizon; each per-axis aggregate contains its plus/minus pair and also passes
    5%;
  - root-cluster 95% lower confidence bounds are above zero;
  - every lower bound uses 10,000 whole-root resamples, seed `20260714`, and the
    empirical fifth percentile of the resampled equal-root aggregate margin, with
    no BCa/studentization;
  - at every horizon and separately on every ordinary/held-out slice, true rollout
    beats repeat-last persistence for zero, every axis family, and all branches,
    with positive relative margin and root-cluster 95% lower bound;
  - for each axis use exactly plus-zero, minus-zero, and plus-minus float32
    `(196,768)` deltas by casting each operand before subtraction; flatten only for
    cosine/L2 norm; use the report's exact per-pair cosine, norm ratio, elementwise
    MSE/zero MSE, at-least-two-signal-pair rule, below-floor absolute gate,
    pair-median root statistic, and across-root median; cosine `>=0.75`, norm ratio
    `[0.75,1.25]`, and MSE margin `>=0.05`;
  - all gates pass separately for ordinary roots, held-out maneuver compositions,
    and held-out starting-state tuples.
- **Verification:** immutable metrics artifact and independent metric-algebra
  review.
- **Failure action:** stop before decoder training; diagnose conditioning/data
  contract. Do not hide a failure with aggregate-only reporting.
- **Authority:** bounded evaluation under the resource authority granted for CS6.

### B3-CS8 — Standalone decoder and D0 ground-truth-token gate

- **Objective:** prove that frozen real DINO tokens contain the required compact
  tracking state before exposing the decoder to model errors.
- **Dependencies:** passing CS7.
- **Work:** implement the <=1.5M learned-query spatial/temporal decoder; freeze
  DINO and world model; train only on ground-truth tokens with amodal center/size
  and visibility labels using the report's locked float32 Smooth-L1/BCE objective
  and explicit amodal-regression mask.
- **DoD:** on each ordinary and held-out test set:
  - visible ADE/FDE `<=8/12 px`;
  - occluded amodal ADE/FDE `<=16/20 px`;
  - visible `log_h` MAE `<=0.12`;
  - visibility AUROC `>=0.95`, Brier `<=0.05`;
  - on the off-frame-excluded fixed-obstacle slice, AUROC/Brier `>=0.95/<=0.05`,
    pre/post positive recall `>=0.95`, and during-obstacle negative recall `>=0.90`;
  - each center slice has at least 100 eligible frames and each visibility slice
    has at least 50 positive and 50 negative labels;
  - no maneuver slice exceeds `1.5x` overall ADE.
- **Metric reduction:** use the report's exact visible/temporary-occlusion masks,
  per-root ADE/FDE/`log_h` reduction, root-balanced Brier, and weighted empirical
  AUROC; off-frame truncation is not temporary occlusion.
- **Verification:** decoder parameter count, frozen-parameter audit, D0 metrics,
  split audit, and loss-mask tests.
- **Failure action:** stop. Do not adapt on predicted tokens or attach the decoder
  to the world model.
- **Authority:** USER-TRAIN; no H20.

### B3-CS9 — D1 predicted-token adaptation and decoder gate

- **Objective:** control train/test mismatch while preserving the passing
  ground-truth-token decoder.
- **Dependencies:** passing CS8 D0 and a frozen passing CS7 world model.
- **Work:** initialize D1 from D0 and fine-tune on a deterministic 50:50
  ground-truth/predicted-token mix by duplicating each selected trajectory as one
  clean row and one stop-gradient prediction row generated under its true action;
  keep DINO and world model frozen and record the checkpoint hash.
- **DoD:** on each ordinary and held-out test set:
  - visible ADE/FDE `<=16/24 px`;
  - occluded amodal ADE/FDE `<=24/32 px`;
  - visible `log_h` MAE `<=0.22`;
  - visibility AUROC `>=0.90`, Brier `<=0.10`;
  - on the same fixed-obstacle slice, AUROC/Brier `>=0.90/<=0.10`, pre/post
    positive recall `>=0.90`, and during-obstacle negative recall `>=0.80`;
  - visible ADE is at most `2x` matched D0 ADE;
  - on clean tokens, all absolute D0 thresholds still pass; center/`log_h` errors
    worsen at most 10%, each global/obstacle AUROC drops at most `0.01`, and each
    Brier rises at most `0.01` relative to the frozen D0 checkpoint;
  - decoded geometry uses the plus-minus pair only: yaw/lateral `delta cx`,
    vertical `delta cy`, and forward `delta log_h`; every horizon/axis has at least
    90% correct signs, and the report's through-origin OLS slope `[0.8,1.2]` and
    uncentered `R0^2>=0.80` pass per axis/slice;
  - each center slice has at least 100 eligible frames and each visibility slice
    has at least 50 positive and 50 negative labels;
  - no maneuver slice exceeds `1.5x` overall ADE.
- **Metric reduction:** use the same frozen masks/root-balanced reductions as D0
  and the exact decoded-geometry estimator; no robust-fit or pair substitution.
- **Verification:** frozen-gradient audit, mixture/split provenance, D0 regression
  check, and D1 metrics.
- **Failure action:** stop before controller interface; report the mismatch rather
  than weakening the output contract.
- **Authority:** USER-TRAIN; no H20.

### B3-CS10 — Offline controller-interface fixture

- **Objective:** prove only that the compact predicted trajectory can be consumed
  by the existing controller boundary.
- **Dependencies:** passing CS3–CS9 and explicit identification/authority for the
  controller API.
- **Work:** on saved fixtures, run RGB -> DINO -> frozen world model -> frozen
  decoder; pass only `(8,4)`, time offsets, and intrinsics ID into an offline
  adapter.
- **DoD:** schema, axes, units, timestamps, finite values, and visibility gating
  pass; no privileged simulator labels/state cross the boundary; the existing
  controller/optimizer accepts the fixture without actuation.
- **Forbidden:** closed-loop simulation, action optimization research, safety
  claims, controller training, flight, or deployment GO.
- **Verification:** focused offline interface test and privilege-field audit.
- **Failure action:** repair only the adapter contract; do not alter a passing
  world model/decoder without a new diagnosis.
- **Authority:** USER-INTERFACE/OFFLINE.

## 6. Promotion and falsification policy

- A completed implementation step is not evidence that its scientific gate passed.
- Each sibling group is the statistical cluster. Report per-horizon and per-axis
  results; aggregate-only results cannot promote.
- Ordinary test, held-out maneuver composition, and held-out starting-state gates
  must all pass.
- One diagnosis-bounded repair is allowed inside an already authorized step. A
  change to model family, simulator engine, root law, decoder output, dataset size,
  resource class, or loss family requires a documented replan and user decision.
- No failed gate authorizes more data, a longer run, GPU/H20, broader simulator
  scope, or controller work.

## 7. Ralph execution rule

The current Ralph run stops before CS4 while either its USER authority gate or
normative-spec availability gate is unmet. Future iterations identify the lowest
eligible step, perform at most one bounded authorized change, run the narrowest
verification, wait for it to finish, and append only verified facts to
`DEV_LOG.md`.

The rejected B3-Sim/B4/CEM plan has no active card. Its internal `AUTO` labels,
approval claims, and completion promises are historical and confer no authority.

## 8. Smallest next step — USER and specification gate

Do not start `B3-CS4` until the complete missing normative specification is
restored or migrated/reviewed and the user separately authorizes the exact
32-root x nine-branch x eight-future CPU smoke plus shared histories. Once both
gates pass, execute only the frozen CS4 card; do not encode DINO, expand to CS5,
train, use H20, or enter controller work.
