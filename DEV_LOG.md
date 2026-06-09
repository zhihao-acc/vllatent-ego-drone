# DEV_LOG — vllatent-ego-drone

Append-only, **newest entry on top**. Read this first each iteration to find the current position,
then re-read the relevant step in `plans/phase-a-data-and-io-contract.md`. Project-level *why* lives in
the vault (`latent-pred-pipeline/`), not here; this log tracks *code state* + step status.

## Step status table

| step | status | date | notes |
|---|---|---|---|
| 1 — scaffold + git + GitHub + codegraph | done | 2026-06-08 | scaffold+git+codegraph green; private repo `zhihao-acc/vllatent-ego-drone` created + pushed direct to github.com (workflow scope added); `origin` resolves, `main` tracks `origin/main` |
| 2 — transcribe I/O contract → docs/io-contract.md | done | 2026-06-08 | DoD item 1; 4 seams + loader tuple + 2 foot-guns transcribed from vault arch §4/§6/§9; DoD grep PASS |
| 3 — pure-tier tuple schemas | done | 2026-06-08 | `vllatent/schemas.py` (StepSample/EpisodeRecord/CacheManifestEntry, frozen+validated) + test_schemas (22 tests) |
| 4 — discrete→4-DoF action mapping | done | 2026-06-08 | `vllatent/actions.py` (Action enum + constants verbatim; apply_delta reproduces env_utils; pose_pair_to_body_delta) + test_actions (64) |
| 5 — AerialVLN JSON audit parser (fixture) | done | 2026-06-08 | `vllatent/audit.py` (parse_episode/audit_episode + AuditReport/QuaternionVerdict + CLI) + tiny & quaternion_trap fixtures + test_audit; `make audit` clean. **`reference_path` schema corrected to 6-wide Euler in 5b** |
| 6 — fetch real dataset JSON slice | done | 2026-06-08 | USER downloaded full splits (Kaggle/Baidu, NOT S3); `fetch_aerialvln_json.sh` finished (slicer); `train.slice.json` (50 eps); CC BY-NC-SA 4.0 |
| 5b — audit on real slice | done | 2026-06-08 | 50/50 ok, ~10198 transitions **0 Δ-mismatches**, all 8 classes, quaternion consistent (34/50 would corrupt yaw w/o reorder) |
| 7–13 (old DINOv3-latent pipeline) | superseded | 2026-06-08 | **replaced by A5.1–A5.18** per `plans/phase-a5-replan-postpivot.md` (post-WorldVLN-pivot re-plan) |
| A5.1 — extract public frame/quaternion primitives → frames.py (M1) | done | 2026-06-08 | `frames.py` owns public `yaw_from_xyzw`/`xyzw_from_yaw`/`wrap_pi`/`reorder_wxyz_to_xyzw`; actions.py+audit.py import them; no private cross-module imports; L1 verified (ROW_WIDTH=6); 102 pure tests green |
| A5.2 — test_frames.py no-flip + NED→FLU→ENU remap math (M2) | done | 2026-06-08 | `frames.py` R_FLU_FROM_FRD/R_ENU_FROM_NED + ned_frd_to_flu/ned_to_enu/remap_waypoint_ned_body_to_flu; `tests/test_frames.py` (7) pins no-flip basis + det=+1; collected by `make test` (102→109); live fly0 wiring Phase D |
| A5.3 — frozen typed Config + from_yaml + validation (H1/H2/L2/L3) | done | 2026-06-08 | frozen `Config` tree (encoder/predictor/distill/trust/data/cache) + `from_yaml` (env-expand, strict unknown-key reject) + boundary validation; replaces orphan `load_config`; swept knobs single-sourced (loader reads from Config, default.yaml trimmed to overrides); trust placeholders for A5.9; test_config (18); 127 pure tests green |
| A5.4 — typed manifest builder fed from Config (M5) | done | 2026-06-08 | `manifest.build_manifest(Config, …)` is the one builder; `empty_manifest` delegates to it; 196/768 from `schemas` constants, version from `CacheConfig`, encoder-id/dtype/convention/dataset(name+license) from `Config`; stubbed `teacher` provenance section (worldvln id+rev, disagreement_source-from-Config, vjepa2 id, render hash); entry-required-keys derived from `CacheManifestEntry.required_keys()` (type-enforced, not hand-kept); 127→134 tests |
| A5.5 — student seams PredictorOutput/TrustReadout/Waypoint (H3) | done | 2026-06-08 | frozen+validated `PredictorOutput.predicted_latents (T,196,768) fp16`, `TrustReadout {p_commit (T,)∈[0,1], k_star∈[0,T] float, sigma≥0}`, `Waypoint.delta_4dof (4,) f32 NED-body`; io-contract §0 references them; teacher `OracleTarget` seam deferred to A5.9; 134→150 tests |
| A5.6 — StepSample history_mask + lang padding-mask (M4) | done | 2026-06-08 | `StepSample` gains `history_mask (H,) bool` (block-causal, real vs zero-pad at episode start) + `lang_mask (M,) bool` (== M of lang_tokens); `MASK_DTYPE=np.bool_`; validation + length cross-check; loader-tuple in io-contract §2 updated (9-tuple); 150→155 tests |
| A5.7 — AuditSummary slice aggregator (M3) | in_progress | 2026-06-09 | CODE done+green: `AuditSummary` + `summarize_episodes` + `--slice/--summary/--split` CLI; the dataset-level checks (all-8-classes UNION, scene-id min..max, splits) computed at SLICE scope (M3 — were mis-scoped per-episode); 162→167 tests. **Real-slice re-run USER-GATED** — command block emitted, awaiting the user's pasted aggregate |
| A5.8 — investigation: WorldVLN determinism/weights/6-DoF/license | done | 2026-06-09 | USER-verified probe of `EmbodiedCity/WorldVLN`: weights complete (~36.9 GB; InfinityStar 4-shard backbone + 1.06 GB action decoder + 0.74 GB VAE); inference **STOCHASTIC by default** (top_k900/top_p0.97/cfg34, per-segment seed) ⇒ K-rollout disagreement FREE (overturns prior "deterministic"); action head **6-DoF [roll,yaw,pitch,x,y,z]** SE(3)-integrated vs our 4-DoF student ⇒ 6→4 projection (A5.9); ckpt env `INFINITY_CKPT`+`ACTIONHEAD_CKPT`; lang enc T5; **LICENSE SPLIT** code CC BY 4.0 / weights `license:other` (flag pre-publication) |
| A5.9 — TeacherOutput/OracleTarget seam + finalize Config placeholders | pending | | PURE/AUTO; after A5.8 |
| A5.10 — DINOv3 student-encoder wrapper | pending | | TORCH; contract AUTO / real-weight USER-GATED |
| A5.11 — frozen WorldVLN teacher wrapper | pending | | TORCH; USER-GATED (server); after A5.8 |
| A5.12 — V-JEPA-2 surprise verifier wrapper | pending | | TORCH; USER-GATED |
| A5.13 — render harness | pending | | SIM; unit AUTO / live USER-GATED (docker+UE4) |
| A5.14 — render→[DINOv3+WorldVLN+V-JEPA-2]→cache + provenance manifest | pending | | SIM+TORCH; manifest AUTO / small-slice USER-GATED |
| A5.15 — distillation loader (StepSample+OracleTarget, masks, H/T from Config) | pending | | TORCH/AUTO |
| A5.16 — loader over real teacher/oracle dump | pending | | USER-GATED |
| A5.17 — size full render→teacher→cache job | pending | | sizing AUTO / bulk USER-GATED |
| A5.18 — Phase-A DoD verification | pending | | USER-GATED final sign-off |

Statuses: `pending` / `in_progress` / `done` / `blocked` / `superseded`.

---

## 2026-06-09 — A5.7: AuditSummary slice aggregator (M3) — CODE done; real-slice USER-GATED (STOP CHECK)
**Status:** A5.7 pending → in_progress (PURE code AUTONOMOUS; the real-slice re-run is USER-GATED per
ralph-rules — command block emitted, awaiting the user's paste).
**What's done.** Closed M3's "dataset-level checks mis-scoped per-episode" by adding `AuditSummary` +
`summarize_episodes(reports, *, splits)` to `vllatent/audit.py`: the SLICE-scope aggregate computes
`all_action_classes_present` as the UNION over the slice, `scene_id_range` as min..max across episodes,
`splits_present` from the slice, plus `n_episodes`/`n_ok`/`n_transitions`/`total_delta_mismatches`/summed
`action_counts`/`n_reorder_consistent`/`n_naive_would_mismatch`, and `ok` = every-episode-ok AND
all-classes. New CLI surface `--slice <file> --summary <out|-> [--split …]` (split auto-inferred from the
filename, e.g. `train.slice.json`→`train`); the per-episode `--episode` path is unchanged (`make audit`
still green). The per-episode `AuditReport` keeps its (per-episode) fields but the summary is now the
authoritative dataset-level source — amends step-5b's ad-hoc no-code check into first-class code.
**Tested.** A5.7 step command `pytest tests/test_audit.py` 7→**12** (+5: slice aggregation reproduces
all-classes-union/scene-range/splits/summed-counts; not-ok when a class missing; empty-slice; `_infer_splits`;
`--slice --summary` CLI on a 2-episode temp slice). Full pure sweep `make test` 162→**167**; `make
import-smoke`/`lint`(ruff)/`typecheck`(mypy, 6 files) clean; `make audit` fixture clean; blob-guard OK.
**Open / next — STOP CHECK (real-slice is USER-GATED + next pure step A5.9 has a design decision).**
Emitting the A5.7 real-slice command block (`--slice data/aerialvln_json/train.slice.json --summary -`;
expect 50/50 ok, all 8 classes, scene range, 0 Δ-mismatch) — A5.7 flips to `done` when the user pastes it.
**A5.9** (`TeacherOutput`/`OracleTarget` seam + finalize Config placeholders) is now UNBLOCKED by A5.8 but
needs a user call on the **6-DoF→4-DoF projection** (teacher emits 6-DoF abs SE(3); student is 4-DoF body
delta) + the **weights-license** question — surfaced for decision, not executed.
**Vault.** No new decision (implements the M3 AuditSummary per the signed-off re-plan).

---

## 2026-06-09 — A5.8: WorldVLN scoping investigation RESOLVED (USER-verified; gates A5.9/A5.11/A5.14)
**Status:** A5.8 pending → done (USER-GATED; user pasted the `EmbodiedCity/WorldVLN` probe output — weights
listing + code-repo greps). The investigation's four unknowns are now answered; this UN-blocks A5.9.
**Findings (4).** (1) **Weights complete + sized:** ~36.9 GB / 11 files — InfinityStar backbone as a 4-shard
safetensors (~35 GB) + `WorldVLN_action_decoder.pt` (1.06 GB) + VAE (0.74 GB). (2) **Inference is STOCHASTIC
by default** — `infer/server.py`: `INFINITY_TOP_K=900`, `TOP_P=0.97`, `CFG=34`, two-phase late_top_k/p,
`lock_seed_across_steps=False` (per-segment `local_seed = seed + segment_index`), backbone
`sample_with_top_k_top_p_…(g_seed,…)`. **This overturns the prior "deterministic inference" assumption** —
K-rollout disagreement is FREE (vary `g_seed`); we do NOT need to engineer MC-dropout / re-enable sampling.
(3) **Action head is 6-DoF, not 4-DoF** — `latent_traj_manifest.py` `delta=np.zeros((T,6))`; `predict_pose.py`
emits `(T,6)` absolute `[roll,yaw,pitch,x,y,z]` integrated via SE(3) (`integrate_trajectory_se3`, ZYX Euler).
(`num_heads=6` are attention heads, NOT DoF.) Our `vllatent` student is 4-DoF `(Δx,Δy,Δz,Δψ)`, roll≡pitch≡0
⇒ the distillation needs an explicit **6→4 projection** (drop/verify-≈0 roll & pitch + absolute→body-delta)
at the teacher→student seam (A5.9). (4) **Checkpoints/encoder:** backbone env `INFINITY_CKPT`; action head env
`ACTIONHEAD_CKPT`/`ACTIONHEAD_REF_CKPT` (the guessed `STAGE2_LATENT2ACTION_CKPT` is the training-stage name,
artifact `stage2_latent2action_combined.pt`); two-stage train (stageA→stageB); language encoder = **T5**
(`T5EncoderModel`), NOT SigLIP/CLIP; backbone @ 81 frames / step 16 / fps 16. **License SPLIT:** GitHub code =
CC BY 4.0; HF weights frontmatter = `license: other` (undeclared) — the weights are what we distill from, so
the permissive read is NOT automatic → **flag for a clarification email before publishing on the student**.
**Decision #2 (disagreement source) RESOLVED.** try-in-order collapses to option (a): **WorldVLN native
stochastic rollouts** (free); AirScape (2507.08885) demoted from fallback to contingency only; V-JEPA-2 surprise
stays the independent second gate. The A5.3 Config placeholders (`disagreement_source`=`worldvln_rollout` is
already the default; `k_rollouts`; `vjepa_surprise_threshold`) are FINALIZED in A5.9.
**Next / open for A5.9 (USER input).** (i) the **6→4 projection** design at the OracleTarget seam (carry the
6-DoF teacher waypoint + project, and verify teacher roll/pitch≈0 on AerialVLN); (ii) the **weights-license**
question (email authors). Probe clone was scratch `/tmp/worldvln_code` (nothing committed). Re-probe the exact
`infer/server.py` rollout call signature when building the A5.11 teacher wrapper (TORCH/USER-GATED), not now.
**Vault.** Findings + the corrected "deterministic→stochastic" caveat recorded in `[[project-latent-pred-arch-locked]]`
+ MEMORY.md (user did this in-session).

---

## 2026-06-08 (PM) — A5.4–A5.6 adversarial-review hardening (0 blockers; 5 real non-blockers fixed)
**Status:** STOP-CHECK gate. A 5-dimension review→adversarial-verify pass (contract-fidelity /
validation-correctness / tier-purity+de-dup / DoD-completeness / test-adequacy) over the `2a89b68..ba0dc04`
diff returned **0 confirmed blockers** — every locked invariant + each step's DoD verified met. It also
surfaced 5 *real, mutation-confirmed* non-blocker gaps in the just-written code, all fixed here before push:
(1) `TrustReadout.sigma` used a one-sided `< 0` check → NaN/+inf slipped through (asymmetric with the
two-sided `k_star`/`p_commit`); now rejects non-finite (`np.isfinite`). (2–4) three `validate_manifest`
clauses had no test — deleting each (`dataset` missing-keys, `teacher` per-key completeness, `teacher.
disagreement_source` enum/`""`-stub) left all tests green; added negative tests for each + the `""` stub
accept. (5) the de-dup assertion `== PATCH_TOKENS` couldn't catch a `196/768` re-hardcode (since
`PATCH_TOKENS==196`); added a monkeypatch test that repoints the manifest-module constants to sentinels
and requires the manifest to follow (a literal would ignore the patch and fail).
**Tested.** Full pure sweep `make test` 155→**162** (+7); `make import-smoke` / `lint` (ruff) /
`typecheck` (mypy, 6 files) clean; manifest CLI round-trip OK; blob-guard OK. No step-status change
(A5.4–A5.6 stay `done`); the sigma change is the only production edit (a not-yet-wired seam, defensive).
**Vault.** No new decision (review-driven hardening of the signed-off A5.4–A5.6 surface).

---

## 2026-06-08 (PM) — A5.6: StepSample history + language padding masks (M4) — STOP CHECK
**Status:** A5.6 pending → done (AUTONOMOUS, pure-tier). **started_step+3 STOP CHECK reached (A5.4–A5.6).**
**What's done.** Closed M4 by making the two variable-validity inputs of the loader tuple explicit before
the loader (A5.15) is written: `StepSample` gains `history_mask (H=3,) bool` (True = real history frame,
False = zero-padding at the block-causal episode start — the old "padded+masked" comment is now a real
field) and `lang_mask (M,) bool` (True = real language token, False = padding so attention ignores the
pad). Added `MASK_DTYPE = np.bool_`; `__post_init__` validates both via `_check_array` and cross-checks
`len(lang_mask) == M == lang_tokens.shape[0]`. Field order keeps each mask beside the array it covers
(`history_latents, history_mask` / `lang_tokens, lang_mask`); the loader tuple in `docs/io-contract.md` §2
is now the 9-tuple `(z_t, history_latents, history_mask, lang_tokens, lang_mask, action_id, z_next,
delta_4dof, future_frame_rgb)` with both mask rows documented.
**Tested.** A5.6 step command `pytest tests/test_schemas.py` (50) green; full pure sweep `make test`
150→**155** (+5: masks-are-real-fields + 4 bad-input rejections — wrong dtype/length, lang length≠M); the
existing `_step_sample` helper updated to supply valid masks; `make import-smoke` / `lint` (ruff) /
`typecheck` (mypy, 6 files) clean; manifest CLI round-trip OK; `make audit` fixture clean (0 Δ-mismatch);
blob-guard OK.
**Open / next — STOP CHECK.** Pushing A5.4–A5.6 to `origin/main` + emitting the loop promise + the
USER-GATED **A5.8** command block (WorldVLN determinism/weights/4-vs-6-DoF/license probe — parallel,
gates A5.9/A5.11/A5.14). Resume after the push at **A5.7** (`AuditSummary` slice aggregator, M3 — PURE
code AUTO, real-slice re-run USER-GATED) → then A5.8-dependent A5.9.
**Vault.** No new decision (closes the M4 mask gap per the signed-off re-plan).

---

## 2026-06-08 (PM) — A5.5: typed student output seams in schemas.py (H3)
**Status:** A5.5 pending → done (AUTONOMOUS, pure-tier).
**What's done.** Promoted the three model-output rows of the I/O contract from prose into frozen,
shape/dtype-validated dataclasses in `vllatent/schemas.py` (review H3 — so a `−trust` / swap-predictor
ablation is a config flag over typed seams, not code surgery in Phase B): (1) `PredictorOutput`
`predicted_latents (T=HORIZON, 196, 768) fp16` (the rollout ẑ_{t+1..t+T} in DINOv3 patch space, cache
dtype to match the `z_next` target); (2) `TrustReadout` = the deployed single-pass head readout
`{p_commit (T,) float ∈ [0,1], k_star float ∈ [0,T] (soft expected horizon Σ_j Π p_i), sigma float ≥ 0}`;
(3) `Waypoint` `delta_4dof (4,) f32` AirSim-NED body, yaw-only (predicted analogue of `StepSample.delta_4dof`;
the NED→FLU→ENU remap stays Phase D). All reuse the existing `_check_array` validator + `eq=False`
(array fields) + `frozen`; scalar `k_star`/`sigma` reject bool and out-of-range. `docs/io-contract.md`
§0 now references the typed seams + notes the teacher `OracleTarget` seam lands in A5.9. Added to `__all__`.
**Tested.** A5.5 step command `pytest tests/test_schemas.py` (45) green; full pure sweep `make test`
134→**150** (+16: PredictorOutput/TrustReadout/Waypoint valid + bad-input rejection); `make import-smoke`
/ `lint` (ruff) / `typecheck` (mypy, 6 files) clean; blob-guard OK.
**Open / next.** A5.6 — `StepSample` `history_mask` + language padding-mask (M4) = **started_step+3 STOP
CHECK** (push + pause). A5.8 (WorldVLN investigation) USER-GATED, parallel — command block at the STOP CHECK.
**Vault.** No new decision (types the locked student-output seams per the signed-off re-plan).

---

## 2026-06-08 (PM) — A5.4: typed manifest builder fed from Config (M5)
**Status:** A5.4 pending → done (AUTONOMOUS, pure-tier).
**What's done.** Replaced the hand-built `empty_manifest()` (which hardcoded the encoder id, 196/768,
the cache version, and the conventions) with a typed `build_manifest(config: Config, *, split, variant,
entries)` that reads everything from the single source of truth: encoder `model_id`/`dtype` +
`dataset.name`/`license` + `cache.{version,quaternion_order,color_order,frame}` from `Config`, and the
fixed DINOv3 shapes `patch_tokens`/`dim` from `schemas.PATCH_TOKENS`/`EMBED_DIM` (kills the third copy of
196/768; M5 de-dup). `empty_manifest()` now just delegates to `build_manifest(Config())`; `CACHE_VERSION`
is the derived `CacheConfig().version` alias (one literal). Added the **stubbed `teacher` provenance
section** for the distillation pivot — `worldvln_model_id`/`worldvln_revision`/`vjepa2_model_id`/
`render_config_hash` are empty stubs (populated at cache build in A5.14), `disagreement_source` is read
from `Config.trust` now (finalized A5.9); validator accepts `""` as the stub value. Added `name`/`license`
to `DataConfig`. The per-entry required keys are now derived from `CacheManifestEntry.required_keys()`
(the no-default fields) so the validator is **type-enforced, not hand-kept** in sync; `validate_manifest`
gained `dataset`/`teacher` section checks. `manifest.py` is still pure tier (now imports `config`+`schemas`,
both numpy/pyyaml).
**Tested.** A5.4 step command `pytest tests/test_smoke.py tests/test_schemas.py` (38) green; full pure
sweep `make test` 127→**134** (7 new build_manifest/required_keys/teacher-stub tests); `make import-smoke`
/ `lint` (ruff) / `typecheck` (mypy, 6 files) clean; manifest CLI emit→validate round-trip OK; blob-guard OK.
**Open / next.** A5.5 — student output seams (`PredictorOutput`/`TrustReadout`/`Waypoint`) in `schemas.py`
(H3) → A5.6 (StepSample masks, M4) = **started_step+3 STOP CHECK** (push + pause). A5.8 (WorldVLN
investigation) stays USER-GATED, parallel — a command block is emitted at the STOP CHECK.
**Vault.** No new decision (implements the M5 typed-manifest de-dup per the signed-off re-plan).

---

## 2026-06-08 (PM) — A5.3: frozen typed Config single-source-of-truth (H1/H2/L2/L3) — STOP CHECK
**Status:** A5.3 pending → done (AUTONOMOUS, pure-tier). **started_step+3 STOP CHECK reached (A5.1–A5.3).**
**What's done.** Replaced the orphan untyped `load_config` dict with a frozen, validated `Config`
dataclass tree (`vllatent/config.py`): `EncoderConfig / PredictorConfig / DistillConfig / TrustConfig /
DataConfig / CacheConfig` + `Config.from_yaml` (env-`${VAR:-default}` expansion + **strict unknown-key
rejection** so an ablation-yaml typo fails fast). Dataclass defaults are the source of truth; the swept
ablation knobs (T/H, predictor depth/heads/mlp_ratio, distill λ-weights+temperature, trust
disagreement_source/k_rollouts/vjepa_surprise_threshold) live ONLY here; the spike-dependent trust knobs
are typed **placeholders finalized in A5.9**. Fixed shapes (196/768) stay `schemas` constants; AirVLN step
sizes stay `actions` constants — both referenced, not duplicated. De-duped HISTORY/HORIZON (now single
literals in `schemas`): `data/loader.py` reads H/T from `Config` (closes L2's local re-declaration);
trimmed `configs/default.yaml` to env overrides only (dropped the fixed-shape + `action:` duplication —
the M5 manifest de-dup lands in A5.4). Boundary validation (positive ints, heads∣EMBED_DIM, enum
disagreement_source, dtype∈{f16,f32}, thresholds∈[0,1]) + immutability. No resume/snapshot (Phase B).
**Tested.** `pytest tests/test_config.py` (18: defaults, from_yaml + env-expand, override, unknown
section/key rejection, validation, FrozenInstanceError) green; full pure sweep `make test` 109→**127**;
`make import-smoke` / `lint` (ruff) / `typecheck` (mypy, 6 files) clean; blob-guard OK; loader imports
config-driven (no torch needed).
**Open / next.** STOP CHECK (started_step+3): pushing + emitting the loop promise. Resume at **A5.4**
(typed manifest builder fed from Config, M5) → A5.5 (student seams, H3) → A5.6 (StepSample masks, M4) →
A5.7 (AuditSummary, M3). **A5.8 (WorldVLN investigation) is USER-GATED** and may run in parallel.
**Vault.** No new decision (implements the H1/H2 typed config SoT per the signed-off re-plan).

---

## 2026-06-08 (PM) — A5.2: no-flip test + NED→FLU→ENU remap math (M2, hard CI gate)
**Status:** A5.2 pending → done (AUTONOMOUS, pure-tier).
**What's done.** Added the body/world remap math to `frames.py` (re-derived vs fly0 `geometry/frames.py`
SEMANTICS; fly0 NOT imported): `R_FLU_FROM_FRD` (FRD body→FLU body) + `R_ENU_FROM_NED` (NED world→ENU
world) — both PROPER rotations (det=+1, no handedness flip) — and `ned_frd_to_flu` / `ned_to_enu` /
`remap_waypoint_ned_body_to_flu` (the 4-DoF seam-(d) stage-1 body remap = `(dx,-dy,-dz,-dyaw)`). Landed
`tests/test_frames.py` (7 tests) as the **hard CI gate** for the #1 foot-gun: no-flip basis (up→up,
down→down, forward→forward, right→right-of-forward) for both body and world frames, proper-rotation
check, waypoint remap sign/involution/magnitude, and an action-semantics survival check (GO_UP stays
+up, MOVE_FORWARD stays +forward) tied to `actions.action_to_delta`. The live closed-loop world-ENU
`WaypointHandoff` (needs odom) stays **Phase D**.
**Tested.** `pytest tests/test_frames.py` (7) green; full pure sweep `make test` 102→**109** (confirms
test_frames is collected by the CI gate); `make lint` (ruff) / `typecheck` (mypy, 6 files) clean.
**Open / next.** A5.3 — frozen typed `Config` + `from_yaml` + validation (H1/H2/L2/L3), the swept-knob
single source-of-truth. **A5.3 is the started_step+3 STOP CHECK** for this loop batch.
**Vault.** No new decision (implements the locked remap math + the M2 guardrail per the re-plan).

---

## 2026-06-08 (PM) — A5.1: extract public frame/quaternion primitives → frames.py (M1)
**Status:** A5.1 pending → done (AUTONOMOUS, pure-tier).
**What's done.** Moved the yaw/quaternion math out of `actions.py` into `frames.py` as the public,
single-owner API: `yaw_from_xyzw`, `xyzw_from_yaw`, `wrap_pi`, + new `reorder_wxyz_to_xyzw` (the
w-FIRST→xyzw foot-gun, previously inlined in `audit.parse_episode`). `actions.py` and `audit.py` now
import these from `frames.py` — **no private `_`-prefixed cross-module imports remain** (M1; the leak
5b had widened is closed). L1 verified: no `DOF+3` pun, `REFERENCE_PATH_ROW_WIDTH = 6` kept. Added the
`== yaw + pi/2` clarifying comment on `apply_delta`'s body-lateral branch (review micro-nit). Behavior
unchanged — math moved verbatim.
**Tested.** `pytest tests/test_actions.py tests/test_audit.py tests/test_smoke.py` (80) + full pure
sweep `make test` (102) green; `make import-smoke` / `lint` (ruff) / `typecheck` (mypy, 6 files clean);
`make audit` on the fixture OK (reorder_consistent=True, 0 Δ-mismatches); blob-guard OK; grep guards
confirm no private cross-module import + no stale `_yaw_from_xyzw`/`_xyzw_from_yaw`/`_wrap_pi`.
**Open / next.** A5.2 — `tests/test_frames.py` no-flip basis + NED→FLU→ENU remap math (M2, hard CI gate;
live fly0 wiring stays Phase D), then A5.3 (typed Config SoT) → STOP CHECK at started_step+3.
**Vault.** No new decision (refactor only; consolidates the #1-foot-gun owner per the re-plan).

---

## 2026-06-08 (PM) — post-pivot re-plan signed off; ralph loop restarting at A5.1
**Status:** Re-plan `plans/phase-a5-replan-postpivot.md` written + **USER-SIGNED-OFF**. Old steps 7–13
**superseded** by A5.1–A5.18. Ralph loop restarting at **A5.1** (pure-tier cheap-wins first).
**What changed.** (1) Backbone pivot — reuse **WorldVLN as a frozen teacher → distil into the student**
(= the latent-prediction transformer + waypoint + trust heads; **DINOv3 is the student's frozen cached
encoder, NOT the student**); oracle = WorldVLN rollout-disagreement + V-JEPA-2 surprise. (2) Code-review
must-fix set absorbed: H1/H2 (typed Config SoT)→A5.3, H3 (typed seams)→A5.5(+A5.9), M1→A5.1, M2→A5.2,
M3→A5.7, M4→A5.6, M5→A5.4(+A5.14). **L1 already resolved** by the 5b Euler fix (`REFERENCE_PATH_ROW_WIDTH=6`)
— the brief's "POSE_ROW=7" was stale.
**3 decisions locked (user).** H3 = student seams now, teacher seam after the A5.8 investigation;
disagreement = try-in-order (re-enable WorldVLN AR sampling → else AirScape native multi-seed; V-JEPA-2
the independent gate); Config = full swept set now with placeholders finalized in A5.9.
**Survives untouched.** `actions.py`, `audit.py` core + real-slice audit, frame conventions, the data
slice (1–6+5b), scaffold/CI, the 4-DoF→fly0-ENU output seam (Phase D).
**Next.** Loop runs A5.1→A5.3 then STOP CHECK (`started_step + 3`); A5.8 is USER-GATED and may run in
parallel. Planning brief: `plans/planning-prompt-2026-06-08-PM-replan-postpivot.md`. Superseded review:
`plans/planning-prompt-2026-06-08-refactor-before-phaseB.md`.

---

## 2026-06-08 — code-review pivot: STOP before step 7; refactor-before-Phase-B planned
**Status:** Phase-A pure/data lane (steps 1–6 + 5b) DONE & green. Forward progress (steps 7–13) PAUSED
pending a plan adjustment driven by a code review. Ralph loop stopped (`.claude/ralph-loop.local.md`
removed).
**Why.** A code review of the pure lane flagged structural issues to fix BEFORE Phase B (the ablation
control surface is forming wrong): **H1/H2** no single typed source-of-truth for the swept knobs
(T/H/depth/K) + untyped/mutable/decorative config (`load_config` orphan); **H3** the output seams
(PredictorOutput / TrustReadout / Waypoint) are prose-only; **M1** quaternion primitives misplaced in
`actions.py` + private cross-module imports (worse after 5b); **M2** `test_frames.py` no-flip-vs-fly0
missing (the #1 foot-gun unmitigated); **M3** audit per-slice checks mis-scoped per-episode; **M4**
`StepSample` lacks history/language masks; **M5** manifest stringly-typed + duplicated constants.
**Artifacts (committed).** `plans/planning-prompt-2026-06-08-refactor-before-phaseB.md` (full review +
planning-agent brief) + `plans/handoff-2026-06-08-refactor-session.md` (cold-start operator brief).
**Next session.** Run the planning agent → `plans/phase-a5-refactor-before-phaseB.md` → user sign-off →
ralph-execute (config single-source-of-truth first; items 1–3 are pure-tier, ~zero callers, cheapest now).

---

## 2026-06-08 — steps 6 + 5b: real AerialVLN slice + audit (schema corrected from real data)
**Status:** step 6 → done; step 5b → done. Step 5's `reference_path` assumption CORRECTED from real data.
**Real-data finding (the load-bearing correction).** The scaffold/plan/fixtures assumed
`reference_path` rows were 7-wide quaternions `[x,y,z,qx,qy,qz,qw]`. The **real AerialVLN** rows are
**6-wide EULER `[x,y,z,pitch,roll,yaw]` (radians)** — pitch=roll≡0 (4-DoF), **yaw = row[5]**; only
`start_rotation` is a quaternion (w-FIRST). Also **`len(reference_path) == len(actions)`** (NOT +1):
`reference_path[0]` is the start pose, `actions[t]` drives `ref[t]→ref[t+1]`, terminal STOP has no
stored next pose. Validated `vllatent.actions` against raw data: **0 mismatches across 39,133
transitions / 200 episodes** (all motion classes) — the action arithmetic was already correct; only the
pose *format* assumption was wrong.
**What's done.** Corrected `frames.py` (drop the bogus `reference_path=xyzw` constant → add
`REFERENCE_PATH_ORIENTATION/ROW_WIDTH/YAW_INDEX`), `schemas.py` (`EpisodeRecord.reference_path` (P,6)),
`audit.py` (Euler `_euler_row_to_pose`, yaw=row[5] quaternion verdict, alignment `len(ref)==len(actions)`
+ start-pose anchor, tuple width 6), `docs/io-contract.md` (foot-gun #1 rewritten: quaternion-vs-Euler),
both fixtures regenerated to the real Euler layout, tests updated (smoke/schemas/audit). Finished
`scripts/fetch_aerialvln_json.sh` (Kaggle/Baidu source doc + local slice-of-N writer for the real
`{"episodes":[...]}`); sliced `train.slice.json` (50/16386).
**Tested (5b on real slice).** `python -m vllatent.audit` over `train.slice.json`: **50/50 ok,
~10,198 transitions, 0 Δ-mismatches, all 8 action classes, 50/50 quaternion reorder-consistent,
34/50 episodes would corrupt yaw without the reorder**; 14 distinct scene_ids. Full pure sweep green
(import-smoke / lint / typecheck / 102 tests / `make audit` / blob-guard). License **CC BY-NC-SA 4.0**
recorded. `data/` gitignored — no JSON committed.
**Open / next.** Step 7 — DINOv3 encoder wrapper (`vllatent/encode/dinov3.py`): contract test is
AUTONOMOUS (monkeypatched backbone, BGR→RGB boundary, `(196,768) fp16`); real-weight smoke is dev-gated.
Then steps 8 (render unit) / 9 (cache manifest) autonomous halves, 10 (loader), 12 (sizing).
**Vault.** File the `reference_path = Euler (not quaternion)` + `len(ref)==len(actions)` finding under
`latent-pred-pipeline/` (corrects the Phase-A data-audit assumption).

---

## 2026-06-08 — step 5: AerialVLN-JSON audit parser (fixture half) + STOP at step 6
**Status:** pending → done (AUTONOMOUS). **Pure lane 2→5 GREEN; loop STOPS at step 6 (USER-GATED).**
**What's done.** `vllatent/audit.py` (pure numpy/stdlib): `parse_episode(dict)→EpisodeRecord` (reorders
`start_rotation` w-FIRST → canonical xyzw — foot-gun #1, schema confirmed against AirVLN `env.py`:
`Quaternionr(x=sr[1],y=sr[2],z=sr[3],w=sr[0])`, `instruction.instruction_text`, `goals[].position`,
`reference_path`=[x,y,z,qx,qy,qz,qw]); `audit_episode(dict)→AuditReport` with `QuaternionVerdict`
(reorder_consistent + **naive_would_mismatch** = flags the would-be wrong yaw if reorder skipped),
`actions[t]↔reference_path[t]→[t+1]` alignment, derived-Δ vs `actions.action_to_delta` (tol 1e-3),
per-action counts, tuple completeness, scene_id range, license. CLI `python -m vllatent.audit
--episode <json> [--report -]`. Two committed fixtures generated by STEPPING `apply_delta`:
`tiny_episode.json` (all 8 action classes, 9 poses, clean) + `quaternion_trap.json` (start yaw 90°;
naive xyzw read mislabels it as 0° → audit flags it).
**Tested.** `pytest -q tests/test_audit.py` → 7 passed; `make audit` exit 0 (tiny clean). Full pure
sweep green: import-smoke / lint / typecheck / `pytest -m "not torch and not sim"` (102 passed) /
`make audit` / blob-guard (fixtures are tiny text, allowed).
**Open / next — STOP CHECK (next step is USER-GATED).** Step 6 = fetch a real AerialVLN slice from S3
(`aerialvln.s3.ap-southeast-2.amazonaws.com/dataset/aerialvln/`). `scripts/fetch_aerialvln_json.sh` is
still a STUB (`exit 2`) — finishing it needs the real split-JSON layout + the user's S3/CN-network
situation, which step 6 surfaces. Loop paused; promise `PHASE A PURE LANE GREEN` emitted; `.claude/
ralph-loop.local.md` removed (deterministic stop). Hand the download block to the user; do NOT
auto-mark step 6 done.
**Vault.** No new decision (audit implements the locked I/O contract + foot-gun #1).

---

## 2026-06-08 — step 4: discrete→continuous-4-DoF action mapping
**Status:** pending → done (AUTONOMOUS).
**What's done.** `vllatent/actions.py` (pure numpy, NO airsim): `Action(IntEnum)` + step constants
transcribed VERBATIM from `third_party/AirVLN/airsim_plugin/airsim_settings.py` (STOP=0…MOVE_RIGHT=7;
FORWARD/LEFT_RIGHT=5, UP_DOWN=2, TURN=15). `action_to_delta(id)→(4,) f32` = canonical body-frame
`(dx,dy,dz,dyaw_deg)`, NED z-down (GO_UP=−z), body-right=+y, lateral=±5. `apply_delta(pose,id)`
reproduces `env_utils.getPoseAfterMakeAction` EXACTLY — incl. the AirSim quaternion↔euler formulas
reproduced in-module (`to_eularian_angles` yaw, `to_quaternion(0,0,yaw)`), pitch/roll forced 0, the
yaw-wrap at ±180, forward `unit_z==0`, and the `(yaw+90°)` body-lateral with LEFT×(−1).
`pose_pair_to_body_delta(a,b)` = the inverse the step-5 audit will use to verify dataset poses vs
quantized deltas.
**Tested.** `pytest -q tests/test_actions.py` → 64 passed: enum/constants, per-action deltas,
`apply_delta` at known starts (forward planar + yaw-following, lateral sign, z up/down, ±15° turn, STOP
identity), and a 6-yaw × 8-action round-trip `apply_delta→derive == action_to_delta` (pre-validates the
step-5 audit). Full pure sweep green: import-smoke / lint / typecheck / `pytest -m "not torch and not
sim"` (95 passed) / blob-guard.
**Open / next.** Step 5 — AerialVLN-JSON audit parser (`vllatent/audit.py`) + tiny_episode &
quaternion_trap fixtures + test_audit; then `make audit` clean. After step 5 → step 6 (S3 download,
USER-GATED) = STOP CHECK.
**Vault.** No new decision (faithful reproduction of the AirVLN ground-truth action arithmetic).

---

## 2026-06-08 — step 3: pure-tier tuple schemas
**Status:** pending → done (AUTONOMOUS).
**What's done.** `vllatent/schemas.py` (numpy + stdlib only, no torch): three frozen dataclasses with
boundary validation — (1) `StepSample` = the loader tuple `(z_t, history_latents, lang_tokens,
action_id, z_next, delta_4dof, future_frame_rgb)` with the locked shapes/dtypes pinned as module
constants (PATCH_TOKENS=196, EMBED_DIM=768, HISTORY=3, HORIZON=4, N_ACTIONS=8, DOF=4; latents fp16,
delta f32, rgb uint8); (2) `EpisodeRecord` = parsed AerialVLN episode (quaternions canonical xyzw,
actions int-aligned with reference_path); (3) `CacheManifestEntry` with `to_dict`/`from_dict` whose
keys satisfy `vllatent.manifest.validate_manifest`. Array records use `eq=False` (numpy `__eq__` is an
array) but stay `frozen`. `__post_init__` raises TypeError/ValueError with specific messages on a
contract breach.
**Tested.** `pytest -q tests/test_schemas.py` → 22 passed (shapes/dtypes, immutability, bad-input
rejection, manifest-entry JSON round-trip + cross-check against the manifest validator). Full pure
sweep green: import-smoke / lint / typecheck / `pytest -m "not torch and not sim"` (31 passed) /
blob-guard.
**Open / next.** Step 4 — discrete→continuous-4-DoF action mapping (`vllatent/actions.py` +
`tests/test_actions.py`), transcribing AirVLN constants + reproducing `env_utils.getPoseAfterMakeAction`.
**Vault.** No new decision (schemas implement the locked I/O contract).

---

## 2026-06-08 — step 2: transcribe I/O contract → docs/io-contract.md
**Status:** pending → done (AUTONOMOUS).
**What's done.** Wrote `docs/io-contract.md` — a *transcription* (not a re-derivation) of the LOCKED
I/O contract from vault `[[arch-design-2026-06-08-latent-pred]]`: the §4 tensor I/O table; the four
seams — (a) action repr = discrete codebook 0–7 → per-step FiLM, with the verbatim AirVLN enum
(STOP=0…MOVE_RIGHT=7) + step constants (FORWARD/LEFT_RIGHT=5, UP_DOWN=2, TURN/TILT=15); (b) language =
frozen SigLIP/CLIP text tower 512→768 → cross-attention; (c) uncertainty = deployed single-pass
horizon head, with K=5 ensemble + V-JEPA-2 marked Phase C (documented, not built); (d) waypoint→EGO =
continuous 4-DoF (Δx,Δy,Δz,Δψ) AirSim-NED body + the NED→FLU→ENU remap marked **NOT executed in
Phase A**. Pinned the loader output tuple (arch §6 item 5) + a "Frame & convention hazards" section
covering both foot-guns (quaternion order `w-FIRST` vs xyzw; `BGR`→RGB) + licenses (CC BY-NC-SA 4.0).
**Tested.** `test -f docs/io-contract.md && grep -q "NOT executed in Phase A" && grep -q "w-FIRST" &&
grep -q "BGR"` → PASS.
**Open / next.** Step 3 — pure-tier tuple schemas (`vllatent/schemas.py` + `tests/test_schemas.py`).
**Vault.** No new decision (pure transcription of the locked arch doc).

---

## 2026-06-08 — step 1: GitHub remote wired → DONE
**Status:** in_progress → done.
**What's done.** Created private GitHub repo `zhihao-acc/vllatent-ego-drone`; wired `origin`
(fetch+push = **direct github.com**, no mirror — direct connect works from this host). Pushed `main`
(`7ff793c`) after adding the `workflow` token scope (required to create `.github/workflows/ci.yml`).
`git ls-remote origin` resolves; `main` tracks `origin/main`.
**Tested.** Re-verified the full step-1 DoD today: codegraph_status healthy (19 files / 87 nodes / 87
edges, `.codegraph/codegraph.db` present + gitignored); `make import-smoke` / `lint` / `typecheck` /
`test` green (9 passed); `ALL=1 bash scripts/check_no_blobs.sh` OK.
**Open / next.** Step 1 complete. Ralph loop now closes steps 2→5 autonomously (io-contract → schemas →
actions → audit+fixtures), then STOPS at step 6 (S3 dataset download, **USER-GATED**).

---

## 2026-06-08 — step 1: scaffold + git + GitHub + codegraph
**Status:** pending → in_progress.
**What's done.** Scaffolded `/home/zh/CODE/vllatent-ego-drone` mirroring upipe, adapted for a training
repo with a **pure / torch / sim** tier split. Authored: `CLAUDE.md` (incl. the Wiki Knowledge Base
fetch-context section + locked-arch pointer + load-bearing invariants), `.claude/ralph-rules.md` (tier
gates + user-gated-step rule + deterministic stop), `DEV_LOG.md`, `README.md`, `.gitignore`,
`pyproject.toml` (package `vllatent`, py3.10, torch/sim extras), `Makefile`, `.github/workflows/ci.yml`
(torch-free pure-tier lane), `docs/TOPOLOGY.md`, `configs/{default,data_audit}.yaml`, `scripts/`
(check_no_blobs + ralph + 4 user-gated command-block stubs), `tests/{conftest,test_smoke}.py`, and the
`vllatent/` package: functional infra (`config.py`, `manifest.py`) + importable stubs
(`schemas/actions/frames/audit` pure tier; `encode/`, `data/`, `render/`, `cache.py` lazy torch/sim).
**Tested.** `make import-smoke`, `make lint`, `make typecheck`, `make test`, `ALL=1 bash scripts/check_no_blobs.sh`,
manifest round-trip (see commit). git init -b main + first commit.
**Open / next.** GitHub repo create + CN-mirror push is **USER-GATED** (command block provided); codegraph
`init`+`index` then verify with `codegraph_status`. Stays `in_progress` until the remote resolves +
codegraph verified. Then ralph closes steps 2→5 autonomously.
**Vault.** Will update `[[dev-decision-2026-07-latent-pred-pipeline]]` §8 to repo `zhihao-acc/vllatent-ego-drone`, package `vllatent`.
