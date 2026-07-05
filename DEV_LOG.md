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
| A5.7 — AuditSummary slice aggregator (M3) | done | 2026-06-09 | `AuditSummary` + `summarize_episodes` + `--slice/--summary/--split` CLI; dataset-level all-classes/scene-range/splits at SLICE scope (M3); 162→167 tests. **Real-slice VERIFIED (user-pasted):** 50/50 ok, 10198 transitions, 0 Δ-mismatch, all 8 classes, 14 scenes ∈[1,26], 34/50 naive-would-mismatch, splits=[train] — reproduces step-5b exactly |
| A5.8 — investigation: WorldVLN determinism/weights/6-DoF/license | done | 2026-06-09 | USER-verified probe of `EmbodiedCity/WorldVLN`: weights complete (~36.9 GB; InfinityStar 4-shard backbone + 1.06 GB action decoder + 0.74 GB VAE); inference **STOCHASTIC by default** (top_k900/top_p0.97/cfg34, per-segment seed) ⇒ K-rollout disagreement FREE (overturns prior "deterministic"); action head **6-DoF [roll,yaw,pitch,x,y,z]** SE(3)-integrated vs our 4-DoF student ⇒ 6→4 projection (A5.9); ckpt env `INFINITY_CKPT`+`ACTIONHEAD_CKPT`; lang enc T5; **LICENSE SPLIT** code CC BY 4.0 / weights `license:other` (flag pre-publication) |
| A5.9 — TeacherOutput/OracleTarget seam + finalize Config placeholders | done | 2026-06-09 | frozen+validated `TeacherOutput.rollouts_pose6 (K,6)` + `rollout_spread()→(6,)`; `OracleTarget {waypoint_4dof (4,) f32, teacher_pose6 (6,), rollpitch_resid, disagreement, vjepa_surprise}` (user-approved shape; finite/bool/dtype validated); `TEACHER_DOF=6`; TrustConfig placeholders FINALIZED (A5.8: worldvln_rollout, stochastic⇒free); io-contract §0 teacher-seam note; focused adversarial review CLEAN; 167→181 tests |
| A5.10 — DINOv3 student-encoder wrapper | done | 2026-06-09 | TORCH; contract (4) + real-weight **encode-smoke GREEN** `(196,768) fp16 cuda`, run live this session (user = operator). **Encoder swapped to timm's NON-GATED `vit_base_patch16_dinov3.lvd1689m`** (same LVD-1689M ViT-B/16 weights; Meta gated repo rejected access) — loader = `timm.create_model`+manual-normalize (pure-torch, no PIL); config model_id + manifest provenance + Makefile + test_config updated; new env `vllatent-ego-drone` (Py3.10). ⚠ `[torch]` extra pulled transformers 5.10/torch 2.12+cu130 (drift vs spec 4.56/2.8-cu12x — pin before A5.14) |
| A5.11 — frozen WorldVLN teacher wrapper | done | 2026-06-11 | client (`vllatent/teacher/worldvln.py`) + **live K-rollout smoke GREEN (user-pasted, H20)**: K=5×T=16 (segment 0), 5 DISTINCT rollouts (seeds 0/65537/…/262148), step-0 spread (6,) all >0 (0.027–0.091), `[teacher-smoke] OK`; health `infinity_loaded=true, points [1,17,33,49]` (`ts_ckpt_loaded` false until first call — stage2 loads lazily). **Wire correction locked:** `[dx,dy,dz,droll,dyaw,dpitch]` cm/deg position-FIRST deltas; K-rollout = K sessions × seed stride 65537. HF layout (actual): `$W/WorldVLN_backbone/{backbone(4-shard),vae}` + `WorldVLN_action_decoder.pt`; T5 separate. 13 contract tests, pure 199→212 |
| A5.12 — V-JEPA-2 surprise verifier wrapper | done | 2026-06-14 | TORCH; `vllatent/verify/vjepa2.py` — frozen V-JEPA-2 ViT-L; surprise `s_j=1−cos(ẑ_j,z_j)` per GT future frame → feeds `OracleTarget.vjepa_surprise`; lazy torch/transformers. **Weights NON-GATED** (`facebook/vjepa2-vitl-fpc64-256`, gated:false, MIT, ~1.30 GB — no DINOv3-style re-host); id single-sourced from `Config.trust.vjepa2_model_id` (+ `build_manifest` records it). 16 pure contract tests. **Real-weight smoke GREEN (user-pasted, cuda):** 587 tensors loaded, `surprise [0.174, 0.208]` (mean 0.191), finite ∈[0,2], `[vjepa-smoke] OK` |
| A5.13 — render harness | done | 2026-06-14 | SIM; teleport+capture, 3 foot-guns (xyzw quat / BGRA→RGB / Lock); 9 unit tests; `scripts/render_aerialvln.sh` real wrapper. **Live render GREEN (user-pasted, fly0-m1):** Connected, 8 RGB frames `(480,640,3)` from `tiny-0001`. **User's 3 smoke fixes (`7e31bf3`):** camera `front_0`→`front_center`, vehicle `Drone_1`→`drone_1`, arm+takeoff+200ms settle (fly0 pattern); script `PYTHONNOUSERSITE=1 -s` (avoid user-site Colosseum airsim shadow); `--vehicle` flag. ⚠ **frames are sim-native `(480,640)` NOT `224²` — A5.14 must square-crop+resize at the render→encode boundary (or set settings.json CaptureSettings=224²) to avoid DINOv3 aspect distortion** |
| A5.13b — frozen CLIP text tower → lang_tokens | done | 2026-06-14 | TORCH; `vllatent/encode/text.py` — frozen CLIP ViT-B/32, instruction→`(M,768)` fp16; native 512→768 zero-pad lift; lazy; id in `Config.encoder.text_model_id` + manifest provenance. **Weights NON-GATED** (`openai/clip-vit-base-patch32`, gated:false, 15M dls). 10 pure contract tests. **Real-weight smoke GREEN (user-pasted, cuda):** `lang_tokens (10,768) float16`, `[text-smoke] OK` (the `UNEXPECTED vision_model.*` keys are benign — CLIPTextModel ignoring the vision tower). Added 2026-06-14 to unblock A5.14's lang_tokens (no text-tower step existed) |
| A5.14 — render→[DINOv3+WorldVLN+V-JEPA-2]→cache + provenance manifest | done | 2026-06-15 | SIM+TORCH; orchestration + mocked test + **small-slice build VERIFIED (user-ran, 5 eps, K=5 WorldVLN rollouts, manifest OK)**. `[torch]` extra PINNED. 251 pure / 5 torch / lint / typecheck / blob green |
| A5.15 — distillation loader (StepSample+OracleTarget, masks, H/T from Config) | done | 2026-06-09 | numpy map-Dataset emits (StepSample,OracleTarget) over the render-once cache; block-causal H-window (H pinned to schemas HISTORY, fail-fast on divergent override), terminal-STOP excluded (len=Σ(N−1)); DEFINES the .npz cache read-contract A5.14 writes + `inspect` CLI (A5.16); torch-free emission (torch only at DataLoader collation); pure 182→190 + torch DataLoader test (4→5) |
| A5.16 — loader over real teacher/oracle dump | done | 2026-06-15 | USER-GATED: inspect over real 5-episode cache GREEN — 987 transitions (H=3), block-causal masks correct, all (StepSample,OracleTarget) tuples well-formed |
| A5.17 — size full render→teacher→cache job | done | 2026-06-15 | sizing doc + guard script (AUTO); build verified (6 episodes total: 5 from A5.14 + 1 incremental). Full 50-ep run deferred to Phase B start |
| A5.18 — Phase-A DoD verification | done | 2026-06-15 | all 3 DoD items verified; code review WARNING→4 HIGHs fixed (wrap_pi docstring, data_audit.yaml stale ref_path+camera+vehicle, np.load fd leak); 0 CRITICAL; **PHASE A COMPLETE** |
| B1.1 — Wire batch_undistort() into pipeline.py | done | 2026-06-19 | undistort wired between stage 2 and 3, gated by undistort_model + K/D |
| B1.2 — Fix MegaSaM confidence np.ones() fallback | done | 2026-06-19 | confidence_source field ("real"/"default"), warnings on fallback |
| B1.3 — Stub GPS Sim(3) alignment | done | 2026-06-19 | AlignmentResult type + NotImplementedError stub |
| B1.4 — Add fixed-clip-length cutting | done | 2026-06-19 | clip_length_seconds=10.0 in IngestConfig, cut_fixed_clips() in preprocess |
| B1.5 — Revise Config for sports pivot | done | 2026-06-19 | vjepa_only default, megasam_vo added, lambda_trust, sports.yaml fixed |
| B1.6 — Create SportsTarget in schemas.py | done | 2026-06-19 | `SportsTarget(waypoint_4dof)` + `Target` union alias; trust mechanism removed 2026-06-25 |
| B1.7 — YouTube pilot: curate + ingest | done | 2026-06-23 | USER-VERIFIED: 15 clips curated (`configs/sports_clips.yaml`), 11 accepted / 4 rejected (ski07/08/14/15), 38 FPV ranges, 173 sub-clips (10s). Filter: motion≥8 AND ¬YOLO(36 classes) AND segment≥10. `--filter-only` run green. `ingest_data/latent_cache/pilot_summary.json` validates. `verify_filter.py` on ski01 confirmed accepted/rejected split |
| B1.7a — Create vllatent/encode/batch.py | done | 2026-06-20 | `vllatent/encode/batch.py` — `encode_frames(frames_dir, device) → (N, 196, 768) fp16`; lazy torch; 5 tests green (mocked encoder, AST purity) |
| B1.7b — Content filter implementation | done | 2026-06-20 | **REVISED B1.7c**: CLIP dropped (0.999 within-domain, zero discrimination). Replaced with YOLO-World `yolov8s-worldv2` (74 FPS, 13M params, open-vocab). 36 rejected classes (drone body+parts, camera/gear, electronics, overlays). `filter_short_segments()` discards accepted runs < 10 frames (2s@5fps). `ultralytics>=8.2.0` added to `[torch]`. Filter: `is_fpv = motion≥8 AND ¬YOLO AND segment≥10`. 44 tests green; all imports lazy (AST-verified) |
| B1.7c — FPV segment extraction + pilot rework | done | 2026-06-20 | **REWORKED**: added `score_frames_from_paths()`, `detect_shot_boundaries_from_paths()`, `filter_video_from_paths()` — path-based memory-efficient filter scoring EVERY frame (no stride sampling). Pilot script uses `filter_video_from_paths()` directly on frame paths; stride variable removed; FPV ranges exact. 35 content filter tests green (8 new path-based) |
| B1.8 — CosFly-Track download + adapter | done | 2026-06-24 | Adapter code done. **DESCOPED:** RGB frames (119 GB) skipped — CARLA urban latents useless for skiing latent prediction (domain mismatch). Trajectory JSONs (~6 GB, `--meta-only`) provide GT deltas for L_wp only. 21 tests green |
| B1.9 — Data quality report script | done | 2026-06-19 | `scripts/data_quality_report.py` — JSON + terminal, 7 tests green |
| B1.9b — Per-clip HTML quality report | done | 2026-06-20 | `vllatent/ingest/visualize.py` + `scripts/clip_report.py` — Plotly offline HTML (5 sections: quality timeline, 3D trajectory, body deltas, VO confidence, latent coherence + summary); 15 tests green; all imports lazy |
| B1.10a — VO validation metrics module | done | 2026-06-24 | `vllatent/ingest/vo_validation.py` — PURE: smoothness (jerk, accel disc MAD-robust, angular spikes), physics plausibility (speed/yaw limits), confidence analysis, scale drift, GO/CONDITIONAL-GO/NO-GO verdict. 18 tests green |
| B1.10b — VO validation CLI + HTML report | done | 2026-06-24 | `scripts/validate_megasam.py` — single-clip + batch mode; Plotly HTML (3D trajectory, speed, yaw rate, confidence, accel); terminal verdict summary; JSON export. Ruff clean |
| B1.10d — Rework MegaSaM parser for real output | done | 2026-06-24 | Parser handles real format: `(T,7)` Lie group w2c → c2w inversion, `motion_prob.npy (T,H,W)` → per-frame confidence, `(T,4)` intrinsics → K matrix. Also `droid.npz` with `cam_c2w`. Legacy fallback kept. 37 tests green |
| B1.10e — MegaSaM 3-step automation script | done | 2026-06-24 | `scripts/run_megasam_pipeline.sh` — DepthAnything → UniDepth → camera_tracking; `run_megasam()` rewired to use it instead of nonexistent `run.py` |
| B1.10c — E2E pipeline test on one sub-clip | done | 2026-06-24 | **USER-VERIFIED GO** on `ski03_fpv00_c000` (50 frames). Full chain: content filter → FPV → subclip → quality gate → MegaSaM → DINOv3 → .npz. Bugs fixed: stale `model=` kwarg, conda `--no-banner`, per-frame fpv_mask leak, xformers sm_120 shims, socks→socks5 proxy |
| B1.10f — Fix shot boundary detection for consistent VO | done | 2026-06-25 | AdaptiveDetector threshold 3.0→2.0 catches all camera switches (ski03: 54→55 BEV cut was missed). Deleted edit_detection.py (hand-crafted heuristics failed twice — histogram+slowmo+block-patterns all false-positive on skiing). E2E stale-dir bug fixed (rmtree before copy). 462 tests green |
| B1.11 — Benchmark DINOv3 ViT-B/16 on Orin NX | done | 2026-06-26 | DEFAULT ASSUMPTION: ViT-B/16 fast enough. D=768 stays. Orin NX benchmark deferred — revisit B1.23 |
| B1.12 — Lock EMBED_DIM + PredictorConfig | done | 2026-06-26 | NO-OP: D=768 confirmed by default assumption. depth=6 (arch research). No code change needed |
| B1.13 — Sports sliding-window loader | done | 2026-06-26 | SportsTrainingDataset + preprocessing pipeline; 30 tests green |
| B1.14 — Collate function for batched training | done | 2026-06-26 | TrainingBatch NamedTuple + collate_sports_batch; 6 torch tests green |
| B1.15 — Block-causal ViT predictor + FiLM | done | 2026-06-26 | LatentPredictor depth=6 D=768 ~57M params; action+dt FiLM; 10 torch tests green |
| B1.16 — Waypoint head | done | 2026-06-26 | WaypointHead MLP D→256→128→4; no TrustHead (removed); 5 torch tests green |
| B1.17 — Full model assembly | done | 2026-06-26 | SportsFollowingModel = predictor + waypoint head; config-driven; 8 torch tests green |
| B1.18 — Loss functions: L_latent + L_wp | done | 2026-06-26 | L_latent smooth L1 beta=0.1 quality-weighted + L_wp confidence-weighted + cosine diag; 15 torch tests green |
| B1.19 — Checkpoint save/load + config snapshot | done | 2026-06-19 | `vllatent/train/checkpoint.py` — save/load + config snapshot + seed_everything; 10 torch tests green; lazy torch import (AST-verified) |
| B1.20 — Training script: overfit-tiny-batch | in_progress | 2026-06-26 | Script written (AUTO); USER-GATED: run on dev box, verify loss < baseline in 200 steps |
| B1.21 — Pre-train sanity check + viz | done | 2026-06-26 | run_sanity_check (7 pure tests) + TrainingLogger JSONL (6 torch tests) green |
| B1.22 — Full training run | superseded | 2026-06-29 | **REPLANNED → B1.21b + B1.22a–e (B-1 = latent only)**; head → B-2a; see plan Group 8 |
| B1.21b — Trust cleanup + remove verify/ | done | 2026-06-29 | dangling trust refs removed from schemas.py docstrings + CLAUDE.md L44 + plan scope rows; empty `vllatent/verify/` (only `__pycache__`) removed; docs-only, 465 pure green |
| B1.22a — train_sports.py upgrade (val/scene-split/warmup/bf16/--latent-only) | done | 2026-06-29 | --latent-only + evaluate()+persistence + split_clips_by_source + train-only NormStats→val + SequentialLR warmup→cosine + ckpt_best/early-stop + bf16(no-scaler) + AdamW param-groups + --no-action-film + --domain-weight(WeightedRandomSampler) + per-worker RNG + frozen TrainConfig(PURE); 478 pure / 64 torch / lint / mypy green; CPU latent-only smoke learns |
| B1.22c — Curate + ingest more REAL YouTube FPV (FRONT-LOADED) | done | 2026-06-29 | USER signaled local 45-candidate data generation/QC passed; source-level QC HTML generated for 44 clips (`cand01`–`cand13`, `cand15`–`cand45`; no `cand14` frames) |
| B1.22b — Generate full dataset ON DEV BOX → rsync .npz to H20 | done | 2026-06-30 | Local 919 `.npz` cache was rsynced/used for H20 B1.22e run; never git-add latents |
| B1.22d — [CONDITIONAL] Game footage → domain=game pretrain slice | superseded | 2026-07-05 | Do not activate game to chase B1 DINO-latent persistence; revisit only after B2 action-policy label/target gates |
| B1.22e — B-1 run: latent predictor (L_latent), DoD beats persistence | done | 2026-07-05 | Closed diagnostic-complete / model-incomplete; absolute, run2, and residual attempts did not produce an accepted latent-world checkpoint |
| B1.22f — Stage 2: waypoint head on frozen predictor | superseded | 2026-06-29 | **→ Phase B-2a** (deferred: MegaSaM scale + prober undecided) |
| B1.22g — Stage 3: conditional joint fine-tune | superseded | 2026-06-29 | **→ Phase B-2a** |
| B1.23 — Jetson inference speed check (encoder+predictor) | superseded | 2026-07-05 | No accepted B1 predictor checkpoint; B2 will benchmark encoder+action policy after B2b |
| B1.24 — Phase B-1 DoD verification (good latent model) | superseded | 2026-07-05 | Original B1 model DoD failed; B1 closed by decision and B2 activated |
| B2.0 — Close B1 and activate B2 Ralph loop | done | 2026-07-05 | Plan/rules/log/AGENTS updated; next active implementation step is B2.1 |
| B2.1 — Pure scale-free action target contract | done | 2026-07-05 | `vllatent/scale_free_targets.py`: locked `[unit_dir_x, unit_dir_y, unit_dir_z, log_speed_ratio]` target, pure numpy, scale-invariant, finite/masked degenerates, target-only API |
| B2.2 — Loader emits B2 action targets additively | done | 2026-07-05 | Sports loader keeps B1 fields and adds B2 scale-free target/input fields plus separate `ActionPolicyBatch`; past input path is causal/no-future |
| B2.3 — Direct scale-free action policy | done | 2026-07-05 | `ScaleFreeActionPolicy`: mean-pooled frozen DINO history/current + history mask + previous observed scale-free action + dt → future action sequence; no B1 predictor or future-label inputs |
| B2.4 — Action losses, metrics, baselines | done | 2026-07-05 | `action_policy_loss` plus direction/angular, path ADE/FDE, speed-ratio, aggregate score, deterministic baselines, and best-baseline margin |
| B2.5 — B2 trainer + local training-policy verification | blocked | 2026-07-05 | Trainer implemented/tested, but local source-split gate failed: best balanced smoke margin +1.99% (< required +10%); no H20 command |
| B2.6 — B2a diagnosis + B1-arch replan | done | 2026-07-05 | Direct-policy B2a is diagnostic only; next target is corrected scale-free supervision plus stronger B1/WAM checkpoint |
| B2.7 — Repair supervision/loss contract | pending | — | Fix speed-ratio outliers/reference-speed masking and align training loss with normalized path metrics |
| B2.8 — Past-only action/camera-history conditioning | pending | — | Add causal scale-free action/camera trajectory history inputs; future labels remain target-only |
| B2.9 — Re-run repaired direct-policy diagnostic | pending | — | Validate corrected signal locally; direct policy remains baseline/probe, not final H20 artifact |
| B2.10 — Control-relevant B1/WAM architecture | pending | — | Latent/world predictor + action head; accepted by action-margin improvement, not raw DINO cosine |
| B2.11 — Local B1-arch training-policy verification | pending | — | Must beat inertia and repaired direct-policy baseline locally before H20 |
| B2.12 — B1-arch H20 USER gate | pending | — | USER-GATED; provide one command only if B2.11 passes |
| B2.13 — H20 scale-free B1-arch WAM run | pending | — | USER-GATED; target artifact is stronger B1-architecture checkpoint |
| B2.14 — B2b readout + Jetson decision | pending | — | Readout before another paid run; Jetson only after useful checkpoint |

Statuses: `pending` / `in_progress` / `done` / `blocked` / `superseded`.

---

## 2026-07-05 — B2.6 diagnosis-driven replan to B1-architecture WAM

**Status:** B2.6 docs/rules replan is done. Next AUTO step is B2.7 supervision/loss repair.

**User correction.** The intended B2 DoD signal is not "a single direct policy model." The target
artifact is a stronger B1-architecture checkpoint: a decoupled latent/world predictor plus action
head, accepted by future-action quality and baseline margin.

**Diagnosis carried forward from B2.5.**
- The direct residual policy was useful as a probe, but failed the source-held-out gate: best
  balanced smoke margin `+1.99%` vs required `+10%`.
- Repeat-last inertia is the strongest baseline, so any accepted world model must add
  non-inertial control information beyond past motion.
- Current `log_speed_ratio` labels can explode when past observed reference speed is tiny, and the
  training loss uses unnormalized cumulative path vectors while evaluation normalizes path shape.
  This target/loss mismatch must be fixed before H20 or architecture-capacity escalation.

**Plan update.** B2.7 repairs scale-free supervision and loss alignment. B2.8 adds past-only
scale-free action/camera-history conditioning. B2.9 reruns the repaired direct policy only as a
diagnostic baseline. B2.10-B2.11 then build and locally verify the control-relevant B1/WAM
architecture. B2.12 is the next USER gate; no H20 command is active before then.

**Verified.**
- `rg -n "B2.7|stronger B1-architecture|loss/metric mismatch|past observed" plans/phase-b-sports-training.md .codex/ralph-rules.md DEV_LOG.md AGENTS.md`
  -> expected B2.6/B2.7/B1-arch hits.
- Stale active-gate scan for the old B2.6/H20 direct-policy phrases
  -> no stale active-gate hits.
- `git diff --check -- AGENTS.md .codex/ralph-rules.md DEV_LOG.md plans/phase-b-sports-training.md`
  -> pass.

---

## 2026-07-05 — B2.5 action trainer and local gate

**Status:** B2.5 trainer is implemented and tested, but the B2a local source-split gate failed.
Do not proceed to the old B2.6 H20 instructions from this state; this was superseded by the B2.6
diagnosis/replan entry above.

**Trainer added.** Added `scripts/train_sports_b2.py`, separate from the B1 latent trainer. It trains
only `ScaleFreeActionPolicy`, writes B2 config snapshots, checkpoints best-by-action-margin, logs
`train_action_metrics.jsonl` / `val_action_metrics.jsonl` / optional `source_action_metrics.jsonl`,
and supports local smoke limits including source-balanced `--max-clips-per-source`.

**Policy correction from local evidence.** The initial source smoke showed the unanchored policy had
to relearn the strongest dumb baseline. `ScaleFreeActionPolicy` is now residual around repeated
`last_action_scale_free`, with the residual head zero-initialized. This preserves the B2.3 input
contract and starts exactly at the deterministic repeat-last baseline.

**Local evidence.**
- Tiny overfit after residual anchor: model score `2.9037`, best baseline `repeat_last=3.0929`,
  margin `+0.1892` (`+6.12%`) over 16 samples.
- Source-split smoke, sorted 120-clip cap, residual depth-2/hidden-128: best margin `+0.0086`
  (`+0.59%`) over best baseline `repeat_last=1.4501`.
- Source-balanced smoke (`--max-clips-per-source 4`, 10 held-out sources): best aggregate model score
  `1.7638`, best baseline `repeat_last=1.7996`, margin `+0.0358` (`+1.99%`), below the required
  `+10%` B2a local gate. At best epoch, 7/10 held-out sources improved, but worst source margin was
  `-18.64%` and best source margin was `+9.01%`.

**Diagnosis.** The trainer can optimize finite losses and the residual policy can slightly improve
on inertia, but current B2 labels/model/training do not clear the local source-split action baseline.
Repeat-last/no-turn remains too strong for the direct visual residual policy in this smoke. Per B2.5,
stop before H20 and replan/diagnose locally rather than spending a B2b run.

**Verified.**
- `/home/zh/miniconda3/envs/vllatent-ego-drone/bin/python -m pytest -q tests/test_train_sports_b2.py tests/test_action_policy.py tests/test_action_metrics.py`
  -> 23 passed.
- Local CPU overfit: `scripts/train_sports_b2.py --overfit-tiny ... --max-steps 50` -> positive
  tiny margin `+6.12%`.
- Local CPU source-balanced smoke: `scripts/train_sports_b2.py ... --max-clips-per-source 4`
  -> best margin `+1.99%`, gate failed.

---

## 2026-07-05 — B2.4 action losses, metrics, and baselines

**Status:** B2.4 is done. Next AUTO step is B2.5 B2 trainer and local training-policy verification.

**Metrics added.** Added `vllatent/train/action_metrics.py` for the locked B2 action vector
`[unit_dir_x, unit_dir_y, unit_dir_z, log_speed_ratio]`. Metrics include direction cosine,
angular error, normalized path-shape ADE/FDE, speed-ratio MAE, and lower-is-better aggregate score.
No yaw-rate metric is included because yaw is not part of the B2.1 contract.

**Baselines and margins.** Added deterministic `repeat_last`, `no_turn`, `zero`, `mean`, and
`linear` baselines plus `score_action_predictions()`, where margin is positive only when the model
beats the best baseline.

**Loss added.** Added masked differentiable `action_policy_loss()` to `vllatent/train/losses.py`,
combining direction, speed-ratio, and path losses with optional sample weighting.

**Verified.**
- `/home/zh/miniconda3/envs/vllatent-ego-drone/bin/python -m pytest -q tests/test_action_metrics.py tests/test_losses.py`
  -> 27 passed.
- `/home/zh/miniconda3/envs/vllatent-ego-drone/bin/ruff check vllatent/train/action_metrics.py vllatent/train/losses.py tests/test_action_metrics.py tests/test_losses.py`
  -> all checks passed.
- `/home/zh/miniconda3/envs/vllatent-ego-drone/bin/python -m py_compile vllatent/train/action_metrics.py vllatent/train/losses.py tests/test_action_metrics.py tests/test_losses.py`
  -> pass.

---

## 2026-07-05 — B2.3 direct scale-free action policy

**Status:** B2.3 is done. Next AUTO step is B2.4 action losses, baselines, and metrics.

**Policy added.** Added `vllatent/model/action_policy.py` with `ScaleFreeActionPolicy`. It mean-pools
cached DINO patch tokens per frame, encodes history/current frame context with a small Transformer,
then combines context with previous observed `last_action_scale_free`, per-horizon `dt_seconds`, and
horizon embeddings to predict `(B, T, SCALE_FREE_ACTION_DIM)`.

**No leakage / no B1 dependency.** The policy forward signature accepts only `history_latents`,
`z_t`, `history_mask`, `last_action_scale_free`, and `dt_seconds`. Tests assert it has no future
target/label or `odom_reference_speed` argument and does not import/call the B1 `LatentPredictor`.

**Verified.**
- `/home/zh/miniconda3/envs/vllatent-ego-drone/bin/python -m pytest -q tests/test_action_policy.py tests/test_heads.py`
  -> 15 passed.
- `/home/zh/miniconda3/envs/vllatent-ego-drone/bin/ruff check vllatent/model/action_policy.py tests/test_action_policy.py`
  -> all checks passed.
- `/home/zh/miniconda3/envs/vllatent-ego-drone/bin/python -m py_compile vllatent/model/action_policy.py tests/test_action_policy.py`
  -> pass.

---

## 2026-07-05 — B2.2 additive loader/collate scale-free action fields

**Status:** B2.2 is done. Next AUTO step is B2.3 direct scale-free action policy.

**Loader changes.** `SportsSample` now keeps existing B1 fields (`target_deltas`, `last_action`) and
additively emits B2 fields: `target_actions_scale_free`, `target_actions_moving_mask`,
`last_action_scale_free`, and `odom_reference_speed`. B2 future labels come from the pre-z-score
velocity-like target path. B2 previous-observed action inputs use a separate causal observed path
(physics clip + velocity, no centered median), so target/future deltas cannot affect B2 model inputs.

**Collate changes.** `TrainingBatch` and `collate_sports_batch()` remain B1-compatible and do not
expose the B2 target fields. Added separate `ActionPolicyBatch` and `collate_action_policy_batch()`
for B2 direct action-policy training.

**Leakage guard.** Added a targeted test that changes only future deltas for a sample. It changes
`target_actions_scale_free` while preserving `last_action_scale_free` and `odom_reference_speed`,
proving the B2 action-like input is past-only. Source split tests remain green.

**B2.1 regression fixed.** The scale-free target helper now returns an ndarray scalar mask for a
single-delta action, with a regression test. This was found by B2.2's `last_action_scale_free`
single-delta path.

**Verified.**
- `/home/zh/miniconda3/envs/vllatent-ego-drone/bin/python -m pytest -q tests/test_sports_loader.py tests/test_collate.py tests/test_scale_free_targets.py`
  -> 66 passed.
- `/home/zh/miniconda3/envs/vllatent-ego-drone/bin/python -m pytest -q tests/test_sports_loader.py tests/test_collate.py tests/test_scale_free_targets.py tests/test_train_split.py`
  -> 76 passed.
- `/home/zh/miniconda3/envs/vllatent-ego-drone/bin/ruff check vllatent/data/sports_loader.py vllatent/data/collate.py vllatent/scale_free_targets.py tests/test_sports_loader.py tests/test_collate.py tests/test_scale_free_targets.py`
  -> all checks passed.
- `/home/zh/miniconda3/envs/vllatent-ego-drone/bin/python -m py_compile vllatent/data/sports_loader.py vllatent/data/collate.py vllatent/scale_free_targets.py tests/test_sports_loader.py tests/test_collate.py tests/test_scale_free_targets.py`
  -> pass.

---

## 2026-07-05 — B2.1 pure scale-free action target contract

**Status:** B2.1 is done. Next AUTO step is B2.2 loader/collate additive B2 action fields.

**Contract locked.** Added `vllatent/scale_free_targets.py` as a PURE numpy module. The B2.1
per-horizon target vector is exactly
`[unit_dir_x, unit_dir_y, unit_dir_z, log_speed_ratio]`; yaw-rate is not part of this target
contract. `ScaleFreeActionTargets` returns only `actions` and `moving_mask`, so future action labels
are target-only and do not package model inputs.

**Scale handling.** `reference_speed_from_deltas()` computes arbitrary-scale speed references for
observed motion. `future_deltas_to_scale_free_targets()` can use an observed reference or an internal
label-only reference, and does not return that reference. Uniform positive rescaling of translation
deltas preserves `unit_dir_xyz` and `log_speed_ratio`. Zero/near-zero motion is finite and masked,
using a stable forward fallback direction.

**Metric speed remains outside Youtube labels.** `metric_speed_command_from_log_ratio()` is an
inference/controller helper only: it uses onboard odom reference speed and clamps commands to
`7.5 m/s - 1e-3`, strictly below the cap. The target-generation signature has no odom speed or speed
cap parameter.

**Verified.**
- `/home/zh/miniconda3/envs/vllatent-ego-drone/bin/python -m pytest -q tests/test_scale_free_targets.py`
  -> 18 passed.
- `/home/zh/miniconda3/envs/vllatent-ego-drone/bin/python -m py_compile vllatent/scale_free_targets.py tests/test_scale_free_targets.py`
  -> pass.
- `/home/zh/miniconda3/envs/vllatent-ego-drone/bin/ruff check vllatent/scale_free_targets.py tests/test_scale_free_targets.py`
  -> all checks passed.

---

## 2026-07-05 — B1 closed; B2 scale-free future-action plan activated

**Status:** B1.22e is closed as **diagnostic-complete / model-incomplete**. B1.23/B1.24 are
superseded. B2.0 is done; B2.1 is the next implementation step. Older historical entries below
still mention B1.22e as `in_progress`; the status table and this entry supersede them.

**User decision.** The project should not keep spending H20 attempts on raw DINO future-latent
cosine. Future action sequence is the target, never an input. Youtube/MegaSaM scale is not trusted;
metric scale comes from onboard odometry during real drone inference. Commanded speed must be
clamped below `7.5 m/s`.

**Research-backed B2 direction.**
- ViNT supports direct future waypoint/action prediction from current/past observations and
  normalizes waypoints by robot top speed, with robot-specific controller unnormalization.
- NoMaD supports future action-sequence policies and later diffusion escalation for multimodal
  futures.
- FutureNav supports action prediction as the inference path, with inverse/forward/future-state
  objectives as auxiliary training support rather than the primary gate.
- Monocular VIO scale ambiguity supports the decision to treat Youtube/MegaSaM translation
  magnitude as scale-free shape supervision, not metric speed.

**Plan/rules changes.**
- `plans/phase-b-sports-training.md` now closes B1, supersedes B1.23/B1.24, and defines ordered
  B2.0-B2.9 diagnostic steps with tier, gate, DoD, tests, and dependencies. Later B2.6 replan
  entry above extends the active queue through B2.14.
- `AGENTS.md` now names B2 as the active repo guidance.
- `.codex/ralph-rules.md` now points to B2 and blocks further B1 latent H20 reruns.
- B2a DoD: local scale-free training-policy gate, including scale-invariance tests, no future-action
  leakage tests, overfit-tiny, and source-split margin over dumb baselines.
- B2b DoD: one user-approved H20 run beats the best scale-free baseline by at least 15%, improves
  on a majority of held-out sources, and saves checkpoint/config/metrics with no metric-flight claim.

**Next AUTO step:** B2.1 pure scale-free action target contract. Do not implement PI-Prober,
diffusion, language, game data, or auxiliary latent/world losses before the direct B2a gate.

---

## 2026-07-05 — B1.22e run2-based residual replan + AUTO implementation

**Status:** B1.22e remains **in_progress**. Do not proceed to B1.23/B1.24. The next step is a
USER-GATED H20 residual run, not another absolute-prediction depth/LR sweep.

**Run2 is now the active baseline.** User-pasted recovery run2 used depth 4, LR `1e-4`, bf16,
AdamW betas `(0.9,0.95)`, `--exclude-source ski03`, `--eval-train`, and `--eval-by-source`.
Best visible val was epoch 25 / step 8008: `val_cos=0.7592678`, persistence `0.8575975`,
margin `-0.0983296`; best visible train eval was epoch 27 / step 8624: `train_cos=0.8002520`,
persistence `0.8685168`, margin `-0.0682648`. Run2 artifacts are not present locally under
`runs/`, so these run2 facts are recorded from the user's paste-back, not re-parsed JSONL.

**Diagnosis/research gate.** Two focused research agents plus local metric review converged on the
same conclusion: because run2 misses persistence even on train, split variance, `ski03`, depth, and
LR are not the primary blocker. The failure is objective/parameterization mismatch: absolute
future-DINO prediction must relearn the large persistent static component before it can learn the
small motion residual that beats `z_t`. Paper-backed direction: JEPA/V-JEPA/DINO-WM/DINO-world
support frozen-feature latent world models; copy-last remains a valid video baseline; residual and
zero-init/identity-start methods support a persistence-residual parameterization.

**Plan revised first.** `plans/phase-b-sports-training.md` now treats run2 as the active baseline,
keeps the DoD unchanged, and defines the approved next AUTO direction:
`z_hat = z_t + delta_hat`, with zero/near-zero residual initialization so the untrained residual
model starts at approximately persistence.

**AUTO code changes.**
- `LatentPredictor` accepts `prediction_mode={"absolute","residual"}`. The default absolute path is
  unchanged. Residual mode adds a zero-initialized `residual_out` projection and returns
  `z_t + delta_hat`.
- `SportsFollowingModel` passes `prediction_mode` through to the predictor.
- `TrainConfig` and `scripts/train_sports.py` expose `--prediction-mode`,
  `--latent-loss-mode {absolute,delta,combined}`, and `--delta-loss-weight`.
- Latent-only training keeps absolute `SmoothL1(z_hat, z_future)` by default; delta and combined
  modes train against `z_future - z_t` as an optimization ablation while evaluation remains on
  reconstructed `z_hat`.
- `evaluate()` now logs `val_min_margin` so the worst horizon is visible alongside average margin.

**TDD/verification.**
- RED gate before production code:
  `/home/zh/miniconda3/envs/vllatent-ego-drone/bin/python -m pytest -q tests/test_predictor.py::TestLatentPredictor::test_residual_mode_zero_init_matches_persistence tests/test_predictor.py::TestLatentPredictor::test_rejects_unknown_prediction_mode tests/test_config.py::test_train_config_recovery_defaults tests/test_train_sports_residual.py`
  → 5 intended failures for missing `prediction_mode` / residual loss plumbing.
- GREEN targeted rerun of the same command → 5 passed.
- `/home/zh/miniconda3/envs/vllatent-ego-drone/bin/python -m pytest -q tests/test_config.py tests/test_evaluate.py tests/test_train_sports_residual.py`
  → 42 passed.
- `/home/zh/miniconda3/envs/vllatent-ego-drone/bin/python -m pytest -q tests/test_predictor.py tests/test_model.py tests/test_config.py tests/test_evaluate.py tests/test_train_sports_residual.py`
  → 64 passed.
- `/home/zh/miniconda3/envs/vllatent-ego-drone/bin/python -m py_compile scripts/train_sports.py vllatent/config.py vllatent/model/predictor.py vllatent/model/sports_model.py vllatent/train/evaluate.py`
  → pass.
- `/home/zh/miniconda3/envs/vllatent-ego-drone/bin/ruff check scripts/train_sports.py vllatent/config.py vllatent/model/predictor.py vllatent/model/sports_model.py vllatent/train/evaluate.py tests/test_predictor.py tests/test_config.py tests/test_evaluate.py tests/test_train_sports_residual.py`
  → all checks passed.

**Next USER gate.** Run the residual H20 command and paste: tail of `val_metrics.jsonl`, tail of
`train_eval_metrics.jsonl`, `source_metrics.jsonl`, steps/sec, GPU memory, and confirmation that
`ckpt_best.pt` + `norm_stats.npz` exist. If train margin is still negative, stop before game/data
scaling and run the delta-loss ablation.

---

## 2026-06-30 — B1.22e recovery replan + AUTO diagnostics

**Status:** B1.22e remains **in_progress**. Do not proceed to B1.23/B1.24. AUTO recovery work is
done; the next step is USER-GATED H20 training with the new diagnostic flags.

**Diagnosis locked into the plan.** The first H20 run failed primarily because the absolute
future-latent objective had to beat a very strong 5 Hz persistence baseline. The metric itself is
apples-to-apples, and `ski03` is not the aggregate root cause (35/1425 val windows). Train-batch
cosine reached only ~0.81 while cache-only train persistence is ~0.87, so the next run must report
full train margin before we spend more time on capacity/data hypotheses. The late collapse/NaN is a
separate stability defect, beginning around step 7510 and first logged as NaN at step 7840.

**Plan revised first.** `plans/phase-b-sports-training.md` now treats the original
depth-6/action-FiLM/LR2e-4 command as a failed baseline, keeps the DoD unchanged, and defines the
recovery decision rule: if train margin stays negative, stop and replan a persistence-residual
predictor (`z_hat = z_t + delta_hat`) before another full run; if train margin is positive but val
margin is negative, focus on split/data scale/generalization.

**AUTO code changes.**
- `TrainConfig` now records AdamW betas, defaults to `(0.9, 0.95)`, and defaults checkpoint
  selection to `early_stop_metric="val_margin"`.
- `train_sports.py` exposes `--adam-beta1/--adam-beta2`, `--exclude-source`, `--eval-train`, and
  `--eval-by-source`.
- Full runs can now exclude orphan/provenance-gap sources such as `ski03` without moving cache
  files; train-eval writes `train_eval_metrics.jsonl`; val source attribution writes
  `source_metrics.jsonl`.
- Optimizer steps now fail fast on non-finite loss or non-finite gradient norm before
  `optimizer.step()`, and `grad_norm` is logged in `train_metrics.jsonl`.
- `SportsTrainingDataset.sample_sources` tracks source id per sample for diagnostics.

**Verified.**
- `/home/zh/miniconda3/envs/vllatent-ego-drone/bin/python -m pytest -q tests/test_config.py tests/test_train_split.py tests/test_sports_loader.py tests/test_train_viz.py tests/test_evaluate.py`
  → 91 passed.
- `/home/zh/miniconda3/envs/vllatent-ego-drone/bin/python -m py_compile scripts/train_sports.py vllatent/config.py vllatent/data/sports_loader.py vllatent/train/viz.py`
  → pass.
- `/home/zh/miniconda3/envs/vllatent-ego-drone/bin/ruff check scripts/train_sports.py vllatent/config.py vllatent/data/sports_loader.py vllatent/train/viz.py tests/test_config.py tests/test_sports_loader.py tests/test_train_viz.py`
  → all checks passed.
- `/home/zh/miniconda3/envs/vllatent-ego-drone/bin/python -m mypy vllatent/schemas.py vllatent/actions.py vllatent/frames.py vllatent/config.py vllatent/manifest.py vllatent/audit.py vllatent/ingest/quality.py vllatent/ingest/ego_motion.py`
  → success, 8 files.
- CLI parse check confirms `--adam-beta*`, `--exclude-source`, `--eval-train`, `--eval-by-source`,
  and `--early-stop-metric {val_cos,val_margin}` are present.
- Direct `_step_optimizer` smoke over a tiny `torch.nn.Linear` returned a finite grad norm for a
  finite loss and raised `FloatingPointError` for a NaN loss before backward/step.

**Next USER gate.** Run the recovery command on H20 and paste: tail of
`val_metrics.jsonl`, tail of `train_eval_metrics.jsonl`, `source_metrics.jsonl`, steps/sec, GPU mem,
and confirmation that `ckpt_best.pt` + `norm_stats.npz` exist. If train margin is still negative,
do not continue with more absolute-prediction sweeps; replan residual prediction first.

---

## 2026-06-30 — B1.22e: H20 latent run completed, DoD NOT met

**Status:** B1.22b effectively **done** (H20 training consumed the 919-file local latent cache);
B1.22e remains **in_progress**. Do not mark B1.22e done or proceed to B1.23/B1.24.

**Artifacts.** Returned H20 run is local under
`runs/h20_b1_latent_lr2e-4_nan_20260630/` and remains ignored/off-git. It contains
`ckpt_best.pt` (651M), `norm_stats.npz`, `val_metrics.jsonl`, `train.log`, and config/log files.

**Run config.** `--latent-only`, depth=6, action-FiLM enabled, bf16, LR `2e-4`, WD `0.05`,
warmup `0.05`, batch 64, val-frac `0.2`, seed 42, early-stop metric `val_cos`, patience 8.
Train log reports 56,741,376 optimized params, split `834 train / 85 val clips`, and
`1425` val windows.

**Best finite val metric (epoch 16 / step 5457).**
- `val_cos=0.7317715287208557`
- `val_persistence=0.8094441294670105`
- `val_margin=-0.07767263054847717`
- per-horizon cos `[0.742104709148407, 0.7347822785377502, 0.7280759215354919, 0.7221232056617737]`
- per-horizon persistence `[0.8690751194953918, 0.819399356842041, 0.7866102457046509, 0.7626919150352478]`
- per-horizon margin `[-0.12697041034698486, -0.08461707830429077, -0.058534324169158936, -0.04056870937347412]`

**Interpretation.** Directional cosine passed the old high-variance `t=1 > 0.7` sanity threshold,
but the predictor did **not** beat persistence at any horizon, so B-1 DoD failed. The negative
margin is least bad at horizon 4, suggesting the persistence baseline is especially strong at
short horizons.

**Stability issue.** Epoch 23 collapsed to `val_cos=0.2157602012`; epoch 24 became `NaN`.
`train.log` shows `L_lat=nan` starting around epoch 24 / step 7840. `ckpt_best` predates the NaN,
but the numerics still need investigation before another paid H20 run.

**Next-session instruction.** `plans/handoff-2026-06-24-b1.10c-onwards.md` was rewritten as the
current handoff. The next session should first inspect data/split, metric/baseline, model size,
action-FiLM, and numerics; research/ponder the failure mode; discuss with the user; then revise
`plans/phase-b-sports-training.md` before executing any model/training changes. Do not blindly
rerun the same B1.22e command.

---

## 2026-06-29 — B1.22c/B1.22e: QC committed + local CUDA preflight before H20

**Status:** B1.22c local data/QC accepted by user → **done**; B1.22b/B1.22e → **in_progress**
(H20 rsync + training paste-back still user-gated). Commit `68d162b` added Codex-local `AGENTS.md`
and `.codex/ralph-rules.md`, persisted content-filter decisions for QC, added
`scripts/qc_report.py`/`qc_lib.py`, made MegaSaM output paths robust, and kept generated
`reports/qc/` ignored.

**QC output.** `python scripts/qc_report.py --clip cand01 ... cand45 --device cuda` generated
44 source-level HTML reports plus `reports/qc/index.html`. `cand14` was skipped because no
`ingest_data/frames/cand14` directory exists. Top-level index reported 44 clips; generated HTML
is ignored and not committed.

**CUDA note.** Sandboxed Codex commands hide `/dev/nvidia*`, so `torch.cuda.is_available()` is
false there. Unsandboxed check is healthy: RTX 5060 Ti visible, torch `2.12.0+cu130`, CUDA 13.0,
bf16 supported. Do not interpret sandbox CUDA=false as an env failure.

**Local B1.22e smoke (unsandboxed CUDA, RTX 5060 Ti).**
- Overfit-tiny, real cache, full depth=6, 56.7M optimized predictor params, fp32, batch 4,
  8 samples: `L_latent` 0.4181 → 0.1038; zeros baseline 0.1215; first beats baseline at step
  120 (`L_latent=0.1203`); `loss_waypoint=0.0` confirmed latent-only. Run dir:
  `runs/b1_22e_smoke_cuda_overfit/` (ignored).
- Tiny scene-split/eval smoke, 12 symlinked clips from two sources (`cand01`/`cand02`), full
  depth=6, fp32, batch 4: split 6 train / 6 val by source, val 249 windows, wrote
  `ckpt_best.pt`, `norm_stats.npz`, `val_metrics.jsonl`; one-epoch functional metric
  `val_cos=0.3013`, persistence `0.8726`, margin `-0.5713` (expected for a one-epoch tiny
  smoke; H20 run owns DoD).

**Next.** User runs H20 paste block: rsync local `.npz` cache to H20, then run
`scripts/train_sports.py --latent-only` with bf16/batch64/depth6 and paste `val_metrics.jsonl`
tail, steps/sec, GPU memory, `ckpt_best.pt`, and `norm_stats.npz` confirmation. Do not mark
B1.22b/B1.22e done until that paste-back lands.

---

## 2026-06-29 — B1.22c: curation tooling + FOLLOW-CAM data decision + 45 candidates

**Status:** B1.22c tooling DONE (AUTO); curation run produced 45 candidates (USER review/ingest pending).
**Data decision (user, load-bearing):** B-1 trains on **FOLLOW-CAM footage — the followed skier IN
FRAME, from behind** (a drone chasing/following a skier), NOT subject-free egocentric POV/helmet-cam.
Why: the model must learn the SUBJECT's dynamics (how the followed person moves/appears), which only
exists when the subject is in frame; pure-POV is a wrong-viewpoint, subject-free distribution and the
domain-blind predictor would be polluted toward subject-free predictions (same logic as game footage).
Pure-POV is at most a conditional environment-dynamics aux later — default EXCLUDE.

**Tooling.** `vllatent/ingest/curate.py` (PURE — gates + 3-level dedup + negative-title filter;
16 tests) + `scripts/curate_sports_clips.py` (yt-dlp search→metadata→gate→dedup; proxy via `--proxy`
+ ALL_PROXY popped). Gates: ≥720p, ≥23fps, aspect 1.3–2.1, 30–1200s, no live. Negative-title filter
drops off-domain "ski" homographs (jet/water-ski, snowmobile), subject-free POV (pov/helmet-cam/
first-person/point-of-view/gopro-line), and review/tutorial/gear meta. Keywords = follow/chase only.

**Result.** 120 raw → 64 deduped → **45 accepted** → `configs/sports_clips_candidates.yaml`
(cand01–45) for review→promotion into `sports_clips.yaml`. ~28 are clean FPV-drone-chase / cinematic-
ski-FPV / auto-follow-drone; ~rest are drone-reviews (real follow footage + talking heads the YOLO
filter salvages) or FPs (e.g. "Chase Jarvis" = a name). Open option: a frame-level person-in-frame
gate in the YOLO content filter for stronger "subject must be present" enforcement (tune permissively).

---

## 2026-06-29 — Data-generation strategy + undistorted preprocessing (user decisions)

**Status:** strategy decided (user); one code change (center-square-crop) shipped; plan revised.
**Why.** Move the slow+cheap half (data gen) off the rented H20 onto the free dev box; rent the
H20 only for the fast+expensive half (training). The full chain (filter → MegaSaM → DINOv3 →
`.npz`) **already ran on the 5060 Ti in B1.10c**, so this is proven, not speculative.

**Decisions.**
- **Generate the full B-1 dataset on the DEV BOX**, then `rsync` only the `.npz` (~2 GB) to the H20;
  H20 = training only. B1.22b retitled accordingly; runbook §E step 3 flipped to dev-box encode.
- **Front-load curation (B1.22c) before generation** → one local pass for the whole dataset.
- **Resolution stays 720p** (`scale=1280:720`); download caps at ≤720p. (1080p deferred — DINOv3
  squashes to 224² anyway.)
- **Aspect = center-square-crop** (NOT stretch): committed in `vllatent/encode/dinov3.py`
  `_center_square_crop` — undistorted 16:9→1:1; no-op for square input. **DEPLOY FOOT-GUN:** the
  on-drone (Jetson/RealSense) preprocessing MUST apply the same crop (Phase D). Invalidates the
  stretch-encoded ski03 cache (regenerated in the local pass). 4 crop tests; 13 encode torch green.
- **Game footage (B1.22d) DEFERRED to a conditional fallback** — build only if real-only B1.22e
  fails to beat persistence with margin; `--domain-weight` plumbing already shipped in B1.22a so it
  needs zero code changes if revived. The predictor is domain-blind ⇒ prefer real-only.

---

## 2026-06-29 — B1.22a: train_sports.py upgrade for the B-1 latent-only run

**Status:** B1.22a pending → **done** (AUTO, TORCH). The single-loop B1.20 script gained
everything the real B-1 run needs; the staged HEAD plumbing (`--stage 2/3`, freeze,
`--init-predictor`, `--head-input`) is intentionally **NOT** here (Phase B-2).

**What shipped.**
- **`--latent-only`** — optimizes `model.predictor` params ONLY (verified: head excluded), calls
  `model.predictor` directly, loss = `L_latent` only (fp32-cast outputs).
- **`vllatent/train/evaluate.py`** — `evaluate()` + `per_horizon_cosine()`: per-horizon val cosine,
  **persistence baseline** `cos(z_t, z_{t+k})`, and margin (the B-1 DoD metric); sample-weighted,
  fp32, no_grad, restores train mode, empty-loader guard.
- **`split_clips_by_source()`** (`sports_loader.py`, PURE) — scene-split holding out WHOLE source
  videos (`stem.split('_')[0]`); no sub-clip leak; guarantees ≥1 train source.
- **train-only `NormStats`** injected into val (no leakage); **`SequentialLR`** linear-warmup→cosine;
  **`ckpt_best.pt`** + early-stop on `val_cos`/`val_margin`; **bf16** default (no GradScaler; fp16
  keeps it); **`build_param_groups`** (`train/optim.py`, AdamW decay/no-decay wd 0.05, embeds/norms/
  biases excluded); **`--no-action-film`** (predictor `use_action_film` flag — dt-FiLM only);
  **`--domain-weight`** down-weights `domain=game` via `WeightedRandomSampler` over the loader's new
  `sample_domains` (default `real`; B1.22d writes `domain=game` into the .npz); per-worker RNG reseed;
  frozen **`TrainConfig`** (PURE, validated) snapshotted to `train_config.json`; `checkpoint.py`
  records `val_metrics`. Writes `val_metrics.jsonl` per eval.
- `--resume` preserved (model/opt/scheduler/step/epoch) for both modes.

**Tested.** 478 pure (split + domain plumbing) + 64 torch (evaluate, optim, action-film ablation,
all affected) green; ruff + mypy(pure) clean. Real-data CPU `--latent-only` smoke (ski03, depth=2,
20 steps): L_latent 0.42→0.20, cosine −0.00→0.36, warmup→cosine LR, ckpt saved.

**Adversarial review** (10-agent workflow): 5 findings → 2 confirmed, 3 dismissed. The "CRITICAL
`GradScaler('cuda')` crash" was an **empirically-disproved false positive** (`torch.amp.GradScaler`'s
first positional IS `device`; verified no crash, env torch 2.12). Applied the one real (defensive)
fix: explicit `.float()` on the latent-only quality weight. **USER-verifiable (B1.22e/dev box):**
the GPU `--overfit-tiny` smoke beating the zeros baseline within 200 steps.

---

## 2026-06-29 — chore: restore green quality gates (lint / typecheck / test)

**Status:** done (loop-health chore; clears the pre-existing debt flagged in the B1.21b entry below).
Not a B1.x step — restores the three ralph gates so B1.22a's verification is meaningful.
- **lint** (`ruff check .`): was 33 findings. Added `exclude = ["_archived"]` to `[tool.ruff]`
  (retired pre-pivot code, not linted; −13); `ruff --fix` auto-fixed `I001`/`F401`/`F541` in live
  scripts/tests + 3 live pkg files (`data/loader.py` drop unused `DOF`,`PATCH_TOKENS`; `ingest/
  ego_motion.py` drop unused `wrap_pi`; `encode/dinov3.py` import grouping — all behaviour-neutral);
  manual `strict=True` on two `zip()`s (`tests/test_checkpoint.py`, B905) + removed a dead
  `median_mag` assign (`tests/test_ingest_ego_motion.py`, F841). → **All checks passed.**
- **typecheck** (`mypy` pure tier): `manifest.py:176` widened `required_entry_keys: tuple[str, ...]`
  (was inferred `tuple[str,str,str]` from the wild-video branch). → **Success, 8 files.**
- **test** (`make test`): added `--ignore=tests/test_data_shapes.py` to the `test`/`test-torch`
  Makefile targets (matches the long-documented pure command). → **465 pure green.**
- **Flagged (NOT fixed here — separate dead-code decision):** `vllatent/data/loader.py` still imports
  the never-created `vllatent.data.base_loader` (legacy A5.15 `CachedLatentDataset`, plan-marked
  "NEEDS REVISION"); it can't import, so `vllatent/data/__main__` + `test_data_shapes.py` are dead.
  The **live** training path uses `data/sports_loader.py` and is unaffected.

---

## 2026-06-29 — B1.21b: stale trust-reference cleanup + remove empty verify/

**Status:** B1.21b pending → **done** (AUTO, docs-only — no behaviour change).
**What.** Removed leftover references to the trust mechanism (deleted 2026-06-25, commit `125576f`):
- `vllatent/schemas.py` docstrings: dropped "trust readout" from the student-output-seams list (L4);
  "trust-oracle disagreement signal" → "rollout disagreement signal" (L198, TeacherOutput).
- `CLAUDE.md` OPEN-list L44: "TrackVLA teacher K + trust thresholds + calibration" → "… K + calibration".
- `plans/phase-b-sports-training.md`: post-pivot scoping row A5.12 `SURVIVES`→`REMOVED` (verifier deleted);
  removed "trust head training (Phase C)" from the B-1-does-NOT-cover list.
- Deleted empty `vllatent/verify/` (only stale `__pycache__`, untracked).
- Kept (correctly): tombstones (plan L571, `heads.py` L6), the completed-B1.5 historical record
  (plan L120), and the B1.21b step/label self-references (plan L812–817, L904, L1049).
**Tests.** 465 pure green (documented cmd `pytest -m "not torch and not sim" --ignore=tests/test_data_shapes.py`);
`schemas.py` clean under ruff + mypy.
**Flagged (pre-existing, NOT introduced by B1.21b — cleared in the following chore commit):** `make test`
errored on stale `tests/test_data_shapes.py`→`vllatent/data/loader.py`→missing `base_loader` (legacy
A5.15 loader); `make lint` had 33 ruff findings (13 in `_archived/`, rest scripts/tests + 4 live files,
mostly `I001` from ruff-version drift); `make typecheck` had 1 mypy error (`manifest.py:176`).

---

## 2026-06-29 — Group 8 SCOPE CUT: B-1 = latent world model only; waypoint head → B-2

**Status:** B-1 Group 8 narrowed (user decision). The waypoint head (former B1.22f/g, `L_wp`,
Stage 2/3, joint control) is **deferred to Phase B-2a**. **B-1 DoD = a good latent prediction
model** (predictor beats a persistence baseline on real held-out val; per-horizon cosine).

**Why.** Two head-side issues are unresolved: (1) **MegaSaM scale inconsistency** — `L_wp`'s
target is the monocular VO delta, whose scale is per-clip ambiguous/drifting (median-speed +
z-score fix the distribution, not cross-clip metric scale); (2) **prober decision** undecided
(MLP vs SkyJEPA PI-Prober vs attentive-pool head). The predictor trains on clean GT DINOv3
latents (no scale ambiguity) → it's the well-posed half. Deferring the head de-risks B-1.

**Plan edits.** Group 8 retitled "Latent World-Model Training (B-1)"; B1.22a rescoped to
`--latent-only` (no `--stage 2/3`/head plumbing); B1.22e is the single B-1 run (predictor on
`L_latent`, game-pretrain variant kept); B1.23 = encoder+predictor latency; B1.24 = latent DoD;
B1.22f/g → new **B-2a** block (waypoint head training + MegaSaM-scale-fix candidates + prober
bake-off). Dependency graph + Open-Decision G updated; action-conditioned-vs-action-free is the
new B-1 sub-decision (default: keep action-FiLM — scale error is tolerable as conditioning).

---

## 2026-06-28 — Group 8 REPLAN: staged training + revised policy + expanded data

**Status:** B1.22 (single-step) **superseded** → **B1.21b + B1.22a–g** (staged). Plan-only
session (no code changed); `plans/phase-b-sports-training.md` Group 8 fully rewritten.

**Why.** User mandate: (1) **staged training** — train the latent predictor and the waypoint
head **separately** (head is downstream of predictor → clean decouple); (2) **check/revise the
training policy**; (3) **more data** incl. game footage (极限国度/Steep) because real YouTube is
too small. Backed by a 9-agent research + adversarial-verify workflow.

**Key outcomes.**
- **Staged design:** Stage 1 predictor-only on `L_latent` → Stage 2 **frozen+eval** predictor,
  head-only on `L_wp` (reads PREDICTED not GT latents) → Stage 3 conditional joint fine-tune
  (cosine-regression abort). Optional single-stage **joint control** for comparison.
- **Policy revision (B1.22a):** current `train_sports.py` has **no val loop / scene-split /
  warmup / early-stop / best-ckpt** and uses fp16+GradScaler. Fixes: bf16 (drop scaler, cast
  loss inputs fp32), linear-warmup→cosine, **scene-split by SOURCE video** (sub-clips of one
  source leak), **train-only NormStats** injected to val, best-ckpt + early-stop, AdamW
  decay/no-decay param groups (wd 0.05), `--stage {1,2,3,joint}`.
- **Data:** full pilot encode is **still pending** (only `ski03_fpv00_c000.npz` cached — B1.7
  ran `--filter-only`); expand real YouTube (~90K frames); **game footage = measured Stage-1
  pretraining source** (pretrain real+game → fine-tune predictor real-only → keep only if real
  val cosine improves; predictor is domain-blind so unguarded game data pollutes it).
- **Ego-Exo4D DEMOTED** (Open Decision #9): wrong motion/domain (no skiing, no high-speed
  ego-translation) — not the volume saviour the plan assumed.
- **Rejected report stays REFERENCE-ONLY:** `training-policy-research-2026-06-25.md`
  (PI-Prober/AdaLN/visual-bottleneck/SkyJEPA/GRPO) — only its staged *schedule* is adopted, with
  the LEAN architecture kept (MLP head, FiLM, no anti-collapse).

**Open (defaults in effect):** data-timing = pilot-now + expand-parallel; joint control = run it;
head-input = locked `mean` + cheap escalation; depth=6 (+ optional 2–4 sweep).
**Loose end:** working tree has **uncommitted** `predictor.py` (SDPA flash-attn) + `train_sports.py`
(bf16/AMP) changes from the B1.20 run — commit before B1.22a builds on them.

---

## 2026-06-28 — Code review fix: 15 findings (3C/4H/5M/3L)

**Status:** B1.20 still **in_progress** (fixes applied, run still USER-GATED).
**What:** 10-agent code review on B1.17–B1.20 surfaced 15 findings. All fixed in one commit.

**CRITICAL (3) — fixed:**
- #1: `loss_out` UnboundLocalError on resume when loop never executes. Init to `None`, guard final save.
- #2: LR scheduler restarts from peak on resume. Now saves/restores `scheduler_state_dict`.
- #3: Config snapshot recorded defaults, not actual CLI args. `Config(predictor=pred_cfg)` embeds real hyperparams.

**HIGH (4) — fixed:**
- #4: **GT future delta leak.** `batch.target_deltas[:,0]` fed as FiLM action — IS the prediction target. Added `last_action` field (`velocities[t-1]`, zeros at clip start) to `SportsSample`/`TrainingBatch`; model uses `batch.last_action`.
- #5: `history_mask` never forwarded to predictor. Now zeros out temporal embeddings for padded history slots.
- #6: `NormStats` never persisted. Now calls `dataset.save_norm_stats()` after construction.
- #7: Infinite loop when `len(dataset) < batch_size`. Guard raises `ValueError`.

**MEDIUM (5) — fixed:**
- #8: NaN `dt_seconds` passed positivity check (IEEE 754). Added `np.isfinite` guard.
- #9: `vo_confidence`/`frame_quality` never validated. Added finite + range checks.
- #10: `sample_weight` field kept (collate computes it, still valid for future use). No semantic change.
- #11: Epoch counter off-by-one on early break. Break moved before `epoch += 1`.
- #12: `TrainingLogger` ZeroDivisionError on `log_every=0`. Added `>= 1` validation.

**LOW (3) — fixed:**
- #13: `compute_baseline_loss` now uses `latent_loss`/`waypoint_loss` (same weighting as training).
- #14: `sanity.py` docstring corrected from "TORCH tier" to "PURE tier".
- #15: `losses.py` top-level `import torch` moved inside functions (lazy import, CI-safe).

**Tests:** 465 pure + 106 torch = 571 all green.

---

## 2026-06-26 — B1.20: Training script (AUTO half — script written)

**Status:** B1.20 pending → **in_progress** (script AUTONOMOUS; run USER-GATED).
**What's done.** `scripts/train_sports.py` — full training script with `--overfit-tiny` mode:

- Loads `SportsTrainingDataset` from cache dir, runs sanity check
- Builds `SportsFollowingModel` from `PredictorConfig` (depth/heads/dropout args)
- Dumb baseline: `loss(zeros, GT)` — must beat within 200 steps
- AdamW + cosine LR schedule + gradient clipping
- `combined_loss` (L_latent beta=0.1 + L_wp, per-sample quality/confidence weighting)
- Per-step logging via `TrainingLogger` (JSONL with per-horizon cosine + L1)
- Checkpoint save every N steps + resume from checkpoint
- Prints baseline comparison (YES/no) at each log step

**USER GATE:** Run on dev box (RTX 5060 Ti) with pilot cache data:
```bash
conda run -n vllatent-ego-drone python scripts/train_sports.py \
    --overfit-tiny --cache-dir ingest_data/latent_cache \
    --run-dir runs/overfit_tiny --device cuda --max-steps 500
```
**DoD:** Loss drops below baseline within 200 steps. Checkpoint saved + resume-tested.

---

## 2026-06-26 — B1.21: Pre-train sanity check + viz

**Status:** B1.21 pending → **done** (AUTO).
**What's done.**

- **`vllatent/train/sanity.py`** — `run_sanity_check(dataset, n_samples=5)`: reads N random samples
  from `SportsTrainingDataset`, validates z_t/history/mask/target shapes+dtypes, checks history_mask[-1]
  is True (z_t slot real), target_deltas finite, dt_seconds positive. Raises `ValueError` on any breach.
  Empty dataset → immediate raise. 7 pure tests.

- **`vllatent/train/viz.py`** — `TrainingLogger`: append-only JSONL logger. `log_step()` records
  step/epoch/loss_total/loss_latent/loss_waypoint/cosine_sim/lr, plus optional per-horizon-step
  cosine similarity and waypoint L1 error breakdowns. `should_log(step)` for periodic logging.
  6 torch tests.

**Tested.** 7 sanity (pure) + 6 viz (torch) = 13 total. Ruff clean.

---

## 2026-06-26 — B1.18: Loss functions

**Status:** B1.18 pending → **done** (AUTO).
**What's done.** `vllatent/train/losses.py` — three loss functions:

- **`latent_loss`**: smooth L1 (beta=0.1, DINO-world precedent) between predicted and GT future
  latents, quality-weighted per sample (`frame_quality.clamp(min=0.1)`).
- **`waypoint_loss`**: smooth L1 between predicted and GT deltas, confidence-weighted per sample
  (`vo_confidence.mean(1).clamp(min=0.05)`). **NOT weighted by frame_quality** (waypoint head
  needs to learn from fast/blurry frames too).
- **`combined_loss`**: `L_total = lambda_latent * L_latent + lambda_waypoint * L_wp`. Returns
  `LossOutput(total, latent, waypoint, cosine_sim)`. Cosine sim is diagnostic (no gradient).

**Tested.** `tests/test_losses.py` (15): scalar output, beta=0.1 vs default, quality/confidence
weighting, zero on identical, differentiable, LossOutput type, floor clamps, lambda scaling,
cosine sim range + perfect, frame_quality NOT applied to L_wp. Ruff clean.

---

## 2026-06-26 — B1.17: Full model assembly

**Status:** B1.17 pending → **done** (AUTO).
**What's done.** `vllatent/model/sports_model.py` — `SportsFollowingModel(nn.Module)` assembles
`LatentPredictor` + `WaypointHead`. Forward takes `TrainingBatch` → `ModelOutput(predicted_latents,
predicted_deltas)`. Encoder NOT part of forward (latents cached). Config-driven via `PredictorConfig`
(depth/heads/mlp_ratio/dropout/history/horizon). `from_config()` class method for convenience.

Waypoint head receives patch-mean-pooled predictor output `(B,T,D)` → `(B,T,4)`.

**Tested.** `tests/test_model.py` (8): output shapes D=768 + D=384, ModelOutput type, from_config,
differentiable (gradients flow), param count depth=6 (40-70M range), config-driven depth (more depth
= more params), eval deterministic. 450 pure + 8 model torch tests green. Ruff clean.

---

## 2026-06-26 — B1.13: Sports sliding-window loader

**Status:** B1.13 pending → **done** (AUTO).
**What's done.** `vllatent/data/sports_loader.py` — `SportsTrainingDataset` map-style Dataset
over ingest `.npz` cache files. Sliding windows of (H+T) frames. Per sample: z_t, history_latents
(GT from cache), history_mask (block-causal), target_latents, target_deltas (preprocessed),
vo_confidence, frame_quality, dt_seconds.

**Delta preprocessing pipeline** (applied at construction): physics hard clip (scaled by dt) →
median filter k=3 → velocity normalize (delta/dt) → per-dimension z-score. `NormStats` saves/loads
for inference denormalization.

**Augmentation** (toggleable): temporal jitter ±1 + Gaussian delta noise (0.05σ).

**Also done:** B1.11/B1.12 accepted default assumption (D=768, depth=6). PredictorConfig default
depth 12→6, dropout=0.1 added (arch research 2026-06-25).

**Tested.** `tests/test_sports_loader.py` (30): preprocessing (physics clip, median filter,
velocity normalize, norm stats roundtrip), dataset shapes/dtypes, block-causal mask ramp-up,
GT history verification, multi-clip, short-clip skip, augmentation, import purity. 450 total green.
Ruff clean.

---

## 2026-06-25 — Remove trust mechanism entirely

**Status:** done (user decision).

Deleted `vllatent/verify/` (V-JEPA-2 surprise verifier), `TrustReadout` from schemas,
`TrustConfig` + `DISAGREEMENT_SOURCES` from config, `vjepa_surprise` from `SportsTarget` /
`OracleTarget` / loader / data quality report, trust section from `sports.yaml`, trust references
from CLAUDE.md / io-contract / arch_diagram / full-run-sizing. Cleaned all tests (420 passed,
11 skipped). Manifest validation no longer checks `disagreement_source` enum.

**Rationale:** User abandoned the trust/commitment-horizon concept. The project focus is
now purely on the latent predictor + waypoint head.

---

## 2026-06-25 — B1.10f: Fix shot boundary detection for consistent VO trajectories

**Status:** new sub-step **done**.

**Problem:** AdaptiveDetector (PySceneDetect) at threshold=3.0 missed obvious hard cuts in skiing
footage — e.g. ski03 frame 54 (behind-skier FPV) → 55 (bird's-eye-view) was NOT detected as a shot
boundary. MegaSaM processed both camera angles as one continuous segment → physically impossible
trajectory leap.

**Root cause:** Skiing footage has high natural frame-to-frame variation (motion 19-28 typical). A cut
between two snowy scenes (motion spike to 38) didn't exceed 3× the local average.

**Fix:** `adaptive_threshold` 3.0 → **2.0** across all `detect_shot_boundaries*` and `filter_video*`
functions. At 2.0, ski03 produces 7 boundaries (was 4) — all verified as real camera switches. No
false positives observed.

**Deleted:** `vllatent/ingest/edit_detection.py` + `tests/test_edit_detection.py`. Two rounds of
hand-crafted edit detection (histogram correlation, motion spikes, slow-mo, block-pattern consistency)
all produced catastrophic false-positive rates on real skiing footage (80+ frames flagged out of 254).
The entire "edit detection" problem was actually just an under-tuned AdaptiveDetector threshold.

**Bug fix:** `scripts/test_e2e_subclip.py` — stale frame directory not cleaned before copy (previous
run's 50 frames persisted; MegaSaM saw 50 instead of 39). Fixed with `shutil.rmtree` before copy.

---

## 2026-06-24 — B1.10 DONE: MegaSaM VO validation — full E2E pipeline verified

**Status:** B1.10c pending → **done** (USER-VERIFIED). All B1.10 sub-steps (a,b,c,d,e) complete.
**Verdict:** **GO** on `ski03_fpv00_c000` — 50 frames, latents `(50,196,768)` fp16, deltas `(49,4)` f32,
all `frame_quality >= threshold`. Output: `reports/e2e_test/cache/ski03_fpv00_c000.npz`.

**B1.10c was redefined** as a full end-to-end pipeline test on one sub-clip (was: run MegaSaM on 3 pilot
clips). Script `scripts/test_e2e_subclip.py` orchestrates: content filter → FPV shot detection → 10s
sub-clip → quality gate → `find_accepted_segments()` → MegaSaM VO (3-step) → DINOv3 encode → `.npz` cache
→ shape/dtype/quality validation.

**Bugs found and fixed during E2E:**
- `pipeline.py`: `run_megasam(..., model=megasam_model)` — stale kwarg from pre-B1.10e signature; fixed
  to `clip_id=segment_id`.
- `run_megasam_pipeline.sh`: `--no-banner` unsupported by user's conda; removed from all 3 `conda run`.
- `content_filter.py`: `extract_fpv_ranges()` ignored per-frame `fpv_mask` — non-FPV frames within FPV
  shots leaked into sub-clips. Fixed: added `fpv_mask` parameter that splits ranges at frame-level
  YOLO/motion rejections. 4 new tests.
- MegaSaM `mega_sam` env: xformers upgrade (0.0.35) broke all C++ extensions + CUDA kernels (sm_120 on
  RTX 5060 Ti unsupported). Fixed via `scripts/megasam_shims/nystrom_shim.py` (NystromAttention replacement
  + `memory_efficient_attention` → PyTorch SDPA monkey-patch + `unbind` shim) + CUDA 13.0 toolkit +
  `.type()→.scalar_type()` + `torch.cuda.amp.autocast→torch.amp.autocast("cuda")` + sm_120 gencode flags.
- DINOv3 encoder: socks:// → socks5:// proxy URL normalization.

**New files:** `scripts/test_e2e_subclip.py`, `scripts/megasam_shims/nystrom_shim.py`,
`scripts/megasam_shims/run_unidepth.py`, `vllatent/io.py`.
**Modified:** `vllatnet/ingest/pipeline.py` (stale kwarg + segment-based processing), `content_filter.py`
(fpv_mask), `quality.py` (find_accepted_segments), `scripts/run_megasam_pipeline.sh` (shims + no-banner),
`scripts/ingest_youtube_pilot.py` + `scripts/verify_filter.py` (pass fpv_mask), `vllatnet/encode/dinov3.py`
(proxy fix), `tests/test_content_filter.py` + `tests/test_ingest_pipeline.py` (new tests).
**Next:** B1.11 (Benchmark DINOv3 ViT-B/16 on Orin NX — CRITICAL GATE, USER-GATED).

---

## 2026-06-24 — B1.8 DONE (DESCOPED): CosFly-Track adapter — trajectory-only, no RGB

**Status:** B1.8 in_progress → **done** (DESCOPED).
**Descope rationale.** CosFly RGB frames are CARLA urban simulator renders (119 GB, ~82K PNGs).
DINOv3 latents of simulated cars and buildings are useless for predicting skiing mountain latents
(complete domain mismatch). The only value is **GT 6-DoF trajectories** (~6 GB trajectory JSONs)
for clean L_wp delta supervision (`vo_confidence=1.0`). Download via `--meta-only` flag; skip
the 119 GB of frames entirely.
**What ships.** `vllatent/ingest/cosfly_adapter.py` (PURE, 21 tests), `scripts/download_cosfly.sh`
(`--meta-only` for trajectory JSONs only). Adapter parses `trajectory.json` → GT deltas `(N-1,4)`.
No latent encoding step — CosFly contributes to L_wp only, not L_latent.
**Impact on B1.13 loader.** The "oversample CosFly to ~40% of batches" plan item needs revision:
CosFly provides delta-only samples (no latents), so it can only feed L_wp, not L_latent. The
loader must handle missing-latent samples or CosFly is L_wp-curriculum only. Revisit in B1.13.

---

## 2026-06-23 — B1.7 USER GATE PASSED: YouTube pilot ingest complete

**Status:** B1.7 done (USER-VERIFIED).
**Result.** 15 skiing FPV clips curated in `configs/sports_clips.yaml`. Pipeline ran with
`--filter-only`: 11 accepted, 4 rejected (ski07: 0/7 FPV, ski08: 2/9, ski14: 13/44, ski15: 0/1).
38 FPV ranges extracted → 173 sub-clips (10s each). Content filter verified on ski01 via
`verify_filter.py` (accepted/rejected split reviewed). `pilot_summary.json` in latent_cache validates.
**Next:** B1.8 (CosFly-Track adapter, USER-GATED) + Group 2 (B1.9 done, B1.9b done, B1.10 pending).

---

## 2026-06-21 — B1.7b REVISED: CLIP → YOLO-World + minimum segment filter

**Status:** B1.7b revised — done (AUTO).
**Problem.** CLIP ViT-B/32 zero-shot scores 0.999 on ALL frames within the same visual domain
(all skiing = snow+mountain+trees). Zero discriminative power for within-domain filtering.
CLIP ignores prepositions (ARO/WinoGround ICLR 2023: 0.50-0.56 compositional accuracy).
**Fix — two changes:**
1. **CLIP → YOLO-World `yolov8s-worldv2`** (74 FPS V100, 13M params, ~1.5 GB VRAM).
   Open-vocabulary object detection via `set_classes()` (text cached, no per-frame re-encoding).
   36 rejected classes in 4 groups: drone body+parts (rotor, propeller, gimbal, landing gear,
   drone arm, motor, battery, RC controller, …), camera/filming gear (GoPro, action camera,
   tripod, monopod, stabilizer, …), electronics (laptop, monitor, phone screen), overlays
   (text overlay, title card, logo, watermark, subtitle).
2. **`filter_short_segments(mask, min_length=10)`** — discards contiguous accepted runs shorter
   than 10 frames (2s at 5fps). Prevents tiny fragments between rejected regions from leaking
   into training data as unusable micro-episodes.
- Filter logic: `is_fpv = motion >= 8.0 AND ¬YOLO AND segment >= 10 frames`
- `ultralytics>=8.2.0` added to `[torch]` extra in `pyproject.toml`
- Phase B plan updated (B1.7b rewritten, decisions/refs/dependency graph updated)
- 44 content filter tests green (was 36). 396 total suite green (was 388).

---

## 2026-06-20 — B1.7c REWORK: path-based content filter (every frame scored, no stride)

**Status:** B1.7c reworked — done (AUTO).
**Problem.** The prior B1.7c stride-sampled ~50 frames per video for the content filter.
Non-FPV content (talking heads, drone closeups, title screens, B-roll) between sampled frames
went undetected and leaked into training data. A 3-second talking-head in a 10-minute video
was invisible to the 50-frame sample.
**Fix — three new path-based functions in `vllatent/ingest/content_filter.py`:**
- `score_frames_from_paths(frame_paths, device, batch_size=32) → (N,) float32` — loads frames
  in bounded batches of `batch_size`, scores each via `_get_clip_scorer()`. Never holds more
  than `batch_size` frames in RAM.
- `detect_shot_boundaries_from_paths(frame_paths, …) → list[int]` — loads frames one at a
  time through PySceneDetect AdaptiveDetector. Never holds more than 1 frame.
- `filter_video_from_paths(frame_paths, device, …) → FilterResult` — composes the two above
  + `classify_shots` + `video_verdict` + `fpv_frame_mask`. Same `FilterResult` as `filter_video()`,
  but the FPV mask covers EVERY frame. No stride. No sampling.
All three use lazy torch/PIL/scenedetect imports (tier rule). Added to `__all__`.
**Pilot script reworked** (`scripts/ingest_youtube_pilot.py`):
- Replaced stride-sampled `filter_video()` call with `filter_video_from_paths(frame_paths, …)`.
- Removed `stride` variable, PIL Image.open sampling loop, unused numpy import.
- FPV ranges from `extract_fpv_ranges(filter_result.shots)` are now exact frame indices (no
  stride-to-full-frame remapping needed).
**Tested.** 35 content filter tests (27 existing + 8 new path-based): `score_frames_from_paths`
(shape, batching, empty-raises), `detect_shot_boundaries_from_paths` (processes all, empty-raises),
`filter_video_from_paths` (full result, mask covers every frame, empty-raises). 387 total tests
pass, 0 regressions. Ruff clean.

---

## 2026-06-20 — B1.7 REPLAN: split into B1.7a + B1.7c sub-steps

**Status:** B1.7 replanned. Three bugs/gaps identified:
1. **Missing `vllatent.encode.batch`** — `pipeline.py:93` imports `encode_frames` from a module that doesn't
   exist. → B1.7a creates `vllatent/encode/batch.py` with `encode_frames(frames_dir, device) → (N,196,768) fp16`.
2. **No segment-level FPV extraction** — the pilot script does whole-video ACCEPT/REJECT but never uses the
   per-frame FPV mask to cut out non-FPV segments (talking heads, title screens, B-roll) or calls
   `cut_fixed_clips()` for 10s clip cutting. → B1.7c adds `extract_fpv_ranges()` to `content_filter.py` and
   reworks the pilot script.
3. **Full download is correct** — yt-dlp needs the full video first for the content filter to identify FPV
   segments. SponsorBlock pre-strips crowdsourced junk. No change needed.

**Plan updated:** `plans/phase-b-sports-training.md` — B1.7a and B1.7c sub-steps added before B1.7.
**Dependency chain:** B1.7a (AUTO) + B1.7b (done) → B1.7c (AUTO) → B1.7 (USER-GATED).

---

## 2026-06-20 — B1.7c: FPV segment extraction + pilot script rework

**Status:** B1.7c pending → **done** (AUTO).
**What's done.**
- `vllatent/ingest/content_filter.py` — added `extract_fpv_ranges(shots) → list[tuple[int,int]]`:
  merges consecutive FPV shots into contiguous frame ranges (non-FPV shots break the range).
  Added to `__all__`. 6 new tests in `tests/test_content_filter.py` (all FPV, mixed, no FPV,
  empty, single, gap).
- `scripts/ingest_youtube_pilot.py` — reworked for segment-level FPV + 10s clip cutting:
  1. Download + extract all frames (unchanged)
  2. Content filter on sampled frames → get verdict + shots
  3. **NEW**: `extract_fpv_ranges(shots)` → FPV frame ranges (stride-mapped to full-frame indices)
  4. **NEW**: `cut_fixed_clips(range_frames, clip_length_frames)` → 10s segments per FPV range
  5. **NEW**: Each sub-clip gets ID `{clip_id}_fpv{range_idx:02d}_c{clip_idx:03d}`, frames copied
     to sub-directory, pipeline runs per sub-clip with `skip_download=True`
  Fallback: if no FPV ranges found, treats entire video as one range. `--filter-only` reports
  FPV ranges + sub-clip counts without running pipeline. Summary tracks per-clip OK/error.
**Tested.** 27 content filter tests (21 existing + 6 new); ruff clean.
**No blockers.** B1.7 (USER-GATED pilot run) is now unblocked.

---

## 2026-06-20 — B1.7a: batch DINOv3 encoder

**Status:** B1.7a pending → **done** (AUTO, TORCH tier).
**What's done.** `vllatent/encode/batch.py` — `encode_frames(frames_dir, device) → (N, 196, 768) fp16`:
loops over sorted `*.jpg` files, loads each via `load_rgb`, encodes via `DinoV3Encoder.encode_rgb`,
stacks into `(N, PATCH_TOKENS, EMBED_DIM)` fp16 numpy array. Lazy torch imports (DinoV3Encoder +
load_rgb imported inside function body). Raises `ValueError` if dir missing, `FileNotFoundError`
if no JPEGs. Resolves the missing import at `pipeline.py:93`.
**Tested.** `tests/test_encode_batch.py` (5 tests, PURE — mocked encoder, no real weights):
shape/dtype contract, empty/missing dir errors, sorted order verification, AST import purity.
**Full suite:** 379 passed, 11 skipped, 0 regressions. Ruff clean.
**No blockers.** B1.7c can proceed.

---

## 2026-06-20 — B1.7: YouTube pilot ingest — script written; USER-GATED (download + pipeline)

**Status:** B1.7 pending → **in_progress** (script AUTONOMOUS; download + pipeline USER-GATED).
**What's done.**
- `vllatent/ingest/acquire.py` — added `sponsorblock: bool = False` parameter to `download_clip()`.
  When True, adds `["--sponsorblock-remove", "all"]` to yt-dlp command (strips sponsor/intro/outro
  segments via SponsorBlock crowdsourced data before frame extraction).
- `scripts/ingest_youtube_pilot.py` — orchestration script:
  1. Loads clips from `configs/sports_clips.yaml` (15 curated skiing FPV clips, already populated)
  2. Downloads each with SponsorBlock enabled (skips already-downloaded)
  3. Extracts frames at 5 fps (from `configs/sports.yaml` IngestConfig)
  4. Runs content filter (B1.7b) — CLIP+PySceneDetect, samples ≤50 frames per clip
  5. REJECTed clips are skipped; ACCEPT/PARTIAL proceed to full pipeline
  6. Full pipeline: quality scoring → MegaSaM VO → DINOv3 encode → .npz cache
  7. Writes `pilot_summary.json` to cache dir
  Flags: `--limit N`, `--device`, `--skip-download`, `--skip-megasam`, `--filter-only`
  Uses config paths throughout (raw_dir/frames_dir/cache_dir from IngestConfig defaults).
- `configs/sports_clips.yaml` — 15 curated clips (ski01-ski15), already populated from B1.6 era.
**USER GATE:** user must run the script and verify output. See command block below.
**DoD (from plan):** 10+ clips downloaded, sponsor segments stripped, content-filtered (FPV accepted),
extracted, quality-scored, MegaSaM-processed, DINOv3-encoded, cached as `.npz`. Manifest validates.

---

## 2026-06-20 — B1.9b: per-clip HTML quality report (Plotly)

**Status:** B1.9b pending → **done** (AUTO, TORCH tier — all plotly imports lazy).
**What's done.** `vllatent/ingest/visualize.py` (~250 LOC) + `scripts/clip_report.py` CLI.
Generates self-contained offline HTML per clip with 5 interactive Plotly sections + summary:
1. **Frame quality timeline** (RdYlGn colorscale, markers+line)
2. **3D ego-motion trajectory** (Scatter3d, colored by speed magnitude, cumulative deltas)
3. **Body-frame deltas** (3-row subplot: dx/dy/dz, dyaw, quality overlay)
4. **VO confidence timeline** (threshold line at 0.3, flagging low-confidence regions)
5. **Latent coherence** (cos_sim z_t/z_{t+1}, threshold at 0.85 for scene change detection)
6. **Summary table** (frames, duration, npz size, means, PASS/FAIL banner)

Helper functions: `compute_latent_coherence` (cosine sim between consecutive latent frames),
`compute_cumulative_trajectory` (cumsum xyz), `compute_speed_magnitudes` (L2 norm xyz).
Plotly CDN for first section, deferred for rest (fast load). CLI:
`python scripts/clip_report.py --cache <dir> --clip <id>`.
**Tested.** `tests/test_visualize.py` (15): HTML structure (7 section checks), write-to-file,
minimal frames (7), latent coherence shape+range, cumulative trajectory correctness,
speed magnitudes, import purity (AST). Ruff clean.
**New dependency:** `plotly` (pip install plotly).
**No blockers.** B1.9b complete.

---

## 2026-06-20 — B1.7b: content filter implementation (CLIP + PySceneDetect)

**Status:** B1.7b pending → **done** (AUTO, TORCH tier — all heavy imports lazy).
**What's done.** `vllatent/ingest/content_filter.py` (~250 LOC) — full content filter pipeline:
- **Shot boundary detection:** PySceneDetect `AdaptiveDetector` wrapper processes in-memory frames
  (no video file needed). Returns sorted frame indices of shot transitions. Configurable
  `adaptive_threshold` (default 3.0) and `min_scene_len` (default 2).
- **CLIP zero-shot FPV scoring:** `_get_clip_scorer()` lazy-loads the existing CLIP ViT-B/32 model
  (same `openai/clip-vit-base-patch32` from `vllatent/encode/text.py`). 8 positive FPV prompts
  (drone, GoPro, POV, egocentric, skiing, MTB, FPV racing, body-mounted) vs 8 negative prompts
  (talking head, text overlay, title screen, static shot, interview, ad, subscribe button,
  stationary). Per-frame score = clipped `(mean_pos - mean_neg + 0.5)` ∈ [0,1].
- **Per-shot majority vote:** `classify_shots()` splits frames at boundaries, per-shot FPV if >50%
  of frames score >= threshold (default 0.25). Returns `ShotClassification` with `ShotInfo` list.
- **Whole-video verdict:** `video_verdict()` — ACCEPT (>=60% FPV shots), REJECT (<30%), PARTIAL.
- **Per-frame FPV mask:** `fpv_frame_mask()` → boolean `(N,)` array for pipeline integration.
- **Thumbnail grid data:** `thumbnail_grid_data()` selects mid-shot representative frames with
  accept/reject labels for human review.
- **Top-level orchestrator:** `filter_video()` → `FilterResult` (verdict, mask, boundaries, shots,
  per-frame scores).
**Tier compliance.** All heavy imports (torch, transformers, scenedetect, PIL) are LAZY — inside
functions only. Module imports on a torch-free box (AST-verified in tests).
**New dependency:** `scenedetect` (`pip install scenedetect[opencv-headless]`).
**Tested.** `tests/test_content_filter.py` (21 tests, PURE — CLIP scorer mocked via `patch`):
SBD (list return, single-scene empty, sharp-cut detected, empty-raises); CLIP scoring
(shape, range [0,1], empty-raises); shot voting (all-FPV, mixed, threshold boundary);
video verdict (ACCEPT, REJECT, PARTIAL, threshold values); FPV mask (shape, values);
thumbnail grid (structure); import purity (AST + sys.modules); integration
(filter_video ACCEPT, filter_video REJECT). 340 total tests pass (1 pre-existing failure
in `test_ingest_preprocess.py` from missing `vllatent.io` — unrelated). Ruff clean.
**No blockers.** B1.7 (YouTube pilot) is now unblocked.

---

## 2026-06-19 — B1.19: checkpoint save/load + config snapshot

**Status:** B1.19 pending → **done** (AUTO, TORCH tier).
**What's done.** `vllatent/train/checkpoint.py` — `save_checkpoint()` / `load_checkpoint()` /
`snapshot_config()` / `seed_everything()`. Save writes model state, optimizer state, epoch,
global step, config (as plain dict), and metrics. Load restores model + optionally optimizer.
Config snapshot writes a YAML file once at training start. `seed_everything` sets torch +
numpy + python random deterministically. All torch imports lazy (inside functions); module
imports pure-box-safe (AST-verified).
**Tested.** `tests/test_checkpoint.py` (10, `@pytest.mark.torch`): YAML round-trip (all sections +
ingest), tuple→list coercion, save/load round-trip (weights + metadata), parent-dir creation,
load-without-optimizer, **resume produces identical gradients** (save at step N, load, verify
step N+1 grad matches), deterministic seed, import purity AST guard. Full pure suite 375+13s
green.
**No blockers.** B1.19 has no dependencies (parallel with B1.15–B1.18, all blocked by B1.11→B1.12).

---

## 2026-06-19 — Phase B start: sports-following training plan

**Status:** Phase A complete. Phase B plan written (`plans/phase-b-sports-training.md`).
**Pivot.** Project pivoted from indoor AerialVLN to autonomous sports-following drone (skiing
primary). WorldVLN teacher retired. Training data = sports FPV video (YouTube + CosFly-Track).
Three blockers identified from research reports: (1) no teacher for L_kd (TrackVLA unreleased),
(2) YouTube skiing FPV 3-5x smaller than assumed, (3) pipeline bugs (undistort unwired, GPS
stubbed, MegaSaM confidence np.ones). Two simplifications: (4) ViT-B/16 may be fast enough on
Orin NX (skip CosPress), (5) language cross-attention is cheap.
**Plan.** 24 steps (B1.1-B1.24) across 8 groups. Critical path: B1.11 (Orin NX benchmark) ->
B1.12 -> B1.15 -> B1.17 -> B1.18 -> B1.20 -> B1.22 -> B1.24. Three parallel tracks before the
encoder gate. User-gated: B1.7/B1.8/B1.10/B1.11/B1.20/B1.22/B1.23/B1.24.
**Superseded plans.** `phase-a5-replan-postpivot.md` (SUPERSEDED banner added).
**Decisions locked:** clip length 10s, encoder working default ViT-B/16 (D=768), GPS Sim(3) stub only.
**Next.** Ralph loop starts at B1.1 (pipeline bug fix, pure-tier cheap-win).

---

## 2026-06-15 — A5.18: second review H1 fixed (224² bottleneck decoupled)
**Second independent code review** found 1 HIGH + 4 MEDIUM + 2 LOW. H1 fixed; MEDIUMs accepted as
Phase B/C gates.
- **H1 (FIXED): 224² bottleneck to teacher/V-JEPA-2.** All frames were center-crop+resized to 224²
  (correct for DINOv3), then the same 224² sent to the WorldVLN server (which upscales to 640²) and
  V-JEPA-2 (which resizes to 256²). Fix: decouple — render once at native → center-crop to square at
  native res (480²) → DINOv3 gets `resize_square(sq, 224)`, teacher and V-JEPA-2 get the native-square
  crop (their processors resize internally from better source). `center_crop_and_resize` split into
  `center_crop_to_square` + `resize_square`. Manifest hash updated. +2 tests (253 total).
  ⚠ **Existing 6-episode .npz files are STALE** (built with 224² teacher/verifier input) — must
  regenerate before Phase B training. Data is gitignored so no commit conflict.
- **M1 (Phase C):** T/H sweep not wired through seam validators (constants lock the current values;
  Phase C explicitly owns the sweep).
- **M2 (Phase B gate):** teacher waypoint frame vs GT body-delta frame unverified. Add a correlation
  check early in Phase B (compare teacher waypoint direction against known GT deltas).
- **M3 (Phase C):** V-JEPA-2 1+1 degenerate mode (shape-correct ≠ discriminative). Phase C GO/NO-GO.
- **M4 (Phase B design):** open-loop teacher targets paired with real observations — inherent to
  world-model distillation; Phase B decides (step-weighting or early-segment-only).
- **L1, L2:** minor guards, won't bite on AerialVLN data.

---

## 2026-06-15 — A5.18 DONE: Phase-A complete (first code review settled)
**Status:** A5.18 in_progress → **done**. **PHASE A COMPLETE.**
**Code review verdict:** WARNING (0 CRITICAL, 4 HIGH, 4 MEDIUM, 3 LOW). All 4 HIGHs fixed:
- HIGH-1: `wrap_pi` docstring `(-pi, pi]` → `[-pi, pi)` (matched implementation).
- HIGH-2: `data_audit.yaml` `reference_path` stale 7-wide quaternion → corrected to 6-wide Euler.
- HIGH-3: `data_audit.yaml` `camera_name`/`vehicle_name` stale (`front_0`/`Drone_1`) → `front_center`/`drone_1`.
- HIGH-4: `cache.py` resume branch `np.load` fd leak → context manager (`with np.load(...) as npz_data`).
**MEDIUMs accepted (no fix needed for Phase A):** M1 zero-padding beyond segment 0 (documented, Phase B);
M2 per-call npz reload (Phase B restructure); M3 uint dtype edge case (not reachable); M4 empty env-var
path (defaults in production YAML).
**Phase-A DoD met:** (1) typed Config SoT + student+teacher seams + io-contract.md; (2) AuditSummary
clean on real slice (50/50, 10,198 transitions); (3) valid `(StepSample, OracleTarget)` tuples from 6
real cached episodes. 253 pure / 5 torch / lint / mypy / blob — all green. Hand-off → **Phase B
(distillation training).**

---

## 2026-06-15 — A5.17 DONE: sizing doc + guard script + build verified
**Status:** A5.17 in_progress → **done**. Sizing doc (`docs/full-run-sizing.md`) + guard script
(`scripts/run_full_cache.sh`) landed (AUTO). User verified incremental build (limit-1, 6th episode
added to existing 5-episode cache). Full 50-episode bulk run deferred to Phase B start.

---

## 2026-06-15 — A5.17 AUTO: sizing doc + guard script (bulk build USER-GATED)
**Status:** A5.17 pending → **in_progress** (sizing AUTO done; bulk USER-GATED).
**What's done (AUTO).** `docs/full-run-sizing.md` — full dataset scope (50 eps, 10,248 poses, 10,198
trainable transitions), per-episode disk (~62 MB, latents dominate at >99.8%), full cache estimate
**~3 GB** compressed, wall-clock **~11–14 hours** (WorldVLN teacher >90% of time: K=5 × ~3 segments ×
160 denoising steps @ 3.5 it/s per rollout), GPU memory budget (local ~2 GB, H20 ~24 GB), prerequisites
checklist. `scripts/run_full_cache.sh` — guard script, exits non-zero without `--i-have-signed-off`,
wraps `python -m vllatent.cache build` with full-slice defaults, resumable.
**Tested (AUTO).** `test -f docs/full-run-sizing.md && grep -q "GB"` PASS; `bash scripts/run_full_cache.sh`
exits 1 without flag; 251 pure / lint / typecheck / blob green.
**Next — USER-GATED:** the full 50-episode bulk build (see command in the sizing doc). This is a
~11–14 hour GPU job. After the user runs it and pastes the output, A5.17 flips to `done`. Then A5.18
(Phase-A DoD sign-off).

---

## 2026-06-15 — A5.16 DONE: loader inspect over real cache GREEN (user-pasted)
**Status:** A5.16 pending → **done**. The USER ran `python -m vllatent.data inspect --cache
data/latent_cache/ --n 4` inside fly0-m1 docker and pasted the output — not agent-fabricated.
**Inspect output (user-pasted).** `cache data/latent_cache/: 5 episodes, 987 transitions (H=3)`.
4 samples: `z_t (196,768) float16` (correct DINOv3 shape/dtype), `action=` valid ints (4,4,3,3),
`hist_mask` block-causal ramp-up correct (`[F,F,T]→[F,T,T]→[T,T,T]`), `lang=(77,768)` (CLIP max-seq
tokens), `waypoint=` 4-float teacher targets in m/deg, `disagree=0.0155–0.0309` (positive, K-rollout
spread), `surprise=0.2086–0.2411` (positive, V-JEPA-2 cosine). All `(StepSample, OracleTarget)` tuples
well-formed. The full distillation-loader round-trip is verified: real .npz → `CachedLatentDataset` →
typed contract objects.

---

## 2026-06-15 — A5.14 DONE: cache pipeline end-to-end verified (small-slice build GREEN)
**Status:** A5.14 in_progress → **done**. The USER ran the small-slice build in fly0-m1 docker with H20
WorldVLN server and pasted the output — not agent-fabricated.
**Small-slice build (user-pasted).** `python -m vllatent.cache build --slice … --limit 5 --out
data/latent_cache/ --teacher-server http://127.0.0.1:8001 --device cuda` → `Connected!`,
`[cache] wrote 5 episodes to data/latent_cache/`, `[cache] manifest OK (teacher provenance populated)`.
WorldVLN server logs show K=5 rollouts per episode (vllatent-k0 through vllatent-k4), each with distinct
seeds (timestamps differ), 160/160 denoising steps per rollout at ~3.48 it/s. Five different instructions
processed across 4 distinct episodes (MSBuild2018 + AirSimNH scenes). All 200 OK from the teacher
server. DINOv3 encode + CLIP text + V-JEPA-2 surprise all ran implicitly (the cache wrote successfully
with the full .npz contract).
**End-to-end pipeline confirmed:** AirSim render (480×640) → center-crop+resize (224²) → DINOv3 latents
(196,768 fp16) + CLIP lang_tokens + WorldVLN K=5 rollout → 6→4 projection → V-JEPA-2 surprise → .npz
write per episode + manifest.json with teacher provenance.
**All Phase-A data seams (5/5) now LIVE-verified:** DINOv3 (A5.10) · WorldVLN teacher (A5.11) · V-JEPA-2
verifier (A5.12) · CLIP text (A5.13b) · AirSim render (A5.13) · **cache orchestration (A5.14)**.
**Remaining:** A5.16 (loader inspect over real cache, USER-GATED) → A5.17 (size full job, USER-GATED) →
A5.18 (Phase-A DoD sign-off, USER-GATED).

---

## 2026-06-15 — A5.14: cache orchestration code + mocked test DONE; small-slice USER-GATED (STOP CHECK)
**Status:** A5.14 pending → **in_progress** (orchestration code + mocked `tests/test_cache_manifest.py`
AUTONOMOUS; the small-slice real build is USER-GATED — command block below).
**What's done.** `vllatent/cache.py` — full render→[DINOv3+CLIP-text+WorldVLN+V-JEPA-2]→cache
orchestration:
- **`center_crop_and_resize`** — center-crop to square + resize to 224² via cv2 `INTER_AREA` (lazy cv2);
  normalizes the sim-native `(480,640)` so DINOv3 and V-JEPA-2 see identical pixels (the render-resolution
  foot-gun from A5.13).
- **`build_episode_cache`** — per episode: render all `reference_path` poses → square-crop+resize →
  DINOv3 `encode_rgb` → CLIP `encode` (once) → WorldVLN `k_rollout_segment` → V-JEPA-2
  `scalar_surprise` per transition → assemble the .npz arrays EXACTLY per the A5.15 read-contract.
- **`build_cache`** — multi-episode: writes per-episode `.npz` + `manifest.json` with teacher/render
  provenance fully populated (`worldvln_model_id`, `worldvln_revision`, `render_config_hash`). Resumable:
  skips episodes whose `.npz` already exists.
- **`_teacher_6to4`** — the 6→4 projection: drop roll/pitch, keep x/y/z (m), convert yaw rad→deg; returns
  `(waypoint_4dof, rollpitch_resid)`.
- **`_disagreement_scalar`** — scalarize the `(6,)` rollout spread over the 4 student-relevant channels
  (yaw,x,y,z) via mean.
- CLI `python -m vllatent.cache build --slice ... --out ...` (USER-GATED; full lazy import of all 5 seams).
All heavy imports lazy (`from __future__ import annotations` + `TYPE_CHECKING` for type hints; seam classes
imported inside functions). Pure import-smoke verified.
**`[torch]` extra PINNED** (A5.10 drift resolved): `torch>=2.8,<2.13`, `transformers>=4.56,<6`,
`timm>=1.0.20,<2` in `pyproject.toml`.
**Tested.** `tests/test_cache_manifest.py` (12 tests, PURE — all 5 seams mocked via `MagicMock`):
center-crop-and-resize (480×640 / already-square / tall / bad-input); 2-episode cache build; manifest
valid + teacher provenance populated + correct entry count; per-episode `.npz` keys/shapes/dtypes match
the A5.15 read-contract EXACTLY (latents/actions/deltas/lang_tokens/waypoint_4dof/teacher_pose6/
rollpitch_resid/disagreement/vjepa_surprise); **round-trip through `CachedLatentDataset`** (all 5
transitions access and construct valid `(StepSample, OracleTarget)` pairs); resumable skip-existing;
oracle target non-negativity. `make test` 239→**251** (+12); `make test-torch` 5; ruff/mypy(pure)/
import-smoke/blob clean.
**Open / next — STOP CHECK (small-slice USER-GATED).** The mocked half is done; the real small-slice
build command block:
```bash
# In fly0-m1 docker (UE4 scene hot on :41451) + ssh tunnel to H20 WorldVLN :8001
HF_ENDPOINT=https://hf-mirror.com \
python -m vllatent.cache build \
  --slice data/aerialvln_json/train.slice.json --limit 5 \
  --scenes-root /opt/aerialvln --out data/latent_cache/ \
  --teacher-server http://127.0.0.1:8001 --device cuda
```
Needs: fly0-m1 docker + UE4 scene running + GPU for DINOv3/CLIP/V-JEPA-2 + H20 WorldVLN server (reuse
from A5.11). After the user pastes back, A5.14 flips to `done`. Then A5.16–A5.18 (all USER-GATED).

---

## 2026-06-14 — A5.13 + A5.13b DONE: both real-weight smokes GREEN (user-verified)
**Status:** A5.13 in_progress → **done**; A5.13b in_progress → **done** (the USER ran both smokes and
pasted output — not agent-fabricated). All 5 model/render components of the cache pipeline are now
real-verified: DINOv3 (A5.10) · WorldVLN teacher (A5.11) · V-JEPA-2 verifier (A5.12) · CLIP text (A5.13b)
· AirSim render (A5.13).
**A5.13b text-smoke (cuda, user-pasted).** `python -m vllatent.encode.text --smoke` →
`lang_tokens (10,768) float16`, `[text-smoke] OK`. The 10-token count = BOS + 8 words + EOS for the test
instruction. The `[transformers] ... UNEXPECTED vual_model.*` load report is **benign** — `CLIPTextModel`
loads the full CLIP checkpoint and ignores the vision tower (the report says so). Frozen CLIP text tower
confirmed end-to-end.
**A5.13 live render (fly0-m1, user-pasted).** `bash scripts/render_aerialvln.sh --episode tiny_episode
--scene 1` → `Connected!`, `[render] scene=1 episode=tiny-0001: 8 RGB frames (480,640,3)`. Teleport →
capture → decode confirmed against a real UE4 scene; foot-guns held. **User landed 3 fixes from the smoke
(`7e31bf3`):** (1) camera `front_0`→**`front_center`** (the real name), (2) vehicle `Drone_1`→**`drone_1`**
(case), (3) **arm+takeoff before first teleport + 200 ms settle after each `simSetVehiclePose`** (fly0
pattern; injected-client tests set `_armed=True` so they skip it). Plus the render script now runs
`env PYTHONNOUSERSITE=1 "$PY" -s` to stop a user-site Colosseum `airsim` from shadowing the conda env, and
the CLI gained `--vehicle`. Gates re-confirmed green after these (239 pure / 5 torch).
**⚠ LOAD-BEARING for A5.14 — render resolution.** Frames came back **`(480,640,3)`** (sim-native), NOT the
`224²` the A5.13 DoD assumed. The container's AirSim `settings.json` CaptureSettings weren't 224². DINOv3's
`encode_rgb` would force-`interpolate` 480×640→224² and **distort aspect ratio** (V-JEPA-2's processor
instead resize-shortest-edge + center-crops). So A5.14 MUST normalize at the render→encode boundary —
**center-crop to square then resize to 224²** (consistent for DINOv3 AND V-JEPA-2), or have the operator
set `settings.json` CaptureSettings=224². Record the chosen transform in the provenance manifest
(training-playbook foot-gun #1: log frame transforms). Do NOT let the encoder silently distort.
**Open / next — A5.14 (now UNBLOCKED, pure orchestration).** render→[square-crop+resize→DINOv3 vision +
CLIP text + WorldVLN teacher + V-JEPA-2 verifier]→`.npz` (the A5.15 read-contract) + provenance manifest;
**pin the `[torch]` extra** (drift flagged in A5.10). Autonomous half = orchestration + mocked
`tests/test_cache_manifest.py`; small-slice build USER-GATED. New cold-start handoff written:
`plans/handoff-2026-06-14-resume-ralph-A5.14.md`.

---

## 2026-06-14 — A5.13b (NEW): frozen CLIP text tower → lang_tokens — contract done; smoke USER-GATED
**Status:** new sub-step added + in_progress (contract AUTONOMOUS; real-weight smoke USER-GATED). **Why it
exists:** wiring A5.14 surfaced a gap — the cache contract (A5.15 loader) needs `lang_tokens (M,768) fp16`
from a "frozen text tower (default SigLIP/CLIP-ViT-B, 512→768)" (io-contract §b), but NO A5.x step built
one, so A5.14 could not produce the cache. User chose **CLIP ViT-B/32**; added A5.13b to the plan.
**Gating researched first (DINOv3 lesson).** `openai/clip-vit-base-patch32` is **NON-GATED** (`gated:false`,
15.3M downloads; probed hf-mirror 2026-06-14) ⇒ no token, no re-host fallback.
**What's done.** `vllatent/encode/text.py` — `ClipTextEncoder(model_id,device,dtype).encode(text) → (M,768)
fp16` (M = real tokens, no padding; the loader sets lang_mask all-True so padding would poison M). CLIP
text width is **512**; `_lift_to_embed_dim` zero-pads 512→768 (documented: the meaningful 512→768 map is
the student's LEARNED cross-attention K/V in Phase B; the frozen cache lift is a reproducible placeholder,
trivially swappable). Frozen (eval/no_grad/requires_grad False), lazy torch/transformers in
`_load_backbone`, `--smoke` CLI + `make text-smoke`. `Config.encoder` gains `text_model_id` (non-empty
validated); `build_manifest` records it in the encoder provenance (audit trail, like the DINOv3 model_id).
Recipe validated on a random-weight CLIPTextModel + the production `encode` path (M=real tokens, zero-pad
tail, fp16, frozen).
**Tested.** `tests/test_text_contract.py` (10, PURE — mocked `_load_backbone` seam, no torch/transformers):
zero-pad lift (first 512 = CLIP, rest 0) + pass-through-at-768 + reject-wider/non-2D; encode shape/dtype/
token-count; **feeds `StepSample.lang_tokens`** (constructs a real StepSample); bad-text raises; model-id
single-source; AST heavy-free. `make test` 229→**239**; `make test-torch` 5; ruff/mypy(pure)/import-smoke/
blob clean.
**Open / next — STOP CHECK (real-weight smoke USER-GATED).** `make text-smoke` command emitted. With
A5.13(live) + A5.13b(smoke) both green, **A5.14 is unblocked** (render→[DINOv3+CLIP-text+WorldVLN+V-JEPA-2]
→cache; pin the `[torch]` extra there).

---

## 2026-06-14 — A5.12: V-JEPA-2 surprise verifier — DONE (real-weight smoke GREEN, user-verified)
**Status:** A5.12 pending → in_progress → **done** (contract half AUTONOMOUS; the real-weight smoke was
USER-GATED and the USER ran it — not an agent-fabricated pass). The independent SECOND trust gate (the
first is the A5.11 WorldVLN K-rollout disagreement).
**Live evidence (user-pasted, 2026-06-14).** `python -m vllatent.verify.vjepa2 --smoke --device cuda
--model-id /tmp/vjepa2-weights` (weights pre-downloaded to a local path): **587 tensors loaded**, real
V-JEPA-2 ViT-L encoder→predictor forward on cuda → `surprise [0.17445292, 0.20802556]` (2 future frames,
mean 0.1912) — finite, ∈[0,2], `[vjepa-smoke] OK`. The moderate (not 0, not 2) values on random RGB are
the expected sanity: random context poorly predicts random future, but their encoder reps aren't fully
orthogonal. The full ẑ-vs-z cosine-surprise pipeline is **live-confirmed end-to-end on real weights.**
**Gating researched FIRST (DINOv3 lesson).** Probed hf-mirror + HF API: `facebook/vjepa2-vitl-fpc64-256`
(ViT-L) is **fully NON-GATED** — `gated:false`, `private:false`, **MIT** license, `model.safetensors`
**1.30 GB** (153k downloads). So — unlike DINOv3, where Meta's gated repo rejected us and timm's re-host
saved A5.10 — **no token and no re-host fallback are needed**; loaded straight via `transformers`. (The
`fpc16-256` variant 401s; `vith`/`vitg` are also non-gated but ViT-L is the spec.) Id single-sourced into
`Config.trust.vjepa2_model_id` so the verifier + the A5.14 manifest provenance never drift.
**Recipe (verified first-hand against the installed `transformers/models/vjepa2/modeling_vjepa2.py`, v5.10.2).**
V-JEPA-2 `VJEPA2Model` = encoder + predictor. `model(pixel_values_videos, context_mask=[ctx_idx],
target_mask=[tgt_idx])` where the masks are **lists of LongTensor INDEX tensors** into the patch dim;
`out.predictor_output.last_hidden_state` = **ẑ** (predictor forecast, projected back to `hidden_size`=1024)
and `out.predictor_output.target_hidden_state` = **z** (`apply_masks(encoder_out, target_mask)` — the
encoder's actual latent at the targets) — token-aligned, SAME space ⇒ cosine compares like-with-like.
Tokens are temporal-major (Conv3d flatten): block `[p·256:(p+1)·256]` = temporal slot p (grid 16²=256
tokens/slot, `tubelet_size`=2). To get clean **per-future-frame** surprise we duplicate each logical frame
`tubelet_size`× (`repeat_interleave`) so frame f → exactly one slot; ctx tokens = first C frames, target =
the J future frames; mean-pool each frame's 256 tokens → (J,D), cosine per row. **Recipe validated
end-to-end twice**: (1) a standalone shrunk random-weight `VJEPA2Model`; (2) the PRODUCTION `_forward`
closure via monkeypatched `from_pretrained` (correct (J,) shape, fp32, finite ≥0, params frozen) — the
USER-GATED smoke is the only thing the real ~1.3 GB weights add.
**What's done.** `vllatent/verify/vjepa2.py` — `VJEPA2SurpriseVerifier(model_id,device,dtype)`:
`surprise(context_rgb,future_rgb) → (J,) fp32 ∈[0,2]` + `scalar_surprise → float` (mean; the OracleTarget
feed); pure `cosine_surprise` helper (zero-norm/NaN→neutral s=1, clip [0,2] absorbs float ε only,
float64-accumulate→fp32); RGB-in (render owns BGR→RGB; the encoder must not flip again); lazy
torch/transformers inside `_load_backbone`; `_smoke`/`--smoke` CLI + `make vjepa-smoke`. `Config.trust`
gains `vjepa2_model_id` (non-empty validated). `build_manifest` now RECORDS `vjepa2_model_id` from
Config immediately (it's a fixed config id, like `disagreement_source` — NOT a build-time fact like
`worldvln_revision`/`render_config_hash`, which stay stubbed for A5.14) — complete audit trail per the
review.
**Tested.** `tests/test_verify_contract.py` (12, PURE — the `_load_backbone` seam returns numpy so no
torch/transformers needed, mirroring the WorldVLN-client test, not the torch-tensor DINOv3 test): cosine
identical/orthogonal/opposite + scale-invariance + zero/non-finite-norm neutrality + shape-mismatch raise;
verifier (J,) shape/dtype/range, per-frame independence, OracleTarget feed, RGB pass-through,
bad-frame/dtype/H,W-mismatch/non-ndarray raises, backbone row-count-mismatch raise, model-id single-source,
AST heavy-free purity; cosine clip [0,2] **both-bounds** + near-float-boundary + scalar-range tests (added
per review — a sign-flip/removed-clip regression can't slip through a `>= 0`-only check). `make test`
212→**229**; `make test-torch` 5; ruff/mypy(pure)/import-smoke/blob clean.
**Adversarial panel (3 skeptics: recipe-vs-source / math-vs-test / purity-Py3.9):** 0 CRITICAL, 2 HIGH,
5 MED/LOW. Recipe agent found **no bugs** (masks=index-tensor lists, ẑ=`predictor_output.last_hidden_state`
vs z=`target_hidden_state` like-with-like, temporal-major token→frame mapping, C/J=1 — all verified vs
source). Fixed: both HIGHs (upper-bound test assertions) + the manifest-provenance MED (record id now) +
the float-boundary MED. **Declined (with reason):** tightening `OracleTarget.vjepa_surprise` to `<= 2` —
it's the LOCKED user-approved seam and the bound can't live in the shared `>= 0` loop (`disagreement` is an
unbounded std-spread); the [0,2] bound is enforced at the verifier (which clips), not the generic seam.
**Open / next.** A5.12 done. Remaining Phase-A.5 is the operator block: **A5.13-live** (fly0-m1 docker +
UE4 render on :41451) and **A5.14** (render → [DINOv3 + WorldVLN + V-JEPA-2] → cache + provenance manifest;
**pin the `[torch]` extra THERE** — the drift flagged in A5.10; reuse the H20 WorldVLN server from A5.11).
A5.16–A5.18 follow. The three model wrappers (DINOv3 A5.10 / WorldVLN A5.11 / V-JEPA-2 A5.12) are now all
real-weight-verified, so A5.14 is pure orchestration over verified parts.
**Vault/memory.** V-JEPA-2 NON-GATED + the predictor-recipe facts to be recorded in memory
(`project_latent_pred_arch_locked`); vault arch-design banner update still deferred to the A5.14
cache-contract freeze (no schema change today).

---

## 2026-06-11 — A5.11 COMPLETE: live K-rollout smoke GREEN on the H20 (user-verified)
**Status:** A5.11 in_progress → **done** (the USER ran the Phase-2 stand-up + smoke and pasted the output —
not an agent-fabricated pass).
**Live evidence (user-pasted).** Server up on the H20 AutoDL container (`autodl-container-9ef943a6c4`), env
`worldvln`; weights at `/root/autodl-tmp/WorldVLN` — **actual HF layout:** `WorldVLN_backbone/backbone/`
(4-shard safetensors + index), `WorldVLN_backbone/vae/model.safetensors`, `WorldVLN_action_decoder.pt`
(NOT the `gpt/`+`vae/` hf_repo naming the upstream resolver special-cases — the flat shard dir /
torch_shard path is what matched); T5 at `/root/autodl-tmp/flan-t5-xl`. Health: `infinity_loaded=true`,
`points [1,17,33,49]`, 640×640 tgt — note `ts_ckpt_loaded=false` until the first predict call (stage2
action head initializes LAZILY per mode; expected, not a failure). Smoke (dev box → ssh tunnel :8001):
**K=5 × T=16 actions (segment 0); 5 DISTINCT step-0 rows** (seeds 0, 65537, 131074, 196611, 262148);
step-0 `rollout_spread (6,)` = [0.068, 0.027, 0.050, 0.055, 0.033, 0.091] — **all six channels > 0**;
`rollouts identical across K: False`; `[teacher-smoke] OK`. The trust-oracle disagreement signal is
**live-confirmed end-to-end** (wire → seam (m,rad) → `TeacherOutput.rollout_spread`).
**Cleanup in this commit.** A5.11 → done; committed the A5.8 probe scripts (`scripts/a5_8_worldvln_meta.sh`,
`scripts/a5_8b_worldvln_probe.sh` — referenced by the A5.8 entry, were untracked); deleted the stale,
superseded `plans/handoff-2026-06-08-resume-ralph-A5.4.md` (self-described disposable; A5.4 long done);
new cold-start brief `plans/handoff-2026-06-11-resume-ralph-A5.12.md`.
**Open / next.** Lowest pending = **A5.12** (V-JEPA-2 surprise verifier — contract half autonomous,
real-weight USER-GATED). Then the remaining operator block: A5.13-live (sim) + A5.14 (cache build; pin the
`[torch]` extra there; can reuse this H20 server — weights persist on `/root/autodl-tmp`). A5.16–A5.18 follow.
**Vault/memory.** A5.11-done + H20 server-reuse facts recorded in memory (`project_latent_pred_arch_locked`
+ equipment note); vault arch-design banner update still deferred to the A5.14 cache-contract freeze.

---

## 2026-06-10 — A5.11: WorldVLN teacher CLIENT done (wire-format CORRECTION); live smoke USER-GATED
**Status:** A5.11 pending → in_progress (client/contract half AUTONOMOUS; the live K-rollout smoke is
USER-GATED — server stand-up command block emitted). **Phase 1 (user) delivered:** the rollout-API dump from
the A5.8 clone (`/tmp/worldvln_code` @ `3409b82`, dump `/tmp/worldvln_rollout_api_dump.txt`) — all facts below
re-verified FIRST-HAND against the clone before coding.
**Load-bearing findings (3, from the live API — they refine A5.8/A5.9 notes).**
(1) **Wire action format CORRECTION:** rows are ``[dx_cm,dy_cm,dz_cm,droll_deg,dyaw_deg,dpitch_deg]`` —
position-FIRST, (cm, deg), per-step **DELTAS** (`_to_cm_deg` server.py:349 converts FROM model-native (m,rad)).
The A5.8 note's ``[roll,yaw,pitch,x,y,z]`` is the training-stats/seam order, NOT the wire; and the seam now
carries deltas, not SE(3)-integrated absolutes (offline `predict_pose.py` integrates; we keep raw deltas) ⇒
**A5.14's abs→body-delta projection step simplifies to: drop roll/pitch + rad→deg yaw** (to match the student's
``delta_4dof`` m/deg). `TeacherOutput` docstring updated (shape/validation UNCHANGED — seam not relitigated).
(2) **Seed semantics:** `local_seed = seed + segment_index` UNLESS `lock_seed_across_steps` — and the released
`infer/config.json` sets it **true** ⇒ one session is seed-stable; **K-rollout disagreement = K sessions with
distinct session_id + seed**, spaced by upstream's own `--candidate_seed_stride 65537` (GRPO tool). Stochasticity
itself (cfg34/top_k900/top_p0.97/tau_video0.4) confirmed unchanged.
(3) **Serving:** FastAPI `infer/run_server.sh` → uvicorn :8001; `POST /v1/predict_delta_actions` (1 segment max
per call; `segment_index=-1` = warmup/no segment; `allow_future_segments=true` = strict closed loop: 1 frame+
instruction → 16 actions, +16 real frames → next 16; released 49/16 → points [1,17,33,49] → 3 segments);
`GET /health`; env `INFINITY_CKPT` (server.py:314) + `STAGE2_LATENT2ACTION_CKPT`; single global async lock.
**What's done.** `vllatent/teacher/worldvln.py` — `WorldVLNTeacherClient` (stdlib urllib + numpy; PNG-b64 frame
encode lazily imports cv2/PIL; transport injectable): `health()`, `predict_segment()` (validates response, raises
actionably on warmup/-1, malformed, unreachable-server), `k_rollout_segment()` (K sessions × stride-65537 seeds,
`reset_session=True`, consistent-shape check) → `(K,T,6)` seam (m,rad); `wire_actions_to_pose6` (order remap
(3,4,5,0,1,2) + deg→rad + cm→m); `teacher_outputs_from_rollouts` → per-step `TeacherOutput`. **Three-unit-system
foot-gun documented** (wire cm/deg · seam m/rad · student m/deg-yaw). Live CLI `python -m vllatent.teacher.worldvln
--episode … --rollouts K --server …` (health + K-rollout + spread + identical-rollouts FAIL check). Upstream clone
never imported/modified. Py3.9-pure-box gotcha fixed (runtime `X | None` in a type alias → TYPE_CHECKING).
**Tested.** `tests/test_teacher_contract.py` (12 tests, MOCKED transport, pure gate): wire→seam order+units
(hand-computed expectations), payload/route fidelity, K distinct sessions+seeds+instruction+reset, (K,T,6)
stacking + per-step spread channel-correctness (dx-varies ⇒ seam[3]>0, yaw 0), warmup −1 raise, malformed raise,
inconsistent-shape raise, Config-default K, health GET, heavy-free AST+sys.modules guard. A **3-skeptic
adversarial panel** (protocol-fidelity vs the clone / conversion-math hand-recomputed / robustness+mutation)
returned 2 holds + 1 holds-with-caveat; the caveat (frame validation unreachable behind the mocked encoder) is
fixed with an unmocked bad-input test. `make test` 199→**212**; `make test-torch` 5;
import-smoke/lint/typecheck(pure)/blob clean.
**Open / next — STOP CHECK (A5.11 live smoke is USER-GATED).** Phase-2 command block emitted (weights download →
`run_server.sh` on the GPU box → `GET /health` → the CLI smoke; paste output to flip A5.11 done). Then A5.12
(V-JEPA-2 verifier). [torch]-extra pin still queued for A5.14.
**Vault.** Wire-format correction + seed-locking recorded in memory (`project_latent_pred_arch_locked`); vault
arch-design update deferred to the A5.14 cache-contract freeze (no schema change today).

---

## 2026-06-09 — A5.13: render harness — MOCK unit half done; live render USER-GATED (STOP CHECK)
**Status:** A5.13 pending → in_progress (mock unit half AUTONOMOUS; live render USER-GATED per ralph-rules —
command block emitted). Front-loaded per the operator's autonomous-first choice.
**Reference discipline.** Per the operator: the AirSim API is COPIED from the end-to-end pipeline
`CODE/vln-ego-drone/fly0-style-pipeline` (`sim/airsim_client.py`) + `third_party/AirVLN` — re-derived into
THIS repo (Phases A–C are standalone, fly0 is NOT imported; we copy the semantics + unit-test them). Ground
truth confirmed: `airsim.Quaternionr(x,y,z,w)` is **xyzw**; `simSetVehiclePose(pose, ignore_collision=True,
vehicle_name='Drone_1')`; `simGetImages([ImageRequest('front_0', ImageType.Scene, False, False)], …)`; Scene
buffer is **4-channel BGRA** → `[:,:,:3]` (BGR) → reverse to RGB; teleport-only needs just `confirmConnection`.
**What's done.** `vllatent/render/harness.py` — `RenderHarness.teleport(pos_ned, yaw)` builds the pose with a
yaw-only **xyzw** quaternion via `frames.xyzw_from_yaw` (== `airsim.to_quaternion(0,0,yaw)`; foot-gun #1);
`capture_rgb()` requests the Scene camera and `decode_scene_to_rgb` drops the BGRA alpha + reverses BGR→RGB
(foot-gun #2) to `(H,W,3)` uint8; `render_reference_row(row6)` does teleport+capture for one Euler row. **Every
`client.X()` is wrapped in one `threading.Lock`** (foot-gun #3 — single-threaded msgpack-RPC). `airsim` import
is LAZY (`_connect`); module imports airsim+cv2-free (no resize here — the DINOv3 processor resizes to 224²,
and the harness owns the BGR→RGB flip so the encoder uses `encode_rgb`, no double-flip). Added the USER-GATED
live CLI `python -m vllatent.render --episode … --scene 1 --out …`.
**Tested.** `tests/test_render_unit.py` (9 tests, MOCKED airsim + fake client — runs in the PURE gate, not
`@pytest.mark.sim`): BGRA→RGB decode (+3-channel + size-mismatch), the yaw→xyzw quaternion, Scene/camera/vehicle
request args, **every client call asserted under the lock**, row-width validation, and an airsim-free import
guard. `make test` 190→**199** (+9); `make test-torch` 5; `make import-smoke`/`lint`(ruff)/`typecheck`(mypy, 6
pure files) clean; blob-guard OK.
**Open / next — STOP CHECK.** A5.15 + A5.13 (autonomous-first batch) done/in_progress + verified; pushing.
The user-gated infra block remains: **A5.11** (WorldVLN-8B teacher; needs the live `infer/server.py` re-probe),
**A5.12** (V-JEPA-2), **A5.13 live render**, **A5.14** (cache build, + pin the `[torch]` extra) — one operator
session. The live-render command block: `python -m vllatent.render --episode fixtures/episodes/tiny_episode.json
--scene 1 --out /tmp/render_smoke/` inside fly0-m1 with the UE4 scene hot on :41451.
**Vault.** No new decision (implements the A5.13 render harness per the signed-off re-plan; AirSim API copied
from the fly0 end-to-end pipeline per the operator's reference-first instruction).

---

## 2026-06-09 — A5.15: distillation loader (StepSample+OracleTarget over the latent cache) — AUTONOMOUS
**Status:** A5.15 pending → done (AUTONOMOUS; TORCH tier but numpy-only emission, no user gate). Front-loaded
per operator choice (autonomous-first; the user-gated WorldVLN/V-JEPA/render/cache block A5.11–A5.14 is batched
for a later operator session).
**What's done.** `vllatent/data/loader.py` — `CachedLatentDataset`, a map-style Dataset emitting the A5.9
per-step distillation pair `(StepSample, OracleTarget)` over a render-once latent cache. It **defines the
per-episode `.npz` read-contract** (latents (N,196,768) fp16 / actions / deltas / lang_tokens / the 5
OracleTarget arrays) + `manifest.json` that **A5.14 will write to**. History = a left-zero-padded block-causal
window ending at t (mask True=real); the terminal STOP (t=N−1, no `z_next`) is excluded → `len = Σ(N_e−1)`.
H/T read from Config; **H is pinned to the arch-locked schemas `HISTORY`** (StepSample fixes that window) — a
divergent override fails fast at construction with a clear message rather than a deep StepSample error.
Numpy-only emission (the typed numpy contract objects, validated per sample) ⇒ imports torch-free; torch enters
only at DataLoader collation (Phase B). Added `python -m vllatent.data inspect --cache <dir> --n N` (A5.16's
real-dump inspector).
**Tested.** `tests/test_data_shapes.py` over a synthetic tiny_dump (tmp, no blobs): len/episode-count, sample
shapes+dtypes, block-causal padding at episode start (`[F,F,T]`) + full window mid-episode (`[T,T,T]`),
cross-episode index routing, the H-lock fail-fast, manifest validity, the `inspect` CLI, **and a real
`torch.utils.data.DataLoader` batch** (`@pytest.mark.torch`). `make test` 182→**190** (+8 pure); `make
test-torch` 4→**5**; `make import-smoke`/`lint`(ruff)/`typecheck`(mypy, 6 pure files) clean; blob-guard OK.
**Open / next.** A5.13 (render harness — autonomous mock unit half; live render USER-GATED) next this batch,
then STOP CHECK → the user-gated A5.11/A5.12/A5.14 infra block.
**Vault.** No new decision (implements the A5.15 loader + defines the cache read-contract per the signed-off re-plan).

---

## 2026-06-09 — A5.10 COMPLETE: encoder swapped to NON-GATED timm DINOv3; real-weight encode-smoke GREEN
**Status:** A5.10 in_progress → **done** (real-weight smoke ran GREEN live this session; user driving as operator,
session "manual operator" — not an agent-fabricated pass).
**Blocker hit + resolved.** Meta's gated `facebook/dinov3-vitb16-pretrain-lvd1689m` **rejected** account `nakacc`'s
license request (HTTP 403 "rejected by the repo's authors" on file resolve; token valid via `whoami`, proxy/mirror
fine — pure gating, not network). Switched the frozen encoder to timm's **non-gated re-host**
`vit_base_patch16_dinov3.lvd1689m` (HF `timm/…`) — **same Meta LVD-1689M ViT-B/16 weights**, verified 85.6M params /
`prefix_tokens=5` / `forward_features (1,201,768)` → trailing-196 = **(196,768)** (locked spec). Non-gated = strictly
**more reproducible for the paper** (no per-reviewer gate request).
**Changes.** `encode/dinov3.py` `_load_backbone`: `transformers.AutoModel/AutoImageProcessor` → `timm.create_model(…,
num_classes=0)` + `resolve_model_data_config` manual normalize (pure-torch, no PIL/torchvision); identical `(1,T,768)`
closure contract, so the monkeypatched contract test is untouched. `config.EncoderConfig.model_id` → timm id
(auto-propagates to manifest provenance). Docstrings/comments + Makefile help/encode-smoke comment de-gated.
`test_config` default-id expectation updated.
**Verified.** `make encode-smoke` → `(196,768) float16 cuda` OK (NO token); `make test-torch` 4 passed; `make test`
182 passed; ruff + mypy(pure tier) clean.
**Env.** Created conda env `vllatent-ego-drone` (Py3.10) — the project torch env (the A5.8 command template's path was
a placeholder; it did not exist). ⚠ **Version drift:** the `[torch]` extra's unbounded lower bounds pulled
**transformers 5.10.2 / torch 2.12.0+cu130** (NOT the spec'd ≥4.56 / ≥2.8-cu12x) — works, but diverges from the
H20-cu12x train env + Jetson deploy parity; **pin `[torch]` before the A5.14 cache run.**
**Open:** (1) user to rotate the HF token shared in plaintext this session; (2) decide `[torch]` version pins.

---

## 2026-06-09 — A5.10: DINOv3 student-encoder wrapper (TORCH tier) — CONTRACT done; real-weight USER-GATED (STOP CHECK: tier boundary)
**Status:** A5.10 pending → in_progress (CONTRACT half AUTONOMOUS; the real-weight encode-smoke is USER-GATED per
ralph-rules — command block emitted, awaiting the user's paste). **First TORCH-tier step = the pure→torch boundary.**
**What's done.** `vllatent/encode/dinov3.py` — frozen DINOv3 ViT-B/16 wrapper (the student's FROZEN cached front-end
encoder, **NOT** the student): `encode_bgr` enforces foot-gun #2 (BGR→RGB flip at the render→encode boundary via the
pure-numpy `bgr_to_rgb`) then `encode_rgb` runs the backbone, takes the **TRAILING `PATCH_TOKENS` (196)** tokens
(robust to the `[CLS]`+4-register prefix — research-confirmed DINOv3 layout = 201 tokens @ 224²), casts fp16, and
`_validate_latent` pins `(196,768)` fp16 (== `LATENT_DTYPE`). **torch/transformers imports are LAZY** (`_load_backbone`
+ methods; `if TYPE_CHECKING: import torch` only) so a torch-free box imports the module (verified: `torch` NOT in
`sys.modules` post-import). Model id single-sourced from `config.EncoderConfig` (corrected the placeholder
`facebook/dinov3-vitb16` → the real **gated** id `facebook/dinov3-vitb16-pretrain-lvd1689m`; DINOv3 license, needs
HF_TOKEN — surfaced as an actionable load error). Added the USER-GATED `make encode-smoke` target.
**Tested.** Contract (AUTO): `tests/test_encode_contract.py` (4 tests, `@pytest.mark.torch`, **monkeypatched backbone
— NO weights**): BGR→RGB reaches the backbone, `(196,768)` fp16 output, trailing-196 drops CLS+registers, bad-input
rejection → `make test-torch` **4 passed** (torch present in the pure env). Pure gate UNCHANGED-GREEN: a torch-FREE
guard in `tests/test_smoke.py` (imports the module + AST-checks no module-level heavy import) lands in `make test`
181→**182**; `make import-smoke`/`lint`(ruff)/`typecheck`(mypy, 6 pure files) clean; blob-guard OK. An **adversarial
4-skeptic verify panel** (tier-purity / contract-fidelity / real-weight-path / DoD) ran the gates + refute-attempts:
**3 holds**; the 4th's issues fixed here (removed dead `return 2`; added the gated-weights HF_TOKEN load error;
clarified the always-fp16 cache comment) — its only remaining flag was "test file untracked" = closed by this commit.
**Open / next — STOP CHECK (pure→torch tier boundary + A5.10 real-weight is USER-GATED).** Emitting the encode-smoke
command block (`HF_TOKEN=… HF_ENDPOINT=https://hf-mirror.com make encode-smoke`; expect `latent (196,768) float16`)
— A5.10 flips to `done` when the user pastes it. Next pending = **A5.11** (frozen WorldVLN teacher wrapper, TORCH,
USER-GATED server) → A5.12 (V-JEPA-2) → A5.13 (render harness).
**Vault.** No new decision (implements the A5.10 DINOv3 wrapper per the signed-off re-plan; the gated-license weights
track is already recorded in the A5.8 entry).

---

## 2026-06-09 — A5.7 real-slice VERIFIED → done; WorldVLN license email SENT
**Status:** A5.7 in_progress → **done** (user pasted the real-slice `AuditSummary`). License-clarification
email to the WorldVLN authors **SENT** (the `license:other` weights track — runs parallel to development;
blocks publication, not Phase-A plumbing).
**Real-slice result (user-run `--slice data/aerialvln_json/train.slice.json --summary -`).** `n_episodes 50,
n_ok 50, n_transitions 10198, total_delta_mismatches 0, all_action_classes_present true` (counts per id
0..7 = 50/5055/1120/936/1323/1329/239/196), `scene_ids` = 14 distinct ∈ [1,26], `n_reorder_consistent 50,
n_naive_would_mismatch 34, splits ["train"], ok true`. **Reproduces step-5b exactly** — the `AuditSummary`
aggregator is now the first-class, committed replacement for 5b's ad-hoc script (M3 closed end-to-end).
**Open / next.** Phase-A pure+contract lane is now fully green through A5.9; A5.7 closed. Next = **A5.10**
(DINOv3 student-encoder wrapper) — the first **TORCH-tier** step (pure→torch transition; needs the
`vllatent-ego-drone` Py3.10+torch env, NOT the pure `vln-ego-drone-1.1`); contract test autonomous,
real-weight encode-smoke USER-GATED. Cold-start brief: `plans/handoff-2026-06-09-resume-ralph-A5.10.md`.
**Vault.** No new decision (verification + the license-track is recorded in the A5.8/A5.9 entries).

---

## 2026-06-09 — A5.9: TeacherOutput/OracleTarget distillation seam + Config finalize (STOP CHECK)
**Status:** A5.9 pending → done (AUTONOMOUS, pure-tier; user-approved seam shape + license-track decision).
**What's done.** Typed the teacher→student distillation seam (the contract Phase-B trains against),
reflecting the A5.8 findings + the user's seam decision ("4-DoF + raw 6-DoF + residual"). `vllatent/schemas.py`:
`TEACHER_DOF=6`; `TeacherOutput.rollouts_pose6 (K,6) float` (K stochastic WorldVLN rollouts —
[roll,yaw,pitch,x,y,z]; A5.8: stochastic-by-default ⇒ K-rollout disagreement is free) + `rollout_spread()`
= per-DoF std `(6,)` (the raw disagreement; A5.14 scalarizes over yaw,x,y,z; Phase-C calibrates);
`OracleTarget` = per-step target paired 1:1 with `StepSample` = `{waypoint_4dof (4,) f32` [6→4-projected:
drop roll/pitch + abs→body-delta, executed in A5.14]`, teacher_pose6 (6,)` provenance`, rollpitch_resid`
[≈0 lossless-projection audit]`, disagreement` [≥0 spread]`, vjepa_surprise` [≥0 independent gate]`}` — all
scalars finite/bool/dtype-validated (applies the A5.5-review `np.isfinite` lesson). **Config finalized:**
the A5.3 trust placeholders are no longer provisional — A5.8 confirmed `disagreement_source="worldvln_rollout"`
(stochastic ⇒ rollout spread free; `airscape_multiseed` kept as contingency); `TrustConfig` + module
docstrings updated. `docs/io-contract.md` §0 documents the teacher seam. **License decision (user):** build
seams now / email WorldVLN authors re: the `license:other` weights in parallel (Phase-A plumbing is
license-agnostic).
**Tested.** A5.9 step command `pytest tests/test_schemas.py tests/test_config.py` (89) green; full pure
sweep `make test` 167→**181** (+14: TeacherOutput valid + rollout_spread channel-correctness + bad-input;
OracleTarget valid + 8 bad-input rejections; Config finalized values); `make import-smoke`/`lint`/`typecheck`
(mypy, 6) clean; blob-guard OK. A **focused adversarial review** (1 agent, ran the gate + adversarial REPL)
returned CLEAN — order/index ([roll,yaw,pitch,x,y,z], axis=0), validation completeness (f32+f64 NaN/inf/bool),
mutation-sensitivity, tier purity, exact 5-field shape all verified.
**Open / next — STOP CHECK (started_step+3: A5.7/A5.8/A5.9; next step is a tier change + user-gated).**
Pushing. Still open: **A5.7 real-slice paste** (flips A5.7 done). Next pending = **A5.10** (DINOv3
student-encoder wrapper) — TORCH tier (lazy import; contract AUTO, real-weight USER-GATED) — a tier
transition from pure→torch, so it pauses here. Author-clarification email for the weights license drafted
for the user.
**Vault.** No new decision (implements the A5.8-informed teacher seam + the user-approved shape per the
signed-off re-plan).

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
