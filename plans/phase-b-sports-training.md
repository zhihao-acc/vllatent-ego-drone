# Phase B: Sports-Following Training Plan

> **Created 2026-06-19.** This is the authoritative Phase B plan. Supersedes
> `plans/phase-a5-replan-postpivot.md` (WorldVLN-centric) and
> `plans/phase-b1-sports-data-pipeline.md` (ingest-only scope). Phase A is complete (253 pure +
> 5 torch tests green, ingest pipeline built with 239 tests).

## Context — why this exists

The vllatent-ego-drone project pivoted 2026-06-19 from indoor AerialVLN to **autonomous
sports-following drone** (skiing primary). Three research reports surfaced blockers that
invalidate large portions of the existing WorldVLN-centric plan:

1. **No teacher for L_kd** — TrackVLA (CoRL 2025) is unreleased; WorldVLN is wrong task domain.
2. **YouTube skiing FPV is 3-5x smaller than assumed** — 30-80 hrs, not 100-500.
3. **Pipeline bugs** — `batch_undistort()` unwired, GPS stubbed, MegaSaM confidence `np.ones()`.

Two simplifications emerged:

4. ViT-B/16 may be fast enough on Orin NX — skip CosPress distillation.
5. Language cross-attention is cheap (+2.76M, <0.5ms).

**User requirements:** (a) history latents = GT from camera, not predicted (B-1); (b) data
quality must be visible/verifiable; (c) fixed clip length (e.g. 10s) matching inference horizon;
(d) YouTube data properly implemented alongside CosFly; (e) clear B-1 scope.

---

## Phase A5 Step Audit

| Step | Verdict | Rationale |
|---|---|---|
| A5.1-A5.7 (pure I/O contracts) | **SURVIVES** | Data-source agnostic. Schemas, frames, config, manifest, audit all valid. |
| A5.8 (WorldVLN investigation) | **INVALIDATED** | WorldVLN retired. Findings are historical record only. |
| A5.9 (TeacherOutput/OracleTarget) | **NEEDS REVISION** | `TeacherOutput` retired. `OracleTarget` needs slimming: drop `teacher_pose6`, `rollpitch_resid`, `disagreement` (WorldVLN fields). Keep `waypoint_4dof` (from MegaSaM) + `vjepa_surprise` (Phase C). |
| A5.10 (DINOv3 encoder) | **SURVIVES** | Works for both ViT-B/16 and ViT-S/16 via timm. |
| A5.11 (WorldVLN teacher) | **INVALIDATED** | WorldVLN retired. `vllatent/teacher/worldvln.py` is dead code — keep as historical, don't import/test. |
| A5.12 (V-JEPA-2 verifier) | **REMOVED** | Trust/verifier mechanism deleted 2026-06-25 (commit `125576f`); `vllatent/verify/` gone. |
| A5.13 (render harness) | **SURVIVES (Phase D only)** | Not used for sports FPV data. |
| A5.13b (CLIP text tower) | **SURVIVES** | Feeds language cross-attention in B-2. |
| A5.14 (render→cache) | **NEEDS REVISION** | AerialVLN cache pipeline replaced by ingest pipeline (`vllatent/ingest/`). `cache.py` kept for historical AerialVLN compat only. |
| A5.15 (loader) | **NEEDS REVISION** | `CachedLatentDataset` structure survives but needs a sports-specific loader with GT history latents, sliding windows, and no OracleTarget dependency. |
| A5.16-A5.17 (loader verify + sizing) | **NEEDS REVISION** | Must re-run on sports ingest cache, not AerialVLN cache. |
| A5.18 (Phase A DoD) | **SURVIVES as record** | Phase A complete. Not re-opened. |
| Config defaults | **NEEDS REVISION** | `disagreement_source="worldvln_rollout"` invalid. `PredictorConfig` depth/heads tied to D=768 which may change. Add `clip_length_seconds` to IngestConfig. |

---

## Phase B-1 Scope (explicit)

**B-1 covers:** Pipeline fixes, schema revision, encoder decision gate, data acquisition
(CosFly-Track + YouTube pilot), data quality dashboards, cache pipeline for sports, sliding-window
loader with GT history latents, predictor + waypoint head, L_latent + L_wp training with
overfit-tiny-batch, full training run on available data.

**B-1 does NOT cover:** Language cross-attention (B-2), auto-captioning (B-2), RefCOCOg
pre-train (B-2), Ego-Exo4D integration (B-3), scheduled sampling (B-2/B-3),
closed-loop deployment (Phase D), CosPress distillation training (only if
encoder gate says ViT-B/16 is too slow — and even then, use Meta's pre-distilled ViT-S/16
from timm first).

---

## Phase B-1 Ordered Steps

`PY` = python interpreter in vllatent-ego-drone env. Tier in {PURE, TORCH, ORCH, RESEARCH, DOC}.
Gate in {AUTO, USER-GATED}. Ralph rules: lowest pending first, pure-tier/fixtures-first,
STOP CHECK at `started_step + 3`, user-gated stays `in_progress`.

---

### Group 0 — Pipeline Bug Fixes + Config/Schema Revision

**B1.1 — Wire `batch_undistort()` into `pipeline.py`.**
- Tier PURE+TOOL / AUTO
- `preprocess.py:123` defines `batch_undistort()` but `pipeline.py` never calls it. Wire between
  frame extraction (stage 2) and quality scoring (stage 3), gated by `IngestConfig.undistort_model`
  (`"pinhole"` = skip, `"fisheye"` = run with K/D from clip YAML entry).
- **Files:** `vllatent/ingest/pipeline.py` (wire call), `vllatent/ingest/preprocess.py` (no change)
- **DoD:** `process_clip()` conditionally calls `batch_undistort()`. Test covers the wiring path.
- **Test:** `$PY -m pytest -q tests/test_ingest_pipeline.py tests/test_ingest_preprocess.py`
- **Deps:** blocks B1.7. Blocked-by: none.

**B1.2 — Fix MegaSaM confidence `np.ones()` fallback.**
- Tier PURE / AUTO
- `megasam.py` lines 103, 116, 124 default to `np.ones()`. Add `confidence_source` field to
  `MegaSamResult` (`"real"` or `"default"`). Log warning on fallback. Flag in quality report.
- **Files:** `vllatent/ingest/megasam.py`, `tests/test_ingest_megasam.py`
- **DoD:** `MegaSamResult.confidence_source` field. Warning logged on fallback.
- **Test:** `$PY -m pytest -q tests/test_ingest_megasam.py`
- **Deps:** none. Blocked-by: none.

**B1.3 — Stub GPS Sim(3) alignment with clear interface in `ego_motion.py`.**
- Tier PURE / AUTO
- Currently just a comment. Define `align_to_gps(poses, gps_track) -> AlignmentResult` with
  proper type annotations, docstring documenting the Umeyama Sim(3) algorithm, and
  `raise NotImplementedError("GPS Sim(3) alignment deferred to custom GoPro+IMU data phase")`.
  `normalize_scale(mode="median_speed")` remains the active path for YouTube + CosFly.
- **Files:** `vllatent/ingest/ego_motion.py`
- **DoD:** Function signature + docstring + NotImplementedError. No test needed (it raises).
- **Test:** `$PY -m pytest -q tests/test_ingest_ego_motion.py` (existing tests stay green)
- **Deps:** none. Blocked-by: none.

**B1.4 — Add fixed-clip-length cutting to `IngestConfig` + `preprocess.py`.**
- Tier PURE / AUTO
- User requires clips pre-cut to a fixed configurable length (default 10s). Add
  `clip_length_seconds: float = 10.0` to `IngestConfig`. Add `cut_fixed_clips()` in
  `preprocess.py` that splits extracted frames into non-overlapping segments of exactly
  `clip_length_seconds * target_fps` frames. Trailing segments shorter than `(HISTORY + HORIZON)`
  frames are discarded. Wire into `pipeline.py` after frame extraction.
- **Files:** `vllatent/config.py` (IngestConfig), `vllatent/ingest/preprocess.py`, `vllatent/ingest/pipeline.py`
- **DoD:** `IngestConfig.clip_length_seconds` validated. `cut_fixed_clips()` splits frames. Pipeline uses it.
- **Test:** `$PY -m pytest -q tests/test_ingest_preprocess.py tests/test_config.py`
- **Deps:** blocks B1.9 (loader). Blocked-by: none.

**B1.5 — Revise Config for sports pivot.**
- Tier PURE / AUTO
- (a) Add `"megasam_vo"`, `"vjepa_only"` to `DISAGREEMENT_SOURCES`; default to `"vjepa_only"`.
  (b) `PredictorConfig` defaults stay depth=12/heads=12 but validator allows D=384-compatible
  combos (heads must divide embed_dim). (c) Add `lambda_trust: float = 0.0` to `DistillConfig`
  (Phase C activates it). (d) Create `configs/sports.yaml` with ingest section.
- **Files:** `vllatent/config.py`, `configs/sports.yaml`, `tests/test_config.py`
- **DoD:** All config changes land. 253+ pure tests green.
- **Test:** `$PY -m pytest -q tests/test_config.py && make import-smoke && make typecheck`
- **Deps:** blocks B1.9 (loader config). Blocked-by: none.

**B1.6 — Create `SportsTarget` in `schemas.py` (slim OracleTarget).**
- Tier PURE / AUTO
- `OracleTarget` has WorldVLN fields (`teacher_pose6`, `rollpitch_resid`, `disagreement`).
  Create a slim `SportsTarget` with just `waypoint_4dof (4,) f32` + `vjepa_surprise: float = 0.0`.
  Sports loader emits `(StepSample, SportsTarget)`. Existing `OracleTarget` unchanged (AerialVLN
  compat). Type alias `Target = OracleTarget | SportsTarget`.
- **Files:** `vllatent/schemas.py`, `tests/test_schemas.py`
- **DoD:** `SportsTarget` validated. Tests pass. `__all__` updated.
- **Test:** `$PY -m pytest -q tests/test_schemas.py`
- **Deps:** blocks B1.9 (loader). Blocked-by: none.

---

### Group 1 — Data Acquisition + Quality

**B1.7a — Create `vllatent/encode/batch.py` (batch DINOv3 encoding).**
- Tier TORCH / AUTO
- `pipeline.py:93` imports `from vllatent.encode.batch import encode_frames` which does not
  exist. Create `vllatent/encode/batch.py` with `encode_frames(frames_dir, device) → (N, 196,
  768) fp16` that iterates over sorted JPEGs in a directory and encodes each via
  `DinoV3Encoder.encode_rgb()`. Lazy torch imports (tier rule).
- **Files:** `vllatent/encode/batch.py` (new), `tests/test_encode_batch.py` (new)
- **DoD:** `encode_frames()` returns stacked `(N, PATCH_TOKENS, EMBED_DIM)` fp16 latents.
  Test covers shape/dtype contract with monkeypatched encoder (no real weights).
- **Test:** `$PY -m pytest -q tests/test_encode_batch.py`
- **Deps:** blocks B1.7. Blocked-by: none.

**B1.7c — Segment-level FPV extraction + rework pilot script.**
- Tier ORCH / AUTO
- The current pilot script does whole-video ACCEPT/REJECT but never extracts FPV-only
  segments or cuts 10s clips. Add `extract_fpv_ranges(shots) → list[tuple[int,int]]` to
  `content_filter.py` that merges consecutive FPV shots into contiguous frame ranges. Rework
  `scripts/ingest_youtube_pilot.py` to: (1) download full video with SponsorBlock, (2) extract
  all frames, (3) run content filter → get FPV mask + shot list, (4) extract contiguous FPV
  ranges, (5) within each FPV range cut 10s clips via `cut_fixed_clips()`, (6) run each clip
  through the pipeline (quality → MegaSaM → DINOv3 → .npz cache). Pipeline uses per-clip
  sub-directories (`{clip_id}_fpv{range_idx}_clip{clip_idx}`).
- **Files:** `vllatent/ingest/content_filter.py` (add `extract_fpv_ranges`),
  `scripts/ingest_youtube_pilot.py` (rework), `tests/test_content_filter.py` (add tests)
- **DoD:** `extract_fpv_ranges()` tested. Pilot script handles PARTIAL videos (FPV-only
  segments extracted). 10s clip cutting integrated.
- **Test:** `$PY -m pytest -q tests/test_content_filter.py`
- **Deps:** blocks B1.7. Blocked-by: B1.7a, B1.7b.

**B1.7 — YouTube pilot: curate 10-15 skiing FPV clips + run full ingest.**
- Tier ORCH / USER-GATED
- Run the reworked `scripts/ingest_youtube_pilot.py` which: downloads with SponsorBlock,
  runs content filter, extracts FPV-only segments, cuts 10s clips, and runs each through
  the full pipeline (quality → MegaSaM → DINOv3 → .npz cache). Exercises the complete
  chain including B1.1 undistort fix, B1.4 clip cutting, B1.7a batch encoding, and B1.7c
  segment-level FPV extraction.
- **Files:** `configs/sports_clips.yaml`
- **DoD:** 10+ clips downloaded, sponsor segments stripped, content-filtered (FPV segments
  extracted), cut to 10s clips, quality-scored, MegaSaM-processed, DINOv3-encoded, cached
  as `.npz`. Manifest validates. Per-clip quality stats + thumbnail grids reviewed.
- **Test:** User runs pipeline + reviews outputs + reviews content filter thumbnail grids.
- **Deps:** blocks B1.8, B1.10. Blocked-by: B1.1, B1.2, B1.4, **B1.7a, B1.7b, B1.7c**.

**B1.7b — Content filter implementation (motion + YOLO-World).**
- Tier TORCH / AUTO
- `vllatent/ingest/content_filter.py`. Two-signal FPV filter:
  - **Motion** (primary): frame-to-frame mean absolute pixel difference. Threshold >= 8.0 at
    5 fps. Catches static/product shots, talking heads, title screens.
  - **YOLO-World** (semantic): open-vocabulary object detection via `yolov8s-worldv2.pt` (13M
    params, 74 FPS on V100). Detects objects that should never appear in FPV training data:
    drone body + parts (rotors, propellers, arms, landing gear, gimbal), cameras, tripods,
    gear, text overlays, logos, watermarks. Text embeddings computed once via `set_classes()`
    and cached — no per-frame re-encoding.
  - Per-frame decision: `is_fpv = motion >= threshold AND no_rejected_objects(YOLO)`
  - **Minimum segment filter**: after per-frame masking, contiguous accepted runs shorter
    than `min_segment_frames` (default 10 = 2s at 5fps) are discarded. Prevents tiny
    fragments between rejected regions from leaking into training data.
  - PySceneDetect `AdaptiveDetector(adaptive_threshold=2.0)` for shot boundary detection (SBD).
    **Threshold tuned in B1.10f** — default 3.0 missed obvious hard cuts in skiing footage
    (high natural frame variation makes cut spikes blend in). 2.0 catches all real camera
    switches with no false positives on pilot data.
  - Per-shot majority vote → ACCEPT (>=60% FPV) / REJECT (<30%) / PARTIAL
  - Thumbnail grid generator for human review
- **CLIP DROPPED (B1.7c finding):** CLIP zero-shot scores 0.999 on all frames within the same
  visual domain (all skiing = snow+mountain). Zero discriminative power for within-domain
  filtering. CLIP ignores prepositions (ARO/WinoGround ICLR 2023: 0.50-0.56 accuracy on
  compositional benchmarks). Replaced by YOLO-World object detection.
- **New dependencies:** `ultralytics>=8.2.0` (in `[torch]` extra), `scenedetect`
- **Files:** `vllatent/ingest/content_filter.py`, `tests/test_content_filter.py`
- **DoD:** Filter runs on sample video. 36 tests green. All imports lazy (AST-verified).
- **Test:** `$PY -m pytest -q tests/test_content_filter.py`
- **Deps:** blocks B1.7. Blocked-by: none.

**B1.8 — CosFly-Track download + adapter.** ✅ DONE 2026-06-24 (DESCOPED)
- Tier TOOL+PURE / USER-GATED (HF download)
- Download 526-trace subset from `AutelRobotics/CosFly`. Adapter converts CosFly
  format (CARLA, GT 6-DoF, 2 Hz) to ingest `.npz` cache with `motion_method="cosfly_gt"`.
- **DESCOPED:** RGB frames skipped (119 GB). Trajectory-only (6 GB). No DINOv3 encoding.
  See **CosFly Suitability Assessment** below for full rationale.
- **Files:** `vllatent/ingest/cosfly_adapter.py`, `scripts/download_cosfly.sh`, `tests/test_cosfly_adapter.py` (21 tests)
- **Status:** Adapter code complete. Download script functional. Integration into training
  loop **deferred** — CosFly contributes to L_wp only (no latents), and two issues must be
  resolved before integration: (1) world-frame vs body-frame deltas, (2) scale mismatch
  (metric vs normalized). See assessment below.
- **Deps:** ~~blocks B1.10, B1.12~~ No longer blocking. Blocked-by: none.

---

**B1.8b — Pipeline redesign: quality-gate before MegaSaM + per-segment processing.** ✅ DONE 2026-06-24 (HOTFIX)
- Tier PURE+ORCH / AUTO
- **Problem:** `process_clip()` runs MegaSaM and DINOv3 on ALL frames including quality-
  rejected ones. `quality_mask` is passive metadata, never a gate. Bad frames produce
  unreliable VO poses and waste compute on encoding. Also, `extract_fpv_ranges()` was
  merging consecutive FPV shots across editing cuts (FIXED 2026-06-24).
- **Corrected pipeline flow per sub-clip:**
  ```
  Stage 1-2:  Download + extract frames (unchanged)
  Stage 2b:   Undistort (unchanged)
  Stage 3:    Quality score ALL frames → quality_mask
  Stage 3b:   Find contiguous accepted segments from quality_mask (NEW)
              - min segment length = HISTORY + HORIZON + 1 = 8 frames
              - segments shorter than this → discarded (can't produce a training sample)
              - if NO segments survive → reject clip, return early
  Stage 4-7:  For EACH accepted segment:
              - Copy/symlink segment frames to segment sub-directory
              - MegaSaM on segment (continuous, no temporal gaps)
              - SE(3) → body-frame deltas
              - DINOv3 encode segment frames only
              - Cache as separate .npz (segment_id = {clip_id}_seg{N:02d})
  ```
- **Key changes:**
  1. `quality.py`: add `find_accepted_segments(mask, min_length) → list[(start, end)]`
  2. `pipeline.py`: `process_clip()` returns `list[ClipPipelineResult]` (one per segment)
  3. `content_filter.py`: `extract_fpv_ranges()` no longer merges shots (already fixed)
  4. `ingest_youtube_pilot.py`: handle list of results per sub-clip
- **Rationale:** Quality filtering is a gate, not a label. MegaSaM needs continuous frames
  but should never process frames we've already decided are bad. DINOv3 should never encode
  frames that won't enter training. Disk stores only useful data.
- **Files:** `vllatent/ingest/quality.py`, `vllatent/ingest/pipeline.py`,
  `scripts/ingest_youtube_pilot.py`, `tests/test_ingest_pipeline.py`
- **DoD:** Quality-rejected frames never reach MegaSaM or DINOv3. Segments are continuous.
  Tests cover: all-accepted (1 segment), split at bad block, all-rejected (0 segments).
- **Test:** `$PY -m pytest -q tests/test_ingest_pipeline.py tests/test_quality.py`
- **Deps:** none. Blocked-by: none.

---

### Group 2 — Data Quality Dashboards

**B1.9 — Data quality report script.**
- Tier PURE / AUTO
- `scripts/data_quality_report.py` reads manifest + `.npz` files and produces JSON + terminal
  report: per-clip frame count, acceptance rate, quality distribution (min/max/mean/p5/p95),
  VO confidence distribution, delta magnitude stats (speed distribution, outlier fraction),
  total counts, MegaSaM confidence_source distribution (real vs defaulted).
  Reuses existing `clip_quality_summary()` and `validate_scale_consistency()` from
  `vllatent/ingest/quality.py`.
- **Files:** `scripts/data_quality_report.py` (new), `vllatent/ingest/quality.py` (reuse)
- **DoD:** Script runs on any manifest directory. JSON + terminal output.
- **Test:** `$PY scripts/data_quality_report.py --cache <fixture_dir>` on synthetic data.
- **Deps:** none. Blocked-by: none.

**B1.9b — Per-clip HTML quality report.**
- Tier PURE+TOOL / AUTO
- New file `vllatent/ingest/visualize.py` (~400 LOC). Generates self-contained Plotly offline
  HTML per clip with 8 sections:
  1. Filmstrip (10 evenly-spaced thumbnails extracted from cached frames)
  2. Quality heatmap timeline (RdYlGn colorscale, `frame_quality` over time)
  3. Content filter results (per-shot motion + YOLO verdicts, color-coded accept/reject)
  4. 3D ego-motion trajectory (interactive Plotly `Scatter3d` + `Cone` headings from
     cumulative deltas, colored by speed magnitude)
  5. Body-frame deltas (3 stacked subplots: dx/dy/dz, dyaw, quality overlay; outliers in red)
  6. VO confidence timeline (low regions flagged below 0.3 threshold)
  7. Latent coherence (`cos_sim(z_t, z_{t+1})` timeline, threshold line at 0.85 for
     scene change detection)
  8. Summary table (frames, duration, npz size, pass/fail verdict)
- **Integration:** called after `_write_clip_npz()` in `pipeline.py` as post-hook. Also
  standalone via `scripts/clip_report.py --cache <dir> --clip <id>`.
- **Dependencies:** `plotly` (likely installed), `Pillow`, `jinja2` — all standard ML env.
- **Files:** `vllatent/ingest/visualize.py` (new), `scripts/clip_report.py` (new),
  `tests/test_visualize.py` (new)
- **DoD:** HTML report generated from synthetic fixture. Opens in browser. All 8 sections render.
- **Test:** `$PY -m pytest -q tests/test_visualize.py`
- **Deps:** none (runs on any cached .npz). Blocked-by: none.

**B1.10 — MegaSaM VO validation on pilot clips.** ✅ DONE 2026-06-24
- Tier RESEARCH+TOOL / USER-GATED
- **Objective.** Validate MegaSaM monocular VO on skiing FPV video. No ground truth available
  (no GPS/IMU), so use physics-plausibility + smoothness proxies. Produce GO / CONDITIONAL-GO
  / NO-GO verdict.

- **B1.10a — Validation metrics module (AUTO).** ✅ DONE
  `vllatent/ingest/vo_validation.py` — pure numpy, no MegaSaM dependency. Functions:
  - `trajectory_smoothness(poses) → SmoothnessReport` — jerk (3rd derivative of position),
    acceleration discontinuity count (jumps > 3σ), angular velocity spikes (> 300°/s).
    Skiing should show gradual curves, not step changes.
  - `physics_plausibility(deltas, fps) → PhysicsReport` — check:
    - Max frame-to-frame displacement (skiing ≤ 22 m/s = 4.4 m/frame at 5 Hz)
    - Altitude change pattern: net descending for downhill skiing
    - Lateral acceleration (turns ≤ 3-5 m/s²)
    - Yaw rate (peak ≤ 180°/s plausible; > 300°/s = failure)
  - `confidence_analysis(confidences) → ConfidenceReport` — distribution stats,
    fraction below 0.3 (low-confidence), longest contiguous low-confidence run.
  - `scale_drift(poses) → DriftReport` — compare displacement magnitude in first vs last
    quarter of trajectory (monocular scale drift shows as speed-up or slow-down over time).
  - `vo_verdict(smoothness, physics, confidence, drift) → Verdict` — combine all checks
    into GO / CONDITIONAL-GO / NO-GO with per-check pass/warn/fail.
  Tests: `tests/test_vo_validation.py` (synthetic trajectories: smooth descent, jerky,
  stationary, circular, physically implausible).

- **B1.10b — Validation visualization (AUTO).** ✅ DONE
  `scripts/validate_megasam.py` — CLI that runs the full validation pipeline:
  1. Parse MegaSaM output (`parse_megasam_output` from `megasam.py`)
  2. Convert to body-frame deltas (`se3_sequence_to_deltas` from `ego_motion.py`)
  3. Run all validation checks from `vo_validation.py`
  4. Generate HTML report (extend `visualize.py` or standalone Plotly):
     - 3D trajectory plot (cumulative deltas, colored by speed)
     - Speed profile over time (flag frames > physics threshold)
     - Yaw rate over time (flag spikes)
     - VO confidence heatmap over trajectory
     - Jerk magnitude timeline
     - Per-check verdict table (smoothness / physics / confidence / drift)
  5. Print terminal summary with GO/CONDITIONAL-GO/NO-GO
  Usage: `python scripts/validate_megasam.py --frames-dir ingest_data/frames/ski01
    --megasam-dir ingest_data/frames/ski01_megasam --fps 5 --out reports/ski01_vo.html`

- **B1.10d — Rework MegaSaM parser for real output format (AUTO).** ✅ DONE
  The existing `parse_megasam_output()` was written against a guessed output format that doesn't
  match MegaSaM's actual 3-step pipeline output. Five critical mismatches:

  | What | Parser expected | MegaSaM actual |
  |---|---|---|
  | poses.npy | `(N, 4, 4)` SE(3) c2w | `(T, 7)` w2c `[x,y,z,qx,qy,qz,qw]` (Lie group) |
  | c2w matrices | from poses.npy directly | `outputs/{scene}_droid.npz` key `cam_c2w (T,4,4)` |
  | confidence | `confidences.npy (N,)` | `motion_prob.npy (T,H/8,W/8)` per-pixel — needs aggregation |
  | intrinsics | `(3, 3)` K matrix | `(T, 4)` vector `[fx,fy,cx,cy]` × 8.0 |
  | run_megasam() | calls `run.py` (nonexistent) | 3-step: DepthAnything → UniDepth → camera_tracking |

  **Rework scope:**
  1. `parse_megasam_output()`: NEW primary path reads `reconstructions/{scene}/` directory:
     - `poses.npy (T,7)` — convert w2c `[x,y,z,qx,qy,qz,qw]` → c2w `(T,4,4)` via
       `SE3(poses).inv()` (quaternion → matrix → invert). Pure numpy, no lietorch.
     - `motion_prob.npy (T,H/8,W/8)` → aggregate per-pixel to per-frame `(T,)` confidence
       via spatial mean (or spatial-percentile for robustness).
     - `intrinsics.npy (T,4)` → build K matrix from first frame `[fx,fy,cx,cy]` (already ×8.0).
     - ALSO support `outputs/{scene}_droid.npz` as alternative path (has `cam_c2w (T,4,4)`
       directly + `intrinsic (3,3)` K matrix; no per-frame confidence in this file).
  2. `run_megasam()`: rewrite as 3-step orchestrator calling MegaSaM's actual scripts:
     - Step 1: `Depth-Anything/run_videos.py` (mono disparity)
     - Step 2: `UniDepth/scripts/demo_mega-sam.py` (metric depth + FoV)
     - Step 3: `camera_tracking_scripts/test_demo.py` (SLAM tracking)
     Each step is a subprocess with proper `--conda-env mega_sam` or `PYTHONPATH` handling.
  3. Keep old format paths (flat `poses.npy (N,4,4)`, `cameras.npz`, `results.json`) as
     fallback for backward compat, but log deprecation warning.
  4. Update `validate_megasam.py` CLI: `--megasam-dir` now points to MegaSaM repo root's
     `reconstructions/{scene}` directory (or the `outputs/{scene}_droid.npz` path).

  **Files:** `vllatent/ingest/megasam.py` (rework), `tests/test_ingest_megasam.py` (update)
  **DoD:** Parser reads real ski01 MegaSaM output correctly. Tests cover new format + old fallback.
    `validate_megasam.py` produces correct verdict on ski01.
  **Test:** `$PY -m pytest -q tests/test_ingest_megasam.py`
  **Deps:** blocks B1.10c, B1.10e. Blocked-by: B1.10a, B1.10b (done).

- **B1.10e — End-to-end MegaSaM automation script (AUTO).** ✅ DONE
  `scripts/run_megasam_pipeline.sh` — one-command wrapper for the 3-step MegaSaM pipeline:
  ```bash
  bash scripts/run_megasam_pipeline.sh --clip-id ski01 \
    --frames-dir ingest_data/frames/ski01 \
    --megasam-dir ~/CODE/MegaSaM
  ```
  Runs DepthAnything → UniDepth → camera_tracking in sequence, with progress logging and
  error checking. Copies final `reconstructions/{scene}/` output back to
  `ingest_data/frames/{clip_id}_megasam/` for our pipeline to pick up.
  Also update `run_megasam()` in `megasam.py` to call this script as subprocess (replacing
  the broken `run.py` call), so `process_clip()` in `pipeline.py` works end-to-end.

  **Files:** `scripts/run_megasam_pipeline.sh` (new), `vllatent/ingest/megasam.py` (update run_megasam)
  **DoD:** Script runs full 3-step pipeline on one clip. `run_megasam()` uses it. `process_clip()`
    works end-to-end from frames → MegaSaM → deltas → cache.
  **Test:** Manual run on ski01 (USER-GATED verify).
  **Deps:** blocks future batch ingest. Blocked-by: B1.10d.

- **B1.10c — End-to-end pipeline test on one sub-clip.** ✅ DONE 2026-06-24 (USER-VERIFIED)
  Verdict: **GO** on `ski03_fpv00_c000` (50 frames). Full chain verified:
  content filter → FPV shot detection → 10s subclip → quality gate →
  MegaSaM VO (3 steps) → DINOv3 encoding → .npz cache.
  Output: `reports/e2e_test/cache/ski03_fpv00_c000.npz`
  — latents (50, 196, 768) fp16, deltas (49, 4) float32, all quality >= threshold.

  **Bugs found & fixed during E2E:**
  - `run_megasam()` called with stale `model=` kwarg (removed)
  - `run_megasam_pipeline.sh`: `--no-banner` unsupported by conda version (removed)
  - `extract_fpv_ranges()` ignored per-frame `fpv_mask` → non-FPV frames leaked
    into sub-clips (fixed: now splits within shots at frame-level rejections)
  - MegaSaM `mega_sam` env: `pip install xformers` upgraded PyTorch 2.0→2.12,
    breaking all C++ extensions + xformers CUDA kernels (sm_120 unsupported).
    Fixed via: `scripts/megasam_shims/nystrom_shim.py` (NystromAttention replacement +
    `memory_efficient_attention` → PyTorch SDPA monkey-patch), CUDA 13.0 toolkit,
    `.type()` → `.scalar_type()`, `torch.cuda.amp.autocast` → `torch.amp.autocast("cuda")`,
    sm_120 gencode flags, libcudart 13.0 runtime linkage.
  - DINOv3 encoder: socks:// → socks5:// proxy URL normalization.

  **Files:** `scripts/test_e2e_subclip.py`, `scripts/megasam_shims/nystrom_shim.py`,
  `scripts/megasam_shims/run_unidepth.py`, `scripts/run_megasam_pipeline.sh`,
  `scripts/validate_megasam.py` (kept for standalone VO validation)
- **Deps:** blocks B1.12. Blocked-by: B1.10d (done), B1.10e (done), B1.8b (done).

- **B1.10f — Fix shot boundary detection for consistent VO trajectories (AUTO).**
  `adaptive_threshold` 3.0 → **2.0** across all `detect_shot_boundaries*` and `filter_video*`
  functions. Root cause: skiing footage has high natural frame-to-frame variation (motion 19-28
  typical); a cut between two snowy scenes (spike to ~38) doesn't exceed 3× the local average
  at threshold=3.0. At 2.0, ski03 produces 7 boundaries (was 4) — all verified as real camera
  switches, no false positives.

  **Deleted:** `vllatent/ingest/edit_detection.py` + `tests/test_edit_detection.py`. Two rounds
  of hand-crafted edit detection (histogram correlation + slow-mo + motion spikes; MAD-robust
  spikes + block-pattern consistency) produced catastrophic false-positive rates on real skiing
  footage (80+ frames out of 254 flagged). The entire "edit detection" problem was an under-tuned
  AdaptiveDetector threshold — no separate module needed.

  **Bug fix:** `scripts/test_e2e_subclip.py` stale frame directory not cleaned before copy
  (previous run's frames persisted; MegaSaM saw 50 instead of 39). Fixed with `shutil.rmtree`
  before fresh copy.

  **Files:** `vllatent/ingest/content_filter.py` (threshold in all 4 functions),
  `scripts/test_e2e_subclip.py`, `scripts/verify_filter.py` (removed edit_mask refs)
  **Impact:** Content filter is now a clean **three-stage pipeline**: motion ≥8.0 → YOLO-World
  object rejection → AdaptiveDetector@2.0 shot boundary split. Each shot = one continuous
  camera recording. MegaSaM runs per-shot segments → no trajectory leaps from camera switches.
  **Test:** `$PY -m pytest -q tests/test_content_filter.py` (462 total green)
  **Deps:** none (hotfix on existing pipeline). Blocked-by: none.

---

### Group 3 — Encoder Decision Gate

**B1.11 — Benchmark DINOv3 ViT-B/16 on Orin NX.**
- Tier RESEARCH / USER-GATED (requires Orin NX hardware access)
- The advisory is contradictory: section 5.2 says 50-80ms, but section 4 says ~6-10ms based on NVIDIA
  benchmarks. Must measure empirically. Run `DinoV3Encoder` with TensorRT FP16 export.
  Measure median/p99 latency at batch=1.
- **Decision:** If ViT-B/16 TRT FP16 < 20ms then keep ViT-B/16, D=768, predictor depth 6 (~28M).
  If > 20ms then use Meta's pre-distilled ViT-S/16 (`vit_small_patch16_dinov3.lvd1689m` via timm),
  D=384, predictor depth 8 (~14M). No CosPress training needed either way.
- **Files:** `scripts/benchmark_encoder_orin.py` (new)
- **DoD:** Written benchmark with latencies. EMBED_DIM decision locked.
- **Test:** User runs on Orin NX.
- **Deps:** **CRITICAL GATE** — blocks B1.13. Blocked-by: none (parallel with all of Group 0-2).

**B1.12 — Lock EMBED_DIM + PredictorConfig from encoder gate.**
- Tier PURE / AUTO
- **Default assumption: ViT-B/16 is fast enough, D=768 stays, no code change needed.** If
  B1.11 shows ViT-B/16 > 20ms, THEN switch to ViT-S/16: update `EMBED_DIM=384` in
  `schemas.py`, `PredictorConfig(depth=8, heads=6)` in `config.py`, CLIP lift from 512->384
  in `encode/text.py`, and cascade through all test fixtures. This step is a NO-OP if
  ViT-B/16 meets the speed target.
- **Files (if change needed):** `vllatent/schemas.py`, `vllatent/config.py`, `vllatent/encode/text.py`, `docs/io-contract.md`, all test files with shape fixtures
- **DoD:** Decision documented. If D changes: all tests green with new D. If D stays: no change.
- **Test:** `$PY -m pytest -q && make import-smoke && make typecheck`
- **Deps:** blocks B1.13, B1.14, B1.15. Blocked-by: B1.11.

---

### Group 4 — Sports Loader (GT history latents, sliding windows)

**B1.13 — Sports sliding-window loader with GT history.**
- Tier TORCH / AUTO
- New `SportsTrainingDataset` in `vllatent/data/sports_loader.py`. Reads `.npz` cache files
  (from ingest pipeline). Produces sliding windows of `(H+T)` frames within each fixed-length
  clip. Per sample:
  - `z_t (P,D) fp16` — current observation latent (GT from cache)
  - `history_latents (H,P,D) fp16` — **GT latents from previous H frames** (NOT predicted)
  - `history_mask (H,) bool` — block-causal padding at clip start
  - `target_latents (T,P,D) fp16` — GT future latents (L_latent targets)
  - `target_deltas (T,4) f32` — GT future 4-DoF deltas (L_wp targets), **preprocessed** (below)
  - `vo_confidence (T,) f32` — per-step VO confidence (for confidence-weighted L_wp)
  - `frame_quality float` — composite quality of z_t frame
  - `dt_seconds (T,) f32` — inter-frame time delta (from cached `timestamps` diffs)
- **Delta preprocessing pipeline** (applied in loader, in order):
  1. Physics hard clip: max displacement 4.0 m/frame at 5 Hz, max dyaw 24 deg/frame
     (thresholds scale proportionally by `dt` for other FPS, e.g. CosFly 2 Hz)
  2. Median filter k=3 on deltas (removes single-frame VO spikes, preserves real turns;
     two consecutive extreme frames survive — correct for mogul skiing)
  3. Convert deltas to velocity: `velocity = delta / dt_seconds` (handles mixed FPS:
     CosFly 2 Hz + YouTube 5 Hz in the same training run, per DINO-world precedent)
  4. Per-dimension z-score normalization (store per-dataset mean/std for inference
     denormalization; computed once at dataset construction, saved alongside manifest)
- **Augmentation** (training only, disabled for val):
  - Temporal jitter: shift window start by +/-1 frame randomly
  - Gaussian noise on deltas: `N(0, 0.05 * std_per_dim)` during training
- **Batch construction:** YouTube-only for B-1. ~~mix CosFly + YouTube sources; oversample
  CosFly-Track to ~40% of batches~~ CosFly integration deferred — see CosFly Suitability
  Assessment. Revisit after B1.20 if L_wp undertrains.
- Clips pre-cut to `clip_length_seconds` (from B1.4), so windows are within 10s segments.
- Reuses manifest reading from existing `vllatent/manifest.py`.
- **Files:** `vllatent/data/sports_loader.py` (new), `tests/test_sports_loader.py` (new)
- **DoD:** Loader emits correct shapes/dtypes on synthetic fixture. Block-causal mask correct
  at clip boundaries. GT history verified (not predicted). Delta preprocessing pipeline tested
  (physics clip, median filter, velocity normalization, z-score). Augmentation toggleable.
  dt_seconds correctly computed from timestamps.
- **Test:** `$PY -m pytest -q tests/test_sports_loader.py`
- **Deps:** blocks B1.16. Blocked-by: B1.4, B1.6, B1.12.

**B1.14 — Collate function for batched training.**
- Tier TORCH / AUTO
- `collate_sports_batch()` converts numpy samples to batched GPU tensors.
  Returns `TrainingBatch` NamedTuple with fields:
  - All fields from B1.13 sample, batched to `(B,...)` tensors
  - `dt_seconds (B,T) f32` — inter-frame time deltas for FiLM conditioning on frame rate
  - `sample_weight (B,) f32` = `frame_quality.clamp(min=0.1) * vo_confidence.clamp(min=0.05)`
    — per-sample loss weight (floors prevent zero-weight samples from vanishing entirely)
- **Files:** `vllatent/data/sports_loader.py` (add collate), `tests/test_sports_loader.py`
- **DoD:** Works with `torch.utils.data.DataLoader`. Shapes/dtypes verified. `dt_seconds` and
  `sample_weight` fields present and correctly computed.
- **Test:** `$PY -m pytest -q tests/test_sports_loader.py -k collate`
- **Deps:** blocks B1.16. Blocked-by: B1.13.

---

### Group 5 — Predictor Architecture

**B1.15 — Block-causal ViT predictor + FiLM action conditioning.**
- Tier TORCH / AUTO
- `vllatent/model/predictor.py` with `LatentPredictor(nn.Module)`. Input: `history_latents
  (B,H,P,D)` + `z_t (B,P,D)` + `action_4dof (B,4) f32` + `dt_seconds (B,T) f32`. Output:
  `predicted_latents (B,T,P,D)`.
  Block-causal mask: each horizon step `t+k` attends to `[history, z_t, t+1..t+k-1]`.
  **Two FiLM conditioning sources:**
  - Action FiLM: action projected to `(scale, shift)` per block, applied after LayerNorm.
  - **dt FiLM:** `dt_embedding = MLP(dt_seconds)` → `(scale, shift)` per block, applied
    alongside action FiLM. Lets the model handle mixed FPS (CosFly 2 Hz vs YouTube 5 Hz)
    in a single training run without separate stages (DINO-world precedent).
  D, depth, heads, dropout from `PredictorConfig`. Default depth=6 (DINO-WM precedent;
  sweep 6-vs-8), dropout=0.1. No language cross-attention in B-1.
  **Architecture research (2026-06-25):** depth=6 chosen over 12 because (a) DINO-WM (closest
  analogue — same frozen DINOv2 encoder, same patch-level prediction) uses depth=6 successfully,
  (b) ~28M params is better matched to our pilot data scale (~8.6K frames), (c) all spatial
  tokens (196) retained — no downsampling (universal in DINO-WM / DINO-world / V-JEPA 2-AC).
- **Files:** `vllatent/model/predictor.py` (new), `tests/test_predictor.py` (new)
- **DoD:** Output shape correct. Param count matches expected (~14M at D=384 or ~28M at D=768).
  Block-causal mask verified (future can't attend to later future). Action FiLM changes output.
  dt FiLM changes output when dt varies. Dropout applied during training.
- **Test:** `$PY -m pytest -q tests/test_predictor.py -m torch`
- **Deps:** blocks B1.17. Blocked-by: B1.12.

**B1.16 — Waypoint head.**
- Tier TORCH / AUTO
- `vllatent/model/heads.py`:
  - `WaypointHead`: MLP `D->256->128->4`. Takes `(B,T,D)` -> `(B,T,4)` predicted deltas.
  - ~~TrustHead stub~~ — trust mechanism REMOVED (commit `125576f`). No stub needed.
- **Files:** `vllatent/model/heads.py` (new), `tests/test_heads.py` (new)
- **DoD:** Shapes correct. MLP produces (B,T,4) from (B,T,D).
- **Test:** `$PY -m pytest -q tests/test_heads.py -m torch`
- **Deps:** blocks B1.17. Blocked-by: B1.12.

**B1.17 — Full model assembly.**
- Tier TORCH / AUTO
- `vllatent/model/sports_model.py` with `SportsFollowingModel(nn.Module)`. Assembles predictor
  + waypoint head. Forward takes `TrainingBatch` -> `(PredictorOutput,
  predicted_deltas (B,T,4))`. Encoder is NOT part of forward (latents cached).
  Config-driven construction from `PredictorConfig`.
- **Files:** `vllatent/model/sports_model.py` (new), `tests/test_model.py` (new)
- **DoD:** End-to-end forward on random input. Output shapes correct.
- **Test:** `$PY -m pytest -q tests/test_model.py -m torch`
- **Deps:** blocks B1.18. Blocked-by: B1.15, B1.16.

---

### Group 6 — Training Loop

**B1.18 — Loss functions: L_latent + L_wp.**
- Tier TORCH / AUTO
- `vllatent/train/losses.py`:
  - `L_latent`: smooth L1 with **beta=0.1** (DINO-world 2025 precedent; NOT default beta=1.0).
    **Quality-weighted**: per-sample L_latent scaled by `frame_quality.clamp(min=0.1)` — low
    quality reduces gradient contribution but floor prevents zero-weight. Log cosine similarity
    as diagnostic (not gradient source).
  - `L_wp`: smooth L1 between predicted and GT future deltas. **Confidence-weighted**: each
    sample's L_wp scaled by `vo_confidence.clamp(min=0.05)`. **NOT weighted by frame_quality**
    — waypoint head needs to learn from all motion patterns including fast/blurry frames.
  - `combined_loss`: `L_total = w_quality * L_latent + lambda_wp * w_vo * L_wp`. Where
    `w_quality = frame_quality.clamp(min=0.1)` and `w_vo = vo_confidence.clamp(min=0.05)`.
    Lambdas from `DistillConfig`.
- **Files:** `vllatent/train/losses.py` (new), `tests/test_losses.py` (new)
- **DoD:** Losses computed correctly. beta=0.1 for smooth L1 verified. Quality weighting on
  L_latent tested. Confidence weighting on L_wp tested (frame_quality NOT applied to L_wp).
  Floor clamps verified. Shapes correct.
- **Test:** `$PY -m pytest -q tests/test_losses.py -m torch`
- **Deps:** blocks B1.20. Blocked-by: B1.17.

**B1.19 — Checkpoint save/load + config snapshot.**
- Tier TORCH / AUTO
- `vllatent/train/checkpoint.py`: `save_checkpoint(model, optimizer, epoch, config, metrics,
  path)` and `load_checkpoint(path)`. Config YAML snapshot written to run dir at train start.
  Deterministic seed for reproducibility.
- **Files:** `vllatent/train/checkpoint.py` (new), `tests/test_checkpoint.py` (new)
- **DoD:** Save/load round-trips. Resume produces identical next-batch gradients.
- **Test:** `$PY -m pytest -q tests/test_checkpoint.py -m torch`
- **Deps:** blocks B1.20. Blocked-by: none (parallel with B1.15-B1.18).

**B1.20 — Training script: overfit-tiny-batch.**
- Tier TORCH / USER-GATED
- `scripts/train_sports.py` with `--overfit-tiny` flag. Loads 8-16 samples from pilot cache.
  Trains predictor + waypoint head for 500 steps. Logs L_latent, L_wp, cosine sim per horizon
  step. Dumb baseline: `L_baseline = loss(zeros, GT)`. Must beat baseline within 200 steps.
  Saves checkpoints every 100 steps. Resume-test: stop at step 200, resume, verify identical
  step-201 gradients.
  **Training playbook gates:** overfit-tiny-batch first, save every config, dumb baseline, boring
  HP (AdamW, cosine LR), resume-test early, watch samples not loss.
- **Files:** `scripts/train_sports.py` (new)
- **DoD:** Loss drops below baseline within 200 steps. Checkpoint saved + resume-tested.
- **Test:** User runs on dev box (RTX 5060 Ti). Reviews loss curve.
- **Deps:** blocks B1.22. Blocked-by: B1.14, B1.17, B1.18, B1.19, and pilot data (B1.7 or B1.8).

---

### Group 7 — Data Quality Visibility in Training

**B1.21 — Pre-train sanity check + sample visualization.**
- Tier TORCH / AUTO
- `vllatent/train/sanity.py`: runs at training start. Reads 5 random samples, verifies
  latent dtype/shape, delta dtype/shape, history mask consistency, transform chain vs manifest.
  `vllatent/train/viz.py`: callback every N steps logs GT vs predicted latent cosine sim,
  GT vs predicted waypoint error, per horizon step. JSON log + optional TensorBoard.
  Training playbook: "log frame transforms (#1 foot-gun)" + "watch samples not loss."
- **Files:** `vllatent/train/sanity.py` (new), `vllatent/train/viz.py` (new), tests
- **DoD:** Sanity check raises on inconsistency. Viz callback produces readable logs.
- **Test:** `$PY -m pytest -q tests/test_sanity.py tests/test_train_viz.py`
- **Deps:** none. Blocked-by: B1.13 (for sanity), B1.18 (for viz).

---

### Group 8 — Latent World-Model Training + Verification (B-1)

> **REPLANNED 2026-06-28 · scope-cut 2026-06-29 · B1.22e recovery 2026-06-30 ·
> residual replan approved 2026-07-05.**
> Supersedes the single-step B1.22.
> **B-1 now trains the latent predictor ONLY — the waypoint head is DEFERRED to B-2** (user
> 2026-06-29). Rationale: the head's `L_wp` target is the **MegaSaM delta**, whose monocular-VO
> **scale is inconsistent/unresolved across clips**, and the head architecture (MLP vs
> PI-Prober vs attentive-pool) is **undecided** — both are open problems. The predictor, by
> contrast, trains on **clean GT DINOv3 latents** with no scale ambiguity, so it is the
> well-posed half. **B-1 DoD = a good latent world model.** Staged training stays the umbrella
> recipe (predictor first in B-1; head second in B-2). Plus a **revised training policy** (the
> current `train_sports.py` has no val loop / scene-split / warmup / early-stop / best-ckpt and
> uses fp16+GradScaler — all fixed here) and an **expanded data plan** (full pilot encode, more
> real YouTube, game footage as a **latent-pretraining** source). The rejected
> `reports/training-policy-research-2026-06-25.md` (AdaLN / visual bottleneck / SkyJEPA / GRPO)
> stays **REFERENCE ONLY**; its **PI-Prober** is reopened as a live **B-2 head-architecture
> candidate** (the "should we adopt the prober?" question). Backed by a 9-agent
> research+adversarial workflow (2026-06-28).

#### Framing — what B-1 Group 8 is (and is not)

**B-1 deliverable = a good latent world model** — an action+dt-conditioned DINOv3-latent
predictor that anticipates future latents better than a persistence baseline, verified on real
held-out skiing FPV. **NOT** a deployable waypoint policy. The run is **overfitting-dominated**
on the pilot (~8.6K frames / 11 source videos → ~7.5K sliding windows; an epoch on H20 is
seconds), so B-1 is an **architecture-validation + data-scale** effort: prove the predictor
learns, then grow the data.

**Deferred to B-2 (unresolved — user decision 2026-06-29):**
- **Waypoint head training** (`L_wp`) — the entire Stage-2/Stage-3 head pipeline.
- **MegaSaM scale inconsistency** — monocular VO recovers translation only up to an unknown,
  per-clip, drifting scale. Current `normalize_scale("median_speed")` + per-dataset velocity
  z-score make the *distribution* comparable but do NOT fix cross-clip metric scale, so absolute
  `L_wp` targets are unreliable. Candidate B-2 fixes (to research): metric-depth anchoring from
  MegaSaM's UniDepth stage; scale-free waypoint parameterization (unit-direction + log-speed);
  per-clip scale alignment; GPS/IMU anchoring on custom data. **B-1 sidesteps this** — the
  predictor's action-FiLM uses the (normalized) delta only as a *soft conditioning hint*, never
  as a regression target, so scale error there is tolerable, not load-bearing.
- **Prober decision** — MLP vs SkyJEPA-style PI-Prober vs attentive-pool head; a B-2
  head-architecture question now that the head is designed in B-2.

**Data reality (corrected 2026-06-28).** Real follow-cam FPV is genuinely scarce, and the
plan's assumed B-3 volume saviour **Ego-Exo4D does not fit** (cooking/music/dance/basketball/
soccer/bouldering — real texture but **no skiing and no sustained high-speed ego-translation
following a subject**; wrong dynamics for our task — see Open Decision #9). **Game footage
(极限国度 / Ubisoft Steep, Riders Republic) has the opposite trade-off — synthetic texture but
exactly the right structure (follow-cam, downhill, alpine, fast translation).** Because the
predictor is **domain-blind** (only action-FiLM + dt-FiLM; no "real vs game" input), game
frames train the *same shared weights* as real — so game data is used as a **measured
latent-pretraining source** (B-1's own `L_latent` objective): pretrain on real+game, then
**fine-tune the predictor on real-only**, and **keep the game contribution only if it improves
cosine on a real held-out val** (safety net against domain pollution). This turns "is there
enough data?" into a measured experiment.

**B1.22e recovery diagnosis (2026-06-30 through run2 paste-back 2026-07-05).** The first H20
depth-6/action-FiLM/bf16 run completed but failed DoD: best finite epoch 16 had `val_cos=0.7318`,
persistence `0.8094`, margin `-0.0777`; epoch 23 collapsed and epoch 24 became NaN. Recovery run2 is
now the active baseline: depth 4, LR `1e-4`, AdamW betas `(0.9,0.95)`, bf16, `--exclude-source ski03`,
`--eval-train`, and `--eval-by-source`. It had cleaner numerics, but still failed persistence on
both held-out val and train. Best visible val was epoch 25 / step 8008 with `val_cos=0.7593`,
`val_persistence=0.8576`, `val_margin=-0.0983`; best visible train eval was epoch 27 / step 8624
with `train_cos=0.8003`, `train_persistence=0.8685`, `train_margin=-0.0683`. Because the predictor
does not beat persistence even on train, stop treating split variance, `ski03`, depth, or LR as the
primary blocker. The metric is apples-to-apples: model and persistence compare against the same
future DINO latents. The root problem is objective/parameterization: at 5 Hz and horizon 1-4,
whole-frame frozen DINO latents are highly persistent, so an absolute predictor must reconstruct the
large static component before learning the small motion residual needed to beat `z_t`. User approved
the next AUTO direction on 2026-07-05: replan to a persistence-residual predictor
`z_hat = z_t + delta_hat`, with a zero/near-zero-initialized residual path so the untrained model
starts at or near persistence. Do not silently relax the DoD; per-horizon margin over persistence
remains the pass gate.

#### A. Revised training policy (replaces "AdamW + cosine, batch-to-GPU")

- **Keep the LOCKED stack** — frozen+cached DINOv3 ViT-B/16 D=768; predictor depth=6 /
  heads=12 / mlp_ratio=4 / dropout=0.1 (~57M params). The `WaypointHead` exists in the
  assembled model but is **UNTRAINED in B-1** (B-2). **No new modules** (no PI-Prober / AdaLN /
  visual bottleneck / readout token / SkyJEPA / GRPO — all REFERENCE-ONLY).
- **Precision:** default `--amp-dtype bf16` on H20 (Hopper sm_90) — **drop GradScaler**; cast
  predictor outputs to **fp32 before** `smooth_l1` / `mean` / `cosine` (latents are fp16 on
  disk; the ~602K-element reductions and the cosine diagnostic must be exact). Keep `fp16`
  (with GradScaler, checkpoint `scaler.state_dict()`) and `fp32` as fallback flags.
- **Overfitting is the primary lever** (small data): keep dropout 0.1 + existing temporal-
  jitter ±1 + Gaussian delta-noise; switch AdamW to **two param groups** — `weight_decay=0.05`
  on weight matrices (`ndim>=2`), `0.0` on biases / LayerNorm / `temporal_embed` / FiLM
  zero-init (replaces the flat `wd=0.01`); grad-clip 1.0.
- **LR + warmup:** add **linear warmup** (~5% of steps) → `CosineAnnealingLR` (`eta_min=1e-6`)
  via `SequentialLR` (current script has NO warmup). AdamW betas (0.9, 0.95). Batch 64 default
  (96 GB fits more, but 64 generalizes better on ~7.5K windows); `grad_accum=1`. DataLoader
  `num_workers 4-8`, `pin_memory`, `drop_last` (train), per-worker/per-epoch RNG reseed.
- **Validation (load-bearing — these decide validity):**
  - **Scene-split by SOURCE video, not sub-clip.** Many sub-clips share a source (e.g. all
    `ski03_*`); splitting by sub-clip leaks. Group by `stem.split('_')[0]`; hold out **2–3
    whole source videos** spanning the sport mix.
  - **NormStats from TRAIN clips only,** injected into val (`SportsTrainingDataset(...,
    norm_stats=train_ds.norm_stats)`) — fixes the current per-dataset leak (`train_sports.py:94`
    computes stats over the whole dataset). NormStats only affect the action-FiLM input in B-1
    (no `L_wp` target). Save the **train** stats as the inference stats.
  - **Best-ckpt + early-stop:** evaluate every epoch / `--eval-every`; keep `ckpt_best.pt`
    (this is what ships to B1.23, **not** `ckpt_final.pt`); patience ~5–10 evals (the 57M
    predictor memorizes fast here). After the failed H20 run, select by **`val_margin`**, not
    absolute `val_cos`, because margin is the DoD signal.
  - **Latent metrics** (the B-1 DoD signal): per-horizon (t=1..4) **val latent cosine** AND the
    margin over a **persistence baseline** (`cos(z_t, z_{t+k})` — "next frame ≈ current frame";
    a latent predictor that can't beat persistence has learned nothing). At N=11 sources, treat
    `val cosine > 0.7` as a DIRECTIONAL high-variance signal (rotate 2–3 folds); the
    **persistence margin** is the more robust pass signal. Optional richer diagnostics:
    per-patch cosine, nearest-neighbour frame retrieval from the predicted latent, per-source
    margin, and a train-set margin pass to distinguish objective failure from generalization.
- **Loss unchanged** (LOCKED): `L_latent` = smooth-L1 β=0.1, frame_quality-weighted, **sole**
  loss in B-1 (no pixel decoder — DINO-WM ablation shows reconstruction *hurts*; `L_wp` is B-2).
  FiLM stays locked (AdaLN parked as a B-2 experiment knob).

#### B. The B-1 training run — latent world model (predictor only)

Staged training, Stage 1 only. The `WaypointHead` is strictly **downstream** of the predictor,
so deferring it to B-2 is a clean cut: B-1 trains the predictor in isolation; B-2 freezes this
predictor and trains the head on top (design in the B-2 section below).

- **TRAIN the predictor only** (`blocks`, `action_film`, `dt_film`, `temporal_embed`,
  `output_norm`). The head is not optimized and `L_wp` is not computed (`--latent-only`).
- **LOSS:** `L_latent` only (smooth-L1 β=0.1, frame_quality-weighted). **TARGET:** GT future
  DINOv3 latents from the cache — clean, no scale ambiguity (the whole reason B-1 is the
  well-posed half). LR 2e-4 (→1e-4/1.5e-4 if unstable or overfitting), warmup→cosine, WD 0.05
  param groups, AdamW betas `(0.9, 0.95)`, bf16, dropout 0.1, batch 64. Early-stop on
  **val latent cosine vs persistence margin** (`--early-stop-metric val_margin` after the failed
  first H20 run).
- **Residual replan (approved 2026-07-05):** add a `z_hat = z_t + delta_hat` predictor mode.
  The residual path must be zero/near-zero initialized so initialization evaluates at approximately
  the persistence baseline. Keep the existing absolute `SmoothL1(z_hat, z_future)` path as the
  default eval-compatible loss, and expose delta-loss ablations:
  `SmoothL1(delta_hat, z_future - z_t)` and a combined absolute+delta loss. Select checkpoints by
  margin, inspect every per-horizon margin, and log the minimum horizon margin so a positive average
  cannot hide a failed short horizon.
- **Action conditioning (kept).** action-FiLM is fed the normalized `last_action` (previous
  observed delta) and dt-FiLM the `dt_seconds` — this is what makes it a *world-action* model,
  not just a video model. The MegaSaM scale problem is **non-load-bearing here** (soft
  conditioning hint, not a target). *Sub-decision (Open G):* run **action-conditioned** (default)
  or pure **action-free** (`--no-action-film`, fully MegaSaM-independent) — default keeps it.
- **HANDOFF:** save `ckpt_best.pt` (predictor) + `norm_stats.npz` (train-only) + config snapshot.
  This is the shipped B-1 artifact and the B-2 Stage-2 starting point.
- **Game-pretraining variant (the data-scale experiment):** first pretrain on **real+game**
  (`domain=game` slice, down-weighted via `frame_quality`), then **fine-tune the predictor on
  real-only**; **keep the game contribution only if real-val cosine improves** over real-only
  training. Measured, not assumed (§Framing).
- **Invariants:** NormStats computed **once** on train clips, reused verbatim (and at inference);
  latents are **not** normalized (raw fp16). `--resume` restores optimizer/scheduler within the
  run. **No EMA / no VICReg / no anti-collapse** (frozen cached target cannot collapse — repo
  invariant).

> **Deferred to B-2 (head training):** freeze this predictor (`requires_grad_(False)` +
> `eval()` + `no_grad`) and train the head on its PREDICTED (not GT) mean-pooled latents; the
> mean-pool-washes-out-heading risk, the `mean_minus_zt` / attentive-pool escalation, the
> MLP-vs-PI-Prober decision, the MegaSaM-scale fix, and an optional joint fine-tune all live in
> B-2. See the Phase B-2 section.

#### C. Data plan

1. **(BLOCKING) Full pilot encode** — today only `ski03_fpv00_c000.npz` is cached; B1.7 ran
   `--filter-only`. Encode all **11 accepted clips** (38 FPV ranges, **173 sub-clips** @10s/5fps,
   ~8.6K frames) on the H20 → 173 `.npz` + manifests. Highest-value B-1 action; hard prerequisite.
2. **Expanded real YouTube** (parallel) — harvest named follow-cam/FPV channels (**Dutch Drone
   Gods**, **Johnny FPV**, **Gab707 / Gabriel Kocher** ski-FPV, **GoPro** Awards/follow-cam,
   **Red Bull / RedBullBike / RedBullSnow**, **TRYP FPV**, **Richard Permin/Fastwood**, wingsuit
   **Soul Flyers / TEEM**) + **AirVuz** curated collections via `yt-dlp --flat-playlist` + title
   regex `(?i)(fpv|follow|chase|one[ -]?take|no cuts|pov)` into a NEW
   `configs/sports_clips_candidates.yaml`. 3-level dedup (`video_id` → frame pHash → sub-clip
   pHash after SBD). SponsorBlock pre-pass. Gates: ≥1080p, native ≥24fps, exclude artificial
   slow-mo (corrupts MegaSaM deltas). Sport + camera-behavior quotas. Realistic target **~5h ≈
   ~90K frames** (~600–700 accepted 10s sub-clips) from ~8–13h curated source (~62% yield).
3. **Game footage as Stage-1 pretraining** (parallel) — Steep / Riders Republic no-commentary
   longplays, **HUD/overlay OFF** (YOLO-World rejects overlays), ≥1080p. `domain=game` tag.
   **No clean-GT-trajectory advantage** (no telemetry/camera-path export in either game → deltas
   carry the same MegaSaM noise as YouTube), so used for **L_latent appearance/dynamics
   diversity** in Stage-1 pretraining, down-weighted via `frame_quality` and gated on real-val
   improvement (Stage 1 scale variant, §B). Never the B-1 architecture-validation training target.
4. **AirSim — DEFER to B-3+.** The A5.13 render harness yields clean GT 6-DoF deltas, but its
   UE4 urban/suburban scenes are domain-wrong and have no moving athlete to follow. Becomes the
   right controllable source only after a snowy scene + animated athlete + scripted follow
   trajectory are authored (substantial build).
5. **Metadata fields** (new, typed) in the clip yaml: `domain {real,game}`, `camera`, `subject`,
   `creator`, `license`, `source_resolution`, `source_fps`, `slowmo`, `time_ranges`, `env`,
   `accept_status`.
6. **Storage/ops:** 196×768 fp16 ≈ 0.29 MB/frame → ~25 GB per 90K frames. Latents/`runs/`/videos
   stay **off git** (no-blobs); rsync only; provenance manifest per clip is the cache key.

#### D. Steps

**B1.21b — Cleanup stale trust references + remove empty `verify/`.**
- Tier PURE+DOC / AUTO
- Remove leftover `trust` references from the trust-mechanism removal (commit `125576f`):
  `schemas.py` docstring, `CLAUDE.md` OPEN-list line ~44, plan references; delete empty
  `vllatent/verify/` (only `__pycache__`). No behaviour change. `PredictorOutput` shape unchanged.
- **DoD:** No `trust` refs remain in those spots; `vllatent/verify/` gone; `make test` + lint +
  typecheck green.
- **Deps:** none (parallel cleanup).

**B1.22a — Upgrade `train_sports.py` for the B-1 latent-only run.**
- Tier TORCH / AUTO
- Add what the single-loop script lacks: **`--latent-only`** predictor-training mode (optimize
  `model.predictor.parameters()`, compute `L_latent` only, skip head/`L_wp`); `@torch.no_grad
  evaluate()` → per-horizon val latent cosine + **persistence-baseline margin** (`cos(z_t,
  z_{t+k})`); `split_clips_by_source()` (group by `stem.split('_')[0]`, hold out whole sources);
  train-only NormStats injected into val; `SequentialLR` warmup→cosine; `ckpt_best.pt` +
  early-stop; `--amp-dtype` default bf16 (drop scaler, cast outputs fp32); AdamW decay/no-decay
  param groups (wd 0.05); `--no-action-film` toggle; `--domain-weight` for the game slice;
  per-worker RNG reseed. Add a frozen **`TrainConfig`** (PURE tier) for the swept knobs;
  `checkpoint.py` records `val_metrics`. **The staged HEAD plumbing (`--stage 2/3`, predictor
  freeze, `--init-predictor`, `--head-input`, joint control) is B-2, NOT built here.**
- **DoD:** new flags parse + run; `evaluate()` returns per-horizon cosine + persistence margin
  on a synthetic 2-source fixture; scene-split holds out whole sources (no window leak); val
  uses train NormStats; bf16 path has no scaler; `--latent-only` optimizes only the predictor;
  existing `--overfit-tiny` smoke still beats the persistence/zeros baseline within 200 steps;
  unit tests for split + evaluate + param-groups; `make test-torch` + lint + typecheck green.
- **Deps:** Blocked-by: B1.18, B1.19, B1.21 (done). Blocks: B1.22e.

**B1.22b — Generate the full B-1 dataset ON THE DEV BOX, then rsync `.npz` to H20.**
- Tier TORCH+ORCH / **USER-GATED** (dev box 5060 Ti — **NOT** H20)
- **Strategy (decided 2026-06-29):** data generation is the slow+cheap half; training is the
  fast+expensive half. The full chain (content filter → FPV extract → quality gate → MegaSaM VO →
  DINOv3 encode) **already ran on the 5060 Ti in B1.10c**, so generate the *entire* B-1 dataset
  locally (overnight, free) and **rsync only the `.npz` (~2 GB for 173 clips) to the H20** — rent
  the H20 purely for the B1.22e training run. Front-load **all** curation (B1.22c) first so this is
  ONE local generation pass. Resolution **720p** (`scale=1280:720`); aspect = **center-square-crop**
  to 224² (committed in DINOv3 preprocessing — undistorted, must be matched on-drone in Phase D).
- Run: `python scripts/ingest_youtube_pilot.py --device cuda` (full; no `--filter-only`/`--skip-megasam`)
  → `ingest_data/latent_cache/*.npz`. Today only `ski03` is cached (and it's stretch-encoded → stale;
  regenerated here under the new crop). BLOCKING prerequisite for B1.22e.
- **DoD:** all accepted clips → `.npz` with valid manifests in `ingest_data/latent_cache`;
  `pilot_summary` reconciles; user pastes encode summary (frame count, fp16 dtype, BGR→RGB flag,
  per-clip OK/error). MegaSaM is the time bottleneck — watch `pilot_summary` for per-clip failures
  (first full multi-clip run; only ski03 was validated in B1.10c). Stays `in_progress` until verified.
  Then `rsync -avP ingest_data/latent_cache/*.npz <H20>:.../ingest_data/latent_cache/` (never git).
- **Deps:** Blocked-by: B1.7 (filter done), B1.10c (E2E encode verified on dev), B1.22c (curation).
  Blocks: B1.22e.

**B1.22c — Curate + promote more REAL YouTube FPV (FRONT-LOADED, before B1.22b generation).**
- Tier RESEARCH+DATA / **USER-GATED** (download/ingest)
- §C.2 — candidates yaml + 3-level dedup + SponsorBlock + **resolution/fps/aspect gates** (≥720p
  source; reject vertical/4:3 oddballs so the pre-crop frames are uniform) + slow-mo exclusion +
  sport/camera quotas + typed metadata fields; promote accepted clips into `configs/sports_clips.yaml`.
  **Do this FIRST** so the whole B-1 dataset is curated before the single local generation pass
  (B1.22b). Curation tooling AUTO; download/ingest USER-GATED (on the dev box, same as B1.22b).
  Target ~5h / ~90K frames near-term. Real-only val maintained. **If this yields enough clean real
  data to beat persistence in B1.22e, the game slice (B1.22d) is dropped.**
- **DoD:** `sports_clips_candidates.yaml` populated; dedup script green; N additional clips
  promoted + encoded; user verifies the additional encode.
- **Deps:** Blocked-by: B1.7. Recommended-before final B1.24 (not blocking the pilot run B1.22e).

**B1.22d — [CONDITIONAL FALLBACK] Game footage (Steep / Riders Republic) as a `domain=game`
latent-pretraining slice.**
- Tier RESEARCH+DATA / **USER-GATED** (capture/ingest) — **DEFERRED 2026-06-29; build ONLY if
  real-only B1.22e fails to beat persistence with margin.**
- Rationale for deferral: the predictor is domain-blind (action+dt FiLM only), so game data
  pollutes the shared weights; game was added purely as a *volume* hedge. If B1.22c scales real
  YouTube enough that B1.22e beats persistence on real held-out val, **drop this step**. The
  `--domain-weight` plumbing (WeightedRandomSampler over `loader.sample_domains`, default `real`)
  shipped in B1.22a, so this can be added later with **zero code changes** beyond ingest.
- §C.3 (if built) — no-commentary longplays, HUD OFF, ≥1080p source; `domain=game` metadata
  written into the `.npz`; ingest through the same pipeline (MegaSaM VO + DINOv3 encode + the same
  center-square-crop) → `.npz` with `domain=game` provenance. Extend `manifest.py` validator with a
  typed game-video distinction (do NOT free-text). Used only as the `L_latent` pretraining slice
  (B1.22e game-pretraining variant), down-weighted, never the validation target.
- **DoD:** game clips ingested + encoded with `domain=game` tag; manifest validates the new
  provenance; user verifies the encode.
- **Deps:** Blocked-by: B1.22a (domain plumbing). Feeds B1.22e game-pretraining variant.

**B1.22e — B-1 training run: latent world model on H20 (predictor only, `L_latent`).** ✅ CLOSED BY DECISION 2026-07-05
- Tier TORCH / **USER-GATED** (H20)
- **Outcome.** B1.22e is diagnostic-complete and model-incomplete. The original absolute run
  failed persistence on validation and later hit non-finite loss. Run2, based on the latest
  recovery code and excluding `ski03`, also failed persistence on both train and validation. The
  residual `z_hat = z_t + delta_hat` follow-up initialized at persistence and quickly fit train
  slightly above persistence, but validation stayed at or below persistence and degraded by epoch 2.
- **Decision.** Do not launch another B1 latent H20 run. Do not activate game pretraining to chase
  the DINO-latent persistence metric. The B1 latent-cosine DoD remains unmet, but the result is
  informative enough to close B1 and move to B2: the deployable signal must be future action
  prediction, not raw future-DINO cosine.
- **Recorded B1 evidence.**
  - Run2 best visible val: `val_cos=0.7593`, persistence `0.8576`, margin `-0.0983`.
  - Run2 best visible train: `train_cos=0.8003`, persistence `0.8685`, margin `-0.0683`.
  - Residual run train epoch 2: margin `+0.0017`; residual val epoch 2: margin `-0.0020`.
- **DoD disposition.** Original B1 "good latent world model" DoD failed. B1 is closed as
  **pipeline/data/training-diagnostics complete**, with no accepted predictor checkpoint.
- **Deps:** Blocked-by: B1.22a, B1.22b. Superseded by B2.0/B2.1.

> **B1.22f / B1.22g (waypoint head Stage 2 + Stage 3) → MOVED to Phase B-2** (deferred
> 2026-06-29: unresolved MegaSaM scale + undecided head architecture). See the Phase B-2 section.

**B1.23 — Jetson Orin NX inference speed check (<50 ms / 20 Hz).** ⛔ SUPERSEDED 2026-07-05
- No accepted B1 predictor checkpoint exists, so encoder+predictor benchmarking is no longer the
  active gate. B2 will benchmark the encoder + direct scale-free action policy when B2b has a
  candidate checkpoint.

**B1.24 — Phase B-1 DoD verification (good latent world model).** ⛔ SUPERSEDED 2026-07-05
- The original B1 model DoD failed. The completion record is now the B1 closure note in
  `DEV_LOG.md` plus this plan revision. The next active Ralph loop starts at B2.0.

#### E. B2 USER-GATED runbook preview

1. **No B1 H20 command remains active.** The previous residual latent command is historical only.
2. **B2a local gate first.** Codex must complete B2.1-B2.5 and record local metrics before any
   H20 command is proposed.
3. **B2.6 USER gate.** Codex provides exactly one paste-ready B2b H20 command after local B2a
   passes. The user approves it or asks for more diagnosis.
4. **B2.7 USER-GATED H20 run.** User runs the approved command and pastes
   `val_action_metrics.jsonl`, train metrics if enabled, source metrics, steps/sec, GPU memory,
   and checkpoint/config existence.
5. **B2.8 readout before another run.** A second paid run requires a recorded decision: pass,
   label-quality diagnosis, attentive-pool escalation, diffusion escalation, or real-odom data.

#### F. B2 verification ladder preview

1. Target contract: translation-scale invariance and finite zero-speed behavior.
2. No leakage: future action labels are targets only, never model inputs.
3. Overfit-tiny: direct action policy beats the best dumb baseline.
4. Local source-split: B2a beats best baseline by at least 10% aggregate or stops with diagnosis.
5. H20 B2b: one approved run beats best baseline by at least 15%, improves on a majority of held-out sources, and saves checkpoint/config/metrics.
6. Jetson: encoder + action policy is benchmarked only after a useful B2 checkpoint exists.

---

## Phase B-2: Scale-Free Future-Action Policy

> **Replanned 2026-07-05 after B1 closure.** B2 is no longer "train a waypoint head on top of a
> successful B1 latent predictor." B1 produced useful diagnostics but no accepted predictor
> checkpoint. B2 therefore starts from the frozen DINOv3 cache and trains a **direct scale-free
> future-action policy**. The future action sequence is the **target**, never an input.

### B2 framing

**Crux.** YouTube + MegaSaM provides useful ego-motion shape, but not trustworthy metric scale.
Metric scale must come from onboard odometry during real drone inference. B2 must therefore learn
scale-free motion intent from video and leave metric conversion to the controller.

**Research basis.**
- ViNT predicts future action waypoints from current/past observations and normalizes relative
  waypoints by robot top speed; deployment unnormalizes through a robot-specific controller.
- NoMaD treats future action sequences as the policy output and uses diffusion only when the action
  distribution is multimodal; this is a later escalation, not the first B2 implementation.
- FutureNav keeps action prediction as the inference path and uses inverse/forward/future-state
  objectives as auxiliary training support. B2 may add auxiliary world/dynamics losses later, but
  action quality is the gate.
- Monocular VIO scale is not observable from YouTube RGB alone. Odom/IMU/range/GPS can resolve
  scale on the real platform; MegaSaM's Youtube scale must not be treated as metric truth.

**B2 locked defaults.**
- **Target representation:** per-horizon scale-free action vector:
  `[unit_dir_x, unit_dir_y, unit_dir_z, log_speed_ratio]`. Raw MegaSaM metric magnitude is
  diagnostics only; yaw-rate targets stay out of the locked B2.1 direct-policy contract.
- **Inference conversion:** `speed_cmd = min(odom_reference_speed * exp(log_speed_ratio), 7.5 m/s)`.
  The controller must clamp strictly below `7.5 m/s`; B2 never learns an absolute Youtube speed.
- **Architecture:** direct action head first. Inputs are frozen DINO history/current latents,
  history mask, `dt_seconds`, and previous observed scale-free action from the past. Do not depend
  on B1 predictor rollout in B2a.
- **Deferred:** PI-Prober, NoMaD-style diffusion, language cross-attention, game pretraining, and
  auxiliary latent/world losses are not allowed until the direct scale-free policy gate is passed or
  fails with a diagnosed reason.

### B2 Ralph-loop protocol

Each B2 step follows the same loop:

1. Read `DEV_LOG.md`, `.codex/ralph-rules.md`, this B2 section, then only the relevant code/tests.
2. Execute the **lowest active B2.x** step.
3. Make one bounded implementation or verification change.
4. Run the listed narrow test before broad tests.
5. Record verified facts in `DEV_LOG.md`.
6. Commit with specific paths only; never `git add -A`.
7. Stop at every USER-GATED step with paste-ready commands. Do not operate H20/SSH/docker for the user.

### B2a — Training-policy gate (AUTO unless noted)

**B2.0 — Close B1 and activate B2 rules.**
- Tier DOC / AUTO
- Update this plan, `AGENTS.md`, `.codex/ralph-rules.md`, and `DEV_LOG.md`: B1.22e closed as
  diagnostic-complete / model-incomplete; B1.23/B1.24 superseded; next active step is B2.1.
- **DoD:** docs agree on B2 active queue and no further B1 H20 latent run is requested.
- **Test:** `rg -n "B2.1|diagnostic-complete|future action sequence is the target" plans/phase-b-sports-training.md .codex/ralph-rules.md DEV_LOG.md`
- **Deps:** blocked-by user approval of B1 closure. Blocks B2.1.

**B2.1 — Pure scale-free action target contract.**
- Tier PURE / AUTO
- Add a pure numpy target module for scale-free future actions. It must expose the locked action
  dimension and transformation utilities from raw/preprocessed velocity-like deltas:
  `[unit_dir_x, unit_dir_y, unit_dir_z, log_speed_ratio]`, plus inverse/control helper docs showing
  odom scale and the `7.5 m/s` cap live outside Youtube training.
- Required invariants:
  - multiplying all translational deltas by any positive scalar leaves `unit_dir_xyz` and
    `log_speed_ratio` unchanged;
  - yaw is not part of the B2.1 target vector;
  - zero/near-zero speed is finite and explicitly masked or mapped to a stable fallback;
  - future actions are never returned as model inputs.
- **DoD:** pure tests prove scale invariance, finite zero-speed behavior, shape/dtype, and no torch import.
- **Test:** `$PY -m pytest -q tests/test_scale_free_targets.py`
- **Deps:** B2.0. Blocks B2.2/B2.3.

**B2.2 — Sports loader emits B2 action targets without breaking B1.**
- Tier TORCH-DATA / AUTO
- Extend `SportsSample` and collation additively: keep `target_deltas` and `last_action` for B1
  compatibility, and add B2 fields for `target_actions_scale_free`, `last_action_scale_free`, and
  `odom_reference_speed` diagnostics. B2 collation returns a separate action-policy batch so B1
  `TrainingBatch` tests and `scripts/train_sports.py` remain reproducible.
- **Leakage rule:** `last_action_scale_free` is computed only from observed past motion. Future
  horizon labels may supervise `target_actions_scale_free` but must never appear in inputs.
- **DoD:** loader tests cover additive shapes, source split still holds, old B1 tests still pass,
  and a targeted leakage test changes future deltas while preserving past inputs.
- **Test:** `$PY -m pytest -q tests/test_sports_loader.py tests/test_collate.py tests/test_scale_free_targets.py`
- **Deps:** B2.1. Blocks B2.3/B2.4.

**B2.3 — Direct scale-free action policy.**
- Tier TORCH / AUTO
- Add a small direct policy module. It reads frozen DINO history/current latents, `history_mask`,
  `dt_seconds`, and previous observed scale-free action, then predicts the full future action
  sequence. It does not call the B1 latent predictor and does not accept future action labels.
- Default architecture: mean-pool patch tokens per frame, temporal tokens over history/current,
  a small Transformer/MLP context block, horizon embeddings, and a per-horizon output head with
  `SCALE_FREE_ACTION_DIM` outputs. Attentive pooling is an escalation only if B2a underfits.
- **DoD:** output shape `(B,T,SCALE_FREE_ACTION_DIM)`, deterministic eval, differentiable, no
  dependency on `target_actions_scale_free` in `forward`.
- **Test:** `$PY -m pytest -q tests/test_action_policy.py tests/test_heads.py`
- **Deps:** B2.2. Blocks B2.4.

**B2.4 — Action losses, baselines, and metrics.**
- Tier TORCH / AUTO
- Add action-policy loss and evaluation metrics:
  - direction cosine / angular error;
  - normalized path-shape ADE/FDE;
  - speed-ratio MAE;
  - yaw-rate MAE;
  - aggregate score where lower is better;
  - margins against repeat-last-motion, no-turn, zero/mean, and linear extrapolation baselines.
- **DoD:** perfect prediction scores near zero error, bad prediction is worse, repeat-last baseline
  is deterministic, metrics are sample-count weighted, and margins are positive only when the
  model beats the best baseline.
- **Test:** `$PY -m pytest -q tests/test_action_metrics.py tests/test_losses.py`
- **Deps:** B2.2. Blocks B2.5.

**B2.5 — B2 trainer and local training-policy verification.**
- Tier TORCH / AUTO
- Add a B2 training script or explicit B2 mode that trains only the direct action policy. It must
  save config snapshots, checkpoint best-by-action-margin, write `val_action_metrics.jsonl`,
  optionally write `source_action_metrics.jsonl`, and support overfit-tiny.
- Default local recipe: depth small, fp32 or bf16 depending device, train source split, val source
  split, no latent loss, no language, no game.
- **B2a DoD:** target invariance tests pass; no-leakage tests pass; tiny overfit beats baseline;
  local source-split smoke beats the best dumb baseline by at least **10%** on aggregate
  scale-free action score or stops with a label/metric diagnosis before any H20 run.
- **Test:** `$PY -m pytest -q tests/test_train_sports_b2.py tests/test_action_policy.py tests/test_action_metrics.py`
- **Deps:** B2.3/B2.4. Blocks B2.6.

**B2.6 — B2a technical readout and USER gate.**
- Tier DOC / **USER-GATED**
- Summarize B2a local evidence in `DEV_LOG.md`: target invariance, leakage, tiny overfit, local
  source-split metrics, best baseline, and per-source failure modes. If the local gate fails, do
  not propose H20; replan labels/metrics first.
- **DoD:** user approves exactly one B2b H20 command block, or asks for another B2a diagnosis.
- **Test:** docs-only plus any failed/passed command outputs recorded.
- **Deps:** B2.5. Blocks B2b.

### B2b — Stronger checkpoint (USER-GATED H20)

**B2.7 — H20 scale-free action-policy run.**
- Tier TORCH / **USER-GATED** (H20)
- User runs the approved B2 command on the existing 919-file cache. Do not operate H20/SSH/docker
  from Codex. Validation stays source-split. Save `ckpt_best.pt`, config snapshot, metrics JSONL,
  and source metrics under `runs/`.
- **B2b DoD:** one H20 run beats the best baseline by at least **15%** aggregate, improves on a
  majority of held-out sources, saves checkpoint/config/metrics, and produces prediction samples
  showing sane direction/yaw/relative-speed behavior. No metric flight claim.
- **Paste-back requested:** tail of `val_action_metrics.jsonl`, tail of `train_action_metrics.jsonl`
  if enabled, source metrics, steps/sec, GPU memory, and checkpoint existence.
- **Deps:** B2.6. Blocks B2.8.

**B2.8 — B2b readout and escalation decision.**
- Tier DOC / **USER-GATED**
- Parse pasted B2b metrics and decide:
  - pass → move to B2.9 Jetson/action-policy speed check;
  - label failure → inspect MegaSaM direction/yaw quality and source filtering;
  - model underfit → try attentive pooling before PI-Prober;
  - multimodal averaging → consider NoMaD-style diffusion;
  - scale/speed limitation → defer to real odom data.
- **DoD:** decision recorded in plan/DEV_LOG before another paid training attempt.
- **Deps:** B2.7. Blocks B2.9 or replanned B2a.

**B2.9 — Jetson action-policy speed check.**
- Tier RESEARCH / **USER-GATED** (Orin NX)
- Benchmark frozen DINOv3 encoder + direct action policy end-to-end. The B2 speed target inherits
  the original onboard constraint: <50 ms / 20 Hz if possible, otherwise conditional smaller policy
  or encoder distillation is opened.
- **DoD:** written benchmark; user-pasted latency; decision whether B2 checkpoint is feasible for
  Phase-D closed-loop experiments.
- **Deps:** B2.8 pass.

### B3 preview — Metric scale and deployment data

- Real drone odometry/IMU supplies metric scale at inference and for future custom training labels.
- Custom GoPro/drone odom collection becomes the first metric waypoint dataset; Youtube remains
  scale-free pretraining/shape supervision.
- Language cross-attention, scheduled sampling, PI-Prober, and controller integration move here
  only after the B2 direct action policy has a validated baseline.

---

## Dependency Graph (B1 closed, B2 active)

```
DONE — B1 DATA / INFRA / DIAGNOSTICS:
  B1.1-B1.22c ✓
  B1.22e closed diagnostic-complete / model-incomplete
  B1.23/B1.24 superseded

ACTIVE — B2 SCALE-FREE FUTURE-ACTION POLICY:
  B2.0  (close B1 + activate B2 rules) ........ user approval
    └─> B2.1  (pure scale-free target contract)
          └─> B2.2  (loader + collate additive B2 action fields)
                ├─> B2.3  (direct scale-free action policy)
                └─> B2.4  (action losses, metrics, baselines)
                      └─> B2.5  (B2 trainer + local training-policy verification)
                            └─> B2.6  (USER gate: approve one B2b H20 command)
                                  └─> B2.7  (USER-GATED H20 action-policy run)
                                        └─> B2.8  (USER gate: readout + escalation decision)
                                              └─> B2.9  (USER-GATED Jetson action-policy speed check)
```

**Critical path (B2):** B2.0 → B2.1 → B2.2 → B2.3/B2.4 → B2.5 → B2.6 USER gate. No H20 run
is allowed before B2.5 passes the local training-policy gate.

**Highest-risk invariant:** future action labels are targets only. They must never be provided to
the model as conditioning inputs. The only action-like input in B2a is previous observed motion.

---

## Decisions Locked This Session

- **Clip length:** 10 seconds default (`IngestConfig.clip_length_seconds = 10.0`). At 5fps = 50
  frames/clip. Training samples are sliding windows of H+T=7 frames within each clip.
- **Encoder working default:** ViT-B/16 (D=768). Code against D=768 now. Change to ViT-S/16
  (D=384) only if the Orin NX benchmark (B1.11) shows ViT-B/16 > 20ms.
- **Predictor depth (2026-06-25 arch research):** Default depth=6 (was 12). Sweep range 6-vs-8
  (was 8-vs-12). DINO-WM (closest analogue) uses depth=6 on DINOv2 latents. ~28M params at
  D=768 (was ~50M). dropout=0.1 added (DINO-WM precedent). 196 spatial tokens retained — no
  downsampling (universal in DINO-WM/DINO-world/V-JEPA-2-AC). Waypoint head stays MLP for B-1;
  PI Prober deferred to B-2 when autoregressive rollout matters.
- **GPS Sim(3):** Stub with interface only. `normalize_scale(mode="median_speed")` is the
  active path for B-1.
- **Content filter (B-1):** Three-stage pipeline: motion ≥8.0 (primary) + YOLO-World
  `yolov8s-worldv2.pt` (semantic object rejection) + PySceneDetect `AdaptiveDetector`
  `adaptive_threshold=2.0` (SBD — tuned from 3.0; B1.10f). Each detected shot = one
  continuous camera recording; MegaSaM runs per-shot → no trajectory leaps. CLIP zero-shot
  DROPPED (scores 0.999 within-domain). Hand-crafted edit detection DELETED (histogram/motion
  spikes/block-patterns all catastrophic FP on skiing). YOLO-World: 74 FPS V100, text cached.
- **Visualization:** Self-contained Plotly HTML per clip, integrated as pipeline post-hook.
- **Delta preprocessing:** physics hard clip → median filter k=3 → velocity normalize
  (delta/dt) → per-dimension z-score normalize. Store mean/std for inference denormalization.
- **Quality weighting:** per-sample loss weights (`frame_quality` → L_latent,
  `vo_confidence` → L_wp), NOT hard exclusion. Floors: 0.1 (quality), 0.05 (confidence).
- **No global temporal smoothing** (Kalman/MA). Confidence-gated median filter only.
- **B-1 augmentation:** temporal jitter ±1 + delta noise only. Patch dropout deferred to B-2.
- **TrackVLA remains unreleased** — B-1 trains without teacher (L_latent + L_wp only).
- **L_latent beta:** smooth L1 with beta=0.1 (DINO-world 2025 precedent).
- **Mixed FPS:** velocity normalization (delta/dt) + FiLM on dt_seconds. Joint training,
  no separate stages. ~~Oversample CosFly-Track to ~40% of batches.~~ CosFly deferred from
  B-1 training loop (see CosFly Suitability Assessment). YouTube-only for B-1.
- **CosFly-Track:** DEFERRED from B-1 training. Adapter code done (trajectory-only, no RGB).
  Two blockers before integration: (1) `poses_to_deltas()` computes world-frame diffs, must
  rotate by yaw for body-frame; (2) scale mismatch (metric vs normalized). Revisit post-B1.20
  if L_wp undertrains. See assessment section.

## Open Decisions

1. **Encoder gate (BLOCKING, B1.11).** Advisory section 5 contradicts itself (6-10ms vs 50-80ms).
   Benchmark resolves. Working default D=768. Only switch to D=384 if benchmark fails.

2. **`action_id` in StepSample.** Sports data has no discrete action. B-1 uses `action_id=0`
   sentinel. B-2 revises schema to make `action_id` optional or replace with continuous action.

3. ~~**L_latent loss type.**~~ **RESOLVED:** Smooth L1 with **beta=0.1** (DINO-world 2025
   precedent, not default beta=1.0). Log cosine sim as diagnostic.

4. ~~**CosFly-Track FPS (2 Hz vs 5 Hz).**~~ **RESOLVED (mechanism), MOOT for B-1:** Velocity
   normalization (delta/dt) + FiLM conditioning on `dt_seconds` in the predictor. Joint
   training (no separate stages). ~~Oversample CosFly-Track to ~40% of batches.~~ CosFly
   deferred from B-1 — YouTube-only. Mechanism ready if CosFly is revisited post-B1.20.

8. **CosFly integration (DEFERRED).** Adapter done, training integration deferred. Two technical
   blockers: world-frame deltas + scale mismatch. Revisit if B1.20 overfit-tiny-batch shows
   L_wp is undertrained on YouTube-only data. See CosFly Suitability Assessment.

9. **Data sufficiency for YouTube-only B-1.** Per vault research §1.3, V-JEPA 2-AC trained a
   300M predictor on 62 hrs (~890K frames, 3 samples/param). Our ~57M predictor needs less but
   still wants ~10-20M frames to converge. YouTube pilot (173 sub-clips, ~8.6K frames) is
   ~1000× short of that — sufficient only for overfit-tiny-batch and **architecture validation**,
   not a deployable model. B-1 validates the architecture; the volume fight is the scale-up.
   - **CORRECTION (2026-06-28): Ego-Exo4D does NOT fit and is DEMOTED** from the "required for
     the comfortable zone" role. Ego-Exo4D is skilled-human-activity video (cooking, music,
     dance, basketball, soccer, bouldering, bike repair) — real texture but **no skiing and no
     sustained high-speed ego-translation following a subject downhill**. Its motion/scene
     structure is wrong for our follow-cam latent-dynamics task. Right texture, wrong dynamics.
   - **Revised volume path:** (a) expanded REAL YouTube follow-cam/FPV (~5h ≈ ~90K frames
     near-term — the best domain match but genuinely supply-limited); (b) **game footage
     (Steep / Riders Republic) as a Stage-1 latent-PRETRAINING source** — synthetic texture but
     the *right* structure (follow-cam, downhill, alpine, fast translation); pretrain on
     real+game → fine-tune predictor on real-only → keep the game contribution only if it
     improves real held-out val cosine (safety net against domain pollution from the
     domain-blind shared predictor). CosFly (~210K, L_wp-only) and AirSim (clean GT but needs a
     bespoke snowy + animated-athlete + scripted-follow build) remain B-3+ supplements.

5. **Scheduled sampling (B-2).** B-1 uses GT history. Deployment needs auto-regressive.
   B-2 introduces curriculum mixing GT and predicted history.

6. ~~**MegaSaM GO/NO-GO (B1.10).**~~ **RESOLVED: GO.** B1.10c E2E verified on `ski03_fpv00_c000`
   (50 frames). B1.10f fixed shot boundary detection (AdaptiveDetector 3.0→2.0) — each shot
   is now one continuous camera recording, eliminating trajectory leaps from undetected cuts.
   MegaSaM produces usable VO on per-shot skiing segments.

7. **Custom data collection timing.** Southern Hemisphere ski season July-Oct 2026. If starting
   now, plan logistics for B-3.

---

## CosFly Suitability Assessment (2026-06-24)

> Based on vault research `sports-data-pipeline-research-2026-06-19.md` §1-2, CosFly-Track
> paper (arXiv 2605.17776), and adapter implementation review.

### What CosFly-Track provides

CosFly-Track (Autel Robotics, May 2026, Apache 2.0) is 526 traces of drone-following-
pedestrian in CARLA simulator. 7 channels: RGB 1280x720, metric depth, semantic seg, GT
6-DoF pose `(x,y,z,pitch,yaw,roll)` Euler, target bbox/state, bilingual instructions,
trajectory-pair metadata. 2 Hz, ~82K frames, 16 urban town variants.

### Why CosFly is deferred from B-1

| Aspect | YouTube FPV | CosFly-Track | Verdict |
|---|---|---|---|
| **L_latent** (predict next DINOv3 latent) | Yes — real skiing frames | **No** — CARLA urban renders domain-mismatch | CosFly unusable |
| **L_wp** (predict waypoint delta) | Yes — MegaSaM VO (noisy) | Yes — GT exact (perfect) | CosFly useful |
| **Delta frame** | Body-frame (SE3 decomposition) | World-frame (simple diffs) | **Inconsistent** |
| **Scale** | Normalized (monocular, ambiguous) | Metric exact (CARLA GT) | **Incompatible** |
| **Frame count** | ~8.6K (pilot), 3.2-8.6M (full curation) | ~82K (2 Hz) | Small addition |

**Three blockers for training integration:**

1. **No latents.** `.npz` files have no `latents` key. Loader crashes on `z_t`/`z_next`
   unpacking. Must add a missing-latent code path in B1.13 and skip L_latent for these samples.

2. **World-frame vs body-frame deltas.** `poses_to_deltas()` computes `dx = x[i+1] - x[i]`
   in CARLA world frame. YouTube's `ego_motion.py` produces body-frame deltas (rotated by
   heading). Mixing both in the same L_wp batch teaches inconsistent coordinate semantics.
   **Fix:** rotate each world-frame delta by the negative of that frame's yaw.

3. **Scale mismatch.** CosFly `scale_mode="exact"` (metres). YouTube `scale_mode="normalized"`
   (median speed = 1.0). Cannot mix in the same L_wp batch without alignment. **Fix:** either
   normalize CosFly to median speed too, or add per-sample scale_mode flag and branch in loss.

### Data sufficiency without CosFly

Per vault research §1.2-1.3:
- **Minimum viable:** ~10-20M frames (0.5-1 sample/param for 22M params)
- **Comfortable:** ~30-50M frames (1.4-2.3 samples/param)
- **V-JEPA 2-AC precedent:** 62 hrs (~890K frames) for 300M predictor → 3 samples/param

**B-1 scope (YouTube pilot only):** ~8.6K frames. Far below minimum viable for a converged
model, but **sufficient for overfit-tiny-batch (B1.20) and architecture validation.** The
purpose of B-1 is proving the pipeline and architecture work, not training a deployable model.

**Full data budget (B-3):** YouTube 3.2-8.6M + Ego-Exo4D ~26M + CosFly ~82K = ~30-35M
frames → 1.4-1.6 samples/param (comfortable zone). **Ego-Exo4D is the load-bearing data
source, not CosFly.** CosFly adds <1% of total frames.

**V-JEPA 2-AC quality argument:** "Clean action labels are disproportionately valuable."
CosFly's GT deltas are perfectly clean, but serve only L_wp (which is the simpler head).
The hard part (latent prediction) gets zero benefit from CosFly. YouTube + MegaSaM with
confidence weighting is the better investment for B-1.

### Decision

- **B-1:** YouTube-only. CosFly adapter code preserved. No training integration.
- **Post-B1.20 gate:** If overfit-tiny-batch shows L_wp undertrained, revisit CosFly.
  Fix body-frame rotation + scale alignment first.
- **B-3:** CosFly is a small bonus (~82K / ~30M = 0.3% of total). Ego-Exo4D integration
  is the priority for scaling data.

---

## Ralph Loop Hand-off

**Current position (2026-06-25):** Groups 0-2 complete (B1.1-B1.10f all done, B1.19 done).
The ingest pipeline is fully operational: three-stage content filter (motion + YOLO + SBD@2.0)
→ per-shot FPV extraction → quality gate → MegaSaM VO → DINOv3 encode → .npz cache.

- **Next:** B1.11 (Orin NX benchmark — **CRITICAL GATE**, USER-GATED). Blocks B1.12→B1.15→model track.
- **Parallel with B1.11:** Nothing left to parallelize — Groups 0-2 are done. B1.13+ all blocked on B1.12.
- Remaining user-gated steps: B1.11, B1.20, B1.22, B1.23, B1.24.
- Always set `--max-iterations` backstop.
- STOP CHECK at `started_step + 3`.

---

## Verification

After plan execution:
1. `make test && make test-torch && make lint && make typecheck` — all green
2. `$PY scripts/data_quality_report.py --cache <pilot_cache>` — quality metrics visible
3. `$PY scripts/clip_report.py --cache <pilot_cache> --clip <id>` — HTML report renders (B1.9b)
4. Content filter thumbnail grids reviewed for pilot clips (B1.7b)
5. `$PY scripts/train_sports.py --overfit-tiny --max-steps 500` — loss below baseline
6. Inspect val metrics from full training run
7. Check Jetson inference speed < 50ms

---

## Research References (2026-06-20)

Sources informing B-1 design decisions. Full research notes in vault.

- **DINO-world** (arXiv 2507.19468): smooth L1 beta=0.1, variable timestamp conditioning via
  FiLM, 66M uncurated videos. Precedent for loss function + FPS handling.
- **YOLO-World** (CVPR 2024, arXiv 2401.17270): open-vocabulary object detection via
  reparam CLIP + YOLOv8. 74 FPS on V100 (49x faster than Grounding DINO). Used for B1.7b
  semantic signal (reject drones/cameras/gear). `yolov8s-worldv2.pt` via ultralytics.
- **PriVi** (CVPR 2026): CLIP+MLP on ~2,500 human labels achieves 90.3% precision for YouTube
  FPV video filtering. Historical reference only — CLIP zero-shot dropped in B1.7c (scores
  0.999 on everything within-domain).
- **Allegro**: 7-stage video processing pipeline — PySceneDetect + DOVER + CLIP alignment.
  Precedent for content filtering architecture (SBD component reused).
- **EgoVid-5M**: CLIP frame-frame consistency >= 0.7, optical flow 3-35 pixels. Quality
  thresholds for ego-centric video curation.
- **DINO-WM** (ICLR 2025): ViT-S/14, D=384, L2 loss, no augmentation, simulator only.
  Earlier baseline; superseded by DINO-world for loss choice.
- **CosFly-Track** (arXiv 2605.17776 + 2605.19120, Autel Robotics): ~82K frames public
  (HF `AutelRobotics/CosFly`), 2 Hz, CARLA urban GT. 526 traces, 7 channels, Apache 2.0.
  **Deferred from B-1 training** — trajectory-only adapter done, but domain mismatch (urban
  pedestrian ≠ skiing), missing latents, world-frame delta + scale issues. L_wp-only
  contribution ≈ 0.3% of full B-3 data budget. Revisit post-B1.20 if L_wp undertrains.
- **TrackVLA** (CoRL 2025): Visual tracking for autonomous agents. Inference code only; no
  weights/training released as of 2026-06. B-1 trains without teacher.
