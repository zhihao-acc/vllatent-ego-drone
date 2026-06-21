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
| A5.12 (V-JEPA-2 verifier) | **SURVIVES** | Independent of teacher choice. Phase C trust oracle. |
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
pre-train (B-2), Ego-Exo4D integration (B-3), scheduled sampling (B-2/B-3), trust head
training (Phase C), closed-loop deployment (Phase D), CosPress distillation training (only if
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
  - PySceneDetect `AdaptiveDetector` for shot boundary detection (SBD)
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

**B1.8 — CosFly-Track download + adapter.**
- Tier TOOL+PURE / USER-GATED (HF download)
- Download 526-trace subset from `AutelRobotics/CosFly`. Write adapter that converts CosFly
  format (CARLA, GT 6-DoF, 2 Hz) to ingest `.npz` cache. CosFly has GT waypoints, so adapter
  writes `deltas` from GT poses, `vo_confidence = 1.0`. Uses `build_manifest_wild_video`
  with `motion_method="cosfly_gt"`.
- **Files:** `vllatent/ingest/cosfly_adapter.py` (new), `scripts/download_cosfly.sh` (new), `tests/test_cosfly_adapter.py` (new)
- **DoD:** Adapter converts entries to `.npz`. Manifest built. 10+ traces converted and inspected.
- **Test:** `$PY -m pytest -q tests/test_cosfly_adapter.py`. User verifies real download + conversion.
- **Deps:** blocks B1.10, B1.12. Blocked-by: none.

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

**B1.10 — MegaSaM VO validation on pilot clips.**
- Tier RESEARCH / USER-GATED
- Run MegaSaM on 3-5 YouTube pilot clips. Inspect 3D trajectory shapes (cumulative deltas).
  Compare against expected motion (e.g., downhill ski run should be roughly linear+descending).
  If any GPS-paired clip is available, run `evo` ATE. Produce GO / CONDITIONAL-GO / NO-GO
  verdict on MegaSaM for skiing FPV. If NO-GO, evaluate DPVO fallback.
- **Files:** written findings note (not committed — vault or dev log)
- **DoD:** Trajectory plots reviewed. Verdict written.
- **Test:** User inspects plots + renders verdict.
- **Deps:** blocks B1.12. Blocked-by: B1.7.

---

### Group 3 — Encoder Decision Gate

**B1.11 — Benchmark DINOv3 ViT-B/16 on Orin NX.**
- Tier RESEARCH / USER-GATED (requires Orin NX hardware access)
- The advisory is contradictory: section 5.2 says 50-80ms, but section 4 says ~6-10ms based on NVIDIA
  benchmarks. Must measure empirically. Run `DinoV3Encoder` with TensorRT FP16 export.
  Measure median/p99 latency at batch=1.
- **Decision:** If ViT-B/16 TRT FP16 < 20ms then keep ViT-B/16, D=768, predictor depth 6 (~50M).
  If > 20ms then use Meta's pre-distilled ViT-S/16 (`vit_small_patch16_dinov3.lvd1689m` via timm),
  D=384, predictor depth 8 (~25M). No CosPress training needed either way.
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
- **Batch construction:** mix CosFly + YouTube sources; oversample CosFly-Track to ~40% of
  batches (CosFly has GT deltas = implicit clean-data curriculum).
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
  D, depth, heads from `PredictorConfig`. No language cross-attention in B-1.
- **Files:** `vllatent/model/predictor.py` (new), `tests/test_predictor.py` (new)
- **DoD:** Output shape correct. Param count matches expected (~25M at D=384 or ~50M at D=768).
  Block-causal mask verified (future can't attend to later future). Action FiLM changes output.
  dt FiLM changes output when dt varies.
- **Test:** `$PY -m pytest -q tests/test_predictor.py -m torch`
- **Deps:** blocks B1.17. Blocked-by: B1.12.

**B1.16 — Waypoint head + trust head stub.**
- Tier TORCH / AUTO
- `vllatent/model/heads.py`:
  - `WaypointHead`: MLP `D->256->128->4`. Takes `(B,T,D)` -> `(B,T,4)` predicted deltas.
  - `TrustHead` (stub): Returns `TrustReadout(p_commit=ones(T), k_star=T, sigma=0)`.
    Phase C replaces with real head.
- **Files:** `vllatent/model/heads.py` (new), `tests/test_heads.py` (new)
- **DoD:** Shapes correct. Trust stub outputs all-commit.
- **Test:** `$PY -m pytest -q tests/test_heads.py -m torch`
- **Deps:** blocks B1.17. Blocked-by: B1.12.

**B1.17 — Full model assembly.**
- Tier TORCH / AUTO
- `vllatent/model/sports_model.py` with `SportsFollowingModel(nn.Module)`. Assembles predictor
  + waypoint head + trust head. Forward takes `TrainingBatch` -> `(PredictorOutput,
  predicted_deltas (B,T,4), TrustReadout)`. Encoder is NOT part of forward (latents cached).
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

### Group 8 — Full Training + Verification

**B1.22 — Full training run: CosFly-Track + YouTube pilot.**
- Tier TORCH / USER-GATED (H20)
- Train on all available data (CosFly 526 traces + YouTube pilot clips). Scene-split sacred
  val: hold out 2-3 complete clips. AdamW, cosine LR, batch size fit to GPU. Log train/val
  loss, per-step cosine sim, waypoint L1 error, speed (steps/sec). Run on AutoDL H20.
  Training playbook: "scene-split sacred val", "boring HP", "track speed."
- **DoD:** Trained checkpoint. Val loss plateaus. Latent cosine sim > 0.7 on val. Waypoint L1
  within 2x of training-set L1 on val.
- **Test:** User monitors H20. Reviews val metrics.
- **Deps:** blocks B1.23. Blocked-by: B1.20, B1.8.

**B1.23 — Jetson inference speed check.**
- Tier RESEARCH / USER-GATED (Orin NX access)
- Export trained model to TorchScript/ONNX, run on Orin NX. Measure encoder (frozen) +
  predictor + waypoint head end-to-end. Target: < 50ms (20 Hz).
  Training playbook: "track speed (Jetson)."
- **DoD:** Written benchmark. GO / CONDITIONAL-GO for deployment speed.
- **Test:** User runs on Orin NX.
- **Deps:** blocks B1.24. Blocked-by: B1.22.

**B1.24 — Phase B-1 DoD verification.**
- Tier DOC / USER-GATED
- Verify: (1) pipeline bugs fixed, (2) schemas revised, (3) encoder decision locked,
  (4) data acquired + quality validated, (5) cache pipeline green, (6) predictor trains
  (overfit passes, full train converges), (7) val metrics acceptable, (8) Jetson speed OK.
- **DoD:** Written Phase B-1 completion note. All tests green.
- **Test:** `make test && make test-torch && make lint && make typecheck`
- **Deps:** blocked-by all B1.x.

---

## Phase B-2 & B-3 (high-level)

### B-2: Language Cross-Attention
- RefCOCOg warm-start: download RefCOCOg (~237K expressions), train Flamingo-style zero-init
  gated cross-attention (every other predictor block) on grounding task.
- Auto-captioning pipeline: SAM-2 + VLM on sports clips -> verified (frame, expression, mask)
  triplets. ~5-20K pairs.
- Integrate cross-attention into predictor. Zero-init gating -> graceful visual-only fallback.
- **Scheduled sampling for history latents**: begin mixing GT + predicted history with increasing
  predicted fraction. This bridges the gap between GT-history training (B-1) and auto-regressive
  deployment (Phase D).
- Validate: language conditioning improves prediction on captioned val clips.

### B-3: Domain Fine-Tune + Scale
- Expand YouTube curation (50-100 clips). Include MTB/snowboard for diversity.
- Ego-Exo4D sports subset integration (soccer, basketball, bouldering — ~26M frames).
- Fine-tune language-conditioned model on full dataset.
- Ablation: measure contribution of each data source + language conditioning.
- Custom GoPro+IMU collection (Southern Hemisphere ski season July-Oct for 2026 timing).

---

## Dependency Graph (B-1)

```
PARALLEL TRACK A (pipeline fixes + content filter, can start immediately):
  B1.1 (undistort wire) ---+
  B1.2 (megasam conf)  ---+
  B1.3 (GPS Sim3)      ---+      B1.7a (batch encode) --+
  B1.4 (clip length)   ---+---> B1.7b (content filter) -+--> B1.7c (FPV extract) --> B1.7 (pilot) --> B1.10
  B1.5 (config revise) ---+                                                                |
  B1.6 (SportsTarget)  ---+                                                                |
                                                                                            |
PARALLEL TRACK B (data + quality dashboards, can start immediately):                       |
  B1.8 (CosFly adapter) ---------------------------------------------------------------+  |
  B1.9 (quality report) ---------------------------------------------------------------+  |
  B1.9b (HTML clip report) — no deps, runs on any .npz --------------------------------+  |
                                                                                        |  |
PARALLEL TRACK C (CRITICAL -- start ASAP):                                             |  |
  B1.11 (Orin NX bench) ---> B1.12 (lock D) --+---------------------------------------+  |
                                                |                                       |  |
GROUP 4 (loader, needs D locked + schema):     |                                       |  |
  B1.13 (sports loader) <-- B1.4,B1.6,B1.12   |                                       |  |
  B1.14 (collate) <-- B1.13                    |                                       |  |
                                                |                                       |  |
GROUP 5 (model, needs D locked):               |                                       |  |
  B1.15 (predictor) <-- B1.12                  |                                       |  |
  B1.16 (heads) <-- B1.12                      |                                       |  |
  B1.17 (assembly) <-- B1.15, B1.16            |                                       |  |
                                                |                                       |  |
GROUP 6 (training):                             |                                       |  |
  B1.18 (losses) <-- B1.17                     |                                       |  |
  B1.19 (checkpoint) -- no block                |                                       |  |
  B1.20 (overfit-tiny) <-- B1.14,B1.17,B1.18,B1.19,data ---------------------------+  |  |
  B1.21 (sanity+viz) <-- B1.13,B1.18           |                                    |  |  |
                                                |                                    |  |  |
GROUP 8 (full training + verify):              |                                    |  |  |
  B1.22 (full train) <-- B1.20, B1.8          -+                                    |  |  |
  B1.23 (Jetson speed) <-- B1.22               |                                    |  |  |
  B1.24 (B-1 DoD) <-- ALL <-------------------+------------------------------------+--+--+
```

**Critical path:** B1.11 -> B1.12 -> B1.15 -> B1.17 -> B1.18 -> B1.20 -> B1.22 -> B1.24
(unchanged — B1.7b is on the data track, not the critical path)

**Parallel tracks before encoder gate:** Track A (B1.1-B1.7b(motion+YOLO)-B1.7c-B1.7), Track B (B1.8-B1.9-B1.9b), Track C (B1.11)

---

## Decisions Locked This Session

- **Clip length:** 10 seconds default (`IngestConfig.clip_length_seconds = 10.0`). At 5fps = 50
  frames/clip. Training samples are sliding windows of H+T=7 frames within each clip.
- **Encoder working default:** ViT-B/16 (D=768). Code against D=768 now. Change to ViT-S/16
  (D=384) only if the Orin NX benchmark (B1.11) shows ViT-B/16 > 20ms.
- **GPS Sim(3):** Stub with interface only. `normalize_scale(mode="median_speed")` is the
  active path for B-1.
- **Content filter (B-1):** Motion (primary) + YOLO-World `yolov8s-worldv2.pt` (semantic object
  rejection) + PySceneDetect `AdaptiveDetector` (SBD). CLIP zero-shot DROPPED — scores 0.999
  within-domain, no discriminative power. YOLO-World: 74 FPS V100, ~1.5 GB VRAM, text cached.
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
  no separate stages. Oversample CosFly-Track to ~40% of batches.

## Open Decisions

1. **Encoder gate (BLOCKING, B1.11).** Advisory section 5 contradicts itself (6-10ms vs 50-80ms).
   Benchmark resolves. Working default D=768. Only switch to D=384 if benchmark fails.

2. **`action_id` in StepSample.** Sports data has no discrete action. B-1 uses `action_id=0`
   sentinel. B-2 revises schema to make `action_id` optional or replace with continuous action.

3. ~~**L_latent loss type.**~~ **RESOLVED:** Smooth L1 with **beta=0.1** (DINO-world 2025
   precedent, not default beta=1.0). Log cosine sim as diagnostic.

4. ~~**CosFly-Track FPS (2 Hz vs 5 Hz).**~~ **RESOLVED:** Velocity normalization (delta/dt) +
   FiLM conditioning on `dt_seconds` in the predictor. Joint training (no separate stages).
   Oversample CosFly-Track to ~40% of batches via weighted sampling.

5. **Scheduled sampling (B-2).** B-1 uses GT history. Deployment needs auto-regressive.
   B-2 introduces curriculum mixing GT and predicted history.

6. **MegaSaM GO/NO-GO (B1.10).** If skiing VO is unusably noisy, evaluate DPVO fallback.
   Escalate before training.

7. **Custom data collection timing.** Southern Hemisphere ski season July-Oct 2026. If starting
   now, plan logistics for B-3.

---

## Ralph Loop Hand-off

- Start at **B1.1** (pipeline bug fix, pure-tier cheap-win).
- **B1.7b (content filter, motion+YOLO-World) is AUTO** — can run immediately, unblocks B1.7.
- **B1.9b (HTML report) is AUTO** — can run anytime, no blockers.
- **B1.11 (Orin NX benchmark) can run in parallel** with Group 0-2.
- User-gated steps: B1.7, B1.8, B1.10, B1.11, B1.20, B1.22, B1.23, B1.24.
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
- **CosFly-Track**: ~100K frames public (HF `AutelRobotics/CosFly`), 2 Hz, CARLA GT.
  Primary synthetic data source for B-1.
- **TrackVLA** (CoRL 2025): Visual tracking for autonomous agents. Inference code only; no
  weights/training released as of 2026-06. B-1 trains without teacher.
