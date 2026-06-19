# Phase B1: Sports-Following Dataset Pipeline

## Context

The project pivoted from indoor AerialVLN to autonomous drone following of a sportsperson (skiing primary). Phase A plumbing (253 pure + 5 torch tests) survives — schemas, encoder, config, manifest, loader are all green. But the data pipeline is entirely different: source is real FPV video (not AirSim renders), action labels come from MegaSaM visual odometry (not discrete oracle actions), there are no language instructions, and no WorldVLN teacher.

**Goal:** Build a pipeline that takes YouTube skiing FPV videos and produces a high-quality cached latent dataset for training. Quality above all — even overfitting on clean data produces convincing demos.

**User decisions (locked):**
1. Cache with DINOv3 ViT-B/16 latents (768-dim); re-encode later if CosPress ViT-S/16 happens
2. Target person tracking deferred to B2
3. Seed dataset: 10-20 hand-curated YouTube skiing FPV clips

## Architecture

```
YouTube URL
  → yt-dlp download
  → ffmpeg frame extraction @ 5 FPS
  → [optional] fisheye undistortion
  → quality scoring (blur, exposure, whiteout)
  → MegaSaM ego-motion extraction → SE(3) camera poses
  → SE(3) → body-frame deltas (dx, dy, dz, dyaw)
  → scale normalization (median displacement = 1.0)
  → DINOv3 ViT-B/16 encode → (196, 768) fp16
  → .npz cache per clip + manifest
```

## Package Layout

```
vllatent/sports/
    __init__.py         # PURE: package marker
    schemas.py          # PURE: SportsSample, SportsClipRecord, SportsClipManifestEntry
    config.py           # PURE: SportsDataConfig
    quality.py          # PURE: frame quality scoring + filtering
    ego_motion.py       # PURE: SE(3) → body-frame deltas, scale normalization
    acquire.py          # TOOL: yt-dlp wrapper
    preprocess.py       # TOOL: ffmpeg frame extraction + undistortion
    megasam.py          # TOOL: MegaSaM subprocess wrapper
    encode.py           # TORCH: batch DINOv3 encoding (reuses DinoV3Encoder)
    cache.py            # ORCH: .npz assembly + manifest
    loader.py           # PURE: SportsDataset (map-style, emits SportsSample)
    pipeline.py         # ORCH: end-to-end per-clip processing
    __main__.py         # CLI: process / batch / inspect
```

## Implementation Steps

### B1.0 — Scaffold + tier wiring
Create `vllatent/sports/__init__.py`. Add `test-sports-pure` and `test-sports-tool` Makefile targets. Add `@pytest.mark.tool` auto-skip to `conftest.py`. Add `sports_data/` to `.gitignore`. Verify `import vllatent.sports` on pure box.

**Files:** `vllatent/sports/__init__.py`, `Makefile`, `tests/conftest.py`, `.gitignore`, `pyproject.toml`
**Test:** `python -c "import vllatent.sports"`
**Tier:** PURE

### B1.1 — Sports schemas (PURE)
New `SportsSample` frozen dataclass — the sports-following training tuple:
```python
@dataclass(frozen=True, eq=False)
class SportsSample:
    z_t: np.ndarray              # (196,768) fp16
    history_latents: np.ndarray  # (H,196,768) fp16
    history_mask: np.ndarray     # (H,) bool
    z_next: np.ndarray           # (196,768) fp16
    delta_4dof: np.ndarray       # (4,) f32 — body-frame (dx,dy,dz,dyaw) from MegaSaM
    vo_confidence: float         # MegaSaM VO confidence [0,1]
    frame_quality: float         # composite quality [0,1]
    dt_seconds: float            # time delta to next frame
```

Key difference from `StepSample`: no `action_id`, no `lang_tokens`/`lang_mask`. Shares `PATCH_TOKENS`, `EMBED_DIM`, `HISTORY`, `LATENT_DTYPE`, `DELTA_DTYPE` from `vllatent/schemas.py` — no duplication.

Also: `SportsClipRecord` (parsed clip metadata) and `SportsClipManifestEntry` (cache entry, with `to_dict()`/`from_dict()` round-trip).

**Files:** `vllatent/sports/schemas.py`, `tests/test_sports_schemas.py`
**Reuses:** `_check_array` pattern from `vllatent/schemas.py`
**Test:** `pytest tests/test_sports_schemas.py -q`
**Tier:** PURE

### B1.2 — Sports config (PURE)
`SportsDataConfig` frozen dataclass with: `raw_dir`, `frames_dir`, `cache_dir`, `clips_yaml`, `target_fps=5.0`, `min_clip_seconds=10.0`, `resolution_hw=(720,1280)`, `megasam_model`, `undistort_model`.

Integrate into `Config` as an optional `sports: SportsDataConfig | None = None` field. Add `"sports"` to `_SECTIONS` with optional handling (absent in YAML → `None`). Existing `Config()` unchanged — all 253 pure tests stay green.

New `configs/sports.yaml` override file. New empty `configs/sports_clips.yaml` template.

**Files:** `vllatent/sports/config.py` (new), `vllatent/config.py` (extend), `configs/sports.yaml`, `configs/sports_clips.yaml`, `tests/test_sports_config.py`
**Test:** `pytest tests/test_sports_config.py tests/test_config.py -q`
**Tier:** PURE

### B1.3 — SE(3) to body-frame deltas (PURE)
Core math module. MegaSaM outputs SE(3) camera poses (OpenCV convention: X-right, Y-down, Z-forward). Convert to body-frame deltas matching `delta_4dof` format.

Functions:
- `se3_to_body_delta(T_prev, T_curr) → (4,) f32`
- `camera_to_drone_body(R, t) → (R_body, t_body)` — OpenCV cam → NED body
- `rotation_to_yaw(R) → float` — extract yaw, drop pitch/roll
- `se3_sequence_to_deltas(poses, fps) → (N-1, 4) f32`
- `normalize_scale(deltas, mode="median_speed") → (N-1, 4) f32`
- `sim3_align(poses, trajectory_metric) → (scale, R, t)` — stub for future GPS

Reuses `wrap_pi()` from `vllatent/frames.py`.

**Files:** `vllatent/sports/ego_motion.py`, `tests/test_sports_ego_motion.py`
**Test:** `pytest tests/test_sports_ego_motion.py -q`
**Tier:** PURE

### B1.4 — Frame quality scoring (PURE)
Frame-level quality metrics, all numpy (no cv2 at runtime for the core scorers):
- `motion_blur_score(frame)` — Laplacian variance via `np.gradient`, [0,1]
- `exposure_score(frame)` — histogram spread, [0,1]
- `snow_whiteout_score(frame)` — fraction near-white pixels, [0,1]
- `composite_quality(frame)` — weighted combination
- `filter_frames(qualities, threshold=0.3) → (N,) bool`

Thresholds tunable via config (added to `SportsDataConfig`).

**Files:** `vllatent/sports/quality.py`, `tests/test_sports_quality.py`
**Test:** `pytest tests/test_sports_quality.py -q`
**Tier:** PURE

### B1.5 — Video acquisition (TOOL)
yt-dlp subprocess wrapper. `download_clip(url, out_dir) → ClipMetadata`. Parses yt-dlp `--dump-json` for metadata (duration, fps, resolution). `download_batch(clips_yaml, out_dir)` processes a YAML clip list. No yt-dlp Python API import — subprocess only.

**Files:** `vllatent/sports/acquire.py`, `tests/test_sports_acquire.py`
**Test:** `pytest tests/test_sports_acquire.py -q -m tool`
**Tier:** TOOL

### B1.6 — Frame extraction + preprocessing (TOOL)
ffmpeg subprocess for frame extraction at target FPS. `extract_frames(video, out_dir, target_fps) → FrameExtraction`. Optional `undistort_fisheye(frame, K, D)` using cv2 (lazy import). Camera params per clip in `clips_yaml`.

**Files:** `vllatent/sports/preprocess.py`, `tests/test_sports_preprocess.py`
**Test:** `pytest tests/test_sports_preprocess.py -q -m tool`
**Tier:** TOOL

### B1.7 — MegaSaM integration (TOOL)
Subprocess wrapper for MegaSaM. `run_megasam(frame_dir, out_dir) → MegaSamResult`. Parses MegaSaM output (SE(3) poses as .npy, per-frame confidence). `MegaSamResult` dataclass with `poses: (N,4,4)`, `confidences: (N,)`, `intrinsics: (3,3)`.

**Highest-risk step** — MegaSaM is research code with an unstandardized output format. The wrapper is structured to adapt once the actual format is confirmed. If MegaSaM proves unsuitable, the wrapper can be swapped for DPVO/DROID-SLAM with the same interface.

**Files:** `vllatent/sports/megasam.py`, `tests/test_sports_megasam.py`
**Test:** `pytest tests/test_sports_megasam.py -q -m tool`
**Tier:** TOOL

### B1.8 — Cache assembly + manifest (PURE + TORCH)

Per-clip `.npz` format:
```
latents:       (N, 196, 768) fp16
deltas:        (N-1, 4)      f32
vo_confidence: (N,)          f32
frame_quality: (N,)          f32
timestamps:    (N,)          f64
quality_mask:  (N,)          bool
```

`build_sports_manifest()` in `vllatent/manifest.py` — same validation infrastructure, different required sections. `motion_source` section replaces `teacher`:
```json
{
  "cache_version": "0.2",
  "encoder": { "model_id": "vit_base_patch16_dinov3.lvd1689m", ... },
  "dataset": { "name": "sports_following", "sport": "skiing", ... },
  "convention": { "color_order": "RGB", "frame": "camera_body" },
  "motion_source": { "method": "megasam", "model": "...", "scale_mode": "normalized" },
  "entries": [{ "clip_id": "ski01", "n_frames": 150, "latent_path": "ski01.npz", ... }]
}
```

Batch encoding wrapper in `vllatent/sports/encode.py` calls existing `DinoV3Encoder.encode_rgb()`.

**Files:** `vllatent/sports/cache.py`, `vllatent/sports/encode.py`, `vllatent/manifest.py` (extend), `tests/test_sports_cache.py`
**Reuses:** `DinoV3Encoder.encode_rgb()`, `build_manifest()` pattern, `validate_manifest()` infrastructure
**Test:** `pytest tests/test_sports_cache.py tests/test_cache_manifest.py -q`
**Tier:** PURE (assembly) + TORCH (encoding)

### B1.9 — Sports loader (PURE)
`SportsDataset` map-style dataset reading the `.npz` cache, emitting `SportsSample`. History windowing with zero-padding at clip start. Quality-mask filtering (only emit frames that pass quality gate). CLI inspect: `python -m vllatent.sports inspect --cache <dir> --n 5`.

**Files:** `vllatent/sports/loader.py`, `tests/test_sports_loader.py`
**Reuses:** `CachedLatentDataset` pattern from `vllatent/data/loader.py`
**Test:** `pytest tests/test_sports_loader.py -q`
**Tier:** PURE

### B1.10 — End-to-end pipeline (ORCH)
Wire all stages: acquire → extract → quality → megasam → ego_motion → encode → cache → manifest.

CLI:
```
python -m vllatent.sports process --url "https://..." --clip-id ski01 --config configs/sports.yaml
python -m vllatent.sports batch --clips configs/sports_clips.yaml --config configs/sports.yaml
python -m vllatent.sports inspect --cache sports_data/latent_cache/ --n 5
```

Resumable (skips completed clips). Clear error messages per stage.

**Files:** `vllatent/sports/pipeline.py`, `vllatent/sports/__main__.py`, `tests/test_sports_pipeline.py`
**Test:** `pytest tests/test_sports_pipeline.py -q`
**Tier:** ORCH

### B1.11 — First real clip (USER-GATED)
Process 1 YouTube skiing FPV clip end-to-end. Inspect every intermediate output. Inspection script `scripts/inspect_sports_clip.py` plots: frame quality histogram, VO confidence over time, 3D trajectory, delta statistics.

**DoD:** User signs off on output quality.

### B1.12 — Seed dataset (USER-GATED)
Curate 10-20 clips. Process in batch. Review quality summary. Adjust thresholds. Target: >5000 accepted frames.

### B1.13 — DoD verification
All tests green (pure + torch + tool). Lint + typecheck clean. No blobs committed. Seed dataset cached with valid manifest.

## Dependency Graph

```
B1.0 (scaffold) ─┬─ B1.1 (schemas) ──────────────────┐
                  ├─ B1.2 (config) ─┬─ B1.5 (acquire) │
                  │                 ├─ B1.6 (preproc)  │
                  │                 └─ B1.7 (megasam)  │
                  ├─ B1.3 (ego_motion) ────────────────┤
                  └─ B1.4 (quality) ───────────────────┤
                                                       │
B1.8 (cache+encode) ← needs B1.1, B1.3, B1.4 ─────────┤
B1.9 (loader) ← needs B1.1, B1.8 ─────────────────────┤
B1.10 (pipeline) ← needs B1.5-B1.9 ───────────────────┘
B1.11 (first clip) ← needs B1.10
B1.12 (seed dataset) ← needs B1.11
B1.13 (DoD) ← needs B1.12
```

**Critical path:** B1.0 → B1.1 → B1.3 → B1.8 → B1.10 → B1.11
**Parallel groups:** {B1.1, B1.2, B1.3, B1.4} after B1.0; {B1.5, B1.6, B1.7} after B1.2

## Key Reuse Points

| Existing code | Reused in | How |
|---|---|---|
| `vllatent/schemas.py` `_check_array()` | `vllatent/sports/schemas.py` | Import directly for validation |
| `vllatent/schemas.py` constants (`PATCH_TOKENS`, `EMBED_DIM`, `HISTORY`, etc.) | `vllatent/sports/schemas.py` | Import, no duplication |
| `vllatent/encode/dinov3.py` `DinoV3Encoder.encode_rgb()` | `vllatent/sports/encode.py` | Wrap for batch processing |
| `vllatent/frames.py` `wrap_pi()` | `vllatent/sports/ego_motion.py` | Import for yaw wrapping |
| `vllatent/manifest.py` validation infrastructure | `vllatent/manifest.py` (extended) | Add `build_sports_manifest()` alongside existing |
| `vllatent/config.py` `Config.from_yaml()` pattern | `vllatent/config.py` (extended) | Add optional `sports` section |
| `vllatent/data/loader.py` `CachedLatentDataset` pattern | `vllatent/sports/loader.py` | Same lazy-load + history-window design |

## What B1 Does NOT Build

- Target person tracking / bounding boxes (B2)
- TrackVLA teacher distillation (B2)
- Unified training loop for sports + AerialVLN (B2)
- CosPress encoder distillation (B2 or separate)
- V-JEPA-2 surprise on sports data (Phase C)
- GPS/IMU metric scale alignment (future, when custom data collected)
- Deblurring (quality filter rejects blurred frames instead)

## Verification

1. `make test` — all 253 pure tests green (existing + new `test_sports_*.py`)
2. `make test-torch` — 5 torch tests green + sports encode test
3. `make lint && make typecheck` — clean
4. `python -m vllatent.sports inspect --cache sports_data/latent_cache/ --n 10` — shapes/dtypes correct
5. `scripts/inspect_sports_clip.py` — visual inspection of trajectories and quality distributions
6. Manual review of 3-5 clips' MegaSaM trajectories plotted in 3D
