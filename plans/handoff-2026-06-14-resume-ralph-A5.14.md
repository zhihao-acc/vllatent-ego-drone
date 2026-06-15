# Cold-start handoff — resume the ralph loop at A5.14 (render→encode→teacher→cache)

> **Created 2026-06-14.** Paste the one-liner at the very bottom into a NEW session, or just say
> "continue the ralph loop." This is the operator brief; the authoritative state lives in the repo files
> it points to. **A5.12 / A5.13 / A5.13b all landed and are real-weight-verified since the last handoff** —
> all five cache-pipeline components (DINOv3 vision, CLIP text, WorldVLN teacher, V-JEPA-2 verifier, AirSim
> render) now have GREEN live smokes. Lowest pending = **A5.14**.

## 0. What this is

Continue the `vllatent-ego-drone` **Phase-A.5 ralph loop**. Lowest pending = **A5.14 — render →
[DINOv3 + CLIP-text + WorldVLN + V-JEPA-2] → cache the teacher/oracle dataset + complete the provenance
manifest (M5)**. Autonomous half = orchestration code + a MOCKED `tests/test_cache_manifest.py`; the
small-slice real build is USER-GATED. Then A5.16–A5.18.

## 1. Read FIRST (authoritative, cheap → expensive)

1. `DEV_LOG.md` — newest on top; the step-status table is the source of truth. Expect: 1–6+5b done,
   A5.1–A5.13 + A5.13b done, A5.15 done, **A5.14 / A5.16 / A5.17 / A5.18 pending**.
2. `.claude/ralph-rules.md` — per-iteration protocol, quality gates, the USER-GATED rule, Test Command Index.
3. `plans/phase-a5-replan-postpivot.md` — each step's tier / gate / DoD / exact test command / deps
   (incl. the A5.13b text-tower entry added 2026-06-14).
4. `CLAUDE.md` — repo invariants, PURE/TORCH/SIM tier split, foot-guns.

## 2. Current position (verified; pushed @ `origin/main`)

- **DONE & real-weight-verified:** A5.1–A5.9 (pure seams), **A5.10** DINOv3 (`(196,768)` fp16 smoke),
  **A5.11** WorldVLN teacher (live K-rollout on H20), **A5.12** V-JEPA-2 verifier (`surprise [0.174,0.208]`
  cuda smoke, `37f6664`), **A5.13** AirSim render (live: 8 frames from `tiny-0001`; user fixes `7e31bf3`),
  **A5.13b** CLIP text tower (`lang_tokens (10,768)` cuda smoke; `c6e745b`), **A5.15** distillation loader
  (DEFINES the `.npz` read-contract A5.14 writes).
- **Gates:** pure `make test` = **239**; `make test-torch` = 5; ruff/mypy(pure)/import-smoke/blob clean.
  Pure env = `PY=/home/zh/miniconda3/envs/vln-ego-drone-1.1/bin/python` (Py3.9 — **no runtime `X|Y` in
  type ALIASES**; annotations OK via `from __future__`). Torch env = `vllatent-ego-drone` (Py3.10,
  transformers 5.10.2 / torch 2.12+cu130 — the drift to PIN in A5.14).
- Loop runs INLINE (no `.claude/ralph-loop.local.md`). Adversarial panels ran on A5.10–A5.13b.

## 3. Resume here — A5.14 (cache build; `vllatent/cache.py` is currently a `NotImplementedError` stub)

**DoD (re-plan §Group 8):** Tier SIM+TORCH · manifest AUTO / small-slice USER-GATED. Orchestrate, per
episode: render each `reference_path` pose → **normalize to 224²** → DINOv3-encode → run WorldVLN teacher +
V-JEPA-2 verifier → write the per-episode `.npz` + the per-step `OracleTarget` fields + update the manifest
(teacher/render provenance populated). Resumable (skip already-cached).

**The `.npz` write-contract — EXACT (from `vllatent/data/loader.py`, the A5.15 reader; do not drift):**
per episode of N poses, `<cache_dir>/<latent_path>.npz` with:
- `latents` (N,196,768) fp16 · `actions` (N,) int · `deltas` (N,4) f32 · `lang_tokens` (M,768) fp16
- `waypoint_4dof` (N,4) f32 · `teacher_pose6` (N,6) f32 · `rollpitch_resid` (N,) f32 ·
  `disagreement` (N,) f32 · `vjepa_surprise` (N,) f32

Plus `<cache_dir>/manifest.json` = `build_manifest(...)` with one entry per episode (`episode_id`,
`n_frames`=N, `latent_path`, `trajectory_id`). Per-transition arrays are stored pose-aligned at length N;
the terminal STOP slot (t=N-1) is unused (the loader excludes it).

**The five components + their verified APIs (all standalone; never import fly0):**
- **render** — `vllatent.render.harness.RenderHarness(host,port,vehicle_name).render_reference_row(row)`
  → `(H,W,3)` uint8 RGB. ⚠ **returns sim-native `(480,640)`, NOT 224²** (see §4).
- **vision** — `vllatent.encode.dinov3.DinoV3Encoder(device).encode_rgb(rgb_224)` → `(196,768)` fp16.
- **text** — `vllatent.encode.text.ClipTextEncoder(device).encode(instruction)` → `(M,768)` fp16 (ONCE per
  episode; `lang_tokens`).
- **teacher** — `vllatent.teacher.worldvln.WorldVLNTeacherClient(server).k_rollout_segment(frames,
  instruction, k, seed_base)` → `(K,T,6)` seam; `teacher_outputs_from_rollouts(...)` → per-step
  `TeacherOutput`. Needs the H20 server (USER-GATED — see §4). 6→4 projection: `waypoint_4dof` = drop
  roll/pitch + rad→deg yaw; `rollpitch_resid` = |roll|+|pitch|; `disagreement` =
  scalarize `TeacherOutput.rollout_spread()` over the student-relevant channels.
- **verifier** — `vllatent.verify.vjepa2.VJEPA2SurpriseVerifier(device).scalar_surprise(context_rgb,
  future_rgb)` → float `vjepa_surprise` (per step; `[0,2]`).
- **deltas** — GT `(dx,dy,dz,dyaw)` body delta per transition via
  `vllatent.actions.pose_pair_to_body_delta` over consecutive `reference_path` poses.

**Test (mocked, AUTO):** `tests/test_cache_manifest.py` — mock the five seams, build a tiny 1–2 episode
cache to `tmp_path`, assert the `.npz` keys/shapes/dtypes EXACTLY match the loader contract and round-trip
through `CachedLatentDataset` (reuse the synthetic-cache pattern in `tests/test_data_shapes.py`), and that
`validate_manifest` is clean with teacher/render provenance populated.

**Test (real small-slice, USER-GATED):** `HF_ENDPOINT=https://hf-mirror.com $PY -m vllatent.cache build
--slice data/aerialvln_json/train.slice.json --limit 5 --scenes-root /opt/aerialvln --out
data/latent_cache/` — needs sim (fly0-m1) + GPU + the WorldVLN H20 server + V-JEPA-2/DINOv3/CLIP weights
all together. Emit the command block; do NOT drive it.

## 4. Load-bearing facts A5.14 depends on

- **⚠ RENDER RESOLUTION (the #1 thing to get right).** The live render returns `(480,640,3)`, not 224².
  DINOv3's `encode_rgb` would force-`interpolate` to 224² and **distort aspect ratio**; V-JEPA-2's processor
  instead resize-shortest-edge + center-crops. **Normalize ONCE at the render→encode boundary —
  center-crop to square, then resize to 224²** — and feed that to BOTH DINOv3 and the V-JEPA-2 context/future
  frames so they see the same pixels. **Record the transform in the manifest** (training-playbook foot-gun
  #1: log frame transforms). Alternatively the operator sets AirSim `settings.json` CaptureSettings=224²,
  but the code should still defensively normalize. Do NOT let the encoder silently distort.
- **`lang_tokens` 512→768.** CLIP ViT-B/32 text width is 512; `ClipTextEncoder` zero-pads to 768 (the real
  512→768 map is the student's learned cross-attn, Phase B). Cache stores the (M,768) fp16 as-is.
- **Teacher = the H20 WorldVLN server (REUSABLE from A5.11).** AutoDL container
  `autodl-container-9ef943a6c4`, conda env `worldvln`, clone `~/WorldVLN.code`, weights
  `/root/autodl-tmp/WorldVLN` (`WorldVLN_backbone/{backbone(4-shard),vae}` + `WorldVLN_action_decoder.pt`),
  T5 `/root/autodl-tmp/flan-t5-xl`, config `/tmp/worldvln_server_config.json` (released config, `infinity.ckpt`
  BLANKED — env `INFINITY_CKPT` is IGNORED unless that field is empty), launch `bash infer/run_server.sh`
  → uvicorn :8001; dev box reaches it via `ssh -N -L 8001:127.0.0.1:8001 <h20>`. Client needs no torch.
  K-rollout = K sessions × seed stride 65537; 1 segment(16 actions)/call; 49/16 ⇒ 3 segments/session.
- **Manifest provenance.** `build_manifest` ALREADY records (from Config): encoder `model_id` +
  `text_model_id`, `disagreement_source`, `vjepa2_model_id`. A5.14 must still populate the build-time stubs:
  `worldvln_model_id`, `worldvln_revision`, `render_config_hash` (M5 completes here).
- **PIN the `[torch]` extra (do it in/with A5.14).** `pyproject.toml [torch]` is unpinned and pulled
  transformers 5.10 / torch 2.12+cu130 on the box (spec wanted `transformers>=4.56,<5`, `torch>=2.8,<2.13`
  cu12x). Reconcile the pin vs the H20/Jetson targets before/with the cache build.
- **Cache writes are torch-free** (numpy `.npz`); torch/airsim/transformers enter only via the lazy
  encoders/verifier/render. Keep `vllatent/cache.py` import-guarded so a pure box imports it.

## 5. Rules that bound the work (unchanged)

- **Per-iteration protocol** (ralph-rules): READ → IDENTIFY lowest pending → REVIEW DoD → EXECUTE
  (lazy heavy imports; mocked contract test closes in CI before any real sim/GPU) → TEST → RECORD in
  `DEV_LOG.md` → COMMIT `feat(phaseA): A5.N — …` (specific `git add`, never `-A`) → STOP CHECK
  (`started_step+3` / user-gated / tier boundary) → push (`git push origin main`).
- **USER-GATED** (never auto-done; emit command blocks): A5.14 small-slice, A5.16, A5.17 bulk, A5.18.
  **SSH HANDS-OFF / docker / UE4 / GPU rental = user-only** — paste blocks only.
- **fly0 reference-first** for AirSim/bridge: consult `/home/zh/CODE/vln-ego-drone/fly0-style-pipeline`
  (+ `third_party/AirVLN`) and copy/re-derive into THIS repo; never import; never modify the upstream clones.
- **Pure gate:** `make PY=$PURE_PY import-smoke lint typecheck test` stays green (239 now). Torch tier:
  `make test-torch` (the pure env HAS torch 2.8 but NOT transformers — keep torch-marked tests
  transformers-free, or they must be pure with a mocked seam like the V-JEPA-2/CLIP/teacher contract tests).
- **LOCKED:** gate = the contribution; student = latent-pred transformer; DINOv3/CLIP = frozen cached
  encoders; no EMA/VICReg; no blobs; `StepSample`/`OracleTarget` seams are user-approved — do not relitigate.

## 6. Open items (carry-over)

- **WorldVLN weights license** (`license:other`): clarification email SENT, awaiting reply. Blocks
  publication, not plumbing.
- **`[torch]` extra unpinned** → pin with A5.14 (see §4).
- **Render resolution** 480×640 → square-crop+resize to 224² at the cache boundary (see §4).
- Rotate the HF token pasted in chat 2026-06-09 (never committed; verified absent from the repo).

---

### Paste this into the new session

```
Continue the vllatent-ego-drone Phase-A.5 ralph loop. Read plans/handoff-2026-06-14-resume-ralph-A5.14.md, then DEV_LOG.md + .claude/ralph-rules.md + plans/phase-a5-replan-postpivot.md, and resume at the lowest pending step A5.14 (render→[DINOv3+CLIP-text+WorldVLN+V-JEPA-2]→cache + provenance manifest). A5.14 is SIM+TORCH: write vllatent/cache.py orchestration (per episode: render reference_path poses → CENTER-CROP-TO-SQUARE + RESIZE to 224² → DINOv3 encode + CLIP-text encode + WorldVLN k-rollout teacher + V-JEPA-2 surprise → write the per-episode .npz EXACTLY per the vllatent/data/loader.py read-contract + update build_manifest provenance) with LAZY heavy imports, and a MOCKED tests/test_cache_manifest.py that mocks the five seams and asserts the .npz keys/shapes/dtypes round-trip through CachedLatentDataset + validate_manifest clean. The manifest/mocked half is autonomous; the small-slice real build (sim + GPU + H20 WorldVLN server + weights) is USER-GATED — emit the command block, do not drive it. PIN the [torch] extra in pyproject with this step. Keep the pure gate green: make PY=/home/zh/miniconda3/envs/vln-ego-drone-1.1/bin/python import-smoke lint typecheck test (239 tests; pure env is Py3.9 — no runtime X|Y in type aliases). STOP CHECK at started_step+3 or any user-gated step. Run iterations INLINE (no ralph-loop.local.md).
```
