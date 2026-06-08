# Phase A — data + I/O contract (executable plan)

> **Ralph reads each iteration:** `DEV_LOG.md` → `.claude/ralph-rules.md` → this file.
> **Repo:** `/home/zh/CODE/vllatent-ego-drone`. **Package:** `vllatent`.
> The vault `[[dev-decision-2026-07-latent-pred-pipeline]]` is the authoritative *why*; this file is the
> executable *what/how*. The architecture is LOCKED (`[[arch-design-2026-06-08-latent-pred]]`).

## §0. Locked context (do NOT relitigate — build around it)

Frozen DINOv3 ViT-B/16, RGB-only, 224×224, last-layer patch tokens **196×768 fp16, CACHED** (render-once
preprocess; training is sim-free). ~120M block-causal latent predictor (D=768, depth=12, heads=12, MLP
3072), H=3, T=4. Discrete action codebook → per-step FiLM. Frozen SigLIP/CLIP text tower (512→768) →
cross-attention. Continuous 4-DoF waypoint head [768→512→256→4]. Trust = deployed single-pass horizon
head; K=5 ensemble teacher + V-JEPA-2 surprise = **Phase C**. Frozen+cached encoder ⇒ **no EMA/VICReg**.
Phase-A item (1) is already done in the vault → in-repo it is *transcribe the I/O contract*, NOT re-design.
**Reuse, do NOT fork** the cloned native loader at `…/fly0-style-pipeline/third_party/AirVLN`.
**Phases A–C are standalone** — no fly0 import; the NED-body→FLU→ENU remap is *documented* for Phase D.

## §1. Env split (read before every step)

| Tier | Imports | Where it runs | CI-safe? |
|---|---|---|---|
| **pure** (`schemas, actions, frames, config, manifest, audit`) | numpy + stdlib | CI / dev host | YES — ralph closes |
| **torch** (`encode/, data/`) | + torch / transformers / timm (lazy) | dev box (5060 Ti) / H20 | code + tiny-fixture tests CI-safe; **real weights/dump dev-gated** |
| **sim** (`render/, cache`) | + airsim (lazy) + AirVLN | `fly0-m1` docker, UE4 hot, port 41451 | code-only CI-safe; **running is USER-GATED** |

Rule: the pure tier imports with numpy only (CI imports it). Every torch/sim import is lazy/guarded so
`make import-smoke` + CI stay GPU-free and AirSim-free.

## §2. Phase-A Definition of Done (verbatim, vault dev-decision §2)

1. Model architecture spec locked (§9 resolved) with a written **I/O contract** (action repr · language
   injection point · uncertainty readout · waypoint→EGO seam). **→ transcribe into `docs/io-contract.md`.**
2. **AerialVLN format audited** and confirmed to yield `(RGB obs, 4-DoF action/waypoint, next obs, language)` tuples.
3. A **data loader** emitting those tuples + a **cached frozen-encoder latent dump** for a small slice.

Phase A is DONE when these three are demonstrably met (Step 13).

## §3. Fixture strategy (so ralph closes code-only steps before any sim/GPU)

Tiny, text/array-only, must pass `scripts/check_no_blobs.sh`. These let CI/ralph close steps 2–5, 7, 9, 10:
- `fixtures/episodes/tiny_episode.json` — hand-authored 4–6-step episode with the REAL key shape
  (`episode_id, trajectory_id, scene_id, instruction.instruction_text, start_position, start_rotation`
  [w,x,y,z], `goals[].position, actions` list[int] 0–7, `reference_path` list of [x,y,z,qx,qy,qz,qw]).
  Poses constructed BY STEPPING the known action deltas so the audit's derived-Δ check has a known-good
  answer. Includes ≥1 of each action class.
- `fixtures/episodes/quaternion_trap.json` — a 2-step episode authored so a naïve (no-reorder) read of
  `start_rotation` vs `reference_path` yields a wrong yaw — the audit's quaternion test must catch it.
- `fixtures/latents/tiny_dump/` — a synthetic cached-latent dump: a few `(196,768)` fp16 arrays + a
  `manifest.json` matching the real schema. Hand-generated numpy-only (no DINOv3). Lets `vllatent.data`
  shape-tests close with no encoder.
- `fixtures/img/synthetic_224.npy` — a deterministic 224×224×3 uint8 BGR array (exercises the BGR→RGB path).

## §4. Steps (ralph-loop executable)

Each step = one ralph iteration → one commit `feat(phaseA): step N — …`. **AUTONOMOUS** = ralph closes
(CI-safe, fixtures). **USER-GATED** = stays `in_progress` until the user pastes confirmation.

### Step 1 — Scaffold + git + GitHub + codegraph  *(done during init)*
- **Do:** scaffold the repo (CLAUDE/README/DEV_LOG/ralph-rules/this plan/.gitignore/Makefile/pyproject/
  configs/docs/scripts/CI + the `vllatent` package stubs); `git init -b main` + first commit; create the
  private GitHub repo `zhihao-acc/vllatent-ego-drone` + CN-mirror remote + push `-u`; `codegraph init`+`index`.
- **DoD / test:** `git log --oneline` shows the scaffold commit; `git ls-remote origin` resolves;
  `make import-smoke && make lint && make typecheck && make test` pass; `ALL=1 bash scripts/check_no_blobs.sh`
  passes; `codegraph_status` healthy + `./.codegraph/codegraph.db` present.
- **Status:** `in_progress` (GitHub push + codegraph verify are USER-GATED / pending).

### Step 2 — Transcribe the I/O contract → `docs/io-contract.md`  *(DoD item 1)* — AUTONOMOUS
- **Do:** transcribe the four locked seams (referencing the vault arch doc as authoritative, NOT
  re-deriving): (a) **action repr** — discrete codebook 0–7 → per-step FiLM, with the enum + step
  constants; (b) **language injection** — frozen SigLIP/CLIP text tower 512→768 → cross-attention;
  (c) **uncertainty readout** — deployed single-pass horizon head; K=5 ensemble + V-JEPA-2 = Phase C
  (documented, not built); (d) **waypoint→EGO seam** — continuous 4-DoF (Δx,Δy,Δz,Δψ) in AirSim-NED body
  + the *documented-only* NED-body→FLU→ENU remap for Phase D (explicitly **NOT executed in Phase A**).
  Pin the loader output tuple (arch §6 item 5) + a "Frame & convention hazards" section (both foot-guns).
- **DoD / test:** `test -f docs/io-contract.md && grep -q "NOT executed in Phase A" docs/io-contract.md && grep -q "w-FIRST" docs/io-contract.md && grep -q "BGR" docs/io-contract.md`.

### Step 3 — Pure-tier tuple SCHEMAS (`vllatent/schemas.py`) — AUTONOMOUS
- **Do:** frozen dataclasses for the loader output tuple, numpy-typed, no torch: `StepSample` =
  `(z_t, history_latents, lang_tokens, action_id, z_next, delta_4dof, future_frame_rgb)` with documented
  shapes/dtypes (`z_t`/`z_next` `(196,768) fp16`; `history` `(3,196,768) fp16`; `action_id ∈ [0,7]`;
  `delta_4dof` `(4,) f32` body-frame). Add `EpisodeRecord` + `CacheManifestEntry`. Each field comments its
  frame/dtype/order.
- **DoD / test:** `pytest -q tests/test_schemas.py` (construct each with synthetic arrays; assert
  shapes/dtypes/immutability; manifest-entry JSON round-trip). `make import-smoke` numpy-only.

### Step 4 — Discrete→continuous-4-DoF action mapping (`vllatent/actions.py`) — AUTONOMOUS
- **Do:** transcribe constants VERBATIM from `third_party/AirVLN/airsim_plugin/airsim_settings.py`
  (FORWARD/LEFT/RIGHT=5, UP_DOWN=2, TURN=15) and reproduce `utils/env_utils.py::getPoseAfterMakeAction`
  arithmetic in pure numpy (no airsim import). Body-frame, yaw-only, NED z-down:
  `MOVE_FORWARD=(+5,0,0,0)`; `MOVE_LEFT/RIGHT=±5` body-lateral (yaw±90°); `GO_UP=(0,0,−2,0)` /
  `GO_DOWN=(0,0,+2,0)`; `TURN_LEFT=−15°`, `TURN_RIGHT=+15°`; `STOP`=identity. Provide
  `action_to_delta(id)→(4,)` and `apply_delta(pose, id)` reproducing env_utils.
- **DoD / test:** `pytest -q tests/test_actions.py` — assert each delta vs enum+constants; assert applied
  pose matches env_utils' arithmetic for a known start (forward `unit_z==0`, ±90° lateral, z-sign up/down,
  ±15° yaw); STOP is identity.

### Step 5 — AerialVLN-JSON AUDIT parser (`vllatent/audit.py`)  *(DoD item 2 — fixture half)* — AUTONOMOUS
- **Do:** `parse_episode(json)→EpisodeRecord` + `audit_episode(record)→AuditReport`. PIN foot-gun #1
  (reorder `start_rotation` [w,x,y,z] vs `reference_path` [...,qx,qy,qz,qw] into canonical xyzw, assert
  consistency); confirm `actions[step]`↔`reference_path[step]` alignment; derive Δ from consecutive poses
  and VERIFY it matches the quantized action delta from `vllatent.actions` within tol; emit `AuditReport`
  (per-action counts, tuple completeness, quaternion verdict, Δ-mismatch list, scene_id range, splits).
  Wire the CLI `python -m vllatent.audit --episode <json> [--report <out|->]`.
- **DoD / test:** `pytest -q tests/test_audit.py` against `fixtures/episodes/tiny_episode.json` (clean
  report, all action classes, derived-Δ matches) AND `fixtures/episodes/quaternion_trap.json` (audit FLAGS
  the would-be wrong-yaw if reorder skipped). Then `make audit` runs clean.

### Step 6 — Obtain the dataset JSON (download a real slice from S3) — **USER-GATED**
- **Do:** finish `scripts/fetch_aerialvln_json.sh` (pull a SLICE of split JSONs from
  `aerialvln.s3.ap-southeast-2.amazonaws.com/dataset/aerialvln/`; keep first N episodes). Confirm + record
  **CC BY-NC-SA 4.0** in `docs/io-contract.md` + `DEV_LOG.md`. Do NOT commit the JSON (gitignored).
- **DoD / test (USER pastes):**
  ```
  bash scripts/fetch_aerialvln_json.sh train 50 data/aerialvln_json
  python -m vllatent.audit --episode data/aerialvln_json/train.slice.json --report -
  ```

### Step 5b — Run the audit on the REAL slice  *(DoD item 2 — real half)* — **USER-GATED**
- **Do:** no new code — run `vllatent.audit` over the real slice; confirm real keys match the
  fixture-pinned schema + the derived-Δ check passes on real data. Record the `AuditReport` in DEV_LOG + vault.
- **DoD / test (USER pastes):** `python -m vllatent.audit --episode data/aerialvln_json/train.slice.json --report data/audit_report.json` → 0 mismatches, all action classes, quaternion verdict OK.

### Step 7 — DINOv3 encoder wrapper (`vllatent/encode/dinov3.py`) — SPLIT
- **Do:** `DinoV3Encoder` — frozen DINOv3 ViT-B/16, RGB 224² → `(196,768) fp16` (eval/no_grad/fp16). Enforce
  foot-gun #2 (RGB input; BGR→RGB upstream + assert at the boundary). Lazy torch/transformers import.
  `make encode-smoke` runs on `fixtures/img/synthetic_224.npy` (CPU ok), asserting shape/dtype. Weights via
  `HF_ENDPOINT=https://hf-mirror.com`.
- **DoD / test:** AUTONOMOUS — `pytest -q tests/test_encode_contract.py` (monkeypatched backbone asserts
  224×224×3 RGB → `(196,768) fp16`, BGR→RGB boundary, `requires_grad=False`). USER-GATED — real-weight
  smoke `HF_ENDPOINT=… make encode-smoke` on the dev box prints `(196,768) torch.float16`.

### Step 8 — Render harness (`vllatent/render/harness.py`) — SPLIT
- **Do:** wrap the AirVLN teleport+capture (reuse `AirVLNSimulatorClientTool.py`; do NOT modify). Per pose:
  build `airsim.Pose` with the reordered quaternion (xyzw; foot-gun #1) →
  `simSetVehiclePose(ignore_collision=True, vehicle_name='Drone_1')` → `simGetImages` on `"front_0"`
  `ImageType.Scene` → uint8 → HxWx3 → `cv2.cvtColor(BGR2RGB)` (foot-gun #2) → 224². **Lock every client
  call** (foot-gun #3). Scene-dependent depth (1&7 DepthVis; else DepthPerspective clip[0,100]/100), raw only.
- **DoD / test:** AUTONOMOUS — `pytest -q tests/test_render_unit.py` (mock client: quaternion reorder
  round-trips a known [w,x,y,z]; BGR→RGB applied; Lock serializes every call via a spy; frame 224×224×3 RGB
  uint8). USER-GATED — live render in `fly0-m1` (user launches scene, waits 41451):
  `python -m vllatent.render --episode fixtures/episodes/tiny_episode.json --scene 1 --out /tmp/render_smoke/`.

### Step 9 — Render→encode→CACHE + manifest/provenance (`vllatent/cache.py`) — SPLIT
- **Do:** orchestrate episode → render each pose (8) → encode RGB (7) → write `(196,768) fp16` per step +
  `manifest.json` (episode_id, scene_id, encoder id+revision, dataset slice id, quaternion order, BGR→RGB
  flag, render config hash, frame count). Deterministic + resumable (skip cached).
- **DoD / test:** AUTONOMOUS — `pytest -q tests/test_cache_manifest.py` (mocked render+encode build from
  `tiny_episode.json`; assert layout + manifest schema (via `vllatent.manifest.validate_manifest`) +
  provenance + resumability no-op). USER-GATED — small-slice build (sim+GPU):
  `HF_ENDPOINT=… python -m vllatent.cache build --episodes data/aerialvln_json/train.slice.json --limit 5 --scenes-root /opt/aerialvln --out data/latent_cache/`.

### Step 10 — Cached-latent torch Dataset / LOADER (`vllatent/data/loader.py`)  *(DoD item 3 — code)* — AUTONOMOUS
- **Do:** torch `Dataset` over a cache dir emitting the full `StepSample` (step 3): build H=3 history
  windows + T=4 horizon targets; pad/mask episode boundaries; load fp16; tokenize language per the text-tower
  contract. Lazy torch import. `python -m vllatent.data inspect --cache <dir> --n N`.
- **DoD / test:** `pytest -q tests/test_data_shapes.py` over `fixtures/latents/tiny_dump/` (every tuple has
  the §3 shapes/dtypes; H=3 windowing + T=4 targets align; boundary mask correct; `action_id∈[0,7]`;
  `delta_4dof` matches `vllatent.actions`).

### Step 11 — Loader over the REAL small-slice dump  *(DoD item 3 — real half)* — **USER-GATED**
- **Do:** no new code — run the loader over the real cache from step 9; confirm well-formed tuples
  end-to-end. Record a sample-batch summary in DEV_LOG.
- **DoD / test (USER pastes):** `python -m vllatent.data inspect --cache data/latent_cache/ --n 4`.

### Step 12 — SIZE the full render→cache job (gated bulk-run sign-off) — SPLIT
- **Do:** `docs/full-run-sizing.md`: **frames** (~845k across 25 scenes — pin exact from the real split
  JSONs after step 6), **GPU-hours** (frames ÷ measured DINOv3-B/16 fp16 throughput on 5060 Ti vs H20),
  **storage** (845k × 196 × 768 × 2 B ≈ ~248 GB fp16 — verify vs measured per-frame size from step 9).
  `scripts/run_full_cache.sh` **refuses** without `--i-have-signed-off`.
- **DoD / test:** AUTONOMOUS — `test -f docs/full-run-sizing.md && grep -q "GB" docs/full-run-sizing.md`;
  `bash scripts/run_full_cache.sh` exits non-zero with "sign-off required" when the flag is absent.
  USER-GATED — the bulk run is a separate explicit user decision (do NOT bulk-execute).

### Step 13 — Final Phase-A DoD verification — **USER-GATED**
- **Do:** no new code — assemble evidence: (1) `docs/io-contract.md` (step 2); (2) clean `AuditReport` on
  the REAL slice (step 5b); (3) `vllatent.data` yields valid tuples from the REAL cached dump (step 11).
  Record the Phase-A completion entry in DEV_LOG + vault.
- **DoD / test (USER pastes all three):**
  ```
  head -40 docs/io-contract.md
  python -m vllatent.audit --episode data/aerialvln_json/train.slice.json --report -
  python -m vllatent.data inspect --cache data/latent_cache/ --n 2
  ```
  Stays `in_progress` until the user pastes all three; ralph does NOT auto-flip Phase A done.

## §5. Run order + where ralph STOPS

```
[1 scaffold]
  ├─ 2 io-contract ─┐
  ├─ 3 schemas → 4 actions → 5 audit(fixture) ─┤  ALL AUTONOMOUS
  │                                            └─ 6 fetch JSON → 5b audit(real)   USER-GATED
  ├─ 7 encode (contract AUTO | real-weights GATED)
  ├─ 8 render  (unit AUTO | live GATED)
  ├─ 9 cache   (manifest AUTO | small-slice GATED)
  ├─ 10 loader (shapes AUTO) → 11 loader on real dump  USER-GATED
  ├─ 12 sizing (AUTO | bulk GATED)
  └─ 13 Phase-A DoD  USER-GATED (final sign-off)
```

Ralph closes autonomously, in order: **2 → 3 → 4 → 5**, then the AUTONOMOUS halves of **7, 8, 9**, then
**10**, then the sizing+guard half of **12**. Natural STOP points: (a) after Step 5 — ask the user for
Step 6 + 5b; (b) after 7/8/9 code lands — user launches docker+UE4+GPU; (c) Step 13 — stays `in_progress`
until the three §2 outputs are pasted (mirror the fly0 "done-with-caveat" rule). A `blocked` step (a real
audit mismatch on the real slice) sends ralph back to fix `actions.py` / the quaternion reorder — not defer.

## §6. Load-bearing foot-guns (also in `docs/io-contract.md`)
1. **Quaternion order:** `start_rotation`=[w,x,y,z]; `reference_path`=[…,qx,qy,qz,qw]; `airsim.Quaternionr` is xyzw → reorder to xyzw. `quaternion_trap.json` fails loudly if skipped.
2. **BGR→RGB:** AirSim `Scene` is BGR (AirVLN leaves `cvtColor` commented); DINOv3 needs RGB. Wrong colours silently poison every cached latent.
3. **Single-threaded msgpack-RPC:** wrap every `client.X()` in a `threading.Lock`.
4. **Scene-dependent depth:** scenes 1 & 7 = DepthVis; others = DepthPerspective (clip[0,100]/100). Encoder is RGB-only — bites only if depth is cached for Phase C.
5. **NED z-down sign:** `GO_UP=−z`, `GO_DOWN=+z`.
6. **No frame-leak:** NED→FLU→ENU remap is documented-only; do NOT import fly0 or execute it in Phase A.
7. **action↔reference_path index alignment:** assert it (env_utils relies on it), don't assume.
8. **No blobs:** only tiny fixtures committed; everything rendered/downloaded/encoded is gitignored.

## §7. Reuse (do not modify — read/replay only)
- `…/third_party/AirVLN/utils/env_utils.py` — ground-truth `getPoseAfterMakeAction` (authority for `actions.py` + audit Δ-check).
- `…/third_party/AirVLN/airsim_plugin/airsim_settings.py` — verbatim action enum + step constants.
- `…/third_party/AirVLN/airsim_plugin/AirVLNSimulatorClientTool.py` — teleport+capture pattern `render/harness.py` wraps.
- `…/unified-pipeline/` — scaffold/workflow conventions this repo mirrors.
