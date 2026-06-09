# Cold-start handoff — resume the ralph loop at A5.10 (first TORCH-tier step)

> **Created 2026-06-09.** Paste the one-liner at the very bottom into a NEW session, or just say
> "continue the ralph loop." This is the operator brief; the authoritative state lives in the repo
> files it points to. The pure/contract lane (A5.1–A5.9) is **done, green, pushed**; A5.10 begins the
> **torch tier**, which is a real environment + tooling transition — read §4 before running anything.

## 0. What this is

Continue the `vllatent-ego-drone` **Phase-A.5 ralph loop**. The plan is user-signed-off. Steps
**A5.1–A5.9 are done** (+ A5.7 real-slice verified); resume at the lowest pending step **A5.10**.

## 1. Read FIRST (authoritative, cheap → expensive)

1. `DEV_LOG.md` — newest entries on top; the step-status table is the source of truth for which step is
   pending. Expect A5.1–A5.9 = `done`, old steps 7–13 = `superseded`, A5.10–A5.18 = `pending`.
2. `.claude/ralph-rules.md` — the per-iteration protocol + quality gates + the user-gated rule + the
   Test Command Index.
3. `plans/phase-a5-replan-postpivot.md` — the executable plan: each step's **tier / gate / DoD / exact
   test command / deps**. A5.10 = **§3 Group 8** (re-spec'd 7–13 successors).
4. `CLAUDE.md` — repo invariants + the PURE/TORCH/SIM tier split + the load-bearing foot-guns.

Confirm current code via codegraph (`codegraph_status` / `codegraph_search`) before trusting any cited
line — the index lags writes ~1s.

## 2. Current position (verified, pushed @ `a954256` on `origin/main`)

- **DONE & green:** A5.1 (`f1f74e6` M1 frames primitives), A5.2 (`e5a2f6a` M2 no-flip CI gate),
  A5.3 (`2a89b68` H1/H2 typed Config), A5.4 (`149d311` M5 typed manifest builder), A5.5 (`b2c6646`
  H3 student seams), A5.6 (`ba0dc04` M4 StepSample masks), `f4108c8` (A5.4–A5.6 review hardening),
  A5.8 (`d7701fc` WorldVLN probe resolved), A5.7 (`34a7c4a` M3 AuditSummary — real-slice VERIFIED
  50/50, 10198 transitions, 0 Δ-mismatch), A5.9 (`a954256` TeacherOutput/OracleTarget seam + Config
  finalized).
- **Pure sweep = 181 tests**; ruff + mypy(6 files) + blob-guard + manifest CLI + `make audit` all clean.
- **Loop ran INLINE** (no `.claude/ralph-loop.local.md` was created → no promise-flush race). Working
  tree clean.

## 3. Resume here — A5.10 (and the A5.11+ teacher/distillation pipeline)

Lowest pending = **A5.10 — DINOv3 student-encoder wrapper** (re-plan §3, was step 7).
- Tier **TORCH** (lazy import) · contract **AUTO** / real-weight **USER-GATED**.
- **DoD:** `vllatent/encode/dinov3.py` — frozen DINOv3 **ViT-B/16**, RGB 224²→`(196,768)` fp16, the
  **BGR→RGB** boundary enforced, **torch import LAZY** (module imports on a torch-free box). Blocks A5.14.
- **Test (contract, AUTO):** `tests/test_encode_contract.py` with a **monkeypatched backbone** (no real
  weights) — assert the BGR→RGB conversion + the `(196,768)` fp16 output shape/dtype. Mark
  `@pytest.mark.torch` (runs under `make test-torch`, not the pure CI gate).
- **Test (real-weight, USER-GATED):** `HF_ENDPOINT=https://hf-mirror.com make encode-smoke` — downloads
  DINOv3 weights + runs a real forward. **Produce the command block; do NOT drive the download.**
- Then **A5.11** (frozen WorldVLN teacher wrapper, USER-GATED server), **A5.12** (V-JEPA-2 verifier),
  **A5.13** (render harness), **A5.14** (render→[DINOv3+WorldVLN+V-JEPA-2]→cache + provenance) — see
  the re-plan dependency graph.

## 4. TIER TRANSITION — A5.10 leaves the pure lane (load-bearing)

- The pure lane used `vln-ego-drone-1.1` (Py3.9) — it has numpy/ruff/mypy/pytest but **may NOT have
  torch**. A5.10's **contract test needs torch** (even monkeypatched, `@pytest.mark.torch`). Check
  `PY=.../vln-ego-drone-1.1/bin/python -c "import torch"`; if absent, the torch-tier tests run in the
  **`vllatent-ego-drone` env (Py3.10 / torch 2.8 / cu12x / transformers≥4.56 / timm≥1.0.20)** on the
  **dev box (RTX 5060 Ti 16 GB)** — set up via `make setup-torch`.
- **Keep the pure gate green regardless:** `vllatent/encode/dinov3.py` must NOT add a module-level torch
  import (lazy/inside-function only), so a torch-free box still imports the package. `make PY=$PY
  import-smoke lint typecheck test` (pure, in `vln-ego-drone-1.1`) must stay green — it does NOT import
  `vllatent.encode.*`. The torch-tier tests are a separate lane: `make test-torch` / `pytest -m torch`.
- **USER-ONLY:** SSH / docker / UE4 / GPU rental / multi-GB weight downloads → emit a command block; the
  user runs them and pastes back. China network: `export HF_ENDPOINT=https://hf-mirror.com`.

## 5. Per-iteration protocol (from ralph-rules)

READ → IDENTIFY lowest pending → REVIEW the DoD/test → EXECUTE (**lazy torch import**; contract test with
a mocked backbone closes in CI before any real weight) → TEST (the step command; fix failures in-iteration)
→ RECORD in `DEV_LOG.md` (flip status + dated entry) → COMMIT `feat(phaseA): A5.N — …` with **specific
`git add`, never `-A`** → STOP CHECK (`started_step + 3`, OR a user-gated step, OR a tier boundary) → push.

## 6. Environment (load-bearing)

- **Pure gate:** `PY=/home/zh/miniconda3/envs/vln-ego-drone-1.1/bin/python`; run make as `make PY=$PY
  import-smoke lint typecheck test`, `make PY=$PY audit`, `ALL=1 bash scripts/check_no_blobs.sh`.
- **Torch tier (A5.10+):** the `vllatent-ego-drone` Py3.10+torch env on the dev box / H20 (`make
  test-torch`). **SSH HANDS-OFF** — paste blocks, never drive ssh/docker/GPU.
- GitHub push works direct (origin = https `zhihao-acc/vllatent-ego-drone`). HF via `hf-mirror.com`.

## 7. LOCKED — do not relitigate (incl. the A5.8/A5.9 decisions)

- The calibrated single-pass trust-horizon **gate is the contribution.** Backbone = **WorldVLN frozen
  teacher → distilled student**; **student = the latent-prediction transformer** (+ waypoint + trust
  heads); **DINOv3 = the student's FROZEN cached encoder, NOT the student.** No EMA / no VICReg.
- **A5.8 facts (verified):** WorldVLN inference is **stochastic by default** ⇒ K-rollout disagreement is
  **free** (vary the seed; `disagreement_source = worldvln_rollout`, AirScape = contingency); the teacher
  action head is **6-DoF `[roll,yaw,pitch,x,y,z]`** (SE(3)) vs the 4-DoF student; ckpts `INFINITY_CKPT` +
  `ACTIONHEAD_CKPT`; lang enc = **T5**; **license SPLIT** (code CC BY 4.0 / weights `license:other` —
  clarification email SENT, awaiting reply; blocks publication, not Phase-A plumbing).
- **A5.9 seam (user-approved):** `OracleTarget = {waypoint_4dof (4,) f32, teacher_pose6 (6,),
  rollpitch_resid, disagreement, vjepa_surprise}`; the **6→4 projection** (drop roll/pitch + abs→body-delta
  + verify roll/pitch≈0) and the disagreement scalarization **execute at A5.14** (cache build), not in the
  seam.
- **Foot-guns:** waypoint = AirSim-NED body, yaw-only (remap is Phase D); **BGR→RGB** before DINOv3
  (record the flag in the manifest); quaternion order (`start_rotation` wxyz → canonical xyzw;
  `reference_path` = 6-wide Euler `[x,y,z,pitch,roll,yaw]`, len==len(actions)); AirSim msgpack-RPC is
  single-threaded → Lock every `client.X()` (render tier). No blobs committed. Don't modify the sibling
  repos / their `third_party/`.

## 8. Loop control

- Deterministic stop = `rm .claude/ralph-loop.local.md` (if you launch via the `ralph-loop:ralph-loop`
  skill — its setup rejects shell subshell chars, so keep `--max-iterations`-style args **free of parens
  and backticks**). Running the iterations INLINE (as the prior two sessions did) avoids the promise-flush
  race entirely and is the simpler path for the pure/contract steps.
- ALWAYS set a `--max-iterations` backstop. STOP CHECK at `started_step + 3`, at any user-gated step, or
  at the pure→torch tier boundary (A5.10 IS that boundary).

## 9. Open items (carry-over)

- **WorldVLN weights license:** clarification email SENT — awaiting the authors' reply. Build continues
  (license-agnostic in Phase A); gate any publication/release of the distilled student on the answer.
- **A5.7:** done (real-slice verified). **A5.8:** done.

---

### Paste this into the new session

```
Continue the vllatent-ego-drone Phase-A.5 ralph loop. Read plans/handoff-2026-06-09-resume-ralph-A5.10.md, then DEV_LOG.md + .claude/ralph-rules.md + plans/phase-a5-replan-postpivot.md, and resume at the lowest pending step A5.10 (DINOv3 student-encoder wrapper). A5.10 is the first TORCH-tier step: write vllatent/encode/dinov3.py with a LAZY torch import + the BGR->RGB boundary + (196,768) fp16 output, and tests/test_encode_contract.py with a MONKEYPATCHED backbone (mark @pytest.mark.torch) — the contract half is autonomous; the real-weight encode-smoke (HF_ENDPOINT=https://hf-mirror.com make encode-smoke) is USER-GATED so produce a command block, do not drive the download. Keep the pure gate green: make PY=/home/zh/miniconda3/envs/vln-ego-drone-1.1/bin/python import-smoke lint typecheck test must still pass (no module-level torch import). STOP CHECK at started_step+3 or the tier boundary. Deterministic stop is rm .claude/ralph-loop.local.md; set --max-iterations 5.
```
