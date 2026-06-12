# Cold-start handoff — resume the ralph loop at A5.12 (V-JEPA-2 verifier)

> **Created 2026-06-11.** Paste the one-liner at the very bottom into a NEW session, or just say
> "continue the ralph loop." This is the operator brief; the authoritative state lives in the repo files
> it points to. A5.10/A5.11/A5.13(mock)/A5.15 all landed since the last handoff; **A5.11's live teacher
> smoke is user-verified GREEN on the H20** — the disagreement signal is confirmed end-to-end.

## 0. What this is

Continue the `vllatent-ego-drone` **Phase-A.5 ralph loop**. Lowest pending = **A5.12 — V-JEPA-2 surprise
verifier wrapper** (contract half AUTONOMOUS / real-weight USER-GATED), then the operator infra block
(A5.13-live render → A5.14 cache build) → A5.16–A5.18.

## 1. Read FIRST (authoritative, cheap → expensive)

1. `DEV_LOG.md` — newest on top; the step-status table is the source of truth. Expect: 1–6+5b done,
   A5.1–A5.11 done, **A5.13 in_progress** (mock half done, live USER-GATED), A5.15 done,
   **A5.12 / A5.14 / A5.16 / A5.17 / A5.18 pending**.
2. `.claude/ralph-rules.md` — per-iteration protocol, quality gates, the user-gated rule, Test Command Index.
3. `plans/phase-a5-replan-postpivot.md` — each step's tier / gate / DoD / exact test command / deps.
4. `CLAUDE.md` — repo invariants, PURE/TORCH/SIM tier split, foot-guns.

## 2. Current position (verified; pushed @ `origin/main`)

- **DONE & green:** A5.1–A5.9 (see prior handoff), **A5.10** (`e8af4f4` — encoder = timm NON-GATED
  `vit_base_patch16_dinov3.lvd1689m`, real encode-smoke GREEN `(196,768) fp16 cuda`), **A5.15** (`2ced77e`
  — distillation loader; DEFINES the per-episode `.npz` cache read-contract A5.14 must write), **A5.13
  mock half** (`84eb251` — render harness, 9 unit tests; live render still USER-GATED), **A5.11**
  (`76d7022` + closure — teacher client + **live K-rollout smoke user-verified**: K=5×T=16, 5 distinct
  rollouts, step-0 spread all-channels > 0, `[teacher-smoke] OK`).
- **Gates:** pure `make test` = **212** tests; `make test-torch` = 5; ruff/mypy(pure)/import-smoke/blob clean.
- Loop runs INLINE (no `.claude/ralph-loop.local.md`). Adversarial verify panels ran on A5.10/A5.13+A5.15/
  A5.11 — all findings fixed before push.

## 3. Resume here — A5.12 (V-JEPA-2 surprise verifier)

**A5.12 (re-plan §3 Group 8):** Tier TORCH (lazy) · contract AUTO / real-weight USER-GATED.
- **DoD:** `vllatent/verify/vjepa2.py` — frozen V-JEPA-2 (ViT-L) surprise `s_j = 1 − cos(ẑ_j, z_j)` on GT
  future frames; lazy imports; feeds `OracleTarget.vjepa_surprise` (≥ 0, validated in schemas).
- **Test (contract, AUTO):** monkeypatched backbone (follow `tests/test_encode_contract.py` /
  `test_teacher_contract.py` patterns — mock at a `_load_*` seam, tests in the pure gate where no torch
  is needed, `@pytest.mark.torch` where it is).
- **Test (real, USER-GATED):** `HF_ENDPOINT=https://hf-mirror.com $PY -m vllatent.verify.vjepa2 --frames
  fixtures/...` — emit the command block; do NOT drive the download. **Check gating first** (DINOv3 lesson:
  Meta gate rejected us → timm re-host saved A5.10; V-JEPA-2 HF repos may also be gated — research the
  exact model id + a non-gated fallback BEFORE writing the loader).
- Blocks A5.14. Then: **A5.13-live** (sim box) and **A5.14** (render→[DINOv3+WorldVLN+V-JEPA-2]→cache;
  **pin the `[torch]` extra in pyproject BEFORE/with A5.14** — unbounded bounds pulled transformers 5.10 /
  torch 2.12+cu130, drift vs H20 cu12x + Jetson).

## 4. A5.11 facts the next steps depend on (live-verified 2026-06-11)

- **Wire ≠ seam ≠ student units (foot-gun):** wire = `[dx_cm,dy_cm,dz_cm,droll_deg,dyaw_deg,dpitch_deg]`
  position-FIRST (cm, deg) per-step **DELTAS**; seam `TeacherOutput.rollouts_pose6` = `[roll,yaw,pitch,x,y,z]`
  **(m, rad)** deltas (`wire_actions_to_pose6` converts); student `delta_4dof` = (m, **deg** yaw).
  **A5.14's 6→4 projection = drop roll/pitch (+ verify `rollpitch_resid`≈0) + rad→deg yaw** — the
  abs→body-delta step is GONE (wire is already deltas; SE(3) integration is offline-only upstream).
- **K-rollout protocol:** released config locks the seed per session ⇒ K rollouts = K sessions
  (`session_id` distinct, `reset_session=True`) × seeds `seed_base + k*65537`. One segment (16 actions)
  per call; 49/16 ⇒ points [1,17,33,49] ⇒ 3 segments/session; `allow_future_segments=true` = strict
  closed loop (1 frame+instruction → 16 actions; +16 real frames → next 16). `segment_index=-1` = warmup.
- **H20 server (REUSABLE for A5.14):** AutoDL container `autodl-container-9ef943a6c4`, conda env `worldvln`,
  clone `~/WorldVLN.code`, weights `/root/autodl-tmp/WorldVLN` (**layout:** `WorldVLN_backbone/backbone/`
  4-shard safetensors, `WorldVLN_backbone/vae/model.safetensors`, `WorldVLN_action_decoder.pt`), T5
  `/root/autodl-tmp/flan-t5-xl`, server config `/tmp/worldvln_server_config.json` (released config with
  `infinity.ckpt` BLANKED — **`INFINITY_CKPT` env is IGNORED unless the config field is empty**), launch
  `bash infer/run_server.sh` → uvicorn :8001. `ts_ckpt_loaded=false` until the first predict (lazy stage2)
  — expected. Client side: dev box via `ssh -N -L 8001:127.0.0.1:8001 <h20>`; the client needs NO torch.
- **Teacher client API:** `WorldVLNTeacherClient(server).k_rollout_segment(frames, instruction, k, seed_base)`
  → `(K,T,6)` seam + `teacher_outputs_from_rollouts(...)` → per-step `TeacherOutput`.

## 5. Rules that bound the work (unchanged)

- **Per-iteration protocol** (ralph-rules): READ → IDENTIFY lowest pending → REVIEW DoD → EXECUTE
  (lazy heavy imports; mocked contract test closes in CI before any real weights) → TEST → RECORD in
  `DEV_LOG.md` → COMMIT `feat(phaseA): A5.N — …` (specific `git add`, never `-A`) → STOP CHECK
  (`started_step+3` / user-gated / tier boundary) → push.
- **USER-GATED** (never auto-done; emit command blocks): A5.12-real, A5.13-live, A5.14 small-slice,
  A5.16, A5.17 bulk, A5.18. **SSH HANDS-OFF** — paste blocks only.
- **fly0 reference-first** for anything AirSim/bridge: consult
  `/home/zh/CODE/vln-ego-drone/fly0-style-pipeline` (+ its `third_party/AirVLN`) and copy/re-derive into
  THIS repo (never import; never modify upstream clones — incl. `/tmp/worldvln_code`).
- **Pure gate:** `PY=/home/zh/miniconda3/envs/vln-ego-drone-1.1/bin/python`; `make PY=$PY import-smoke
  lint typecheck test` must stay green (the pure env is **Py3.9**: no runtime `X | Y` in type aliases —
  bit us in A5.11; annotations are fine via `from __future__ import annotations`). Torch tier:
  `make test-torch` (the pure env HAS torch 2.8, so torch-marked tests run there too).
- **LOCKED:** gate = the contribution; student = latent-pred transformer (+ waypoint + trust heads);
  DINOv3 = frozen cached encoder, NOT the student; no EMA/VICReg; no blobs; OracleTarget seam shape is
  user-approved — do not relitigate.

## 6. Open items (carry-over)

- **WorldVLN weights license** (`license:other`): clarification email SENT, awaiting reply. Blocks
  publication, not plumbing.
- **`[torch]` extra unpinned** → pin (`transformers>=4.56,<5`, `torch>=2.8,<2.13` cu12x) BEFORE/with A5.14.
- **Rotate the HF token** pasted in chat 2026-06-09 (never committed; verified absent from the repo).
- **V-JEPA-2 gating unknown** — research model id + access before coding A5.12's loader (see §3).

---

### Paste this into the new session

```
Continue the vllatent-ego-drone Phase-A.5 ralph loop. Read plans/handoff-2026-06-11-resume-ralph-A5.12.md, then DEV_LOG.md + .claude/ralph-rules.md + plans/phase-a5-replan-postpivot.md, and resume at the lowest pending step A5.12 (V-JEPA-2 surprise verifier wrapper). A5.12 is TORCH-tier: research the exact V-JEPA-2 HF model id AND whether it is gated (DINOv3 lesson — find the non-gated fallback first), then write vllatent/verify/vjepa2.py (frozen ViT-L, surprise s_j = 1 - cos(zhat_j, z_j) on GT future frames, LAZY imports, feeds OracleTarget.vjepa_surprise >= 0) + a monkeypatched contract test following tests/test_encode_contract.py / test_teacher_contract.py patterns. The contract half is autonomous; the real-weight smoke is USER-GATED (emit the command block, HF_ENDPOINT=https://hf-mirror.com, do not drive the download). Keep the pure gate green: make PY=/home/zh/miniconda3/envs/vln-ego-drone-1.1/bin/python import-smoke lint typecheck test (212 tests; pure env is Py3.9 — no runtime X|Y in type aliases). STOP CHECK at started_step+3 or any user-gated step. Run iterations INLINE (no ralph-loop.local.md).
```
