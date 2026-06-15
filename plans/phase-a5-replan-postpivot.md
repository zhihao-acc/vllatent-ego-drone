# Phase-A re-plan (post-WorldVLN-pivot + code review)

> **Created 2026-06-08 (PM) by the planning agent.** Supersedes steps 7–13 of
> `plans/phase-a-data-and-io-contract.md` and consumes the brief in
> `plans/planning-prompt-2026-06-08-PM-replan-postpivot.md`. **Sign-off: user-approved
> 2026-06-08.** Output of the planning task = this doc; no production code written, no step
> executed. Hand-off → ralph loop (pure-tier / cheap-wins first; user-gated steps flagged).

## Context — why this exists

`vllatent-ego-drone` is the latent-prediction aerial-VLN repo. Phase A (the pure/data and
I/O-contract lane) was built to steps **1–6 + 5b**, all green @ commit `a0a3bb6`, and the
ralph loop was deliberately **stopped** (`.claude/ralph-loop.local.md` removed) *before*
step 7. Two things then landed that make the original 13-step plan no longer correct as
written, and this re-plan reconciles **both**:

1. **The backbone pivot (user-approved 2026-06-08 PM).** We no longer train the latent
   predictor from scratch. We **reuse WorldVLN (`2605.15964`, base InfinityStar-8B
   `2511.04675`, HF `EmbodiedCity/WorldVLN`, CC BY 4.0) as a FROZEN TEACHER and distil it**
   into a small single-pass on-board **student** = the latent-prediction transformer +
   waypoint head + trust-horizon head. **DINOv3 is the student's frozen, cached front-end
   encoder — not the student.** The contribution is unchanged: the **calibrated single-pass
   trust-horizon gate is the paper**. The trust oracle (old K=5 self-trained ensemble) is
   replaced by **WorldVLN rollout-disagreement + an independent V-JEPA-2 surprise verifier
   (`2506.09985`)**.
2. **A code review of the pure lane** flagged structural issues to fix *before* Phase B,
   chiefly that the ablation control surface ("flip an ablation via config, not code
   surgery") is forming wrong. A follow-up softened the WARN toward APPROVE because much of
   it graded *unbuilt* steps — so this re-plan absorbs only the genuine, built-surface
   findings and does **not** treat planned/Phase-B/D work as Phase-A holes.

---

## §1 — Current position (verified against live tree, not assumed)

- **Done & green @ `a0a3bb6`:** steps 1–6 + 5b. Pure lane passes `import-smoke / lint /
  typecheck / pytest (102 tests) / make audit / check_no_blobs`.
- **Real-data correction already applied (`95bcb8e`):** `reference_path` is **6-wide EULER
  `[x,y,z,pitch,roll,yaw]` (radians)**, `len(reference_path) == len(actions)`. Real-slice
  audit: 50/50 episodes ok, ~10,198 transitions, **0 Δ-mismatches**, all 8 action classes,
  quaternion reorder-consistent.
- **Loop:** stopped before step 7 (`.claude/ralph-loop.local.md` absent).
- **Working tree:** deleted `plans/handoff-2026-06-08-refactor-session.md` (was never
  committed; this re-plan replaces the need for it); untracked
  `plans/planning-prompt-2026-06-08-PM-replan-postpivot.md` (the in-repo source of this
  task's brief).
- **Findings vs. live code:** **H1, H2, H3, M1, M2, M3, M4, M5 all CONFIRMED present.**
  **L1 is already RESOLVED** — the step-5b Euler fix replaced the `DOF+3` pun with an
  explicit `REFERENCE_PATH_ROW_WIDTH = 6`; the row is **6-wide Euler, not 7-wide
  quaternion**, so the brief's "`POSE_ROW = 7 (3 pos + 4 quat)`" suggestion is itself stale
  and is dropped. L2→subsumed by H1; L3→subsumed by H2.

---

## §2 — What changed (one paragraph each)

**The pivot.** Predictor origin flips from *trained-from-scratch* to *distilled from a
frozen WorldVLN teacher*. Student = latent-prediction transformer + waypoint + trust head
(DINOv3 frozen encoder in front, cached). Oracle = WorldVLN rollout-disagreement +
V-JEPA-2 surprise. Output seam unchanged: student emits 4-DoF `(Δx,Δy,Δz,Δψ)` in AirSim-NED
body; NED→FLU→ENU remap math is A–C, live fly0 wiring is Phase D. Load-bearing OPEN unknowns
to retire early: WorldVLN's released inference is **deterministic** (so rollout-disagreement
is not free), weight/size confirmation, **4-DoF (paper) vs 6-DoF (`WorldVLN.code`)** action
I/O, and licenses (WorldVLN CC BY 4.0; AerialVLN CC BY-NC-SA 4.0 non-commercial).

**The review (corrected must-fix set).** Absorb the findings on the *built* surface: H1/H2
(single typed Config source-of-truth — highest leverage), H3 (type the output seams), M1
(quaternion primitives misplaced + private cross-module imports), M2 (no-flip guardrail has
no test), M3 (audit per-slice checks mis-scoped per-episode), M4 (StepSample lacks
history/language masks), M5 (manifest stringly-typed + duplicated constants). Do **not**
plan remap/encoder/predictor/trust/resume as Phase-A holes — they are correctly Phase B/D.

---

## §3 — Ordered step list

Tier ∈ {PURE, TORCH, SIM}. Gate ∈ {AUTO, USER-GATED}. `PY=/home/zh/miniconda3/envs/vln-ego-drone-1.1/bin/python`.
Ordering: pure-tier cheap-wins → config source-of-truth → typed seams → data-contract
completions → pivot investigation (gates the teacher contract) → teacher distillation
contract → re-spec'd 7–13 successors. Ralph rule preserved: lowest pending step first,
fixtures-first, STOP CHECK at `started_step + 3`, user-gated steps stay `in_progress` until
the user pastes verification.

### Group 1 — Pure-tier cheap-wins (≈zero architecture risk, land first)

**A5.1 — Extract public frame/quaternion primitives into `frames.py` (M1, L1, micro-nit).**
- Tier PURE · AUTO. **Why now:** cheapest it will ever be (~0 downstream callers); closes the
  leaky private import that step-5b *worsened*; pre-positions `frames.py` as owner of the #1
  frame invariant.
- **DoD:** `frames.py` exports **public** `yaw_from_xyzw`, `xyzw_from_yaw`, `wrap_pi`,
  `reorder_wxyz_to_xyzw`; `actions.py` and `audit.py` import the public API (no `_private`
  cross-module imports remain). Verify no `DOF+3` pun remains (L1 already resolved → keep
  `REFERENCE_PATH_ROW_WIDTH = 6`). Add the `# == yaw + π/2` comment on `apply_delta`'s
  lateral branch.
- **Test:** `$PY -m pytest -q tests/test_actions.py tests/test_audit.py tests/test_smoke.py`
  + `make import-smoke && make lint && make typecheck`; `! grep -rn "from vllatent.actions import .*_" vllatent/audit.py`.
- **Deps:** blocks A5.2. Blocked-by: none.

### Group 2 — No-flip guardrail (close M1's companion risk)

**A5.2 — `test_frames.py` no-flip + NED→FLU→ENU remap math (M2), hard CI gate.**
- Tier PURE · AUTO. **Why now:** mitigates the #1 foot-gun (frame re-derivation) at its
  cheapest; CLAUDE.md sanctions *remap math + unit tests* as A–C work (only the live fly0
  `WaypointHandoff` wiring is Phase D). Risk if deferred: silently slips to Phase D.
- **DoD:** `frames.py` holds the pure-numpy NED→FLU→ENU remap; `tests/test_frames.py`
  asserts the no-flip basis vs fly0 semantics (up→up, down→down, right→right-of-forward,
  forward→forward) and is wired into the CI `make test` gate. Live fly0 wiring explicitly
  marked Phase D (not implemented here).
- **Test:** `$PY -m pytest -q tests/test_frames.py` + `make test`.
- **Deps:** blocked-by A5.1. Blocks: none (live seam is Phase D).

### Group 3 — Config source-of-truth (highest leverage)

**A5.3 — Frozen `Config` dataclass tree + `from_yaml` + validation (H1, H2, L2, L3).**
- Tier PURE · AUTO. **Why now:** the ablation control surface; pure-tier with ~0 callers
  today; every later step (manifest, loader, teacher seam) reads from it.
- **Knob set (Q3 = full-set-now, with typed placeholders):** Config exposes the SWEPT knobs
  `T` (horizon), `H` (history), distillation `λ`-weights + temperature, **student-transformer
  depth**, and the spike-dependent placeholders `disagreement_source`, `K`, `vjepa_surprise_threshold`
  (finalized in A5.9). LOCKED-fixed shapes (DINOv3 196/768; predictor D=768/depth-12/heads-12)
  stay plain module constants — only swept knobs move into Config.
- **DoD:** frozen `Config` tree with `from_yaml` + boundary validation (mirrors how
  `StepSample` guards its boundary); `schemas.py`/`actions.py`/`manifest.py` read swept knobs
  from Config (no duplicated `HORIZON`/`HISTORY`/step-constants across files); the orphan
  untyped `load_config` is removed/replaced; `Config.from_yaml` has tests (absorbs the L3
  "load_config test"). Immutability + boundary-validation rules satisfied. **No resume/
  snapshot here (correctly Phase B).**
- **Test:** `$PY -m pytest -q tests/test_config.py` + `make import-smoke && make typecheck && make test`.
- **Deps:** blocks A5.4, A5.9, A5.15. Blocked-by: none.

**A5.4 — Typed manifest builder fed from Config (M5).**
- Tier PURE · AUTO. **Why now:** de-dup rides on A5.3; kills the third copy of 196/768 in
  `empty_manifest()`.
- **DoD:** typed manifest builder consuming `Config` (no hardcoded encoder-id/196/768);
  `CacheManifestEntry` ↔ validation key-agreement is type-enforced, not hand-kept. Teacher-
  provenance fields (WorldVLN id+rev, disagreement-source, V-JEPA-2 id, render-cfg hash) are
  *stubbed in the type now*, populated in A5.14.
- **Test:** `$PY -m pytest -q tests/test_smoke.py tests/test_schemas.py` (manifest round-trip).
- **Deps:** blocked-by A5.3. Blocks: A5.14.

### Group 4 — Typed seams (H3, student now / teacher later — Q1)

**A5.5 — Student output seams in `schemas.py` (H3, Q1 = student seams now).**
- Tier PURE · AUTO. **Why now:** pure-tier, CI-gateable; makes −trust / swap-predictor a
  config flag over typed seams instead of code surgery in Phase B.
- **DoD:** frozen `PredictorOutput` (`ẑ_{t+1..T}` `(T,196,768)`), `TrustReadout`
  (`{p_j∈[0,1]^T, k*, σ}`), `Waypoint` (`(4,)` NED-body) dataclasses added with shape/dtype
  validation; `docs/io-contract.md` updated to reference the typed seams.
- **Test:** `$PY -m pytest -q tests/test_schemas.py`.
- **Deps:** blocked-by: none. Blocks: A5.15 (loader/targets reference these).

### Group 5 — Data-contract completions

**A5.6 — Add `history_mask` + language padding-mask to `StepSample` (M4).**
- Tier PURE · AUTO. **Why now:** must reopen the step-3 contract *before* the loader (A5.15)
  is written; the student still has history + language under the pivot, so masks are real.
- **DoD:** `StepSample` gains `history_mask` (block-causal at episode start) and a language
  padding mask; validation updated; the "padded+masked" comment becomes a real field.
- **Test:** `$PY -m pytest -q tests/test_schemas.py`.
- **Deps:** blocks A5.15. Blocked-by: none.

**A5.7 — `AuditSummary` slice aggregator (M3); amend step 5b.**
- Tier PURE (code) · USER-GATED (re-run on real slice). **Why now:** `all_action_classes_present`
  / `scene_id_range` / `splits_present` are dataset-level; computing them per-episode is
  meaningless on real data. Step 5b passed via an ad-hoc script — make it first-class.
- **DoD:** `AuditSummary` aggregates across the slice; step 5b's "no new code" is dropped;
  the real-slice audit is re-run through the aggregator and reproduces 50/50 ok / all 8
  classes / scene_id∈[1,25] / splits-present **at slice scope**. Stays `in_progress` until
  the user pastes the aggregated report.
- **Test (code):** `$PY -m pytest -q tests/test_audit.py` + `make audit`.
- **Test (real, user-gated):** `$PY -m vllatent.audit --slice data/aerialvln_json/train.slice.json --summary -`.
- **Deps:** blocked-by: none.

### Group 6 — Pivot scoping investigation (gates the teacher contract)

**A5.8 — Scoping investigation: WorldVLN determinism + weights/size + 4-DoF↔6-DoF + license.**
- Tier TORCH/research · USER-GATED (download via `hf-mirror.com`; read `WorldVLN.code`).
  **Why now:** MUST land before the teacher seam (A5.9) and the teacher-cache contract
  (A5.14) freeze — the cache file format depends on what the disagreement signal actually is.
- **Disagreement strategy (Q2 = try-in-order, commit on evidence):** (a) attempt to
  re-enable stochastic AR sampling in *our* WorldVLN wrapper so rollouts differ → if
  infeasible, (d) adopt **AirScape (`2507.08885`, MIT, native multi-seed)** as the
  disagreement teacher; **(c) V-JEPA-2 surprise stays the independent second gate either way.**
- **DoD (a written findings note, not production code):** resolves — (1) is WorldVLN
  inference stochastic-able? which option (a/d) wins on a tiny probe; (2) weights complete +
  size confirmed (`INFINITY_CKPT`, `STAGE2_LATENT2ACTION_CKPT`, ≈8B); (3) action I/O pinned
  (4-DoF with roll/pitch≡0 vs the code's 6-DoF); (4) licenses confirmed. Stays `in_progress`
  until the user pastes the probe output.
- **Test:** user pastes: tiny multi-rollout probe showing non-identical rollouts (or the
  AirScape fallback producing K varied rollouts) + `hf` weight listing.
- **Deps:** blocks A5.9, A5.11, A5.14. Blocked-by: none (can run in parallel with Groups 1–5).

### Group 7 — Teacher distillation contract (post-investigation)

**A5.9 — `TeacherOutput` / `OracleTarget` distillation-target seam; finalize Config placeholders.**
- Tier PURE · AUTO. **Why now:** this tuple is literally the contract Phase B distillation
  trains against; typed *after* A5.8 so it reflects facts, not guesses (Q1).
- **DoD:** frozen `TeacherOutput` / `OracleTarget` dataclass = (WorldVLN waypoint +
  disagreement statistic [per A5.8 source] + V-JEPA-2 surprise); the A5.3 placeholders
  `disagreement_source` / `K` / `vjepa_surprise_threshold` are finalized to the chosen values.
- **Test:** `$PY -m pytest -q tests/test_schemas.py tests/test_config.py`.
- **Deps:** blocked-by A5.8, A5.3. Blocks: A5.14, A5.15.

### Group 8 — Re-spec'd 7–13 successors (teacher/distillation pipeline)

**A5.10 (was 7) — DINOv3 student-encoder wrapper.** Tier TORCH (lazy) · contract AUTO /
real-weight USER-GATED. DoD: `vllatent/encode/dinov3.py`, frozen ViT-B/16, RGB 224²→(196,768)
fp16, RGB boundary enforced, lazy torch import. Test: `$PY -m pytest -q tests/test_encode_contract.py`
(monkeypatched); real-weight: `HF_ENDPOINT=https://hf-mirror.com make encode-smoke`.
Blocks A5.14.

**A5.11 (new) — Frozen WorldVLN teacher wrapper.** Tier TORCH (lazy) · USER-GATED (server).
DoD: wrapper runs WorldVLN inference → 4-DoF waypoints + the disagreement signal sourced in
A5.8; lazy import; never modifies the upstream clone. Test (user-gated): `HF_ENDPOINT=https://hf-mirror.com $PY -m vllatent.teacher.worldvln --episode fixtures/episodes/tiny_episode.json --rollouts K`.
Blocked-by A5.8. Blocks A5.14.

**A5.12 (new) — V-JEPA-2 surprise verifier wrapper.** Tier TORCH (lazy) · USER-GATED. DoD:
frozen ViT-L surprise `s_j = 1 − cos(ẑ_j, z_j)` on GT future frames; lazy import. Test
(user-gated): `HF_ENDPOINT=https://hf-mirror.com $PY -m vllatent.verify.vjepa2 --frames fixtures/...`.
Blocks A5.14.

**A5.13 (was 8) — Render harness.** Tier SIM (lazy airsim) · unit AUTO / live USER-GATED.
DoD: AirSim teleport+capture, BGR→RGB, quaternion reorder, **every client call lock-wrapped**,
224² RGB uint8. Test: `$PY -m pytest -q tests/test_render_unit.py` (mock); live:
`$PY -m vllatent.render --episode fixtures/episodes/tiny_episode.json --scene 1 --out /tmp/render_smoke/`
(in `fly0-m1` docker; user launches UE4 scene, waits port 41451). Blocks A5.14.

**A5.13b (added 2026-06-14) — Frozen CLIP text tower → `lang_tokens`.** Tier TORCH (lazy) · contract
AUTO / real-weight USER-GATED. **Why:** the cache contract (A5.15 loader) needs `lang_tokens (M,768)
fp16` from a frozen text tower, but no A5.x step built one — A5.14 cannot produce the cache without it.
DoD: `vllatent/encode/text.py`, frozen CLIP ViT-B/32 text tower (NON-GATED `openai/clip-vit-base-patch32`),
instruction → `(M,768)` fp16 per-token tokens (M real tokens; native 512 → 768 zero-pad lift, the real
512→768 map being the student's learned cross-attn in Phase B); lazy import; id single-sourced in
`Config.encoder.text_model_id` + recorded in `build_manifest` encoder provenance. Test:
`$PY -m pytest -q tests/test_text_contract.py` (monkeypatched); real-weight:
`HF_ENDPOINT=https://hf-mirror.com make text-smoke`. Blocks A5.14.

**A5.14 (was 9) — Render → [DINOv3 + WorldVLN + V-JEPA-2] → cache teacher/oracle dataset +
extended provenance manifest (M5 completes here).** Tier SIM+TORCH · manifest AUTO /
small-slice USER-GATED. DoD: orchestrate episode → render → encode → run teacher + verifier →
write fp16 latents + `OracleTarget` + manifest (teacher provenance populated). Test:
`$PY -m pytest -q tests/test_cache_manifest.py` (mocked); small-slice (user-gated):
`HF_ENDPOINT=https://hf-mirror.com $PY -m vllatent.cache build --slice data/aerialvln_json/train.slice.json --limit 5 --scenes-root /opt/aerialvln --out data/latent_cache/`.
Blocked-by A5.4, A5.9, A5.10, A5.11, A5.12, A5.13. Blocks A5.16/17/18.

**A5.15 (was 10) — Distillation loader.** Tier TORCH (lazy) · AUTO (code-only). DoD:
`vllatent/data/loader.py` torch `Dataset` emitting `(StepSample` student-inputs, `OracleTarget`
teacher-targets`)`, honoring the A5.6 masks, with `H`/`T` read from Config (A5.3). Test:
`$PY -m pytest -q tests/test_data_shapes.py` (shapes/dtypes over tiny_dump fixture).
Blocked-by A5.5, A5.6, A5.9, A5.3. Blocks A5.16/18.

**A5.16 (was 11) — Loader over REAL small teacher/oracle dump.** Tier TORCH · USER-GATED (no
new code). DoD: confirms well-formed `(student-input, teacher-target)` tuples end-to-end from
the A5.14 cache. Test: `$PY -m vllatent.data inspect --cache data/latent_cache/ --n 4`.
Blocked-by A5.14, A5.15.

**A5.17 (was 12) — SIZE the full render→teacher→cache job.** Tier doc/SIM · sizing AUTO /
bulk USER-GATED. DoD: `docs/full-run-sizing.md` (now includes WorldVLN-8B inference + V-JEPA-2
+ DINOv3 cost, server-side); `scripts/run_full_cache.sh` refuses without `--i-have-signed-off`.
Test: `test -f docs/full-run-sizing.md && grep -q "GB" docs/full-run-sizing.md`;
`bash scripts/run_full_cache.sh` exits non-zero without the flag. Blocked-by A5.14.

**A5.18 (was 13) — Phase-A DoD verification.** Tier evidence · USER-GATED. DoD: assemble (1)
`docs/io-contract.md` + typed student+teacher seams, (2) clean `AuditSummary` on the real
slice, (3) valid `(student-input, teacher-target)` distillation tuples from a real cached
dump. Stays `in_progress` until the user pastes all three. Blocked-by A5.14, A5.15, A5.7.

---

## §4 — Findings-disposition table (nothing silently dropped)

| ID | Finding | Disposition |
|----|---------|-------------|
| **H1** | No single source of truth for swept knobs | **Absorbed → A5.3** (typed `Config`; pivoted knob set). |
| **H2** | Untyped/unvalidated/mutable config; `load_config` orphan | **Absorbed → A5.3** (frozen tree + `from_yaml` + validation; no resume/snapshot — Phase B). |
| **H3** | Output seams prose-only | **Split (Q1): student seams → A5.5 now; teacher `OracleTarget` seam → A5.9 after A5.8.** |
| **M1** | Quaternion primitives misplaced + private cross-module imports | **Cheap-win → A5.1** (public `frames.py` module). |
| **M2** | No-flip guardrail has no test | **Scheduled → A5.2** (remap math + no-flip test, hard CI gate; live wiring stays Phase D). |
| **M3** | Audit per-slice checks mis-scoped per-episode | **Absorbed → A5.7** (`AuditSummary`; amend 5b, re-run on real slice). |
| **M4** | `StepSample` lacks history/language masks | **Absorbed → A5.6** (added before the loader). |
| **M5** | Manifest stringly-typed + duplicated constants | **Absorbed → A5.4** (typed builder from Config) **+ A5.14** (teacher provenance). |
| **L1** | `DOF+3` pose-row pun | **RESOLVED already** (step-5b Euler fix → `REFERENCE_PATH_ROW_WIDTH=6`, 6-wide Euler). Verify-only in A5.1. Brief's "POSE_ROW=7" suggestion **dropped as stale**. |
| **L2** | `HORIZON` duplicated schemas↔loader | **Subsumed by H1 → A5.3.** |
| **L3** | `load_config` orphan + untested | **Subsumed by H2 → A5.3** (test the replacement, not the orphan). |
| micro-nit | `apply_delta` lateral `# == yaw + π/2` | **Folded into A5.1.** |
| remap / encoder / predictor / trust / resume | — | **NOT Phase-A holes** (Phase B/D). Not scheduled here. |

---

## §5 — Supersession + what survives untouched

**Superseded** (kept as record; point to the vault PIVOT banners in
`arch-design-2026-06-08-latent-pred` and `dev-decision-2026-06-07-latent-pred-pipeline`):
old `phase-a-data-and-io-contract.md` **steps 7–13** → replaced by **A5.10–A5.18** (the
DINOv3-only latent-cache pipeline becomes a teacher/oracle distillation-cache pipeline).

**Survives untouched — do not redo:** `actions.py` (discrete→4-DoF), the `audit.py` core +
real-slice audit (Euler poses, quaternion order, `len(ref)==len(actions)` alignment), the
frame conventions, the data slice (steps 1–6 + 5b), scaffold/CI/tier-split, and the **4-DoF →
fly0-ENU output seam (Phase D, unchanged)**. No completed work is wasted.

---

## §6 — Phase-A DoD (restated for the pivot)

Phase A is done when **both** hold: (1) the data/contract lane is green — typed `Config`
source-of-truth, typed student seams (`PredictorOutput`/`TrustReadout`/`Waypoint`) + teacher
seam (`OracleTarget`), `StepSample` masks, `AuditSummary` clean on the real slice, public
`frames.py` + no-flip CI gate; **and** (2) a **small teacher/oracle distillation slice is
produced and loadable** — `(student-input, teacher-target)` tuples from a real cached dump
with the extended provenance manifest. That is the hand-off into **Phase B (distillation
training)**.

---

## §7 — Open risks / kill criteria

- **Disagreement investigation (A5.8) fails** → fall back along Q2's chain: AirScape native
  multi-seed (d) → MC-dropout / latent-perturbation (b) → V-JEPA-2-surprise-only (c).
  **Kill criterion:** if no source yields a disagreement signal that separates good/bad
  rollouts on a tiny probe, escalate — re-evaluate the oracle design (and Path A from the
  dev-decision history) before building A5.14.
- **WorldVLN weights unusable/incomplete** → re-evaluate backbone (AirScape-5B as alternate
  teacher) or revert toward the pre-pivot from-scratch predictor.
- **4-DoF vs 6-DoF unresolved** → block the distillation contract (A5.9/A5.14) until pinned
  (roll/pitch≡0 mapping confirmed).
- **Licenses:** AerialVLN **CC BY-NC-SA 4.0 (non-commercial)** — flag for publication;
  WorldVLN **CC BY 4.0** attribution required.
- **M2 slippage:** no-flip test/remap silently sliding to Phase D — mitigated by the A5.2
  hard CI gate.
- **Tier purity:** any `torch`/`airsim`/`transformers` import leaking into the pure tier —
  CI hard-gate catches; keep every such import lazy.

---

## §8 — Decisions locked this session (human sign-off)

1. **H3 scope:** student seams now (A5.5); teacher `OracleTarget` seam after the A5.8
   investigation (A5.9).
2. **Disagreement source:** try-in-order, commit on evidence — re-enable WorldVLN sampling
   → else AirScape multi-seed; V-JEPA-2 surprise is the independent gate either way (A5.8).
3. **Config knobs:** full swept set now with typed placeholders for `disagreement_source` /
   `K` / `vjepa_surprise_threshold`, finalized in A5.9 (A5.3).

---

## Hand-off to the ralph loop

- Start at **A5.1** (cheap-wins), pure-tier/fixtures-first; STOP CHECK at `started_step + 3`.
- **User-gated steps** (A5.7 real-run, A5.8, A5.10 real-weight, A5.11, A5.12, A5.13 live,
  A5.14 small-slice, A5.16, A5.17 bulk, A5.18) stay `in_progress` until the user pastes
  verification — never auto-mark done.
- **A5.8 can run in parallel with Groups 1–5** (it gates only A5.9/A5.11/A5.14).
- Always set a `--max-iterations` backstop; deterministic stop = `rm .claude/ralph-loop.local.md`.
- Dependency graph (acyclic): A5.1→A5.2; A5.3→{A5.4,A5.9,A5.15}; A5.8→{A5.9,A5.11,A5.14};
  A5.{4,9,10,11,12,13}→A5.14→{A5.16,A5.17,A5.18}; {A5.5,A5.6}→A5.15; A5.7→A5.18.
