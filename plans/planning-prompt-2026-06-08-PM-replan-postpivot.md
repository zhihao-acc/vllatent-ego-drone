# Planning-agent brief — re-plan Phase A (post-WorldVLN-pivot + code review)

> **Created 2026-06-08 (PM).** Hand this to the planning agent at the start of the next session.
> **Supersedes** `plans/planning-prompt-2026-06-08-refactor-before-phaseB.md` *for the planning task*
> (that pre-pivot file is still required reading — it holds the full code-review detail this brief distils).
> **Output of your work** = a single re-plan doc (`plans/phase-a5-replan-postpivot.md`) + sign-off, **NOT code.**

---

## 0. Your role and the one objective

You are the **planning agent**. Produce **one coherent Phase-A re-plan** that simultaneously:
1. **Absorbs the backbone pivot** (we now reuse **WorldVLN as a frozen teacher and distil it**, instead of
   training a predictor from scratch), and
2. **Lands the genuine code-review refactor items** (the ablation control surface is forming wrong),
3. sequenced **pure-tier-first, ralph-executable**, with **DoDs + exact test commands + dependency order +
   tier/user-gating** per step.

You **do not write production code.** You read, reconcile, and write the plan, then **present it for the
user's sign-off** before any ralph execution. Honor the project's loop: **plan → sign-off → ralph-execute.**

A note on stance, because it matters here: a previous review over-reached by grading **unbuilt** steps as if
they were gaps (see §2B). Do **not** repeat that. Plan the work that is actually next; mark deferrals
honestly with a reason; do not relitigate decisions that are locked (§4).

---

## 1. Locate current dev status FIRST (read in this order, cheap → expensive)

Verify everything against the live tree — the review's line numbers and the memories are point-in-time and
may be stale. Use `codegraph_status` / `codegraph_files` / `codegraph_search` to confirm current code before
trusting any cited `file:line`.

**Repo (`/home/zh/CODE/vllatent-ego-drone`):**
1. `DEV_LOG.md` — find the exact current position. (Expect: steps 1–6 + 5b **done & green @ a commit**;
   loop **paused before step 7**; ralph stopped via `rm .claude/ralph-loop.local.md`.)
2. `plans/phase-a-data-and-io-contract.md` — the original 13-step plan + per-step DoDs/test commands.
3. `plans/planning-prompt-2026-06-08-refactor-before-phaseB.md` — the **full code review** (all of
   H1–H3 / M1–M5 / L1–L3 + the reuse audit). §2 below distils it **plus the post-review correction** that
   file predates.
4. `plans/handoff-2026-06-08-refactor-session.md` — cold-start operator brief (if present).
5. `CLAUDE.md` — repo invariants + the PURE/TORCH/SIM tier split + the load-bearing foot-guns.
6. Current code via codegraph: `vllatent/{schemas,actions,frames,config,manifest,audit}.py`,
   `configs/*.yaml`, `tests/`. Confirm what each module actually contains today.

**Vault (`/home/zh/Documents/Obsidian Vault/projects/vln-ego-drone/latent-pred-pipeline/`) — read the
PIVOT banners at the top of each; they are authoritative and post-date the body:**
7. `arch-design-2026-06-08-latent-pred.md` — **PIVOT banner** = the new network (teacher/student split,
   I/O, the load-bearing OPEN items, the Phase-A rework note). The `⚠ SUPERSEDED` marks show what changed.
8. `dev-decision-2026-06-07-latent-pred-pipeline.md` — **PIVOT banner** + §1/§8 (WorldVLN is now
   teacher/substrate **and** baseline) + §11 verified-IDs line.
9. `environment-and-equipment.md` — H20 train / **Jetson Orin NX 16 GB deploy (binding)** / `fly0-m1`
   docker + AerialVLN scenes / conda env / **HF via `hf-mirror.com`, github direct** / **SSH hands-off**.
10. `training-playbook.md` — Phase-B SOPs (save-every-config, scene-split val, log frame transforms,
    overfit-tiny-batch). Relevant to H2/M5 framing but mostly **Phase B**, not Phase A.

**Auto-memory** (`~/.claude/projects/-home-zh/memory/`): `project-latent-pred-arch-locked` (now
pivot-headed), `project-vllatent-phaseA-pure-done`, `project-vllatent-equipment`,
`feedback-ralph-loop-promise-flush-race`, `feedback-dont-modify-third-party`, `first-training-playbook`.

---

## 2. The two things that changed since the plan was written — you must reconcile BOTH

### 2A. The pivot (authoritative — match it)

We **reuse WorldVLN as a FROZEN TEACHER and distil it** into a small single-pass on-board student that
carries the gate. **The contribution is unchanged — the calibrated single-pass trust-horizon gate is the
paper.** Concretely:

- **Teacher (offline / server, frozen):** WorldVLN (`2605.15964`, InfinityStar-8B, CC BY 4.0, HF
  `EmbodiedCity/WorldVLN`) → instruction + observations → latent rollout → **4-DoF `(Δx,Δy,Δz,Δψ)`
  waypoints**. Provides **policy targets** + the **trust-horizon oracle**.
- **Trust-horizon oracle (replaces the old K=5 self-trained ensemble):** WorldVLN **rollout disagreement**
  + an **independent V-JEPA-2 surprise** verifier (`2601.10553`).
- **Student (deployed Jetson Orin NX 16 GB, single pass):** frozen **DINOv3** encoder (unchanged, now the
  *student's* encoder) + a **small predictor trained by DISTILLATION from WorldVLN** + waypoint head + the
  **calibrated single-pass trust-horizon head**.
- **Output seam unchanged:** the 4-DoF → fly0 NED→FLU→ENU remap is identical (still Phase-D for the live
  seam).

**⚠ Pivot OPEN items — schedule these as early Phase-A spikes (they gate the design):**
- **WorldVLN's released inference is DETERMINISTIC** (no exposed temperature/seed/rollouts; the paper's
  G=4 was RL-training-only). The "K-rollout disagreement" is **not free** — a spike must determine the
  source: (a) re-enable stochastic AR sampling in *our* wrapper (check `WorldVLN.code`), (b) MC-dropout /
  latent-perturbation over the frozen teacher, (c) the independent V-JEPA-2 surprise, and/or (d)
  **AirScape** (`2507.08885`, MIT, native multi-seed) as an auxiliary disagreement teacher.
- **Verify weights + size** (HF via `hf-mirror.com`; ckpts `INFINITY_CKPT`, `STAGE2_LATENT2ACTION_CKPT`;
  ≈8 B).
- **Reconcile action I/O:** `WorldVLN.code` lists **6-DoF** deltas `[dx,dy,dz,droll,dyaw,dpitch]`; the
  paper says **4-DoF**; our seam is 4-DoF (roll/pitch≡0). Pin before any distillation contract.
- **Licenses:** WorldVLN **CC BY 4.0**; AerialVLN data **CC BY-NC-SA 4.0** (non-commercial).

### 2B. The code review — distilled, WITH the post-review correction

The reviewer's intent: **the repo's whole deliverable is "flip an ablation via config, not code surgery,"
and that ablation control surface is forming wrong in Phase A.** But a follow-up exchange corrected the
report — the WARN softens toward APPROVE, because much of it graded **unbuilt** steps. Use this corrected
map (do **not** treat the "planned/over-reach" rows as Phase-A holes):

| ID | Finding (one line) | Plan status | Disposition for the re-plan |
|----|---|---|---|
| **M3** | audit computes `all_action_classes_present` / `scene_id_range` / `splits_present` **per-episode**; DoD wants them **per-slice** | **COLLIDES** with step 5b's "no new code" | **Must absorb.** Add an `AuditSummary` slice aggregator; amend 5b to drop "no new code". Real data won't have all 8 classes per episode. |
| **M4** | `StepSample` has no `history_mask` / language padding mask | **COLLIDES** — step 10 would reopen the "done" frozen step-3 contract | **Must absorb.** Add masks to the contract **now**, before the loader (step 10) is written. (Student still has history + language under the pivot → still needed.) |
| **H1** | no single source of truth for swept knobs; `T/H/depth/K` hardcoded + duplicated across `schemas`/`loader`/`yaml`/`manifest`; `load_config` orphan | unscheduled; legitimately an **A→B-boundary** task | **Pull forward — cheapest now (pure-tier, ~0 callers).** And see the pivot note below: the *knob set itself changed*. |
| **H2** | config is untyped `dict[str,Any]`, no schema/validation/immutability; violates the repo's own boundary-validation + immutability rules | unscheduled (resume/snapshot is correctly **Phase B**) | Frozen `Config` dataclass tree + `from_yaml` + validation (with H1). Add the `load_config` test. **Do not** build resume/snapshot here. |
| **M1** | quaternion math (`_yaw_from_xyzw`, `_xyzw_from_yaw`, `_wrap_pi`) lives in `actions.py`; `audit.py` imports its **private** helpers; `frames.py` owns only string constants | unscheduled | **Cheap win — do now.** Extract a `frames.py`-owned **public** quaternion module; `actions.py` + `audit.py` import it. Pre-positions `frames.py` for the Phase-D remap. |
| **M2** | no-flip `test_frames.py` vs fly0 semantics is asserted as an A–C invariant but **has no step** | unscheduled guardrail | **Give it a step** (hard CI gate). It mitigates the #1 foot-gun (frame re-derivation); risk is it silently slips to Phase D. |
| **M5** | provenance manifest stringly-typed + triple-duplicated constants | partly planned (step 9) | de-dup rides on **H1**; build a typed manifest builder fed from the config. **Extend provenance for the teacher** (WorldVLN model id+rev, disagreement-source, V-JEPA-2 id, render cfg hash). |
| **L1** | `DOF + 3` pun for pose-row width (`audit.py`) | unscheduled | **Cheap win** — `POSE_ROW = 7` (3 pos + 4 quat). |
| **H3** | output seams (`PredictorOutput`, `TrustReadout`, `Waypoint`) are prose-only in `io-contract.md` | reviewer **conceded**: pull-forward advice, **not a Phase-A gap** | **Decide deliberately** (see pivot note). Under the pivot these seams = the **distillation contract**, so pulling them forward is more attractive than at review time. |
| remap / encoder / predictor / trust / resume | — | **planned/Phase B/D** | **Do NOT plan as Phase-A holes.** The reviewer over-reached here. |

**Cheap-wins bundle (no architecture risk, land first):** M1 private-import → public extraction, the
`load_config` test, L1 `POSE_ROW`.

---

## 3. The reconciliation you must perform (where the pivot reshapes the review)

This is the heart of the task — don't apply the review mechanically; refract it through the pivot:

1. **H1 config-SoT — the knob set CHANGED.** The pre-pivot swept knobs were `T / H / predictor-depth /
   heads / ensemble-K`. Under the pivot they become: **disagreement source + K (number of WorldVLN
   rollouts / perturbation samples)**, **distillation weights/temperature**, **student depth**, `T / H`,
   and the V-JEPA-2 gate threshold(s). Define the **typed `Config` against the *pivoted* ablation surface**
   — this makes H1 *more* central, and it's still pure-tier and cheapest now. Keep the LOCKED-but-fixed
   shapes (DINOv3 196/768) as constants; only the **swept** knobs move into config.
2. **H3 output seams — add the teacher/oracle contract.** Beyond `PredictorOutput` / `TrustReadout` /
   `Waypoint`, the pivot introduces a **`TeacherOutput` / `OracleTarget`** seam = (WorldVLN waypoint +
   rollout-disagreement statistic + V-JEPA-2 surprise) — i.e. the **distillation target tuple**. If you
   pull H3 forward, define this one too; it is the contract Phase B's distillation trains against.
3. **Steps 7–13 — RE-SPEC around the teacher.** The DINOv3-centric pending steps change meaning:
   - old step 7 (DINOv3 encoder wrapper) → **stand up the frozen WorldVLN teacher** (download via
     hf-mirror; run its inference; resolve the determinism spike in §2A) **+** keep a DINOv3 *student*
     encoder wrapper.
   - old steps 8–9 (render → DINOv3-encode → cache) → **render AerialVLN → run frozen WorldVLN → cache the
     teacher/oracle dataset** (waypoints + rollout/latent + disagreement stat + V-JEPA-2 surprise) with the
     extended provenance manifest (M5).
   - old steps 10–11 (cached-latent loader) → **loader that feeds the distillation** (student inputs +
     teacher/oracle targets), honoring the M4 masks.
   - old steps 12–13 (sizing + DoD) → re-size the teacher-inference + cache job; re-state the Phase-A DoD
     for the distillation dataset.
   Mark the old DINOv3-latent steps **SUPERSEDED** in the re-plan (keep them as the record).
4. **What's untouched (say so explicitly, so no one re-does it):** the pure/data lane — `actions.py`
   (discrete→4-DoF), `audit.py` + the real-slice audit (Euler poses, quaternion order, `len(ref)==len`
   alignment), the frame conventions, the data slice, scaffold/CI — **survives**, and the **4-DoF →
   fly0-ENU output seam is unchanged**. No completed work is wasted.

---

## 4. Constraints & invariants — honor (do not relitigate)

- **LOCKED, do not reopen:** the gate is the contribution; the pivot (WorldVLN frozen teacher → distilled
  student); DINOv3 as the student encoder; **no EMA / no VICReg** (frozen+cached target); reuse-don't-fork.
- **Tier discipline (load-bearing):** PURE (`schemas/actions/frames/config/manifest/audit`, numpy/pyyaml
  only — CI hard-gates) / TORCH (lazy) / SIM (lazy). Every refactor item in §2B is **pure-tier** — keep it
  there. The pure tier must never gain a torch/airsim/transformers import.
- **Foot-guns:** frame no-flip (re-derive vs fly0 `frames.py` semantics, never hand-roll — that's M2);
  BGR→RGB at the render→encode boundary; quaternion order (`start_rotation` w-first; `reference_path` Euler
  6-wide; `airsim.Quaternionr` xyzw); **no blobs** committed.
- **Ralph discipline:** lowest pending step first; **pure-tier / fixtures-first**; STOP CHECK at
  `started_step + 3`; **user-gated steps** (render / cache / H20 / docker / WorldVLN weights / network)
  stay `in_progress` until the user pastes verification — **never auto-mark done**. Deterministic stop =
  `rm .claude/ralph-loop.local.md`; always set a `--max-iterations` backstop.
- **SSH hands-off / manual ops:** anything on the H20, any UE4/AirSim launch, any large download — give the
  user a **command block to paste**; never drive it. github direct; HF via `hf-mirror.com`.
- **Research-before-implement:** for any new external dependency (WorldVLN inference, V-JEPA-2 surprise,
  AirScape), confirm the API against the upstream repo/docs before writing the wrapper — don't guess.
- **Don't modify `third_party/` or the sibling repos.** Workarounds live in this repo.

---

## 5. Deliverable — what to WRITE (and where)

Write **`plans/phase-a5-replan-postpivot.md`**, structured as:

1. **Current position** (verified from DEV_LOG + codegraph, not assumed): which steps are done, the commit,
   what's paused.
2. **What changed** (1 short paragraph each): the pivot; the review's corrected must-fix set.
3. **Ordered step list** with, per step: **id + title**, **tier** (pure/torch/sim), **autonomous vs
   user-gated**, **DoD**, **exact test command(s)**, **dependencies (blocks/blocked-by)**, and a one-line
   **why-now**. Put the **pure-tier refactor items first** (cheap-wins bundle → H1/H2 config-SoT →
   H3+teacher seam → M1/M2 frames+no-flip-test → M4 masks → M3 aggregator), **then** the re-spec'd
   teacher/distillation steps (7–13 successors), **then** the pivot spikes where they're needed (the
   determinism/weights/DoF spike should land **before** the teacher-cache contract is frozen).
4. **A findings-disposition table** = the §2B table with your final per-item decision (absorbed-into-step-N
   / scheduled-at-A→B / cheap-win-now / deferred-with-reason). Nothing silently dropped.
5. **Supersession block:** mark the old steps 7–13 superseded, point to the pivot banners; note what
   survives untouched (§3.4).
6. **Phase-A DoD (restated for the pivot):** the data/contract lane is green **and** a small
   teacher/oracle distillation slice is produced + loadable — the hand-off into Phase B (distillation
   training).
7. **Open risks / kill criteria** carried forward (e.g., determinism spike fails → fall back to
   MC-dropout/AirScape/V-JEPA-2-only disagreement; weights unusable → re-evaluate Path A from the
   dev-decision history).

---

## 6. Your planning-task Definition of Done

- The re-plan doc exists, is internally consistent, and **every** H/M/L item + the pivot has an explicit
  disposition.
- **You have NOT executed any step or written production code.**
- You **present the plan to the user for sign-off**, calling out the 2–3 decisions that genuinely need a
  human (e.g., "pull H3 forward as the distillation contract, yes/no?"; "disagreement-source spike
  ordering"; "confirm the pivoted ablation-knob list for the typed Config").
- On sign-off → hand to the ralph loop (config-SoT / cheap-wins first), user-gated steps flagged.

**Do NOT:** grade unbuilt steps as gaps · relitigate the pivot or the gate-is-contribution lock · modify
`third_party`/siblings · break the pure tier · auto-mark user-gated steps done · write code this session.
