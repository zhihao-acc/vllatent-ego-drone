# New-session handoff — vllatent-ego-drone (2026-06-08, post Phase-A pure lane → refactor pivot)

> Paste this to start the next session. It is a cold-start operator brief. The Phase-A pure lane is
> done; a code review triggered a **"refactor before Phase B"** pivot. Your job: run the planning agent
> to adjust the plan, get user sign-off, then execute via the ralph loop.

---

ROLE: You are the operator for **vllatent-ego-drone**. The Phase-A pure/data lane is complete and
green; a code review found structural issues to fix **before Phase B**. (1) Run the planning agent to
produce the adjusted refactor plan, (2) show the user and get sign-off, (3) execute the refactor via the
ralph loop. Do NOT relitigate the LOCKED architecture. Do NOT drive SSH/docker/UE4/GPU yourself
(user-gated); you give command blocks. Local git (commit + push to the existing `origin`) is fine.

## Repo + source of truth (read in order before acting)
1. `CLAUDE.md` — LOCKED arch · tier split (pure/torch/sim) · Wiki-KB pointers · load-bearing invariants.
2. `.claude/ralph-rules.md` — per-iteration protocol · quality gates · user-gated rule · deterministic stop.
3. `DEV_LOG.md` — current position (newest entry on top); the step-status table.
4. `plans/phase-a-data-and-io-contract.md` — the original Phase-A steps 1–13.
5. `plans/phase-a5-refactor-before-phaseB.md` — **the adjusted refactor plan (produced by step 1 below).**
6. `docs/io-contract.md` — the four seams + the foot-guns.
- Vault (authoritative INTENT, do NOT relitigate): `…/latent-pred-pipeline/` —
  `arch-design-2026-06-08-latent-pred` (LOCKED) + `dev-decision-2026-06-07`.

## Current state (done · committed · pushed · `origin` synced @ `95bcb8e`)
- **Steps 1–6 + 5b done & green** (see DEV_LOG): scaffold/git/codegraph · io-contract.md · schemas.py ·
  actions.py · audit.py + fixtures · real slice fetched + audited clean (50/50, 0 Δ-mismatches over
  ~10,198 transitions, all 8 classes, quaternion consistent).
- Pure lane gates green: `make import-smoke / lint / typecheck / test` (102) `/ make audit /
  ALL=1 scripts/check_no_blobs.sh`.
- Real data in place, **gitignored** under `data/aerialvln_json/` (`{train,val_seen,val_unseen,test}.json`
  + `train.slice.json` 50 eps + `audit_report.json`). Never commit it.
- **Key real-data correction (commit `95bcb8e`)**: `reference_path` rows are **6-wide EULER
  `[x,y,z,pitch,roll,yaw]` (rad, yaw=row[5])**, NOT a 7-wide quaternion; **`len(reference_path)==len(actions)`**.
- The ralph loop is **STOPPED** (`.claude/ralph-loop.local.md` removed). Steps **7–13 are NOT started.**

## Immediate task (in order)
1. **Run the planning agent** with `plans/planning-prompt-2026-06-08-refactor-before-phaseB.md` (ready).
   It produces `plans/phase-a5-refactor-before-phaseB.md` — a ralph-executable refactor plan covering
   the must-fix list (config single-source-of-truth → output-seam dataclasses → frames/quaternion
   consolidation + test_frames no-flip-vs-fly0 → StepSample masks → AuditSummary), then re-sequences
   steps 7–13 to consume them.
2. **Show the user the adjusted plan and get sign-off** before executing (the review reorders
   priorities: config SoT first; items 1–3 are pure-tier, CI-gated, ~zero callers → cheapest now).
3. **Launch the ralph loop** on the A.5 steps using the same mechanics as before (see below).

## Toolchain / env
- Local python for ALL make/pytest/ruff/mypy (no py3.10 locally; training env is H20-only):
  `PY=/home/zh/miniconda3/envs/vln-ego-drone-1.1/bin/python`
  → `make PY=$PY import-smoke && make PY=$PY lint && make PY=$PY typecheck && make PY=$PY test`
- **Do NOT wrap python in `conda run`** (RoboStack activate.d hook bug) — call the binary path directly.
- codegraph MCP: pass `projectPath=/home/zh/CODE/vllatent-ego-drone` (the server is launched outside the
  project). Consult it before editing.

## Ralph loop mechanics + gotchas (from prior runs)
- Launch the `ralph-loop:ralph-loop` skill. **Strip shell-special chars from the prompt args** —
  parentheses `()`, backticks, `;` trip the setup script's subshell guard (the per-iteration protocol
  lives in `.claude/ralph-rules.md`, which the loop re-reads each iteration, so the prompt can be terse).
- ALWAYS keep a `--max-iterations` backstop and a `--completion-promise`.
- **Deterministic stop = `rm .claude/ralph-loop.local.md`** (the stop hook can re-feed even after a valid
  `<promise>`). Monitor: `grep '^iteration:' .claude/ralph-loop.local.md`. Cancel: `/cancel-ralph`.
- One step → one commit `refactor(phaseA5): step N — …` with **specific `git add`** (never `-A`/`.`).
- **User-gated steps** (render/cache/H20/docker/network/S3) stay `in_progress` until the user pastes
  verification — NEVER auto-mark done; produce the command block, the user runs it.
- **GateGuard**: the first Bash call each session must be preceded by stating (1) the user request and
  (2) what the command verifies.

## Load-bearing foot-guns (enforce in any code written)
1. **Orientation formats** (CONFIRMED on real data): `start_rotation` is a **quaternion w-FIRST
   `[w,x,y,z]`** → reorder to canonical `xyzw`; `reference_path` is **EULER `[x,y,z,pitch,roll,yaw]`
   (rad, yaw=row[5], 6-wide)**, NOT a quaternion. `len(reference_path)==len(actions)`. The
   `quaternion_trap.json` fixture must keep failing loudly if the start reorder is skipped.
2. **BGR→RGB** before DINOv3 (AirSim Scene is BGR) — record the flag in the manifest.
3. **AirSim msgpack-RPC single-threaded** → Lock every `client.X()` call.
4. **NED z-down** (`GO_UP=−z`).  5. **Phases A–C standalone** — no fly0/navdreamer import.
6. **No EMA/VICReg** (frozen+cached target).  7. **No blobs** — only tiny fixtures under `fixtures/`.
8. **Pure tier stays pure** — `schemas, actions, frames, config, manifest, audit` import numpy/pyyaml
   only; CI imports them. Guard every torch/airsim import (lazy/in-function).

## Guardrails
- Architecture is **LOCKED** (`arch-design-2026-06-08-latent-pred`) — the refactor only re-organizes how
  the **swept knobs** (T, H, predictor depth/heads, ensemble K) are sourced (→ config single source of
  truth); the DINOv3/AirVLN-intrinsic shapes (196/768) stay constants. Do NOT relitigate the design.
- Do NOT modify `third_party/` or the sibling `fly0`/`navdreamer` repos.
- After ~3 failed patches on the same root failure: stop, use WebSearch + Explore + read the AirVLN
  source before patching further.

## The code review (why we pivoted) — short form
HIGH: H1 no single source of truth for swept ablation knobs (duplicated constants; config is
decorative) · H2 config is an untyped/unvalidated/mutable dict (violates the repo's own
immutability+validation rules) · H3 only the loader-INPUT tuple is typed — the OUTPUT seams
(PredictorOutput/TrustReadout/Waypoint) the ablations pivot on are prose-only. MEDIUM: M1
quaternion/frame primitives misplaced in actions.py + reached via private imports (worse after 5b) ·
M2 test_frames no-flip-vs-fly0 missing (the #1 foot-gun is unmitigated) · M3 audit per-slice checks
mis-scoped to per-episode · M4 StepSample lacks history/language masks · M5 manifest stringly-typed +
triple-duplicated constants. Full text + the must-fix list: in the planning prompt
(`plans/planning-prompt-2026-06-08-refactor-before-phaseB.md`).

## TODO carried forward (not blocking the refactor)
- File the real-data finding (`reference_path` = Euler, not quaternion; `len(ref)==len(actions)`;
  5b audit clean) in the vault `latent-pred-pipeline/` (corrects the Phase-A data-audit assumption).
