# Planning-agent prompt — "Refactor before Phase B" (2026-06-08)

> Hand this verbatim to a planning agent (`planner` / `architect`). **Planning only — produce a plan
> document, do NOT write code or modify any file except the new plan.**

---

ROLE: You are the planning agent for **vllatent-ego-drone**. A code review of the Phase-A pure lane
(steps 1–5, committed) plus the step-5b/6 real-data reconciliation has surfaced **structural issues
that must be fixed BEFORE Phase B**. Produce an ADJUSTED, ralph-executable plan — a **"Phase A.5:
refactor before Phase B"** — that front-loads the must-fixes by leverage and then re-sequences the
remaining Phase-A steps (7–13) to consume the new structures. Do NOT relitigate the locked
architecture. Do NOT write code.

## Repo + source of truth
- Repo: `/home/zh/CODE/vllatent-ego-drone` (git `main`, package `vllatent`).
- READ IN ORDER: `CLAUDE.md` (LOCKED arch · tier split · invariants) → `.claude/ralph-rules.md`
  (iteration protocol + quality gates) → `DEV_LOG.md` (current position; newest on top) →
  `plans/phase-a-data-and-io-contract.md` (the executable plan you are adjusting) → `docs/io-contract.md`
  (the seams). Use the codegraph MCP (`mcp__codegraph__*`, projectPath the repo) to confirm
  caller counts before claiming "zero downstream callers."
- Vault (authoritative INTENT — do NOT relitigate):
  `/home/zh/Documents/Obsidian Vault/projects/vln-ego-drone/latent-pred-pipeline/` —
  `arch-design-2026-06-08-latent-pred` (LOCKED network spec + I/O contract + data-audit spec) +
  `dev-decision-2026-06-07-latent-pred-pipeline` (phases/DoD).

## Current state (done, committed, pushed, green)
- **Steps 1–5 done**: scaffold+git+codegraph · `docs/io-contract.md` · `vllatent/schemas.py`
  (StepSample/EpisodeRecord/CacheManifestEntry) · `vllatent/actions.py` (Action enum + `apply_delta`
  reproducing `env_utils.getPoseAfterMakeAction`) · `vllatent/audit.py` + fixtures.
- **Steps 6 + 5b done**: real AerialVLN slice fetched (Kaggle `shuboliu/aerialvln` / Baidu mirror —
  NOT the S3 path the scaffold guessed) and audited clean — **50/50 episodes ok, 0 Δ-mismatches over
  ~10,198 transitions, all 8 action classes, quaternion reorder consistent**.
- **Real-data correction already applied (commit `95bcb8e`)**: `reference_path` rows are **6-wide
  EULER `[x,y,z,pitch,roll,yaw]` (radians, yaw=row[5])**, NOT a 7-wide quaternion; **`len(reference_path)
  == len(actions)`**. The action arithmetic was already correct (0 mismatches vs 39,133 real
  transitions). NOTE: the code review below was written against the **step-5 commit** (it still
  describes `reference_path` as a quaternion and cites `DOF+3=7` at audit.py:211 in L1) — its
  structural findings remain valid; reconcile L1/M3 with the now-Euler layout.
- Pure lane green: `import-smoke` / `lint` / `typecheck` / `pytest` (102) / `make audit` / blob-guard.
- The ralph loop is STOPPED (`.claude/ralph-loop.local.md` removed).

## The architecture is LOCKED — build the refactor AROUND it (do NOT relitigate)
Frozen DINOv3 ViT-B/16, RGB-only 224², last-layer patch tokens **196×768 fp16 CACHED**; ~120M
block-causal predictor (D=768, depth=12, heads=12, MLP=3072); discrete codebook→per-step FiLM action;
frozen SigLIP/CLIP text tower 512→768→cross-attention; **H=3 / T=4**; trust = deployed single-pass
horizon head (K=5 ensemble + V-JEPA-2 surprise = **Phase C**); continuous 4-DoF waypoint head
[768→512→256→4]; **NO EMA / NO VICReg** (frozen+cached target). Distinguish **DINOv3-intrinsic shapes**
(196 tokens, 768 dim — stay module constants) from **SWEPT ablation knobs** (T, H, predictor
depth/heads, ensemble K — must become config-driven).

---

## INPUT — code review findings to address (verbatim)

### 🔴 HIGH — fix before Phase B

**H1. No single source of truth for the ablation knobs.** The swept hyperparameters — HORIZON (T,
sweep 2–6 in C), HISTORY (H), predictor depth/heads (8-vs-12 in B), ensemble K — exist as hardcoded
module constants duplicated across modules with nothing reconciling them:
- `HORIZON=4` / `HISTORY=3` in `schemas.py` and re-declared in `data/loader.py`
- `predictor.horizon/history/depth/heads` in `configs/default.yaml`
- step constants in `actions.py` and `configs/default.yaml` and `configs/data_audit.yaml`
- patch_tokens/dim hardcoded a third time in `manifest.empty_manifest()`

The config file is currently decorative — `schemas.py` and `loader.py` don't read horizon from it;
`load_config` has zero callers (codegraph-confirmed). For a repo whose entire deliverable is "flip the
ablation via config, not code surgery," a horizon sweep today means editing the constant in
`schemas.py`, the constant in `loader.py`, the yaml, and praying they agree. Single biggest structural
risk, forming in Phase A. **Fix:** one typed config object is the source of truth; modules read
T/H/depth/K from it; the LOCKED-but-fixed shapes (196/768, DINOv3-intrinsic) stay constants, the swept
ones do not.

**H2. The config is an untyped `dict[str, Any]` with no schema, no validation, no immutability**
(`config.py`). Most ablation-load-bearing surface in the repo, and the only boundary with zero
validation — while `schemas.py` rigorously validates the data tuples. Violates the repo's own two
CRITICAL rules (boundary schema-validation; immutability). No config-snapshot / resume /
"save-every-config" architecture, which the training-playbook marks as a Phase-B SOP. `load_config` is
orphan + untested (implemented, 0 callers, no test) — env-expansion behavior unverified. **Fix:** a
frozen `Config` dataclass tree with `from_yaml` + validation, mirroring how `StepSample` guards its
boundary.

**H3. Only the loader-INPUT tuple is a typed contract; the OUTPUT seams the ablations pivot on are
prose-only.** `StepSample` is a clean frozen dataclass. But the seams that −trust / −verifier /
swap-predictor flip across — predictor output `ẑ_{t+1..T}` (T,196,768), trust readout {p_j, k*, σ},
waypoint (4,) + its remap stages — live only as tables in `io-contract.md`. These are pure-tier,
numpy-typeable, CI-gateable today. If not pinned now, Phase B threads them as bare tuples/dicts and the
−trust ablation becomes code surgery instead of a config flag. **Fix:** add `PredictorOutput`,
`TrustReadout`, `Waypoint` frozen dataclasses to `schemas.py` now.

### 🟡 MEDIUM

**M1. Quaternion/frame primitives live in the wrong module, and cross-module use is via private
symbols.** The yaw/quaternion math (`_yaw_from_xyzw`, `_xyzw_from_yaw`, `_wrap_pi`) sits in
`actions.py`, and `audit.py` reaches into `actions.py`'s private helpers (codegraph-confirmed
cross-module caller). Meanwhile `frames.py` — designated owner of the #1 frame invariant — holds only
constants. The frame/quaternion concern is scattered across three modules and owned by none. **Fix:**
extract a `frames.py`-owned quaternion module (public `yaw_from_xyzw`, `xyzw_from_yaw`, `wrap_pi`,
`reorder_wxyz_to_xyzw`); `actions.py` and `audit.py` both import it. Pre-positions `frames.py` for the
Phase-D NED→FLU→ENU remap it must host. *(Note: the step-5b correction ADDED more private cross-module
imports — `audit.py` now imports `_xyzw_from_yaw`/`_yaw_from_xyzw`/`_wrap_pi`/`Pose` from `actions.py` —
so M1 is now more urgent.)*

**M2. The re-derivation safety net is not in place.** Not importing fly0 (standalone A–C) and
re-deriving the frame remap in `frames.py` is defensible — provided `test_frames.py` pins the no-flip
basis against fly0's `frames.py` semantics. Right now `frames.py` has no remap logic, `test_frames.py`
doesn't exist, only the string constants are tested. The re-implementation risk (the #1 foot-gun) is
unmitigated. **The no-flip test must land WITH the frames logic, as a hard gate — not "after."**

**M3. The audit mis-scopes per-slice checks to per-episode.** `all_action_classes_present`,
`scene_id_range`, `splits_present` are dataset-level concepts, but `audit_episode` computes them
per-episode and `_main` loops without aggregating. On real data most episodes won't contain all 8
classes, so per-episode `all_action_classes_present` is meaningless, and DoD item 2's "all 8 classes
across the slice / splits present / scene_id∈[1,25]" is not computed anywhere. The fixture passes only
because it's contrived. **Fix:** an `AuditSummary` aggregator over the slice. *(Note: step 5b passed via
an ad-hoc script-level aggregation — make it first-class + tested.)*

**M4. `StepSample` omits the masks the predictor will need.** `history_latents` is documented
"padded+masked at episode start" but there's no `history_mask` field; `lang_tokens` is variable-M with
no padding mask for batching. Block-causal masking at episode boundaries and language padding are real
Phase-B needs the current contract can't express, so step 10 will invent out-of-contract masks. **Add
them to the seam now.**

**M5. Provenance manifest is stringly-typed and triple-duplicated.** `empty_manifest()` hardcodes
encoder id / 196 / 768 (a third copy of H1's constants); `CacheManifestEntry.to_dict()` ↔
`validate_manifest` key-agreement is hand-maintained, guarded by exactly one test. The manifest is
reproducibility-load-bearing per the training-playbook. **Fix:** a typed manifest builder fed from the
H1 single config, replacing the hand-kept dict.

### 🟢 LOW
- **L1.** `audit.py` used `DOF + 3` (=7) for the pose-row width — a semantic pun. Use an explicit
  constant. *(Partly addressed by the step-5b Euler correction, which switched to a 6-wide row +
  `REFERENCE_PATH_ROW_WIDTH`; verify no `DOF+3` pun remains and the pose-width constants are explicit.)*
- **L2.** `HORIZON` defined in `schemas.py` but documented "not a StepSample field"; `loader.py`
  re-declares it locally instead of importing. (Subsumed by H1.)
- **L3.** `load_config` orphan + untested. (Subsumed by H2.)

### Reuse audit (verdicts)
- fly0 `geometry/frames.py` (NED↔ENU / no-flip): **re-derived, not reused — by design**; guardrail
  (test_frames vs fly0) **missing → M2**.
- `env_utils.getPoseAfterMakeAction`: **reproduced, not imported — faithful + well-tested** (64 cases).
- EMA/VICReg/leakage-free machinery: **absent — clean** (design correctly deletes the subsystem). ✅
- upipe scaffold conventions (tier split, ralph, CLAUDE/DEV_LOG): **reused — strongest part of repo.** ✅
- DINO-WM offline-feature caching / frozen-split: **render-once→cache→provenance manifest is
  structurally consistent and more rigorous.** ✅

### Secondary pass (correctness/style): clean
Immutability ✅ · error handling ✅ · naming ✅ · file sizes ✅ (≤283 lines) · behavioral tests ✅ ·
action math correct (traced). Micro-nit: `apply_delta` lateral branch uses
`math.radians(math.degrees(yaw) + 90)` — faithful but a one-line `# == yaw + π/2` comment would help.

### "Refactor before Phase B" — the must-fix list
1. **Single typed config as source of truth (H1+H2+M5+L2):** frozen Config tree, `from_yaml` +
   validation; modules read swept knobs (T/H/depth/K) from it; manifest builder + loader fed from it.
   **Highest leverage — the ablation control surface.**
2. **Pin the output-seam contracts (H3):** `PredictorOutput`, `TrustReadout`, `Waypoint` frozen
   dataclasses in `schemas.py`, so −trust/swap-predictor are config flags over typed seams.
3. **Consolidate quaternion/frame primitives into `frames.py` (M1)** and land `test_frames.py`
   no-flip-vs-fly0 with the remap logic (M2) — close the leaky private import + the unmitigated
   re-derivation risk together.
4. **Add `history_mask` / language padding-mask to `StepSample` (M4)** before step-10 loader.
5. **Add an `AuditSummary` slice aggregator (M3)** — already needed; make it first-class.

Items 1–3 are pure-tier, CI-gated, ~zero downstream callers today — cheapest they will ever be.

---

## YOUR DELIVERABLE — `plans/phase-a5-refactor-before-phaseB.md`
A numbered, **ralph-executable** plan mirroring `plans/phase-a-data-and-io-contract.md`:
- Each step = one ralph iteration → one commit `refactor(phaseA5): step N — …`, with **Do / DoD +
  exact test command (`PY=/home/zh/miniconda3/envs/vln-ego-drone-1.1/bin/python`) / AUTONOMOUS|USER-GATED**.
- Order strictly by leverage; the config SoT (must-fix #1) first, since the seam/manifest/loader steps
  depend on it.
- Cover all five must-fixes + L1; each step names the files it touches and (codegraph-confirmed) the
  current downstream-caller count.
- A "§ Re-sequence of steps 7–13" section: how encoder/render/cache/loader/sizing change to consume the
  new config + typed seams + masks (e.g., loader reads T/H from config and emits the masked
  `StepSample`; manifest built from config; encoder/render/cache intent unchanged).
- A 1-paragraph migration-order rationale + a back-out note per step.

## Guardrails (binding)
- **Pure tier stays pure** (`schemas, actions, frames, config, manifest, audit` import numpy/pyyaml
  only; CI imports them). Config + seams + frames consolidation are all pure-tier.
- **Immutability + boundary validation** (the repo's two CRITICAL rules) for every new type.
- **Fixtures-first**, no blobs, no fly0/navdreamer import in A–C, no EMA/VICReg, don't modify
  `third_party/` or sibling repos, **do not relitigate the LOCKED architecture** (only re-organize how
  its swept knobs are sourced).
- Keep `196`/`768` and the DINOv3/AirVLN-intrinsic constants as constants; move only the **swept** knobs
  (T, H, depth, heads, K) into the config.
- Planning only: write **just** `plans/phase-a5-refactor-before-phaseB.md`. No code, no other edits.
