# DEV_LOG — vllatent-ego-drone

Append-only, **newest entry on top**. Read this first each iteration to find the current position,
then re-read the relevant step in `plans/phase-a-data-and-io-contract.md`. Project-level *why* lives in
the vault (`latent-pred-pipeline/`), not here; this log tracks *code state* + step status.

## Step status table

| step | status | date | notes |
|---|---|---|---|
| 1 — scaffold + git + GitHub + codegraph | done | 2026-06-08 | scaffold+git+codegraph green; private repo `zhihao-acc/vllatent-ego-drone` created + pushed direct to github.com (workflow scope added); `origin` resolves, `main` tracks `origin/main` |
| 2 — transcribe I/O contract → docs/io-contract.md | done | 2026-06-08 | DoD item 1; 4 seams + loader tuple + 2 foot-guns transcribed from vault arch §4/§6/§9; DoD grep PASS |
| 3 — pure-tier tuple schemas | done | 2026-06-08 | `vllatent/schemas.py` (StepSample/EpisodeRecord/CacheManifestEntry, frozen+validated) + test_schemas (22 tests) |
| 4 — discrete→4-DoF action mapping | done | 2026-06-08 | `vllatent/actions.py` (Action enum + constants verbatim; apply_delta reproduces env_utils; pose_pair_to_body_delta) + test_actions (64) |
| 5 — AerialVLN JSON audit parser (fixture) | done | 2026-06-08 | `vllatent/audit.py` (parse_episode/audit_episode + AuditReport/QuaternionVerdict + CLI) + tiny & quaternion_trap fixtures + test_audit; `make audit` clean. **`reference_path` schema corrected to 6-wide Euler in 5b** |
| 6 — fetch real dataset JSON slice | done | 2026-06-08 | USER downloaded full splits (Kaggle/Baidu, NOT S3); `fetch_aerialvln_json.sh` finished (slicer); `train.slice.json` (50 eps); CC BY-NC-SA 4.0 |
| 5b — audit on real slice | done | 2026-06-08 | 50/50 ok, ~10198 transitions **0 Δ-mismatches**, all 8 classes, quaternion consistent (34/50 would corrupt yaw w/o reorder) |
| 7 — DINOv3 encoder wrapper | pending | | contract test AUTONOMOUS / real weights USER-GATED |
| 8 — render harness (teleport+capture) | pending | | unit AUTONOMOUS / live render USER-GATED (docker+UE4) |
| 9 — render→encode→cache + manifest | pending | | manifest test AUTONOMOUS / small-slice build USER-GATED |
| 10 — cached-latent loader | pending | | `vllatent/data/loader.py` + test_data_shapes (tiny_dump) — DoD item 3 code |
| 11 — loader over real dump | pending | | USER-GATED (depends on 9) — DoD item 3 |
| 12 — size full render→cache job | pending | | sizing + guard AUTONOMOUS / bulk run USER-GATED |
| 13 — Phase-A DoD verification | pending | | USER-GATED final sign-off; do NOT auto-flip done |

Statuses: `pending` / `in_progress` / `done` / `blocked`.

---

## 2026-06-08 — steps 6 + 5b: real AerialVLN slice + audit (schema corrected from real data)
**Status:** step 6 → done; step 5b → done. Step 5's `reference_path` assumption CORRECTED from real data.
**Real-data finding (the load-bearing correction).** The scaffold/plan/fixtures assumed
`reference_path` rows were 7-wide quaternions `[x,y,z,qx,qy,qz,qw]`. The **real AerialVLN** rows are
**6-wide EULER `[x,y,z,pitch,roll,yaw]` (radians)** — pitch=roll≡0 (4-DoF), **yaw = row[5]**; only
`start_rotation` is a quaternion (w-FIRST). Also **`len(reference_path) == len(actions)`** (NOT +1):
`reference_path[0]` is the start pose, `actions[t]` drives `ref[t]→ref[t+1]`, terminal STOP has no
stored next pose. Validated `vllatent.actions` against raw data: **0 mismatches across 39,133
transitions / 200 episodes** (all motion classes) — the action arithmetic was already correct; only the
pose *format* assumption was wrong.
**What's done.** Corrected `frames.py` (drop the bogus `reference_path=xyzw` constant → add
`REFERENCE_PATH_ORIENTATION/ROW_WIDTH/YAW_INDEX`), `schemas.py` (`EpisodeRecord.reference_path` (P,6)),
`audit.py` (Euler `_euler_row_to_pose`, yaw=row[5] quaternion verdict, alignment `len(ref)==len(actions)`
+ start-pose anchor, tuple width 6), `docs/io-contract.md` (foot-gun #1 rewritten: quaternion-vs-Euler),
both fixtures regenerated to the real Euler layout, tests updated (smoke/schemas/audit). Finished
`scripts/fetch_aerialvln_json.sh` (Kaggle/Baidu source doc + local slice-of-N writer for the real
`{"episodes":[...]}`); sliced `train.slice.json` (50/16386).
**Tested (5b on real slice).** `python -m vllatent.audit` over `train.slice.json`: **50/50 ok,
~10,198 transitions, 0 Δ-mismatches, all 8 action classes, 50/50 quaternion reorder-consistent,
34/50 episodes would corrupt yaw without the reorder**; 14 distinct scene_ids. Full pure sweep green
(import-smoke / lint / typecheck / 102 tests / `make audit` / blob-guard). License **CC BY-NC-SA 4.0**
recorded. `data/` gitignored — no JSON committed.
**Open / next.** Step 7 — DINOv3 encoder wrapper (`vllatent/encode/dinov3.py`): contract test is
AUTONOMOUS (monkeypatched backbone, BGR→RGB boundary, `(196,768) fp16`); real-weight smoke is dev-gated.
Then steps 8 (render unit) / 9 (cache manifest) autonomous halves, 10 (loader), 12 (sizing).
**Vault.** File the `reference_path = Euler (not quaternion)` + `len(ref)==len(actions)` finding under
`latent-pred-pipeline/` (corrects the Phase-A data-audit assumption).

---

## 2026-06-08 — step 5: AerialVLN-JSON audit parser (fixture half) + STOP at step 6
**Status:** pending → done (AUTONOMOUS). **Pure lane 2→5 GREEN; loop STOPS at step 6 (USER-GATED).**
**What's done.** `vllatent/audit.py` (pure numpy/stdlib): `parse_episode(dict)→EpisodeRecord` (reorders
`start_rotation` w-FIRST → canonical xyzw — foot-gun #1, schema confirmed against AirVLN `env.py`:
`Quaternionr(x=sr[1],y=sr[2],z=sr[3],w=sr[0])`, `instruction.instruction_text`, `goals[].position`,
`reference_path`=[x,y,z,qx,qy,qz,qw]); `audit_episode(dict)→AuditReport` with `QuaternionVerdict`
(reorder_consistent + **naive_would_mismatch** = flags the would-be wrong yaw if reorder skipped),
`actions[t]↔reference_path[t]→[t+1]` alignment, derived-Δ vs `actions.action_to_delta` (tol 1e-3),
per-action counts, tuple completeness, scene_id range, license. CLI `python -m vllatent.audit
--episode <json> [--report -]`. Two committed fixtures generated by STEPPING `apply_delta`:
`tiny_episode.json` (all 8 action classes, 9 poses, clean) + `quaternion_trap.json` (start yaw 90°;
naive xyzw read mislabels it as 0° → audit flags it).
**Tested.** `pytest -q tests/test_audit.py` → 7 passed; `make audit` exit 0 (tiny clean). Full pure
sweep green: import-smoke / lint / typecheck / `pytest -m "not torch and not sim"` (102 passed) /
`make audit` / blob-guard (fixtures are tiny text, allowed).
**Open / next — STOP CHECK (next step is USER-GATED).** Step 6 = fetch a real AerialVLN slice from S3
(`aerialvln.s3.ap-southeast-2.amazonaws.com/dataset/aerialvln/`). `scripts/fetch_aerialvln_json.sh` is
still a STUB (`exit 2`) — finishing it needs the real split-JSON layout + the user's S3/CN-network
situation, which step 6 surfaces. Loop paused; promise `PHASE A PURE LANE GREEN` emitted; `.claude/
ralph-loop.local.md` removed (deterministic stop). Hand the download block to the user; do NOT
auto-mark step 6 done.
**Vault.** No new decision (audit implements the locked I/O contract + foot-gun #1).

---

## 2026-06-08 — step 4: discrete→continuous-4-DoF action mapping
**Status:** pending → done (AUTONOMOUS).
**What's done.** `vllatent/actions.py` (pure numpy, NO airsim): `Action(IntEnum)` + step constants
transcribed VERBATIM from `third_party/AirVLN/airsim_plugin/airsim_settings.py` (STOP=0…MOVE_RIGHT=7;
FORWARD/LEFT_RIGHT=5, UP_DOWN=2, TURN=15). `action_to_delta(id)→(4,) f32` = canonical body-frame
`(dx,dy,dz,dyaw_deg)`, NED z-down (GO_UP=−z), body-right=+y, lateral=±5. `apply_delta(pose,id)`
reproduces `env_utils.getPoseAfterMakeAction` EXACTLY — incl. the AirSim quaternion↔euler formulas
reproduced in-module (`to_eularian_angles` yaw, `to_quaternion(0,0,yaw)`), pitch/roll forced 0, the
yaw-wrap at ±180, forward `unit_z==0`, and the `(yaw+90°)` body-lateral with LEFT×(−1).
`pose_pair_to_body_delta(a,b)` = the inverse the step-5 audit will use to verify dataset poses vs
quantized deltas.
**Tested.** `pytest -q tests/test_actions.py` → 64 passed: enum/constants, per-action deltas,
`apply_delta` at known starts (forward planar + yaw-following, lateral sign, z up/down, ±15° turn, STOP
identity), and a 6-yaw × 8-action round-trip `apply_delta→derive == action_to_delta` (pre-validates the
step-5 audit). Full pure sweep green: import-smoke / lint / typecheck / `pytest -m "not torch and not
sim"` (95 passed) / blob-guard.
**Open / next.** Step 5 — AerialVLN-JSON audit parser (`vllatent/audit.py`) + tiny_episode &
quaternion_trap fixtures + test_audit; then `make audit` clean. After step 5 → step 6 (S3 download,
USER-GATED) = STOP CHECK.
**Vault.** No new decision (faithful reproduction of the AirVLN ground-truth action arithmetic).

---

## 2026-06-08 — step 3: pure-tier tuple schemas
**Status:** pending → done (AUTONOMOUS).
**What's done.** `vllatent/schemas.py` (numpy + stdlib only, no torch): three frozen dataclasses with
boundary validation — (1) `StepSample` = the loader tuple `(z_t, history_latents, lang_tokens,
action_id, z_next, delta_4dof, future_frame_rgb)` with the locked shapes/dtypes pinned as module
constants (PATCH_TOKENS=196, EMBED_DIM=768, HISTORY=3, HORIZON=4, N_ACTIONS=8, DOF=4; latents fp16,
delta f32, rgb uint8); (2) `EpisodeRecord` = parsed AerialVLN episode (quaternions canonical xyzw,
actions int-aligned with reference_path); (3) `CacheManifestEntry` with `to_dict`/`from_dict` whose
keys satisfy `vllatent.manifest.validate_manifest`. Array records use `eq=False` (numpy `__eq__` is an
array) but stay `frozen`. `__post_init__` raises TypeError/ValueError with specific messages on a
contract breach.
**Tested.** `pytest -q tests/test_schemas.py` → 22 passed (shapes/dtypes, immutability, bad-input
rejection, manifest-entry JSON round-trip + cross-check against the manifest validator). Full pure
sweep green: import-smoke / lint / typecheck / `pytest -m "not torch and not sim"` (31 passed) /
blob-guard.
**Open / next.** Step 4 — discrete→continuous-4-DoF action mapping (`vllatent/actions.py` +
`tests/test_actions.py`), transcribing AirVLN constants + reproducing `env_utils.getPoseAfterMakeAction`.
**Vault.** No new decision (schemas implement the locked I/O contract).

---

## 2026-06-08 — step 2: transcribe I/O contract → docs/io-contract.md
**Status:** pending → done (AUTONOMOUS).
**What's done.** Wrote `docs/io-contract.md` — a *transcription* (not a re-derivation) of the LOCKED
I/O contract from vault `[[arch-design-2026-06-08-latent-pred]]`: the §4 tensor I/O table; the four
seams — (a) action repr = discrete codebook 0–7 → per-step FiLM, with the verbatim AirVLN enum
(STOP=0…MOVE_RIGHT=7) + step constants (FORWARD/LEFT_RIGHT=5, UP_DOWN=2, TURN/TILT=15); (b) language =
frozen SigLIP/CLIP text tower 512→768 → cross-attention; (c) uncertainty = deployed single-pass
horizon head, with K=5 ensemble + V-JEPA-2 marked Phase C (documented, not built); (d) waypoint→EGO =
continuous 4-DoF (Δx,Δy,Δz,Δψ) AirSim-NED body + the NED→FLU→ENU remap marked **NOT executed in
Phase A**. Pinned the loader output tuple (arch §6 item 5) + a "Frame & convention hazards" section
covering both foot-guns (quaternion order `w-FIRST` vs xyzw; `BGR`→RGB) + licenses (CC BY-NC-SA 4.0).
**Tested.** `test -f docs/io-contract.md && grep -q "NOT executed in Phase A" && grep -q "w-FIRST" &&
grep -q "BGR"` → PASS.
**Open / next.** Step 3 — pure-tier tuple schemas (`vllatent/schemas.py` + `tests/test_schemas.py`).
**Vault.** No new decision (pure transcription of the locked arch doc).

---

## 2026-06-08 — step 1: GitHub remote wired → DONE
**Status:** in_progress → done.
**What's done.** Created private GitHub repo `zhihao-acc/vllatent-ego-drone`; wired `origin`
(fetch+push = **direct github.com**, no mirror — direct connect works from this host). Pushed `main`
(`7ff793c`) after adding the `workflow` token scope (required to create `.github/workflows/ci.yml`).
`git ls-remote origin` resolves; `main` tracks `origin/main`.
**Tested.** Re-verified the full step-1 DoD today: codegraph_status healthy (19 files / 87 nodes / 87
edges, `.codegraph/codegraph.db` present + gitignored); `make import-smoke` / `lint` / `typecheck` /
`test` green (9 passed); `ALL=1 bash scripts/check_no_blobs.sh` OK.
**Open / next.** Step 1 complete. Ralph loop now closes steps 2→5 autonomously (io-contract → schemas →
actions → audit+fixtures), then STOPS at step 6 (S3 dataset download, **USER-GATED**).

---

## 2026-06-08 — step 1: scaffold + git + GitHub + codegraph
**Status:** pending → in_progress.
**What's done.** Scaffolded `/home/zh/CODE/vllatent-ego-drone` mirroring upipe, adapted for a training
repo with a **pure / torch / sim** tier split. Authored: `CLAUDE.md` (incl. the Wiki Knowledge Base
fetch-context section + locked-arch pointer + load-bearing invariants), `.claude/ralph-rules.md` (tier
gates + user-gated-step rule + deterministic stop), `DEV_LOG.md`, `README.md`, `.gitignore`,
`pyproject.toml` (package `vllatent`, py3.10, torch/sim extras), `Makefile`, `.github/workflows/ci.yml`
(torch-free pure-tier lane), `docs/TOPOLOGY.md`, `configs/{default,data_audit}.yaml`, `scripts/`
(check_no_blobs + ralph + 4 user-gated command-block stubs), `tests/{conftest,test_smoke}.py`, and the
`vllatent/` package: functional infra (`config.py`, `manifest.py`) + importable stubs
(`schemas/actions/frames/audit` pure tier; `encode/`, `data/`, `render/`, `cache.py` lazy torch/sim).
**Tested.** `make import-smoke`, `make lint`, `make typecheck`, `make test`, `ALL=1 bash scripts/check_no_blobs.sh`,
manifest round-trip (see commit). git init -b main + first commit.
**Open / next.** GitHub repo create + CN-mirror push is **USER-GATED** (command block provided); codegraph
`init`+`index` then verify with `codegraph_status`. Stays `in_progress` until the remote resolves +
codegraph verified. Then ralph closes steps 2→5 autonomously.
**Vault.** Will update `[[dev-decision-2026-07-latent-pred-pipeline]]` §8 to repo `zhihao-acc/vllatent-ego-drone`, package `vllatent`.
