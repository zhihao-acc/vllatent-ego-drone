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
| 4 — discrete→4-DoF action mapping | pending | | `vllatent/actions.py` vs AirVLN env_utils + test_actions |
| 5 — AerialVLN JSON audit parser (fixture) | pending | | `vllatent/audit.py` + tiny + quaternion_trap fixtures |
| 6 — fetch real dataset JSON slice | pending | | USER-GATED (S3 / network / license) |
| 5b — audit on real slice | pending | | USER-GATED (depends on 6) — DoD item 2 |
| 7 — DINOv3 encoder wrapper | pending | | contract test AUTONOMOUS / real weights USER-GATED |
| 8 — render harness (teleport+capture) | pending | | unit AUTONOMOUS / live render USER-GATED (docker+UE4) |
| 9 — render→encode→cache + manifest | pending | | manifest test AUTONOMOUS / small-slice build USER-GATED |
| 10 — cached-latent loader | pending | | `vllatent/data/loader.py` + test_data_shapes (tiny_dump) — DoD item 3 code |
| 11 — loader over real dump | pending | | USER-GATED (depends on 9) — DoD item 3 |
| 12 — size full render→cache job | pending | | sizing + guard AUTONOMOUS / bulk run USER-GATED |
| 13 — Phase-A DoD verification | pending | | USER-GATED final sign-off; do NOT auto-flip done |

Statuses: `pending` / `in_progress` / `done` / `blocked`.

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
