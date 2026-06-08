# DEV_LOG — vllatent-ego-drone

Append-only, **newest entry on top**. Read this first each iteration to find the current position,
then re-read the relevant step in `plans/phase-a-data-and-io-contract.md`. Project-level *why* lives in
the vault (`latent-pred-pipeline/`), not here; this log tracks *code state* + step status.

## Step status table

| step | status | date | notes |
|---|---|---|---|
| 1 — scaffold + git + GitHub + codegraph | in_progress | 2026-06-08 | local scaffold + git init + first commit done; **GitHub create + push USER-GATED** (mirror block); codegraph init pending verify |
| 2 — transcribe I/O contract → docs/io-contract.md | pending | | DoD item 1 (transcribe, not re-design) |
| 3 — pure-tier tuple schemas | pending | | `vllatent/schemas.py` + test_schemas |
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
