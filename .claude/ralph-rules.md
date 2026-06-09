# Ralph rules — vllatent-ego-drone Phase A

> **Pinned invariant (read every iteration).** The waypoint output is **AirSim-NED body, yaw-only** —
> remap NED→FLU→ENU, **never hand-roll**, keep `tests/test_frames.py` green. **Cached latents are
> render-once** (log provenance). Data foot-guns: **BGR→RGB** before the encoder; **quaternion order**
> `start_rotation`=wxyz vs `reference_path`=xyzw → reorder to canonical **xyzw**. AirSim msgpack-RPC is
> **single-threaded → Lock every `client.X()`**. Phases A–C are **standalone** (no sibling import).

## Per-iteration protocol

1. **READ** — `DEV_LOG.md` (find position), this file, **`plans/phase-a5-replan-postpivot.md`** (the step
   DoD + exact test command; it **supersedes steps 7–13** of `plans/phase-a-data-and-io-contract.md`).
2. **IDENTIFY** — the lowest `pending` (or `in_progress` you own) step in the DEV_LOG step-status table
   (the **A5.x** rows; the old `7–13` row is `superseded` — skip it). Lowest pending now = **A5.1**.
3. **REVIEW** — re-read the step's DoD + test command; `ls` the relevant subtree; verify prior outputs exist.
4. **EXECUTE** — implement per the quality gates. **Pure-tier / fixtures-first**: write the numpy-only
   logic + its mocked/fixture test so it closes in CI before any sim/GPU is touched. Guard every
   torch/airsim import (lazy) so pure import-smoke stays clean.
5. **TEST** — run the step's command from the Test Command Index. Fix failures **in-iteration**; do not
   defer. If a failure is a real frame/sign bug, fix `actions.py` / `frames.py` / the audit reorder —
   don't paper over it.
6. **RECORD** — update `DEV_LOG.md`: flip the step status, add the dated entry (Status / What's done /
   Tested / Open-next / Vault).
7. **COMMIT** — `git add <specific files>` (**never** `-A` / `.`); commit `feat(phaseA): step N — <one-line>`
   (+ optional body with a `Tested:` line). Push at the STOP CHECK.
8. **STOP CHECK** — if the next pending step is `>= started_step + 3`, OR the next step is **user-gated**,
   OR a test is un-fixably failing: push, output `<promise>PHASE A …</promise>`, and pause.

## Quality gates

- **Pure tier stays pure** — `vllatent.{schemas,actions,frames,config,manifest,audit}` import with
  numpy/pyyaml only. CI imports them. Do NOT add torch/transformers/timm/airsim to these modules.
- **Torch tier import-guarded** — `vllatent.encode.*`, `vllatent.data.*` defer all heavy imports
  (lazy / inside functions); a torch-free box imports the module without crashing; `@pytest.mark.torch`
  tests run only where torch is installed (`make test-torch`).
- **Sim tier import-guarded** — `vllatent.render.*`, `vllatent.cache` defer `airsim`; run only in
  `fly0-m1`; every `client.X()` under a `threading.Lock`.
- **Frame discipline** — any code touching a pose / action delta / NED↔ENU remap names the frame of its
  input(s) in a comment.
- **Cache provenance** — any code that writes a cached latent writes/updates the cache manifest
  (encoder id+revision, dataset slice, quaternion order, BGR→RGB flag, render config hash).
- **No EMA / no VICReg** — frozen+cached encoder; reject any target-EMA / anti-collapse term.
- **No blobs** — never commit weights / `runs/` / cached latents / downloaded JSON / videos / large
  `.npy`. `scripts/check_no_blobs.sh` rejects them; only tiny `fixtures/` are allowed.

## User-gated steps (the load-bearing rule)

Any step that **renders frames**, **dumps cached latents**, downloads the **dataset / model weights**,
or touches the **H20 / AutoDL / docker / SSH / WorldVLN-server**, stays **`in_progress`** until the USER
pastes back the verification output. **Never auto-mark such a step `done`.** The agent produces the exact
command BLOCK (it does NOT drive SSH, launch UE4, run `docker exec`, rent/operate a GPU, download multi-GB
weights, or provide keys); the user pastes it and returns the result. Under the A5.x re-plan the
USER-GATED steps are: **A5.7** (real-slice audit re-run), **A5.8** (WorldVLN determinism/weights probe),
**A5.10** (DINOv3 real weights), **A5.11** (WorldVLN teacher, server), **A5.12** (V-JEPA-2 weights),
**A5.13** (live docker+UE4 render), **A5.14** (small-slice cache build), **A5.16** (loader over the real
dump), **A5.17** (bulk run), **A5.18** (final sign-off). The pure-tier steps (**A5.1–A5.6, A5.9**, the
code/mocked halves of A5.7/A5.10/A5.13/A5.14, the loader A5.15, and the sizing guard of A5.17) auto-close
in the loop.

## Deterministic stop / backstop

- Launch with a `--max-iterations` backstop (default 10) and a `--completion-promise`
  (`PHASE A PURE LANE GREEN`). On a STOP CHECK: push, emit `<promise>…</promise>`, pause.
- **Deterministic stop = `rm .claude/ralph-loop.local.md`** (the loop's local, gitignored state file).
  Deleting it halts the loop cleanly even if the stop hook would otherwise re-feed.
- Also stop when a TEST fails un-fixably (mark the step `blocked` in `DEV_LOG.md`) or a user-only action
  is needed.

## Test Command Index

| Surface | Command |
|---|---|
| Pure import-smoke (numpy/pyyaml only) | `make import-smoke` |
| Lint + types | `make lint && make typecheck` |
| Pure unit tests | `pytest -q -m "not torch and not sim"` |
| Torch-tier tests (dev box / H20) | `make test-torch` (`pytest -q -m torch`) |
| Cache-manifest round-trip | `python -m vllatent.manifest --emit-empty \| python -m vllatent.manifest --validate -` |
| AerialVLN audit (fixture, after step 5) | `make audit` (`python -m vllatent.audit --episode fixtures/episodes/tiny_episode.json`) |
| Blob guard (CI mode) | `ALL=1 bash scripts/check_no_blobs.sh` |
| Render (sim, USER-GATED, in `fly0-m1`) | `bash scripts/render_aerialvln.sh` |
| Encode→cache (torch/sim, USER-GATED) | `bash scripts/encode_latents.sh` |

## git remote

Private, https, account `zhihao-acc`: `origin = https://github.com/zhihao-acc/vllatent-ego-drone.git`
with push set to the `ghfast.top` mirror (CN). Push at each STOP CHECK with `git push origin main`.

## DO NOT

- Skip the READ phase · modify the fly0/navdreamer siblings or their `third_party/` · `git add -A` /
  `git add .` · commit weights / `runs/` / cached latents / downloaded JSON / videos / large `.npy` ·
  drive UE4 / `docker exec` / SSH-to-H20 / rent or operate a GPU / provide keys (all user-only — produce
  a command block) · import fly0/navdreamer in Phases A–C · add EMA / VICReg · relitigate the LOCKED
  architecture · after ~3 failed patches on the same root failure, stop and use WebSearch + Explore +
  read the AirVLN source before patching further.
