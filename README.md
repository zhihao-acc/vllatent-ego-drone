# vllatent-ego-drone

A compact latent world-action model for FPV sports following. The active Phase B path uses cached
DINOv3 latents plus past scale-free action/path history to predict future scale-free actions.

```
observed DINO latents + past scale-free history -> direct policy / WAM -> future scale-free actions
```

The current target is `[unit_dir_x, unit_dir_y, unit_dir_z, log_speed_ratio]`. Future actions are
labels only; model conditioning is limited to observed latents and past observed action/path history.

> Architecture is **LOCKED** — see the vault `latent-pred-pipeline/arch-design-2026-06-08-latent-pred`.
> Do not relitigate it; build around it. See `CLAUDE.md` for full agent context.

## Layout

```
vllatent/   schemas actions frames config manifest audit    # PURE tier (numpy/pyyaml; CI-gated)
            encode/ data/ ingest/ model/ train/             # TORCH + sports-training tiers
plans/      phase-b-sports-training.md                      # active Phase-B plan
docs/       TOPOLOGY.md  io-contract.md
configs/    default.yaml  data_audit.yaml
fixtures/   tiny episodes + synthetic latents (committed, tiny)
scripts/    check_no_blobs.sh  ralph.sh  training/ingest helpers
```

## Quickstart (pure tier — dev box / CI)

```bash
make setup          # ruff, mypy, pytest, numpy, pyyaml (NO torch)
make import-smoke    # pure tier imports with numpy/pyyaml only
make lint && make typecheck && make test
```

The torch tier (`pip install -e ".[torch]"` -> `make test-torch`) runs on the dev box / H20.
The old AirSim render-cache path is historical; Phase B trains on sports latent caches.

## Workflow

Iterated under a **ralph loop** (`make ralph` prints the launch command). Each step is tracked in
`DEV_LOG.md` against the active plan. Phase B is the sports-following pivot: local diagnostics must
pass before any H20 training command is prepared.

Historical Phase-A AerialVLN artifacts are retained only where they still support pure contracts.
