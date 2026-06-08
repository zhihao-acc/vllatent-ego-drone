# vllatent-ego-drone

A **compact latent world-action model for aerial vision-language navigation**. The contribution is
**trust-aware commitment**: a frozen perception backbone + a small action/language-conditioned latent
predictor + a trust layer that decides how far ahead to commit.

```
RGB 224² → [DINOv3 ViT-B/16, frozen, cached] → latent predictor (~120M) → 4-DoF waypoint + trust horizon
```

Action is **discrete-in** (AerialVLN 8-way) / **continuous-4-DoF-out** (Δx, Δy, Δz, Δψ). The frozen,
cached encoder makes training **sim-free** after a one-time render→encode→cache preprocess.

> Architecture is **LOCKED** — see the vault `latent-pred-pipeline/arch-design-2026-06-08-latent-pred`.
> Do not relitigate it; build around it. See `CLAUDE.md` for full agent context.

## Layout

```
vllatent/   schemas actions frames config manifest audit   # PURE tier (numpy/pyyaml; CI-gated)
            encode/ data/                                   # TORCH tier (lazy torch; make test-torch)
            render/ cache                                   # SIM tier (lazy airsim; fly0-m1 only)
plans/      phase-a-data-and-io-contract.md                 # the executable Phase-A plan
docs/       TOPOLOGY.md  io-contract.md(step2)  full-run-sizing.md(step12)
configs/    default.yaml  data_audit.yaml
fixtures/   tiny episodes + synthetic latents (committed, tiny)
scripts/    check_no_blobs.sh  ralph.sh  + user-gated command-block stubs
```

## Quickstart (pure tier — dev box / CI)

```bash
make setup          # ruff, mypy, pytest, numpy, pyyaml (NO torch)
make import-smoke    # pure tier imports with numpy/pyyaml only
make lint && make typecheck && make test
```

The torch tier (`pip install -e ".[torch]"` → `make test-torch`) runs on the dev box / H20; the sim tier
runs only inside the `fly0-m1` docker. See `docs/TOPOLOGY.md`.

## Workflow

Iterated under a **ralph loop** (`make ralph` prints the launch command). Each step is tracked in
`DEV_LOG.md` against `plans/phase-a-data-and-io-contract.md`. Phase A = plumbing + data; the
render/encode/cache job is sized but **not bulk-run** until explicit sign-off.

License of the AerialVLN dataset used in Phase A: **CC BY-NC-SA 4.0** (non-commercial).
