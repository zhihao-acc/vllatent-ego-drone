# vllatent-ego-drone

A compact latent world-action model for FPV sports following. The active Phase B path uses cached
DINOv3 latents plus candidate future camera/drone plans to predict future human-centric world state.

```
observed human/camera history + candidate future 6-D camera/drone plan
    -> future person/world latents + person-state trajectory
```

The active B3 plan-token contract is:

```python
PLAN_TOKEN_DIM = 6
PLAN_TOKEN_FIELDS = [
    "unit_dir_x",
    "unit_dir_y",
    "unit_dir_z",
    "log_speed_ratio",
    "yaw_rate_norm",
    "valid",
]
```

Future candidate plans are model inputs. Future person/world targets are labels only:
future latents, person state `(cx, cy, log_h, visibility)`, masks, and confidences
must not enter model `forward`. B2.11c remains evidence and a partial proposal prior;
B2.12/H20 action-imitation training is inactive.

B3 cache person labels keep detector visibility and supervision validity separate:
`person_visible` means a sanitized detector box exists, while `person_state_valid`
means the label passed the stricter followed-subject trackability gate. As of the
latest local refire, B3.4 remains blocked (`G0`/`K2` fail), so do not start B3.5.

> Architecture is **LOCKED** — see the vault `latent-pred-pipeline/arch-design-2026-06-08-latent-pred`.
> Do not relitigate it; build around it. See `CLAUDE.md` for full agent context.

## Layout

```
vllatent/   schemas actions frames config manifest audit    # PURE tier (numpy/pyyaml; CI-gated)
            encode/ data/ ingest/ model/ train/             # TORCH + sports-training tiers
plans/      phase-b3-human-conditioned-world-model.md        # active Phase-B3 plan
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
`DEV_LOG.md` against the active plan. Phase B-3 is the human-conditioned world-model pivot:
local G0/K1/K2/G1 gates must pass before any H20 training command is prepared.

Historical Phase-A AerialVLN artifacts are retained only where they still support pure contracts.
