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
means the label passed the stricter followed-subject trackability gate. B3.4's
old `0.95` AUROC probe is diagnostic only now. B3.4/B3.4a and the depth-6 B3.5
model are complete. B3.6 remains blocked after the review-backed real-transition-
verifier repair: corrected tiny evaluation passes G1b but fails G1a's null-plan
margin and G1d's aggregate counterfactual margins and yaw-geometry requirement.
B3.7/H20 is therefore ineligible.

The active contract and gate definitions live in
`plans/phase-b3-human-conditioned-world-model.md`; `DEV_LOG.md` records the latest
verified result. See `CLAUDE.md` for broader context and historical invariants.

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

The torch tier (`pip install -e ".[torch]"` -> `make test-torch`) runs on the dev box. H20 is
user-gated and currently ineligible while B3.6 is blocked.
The old AirSim render-cache path is historical; Phase B trains on sports latent caches.

## Workflow

Iterated under a **Ralph loop** using `.codex/ralph-rules.md`. Each step is tracked in `DEV_LOG.md`
against the active B3 plan. G0/K1/K2 are completed diagnostics; the corrected G1 gates must pass
before any H20 training command is prepared.

Historical Phase-A AerialVLN artifacts are retained only where they still support pure contracts.
