# vllatent-ego-drone

A compact causal latent world model for sports-following drones.

```text
three observed DINO latent frames + eight future camera/drone actions
    -> eight future DINO latent frames
    -> standalone decoder
    -> eight future (cx, cy, log_h, p_visible) rows
```

Current simulator records use four requested body-FRD command channels in SI
units, with `dt_seconds` stored separately:

```python
COMMAND_FIELDS = (
    "v_forward_m_s",
    "v_right_m_s",
    "v_down_m_s",
    "yaw_rate_rad_s",
)
```

The historical passive-video six-field token remains at
`vllatent.plan_tokens` for compatibility and is never reinterpreted as this
simulator interface.

## Status

- B3-CS1 renderer-neutral contracts: complete 2026-07-15.
- B3-CS2 deterministic skier proof: complete 2026-07-15.
- B3-CS3 Blender 4.5.11/Cycles CPU feasibility proof: complete 2026-07-20.
- B3-CS4 paired 32-root x nine-branch smoke: next USER gate; not started.

CS4 also waits for restoration or reviewed complete migration of normative
CS4+ clauses that the original draft delegated to two reports absent from Git.
B3.6 remains blocked and B3.7/H20 remains ineligible.

## Layout

```text
vllatent/sim/       renderer-neutral deterministic contracts/mechanics (PURE)
scripts/blender/    isolated Blender bridge used by the completed CS3 proof
manifests/b3_cs3/   tracked CS3 asset/rig/scene authority
vllatent/           legacy PURE/TORCH compatibility code pending CS5–CS7 replacement
plans/              active B3-CS plan plus historical evidence
tests/              focused PURE, Blender-boundary, and retained compatibility tests
```

Phase-A AirSim/AerialVLN runtime paths and the B2 direct action policy have been
removed. Passive ingest/loader code remains only until CS5; old model/loss/metric
wrappers remain until CS6/CS7 replacements.

## Local PURE verification

```bash
make setup
make import-smoke
make typecheck
make test
```

The default lane requires only NumPy/PyYAML and gates all renderer-neutral
`vllatent.sim` modules. Torch is optional via `.[torch]`. No AirSim extra exists.

## Workflow

Work proceeds under `.codex/ralph-rules.md`. The current loop stops before CS4
until its normative-spec blocker and USER data-generation gate are both cleared.
No H20 command is eligible.
