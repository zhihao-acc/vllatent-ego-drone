# Topology — active B3-CS boundaries

The active state is defined by
`plans/phase-b3-causal-ski-sim-latent-decoder.md`, `.codex/ralph-rules.md`, and
the newest verified `DEV_LOG.md` entry.

## Execution tiers

| Tier | Code | Runtime | Current state |
|---|---|---|---|
| PURE | `vllatent/sim/`, `schemas.py`, `config.py`, `manifest.py`, selected ingest math | CPython + NumPy/PyYAML, CI | active and gated |
| TORCH | `encode/`, `data/`, `model/`, `train/`, remaining ingest | local torch environment | no active training authority |
| BLENDER | `scripts/blender/b3_cs3_bridge.py` | pinned Blender 4.5.11, Cycles CPU | CS3 proof complete; no new render authorized |
| CONTROLLER | external existing controller boundary | offline only after CS10 | out of current scope |

The completed CS3 proof used local Blender/Cycles CPU and tracked immutable
manifests under `manifests/b3_cs3/`. Renderer-neutral mechanics never import
`bpy`; the bridge is the only Blender boundary. AirSim and the historical
`fly0-m1` topology are no longer repository runtime tiers.

## Resource map

| Resource | Role |
|---|---|
| Dev box | PURE development, completed CS3 CPU proof, future explicitly authorized local cards |
| AutoDL H20 | no active role; B3.7/H20 remains ineligible and SSH is hands-off |
| Lab GPUs | no active role |
| Jetson Orin NX | possible future deployment target only after all model/decoder/interface gates |

Any new Blender CPU generation, encoding, training, GPU/H20, SSH, Docker,
controller, publication, or real-flight operation requires explicit authority.

## Planned causal data flow

```text
one deterministic skier root/history
        ├── zero camera branch
        ├── +/- yaw
        ├── +/- forward
        ├── +/- lateral
        └── +/- vertical
                 │
                 ▼
paired RGB/masks/labels -> frozen DINO patch latents
                 │
                 ▼
H=3 latents + T=8 four-channel actions + separate dt
                 │
                 ▼
T=8 predicted latents -> standalone (cx, cy, log_h, p_visible) decoder
```

CS4 is the next USER gate, but it cannot execute until its missing delegated
normative specification is restored or completely migrated and reviewed.
