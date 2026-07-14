# Topology — hardware, processes, and where each tier runs

The active architecture and gate state are defined by
`plans/phase-b3-human-conditioned-world-model.md` and `DEV_LOG.md`. This document maps the
repository tiers onto hardware and processes.

## Hardware

| Box | Spec | Role |
|---|---|---|
| Dev box | RTX 5060 Ti 16 GB, Ubuntu 24.04 | pure/torch development, ingest, and corrected B3 local gates |
| AutoDL H20 | NVLink ~96 GB (saved navdreamer/Wan2.1 mirror, torch 2.8/CUDA 12.x) | USER-GATED B3.7 training only after B3.6 passes; currently ineligible and **SSH HANDS-OFF** |
| Lab | 5× RTX 4090 (24 GB) | optional future work; not part of the active B3.6 gate |
| `fly0-m1` docker | ROS Noetic, `--gpus all --net=host`, AerialVLN scenes at `/opt/aerialvln/{MSBuild2018,AirSimNH,Blocks}` | historical Phase-A render host; not part of active B3 |
| Jetson Orin NX 16 GB | drone companion computer | **deploy** target — the binding size constraint |

## Tier → where it runs

| Tier | Modules | Runs on | Deps |
|---|---|---|---|
| **PURE** | `schemas, actions, frames, config, manifest, audit, ingest/quality, ingest/ego_motion` | CI / any box | numpy, pyyaml |
| **TORCH** | `encode/, data/, model/, train/`, remaining ingest tools | dev box; H20 only after B3.6 | torch 2.8, timm>=1.0.20, declared ingest extras |
| **SIM** | retired in active code | `fly0-m1` docker only, if a historical Phase-A render must be reproduced | airsim, msgpack-rpc; `/opt/aerialvln` scenes |

CI gates ONLY the pure tier (torch-free). The torch tier is covered by `make test-torch`
on tiny fixtures. No active repo module imports AirSim.

## Manual operations (agent gives a command block; the USER runs it)

1. Launch UE4 + AirSim scene in `fly0-m1` (wait for "Listening on port 41451").
2. Restart UE4 every ~4–6 h.
3. Anything over SSH / on the H20 (env build, training, rsync). Do not prepare an H20 run while
   B3.6 is blocked.
4. GitHub repo create + push (CN mirror chain).
5. `sudo` / host-state ops.

## Network (China)

- GitHub: mirror chain `ghfast.top`, `gh.llkk.cc`, `mirror.ghproxy.com`.
- HuggingFace: `export HF_ENDPOINT=https://hf-mirror.com` (DINOv3 and explicitly approved model/data access).

## Data flow (active Phase B-3)

```
Sports FPV videos / frame clips        [ingest_data/, gitignored]
        │  content filter + MegaSaM VO + DINOv3 encode
        ▼
   latent .npz clips with person tracks and scale-free 6-D future plans
        │
        ▼
          cached (196,768) fp16 observed/future latents   [ingest_data/latent_cache/, gitignored]
        │
        ▼
          SportsTrainingDataset → depth-6 plan-conditioned latent/person predictor
                                      │
                                      └─► corrected local G1 gates before any H20 run
```
