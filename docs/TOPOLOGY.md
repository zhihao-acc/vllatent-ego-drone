# Topology — hardware, processes, and where each tier runs

The architecture is LOCKED (vault `[[arch-design-2026-06-08-latent-pred]]`). This doc maps
the **tiers** (CLAUDE.md "Tier split") onto **hardware/processes**.

## Hardware

| Box | Spec | Role |
|---|---|---|
| Dev box | RTX 5060 Ti 16 GB, Ubuntu 24.04 | pure-tier dev + small-slice encode; CI mirror |
| AutoDL H20 | NVLink ~96 GB (saved navdreamer/Wan2.1 mirror, torch 2.8/CUDA 12.x) | **training** (Phase B+) + full encode→cache job. **SSH HANDS-OFF** |
| Lab | 5× RTX 4090 (24 GB) | optional; K=5 ensemble teacher (Phase C) |
| `fly0-m1` docker | ROS Noetic, `--gpus all --net=host`, AerialVLN scenes at `/opt/aerialvln/{MSBuild2018,AirSimNH,Blocks}` | **render** (Phase A) + closed-loop (Phase D). UE4 launched MANUALLY |
| Jetson Orin NX 16 GB | drone companion computer | **deploy** target — the binding size constraint |

## Tier → where it runs

| Tier | Modules | Runs on | Deps |
|---|---|---|---|
| **PURE** | `schemas, actions, frames, config, manifest, audit` | CI / any box | numpy, pyyaml |
| **TORCH** | `encode/, data/` | dev box / H20 | torch 2.8, transformers≥4.56, timm≥1.0.20 |
| **SIM** | `render/, cache` | `fly0-m1` docker only | airsim, msgpack-rpc; `/opt/aerialvln` scenes |

CI gates ONLY the pure tier (torch-free). The torch tier is covered by `make test-torch`
on tiny fixtures; the sim tier runs only in the container.

## Manual operations (agent gives a command block; the USER runs it)

1. Launch UE4 + AirSim scene in `fly0-m1` (wait for "Listening on port 41451").
2. Restart UE4 every ~4–6 h.
3. Anything over SSH / on the H20 (env build, training, rsync of the latent cache).
4. GitHub repo create + push (CN mirror chain).
5. `sudo` / host-state ops.

## Network (China)

- GitHub: mirror chain `ghfast.top`, `gh.llkk.cc`, `mirror.ghproxy.com`.
- HuggingFace: `export HF_ENDPOINT=https://hf-mirror.com` (DINOv3 / V-JEPA-2 / datasets).

## Data flow (Phase A)

```
AerialVLN JSON (actions + reference_path + language)   [data/, gitignored]
        │  teleport-to-pose replay in fly0-m1 (sim tier)
        ▼
   rendered RGB (BGR→RGB)            ──► frozen DINOv3 ViT-B/16 (torch tier)
        │                                        │
        └────────────────────────────────────────┘
                          ▼
          cached (196,768) fp16 latents + manifest   [data/latent_cache/, gitignored]
                          ▼
          CachedLatentDataset (torch tier) → StepSample tuples  → Phase B training (sim-free)
```
