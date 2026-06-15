# Full cache-build sizing — AerialVLN train split

> **Created 2026-06-15 (A5.17).** Estimates derived from the 5-episode small-slice build
> (A5.14 verified 2026-06-15). Update after the full build completes.

## Dataset scope

| Metric | Value | Source |
|--------|-------|--------|
| Episodes (train split) | 50 | `data/aerialvln_json/train.slice.json` |
| Total poses | 10,248 | sum of `len(reference_path)` |
| Total transitions (trainable) | 10,198 | poses − 50 terminal STOPs |
| Mean poses/episode | 205.0 | |
| Min / max poses | 35 / 735 | |

## Per-episode disk (uncompressed .npz arrays)

For an episode with N poses:

| Array | Shape | Dtype | Bytes/episode (N=205) |
|-------|-------|-------|----------------------|
| `latents` | (N, 196, 768) | fp16 | N × 301,056 = **61.7 MB** |
| `actions` | (N,) | int64 | N × 8 = 1.6 KB |
| `deltas` | (N, 4) | f32 | N × 16 = 3.2 KB |
| `lang_tokens` | (77, 768) | fp16 | 118,272 = **115.5 KB** |
| `waypoint_4dof` | (N, 4) | f32 | N × 16 = 3.2 KB |
| `teacher_pose6` | (N, 6) | f32 | N × 24 = 4.8 KB |
| `rollpitch_resid` | (N,) | f32 | N × 4 = 0.8 KB |
| `disagreement` | (N,) | f32 | N × 4 = 0.8 KB |
| `vjepa_surprise` | (N,) | f32 | N × 4 = 0.8 KB |
| **Total per episode** | | | **~61.8 MB** |

Latents dominate (>99.8% of disk). fp16 `(196, 768)` = 294 KB per frame.

## Full cache disk estimate

| Scenario | Episodes | Est. uncompressed | Est. compressed (.npz) |
|----------|----------|-------------------|----------------------|
| Small slice (verified) | 5 | ~300 MB | ~250–300 MB |
| Full train split | 50 | **~3.0 GB** | **~2.5–3.0 GB** |

npz uses zip-store (minimal compression on fp16 latents), so compressed ≈ 85–100% of
uncompressed. **Budget 3.5 GB** for the full cache + manifest.

## Timing estimate (per component, per episode, N=205 mean)

| Component | Device | Est. time/episode | Notes |
|-----------|--------|-------------------|-------|
| **AirSim render** | CPU (sim) | ~20–40 s | 205 teleport+capture calls, single-threaded RPC |
| **Center-crop + resize** | CPU | ~2 s | 205 × cv2 INTER_AREA, trivial |
| **DINOv3 encode** | GPU | ~5–10 s | 205 × ViT-B/16 forward (224², fp16) |
| **CLIP text encode** | GPU | ~0.5 s | 1 per episode (instruction → (77,768)) |
| **WorldVLN K=5 rollouts** | H20 GPU | **~12–15 min** | 3 segments × 5 rollouts × ~160 steps @ 3.5 it/s |
| **V-JEPA-2 surprise** | GPU | ~10–20 s | 204 transition pairs × ViT-L forward |
| **Total per episode** | | **~13–17 min** | **WorldVLN dominates** |

## Full build wall-clock

| Scenario | Episodes | Est. time | Notes |
|----------|----------|-----------|-------|
| Full train (sequential) | 50 | **~11–14 hours** | WorldVLN-bound |

The build is **resumable** (skips episodes whose `.npz` already exists). If the H20 server
drops, restart the same command — completed episodes are not re-processed.

## Bottleneck analysis

1. **WorldVLN teacher (>90% of wall-clock).** The 8B diffusion backbone runs 160 denoising
   steps per rollout at ~3.5 it/s on H20. K=5 rollouts × ~3 segments/episode = 15 rollout
   calls, each ~46 seconds.
2. **AirSim render** is second (~3–5% of wall-clock). Single-threaded RPC, one teleport+capture
   per pose.
3. **DINOv3 / CLIP / V-JEPA-2** are negligible (<2% combined on a modern GPU).

## GPU memory

| Model | VRAM (fp16) | Where |
|-------|-------------|-------|
| WorldVLN backbone (~8B) | ~20–24 GB | H20 server (remote) |
| WorldVLN VAE + action decoder | ~2 GB | H20 server (remote) |
| DINOv3 ViT-B/16 | ~0.3 GB | fly0-m1 local GPU |
| CLIP ViT-B/32 text | ~0.2 GB | fly0-m1 local GPU |
| V-JEPA-2 ViT-L | ~1.2 GB | fly0-m1 local GPU |
| **Local total** | **~2 GB** | fits any GPU with the sim |
| **Remote total** | **~24 GB** | H20 NVLink |

## Prerequisites checklist

Before running the full build:

- [ ] fly0-m1 docker running, UE4 scene hot on port 41451
- [ ] H20 WorldVLN server running on port 8001 (ssh tunnel forwarded to localhost:8001)
- [ ] `data/aerialvln_json/train.slice.json` present (50 episodes)
- [ ] ~4 GB free disk in `data/latent_cache/`
- [ ] DINOv3 + CLIP + V-JEPA-2 weights cached in `HF_HOME`
- [ ] Confirm small-slice build (A5.14) already verified

## Command

```bash
# Use scripts/run_full_cache.sh (A5.17 guard script — refuses without --i-have-signed-off)
bash scripts/run_full_cache.sh --i-have-signed-off
```
