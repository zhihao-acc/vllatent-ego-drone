# vllatent-ego-drone (`vllatent`) — Current Agent Context

This file is a concise compatibility entrypoint. `AGENTS.md`,
`.codex/ralph-rules.md`, the newest `DEV_LOG.md` entry, and
`plans/phase-b3-causal-ski-sim-latent-decoder.md` are authoritative.

Read them in that order, then read only the files named by the active card. Do
not consult Obsidian. The previous passive-video B3 plan is historical evidence;
Phase-A AirSim/AerialVLN and controller/CEM queues are retired.

## Current state

- B3-CS1/CS2 completed on 2026-07-15.
- B3-CS3 deterministic Blender feasibility completed on 2026-07-20.
- B3-CS4 is next, but has not started. It needs explicit USER CPU
  data-generation authority.
- Two reports referenced by the original draft plan are absent and were never
  tracked. Because they contain delegated CS4+ numeric/formula contracts, CS4
  also waits for report restoration or a reviewed complete plan migration.
- B3.6 remains blocked; B3.7/H20 remains ineligible.

## Locked boundary

```text
H=3 observed DINO latents + T=8 requested camera actions + separate dt
    -> T=8 predicted DINO latents
    -> standalone decoder -> T=8 (cx, cy, log_h, p_visible)
```

Simulator actions are four SI body-FRD channels
`[v_forward_m_s, v_right_m_s, v_down_m_s, yaw_rate_rad_s]`. They are not the
historical six-field passive-video token. Nine siblings share one exact skier
future and one root-group split; requested and achieved SE(3) stay separate;
camera/branch/visibility/pixels never enter the skier digest.

`vllatent/sim/` is renderer-neutral PURE code. The Blender boundary is
`scripts/blender/b3_cs3_bridge.py`. No new rendering, data generation/encoding,
training, GPU/H20, controller, SSH, Docker, publication, or flight work is
authorized merely because CS3 passed.

Use exact-path staging only, preserve unrelated worktree changes, and never
commit generated data, caches, weights, archives, or source assets.
