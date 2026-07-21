# vllatent-ego-drone (`vllatent`) — Codex Project Guidance

Read this before working in the repository. When guidance differs, prefer the
newest verified `DEV_LOG.md` entry, `.codex/ralph-rules.md`, and
`plans/phase-b3-causal-ski-sim-latent-decoder.md`.

## Current position

The active research path is a causal sports-following visual world model:

```text
three observed DINO latent frames + eight future camera/drone action steps
    -> eight future DINO latent frames
    -> standalone decoder
    -> eight future (cx, cy, log_h, p_visible) rows
```

`B3-CS1` and `B3-CS2` completed on 2026-07-15. `B3-CS3` completed on
2026-07-20 with a deterministic Blender 4.5.11/Cycles CPU feasibility proof.
`B3-CS4` is the lowest pending card and is a USER-gated 32-root x nine-branch
CPU data-generation smoke. It has not started.

The draft plan originally delegated several CS4+ numeric/formula contracts to
two reports that are absent from this repository and were never tracked by Git.
CS4 is therefore blocked on both explicit USER data-generation authority and
either restoration of those reports or a separately reviewed, complete migration
of their missing normative clauses into the active plan. Never invent or weaken
those clauses.

B3.6 remains blocked; B3.7/H20 remains ineligible. The old B3.8/CEM queue and
Phase-A AirSim/AerialVLN paths are retired.

## Required read order

1. `AGENTS.md`.
2. `.codex/ralph-rules.md`.
3. `DEV_LOG.md`.
4. `plans/phase-b3-causal-ski-sim-latent-decoder.md`.
5. Only files named by the active card.

`plans/phase-b3-human-conditioned-world-model.md` is optional historical
evidence, not active authority. Do not consult Obsidian for this queue. Do not
cite the two absent report paths as if they were available authority.

## Current B3-CS contracts

- Simulator commands are four requested body-FRD channels
  `[v_forward_m_s, v_right_m_s, v_down_m_s, yaw_rate_rad_s]` in SI units.
  `dt_seconds` is separate and fixed at `0.2`. Zero and pure yaw are valid.
- The exact nine branches are zero and plus/minus yaw, forward, lateral, and
  vertical. Siblings share one `root_id`, one `split_group_id`, one history, and
  the exact skier future.
- Requested and achieved SE(3) records remain distinct. Transforms are named
  `T_target_from_source`, row-major float64.
- The canonical skier digest excludes camera, branch, command, visibility,
  pixels, images, and render results.
- Forecast changes must be identifiable from history. No hidden target,
  maneuver, ramp, or trigger may begin after history.
- The historical six-field passive-video token's canonical definition remains
  at `vllatent.plan_tokens`; it must never be interpreted as the simulator
  command.
- Future latents and person/simulator state are labels only and never model
  inputs. The decoder emits only `(cx, cy, log_h, p_visible)`.
- IMU, VINS, radar, depth, terrain safety, obstacle avoidance, trajectory
  optimization, and low-level control remain controller-side.

## Tier boundaries

| Tier | Current modules | Rule |
|---|---|---|
| PURE | `schemas`, `config`, `manifest`, `ingest/quality`, `ingest/ego_motion`, all `vllatent/sim/` | stdlib + NumPy/PyYAML only; no `bpy`, torch, timm, transformers, AirSim, wall-clock, or RNG |
| TORCH | `encode/`, `data/`, `model/`, `train/`, remaining ingest tools | optional heavy imports remain lazy where supported |
| BLENDER | `scripts/blender/b3_cs3_bridge.py` and audited CS3 artifacts | Blender-bundled runtime only; any new CS4 render/data generation is USER-gated |

The old top-level `vllatent.actions`, `vllatent.frames`, and `vllatent.audit`
modules no longer exist. Do not recreate them or add a fake
`vllatent.sim.plan_tokens` alias.

## Data, artifact, and git rules

- Do not commit weights, `.npz`, videos, frames, caches, generated QC reports,
  downloaded archives, or paid/source assets.
- Preserve canonical CS3 manifests and proof artifacts. New renders, data,
  encoding, training, GPU/H20, controller, SSH, Docker, GUI, publication, and
  real-flight work require their own explicit authority.
- Use exact paths for staging. Never use `git add -A` or `git add .`.
- Preserve unrelated dirty-worktree changes. Do not restore or rewrite user work.
- P1 passive ingest/loader removal waits for CS5. Old model/loss/metric wrapper
  removal waits for CS6/CS7 replacements.

## Verification

Run the narrow affected tests first. For PURE changes, run the import/AST gate,
focused pytest, Ruff, mypy, and `git diff --check`. Wait for commands to exit.
Do not run broad training or hardware checks merely for activity.

## Ralph

Follow `.codex/ralph-rules.md`. A Ralph run stops at CS4 until both the missing-
normative-spec blocker and the explicit USER data-generation gate are cleared.
