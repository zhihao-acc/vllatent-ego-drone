# B3-CS I/O contract

This document summarizes the implemented CS1–CS3 boundary. Exact executable
schemas and constants live in `vllatent/sim/`; the active queue and gates live in
`plans/phase-b3-causal-ski-sim-latent-decoder.md`.

The retired Phase-A discrete AirSim/AerialVLN tuple and waypoint seam are not
current interfaces. The historical passive-video six-field token's canonical
definition remains at `vllatent.plan_tokens` until CS5 compatibility migration.

## 1. Forecast boundary

```text
history_latents:  (H=3, 196, 768)
future_command:   (T=8, 4)
dt_seconds:       (T=8,) separate
        -> predicted_latents: (T=8, 196, 768)
        -> decoder target:     (T=8, 4) = (cx, cy, log_h, p_visible)
```

Future latents, labels, masks, visibility, simulator state, and skier state are
targets/audit data only and must not enter model `forward`.

## 2. Requested camera command

Canonical storage is little-endian float64 SI. Model loading later casts command
and `dt` independently to float32 without normalization.

```text
requested_command.shape = (8, 4)
requested_command fields =
  [v_forward_m_s, v_right_m_s, v_down_m_s, yaw_rate_rad_s]
dt_seconds.shape = (8,); every value = 0.2 s
record_valid.shape = (8,); zero and pure-yaw rows are valid
```

Exactly nine constant programs exist, in frozen order:

| branch | nonzero command |
|---|---|
| `zero` | none |
| `yaw_plus`, `yaw_minus` | `yaw_rate = +/-pi/15 rad/s` |
| `forward_plus`, `forward_minus` | `v_forward = +/-1.0 m/s` |
| `lateral_plus`, `lateral_minus` | `v_right = +/-0.75 m/s` |
| `vertical_plus`, `vertical_minus` | `v_down = +/-0.50 m/s` |

The six-field passive-video token is neither shape-compatible nor
meaning-compatible with this record.

## 3. Frames, signs, and SE(3)

Semantic rig/body axes are FRD: `+forward`, `+right`, `+down`; positive yaw turns
forward toward right about `+down`. Blender camera axes are `+X` right, `+Y` up,
`-Z` optical forward. The frozen rotation is:

```text
R_cam_from_rig = [[0, 1, 0],
                  [0, 0,-1],
                  [-1,0, 0]]
R_rig_from_cam = R_cam_from_rig.T
t_cam_from_rig = t_rig_from_cam = [0,0,0] m
```

Transforms are row-major little-endian float64 matrices named
`T_target_from_source`. The trajectory stores all of these separately:

- `requested_command` and `dt_seconds`;
- authoritative `T_world_from_rig`;
- `requested_T_rig0_from_rig_t`;
- `achieved_T_rig0_from_rig_t`.

Requested and achieved transforms must never alias or overwrite each other.
Future frame `k` is captured after one command integration step. History ticks
`-2,-1,0` use the serialized initial camera under zero command.

The fixed camera contract is 224 x 224, 24 mm lens, 36 mm horizontal sensor,
zero shift, square pixels, clip range `[0.1,500] m`, and depth of field disabled.

Expected image signs relative to the matched zero sibling are frozen:

| command | expected effect |
|---|---|
| positive yaw or positive lateral/right | `cx` decreases |
| negative yaw or negative lateral/right | `cx` increases |
| positive forward | `log_h` increases |
| negative forward | `log_h` decreases |
| positive down | `cy` decreases |
| negative down | `cy` increases |

## 4. Root, sibling, and digest identity

`root_id` identifies the complete initial scene/camera/skier state and absolute-
tick continuation schedule. `split_group_id` is the same indivisible group for
the shared history and all nine siblings. No sibling or frame may cross a split.

The canonical skier digest includes deterministic root, ski/contact, phase,
pose, bone-local, and seed state. It explicitly rejects camera, branch, command,
record-valid, visibility, render, RGB/image, mask, and pixel fields. Thus every
sibling has the exact same skier digest at the same absolute tick.

Canonical serialization is versioned, NFC-normalized, C-order, little-endian,
finite, and hash-addressed with SHA-256. Arrays exposed by frozen records are
backed by immutable buffers.

## 5. Continuation and skier proof

Schedules use absolute integer ticks. Any active maneuver target/ramp begins no
later than history tick `-2`; hidden starts in future ticks `1..8` are invalid.
Non-steady futures require visible history cues, and terminal-state keys may not
map to different continuation laws or target parameters.

`vllatent.sim.skier`, `pose`, `rig`, `scene`, and their audit modules own the
versioned float64 slope-plane mechanics, ground-root/armature separation,
root-free animation, ski/contact realization, continuation audit, and CS3
canonical root envelope.

## 6. Labels

The target set is person/skinned body, clothing, helmet, and boots. Skis, poles,
detached equipment, terrain, and obstacles are excluded. Camera-independent
world triangles are rasterized on an unbounded integer pixel lattice at pixel
centers; separate full-amodal, in-crop amodal, and depth-tested visible results
produce the frozen label record.

Current implemented constants include a 224-square crop,
`p_visible` threshold `0.20`, occlusion-ratio threshold `0.80`, and padded center
range `[-0.25,1.25]`. Visibility is camera-relative and never enters root/skier
identity.

## 7. Import boundary and current gate

All `vllatent/sim/*.py` modules are PURE: stdlib + NumPy/PyYAML only, with no
`bpy`, torch, AirSim, wall-clock, UUID, secrets, or RNG imports. Blender code is
isolated in `scripts/blender/b3_cs3_bridge.py`.

CS1–CS3 are complete. CS4 is not authorized and also lacks an available complete
normative CS4+ specification because two delegated reports were never tracked.
Do not generate data until both blockers are resolved.
