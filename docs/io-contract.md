# I/O Contract — vllatent-ego-drone (Phase A)

> **What this is.** A *transcription* of the **LOCKED** I/O contract for in-repo reference. The
> authoritative source is the vault design doc
> `[[arch-design-2026-06-08-latent-pred]]` (§4 I/O contract · §6 data-audit spec · §9.3–9.7 seam
> choices) and `[[dev-decision-2026-06-07-latent-pred-pipeline]]` (phases / DoD). **This file does not
> re-derive or relitigate** the architecture — it pins the four seams + the loader tuple + the data
> foot-guns so code in steps 3–10 has a single in-repo reference. Phase-A DoD item (1).

## 0. Tensor I/O table (arch §4) — the seam definition

| Symbol | Shape / type | Frame / units | Source |
|---|---|---|---|
| `rgb_t` | `(224,224,3)` uint8 | camera-optical, **BGR from AirSim → convert to RGB** | AirSim render @ GT pose |
| `z_t` (obs latent) | `(196, 768)` fp16 | DINOv3 patch space | frozen encoder, **cached** |
| `history_latents` | `(H=3, 196, 768)` fp16 | — | cached latents |
| `lang_tokens` `L` | `(M, 768)` fp16 | text-embed space | frozen text tower, **cached/episode** |
| `action_id` (in) | scalar id `∈ {0..7}` | AerialVLN discrete | dataset `actions[t]` |
| `ẑ_{t+1..t+T}` | `(T=4, 196, 768)` fp16 | DINOv3 space | predictor output |
| trust readout | `{ p_j∈[0,1]^T, k*∈[0,T], σ∈ℝ_+ }` | — | trust head (1 pass) |
| waypoint (native) | `(4,)` = `(Δx,Δy,Δz,Δψ)` | **AirSim-NED body, yaw-only** | waypoint head |
| → remap | `(4,)` body-FLU delta | fly0 convention | remap layer (**Phase D**) |
| → handoff | `WaypointHandoff.G_world (3,)` ENU **+ Δψ** | **world ENU** | fly0 `frames.py` + odom (**Phase D**) |

> **Typed student seams (Phase A, `vllatent/schemas.py`).** The model-output rows above are frozen,
> shape/dtype-validated dataclasses (review H3) so an ablation is a config flag, not code surgery:
> `ẑ_{t+1..t+T}` → `PredictorOutput.predicted_latents` `(T,196,768)` fp16; trust readout →
> `TrustReadout {p_commit (T,) ∈ [0,1], k_star ∈ [0,T], sigma ≥ 0}`; waypoint (native) →
> `Waypoint.delta_4dof (4,)` f32, AirSim-NED body. The loader-input tuple is `StepSample` (§2). The
> teacher-side `OracleTarget` distillation seam is typed later (A5.9, after the A5.8 investigation).

---

## 1. The four locked seams

### (a) Action representation — discrete codebook `0..7` → per-step FiLM

- **Input is discrete.** AerialVLN is **discrete-native**: an 8-way action id (7 motion + STOP). The
  codebook (8×768) embeds the id → small MLP → per-step **FiLM** `(γ_t, β_t)` applied as
  `(1+γ_t)⊙h + β_t` inside each predictor block (ReL-NWM mechanism). No delta-derivation noise on the
  *input* (arch §9.3).
- **Action enum** — transcribed verbatim from
  `third_party/AirVLN/airsim_plugin/airsim_settings.py::_DefaultAirsimActions`:

  | id | name | id | name |
  |---|---|---|---|
  | 0 | `STOP` | 4 | `GO_UP` |
  | 1 | `MOVE_FORWARD` | 5 | `GO_DOWN` |
  | 2 | `TURN_LEFT` | 6 | `MOVE_LEFT` |
  | 3 | `TURN_RIGHT` | 7 | `MOVE_RIGHT` |

- **Step constants** — verbatim from `_DefaultAirsimActionSettings`:
  `FORWARD_STEP_SIZE=5`, `LEFT_RIGHT_STEP_SIZE=5`, `UP_DOWN_STEP_SIZE=2`, `TURN_ANGLE=15`,
  `TILT_ANGLE=15` (metres / metres / metres / degrees / degrees). Pitch/roll ≡ 0 ⇒ effective 4-DoF.
- **Output is continuous** (seam d): the *input* uses the discrete id; the *output target* uses the
  continuous delta derived from consecutive GT poses. The exact arithmetic (lateral move via yaw±90°,
  NED z-down sign) is transcribed into `vllatent/actions.py` (step 4) by reproducing
  `third_party/AirVLN/utils/env_utils.py::getPoseAfterMakeAction` — **reproduced, not re-designed**.

### (b) Language injection — frozen text tower (512→768) → cross-attention

- **Frozen** text encoder (default SigLIP/CLIP-ViT-B text tower, 512-d projected → 768; alt
  BGE-small/MiniLM for long instructions) → per-token embeddings `L ∈ ℝ^{M×768}` used as **prefix
  tokens** read via **cross-attention** in every predictor block ("Chain of World" / CoWVLA
  mechanism **only — NOT its Emu3-8.5 B base**). Cached once per episode (arch §9.4).
- **Division of labour:** language = global, slow-changing "what/where" → cross-attn; action =
  per-step "motion" → FiLM (seam a).

### (c) Uncertainty readout — deployed **single-pass** horizon head

- **Deployed student (the model that ships):** one trust/horizon head over the rollout features →
  per-step "still-trusted" probability `p_j` → soft expected horizon `k* = Σ_j Π_{i≤j} p_i` + a
  scalar `σ`. **One forward pass** (arch §9.6).
- **Offline teacher — Phase C, documented here, NOT built in Phase A:** K=5 predictor ensemble
  inter-member latent disagreement `u_j` **AND** an independent V-JEPA-2 surprise `s_j`; consensus
  horizon `k*_teacher = max j s.t. u_j<ε ∧ s_j<δ` (two orthogonal gates). Distilled into the student
  (Huber on `k*` + BCE on `p_j`), isotonic-calibrated; ECE/UCE + AUROC = the Phase-C GO/NO-GO.
- **Phase A builds none of the teacher / ensemble / V-JEPA-2** — it is documented for Phase C only.

### (d) Waypoint → EGO seam — continuous 4-DoF `(Δx,Δy,Δz,Δψ)` in AirSim-NED body

- **Waypoint head:** pool predicted patch tokens at the committed step `k*` (soft-weighted by `p_j`)
  → MLP `[768→512→256→4]` → `(Δx,Δy,Δz,Δψ)` with per-DoF `tanh × max_range`. Native convention =
  **AirSim-NED body, yaw-only** (arch §9.7).
- **Remap chain — DOCUMENTED ONLY; this remap is `NOT executed in Phase A`:**
  `native AirSim-NED body` → `fly0 body-FLU delta` → `fly0 geometry/frames.py` (`R_FLU_FROM_FRD`,
  `R_ENU_FROM_NED`) + `odom` → `world ENU + Δψ` → `WaypointHandoff` / `PoseStamped` → **frozen
  EGO-Planner** (z & yaw config-unfixed for this loop).
- In Phases A–C, `vllatent/frames.py` **re-derives** the remap and unit-tests it against fly0's
  `geometry/frames.py` **semantics** — but **fly0 is NEVER imported** (A–C are standalone). The live
  closed-loop seam and the `WaypointHandoff` yaw-field extension are **Phase D**.

---

## 2. Loader output tuple (arch §6 item 5)

Per step the cached-latent loader (`vllatent/data/loader.py`, step 10) emits the `StepSample`
(`vllatent/schemas.py`, step 3):

```
(z_t, history_latents, lang_tokens, action_id, z_next, delta_4dof, future_frame_rgb)
```

| field | shape / dtype | frame / notes |
|---|---|---|
| `z_t` | `(196, 768)` fp16 | DINOv3 patch tokens (cached), obs at step t |
| `history_latents` | `(H=3, 196, 768)` fp16 | cached latents `z_{t-2..t}`; padded+masked at episode start |
| `lang_tokens` | `(M, 768)` fp16 | frozen text-tower output; cached per episode |
| `action_id` | `int ∈ [0, 7]` | AerialVLN discrete `actions[t]` |
| `z_next` | `(196, 768)` fp16 | DINOv3 latent of the next obs = prediction **target** |
| `delta_4dof` | `(4,)` float32 | `(Δx,Δy,Δz,Δψ)` **AirSim-NED body, yaw-only**, derived from poses |
| `future_frame_rgb` | uint8 RGB (optional) | GT future frame for the Phase-C V-JEPA-2 verifier |

Confirms Phase-A DoD item (2): AerialVLN yields `(RGB obs, 4-DoF action/waypoint, next obs, language)`
tuples.

---

## 3. Frame & convention hazards (the load-bearing data foot-guns)

### Foot-gun #1 — orientation formats (audit + render) — CONFIRMED against real data (step 5b)

The two orientation fields use **different representations** — that is the real foot-gun:

| Field | Format | Note |
|---|---|---|
| AerialVLN `start_rotation` | **quaternion `[w, x, y, z]` — `w-FIRST`** | episode start orientation |
| `airsim.Quaternionr(x, y, z, w)` | quaternion `xyzw` | sim API |
| **Canonical internal order** | quaternion **`xyzw`** | `vllatent/frames.py` pins this |
| AerialVLN `reference_path` row | **Euler `[x, y, z, pitch, roll, yaw]` (radians, 6-wide)** | per-pose; pitch=roll≡0; **yaw = `row[5]`** — NOT a quaternion |

Reorder `start_rotation` (`w-FIRST`) → canonical `xyzw` **before any use**; read `reference_path`
orientation as Euler (**yaw = `row[5]`**) — do **not** mistake the 6-wide row for a 7-wide quaternion.
Index alignment (confirmed on real data): **`len(reference_path) == len(actions)`**; `reference_path[0]`
is the start pose and `actions[t]` drives `reference_path[t] → reference_path[t+1]` (the terminal STOP
has no stored next pose). The `fixtures/episodes/quaternion_trap.json` fixture is authored so a naïve
(no-reorder) read of `start_rotation` yields a wrong yaw — the audit **flags it loudly**
(`naive_would_mismatch`).

### Foot-gun #2 — `BGR` → RGB at the render→encode boundary

AirSim `Scene` images are **`BGR`** (AirVLN leaves the `cvtColor` call commented out); DINOv3 expects
**RGB**. Convert at the render→encode boundary (`cv2.cvtColor(..., BGR2RGB)`) and **record the BGR→RGB
flag in the cache manifest**. Wrong channel order silently poisons every cached latent.

### Other pinned conventions

- **NED z-down sign:** `GO_UP = −z`, `GO_DOWN = +z`.
- **4-DoF only:** pitch = roll ≡ 0 ⇒ `(x, y, z, yaw)`; lateral moves (`MOVE_LEFT`/`MOVE_RIGHT`) are
  **body-relative** (yaw±90°), not world-axis.
- **AirSim msgpack-RPC is single-threaded** → wrap **every** `client.X()` call in a `threading.Lock`
  (render tier, step 8). tornado IOLoop is not re-entrant.
- **action ↔ reference_path index alignment:** `actions[t]` is index-aligned with the pose pair
  `reference_path[t] → reference_path[t+1]`. **Assert it; do not assume** (audit step 5).
- **Scene-dependent depth (Phase C only):** scenes 1 & 7 = `DepthVis`; others = `DepthPerspective`,
  clip `[0,100]/100`. The encoder is **RGB-only**, so this bites only if depth is cached for Phase C.
- **No frame-leak:** the NED→FLU→ENU remap is **documented-only** in A–C; do **NOT** import fly0 or
  execute the remap in Phase A.

---

## 4. Licenses (relevant at publication / productization)

- **AerialVLN dataset:** `CC BY-NC-SA 4.0` — **non-commercial**, share-alike. Recorded again in
  `DEV_LOG.md` at step 6 when the real slice is fetched.
- **DINOv3 weights:** Meta custom non-OSI license (commercial OK with attribution; ITAR/military
  prohibited) — fine for academic sim work.
- **V-JEPA-2** (Phase-C verifier): Apache-2.0.

---

## 5. Source of truth

Vault `[[arch-design-2026-06-08-latent-pred]]` is **authoritative** for the locked architecture and
I/O contract; `[[dev-decision-2026-06-07-latent-pred-pipeline]]` for phases/DoD. This repo is
authoritative for **code state**. This file transcribes the contract; any change to the contract is a
new vault decision entry, not an edit here.
