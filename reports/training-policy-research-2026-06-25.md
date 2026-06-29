# Training Policy: SkyJEPA PI Prober Integration + Vision-Conditioned Dynamics

**Date:** 2026-06-25 (revised)
**Author:** Research synthesis from 3 parallel deep-research agents + revision session
**Status:** REVISED — single recommended training policy. Ready for implementation.
**Scope:** Handoff task 2 — "training policy research" for vllatent-ego-drone Phase B+C.

---

## Executive Summary

This document defines the **single recommended training policy** for vllatent-ego-drone,
incorporating three locked decisions:

1. **PI Prober replaces the MLP waypoint head.** The SkyJEPA PI Prober architecture —
   residual corrections on a hardcoded kinematic prior + SO(3) exponential map + Euler
   integration — is the action decoder. It produces physically feasible 4-DoF trajectories
   by design, not by learning to approximate physics.

2. **Option B — Image as conditional input.** The latent space mismatch between DINOv3
   (196×768 image patch tokens) and SkyJEPA (24D GRU hidden dynamics state) is resolved by
   keeping vision and dynamics in their native spaces. DINOv3 image latents are compressed
   to a conditioning signal via a learned **visual bottleneck** (cross-attention stem →
   16 tokens → linear → conditioning vector). This vector modulates the dynamics predictor
   via **AdaLN** (adaptive LayerNorm), telling it "where the target is moving" without
   forcing image features into a dynamics latent space they were never designed for.
   The PI Prober operates on the dynamics latent, not on image features.

3. **Privileged-teacher distillation** (Loquercio et al. pattern). A two-phase training
   policy: Phase B trains the visual latent predictor + a vision-conditioned PI Prober
   (supervised, MegaSaM GT actions). Phase C introduces a privileged dynamics teacher
   (SkyJEPA trained on real drone odom/IMU) and distills its trajectory expertise into the
   vision-conditioned student. This is the same paradigm that achieved zero-shot sim-to-real
   at 60 km/h in aggressive drone flight (Loquercio et al., Science Robotics 2021).

**Trust mechanism is REMOVED.** No trust heads, commitment horizons, or V-JEPA-2 surprise
gates. The architecture is simpler than what the prior report proposed.

---

## 1. SkyJEPA — Architecture Reference

### 1.1 Paper Identity

| Field | Value |
|---|---|
| Title | SkyJEPA: Learning Long-Horizon World Models for Zero-Shot Sim-to-Real Control of Quadrotors |
| ArXiv | 2606.23444 (v2, 2026-06-23) |
| Authors | Pratyaksh Rao (UC Berkeley), Wancong Zhang (NYU), Randall Balestriero (Brown), Yann LeCun (NYU), Giuseppe Loianno (UC Berkeley) |
| License | Paper CC BY 4.0, Code Apache 2.0 |
| Open-source | GitHub `arplaboratory/SkyJEPA` — repo exists (Apache-2.0, public) but **placeholder only** as of 2026-06-25: README + LICENSE + GIFs, no code/weights. "Code release: will be released soon." Assume available for architecture pattern adoption; re-implement from paper if weights delayed. |

### 1.2 Architecture (~99K params total)

```
State history (H=10 @ 20Hz)          Action history (H=10 @ 20Hz)
  x_t = [p(3), v(3), R(9), ω(3)]      a_t = [f0, f1, f2, f3] (motor forces, N)
  = 18-dim per step                     = 4-dim per step
         │                                    │
    [TCN channels 8,8,16]              [TCN channels 4,4,8]
         │                                    │
         └──────────┬─────────────────────────┘
                    │
              [GRU hidden=24]  ──unroll T=20 steps (1.0s)──►  ŝ_{t+1..t+T}
                    │
        ┌───────────┴───────────┐
        │   Stage 2 (frozen)    │
        │                       │
   [PI Prober MLP]              │
        │                       │
   Δv̇ (3) residual accel   K (3×4) angular accel matrix
        │                       │
   ┌────┴───────────────────────┘
   │  Nominal kinematic model (HARDCODED):
   │    v̇ = (Σf_i/m)·R·e₃ − g + Δv̇        (translational)
   │    Δτ = K · a                           (angular)
   │    p_{t+1} = p_t + v_t·dt              (Euler integration)
   │    v_{t+1} = v_t + v̇_t·dt
   │    R_{t+1} = R_t · exp([ω_t]_× · dt)  (SO(3) exp map)
   │    ω_{t+1} = ω_t + Δτ_t·dt
   └──► physical state trajectory
```

**Key design decisions:**
- Attitude = full rotation matrix R(9), not quaternion — avoids discontinuities
- Prober predicts RESIDUALS on physics, not raw state — constrains drift
- SO(3) exponential map (Rodrigues' formula) is hardcoded, not learned
- SIGReg anti-collapse (lambda=0.02) instead of VICReg — matches full distribution
  via Cramer-Wold theorem + Epps-Pulley test on M=17 random 1D projections
- Two-stage training: (1) latent dynamics (L_pred MSE + L_SIGReg), (2) PI prober
  (state recovery MSE, stop-gradient on latents)

### 1.3 Results

| Metric | SkyJEPA (JEPA+PI Prober) | Predictive baseline |
|---|---|---|
| Open-loop position RMSE (60 steps) | **1.43 m** | 8.80 m (6.1x worse) |
| Open-loop attitude error | **4.71 deg** | 53.4 deg (11.3x worse) |
| Closed-loop circle tracking (real) | **0.24 m** | 0.39 m |
| Closed-loop lemniscate (real, aggressive) | **0.45 m** | 0.61 m |
| Orin NX latency (512 MPPI samples, H=15) | **~10 ms** | — |

Zero-shot sim-to-real. Domain randomization: mass ±50%, inertia ±30%, motor/drag/thrust
coefficients varied across 500 configurations, 20K trajectories.

### 1.4 What We Take from SkyJEPA

We adopt the **PI Prober architecture pattern** — not SkyJEPA's latent encoder. The PI
Prober is a function `latent → (Δv̇, K) → physics integration → trajectory`. It is
decoupled from how the latent is produced. In SkyJEPA, a TCN+GRU produces the latent from
18D state history. In our system, a transformer predictor produces the latent from DINOv3
image tokens + action history. The PI Prober doesn't care — it receives a fixed-size latent
vector and outputs residual corrections on the kinematic prior.

---

## 2. The Latent Space Mismatch — Analysis and Resolution

### 2.1 The Problem

| Space | Producer | Shape | Content |
|---|---|---|---|
| SkyJEPA dynamics latent | TCN+GRU on 18D state | (24,) per step | Quadrotor dynamics: position, velocity, SO(3), angular velocity |
| DINOv3 image latent | Frozen ViT-B/16 on RGB 224² | (196, 768) per frame | Visual scene: spatial structure, object identity, appearance |

The PI Prober expects a dynamics-like latent (compact, low-dimensional, encoding physical
state). DINOv3 outputs a high-dimensional visual representation (196 spatial tokens × 768
channels = 150K dimensions). These are fundamentally different kinds of information.

### 2.2 Decision: Image as Conditional Input (Option B)

**LOCKED.** Vision conditions the dynamics model — it does not replace it or merge into it.
DINOv3 image latents are compressed to a conditioning signal via a learned visual bottleneck
(cross-attention stem). This conditioning vector modulates the predictor via AdaLN. The
PI Prober operates on the predictor's dynamics-compatible latent space, not on raw image
features.

This is the dominant pattern in the literature (HPT, pi0, V-JEPA-2-AC, DINO-WM, TD-MPC2)
and is the exact paradigm that achieved 60 km/h zero-shot sim-to-real in Loquercio et al.
(2021). Specific architectural mechanism in §3.

**Why not the alternatives:**
- **Pure SkyJEPA (Option A):** Abandoning vision removes the core thesis (visual
  anticipation). Our MegaSaM VO data provides only 4-DoF deltas, not the 18D state +
  4D motor commands SkyJEPA requires. Not viable.
- **Latent alignment (Option C):** Forcing 150K-dim image features into a 24-dim dynamics
  space via alignment losses discards spatial information. No precedent for this kind of
  cross-scale alignment in robotics. Fragile (collapse, mode dropping).

### 2.3 Literature Support for Option B

The "vision conditions dynamics" pattern is well-established:

| System | Vision → Dynamics Mechanism | Result |
|---|---|---|
| **Loquercio et al. (2021)** | Privileged teacher (full state) → sensorimotor student (camera only). Student learns to map visual features to the same action space. | 60 km/h zero-shot sim-to-real. Science Robotics. |
| **pi0 (2024)** | SigLIP vision tokens + proprioception → blockwise causal transformer → flow matching actions. Vision and state stay in separate token spaces, fused by attention. | State of the art in robotic manipulation. |
| **HPT (NeurIPS 2024)** | Cross-attention stems compress vision (ResNet) and state (MLP) each to 16 tokens → shared GPT trunk. Equal representation size forces balanced attention. | Cross-embodiment generalization. |
| **V-JEPA-2-AC (2025)** | Frozen ViT-g vision + 7D proprioception → chronological interleave with 3D-RoPE → block-causal transformer. | 62 hrs robot data suffices. |
| **TD-MPC2 (ICLR 2024)** | CNN vision + MLP state → task embedding concat to all components. State modulates vision. | Multi-task, multi-domain. |
| **"What Drives Success in JEPA WMs" (2512.24497)** | Ablation: AdaLN conditioning of action/state > concatenation > sequence conditioning. | Systematic ablation on JEPA world models. |
| **DINO-WM (ICML 2025)** | Frozen DINOv2 + action broadcast to all 256 patches. Vision provides the spatial map; action tells where to move. | 0.98 success rate, maze navigation. |

**The consensus mechanism is AdaLN** — the action/state embedding modulates every
transformer layer via adaptive LayerNorm: `y = γ(s) · LayerNorm(x) + β(s)`. This is the
FiLM pattern applied at LayerNorm (more numerically stable). Ablation-proven superior to
token concatenation and sequence conditioning.

---

## 3. Recommended Architecture

### 3.1 Overview

The diagram below shows the **fully assembled architecture** (Stage 3+). Components marked
with their training stage: [S1] = Stage 1 only, [S2+] = added at Stage 2, [all] = all stages.

```
                ┌───────────────────────────────────────────────────────────┐
                │                PERCEPTION [all: frozen]                   │
                │                                                           │
  RGB 224² ──►  │  [DINOv3 ViT-B/16, FROZEN, CACHED] → z_vis (196×768)    │
                │         │                                                 │
                │    [Cross-Attn Stem: 16 learned queries attend to  [S2+] │
                │     196 DINOv3 tokens → 16 visual summary tokens]        │
                │         │                                                 │
                │    [MLP pool: 16×768 → mean → MLP(768→256)]        [S2+] │
                │         │                                                 │
                │         ▼                                                 │
                │    c_vis (256,) — visual conditioning vector              │
                └─────────────┬─────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────────────────────────────────────┐
                │                      PREDICTOR                            │
                │                                                           │
                │  Input: history latents z_{t-2..t} (3×196×768)            │
                │       + action history a_{t-2..t-1} (2×4)                 │
                │                                                           │
                │  [Block-Causal ViT, depth from PredictorConfig]           │
                │   + AdaLN conditioning from c_vis (§2.3 ablation winner)  │
                │   + AdaLN dt conditioning (handle mixed FPS)              │
                │   + block-causal mask (H=3 history, T=4 horizon)          │
                │                                                           │
                │  Output: ẑ_{t+1..t+4} (4×196×768) — predicted latents    │
                │       +  h_{t+1..t+4} (4×768) — predictor readout tokens │
                └─────────────┬─────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────────────────────────────────────┐
                │                   PI PROBER (action decoder)              │
                │                                                           │
                │  Input: h_t (768,) — predictor readout token               │
                │       + c_vis (256,) — visual conditioning vector         │
                │       + current state estimate (v_t, R_t from odom/VO)    │
                │                                                           │
                │  [MLP: 768 + 256 + state_dim → 256 → 128]                │
                │       │                              │                    │
                │  Δv̇ (3) residual accel          K (3×4) angular accel    │
                │       │                              │                    │
                │  ┌────┴──────────────────────────────┘                    │
                │  │  Kinematic prior (HARDCODED):                          │
                │  │    v̇ = Δv̇ + gravity_term                              │
                │  │    p_{t+1} = p_t + v_t·dt + 0.5·v̇·dt²                 │
                │  │    R_{t+1} = R_t · exp([ω_update]_× · dt)             │
                │  │                                                        │
                │  │  → 4-DoF output: (dx, dy, dz, dyaw)                   │
                │  └──► physically feasible waypoint                        │
                └───────────────────────────────────────────────────────────┘
```

### 3.2 Component Details

**Dimension definitions:**
- **D_pred = EMBED_DIM = 768.** The predictor readout token has the same width as the
  transformer's residual stream. No projection needed.
- **D_cond = 256.** The visual conditioning vector. Chosen as a bottleneck: large enough to
  carry spatial information (target location, terrain shape), small enough to not dominate
  the AdaLN modulation.

**Visual Bottleneck (Cross-Attention Stem).** 16 learned query tokens attend to the 196
DINOv3 patch tokens via multi-head cross-attention. This compresses 196×768 = 150K dims
to 16×768 = 12K dims. A mean-pool + MLP further compresses to D_cond = 256.
This is the HPT pattern — cross-attention stems with a fixed number of queries produce a
fixed-size representation regardless of vision encoder output size.

Why 16 queries: HPT ablation showed 16 is the sweet spot — fewer loses spatial detail,
more adds compute without quality gain. The queries learn to attend to task-relevant visual
features (target location, terrain, obstacles).

**Predictor (Block-Causal ViT).** Unchanged from current design except for one addition:
a **learned readout token** is prepended to each horizon step's token sequence. This is a
standard ViT CLS token — a single learned embedding (768,) that attends to all 196 patch
tokens within its block-causal step and accumulates a summary representation h_t.

The predictor outputs:
- ẑ_{t+1..t+T} (4×196×768) — predicted patch latents (for L_latent)
- h_{t+1..t+4} (4×768) — readout tokens per horizon step (for PI Prober)

The visual conditioning vector c_vis enters via AdaLN — each transformer layer's LayerNorm
is modulated by γ(c_vis), β(c_vis). This replaces the current FiLM action conditioning.

**Implementation note:** The readout token requires modifying `PredictorConfig` (add
`use_readout_token: bool = True`), `PredictorOutput` in schemas.py (add
`readout_tokens: np.ndarray  # (T, D_pred) fp16`), and the predictor forward pass. This
is a new component — it does not exist in the current codebase. The readout token is
initialized as a learned parameter (nn.Parameter, shape (1, 768), normal init σ=0.02) and
prepended to each block-causal step's token sequence before the transformer layers.

**PI Prober (Action Decoder).** Replaces the current `D→256→128→4` MLP. Takes the
predictor's CLS token (h_t), the visual conditioning vector (c_vis), and the current state
estimate (velocity, orientation from VO/odom). Outputs residual corrections (Δv̇, K) on
a hardcoded kinematic model. The kinematic model integrates these into a physically
feasible 4-DoF waypoint.

Key adaptations from SkyJEPA's PI Prober:
- **Input:** SkyJEPA uses GRU hidden (24D). We use predictor readout token (768) + visual
  conditioning (256) + state estimate. Total input dim is larger but the MLP handles it.
- **State estimate:** In Phase B, we use MegaSaM VO velocity estimates (noisy but available).
  In Phase D, VINS-Mono provides clean state. The PI Prober is robust to state noise because
  it predicts RESIDUALS — errors in the state estimate affect the kinematic prior, but the
  learned residuals compensate.
- **Action space:** SkyJEPA outputs 18D full state predictions. We output 4-DoF (dx,dy,dz,
  dyaw). The kinematic integration is simplified: no individual motor forces, just aggregate
  thrust direction + yaw rate. This matches our waypoint head contract.
- **SO(3) integration:** Retained from SkyJEPA for yaw update. dyaw is extracted from
  R_{t+1} = R_t · exp([0, 0, Δyaw]_× · dt) via atan2.

### 3.3 Simplified PI Prober for Phase B

For Phase B (no real-time odom), we use a **simplified PI Prober** that doesn't require
live state estimation:

```
Input: h_t (768,) — predictor readout token (D_pred = EMBED_DIM = 768)
     + c_vis (256,) — visual conditioning vector (D_cond = 256)

MLP: (768 + 256) → 256 → 128

Output head 1: Δv (3) — residual velocity correction
Output head 2: Δyaw (1) — residual yaw rate correction

Kinematic prior (SIMPLIFIED for 4-DoF):
  dp = v_prev · dt + Δv · dt    (position delta)
  dyaw = yaw_rate_prev · dt + Δyaw · dt

  4-DoF output: (dp_x, dp_y, dp_z, dyaw)
```

**Velocity prior source (v_prev / yaw_rate_prev):**

- **Training:** v_prev = delta_4dof[t-1] / dt (the previous GT waypoint delta, equivalent
  to differencing two consecutive GT positions). This is available from `StepSample` and
  `SportsTarget` data. Concretely, `v_prev = delta_4dof[t-1][:3] / dt_seconds` and
  `yaw_rate_prev = delta_4dof[t-1][3] / dt_seconds`. For `t=0` (no predecessor), v_prev
  and yaw_rate_prev are zero vectors (stationary prior).

- **Inference:** The model's own previous output serves as v_prev (autoregressive). The
  first step uses zero velocity (stationary start). This creates a mild autoregressive
  dependency, but drift is bounded because the residual Δv compensates — if v_prev drifts,
  the network learns to predict larger Δv corrections. SkyJEPA's PI Prober operates the
  same way at deployment (MPPI rolls out its own predictions autoregressively).

- **dt source:** `StepSample.dt_seconds` when available; falls back to `1.0 / target_fps`
  (0.2s at 5 fps) when None. The kinematic prior uses dt directly.

This is the PI Prober concept applied to our simpler action space. The residual-on-prior
structure is preserved — the network predicts corrections to a "continue current motion"
baseline (kinematic prior = constant velocity assumption). During supervised training with
GT v_prev, the residual structure is mathematically equivalent to a reparameterized MLP
(the network can learn `Δv = (target - v_prev·dt) / dt`). The architectural benefit is
an inductive bias: at inference with autoregressive v_prev, the constant-velocity prior
provides a reasonable default when visual input is ambiguous, and the structure composes
directly with MPPI planning in Phase D. The full SO(3) PI Prober with rotation matrix and
motor forces is Phase D when we have real drone state from VINS.

### 3.4 Why Not Direct MLP → 4-DoF?

The current plan has a simple MLP waypoint head: `D→256→128→4`. This works but has no
physics constraint. The PI Prober's advantage:

1. **Bounded drift.** Residual corrections on a kinematic prior cannot produce arbitrarily
   large jumps. The kinematic prior constrains the output to physically plausible deltas.
2. **Better extrapolation.** When the visual input is ambiguous, the kinematic prior
   (constant velocity) is a reasonable default. A pure MLP might output nonsense.
3. **Composability with MPPI.** In Phase D, the PI Prober integrates naturally with MPPI
   planning — sample action sequences, roll out the PI Prober's kinematic model, evaluate
   trajectories. This is how SkyJEPA uses it at deployment.

---

## 4. Training Policy — The Five-Stage Pipeline

### 4.1 Overview

```
Stage 0          Stage 1              Stage 2            Stage 3           Stage 4
┌──────┐   ┌──────────────┐   ┌──────────────────┐   ┌──────────────┐   ┌──────────┐
│Freeze│   │Video pre-trn │   │Warm-up: action   │   │Joint fine-   │   │Privileged│
│DINOv3│──►│predictor only│──►│blocks + PI Prober│──►│tune all      │──►│teacher   │
│encode│   │L_latent      │   │(predictor frozen)│   │L_lat + L_wp  │   │distill.  │
│cache │   │(no actions)  │   │L_latent + L_wp   │   │              │   │(Phase C) │
└──────┘   └──────────────┘   └──────────────────┘   └──────────────┘   └──────────┘
```

### 4.2 Stage 0 — Freeze and Cache (DONE)

Encode all training frames with frozen DINOv3 ViT-B/16. Store as (N, 196, 768) fp16 on
disk. This is the existing pipeline (B1.7 pilot, B1.10c E2E verified).

### 4.3 Stage 1 — Video Pre-Training (Predictor Only)

**What trains:** Predictor transformer only. Visual bottleneck is NOT used (no conditioning
needed — this is pure next-latent prediction).

**Loss:** L_latent = Smooth L1 (β=0.1) between predicted latents ẑ_{t+1..t+T} and ground
truth z_{t+1..t+T} from the cache.

**Data:** ALL available skiing FPV video. No action labels needed. This includes frames
that fail quality gating (fine for visual prediction, just not for VO-based action labels).

**Why:** Overwhelmingly supported by the literature (+4–68% across 8 papers, §3.1 in prior
report). The gap is largest when downstream labeled data is scarce — which is our situation.

**Duration:** Until val L_latent plateaus. Expected: 50-100 epochs on pilot data.

**Literature precedent:**
- DINO-world: +12.5pp from video pre-training
- DINO-WM: frozen DINOv2 + causal ViT predictor + L2 loss → 0.98 success rate
- APV (ICML 2022): +27.5pp from action-free pre-training
- VideoVLA (NeurIPS 2025): +67.8pp (largest gap, small downstream data)

### 4.4 Stage 2 — Warm-Up (Action Blocks + PI Prober, Predictor Frozen)

**What trains:** Visual bottleneck (cross-attention stem + MLP), PI Prober, AdaLN
conditioning layers. The pre-trained predictor is FROZEN.

**Why freeze the predictor:** CoLA-World finding — without this warm-up, action conditioning
layers fight the pre-trained representations, causing "VQ codebook collapse." The action
blocks are initialized as identity with layer_scale=0 (the model starts as if actions don't
exist, gradually learning to use them).

**Loss:**
```
L = λ_latent · L_latent(ẑ, z)  +  λ_wp · L_wp(ŵ, w_gt)

where:
  L_latent = Smooth L1 (β=0.1) on predicted latents (unchanged from Stage 1)
  L_wp     = MSE on 4-DoF waypoint (PI Prober output vs MegaSaM GT delta)
  λ_latent = 1.0, λ_wp = 0.1 (ramp to 1.0 over warm-up)
```

**Data:** YouTube pilot dataset with MegaSaM ego-motion deltas (action labels required).

**Duration:** ~8K steps (CoLA-World recommendation). Monitor: L_wp should decrease while
L_latent stays approximately constant (confirming the predictor isn't being corrupted).

**Key implementation detail:** The PI Prober's kinematic prior uses the previous GT
`delta_4dof[t-1] / dt` as `v_prev` during training (see §3.3 for full specification).
For `t=0` steps, v_prev = 0 (stationary prior).

### 4.5 Stage 3 — Joint Fine-Tune (Everything)

**What trains:** Predictor + visual bottleneck + PI Prober + AdaLN layers — everything
except the frozen DINOv3 encoder.

**Loss:**
```
L = λ_latent · L_latent(ẑ, z)  +  λ_wp · L_wp(ŵ, w_gt)

λ_latent = 1.0, λ_wp = 1.0 (equal weight after warm-up)
```

**Data:** YouTube pilot dataset initially. CosFly-Track trajectory JSONs are available but
integration is pending (B1.8 descoped RGB, trajectory-only adapter done but world-frame vs
body-frame delta conversion and scale mismatch are unresolved). Add CosFly when resolved.

**Duration:** Until val metrics plateau. Monitor:
- L_latent on held-out clips (visual prediction quality)
- L_wp on held-out clips (action prediction quality)
- Overfit-tiny-batch first (training-playbook SOP)

**Key finding from DINO-world ablation:** fine-tuning the pre-trained predictor (59.4%)
outperforms keeping it frozen (49.4%). But: DINO-WM found that adding a decoder
reconstruction loss HURTS (0.80 vs 0.92). L_latent should be the sole predictor loss
for the latent stream — no pixel-space decoder.

### 4.6 Stage 4 — Privileged Teacher Distillation (Phase C)

This stage runs in **Phase C** when we have drone flight data (real odom/IMU from VINS).

**The Loquercio Pattern (Science Robotics 2021):**
1. Train a **privileged teacher** with full state access: SkyJEPA trained on real drone
   odom/IMU data (18D state + 4D motor commands). This teacher has perfect dynamics
   knowledge but no vision.
2. The privileged teacher produces trajectory predictions that are physically optimal.
3. **Distill** the teacher's trajectory predictions into the vision-conditioned student
   (our model from Stages 1-3). The student learns to produce teacher-quality trajectories
   from visual input alone.

**Distillation loss:**
```
L_kd = MSE(student_trajectory, teacher_trajectory)  [stop-gradient on teacher]
```

This is behavioral cloning from a privileged expert — the same mechanism that achieved
zero-shot sim-to-real at 60 km/h for aggressive aerial maneuvers (Loquercio et al., 2021).

**Why this works:** The privileged teacher (SkyJEPA) knows the optimal trajectory from
dynamics alone. The student (our model) must learn to infer dynamics-quality trajectories
from visual input. The distillation bridges the gap between "what the drone should do
physically" and "what the camera sees."

**SkyJEPA's own two-stage training applies HERE:**
1. Train SkyJEPA's latent dynamics (L_pred + L_SIGReg) on drone flight logs
2. Train SkyJEPA's PI Prober (state recovery MSE, stop-gradient on latent encoder)
3. Freeze the trained SkyJEPA as teacher
4. Distill into our student via L_kd

**Data requirement for Phase C:** Simulation-generated trajectories are the **primary and
necessary** data source. Phase C cannot depend on real flight data from Phase D (circular
dependency — Phase D needs a Phase C model). SkyJEPA's paper used 20K trajectories
(500 configs × 40 trajs, 10s each, 20Hz) in simulation with domain randomization (mass
±50%, inertia ±30%, motor time constants [0.01, 0.1]s, drag [0.1, 0.5], thrust/torque
±50%). We generate equivalent data in IsaacLab or gym-pybullet-drones with similar domain
randomization. Training: Adam, LR warmup 0→5e-3 over 4K steps, cosine decay to 1e-4,
batch size 2048, 50 epochs (Stage 1). Real flight logs (Phase D) are a refinement
iteration, not the Phase C entry point.

**SkyJEPA PI Prober internals (from paper — code unreleased):** The paper does NOT specify
the PI Prober MLP architecture (hidden dims, activations, number of layers). The output is
15 scalars: Δv̇ ∈ ℝ³ (residual translational acceleration) + K ∈ ℝ³ˣ⁴ (angular acceleration
matrix, where Δτ = K @ a maps 4D motor forces to 3D torque). The state representation is
18D: [p(3), v(3), r_x(3), r_y(3), r_z(3), ω(3)] — rotation matrix stored as three column
vectors. Since we are re-implementing from the paper description, our MLP design is free.

### 4.7 Optional Stage 5 — GRPO on Trajectory Quality (Phase C/D)

After distillation, optional GRPO fine-tuning in imagination:

1. From a training state, roll out the model G=8-16 times (different action samples)
2. Score each trajectory with reward function
3. Compute within-group relative advantages (VD-GRPO — subtract mean only, no variance
   division, per Plan-R1 2505.17659)
4. Update policy (predictor + heads) with policy gradient

**Reward signals (from GC-VAT, NeurIPS 2025):**
```
r_total = w_prog · r_progress(Δ_distance_to_target)
        + w_fov · r_centering(GC-VAT projection-aware, NOT Euclidean)
        - w_smooth · r_jerk(||a_t - a_{t-1}||²)
        - w_effort · r_effort(||a_t||²)
```

**Phased curriculum** (arXiv 2603.05113):
- Phase 1: r_progress + r_centering only (learn the task)
- Phase 2: add r_jerk (learn smooth behavior)
- Phase 3: velocity adaptation (fast when far, slow when near)

This is cheap: SRPO achieved 99.2% from 48.9% in only 200 RL steps.

---

## 5. Distillation Literature — Extracted Training Recipes

### 5.1 Loquercio et al. (2021) — "Learning High-Speed Flight in the Wild"

The most directly relevant precedent for our pipeline.

| Stage | What | Loss | Frozen | Data |
|---|---|---|---|---|
| 1. Privileged teacher | CNN policy with full state access (depth, velocity, obstacle positions) | PPO (RL reward: progress + collision avoidance) | — | Sim (varied envs, domain randomization) |
| 2. Sensorimotor student | ResNet-18 student from monocular camera only | DAgger-style behavioral cloning from teacher rollouts | Teacher | Sim → real (sim2real transfer) |

**Key insight:** The teacher never sees camera images. The student never sees privileged
state. Distillation bridges the modality gap. This achieved 60 km/h zero-shot sim-to-real
in aggressive autonomous flight through forests and buildings.

**Relevance to us:** Replace "sim with privileged state" with "SkyJEPA with odom/IMU".
Replace "ResNet-18 student" with "DINOv3 + predictor + PI Prober". Same paradigm.

### 5.2 SkyJEPA (2026) — Two-Stage Latent Dynamics + PI Prober

| Stage | What trains | Loss | Frozen | Data |
|---|---|---|---|---|
| 1. Latent dynamics | TCN encoders + GRU | L_pred (MSE on predicted state) + L_SIGReg (λ=0.02) | — | 20K trajectories (500 configs × 40 trajs) |
| 2. PI Prober | PI Prober MLP only | State recovery MSE | TCN + GRU (stop-gradient) | Same data |

**Key insight:** Stop-gradient on the latent encoder during PI Prober training. This
prevents the PI Prober from corrupting the latent space. The latent space is locked after
Stage 1; the PI Prober learns to decode from it without modifying it.

**Relevance to us:** Our Stage 2 warm-up applies the same principle — freeze the predictor
while the PI Prober learns to decode from its latent space.

### 5.3 DINO-world (2025) — Staged Video Pre-Training

| Stage | What trains | Loss | Frozen | Data |
|---|---|---|---|---|
| 1. Video pre-train | Predictor | L_latent (Smooth L1) | DINOv3 encoder | Action-free video |
| 2. Warm-up | Action blocks only | L_latent + L_wp | DINOv3 + predictor | Action-paired data |
| 3. Joint fine-tune | Predictor + action blocks + heads | L_latent + L_wp | DINOv3 encoder | Action-paired data |

**Key ablation:** Video pre-training + warm-up + joint = 59.4%. Without video pre-train = 46.9%.
Without warm-up = collapse. Without joint fine-tune (keep predictor frozen) = 49.4%.

**Relevance to us:** Directly adopted as our Stages 1-3.

### 5.4 CoLA-World (2025) — Warm-Up Prevents Collapse

**Critical finding:** When action conditioning layers are added to a pre-trained predictor
without a warm-up period, "VQ codebook collapse" occurs — the action blocks overwrite the
pre-trained representations. Solution: freeze predictor for ~8K steps while action blocks
align. Initialize action blocks as identity with layer_scale=0.

**Relevance to us:** Adopted directly in Stage 2.

### 5.5 DAgger (2011) and Variants

**DAgger** (Dataset Aggregation) addresses distribution shift in behavioral cloning:
1. Train student on teacher demonstrations
2. Run student in environment, collect its own states
3. Query teacher for actions at student's states (not teacher's states)
4. Aggregate new data into training set, retrain

**Relevance to us:** Phase C distillation could use DAgger-style online aggregation if
the student's trajectory distribution diverges from the teacher's. In practice, offline
behavioral cloning (MSE on teacher trajectories) works well when the teacher's coverage
is broad (which SkyJEPA with domain randomization provides).

### 5.6 TrackVLA (CoRL 2025)

| Component | Detail |
|---|---|
| Architecture | π0.5-based VLA with temporal compression + spatial grounding |
| Training | 1.7M visual tracking samples, behavioral cloning |
| Action space | Velocity + yaw rate (continuous), 10 Hz |
| Relevance | Purpose-built for drone visual tracking — closest existing system to our task |

**Status:** Code released at github.com/wsakobe/TrackVLA but weights may be restricted.
If available in Phase C, TrackVLA provides an alternative teacher signal (waypoint soft
labels from a tracking-specific VLA, supplementing SkyJEPA's dynamics-only expertise).

---

## 6. Encoder Decision

### 6.1 Current Lock: DINOv3 ViT-B/16 (D=768)

The encoder is locked at ViT-B/16 (D=768) per CLAUDE.md and the Phase B plan. The B1.11
Orin NX benchmark gate determines whether to stay with ViT-B/16 or downgrade to ViT-S/16.

### 6.2 Upgrade Path (Future)

The training policy research previously recommended upgrading to ViT-L/16 (D=1024) for
better dense features. This remains a valid optimization but is NOT part of the current
training policy. Reasons:

1. B1.11 gate hasn't fired yet — we don't know if ViT-B/16 fits Orin NX
2. Upgrading D changes all shape fixtures, predictor config, and cached latents
3. The priority is to prove the pipeline works at D=768 first
4. ViT-B/16 is ablation-proven sufficient (DINO-WM achieved 0.98 success with DINOv2
   ViT-S/14, which is smaller)

**Decision: Keep D=768 for Phase B. Evaluate D=1024 upgrade after the pipeline is proven.**

---

## 7. 3DGS for Simulation and Validation

### 7.1 Timing

3DGS simulation is **NOT part of Phases B or C training**. It is for:
- **Phase C/D:** GRPO reward computation (roll out predictor in 3DGS, score trajectories)
- **Phase D:** Closed-loop validation before real deployment
- **Data augmentation:** Novel viewpoint rendering (low priority)

### 7.2 Proven Systems

| System | ArXiv | Key Result |
|---|---|---|
| **GRaD-Nav** | 2503.03984 (IROS 2025) | Differentiable 3DGS + drone dynamics, zero-shot sim-to-real |
| **GRaD-Nav++** | 2506.14009 | + VLM for language-conditioned navigation |
| **SOUS VIDE / FiGS** | 2412.16346 (RA-L 2025) | 130 FPS, 105 hardware flights, robust to 30% mass, 40 m/s wind |
| **FalconGym 2.0** | 2510.02248 | GSplat + Edit API, 4D time-varying, 98.6% real success |
| **Mobile-GS** | 2603.11531 (ICLR 2026) | 40-80 FPS on mobile SoC, 4.8 MB model |

### 7.3 When to Build

Not until we have: (a) a trained predictor from Stages 1-3, and (b) a reward function
defined for GRPO. Build 3DGS sim from existing FPV video via MonoGS (monocular SLAM) for
coarse GRPO rewards. Higher-quality 3DGS requires deliberate capture (Southern Hemisphere
ski season July-Oct 2026).

---

## 8. Implementation Roadmap — What Changes in the Plan

### 8.1 Phase B Changes (Immediate)

| Component | Current Plan | Revised |
|---|---|---|
| Waypoint head (B1.16) | MLP D→256→128→4 | **Simplified PI Prober** (residual velocity + kinematic prior) |
| Action conditioning | FiLM | **AdaLN** (ablation-proven superior, arXiv 2512.24497) |
| Training schedule | Joint L_latent + L_wp from scratch | **Staged: video pre-train → warm-up → joint** (Stages 1-3) |
| Visual bottleneck | Not planned | **Cross-attention stem (16 queries) + MLP → 256** |
| Predictor readout | No CLS/readout token | **Add learned readout token (768,) per horizon step** |
| Encoder | DINOv3 ViT-B/16 (D=768) | **Unchanged** (prove pipeline first, upgrade later) |

**New components to build (not in existing codebase):**
1. **Readout token** — learned (1, 768) prepended to each horizon step in the predictor.
   Modify `PredictorConfig` + `PredictorOutput` + predictor forward pass. See §3.2.
2. **Visual bottleneck** — cross-attention stem (16 queries) + mean-pool + MLP(768→256).
   New module `vllatent/model/visual_bottleneck.py`.
3. **AdaLN layers** — replace FiLM conditioning in predictor. γ(c_vis), β(c_vis) modulate
   each transformer layer's LayerNorm. New module or integrated into predictor blocks.
4. **PI Prober** — MLP (768+256→256→128→4) with kinematic prior integration. Replaces
   MLP waypoint head. New module `vllatent/model/pi_prober.py`.

**Stale trust references to clean up (trust mechanism removed 2026-06-25):**
- `vllatent/schemas.py` docstring lines 4, 198 — still mention "trust readout"
- `CLAUDE.md` line 44 — mentions "trust thresholds + calibration (Phase C)"
- `plans/phase-b-sports-training.md` lines 57-58 — mentions "trust head training (Phase C)"
- `vllatent/verify/` — directory exists but only contains `__pycache__`; safe to delete

### 8.2 Phase C Changes (Future)

| Component | Addition |
|---|---|
| Privileged teacher | SkyJEPA trained on **sim-generated** trajectories (IsaacLab/pybullet, domain-randomized) |
| Distillation loss | L_kd = MSE(student_trajectory, teacher_trajectory) with stop-gradient |
| Full PI Prober | SO(3) exp map + motor force decomposition (replaces simplified version) |
| Language cross-attention | CLIP text tokens → cross-attention in predictor (B-2 originally) |
| GRPO | VD-GRPO on trajectory quality (GC-VAT centering reward + smoothness) |

### 8.3 Phase D Changes (Future)

| Component | Addition |
|---|---|
| 3DGS sim | MonoGS reconstruction → differentiable rollout environment |
| Closed-loop | MPPI planning with PI Prober forward model |
| Real deployment | PX4 + VINS + Orin NX, full SO(3) PI Prober with live state |

### 8.4 Updated Architecture Diagram

```
Phase B (current):
                                                                 ┌─ Phase C ──┐
  Stage 1         Stage 2         Stage 3                       │             │  Phase D
  ┌─────────┐   ┌───────────┐   ┌───────────┐   ┌──────────────┴──┐   ┌─────┴────────┐
  │Video    │   │Warm-up:   │   │Joint      │   │Privileged       │   │Closed-loop   │
  │pre-trn  │──►│PI Prober  │──►│fine-tune  │──►│teacher distill. │──►│MPPI + 3DGS   │
  │L_lat    │   │+ vis btlnk│   │L_lat+L_wp │   │SkyJEPA teacher  │   │real deploy   │
  │(no act) │   │(pred frzn)│   │           │   │+ GRPO (opt.)    │   │              │
  └─────────┘   └───────────┘   └───────────┘   └─────────────────┘   └──────────────┘
```

---

## 9. Key References

### Must-Read (directly inform the training policy)

| Paper | ArXiv / Venue | Why |
|---|---|---|
| **SkyJEPA** | 2606.23444 | PI Prober architecture, SIGReg, dynamics JEPA, Orin NX |
| **Loquercio et al.** | Science Robotics 2021 | Privileged teacher → sensorimotor student, 60 km/h sim-to-real |
| **DINO-world** | 2507.19468 | Staged pre-training, action blocks, Smooth L1 β=0.1 |
| **"What Drives Success in JEPA WMs"** | 2512.24497 | AdaLN > concat for action conditioning, DINOv3 best |
| **DINO-WM** | 2411.04983 (ICML 2025) | Frozen DINOv2, L2 latent loss, decoder loss hurts |
| **CoLA-World** | 2510.26433 | Warm-up prevents VQ collapse |
| **HPT** | 2409.20537 (NeurIPS 2024) | Cross-attention stems, equal vision/state representation |
| **V-JEPA-2-AC** | 2506.09985 | Frozen ViT-g, 300M predictor, 62 hrs robot data suffices |
| **GC-VAT** | 2412.00744 (NeurIPS 2025) | Projection-aware centering reward |
| **VD-GRPO (Plan-R1)** | 2505.17659 | Variance-decoupled GRPO for trajectory domains |
| **TrackVLA** | CoRL 2025 | Visual tracking VLA, 1.7M samples, velocity+yaw output |
| **WorldRFT** | 2512.19133 | GRPO on latent WM, trajectory Gaussianization |

### Secondary (inform specific design decisions)

| Paper | ArXiv | Relevance |
|---|---|---|
| pi0 | 2410.24164 | Flow matching action head, blockwise causal masking |
| TD-MPC2 | 2310.16828 | SimNorm, decoder-free WM, MPPI planning |
| DreamerV3 | 2301.04104 | RL in imagination, percentile return normalization |
| APV | 2203.13880 | Action-free WM pre-training |
| LAPA | 2410.11758 | VQ-VAE latent actions, internet video pre-training |
| Dream to Fly | 2501.14377 | DreamerV3 for drones, 9 m/s real flight |
| GRaD-Nav | 2503.03984 | Differentiable 3DGS + drone dynamics |
| SOUS VIDE / FiGS | 2412.16346 | 3DGS drone sim, 105 hardware flights |
| SRPO | 2511.15605 | 99.2% from 48.9% in 200 RL steps |
| DAgger | Ross et al. 2011 | Dataset aggregation for behavioral cloning |

---

## 10. Decision Summary

| Decision | Choice | Justification |
|---|---|---|
| **Action decoder** | PI Prober (simplified Phase B, full Phase D) | Physics-informed, bounded drift, composable with MPPI |
| **Latent mismatch** | Option B — vision conditions dynamics via AdaLN | Literature consensus (5+ papers), preserves visual anticipation |
| **Visual bottleneck** | Cross-attention stem (16 queries → 256) | HPT pattern, fixed-size regardless of encoder output |
| **Training schedule** | 5-stage: pre-train → warm-up → joint → distill → GRPO | DINO-world + CoLA-World + Loquercio consensus |
| **Privileged teacher** | SkyJEPA on sim-generated data (Phase C) | Loquercio paradigm + SkyJEPA's PI Prober expertise |
| **Encoder** | DINOv3 ViT-B/16, D=768 (unchanged) | Prove pipeline first, upgrade later |
| **Trust mechanism** | REMOVED | Simplicity; trust was speculative, not literature-grounded |
| **GRPO** | Phase C/D (after distillation) | Needs reward function + rollout environment |
| **3DGS sim** | Phase C/D | Not blocking training pipeline |

---

*End of revised training policy. This is a single recommended pipeline — specific,
literature-grounded, and implementable. Next step: update `plans/phase-b-sports-training.md`
to reflect the PI Prober and staged training decisions.*
