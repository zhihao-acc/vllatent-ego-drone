# Ralph Rules - vllatent-ego-drone Phase B-3

This Claude-facing file mirrors the active Codex workflow. The canonical current
state is, in order, `DEV_LOG.md`, `.codex/ralph-rules.md`, and
`plans/phase-b3-human-conditioned-world-model.md`. Follow those files if this
summary ever differs.

## Read And Execute

1. Read the three canonical files above; use `plans/phase-b-sports-training.md`
   only for historical B1/B2 evidence.
2. Identify the lowest actionable B3 step and state one bounded diagnosis and
   completion criterion.
3. Add a failing regression test before changing behavior.
4. Implement the smallest evidence-supported repair.
5. Run focused tests and Ruff, then record exact verified results in the dev log
   and B3 plan.
6. Commit only when the user asks, with explicit paths; never use `git add -A`
   or `git add .`.

Phase-A plans, AerialVLN seams, and B1/B2 reports are historical reference, not
the active queue.

## Current Position

- B3.0-B3.5 are complete.
- B3.6 remains blocked after the review-backed real-transition-verifier repair.
  The corrected tiny run passes G1b, but G1a misses the null-plan margin and
  G1d misses the shuffled/flipped aggregate-margin and yaw-geometry requirements.
- Do not run the source-held-out gate, data/capacity scaling, B3.7, or H20 until
  a defensible conditioning repair passes the corrected tiny protocol.
- B2.12/H20 action-imitation training remains inactive.

## B3 Invariants

- Frozen DINOv3 ViT-B/16, D=768, cached fp16 patch latents.
- Depth 6, H=3, T=8 for the serious local model.
- Candidate future plan tokens are
  `[unit_dir_x, unit_dir_y, unit_dir_z, log_speed_ratio, yaw_rate_norm, valid]`.
- Future latent/person targets, masks, and confidences are labels only and never
  model inputs.
- Translation is scale-free; metric speed is controller-side and clamped
  strictly below `7.5 m/s`.
- Source splits are by source video, not sub-clip.
- Do not introduce language, diffusion, game data, SAM2, PI-Prober, metric
  waypoint training, or EGO-Planner before deterministic B3 gates pass.

## Data, Hardware, And Verification

- Never modify or delete retained dataset files while diagnosing B3.6 unless the user explicitly
  approves a reviewed cleanup manifest.
- Never commit weights, `.npz`, videos, frames, caches, generated reports, or
  run logs.
- H20, SSH, docker, Orin, video downloads, and multi-GB jobs are user-gated. No
  H20 command is permitted before B3.7 becomes eligible.
- For B3.6, rerun tiny overfit before any fixed source-held-out comparison.
- Before CUDA, verify `nvidia-smi -L`, `/dev/nvidia*`, and
  `torch.cuda.device_count()` in the intended execution mode.
- Run `git diff --check` before handoff and do not treat partial job output as
  completion.
- A local Claude Ralph loop can be stopped by removing
  `.claude/ralph-loop.local.md`.
