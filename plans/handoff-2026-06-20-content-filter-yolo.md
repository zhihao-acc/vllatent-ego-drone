# Handoff — B1.7c Content Filter: Replace CLIP with YOLO

**Date:** 2026-06-20
**Branch:** main @ `aad34b9`
**Position:** B1.7c (in_progress) → B1.7 (USER-GATED)
**Group:** B-1 Group 1 (Data Acquisition + Quality). Do NOT move to Group 2 without user permission.

---

## What's done

The content filter (`vllatent/ingest/content_filter.py`) uses two signals:
1. **Motion** (frame-to-frame pixel difference) — works well, catches static/product shots.
2. **CLIP zero-shot** — DOES NOT WORK for within-domain filtering. All frames in a skiing video
   score 0.999 because they share the same visual scene (snow, mountain, trees). CLIP cannot
   distinguish perspective or viewpoint.

Current state: motion does all the real work (threshold ≥ 8.0). CLIP is dead weight.

Verification script: `scripts/verify_filter.py --frames <dir> --device cuda` splits frames into
`accepted/` and `rejected/` directories. The user confirmed motion-based filtering works but
CLIP contributes nothing.

---

## What's next — research & replan FIRST, then execute

**User directive:** "use YOLO for sure. Drop what's little impact."

### Research phase (do this first)

1. **YOLO for zero-shot object detection** — research which YOLO variant supports zero-shot /
   open-vocabulary detection (e.g., YOLO-World, Grounding DINO, OWL-ViT). The goal: query
   for objects like "drone", "quadcopter", "camera", "electronic device", "person facing viewer"
   and reject frames where these are detected.

2. **Evaluate options:**
   - YOLO-World (real-time, open-vocabulary)
   - Grounding DINO (zero-shot, heavier)
   - OWL-ViT (zero-shot, medium weight)
   - ultralytics YOLO with custom prompts

3. **Determine which signals to keep:**
   - Motion (HIGH impact) → KEEP as primary signal
   - YOLO object detection → ADD as semantic signal (catches what motion misses: slow pan
     over a drone, handheld gear shots with slight motion)
   - CLIP → DROP (0.999 on everything, no discrimination)
   - PySceneDetect shot boundaries → DROP or keep only for video-level verdict display

### Implementation plan (after research)

1. Add YOLO-based frame rejection to `content_filter.py`
2. Remove CLIP scorer (or move to a separate "cross-domain pre-filter" that only runs at
   video-level, not per-frame)
3. Remove PySceneDetect if not needed
4. Final filter logic: `is_fpv = motion >= threshold AND no_rejected_objects_detected`
5. Update tests
6. Run `scripts/verify_filter.py` on ski01 — user verifies
7. Run full `scripts/ingest_youtube_pilot.py` — user verifies (B1.7 USER GATE)

---

## Key files

- `vllatent/ingest/content_filter.py` — the filter module (CLIP + motion, needs YOLO)
- `scripts/verify_filter.py` — visual verification tool
- `scripts/ingest_youtube_pilot.py` — pilot ingest script (uses filter_video_from_paths)
- `tests/test_content_filter.py` — 36 tests
- `plans/phase-b-sports-training.md` — authoritative Phase B plan
- `DEV_LOG.md` — step status table

---

## Constraints

- **Tier split:** all heavy imports (torch, ultralytics, transformers) must be LAZY (inside functions).
  Module must import on a torch-free box.
- **TDD:** write tests first (RED), implement (GREEN).
- **SSH HANDS-OFF:** paste command blocks, never drive ssh.
- **User-gated steps:** stay in_progress until user pastes verification.
- **Do NOT advance past Group 1** (B1.7/B1.8) without explicit user permission.
- Commit format: `fix(phaseB): B1.7c — description`

---

## Environment

```bash
conda activate vllatent-ego-drone
# Run filter verification:
env -u ALL_PROXY -u all_proxy HF_ENDPOINT=https://hf-mirror.com \
  python scripts/verify_filter.py --frames ingest_data/frames/ski01 --device cuda
# Run tests:
python -m pytest tests/test_content_filter.py -q
```

---

## Research findings (from this session)

- CLIP ViT-B/32 cannot distinguish viewpoint/perspective. It only reads scene content.
  Within the same visual domain (all frames have snow/mountain), it scores everything 0.999.
- CLIP ignores prepositions ("from" vs "of") — tested at 0.50-0.56 accuracy on compositional
  benchmarks (ARO/WinoGround ICLR 2023).
- Brand names ("GoPro") trigger via typographic attacks — the logo on the device matches the
  text token regardless of context.
- Motion (temporal difference) is the correct primary signal for FPV vs static content.
- YOLO/object detection is the correct semantic signal for detecting specific objects (drones,
  cameras, gear) that should never appear in training data.
