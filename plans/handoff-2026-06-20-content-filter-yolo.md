# Handoff — Continue Ralph Loop from B1.8

**Date:** 2026-06-23
**Branch:** main @ `e0be47c`
**Position:** B1.7 DONE → B1.8 (next) + Group 2 (B1.10 pending)
**Group:** B-1. User has authorized moving to B1.8 + Group 2.

---

## What's done (Group 0 + Group 1)

All of Group 0 (B1.1–B1.6) and Group 1 (B1.7a/7b/7c/7) are **done**.

### B1.7 pilot results (user-verified 2026-06-23)

- 15 skiing FPV clips curated in `configs/sports_clips.yaml`
- 11 accepted, 4 rejected (ski07: 0/7, ski08: 2/9, ski14: 13/44, ski15: 0/1)
- 38 FPV ranges → 173 sub-clips (10s each)
- Content filter: motion >= 8.0 AND no YOLO detections (36 rejected classes) AND segment >= 10 frames
- `ingest_data/latent_cache/pilot_summary.json` validates
- `verify_filter.py` on ski01 confirmed accepted/rejected split

### Previously done (Group 2 partial)

- **B1.9** — data quality report script: done
- **B1.9b** — per-clip HTML quality report: done

---

## What's next — in order

### 1. B1.8 — CosFly-Track download + adapter (USER-GATED)

- Tier TOOL+PURE
- Download 526-trace subset from `AutelRobotics/CosFly` on HuggingFace.
- Write `vllatent/ingest/cosfly_adapter.py` that converts CosFly format (CARLA, GT 6-DoF, 2 Hz)
  to ingest `.npz` cache. CosFly has GT waypoints → adapter writes `deltas` from GT poses,
  `vo_confidence = 1.0`. Uses `build_manifest_wild_video` with `motion_method="cosfly_gt"`.
- **Files:** `vllatent/ingest/cosfly_adapter.py` (new), `scripts/download_cosfly.sh` (new),
  `tests/test_cosfly_adapter.py` (new)
- **DoD:** Adapter converts entries to `.npz`. Manifest built. 10+ traces converted and inspected.
- **Test:** `$PY -m pytest -q tests/test_cosfly_adapter.py`. User verifies real download + conversion.
- **User gate:** user runs download + inspects converted traces.

### 2. B1.10 — MegaSaM VO validation on pilot clips (USER-GATED)

- Tier RESEARCH
- Run MegaSaM on 3-5 YouTube pilot clips from B1.7. Inspect 3D trajectory shapes.
- Compare against expected motion (downhill ski = roughly linear + descending).
- Produce GO / CONDITIONAL-GO / NO-GO verdict on MegaSaM for skiing FPV.
- **User gate:** user inspects trajectory plots + renders verdict.

### 3. B1.11 — Benchmark DINOv3 ViT-B/16 on Orin NX (USER-GATED, CRITICAL GATE)

- Can run in parallel with B1.8/B1.10. Blocks everything in Group 3+.
- **User gate:** user runs benchmark on Orin NX hardware.

---

## Constraints

- **Tier split:** all heavy imports (torch, ultralytics, transformers) must be LAZY (inside functions).
- **TDD:** write tests first (RED), implement (GREEN).
- **SSH HANDS-OFF:** paste command blocks, never drive ssh.
- **User-gated steps:** stay `in_progress` until user pastes verification.
- **Do NOT advance past the steps above** without explicit user permission.
- Commit format: `feat(phaseB): B1.8 — description`

---

## Environment

```bash
conda activate vllatent-ego-drone
# Run tests:
python -m pytest tests/ -q --ignore=tests/test_data_shapes.py
# Run content filter verification:
env -u ALL_PROXY -u all_proxy HF_ENDPOINT=https://hf-mirror.com \
  python scripts/verify_filter.py --frames ingest_data/frames/ski01 --device cuda
```

---

## Key files

- `plans/phase-b-sports-training.md` — authoritative Phase B plan
- `DEV_LOG.md` — step status table (read first each iteration)
- `configs/sports_clips.yaml` — 15 curated skiing clips
- `ingest_data/latent_cache/pilot_summary.json` — B1.7 results
- `vllatent/ingest/content_filter.py` — motion + YOLO-World filter (44 tests)
- `scripts/ingest_youtube_pilot.py` — pilot ingest orchestrator
