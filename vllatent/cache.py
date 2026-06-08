"""Render -> encode -> CACHE orchestrator + provenance (SIM tier) — Phase-A step 9.

For an episode: render each reference_path pose (vllatent.render) -> encode RGB
(vllatent.encode) -> write (196,768) fp16 latents per step + update the cache
manifest (vllatent.manifest). Deterministic + resumable (skip already-cached).
The manifest pins encoder id+revision, dataset slice, quaternion order, BGR->RGB
flag, render config hash (so a cache is reproducible/auditable).

Heavy imports are LAZY. Manifest logic is tested with mocked render+encode
(CI-safe); RUNNING the small-slice build needs sim + GPU together — USER-GATED.

CLI:  python -m vllatent.cache build --episodes <json> --limit N --scenes-root /opt/aerialvln --out <dir>

See plans/phase-a-data-and-io-contract.md step 9/12.
"""
from __future__ import annotations


def build_cache(*args, **kwargs):  # pragma: no cover - implemented in step 9
    raise NotImplementedError("vllatent.cache.build_cache lands in Phase-A step 9")


__all__ = ["build_cache"]
