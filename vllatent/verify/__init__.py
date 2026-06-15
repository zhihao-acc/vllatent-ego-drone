"""TORCH tier — frozen V-JEPA-2 surprise verifier (the independent second trust gate).

The verifier (``vjepa2.py``) runs Meta's frozen V-JEPA-2 video JEPA: given the observed
context frames, its predictor forecasts the latent of the GT future frames; the *surprise*
``s_j = 1 - cos(ẑ_j, z_j)`` between that forecast ``ẑ`` and the actually-encoded future
latent ``z`` is the disagreement-independent signal that feeds ``OracleTarget.vjepa_surprise``
(Phase-C gate). Heavy imports (torch / transformers) are LAZY (inside functions/methods) so a
torch-free box imports this package without crashing; the real-weight forward is the USER-GATED
smoke. Tested via the monkeypatched contract test (``tests/test_verify_contract.py``).
"""
