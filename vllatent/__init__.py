"""Causal ski-simulation contracts and latent world-model components.

The active B3-CS simulator interface is four body-FRD command channels in SI
units plus a separate ``dt_seconds``.  The historical six-field passive-video
token remains a distinct compatibility contract and is not a simulator command.

Tier discipline (see ``AGENTS.md``):
  PURE  (NumPy/PyYAML only, CI-gated): schemas, config, manifest, selected ingest,
                                      and all of sim/
  TORCH:                               encode/, data/, model/, train/
"""

__version__ = "0.0.1"
