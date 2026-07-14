"""Human- and plan-conditioned latent world model for sports-following drones.

The active B3 path consumes cached DINOv3 history plus a candidate 6-D future
camera plan and predicts future patch latents and person state.

Tier discipline (see CLAUDE.md "Tier split"):
  PURE  (numpy/pyyaml only, CI-gated): schemas, actions, frames, config, manifest, audit
  TORCH:                               encode/, data/, model/, train/
"""

__version__ = "0.0.1"
