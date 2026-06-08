"""vllatent — compact latent world-action model for aerial VLN.

Single namespaced top-level package. Phase A is plumbing + data:
the discrete->continuous-4DoF action mapping, the AerialVLN JSON audit, the
frozen-DINOv3 encode->cache pipeline, and the cached-latent loader.

Tier discipline (see CLAUDE.md "Tier split"):
  PURE  (numpy/pyyaml only, CI-gated): schemas, actions, frames, config, manifest, audit
  TORCH (heavy imports lazy):          encode/, data/
  SIM   (airsim lazy; fly0-m1 only):   render/, cache

Phases A-C are STANDALONE — no sibling (fly0/navdreamer) import.
"""

__version__ = "0.0.1"
