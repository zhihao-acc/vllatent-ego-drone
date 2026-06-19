"""Sports-following drone dataset pipeline (Phase B1).

Video-based data pipeline: YouTube FPV clips → MegaSaM ego-motion → DINOv3 latent cache.

Tier split (mirrors vllatent core):

| Tier     | Modules                                  | Imports                     |
|----------|------------------------------------------|-----------------------------|
| **PURE** | schemas, config, quality, ego_motion     | numpy/pyyaml/stdlib         |
| **TORCH**| encode                                   | + torch/timm (LAZY)         |
| **TOOL** | acquire, preprocess, megasam             | subprocess (yt-dlp/ffmpeg)  |
| **ORCH** | cache, loader, pipeline, __main__        | orchestrates all tiers      |
"""
