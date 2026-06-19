"""Wild-video ingestion pipeline — YouTube FPV clips to cached DINOv3 latents.

Tier split:

| Tier     | Modules                          | Imports                    |
|----------|----------------------------------|----------------------------|
| **PURE** | quality, ego_motion              | numpy/stdlib               |
| **TOOL** | acquire, preprocess, megasam     | subprocess (yt-dlp/ffmpeg) |
| **ORCH** | pipeline, __main__               | orchestrates all tiers     |

Manifest building/validation lives in ``vllatent.manifest``.
Batch DINOv3 encoding lives in ``vllatent.encode.batch``.
"""
