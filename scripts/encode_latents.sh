#!/usr/bin/env bash
# Encode rendered RGB -> frozen DINOv3 (196,768) fp16 latents -> cache — step 7/9 (USER-GATED).
#
# Runs on a GPU box: dev 5060 Ti 16 GB (small slice) or AutoDL H20 ~96 GB (full job,
# SSH HANDS-OFF — agent gives the block, user pastes). HF weights via the mirror:
#   export HF_ENDPOINT=https://hf-mirror.com
#
# STUB at scaffold time — the real encode/cache CLI lands in steps 7 + 9.
set -euo pipefail

echo "[encode_latents] Phase-A step 7/9 stub (USER/DEV-GATED: GPU + HF weights)."
echo "  Intended: HF_ENDPOINT=https://hf-mirror.com python -m vllatent.cache build \\"
echo "              --episodes <json> --limit N --scenes-root /opt/aerialvln --out data/latent_cache/"
echo "Not implemented yet (steps 7, 9)."
exit 2
