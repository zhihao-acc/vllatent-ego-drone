#!/usr/bin/env bash
# A5.8 Tier-1: list WorldVLN weights metadata WITHOUT downloading blobs.
# Run with:  bash scripts/a5_8_worldvln_meta.sh   (from repo root)
set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com
PY=/home/zh/miniconda3/envs/vln-ego-drone-1.1/bin/python   # any env w/ huggingface_hub

"$PY" -m pip install -q -U "huggingface_hub[cli]" 2>/dev/null || true

# (1) WEIGHTS: list every file + size + license WITHOUT downloading blobs
"$PY" - <<'PYEOF'
import os
# httpx (new huggingface_hub) rejects the legacy socks:// scheme our ALL_PROXY exports.
# hf-mirror.com is directly reachable in CN, so drop the proxy for THIS process only
# (does NOT unset ALL_PROXY at the shell level — other tools keep their proxy).
for k in ("ALL_PROXY","all_proxy","HTTP_PROXY","http_proxy","HTTPS_PROXY","https_proxy"):
    os.environ.pop(k, None)
from huggingface_hub import HfApi
info = HfApi().repo_info("EmbodiedCity/WorldVLN", files_metadata=True)
tot = 0
for s in sorted(info.siblings, key=lambda x: -(x.size or 0)):
    tot += s.size or 0
    print(f"{(s.size or 0)/1e9:8.3f} GB  {s.rfilename}")
print(f"--- TOTAL ~ {tot/1e9:.2f} GB / {len(info.siblings)} files ---")
print("cardData.license:", (info.cardData or {}).get("license"))
PYEOF
