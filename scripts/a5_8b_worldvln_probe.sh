#!/usr/bin/env bash
# A5.8b: download WorldVLN CODE/config/license ONLY (no weight blobs), then probe
# for sampling/determinism, action-head dims, checkpoint names, and license.
# Run with:  bash scripts/a5_8b_worldvln_probe.sh   (from repo root)
set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com
PY=/home/zh/miniconda3/envs/vln-ego-drone-1.1/bin/python

"$PY" - <<'PYEOF'
import os
# new huggingface_hub -> httpx rejects legacy socks:// ; mirror is direct in CN.
for k in ("ALL_PROXY","all_proxy","HTTP_PROXY","http_proxy","HTTPS_PROXY","https_proxy"):
    os.environ.pop(k, None)
from huggingface_hub import snapshot_download
p = snapshot_download(
    "EmbodiedCity/WorldVLN",
    allow_patterns=["*.py","*.json","*.md","*.txt","*.yaml","*.yml","LICENSE*"],
    local_dir="/tmp/worldvln_probe/code",
)
print("code at:", p)
PYEOF

C=/tmp/worldvln_probe/code
echo
echo "### files pulled ###"
find "$C" -type f | sed "s#$C/##" | sort

echo
echo "### (a) sampling / determinism signals ###"
grep -rniE "do_sample|temperature|top_p|top_k|greedy|argmax|multinomial|\.sample\(|manual_seed|generator=" "$C" --include='*.py' | head -40 || true

echo
echo "### (b) action head: 4-DoF vs 6-DoF (roll/pitch==0?) ###"
grep -rniE "action_dim|num_actions|\bdof\b|roll|pitch|yaw|waypoint|out_features|Linear\([0-9]" "$C" --include='*.py' | head -40 || true

echo
echo "### (c) checkpoint names (expect INFINITY_CKPT / STAGE2_LATENT2ACTION_CKPT, ~8B InfinityStar) ###"
grep -rniE "INFINITY_CKPT|STAGE2_LATENT2ACTION_CKPT|checkpoint|\.pth|\.safetensors|from_pretrained" "$C" --include='*.py' | head -30 || true

echo
echo "### (d) license ###"
sed -n '1,15p' "$C"/LICENSE* 2>/dev/null || grep -rni "license" "$C"/*.md | head
