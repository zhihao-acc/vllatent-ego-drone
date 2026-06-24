#!/usr/bin/env bash
# Download CosFly-Track dataset from HuggingFace (AutelRobotics/CosFly).
#
# Recommended: --meta-only (~6 GB trajectory JSONs for GT delta supervision).
# Full RGB is 119 GB of CARLA urban renders — not useful for skiing latent model.
#
# Usage:
#   bash scripts/download_cosfly.sh [--out DIR] [--meta-only]
#
# Requires: hf CLI (pip install huggingface_hub)
# Uses HF_ENDPOINT env var if set (for mirrors like hf-mirror.com).
set -euo pipefail

OUT_DIR="ingest_data/cosfly"
META_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)       OUT_DIR="$2"; shift 2 ;;
        --meta-only) META_ONLY=true; shift ;;
        -h|--help)
            echo "Usage: $0 [--out DIR] [--meta-only]"
            echo "  --out DIR     Output directory (default: ingest_data/cosfly)"
            echo "  --meta-only   Download only trajectory.json files (~few MB), skip frames"
            exit 0
            ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "[cosfly] Downloading CosFly-Track from AutelRobotics/CosFly"
echo "[cosfly] Output: ${OUT_DIR}"
echo "[cosfly] HF_ENDPOINT: ${HF_ENDPOINT:-https://huggingface.co}"

mkdir -p "${OUT_DIR}"

if ! command -v hf &>/dev/null; then
    echo "[cosfly] ERROR: hf CLI not found. Install with: pip install huggingface_hub"
    exit 1
fi

if [[ "${META_ONLY}" == true ]]; then
    echo "[cosfly] Meta-only mode: downloading trajectory.json + perturbation_report.json only"
    hf download "AutelRobotics/CosFly" \
        --repo-type dataset \
        --local-dir "${OUT_DIR}" \
        --include "data_v7/*/trajectory_*/*/trajectory.json" \
                  "data_v7/*/trajectory_*/*/perturbation_report.json"
else
    echo "[cosfly] Selective download: trajectory.json + rgb.png (skipping depth/instance/debug)"
    echo "[cosfly] WARNING: rgb.png alone is ~119 GB. Consider --meta-only instead."
    hf download "AutelRobotics/CosFly" \
        --repo-type dataset \
        --local-dir "${OUT_DIR}" \
        --include "data_v7/*/trajectory_*/*/trajectory.json" \
                  "data_v7/*/trajectory_*/*/perturbation_report.json" \
                  "data_v7/*/trajectory_*/*/frames_playback/*/rgb.png"
fi

echo ""
echo "[cosfly] Download complete."
echo "[cosfly] Next: convert with the adapter:"
echo "  python -c \""
echo "    from vllatent.ingest.cosfly_adapter import convert_dataset"
echo "    results = convert_dataset('${OUT_DIR}/data_v7', 'ingest_data/latent_cache', limit=10)"
echo "    print(f'Converted {len(results)} traces')"
echo "  \""
