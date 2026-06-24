#!/usr/bin/env bash
# Download CosFly-Track dataset from HuggingFace (AutelRobotics/CosFly).
#
# Usage:
#   bash scripts/download_cosfly.sh [--out DIR] [--limit N]
#
# Requires: huggingface-cli (pip install huggingface_hub)
# Uses HF_ENDPOINT env var if set (for mirrors like hf-mirror.com).
set -euo pipefail

OUT_DIR="ingest_data/cosfly"
LIMIT=""
REPO="AutelRobotics/CosFly"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)    OUT_DIR="$2"; shift 2 ;;
        --limit)  LIMIT="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--out DIR] [--limit N]"
            echo "  --out DIR   Output directory (default: ingest_data/cosfly)"
            echo "  --limit N   Download only the first N trace directories"
            exit 0
            ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "[cosfly] Downloading CosFly-Track from ${REPO}"
echo "[cosfly] Output: ${OUT_DIR}"
echo "[cosfly] HF_ENDPOINT: ${HF_ENDPOINT:-https://huggingface.co}"

mkdir -p "${OUT_DIR}"

if ! command -v huggingface-cli &>/dev/null; then
    echo "[cosfly] ERROR: huggingface-cli not found. Install with: pip install huggingface_hub"
    exit 1
fi

if [[ -n "${LIMIT}" ]]; then
    echo "[cosfly] Limiting to first ${LIMIT} trace directories"
    echo "[cosfly] Fetching file list..."

    # Get the list of trajectory directories and take the first N
    huggingface-cli download "${REPO}" \
        --repo-type dataset \
        --local-dir "${OUT_DIR}" \
        --include "data_v7/*" \
        2>&1 | head -20

    echo "[cosfly] NOTE: --limit only applies to the adapter (convert_dataset --limit N)."
    echo "[cosfly] The HF download fetches the full dataset; use convert_dataset(limit=N) to process a subset."
else
    huggingface-cli download "${REPO}" \
        --repo-type dataset \
        --local-dir "${OUT_DIR}" \
        --include "data_v7/*"
fi

echo ""
echo "[cosfly] Download complete."
echo "[cosfly] Next: convert with the adapter:"
echo "  python -c \""
echo "    from vllatent.ingest.cosfly_adapter import convert_dataset"
echo "    results = convert_dataset('${OUT_DIR}/data_v7', 'ingest_data/latent_cache', limit=10)"
echo "    print(f'Converted {len(results)} traces')"
echo "  \""
