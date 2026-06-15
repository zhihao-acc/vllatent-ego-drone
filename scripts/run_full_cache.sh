#!/usr/bin/env bash
# Full cache build — all 50 train episodes (A5.17, USER-GATED).
#
# Refuses to run without --i-have-signed-off (guards against accidental multi-hour GPU jobs).
# Wraps `python -m vllatent.cache build` with the full-slice arguments. Resumable: episodes
# whose .npz already exists are skipped.
#
# Prerequisites (see docs/full-run-sizing.md):
#   1. fly0-m1 docker running, UE4 scene hot on port 41451
#   2. H20 WorldVLN server on port 8001 (ssh tunnel to localhost:8001)
#   3. data/aerialvln_json/train.slice.json present (50 episodes)
#   4. ~4 GB free disk for data/latent_cache/
#   5. DINOv3 + CLIP + V-JEPA-2 weights cached in HF_HOME
#
# Usage:
#   bash scripts/run_full_cache.sh --i-have-signed-off [--limit N] [--device DEVICE]
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_DIR}"

SIGNED_OFF=false
LIMIT=""
DEVICE="cuda"
HOST="127.0.0.1"
PORT="41451"
TEACHER_SERVER="http://127.0.0.1:8001"
SCENES_ROOT="/opt/aerialvln"
OUT="data/latent_cache/"
SLICE="data/aerialvln_json/train.slice.json"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --i-have-signed-off) SIGNED_OFF=true; shift ;;
    --limit)             LIMIT="$2";       shift 2 ;;
    --device)            DEVICE="$2";      shift 2 ;;
    --host)              HOST="$2";        shift 2 ;;
    --port)              PORT="$2";        shift 2 ;;
    --teacher-server)    TEACHER_SERVER="$2"; shift 2 ;;
    --scenes-root)       SCENES_ROOT="$2"; shift 2 ;;
    --out)               OUT="$2";         shift 2 ;;
    --slice)             SLICE="$2";       shift 2 ;;
    -h|--help)           sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "run_full_cache: unknown arg '$1'" >&2; exit 2 ;;
  esac
done

if [[ "${SIGNED_OFF}" != "true" ]]; then
  cat >&2 <<'EOF'
ERROR: full cache build refused — pass --i-have-signed-off to confirm.

This will run for ~11-14 hours (50 episodes × K=5 WorldVLN rollouts on H20) and write
~3 GB to data/latent_cache/. Read docs/full-run-sizing.md first.

  bash scripts/run_full_cache.sh --i-have-signed-off
EOF
  exit 1
fi

if [[ ! -f "${SLICE}" ]]; then
  echo "ERROR: slice not found: ${SLICE}" >&2
  exit 1
fi

cat >&2 <<EOF
[run_full_cache] A5.17 full cache build (USER-GATED).
  slice:    ${SLICE}
  out:      ${OUT}
  device:   ${DEVICE}
  teacher:  ${TEACHER_SERVER}
  scenes:   ${SCENES_ROOT}
  sim:      ${HOST}:${PORT}
  limit:    ${LIMIT:-all}
  Resumable: existing .npz files are skipped.
EOF

LIMIT_ARG=""
if [[ -n "${LIMIT}" ]]; then
  LIMIT_ARG="--limit ${LIMIT}"
fi

exec python -m vllatent.cache build \
  --slice "${SLICE}" \
  --out "${OUT}" \
  --scenes-root "${SCENES_ROOT}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --teacher-server "${TEACHER_SERVER}" \
  --device "${DEVICE}" \
  ${LIMIT_ARG}
