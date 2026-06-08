#!/usr/bin/env bash
# Fetch a SLICE of the AerialVLN split JSONs from S3 — Phase-A step 6 (USER-GATED).
#
# Pixels are NOT in these JSONs (they render from the sim). The JSONs carry the
# discrete `actions`, the `reference_path` poses, and the instructions — i.e. the
# audit + label source. They are gitignored (data/) and blob-guarded; never commit them.
#
# License: AerialVLN is CC BY-NC-SA 4.0 (non-commercial). Record the confirmation
# in DEV_LOG.md + docs/io-contract.md.
#
# STUB at scaffold time — the real download wiring lands in step 6. The agent does
# NOT run network/credentialed pulls; the USER pastes this block and returns output.
set -euo pipefail

SLICE="${1:-train}"          # train | val_seen | val_unseen | test
LIMIT="${2:-50}"             # number of episodes to keep in the slice
OUT="${3:-data/aerialvln_json}"
S3_BASE="https://aerialvln.s3.ap-southeast-2.amazonaws.com/dataset/aerialvln"

echo "[fetch_aerialvln_json] Phase-A step 6 stub."
echo "  Intended (USER pastes once step 6 lands):"
echo "    mkdir -p ${OUT}"
echo "    wget -O ${OUT}/${SLICE}.json ${S3_BASE}/${SLICE}.json   # or via a CN mirror if blocked"
echo "    python - <<'PY'  # keep first ${LIMIT} episodes -> ${OUT}/${SLICE}.slice.json"
echo "    # (slice writer lands in step 6)"
echo "    PY"
echo "    python -m vllatent.audit --episode ${OUT}/${SLICE}.slice.json --report -"
echo ""
echo "Not implemented yet (step 6). License to confirm: CC BY-NC-SA 4.0."
exit 2
