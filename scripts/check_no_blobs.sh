#!/usr/bin/env bash
# Pre-commit guard: reject accidental large/binary commits.
# Adapted from upipe/fly0 for a training repo (adds .npy/.npz; allows tiny fixtures).
#
# Rejects:
#   - any STAGED file larger than MAX_KB (default 512 KB), EXCEPT under fixtures/
#   - any STAGED file with a weights/video/model/archive/array extension
#     (defence-in-depth over .gitignore), EXCEPT cached-array fixtures under fixtures/
#
# Usage:
#   bash scripts/check_no_blobs.sh        # checks staged files (use as a pre-commit hook)
#   ALL=1 bash scripts/check_no_blobs.sh  # checks ALL tracked files (CI mode)
#
# Exit 0 = clean, exit 1 = a blob slipped through.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_DIR}"

MAX_KB="${MAX_KB:-512}"
# weights / video / model / archive  +  raw arrays (.npy/.npz: cached latents must never enter git)
BLOCKED_RE='\.(pt|pth|ckpt|safetensors|onnx|bin|h5|mp4|avi|mov|mkv|zip|7z|tar|tgz|bag|npy|npz)$'

if [[ "${ALL:-0}" == "1" ]]; then
  mapfile -t FILES < <(git ls-files)
else
  mapfile -t FILES < <(git diff --cached --name-only --diff-filter=ACM)
fi

fail=0
for f in "${FILES[@]}"; do
  [[ -z "${f}" ]] && continue
  [[ -f "${f}" ]] || continue

  # Tiny committed fixtures (hand-authored episodes, synthetic latents) are allowed,
  # but still size-capped so a real capture can't sneak in under fixtures/.
  is_fixture=0
  if [[ "${f}" == fixtures/* ]]; then is_fixture=1; fi

  if [[ "${is_fixture}" -eq 0 ]] && echo "${f}" | grep -qiE "${BLOCKED_RE}"; then
    echo "BLOB-REJECT (extension): ${f}"
    fail=1
    continue
  fi

  size_kb=$(( ( $(wc -c < "${f}") + 1023 ) / 1024 ))
  if (( size_kb > MAX_KB )); then
    echo "BLOB-REJECT (size ${size_kb} KB > ${MAX_KB} KB): ${f}"
    fail=1
  fi
done

if (( fail != 0 )); then
  echo ""
  echo "Refusing the commit: weights / videos / cached latents / large blobs do not belong in git."
  echo "They live in runs/ , weights/ , latents/ , cache/ , data/ (all gitignored; see docs/TOPOLOGY.md)."
  echo "Tiny fixtures go under fixtures/ and must stay <= ${MAX_KB} KB."
  exit 1
fi

echo "check_no_blobs: OK (no staged weights/videos/cached-latents/large blobs)."
