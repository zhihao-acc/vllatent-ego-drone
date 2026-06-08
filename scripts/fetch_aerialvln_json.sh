#!/usr/bin/env bash
# Fetch + SLICE AerialVLN split JSONs — Phase-A step 6.
#
# The annotations (AerialVLN + AerialVLN-S, <100MB total) are NOT on the S3 path the
# scaffold guessed. They live on Kaggle (shuboliu/aerialvln) with a Baidu Netdisk CN
# mirror. The DOWNLOAD is USER-GATED (network/credentialed); this script does the LOCAL
# SLICE (CI-safe) once the full split JSON is present, and prints the download block if not.
#
# Real schema (confirmed Phase-A step 5b): top-level {"episodes":[...]}; each episode has
#   episode_id / trajectory_id / scene_id / instruction.instruction_text / start_position /
#   start_rotation ([w,x,y,z] quaternion, w-FIRST) / goals[].position /
#   reference_path ([x,y,z,pitch,roll,yaw] EULER radians, 6-wide) / actions (list[int] 0..7).
# Invariant: len(reference_path) == len(actions).
#
# License: AerialVLN is CC BY-NC-SA 4.0 (non-commercial). Never commit the JSONs (data/ is gitignored).
set -euo pipefail

SPLIT="${1:-train}"                  # train | val_seen | val_unseen | test
LIMIT="${2:-50}"                     # number of episodes to keep in the slice
OUT="${3:-data/aerialvln_json}"
PY="${PY:-python3}"                  # any python3 (stdlib json only)
SRC="${SRC:-${OUT}/${SPLIT}.json}"   # the full split JSON (downloaded by the user)
DEST="${OUT}/${SPLIT}.slice.json"

mkdir -p "${OUT}"

if [[ ! -f "${SRC}" ]]; then
  cat >&2 <<EOF
[fetch_aerialvln_json] Full split JSON not found: ${SRC}
Download it first (USER-GATED). The annotations are a single <100MB zip:

  Kaggle (needs ~/.kaggle/kaggle.json):
    kaggle datasets download -d shuboliu/aerialvln -p ${OUT} && ( cd ${OUT} && unzip -o aerialvln.zip )

  CN mirror — Baidu Netdisk (extract code cgwh):
    https://pan.baidu.com/s/1mhNeqDjipXULMa2PfTaZKQ?pwd=cgwh
    (download via the Baidu client / BaiduPCS-Go, unzip so ${SRC} exists)

Then re-run:  bash scripts/fetch_aerialvln_json.sh ${SPLIT} ${LIMIT} ${OUT}
License to record: CC BY-NC-SA 4.0 (non-commercial).
EOF
  exit 2
fi

"${PY}" - "${SRC}" "${DEST}" "${LIMIT}" <<'PY'
import json, sys
src, dest, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
data = json.load(open(src))
eps = data["episodes"] if isinstance(data, dict) and "episodes" in data else data
sliced = list(eps)[:limit]
json.dump({"episodes": sliced}, open(dest, "w"))
print(f"[fetch_aerialvln_json] sliced {len(sliced)}/{len(eps)} episodes -> {dest}")
PY

echo "[fetch_aerialvln_json] License: CC BY-NC-SA 4.0 (non-commercial). data/ is gitignored — do not commit."
echo "[fetch_aerialvln_json] Audit:  ${PY} -m vllatent.audit --episode ${DEST} --report -"
