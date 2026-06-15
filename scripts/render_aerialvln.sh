#!/usr/bin/env bash
# Render RGB by replaying AerialVLN reference_path poses in the AirSim sim — A5.13 (USER-GATED).
#
# Thin wrapper around `python -m vllatent.render` (the live render CLI landed in A5.13). Runs ONLY
# inside the fly0-m1 docker with UE4 + a scene hot on port 41451. The agent does NOT launch UE4 /
# docker / SSH — the USER launches the scene, waits for "Listening on port 41451", then runs this.
# Foot-guns are enforced in vllatent.render: quaternion reorder (xyzw, #1), BGRA->BGR->RGB (#2),
# single-threaded msgpack-RPC Lock on every client call (#3).
#
# Usage (inside fly0-m1, after the scene is up on :41451):
#   bash scripts/render_aerialvln.sh [--episode <json>] [--scene <id>] [--out <dir>] [--host H] [--port P]
# Defaults: episode=fixtures/episodes/tiny_episode.json, scene=1, out=/tmp/render_smoke/, host=127.0.0.1, port=41451
# Override the interpreter with PY=... (e.g. PY=/home/zh/miniconda3/envs/vllatent-ego-drone/bin/python).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_DIR}"

PY="${PY:-python}"
EPISODE="fixtures/episodes/tiny_episode.json"
SCENE="1"
OUT="/tmp/render_smoke/"
HOST="127.0.0.1"
PORT="41451"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --episode) EPISODE="$2"; shift 2 ;;
    --scene)   SCENE="$2";   shift 2 ;;
    --out)     OUT="$2";     shift 2 ;;
    --host)    HOST="$2";    shift 2 ;;
    --port)    PORT="$2";    shift 2 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "render_aerialvln: unknown arg '$1'" >&2; exit 2 ;;
  esac
done

cat >&2 <<EOF
[render_aerialvln] A5.13 live render (USER-GATED: fly0-m1 docker + UE4 scene on port ${PORT}).
  Launch ONE scene MANUALLY in the container first and wait for "Listening on port 41451":
    /opt/aerialvln/MSBuild2018/LinuxNoEditor/MSBuild2018.sh -windowed -ResX=1280 -ResY=720
    cd /opt/aerialvln/AirSimNH/LinuxNoEditor && ./AirSimNH.sh -windowed -ResX=1280 -ResY=720
    /opt/aerialvln/Blocks/Blocks/LinuxNoEditor/Blocks.sh
  Then this replays ${EPISODE} (scene ${SCENE}) -> per-pose uint8 RGB .npy in ${OUT}.
EOF

exec env PYTHONNOUSERSITE=1 "${PY}" -s -m vllatent.render \
  --episode "${EPISODE}" --scene "${SCENE}" --out "${OUT}" --host "${HOST}" --port "${PORT}"
