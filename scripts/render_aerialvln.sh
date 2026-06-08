#!/usr/bin/env bash
# Render RGB(+depth) by replaying reference_path poses in the AirSim sim — step 8 (USER-GATED).
#
# Runs ONLY inside the fly0-m1 docker with UE4 + a scene hot on port 41451. The
# agent does NOT launch UE4 / docker / SSH — the USER launches the scene, waits for
# "Listening on port 41451", then runs this. Foot-guns enforced in vllatent.render:
# quaternion reorder (xyzw), BGR->RGB, single-threaded Lock.
#
# Manual scene launch (USER, in the container; pick one; wait for port 41451):
#   /opt/aerialvln/MSBuild2018/LinuxNoEditor/MSBuild2018.sh -windowed -ResX=1280 -ResY=720
#   cd /opt/aerialvln/AirSimNH/LinuxNoEditor && ./AirSimNH.sh -windowed -ResX=1280 -ResY=720
#   /opt/aerialvln/Blocks/Blocks/LinuxNoEditor/Blocks.sh
#
# STUB at scaffold time — the real harness CLI lands in step 8.
set -euo pipefail

echo "[render_aerialvln] Phase-A step 8 stub (USER-GATED: docker + UE4 + port 41451)."
echo "  Intended: python -m vllatent.render --episode <episode.json> --scene <id> --out <dir>"
echo "Not implemented yet (step 8)."
exit 2
