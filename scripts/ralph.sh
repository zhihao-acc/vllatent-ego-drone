#!/usr/bin/env bash
# Launch the Ralph loop for vllatent-ego-drone Phase-A step iteration.
#
# The Ralph loop re-feeds the SAME prompt each iteration; the agent reads
# DEV_LOG.md (position) -> .claude/ralph-rules.md (protocol) ->
# plans/phase-a-data-and-io-contract.md (step DoD), executes the lowest pending
# step, tests, records, and commits `feat(phaseA): step N — ...`.
#
# This script just PRINTS the canonical /ralph-loop invocation — start it from a
# Claude Code session in this repo (the loop's stop-hook runs in that session).
# It stops at the first user-gated step (6: S3 download; 7/8/9 live render/encode/cache).
set -euo pipefail

cat <<'EOF'
Start the Ralph loop from a Claude Code session in this repo with:

  /ralph-loop Iterate vllatent-ego-drone Phase A. Each iteration follow .claude/ralph-rules.md exactly: READ DEV_LOG.md + ralph-rules + plans/phase-a-data-and-io-contract.md, IDENTIFY the lowest pending step, REVIEW its DoD, EXECUTE (pure-tier / fixtures-first), TEST with the step command, RECORD in DEV_LOG.md, COMMIT feat(phaseA): step N. User-gated steps (render/cache/H20/docker/network) stay in_progress until the user pastes verification — never auto-mark done. Stop at the first user-gated step. --max-iterations 10 --completion-promise 'PHASE A PURE LANE GREEN'

Monitor:   grep '^iteration:' .claude/ralph-loop.local.md
Cancel:    /cancel-ralph   (or: rm .claude/ralph-loop.local.md  for a deterministic stop)
EOF
