#!/usr/bin/env bash
# Print the Ralph-loop launch prompt for the active vllatent-ego-drone B3 queue.
#
# Each iteration reads DEV_LOG.md (position), .codex/ralph-rules.md (protocol),
# and plans/phase-b3-human-conditioned-world-model.md (DoD). B3.6 is currently
# blocked on corrected tiny G1a/G1d plan-dependence gates.
#
# This script only prints the Claude `/ralph-loop` invocation. Project protocol
# remains canonical in `.codex/ralph-rules.md`; Claude keeps loop state locally.
set -euo pipefail

cat <<'EOF'
Start the Ralph loop from a Claude Code session in this repo with:

  /ralph-loop Continue vllatent-ego-drone Phase B3. Each iteration follow .codex/ralph-rules.md exactly: READ DEV_LOG.md + .codex/ralph-rules.md + plans/phase-b3-human-conditioned-world-model.md, IDENTIFY the lowest actionable B3 step, REVIEW its DoD, make one bounded review-supported improvement, TEST with the narrowest relevant check, and RECORD verified facts in DEV_LOG.md. Current stop: B3.6 remains blocked because the corrected tiny protocol misses G1a/G1d; do not run the source-held-out gate, data/capacity scaling, or B3.7/H20 until a defensible counterfactual-conditioning repair passes corrected tiny G1a/G1b/G1d. Stop when the blocker repeats, user action is required, the next step is user-gated, the backstop is reached, or the completion promise is satisfied. Commit only when the user asks and stage specific paths only. --max-iterations 10 --completion-promise 'B3 HUMAN-CONDITIONED WORLD MODEL READY FOR USER-GATED H20 TRAINING'

Monitor:   grep '^iteration:' .claude/ralph-loop.local.md
Cancel:    /cancel-ralph   (or remove .claude/ralph-loop.local.md for a deterministic stop)
EOF
