#!/usr/bin/env bash
# Print the Ralph-loop launch prompt for the active B3-CS queue. This script
# performs no implementation or external operation.
set -euo pipefail

cat <<'EOF'
Start the Ralph loop from a Claude Code session in this repo with:

  /ralph-loop Continue vllatent-ego-drone B3-CS. Follow .codex/ralph-rules.md. READ, in order: AGENTS.md; .codex/ralph-rules.md; DEV_LOG.md; plans/phase-b3-causal-ski-sim-latent-decoder.md; then only files named by the active card. Do not consult Obsidian. Current verified state: CS1/CS2 complete 2026-07-15; CS3 complete 2026-07-20; CS4 is lowest pending. STOP before CS4 mutation unless both gates are satisfied: explicit USER authority for the frozen 32-root x nine-branch x eight-future CPU data-generation smoke, and restoration or reviewed complete migration of the absent normative CS4+ report clauses. Never invent missing equations/constants/thresholds. B3.6 remains blocked and B3.7/H20 remains ineligible. --max-iterations 10 --completion-promise 'B3-CS4 PAIRED CAUSAL SMOKE COMPLETE'

Monitor: grep '^iteration:' .claude/ralph-loop.local.md
Cancel:  /cancel-ralph
EOF
