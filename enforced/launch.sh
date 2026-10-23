#!/bin/bash
# Start an autoresearch run in the background (survives logout). Arguments go to autoresearch.py start:
#   ./launch.sh --strategy search --batch 4                        # Claude Opus 5.5, effort medium, 400 experiments
#   ./launch.sh --strategy search --batch 4 --budget 8 --tag try   # a short run
#   ./launch.sh --strategy linear                                  # one experiment per round
#   ./launch.sh --run 2026-10-12/search-claude-medium              # resume a run
# Needs a committed repository (or --allow-dirty), the prepared data, a baseline measured on
# harness/baseline_model.py, and a Claude login (~/.ar_claude_token or ANTHROPIC_API_KEY).
# Outputs: $AR_WORK/outputs/<date>/<tag>/ ; launcher log: $AR_WORK/logs/enforced/<time>_<args>.log
cd "$(dirname "$0")"
source ../env.sh
mkdir -p "$AR_WORK/logs/enforced"
log="$AR_WORK/logs/enforced/$(date +%Y%m%d-%H%M%S)_$(echo "$*" | tr -c 'A-Za-z0-9_.-' '_' | cut -c1-80).log"
setsid nohup python autoresearch.py start "$@" > "$log" 2>&1 < /dev/null &
echo "started (pid $!); log: $log"
