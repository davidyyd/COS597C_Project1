#!/bin/bash
# Start an autoresearch run in the background (survives logout). Arguments go to autoresearch.py start:
#   ./launch.sh --arm vanilla --strategy search --batch 4             # Claude Opus 5.5, effort medium, 400 experiments
#   ./launch.sh --arm free --strategy search --batch 4 --budget 100 --tag free-short
#   ./launch.sh --run 2026-10-12/search-claude-medium-vanilla         # resume a run
# Needs the repository's env.sh set up (Python environment, data), a clean git tree (or --allow-dirty), and a
# Claude login (~/.ar_claude_token from `claude setup-token`, ANTHROPIC_API_KEY, or an existing `claude` login).
# Outputs: $AR_WORK/outputs/<date>/<tag>/ ; launcher log: $AR_WORK/logs/<time>_<args>.log
cd "$(dirname "$0")"
source ../env.sh
mkdir -p "$AR_WORK/logs"
name=$(date +%Y%m%d-%H%M%S)_$(echo "$*" | tr -c 'A-Za-z0-9_.-' '_' | cut -c1-80)
setsid nohup python autoresearch.py start "$@" > "$AR_WORK/logs/$name.log" 2>&1 < /dev/null &
echo "started (pid $!); log: $AR_WORK/logs/$name.log"
