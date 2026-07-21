#!/usr/bin/env bash
# Watch x40+x39 TB runners: rsync x39 results, merge, restart dead loops, log status.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
X39="${X39_HOST:-hiescha@cmtcdeu89976739.rd.corpintra.net}"
PORT="${SSH_PORT:-42022}"
INTERVAL="${DUAL_MONITOR_SEC:-120}"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$ROOT/results/agent_bench/aa_index/dual_monitor_${STAMP}.log"
STATUS="$ROOT/results/agent_bench/aa_index/DUAL_NODE_STATUS.txt"

mkdir -p "$ROOT/results/agent_bench/aa_index/terminal-bench-v2/claude-code-x39"

log() { echo "[$(date +%Y-%m-%dT%H:%M:%S)] $*" | tee -a "$LOG"; }

log "dual_node_monitor start interval=${INTERVAL}s"

while true; do
  # rsync x39 job results → x40 for merge
  rsync -az \
    -e "ssh -p $PORT" \
    "$X39:~/Projects/Work/llm-bench/results/agent_bench/aa_index/terminal-bench-v2/claude-code-x39/" \
    "$ROOT/results/agent_bench/aa_index/terminal-bench-v2/claude-code-x39/" \
    2>/dev/null || log "WARN rsync x39 failed"

  PY="$ROOT/.venv/bin/python"
  [[ -x "$PY" ]] || PY=python3
  summary=$("$PY" "$ROOT/agent_bench/scripts/dual_node_merge_tb.py" --print 2>/dev/null || echo "merge failed")
  log "MERGE $summary"

  # health checks
  KEY=$(tr -d '[:space:]' < "$HOME/llm-serving/aa-index-key" 2>/dev/null || true)
  x40_code=$(curl -s -o /dev/null -w '%{http_code}' -m 10 -H "Authorization: Bearer $KEY" http://127.0.0.1:4000/v1/models 2>/dev/null || echo 000)
  x39_code=$(ssh -p "$PORT" "$X39" "curl -s -o /dev/null -w '%{http_code}' -m 10 -H 'Authorization: Bearer $KEY' http://127.0.0.1:4000/v1/models" 2>/dev/null || echo 000)

  x40_h=$(pgrep -cf 'harbor (run|job)' 2>/dev/null || echo 0)
  x39_h=$(ssh -p "$PORT" "$X39" "pgrep -cf 'harbor (run|job)' || echo 0" 2>/dev/null || echo 0)

  {
    echo "updated $(date)"
    echo "litellm x40=$x40_code x39=$x39_code"
    echo "harbor procs x40=$x40_h x39=$x39_h"
    echo "$summary"
    echo "tmux: $(tmux ls 2>/dev/null | tr '\n' ' ')"
  } >"$STATUS"

  # restart dead tmux loops (not while harbor active on that node)
  if ! tmux has-session -t dual-x40 2>/dev/null; then
    if [[ "$x40_h" == "0" ]]; then
      log "restart dual-x40 tmux"
      tmux new-session -d -s dual-x40 -c "$ROOT" -- bash agent_bench/scripts/dual_node_tb_loop.sh x40
    fi
  fi
  if ! ssh -p "$PORT" "$X39" "tmux has-session -t dual-x39" 2>/dev/null; then
    if [[ "$x39_h" == "0" ]]; then
      log "restart dual-x39 tmux"
      ssh -p "$PORT" "$X39" "tmux new-session -d -s dual-x39 -c ~/Projects/Work/llm-bench -- bash agent_bench/scripts/dual_node_tb_loop.sh x39"
    fi
  fi

  sleep "$INTERVAL"
done
