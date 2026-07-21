#!/usr/bin/env bash
# Stop Harbor/matrix/watchers on this host (x40 or x39).
set -euo pipefail
echo "=== stop_aa_benchmarks $(hostname) $(date) ==="
ROOT="${REPO:-$HOME/Projects/Work/llm-bench}"
mkdir -p "$ROOT/results/agent_bench/aa_index"
touch "$ROOT/results/agent_bench/aa_index/DUAL_NODE_MODE"

for sig in TERM TERM KILL; do
  pkill -"$sig" -f 'harbor (run|job)' 2>/dev/null || true
  pkill -"$sig" -f 'agent_bench.run_matrix' 2>/dev/null || true
  pkill -"$sig" -f 'queue_near_cap_content_fail_retries' 2>/dev/null || true
  pkill -"$sig" -f 'dual_node_tb_' 2>/dev/null || true
  sleep 2
done

for s in aa-ws aa-watch aa-guard aa-night near-cap dual-x40 dual-x39 dual-monitor; do
  tmux kill-session -t "$s" 2>/dev/null || true
done

sleep 3
left=$(pgrep -af 'harbor (run|job)|run_matrix' 2>/dev/null | grep -v pgrep || true)
if [[ -n "$left" ]]; then
  echo "WARN: still running:" >&2
  echo "$left" >&2
  exit 1
fi
echo "OK: benchmarks stopped"
