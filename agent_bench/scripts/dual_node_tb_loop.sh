#!/usr/bin/env bash
# Resume loop for one node's TB partition until tech failures exhausted or idle.
set -euo pipefail
NODE="${1:?x40|x39}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$ROOT/results/agent_bench/aa_index/dual_${NODE}_loop_${STAMP}.log"
MAX_ROUNDS="${MAX_TECH_ROUNDS:-24}"
ROUND=0

echo "dual_node_tb_loop $NODE start $(date)" | tee "$LOG"

while (( ROUND < MAX_ROUNDS )); do
  ROUND=$((ROUND + 1))
  echo "=== round $ROUND $(date) ===" | tee -a "$LOG"
  if ! bash "$ROOT/agent_bench/scripts/dual_node_tb_runner.sh" "$NODE" >>"$LOG" 2>&1; then
    echo "runner exit nonzero round $ROUND" | tee -a "$LOG"
  fi
  sleep 30
  # stop if no harbor running and no pending tech in this node's jobs dir
  if ! pgrep -f "harbor (run|job)" >/dev/null 2>&1; then
    PY="$ROOT/.venv/bin/python"
    [[ -x "$PY" ]] || PY=python3
    if [[ "$NODE" == "x39" ]]; then
      OUT="$ROOT/results/agent_bench/aa_index/terminal-bench-v2/claude-code-x39"
    else
      OUT="$ROOT/results/agent_bench/aa_index/terminal-bench-v2/claude-code-x40"
    fi
    tech=$("$PY" -c "
import json
from pathlib import Path
from agent_bench.tech_failures import classify_result
out = Path('$OUT')
n = 0
if out.is_dir():
    for j in out.iterdir():
        if not j.is_dir(): continue
        for t in j.iterdir():
            rj = t / 'result.json'
            if not rj.is_file(): continue
            r = json.loads(rj.read_text())
            if classify_result(r) == 'tech':
                n += 1
print(n)
" 2>/dev/null || echo 1)
    if [[ "${tech:-1}" == "0" ]]; then
      echo "no tech left on $NODE — loop idle exit" | tee -a "$LOG"
      break
    fi
  fi
  sleep 60
done

echo "dual_node_tb_loop $NODE done rounds=$ROUND $(date)" | tee -a "$LOG"
