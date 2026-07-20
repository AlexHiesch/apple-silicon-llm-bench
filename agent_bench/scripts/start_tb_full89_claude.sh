#!/usr/bin/env bash
# Start Terminal Bench 2.0 full 89 for claude-code, short tasks first.
# Archives the active partial job (47-cohort) if present.
set -euo pipefail
REPO="${REPO:-$HOME/Projects/Work/llm-bench}"
OUT="$REPO/results/agent_bench/aa_index/terminal-bench-v2/claude-code"
STAMP=$(date +%Y%m%d_%H%M%S)
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$HOME/aa-index-runner-home/.local/bin:${PATH:-/usr/bin:/bin}"

# Workstation gateway (same as run_aa_index_workstation.sh)
export OPENAI_API_KEY="${OPENAI_API_KEY:-$(cat "$HOME/llm-serving/aa-index-key" 2>/dev/null || true)}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-$OPENAI_API_KEY}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:4000/v1}"
export OPENAI_API_BASE="$OPENAI_BASE_URL"
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-http://127.0.0.1:4000}"
export HARBOR_OPENAI_BASE="${HARBOR_OPENAI_BASE:-http://host.docker.internal:4000/v1}"
export HARBOR_ANTHROPIC_BASE="${HARBOR_ANTHROPIC_BASE:-http://host.docker.internal:4000}"
export LLM_MODEL="${LLM_MODEL:-thinkingcap}"

cd "$REPO"

# Stop in-flight harbor/matrix briefly.
pkill -TERM -f 'agent_bench.run_matrix' 2>/dev/null || true
sleep 5
pkill -9 -f 'agent_bench.run_matrix' 2>/dev/null || true
pkill -TERM -f 'harbor job resume' 2>/dev/null || true
sleep 3

for job in "$OUT"/terminal-bench-v2__claude-code__*; do
  [[ -d "$job" ]] || continue
  [[ -f "$job/config.json" ]] || continue
  if grep -q '"tb_full_ordered": true' "$job/config.json" 2>/dev/null; then
    echo "keep full-89 job: $(basename "$job")"
    continue
  fi
  dest="$OUT/_archived_partial_${STAMP}_$(basename "$job")"
  echo "archive partial job -> $dest"
  mv "$job" "$dest"
done

export TB_FULL_89=1
export TB_FORCE_FRESH=1
export AGENT_TIMEOUT_MULT="${AGENT_TIMEOUT_MULT:-1.0}"

echo "=== task order (first 20) ==="
.venv/bin/python - <<'PY'
from pathlib import Path
from agent_bench.run_harbor import tb_full_ordered_include_names, SUITE_DATASETS

out = Path("results/agent_bench/aa_index/terminal-bench-v2/claude-code")
ds = Path(SUITE_DATASETS["terminal-bench-v2"]["local"])
tasks = tb_full_ordered_include_names(out, ds)
print(f"to_run={len(tasks)}")
for t in tasks[:20]:
    print(" ", t)
if len(tasks) > 20:
    print(f"  ... +{len(tasks)-20} more")
PY

echo "=== harbor start ==="
.venv/bin/python -m agent_bench.run_harbor \
  --agent claude-code \
  --suite terminal-bench-v2 \
  --n-attempts 1 \
  --n-concurrent "${N_CONCURRENT:-2}" \
  --agent-timeout-multiplier "$AGENT_TIMEOUT_MULT" \
  --tb-full-ordered \
  --tb-force-fresh

echo "=== restart aa-ws ==="
tmux kill-session -t aa-ws 2>/dev/null || true
sleep 1
tmux new-session -d -s aa-ws -c "$REPO" -- bash -lc \
  "export TB_FULL_89=1; export AGENT_TIMEOUT_MULT=$AGENT_TIMEOUT_MULT; bash agent_bench/scripts/run_aa_index_workstation_tp2_128k.sh 2>&1 | tee -a results/agent_bench/aa_index/overnight_ws_full89_${STAMP}.log"
