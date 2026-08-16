#!/usr/bin/env bash
# Mark near-cap content_fails and start a 2.5× include-only Harbor job when idle.
# Safe to run while the gaps job is still finishing — waits for harbor to exit.
set -euo pipefail
REPO="${REPO:-$HOME/Projects/Work/llm-bench}"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$HOME/aa-index-runner-home/.local/bin:${PATH:-/usr/bin:/bin}"
export AGENT_TIMEOUT_MULT="${AGENT_TIMEOUT_MULT:-2.5}"
export N_CONCURRENT="${N_CONCURRENT:-2}"

# Workstation gateway (same as start_tb_full89_claude.sh)
export OPENAI_API_KEY="${OPENAI_API_KEY:-$(cat "$HOME/llm-serving/aa-index-key" 2>/dev/null || true)}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-$OPENAI_API_KEY}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:4000/v1}"
export OPENAI_API_BASE="$OPENAI_BASE_URL"
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-http://127.0.0.1:4000}"
export HARBOR_OPENAI_BASE="${HARBOR_OPENAI_BASE:-http://host.docker.internal:4000/v1}"
export HARBOR_ANTHROPIC_BASE="${HARBOR_ANTHROPIC_BASE:-http://host.docker.internal:4000}"
export LLM_MODEL="${LLM_MODEL:-thinkingcap}"

STAMP=$(date +%Y%m%d_%H%M%S)
LOG="results/agent_bench/aa_index/near_cap_retries_${STAMP}.log"
mkdir -p results/agent_bench/aa_index

echo "=== near-cap retries @ ${AGENT_TIMEOUT_MULT}x → $LOG ==="
# Prefer venv python when present.
PY="${REPO}/.venv/bin/python"
[[ -x "$PY" ]] || PY=python3

"$PY" agent_bench/scripts/queue_near_cap_content_fail_retries.py \
  --apply \
  --start \
  --mult "$AGENT_TIMEOUT_MULT" \
  --n-concurrent "$N_CONCURRENT" \
  2>&1 | tee -a "$LOG"
