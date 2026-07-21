#!/usr/bin/env bash
# Run assigned TB tasks on this node (x40 or x39). Args: x40|x39
set -euo pipefail
NODE="${1:?usage: dual_node_tb_runner.sh x40|x39}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$HOME/aa-index-runner-home/.local/bin:${PATH:-/usr/bin:/bin}"

SPLIT="$ROOT/results/agent_bench/aa_index/dual_node_tb_split.json"
[[ -f "$SPLIT" ]] || { echo "missing $SPLIT"; exit 1; }

TASKS=$(python3 - <<PY
import json
from pathlib import Path
s=json.loads(Path("$SPLIT").read_text())
print(",".join(s["$NODE"]))
PY
)

export AGENT_TIMEOUT_MULT="${AGENT_TIMEOUT_MULT:-2.5}"
export N_CONCURRENT="${N_CONCURRENT:-2}"
export CLAUDE_CODE_MAX_OUTPUT_TOKENS="${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-16384}"

# Local LiteLLM on each node (hostNetwork :4000)
export HOST_GATEWAY="http://127.0.0.1:4000"
export DOCKER_GATEWAY="http://host.docker.internal:4000"
export OPENAI_BASE_URL="$HOST_GATEWAY/v1"
export OPENAI_API_BASE="$OPENAI_BASE_URL"
export ANTHROPIC_BASE_URL="$HOST_GATEWAY"
export HARBOR_OPENAI_BASE="$DOCKER_GATEWAY/v1"
export HARBOR_ANTHROPIC_BASE="$DOCKER_GATEWAY"

export OPENAI_API_KEY="${OPENAI_API_KEY:-$(tr -d '[:space:]' < "$HOME/llm-serving/aa-index-key")}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-$OPENAI_API_KEY}"
export LLM_MODEL="${LLM_MODEL:-thinkingcap}"

# Corp proxy for image pulls
export HTTP_PROXY="${HTTP_PROXY:-http://localhost:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-$HTTP_PROXY}"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
export HARBOR_HTTP_PROXY="${HARBOR_HTTP_PROXY:-http://host.docker.internal:3128}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,host.docker.internal,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,.svc,.cluster.local,.corpintra.net}"

if [[ "$NODE" == "x39" ]]; then
  # x39 has no local squid; use x40 corp proxy for docker pulls.
  X40_PROXY="${X40_PROXY:-http://cmtcdeu89976740.rd.corpintra.net:3128}"
  export HTTP_PROXY="$X40_PROXY"
  export HTTPS_PROXY="$X40_PROXY"
  export http_proxy="$HTTP_PROXY"
  export https_proxy="$HTTPS_PROXY"
  export HARBOR_HTTP_PROXY="${HARBOR_HTTP_PROXY:-$X40_PROXY}"
fi

if [[ "$NODE" == "x39" ]]; then
  JOBS_SUB="claude-code-x39"
else
  JOBS_SUB="claude-code"
fi
export AA_JOBS_SUB="$JOBS_SUB"

STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$ROOT/results/agent_bench/aa_index/dual_${NODE}_${STAMP}.log"
mkdir -p "$ROOT/results/agent_bench/aa_index"

echo "=== dual_node_tb_runner $NODE tasks=$TASKS mult=$AGENT_TIMEOUT_MULT n=$N_CONCURRENT ===" | tee "$LOG"

PY="${ROOT}/.venv/bin/python"
[[ -x "$PY" ]] || PY=python3

TECH=(
  UnknownApiError AgentTimeoutError CancelledError NetworkConnectionError
  NonZeroAgentExitCodeError ContextWindowExceededError RateLimitError
  ApiRateLimitError TimeoutError RuntimeError ValueError
  AgentSetupTimeoutError EnvironmentStartTimeoutError VerifierTimeoutError
  ApiUsageLimitError
)

if [[ "$NODE" == "x39" ]]; then
  OUT="$ROOT/results/agent_bench/aa_index/terminal-bench-v2/claude-code-x39"
else
  OUT="$ROOT/results/agent_bench/aa_index/terminal-bench-v2/claude-code-x40"
fi
mkdir -p "$OUT"
export AA_JOBS_DIR="$OUT"

FRESH_FLAG=()
if [[ -z "$(find "$OUT" -maxdepth 1 -type d -name 'terminal-bench-v2__claude-code__*' 2>/dev/null | head -1)" ]]; then
  FRESH_FLAG=(--tb-force-fresh)
fi

run_once() {
  local extra=()
  for e in "${TECH[@]}"; do extra+=(--filter-error-type "$e"); done
  "$PY" -m agent_bench.run_harbor \
    --agent claude-code \
    --suite terminal-bench-v2 \
    --n-attempts 1 \
    --n-concurrent "$N_CONCURRENT" \
    --agent-timeout-multiplier "$AGENT_TIMEOUT_MULT" \
    --tb-include-only "$TASKS" \
    --resume \
    "${FRESH_FLAG[@]}" \
    "${extra[@]}"
}

run_once 2>&1 | tee -a "$LOG"
