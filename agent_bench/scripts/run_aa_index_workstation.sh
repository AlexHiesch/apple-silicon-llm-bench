#!/usr/bin/env bash
# AA Coding Agent Index against the HP Z8 workstation ThinkingCap gateway.
#
# Inference: LiteLLM on cudbench / cmtcdeu89976740 (:4000) → native vLLM replicas
# Harness:   Harbor/Pier on THIS machine (Mac or workstation), pointing at the gateway.
#
# Env (optional overrides):
#   WORKSTATION_BASE   default https://cudbench.app.corpintra.net
#   OPENAI_API_KEY / ANTHROPIC_API_KEY  (LiteLLM key; falls back to hpllm/.local-ai-key)
#   LLM_MODEL          default thinkingcap
#   N_ATTEMPTS         default 1
#   AGENT_TIMEOUT_MULT default 1.5  (A6000 is much faster than MLX Mac)
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PATH="${HOME}/.local/bin:${HOME}/.npm-global/bin:${PATH}"

HPLLM_KEY_FILE="${HPLLM_KEY_FILE:-$ROOT/../hpllm/.local-ai-key}"
if [[ -z "${OPENAI_API_KEY:-}" && -f "$HPLLM_KEY_FILE" ]]; then
  OPENAI_API_KEY="$(tr -d '[:space:]' < "$HPLLM_KEY_FILE")"
fi
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  ANTHROPIC_API_KEY="${OPENAI_API_KEY:-}"
fi
export OPENAI_API_KEY ANTHROPIC_API_KEY
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "FATAL: set OPENAI_API_KEY or put key in $HPLLM_KEY_FILE" >&2
  exit 1
fi

WORKSTATION_BASE="${WORKSTATION_BASE:-https://cudbench.app.corpintra.net}"
# LiteLLM serves OpenAI at /v1 and Anthropic at /v1/messages (base without /v1 for Anthropic).
export LLM_MODEL="${LLM_MODEL:-thinkingcap}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-$WORKSTATION_BASE/v1}"
export OPENAI_API_BASE="$OPENAI_BASE_URL"
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-$WORKSTATION_BASE}"
export HARBOR_OPENAI_BASE="$OPENAI_BASE_URL"
export HARBOR_ANTHROPIC_BASE="$ANTHROPIC_BASE_URL"
export PIER_OPENAI_BASE="$OPENAI_BASE_URL"
export PIER_ANTHROPIC_BASE="$ANTHROPIC_BASE_URL"
export CLAUDE_CODE_USE_BEDROCK=0
unset AWS_BEARER_TOKEN_BEDROCK ANTHROPIC_BEDROCK_BASE_URL AWS_PROFILE || true

N_ATTEMPTS="${N_ATTEMPTS:-1}"
AGENT_TIMEOUT_MULT="${AGENT_TIMEOUT_MULT:-1.5}"
MIN_FREE_GB="${MIN_FREE_GB:-30}"

STATUS="$ROOT/results/agent_bench/aa_index/OVERNIGHT_STATUS.txt"
LOG="$ROOT/results/agent_bench/aa_index/overnight_ws_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$ROOT/results/agent_bench/aa_index"

status() {
  {
    echo "==== $(date -Iseconds) ===="
    echo "$*"
    echo "gateway: $WORKSTATION_BASE model=$LLM_MODEL"
    curl -sf -m 8 -H "Authorization: Bearer $OPENAI_API_KEY" \
      "$OPENAI_BASE_URL/models" | head -c 200 || echo "gateway: DOWN"
    echo
    df -h /System/Volumes/Data 2>/dev/null | tail -1 || df -h / | tail -1
    echo
  } | tee -a "$STATUS"
}

wait_gateway() {
  local tries="${1:-60}"
  for i in $(seq 1 "$tries"); do
    if curl -sf -m 8 -H "Authorization: Bearer $OPENAI_API_KEY" \
      "$OPENAI_BASE_URL/models" >/dev/null 2>&1; then
      echo "gateway up"
      return 0
    fi
    sleep 5
  done
  echo "FATAL: gateway not healthy at $OPENAI_BASE_URL/models" >&2
  return 1
}

ensure_docker() {
  if ! docker info >/dev/null 2>&1; then
    echo "FATAL: Docker daemon not running" >&2
    exit 1
  fi
}

cd "$ROOT"
exec > >(tee -a "$LOG") 2>&1

status "workstation overnight start (k=$N_ATTEMPTS timeout_mult=$AGENT_TIMEOUT_MULT)"
ensure_docker
wait_gateway 90

# Smoke one cheap completion so we fail fast if vLLM is down behind LiteLLM
python3 - <<PY || { echo "FATAL: gateway completion smoke failed"; exit 1; }
import json, os, urllib.request
url = os.environ["OPENAI_BASE_URL"].rstrip("/") + "/chat/completions"
req = urllib.request.Request(
    url,
    data=json.dumps({
        "model": os.environ["LLM_MODEL"],
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "ping"}],
    }).encode(),
    headers={
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        "Content-Type": "application/json",
    },
    method="POST",
)
with urllib.request.urlopen(req, timeout=120) as r:
    body = r.read().decode()
print("smoke ok:", body[:180])
PY

# Pier Squid may still need local egress patch when Pier runs on Mac
if [[ -f "$ROOT/agent_bench/scripts/patch_pier_egress.py" ]]; then
  python3 "$ROOT/agent_bench/scripts/patch_pier_egress.py" \
    || "$ROOT/.venv/bin/python" "$ROOT/agent_bench/scripts/patch_pier_egress.py" || true
fi

status "launching matrix via workstation ThinkingCap"

.venv/bin/python -m agent_bench.run_matrix \
  --matrix \
  --profile aa-index \
  --model "$LLM_MODEL" \
  --skip-unavailable \
  --n-attempts "$N_ATTEMPTS" \
  --n-concurrent "${N_CONCURRENT:-1}" \
  --suite-major \
  --suite-order terminal-bench-v2 swe-atlas-qna deepswe \
  --exclude-deepswe-touched \
  --resume-harbor \
  --harbor-retry-error UnknownApiError \
  --harbor-retry-error AgentTimeoutError \
  --harbor-retry-error CancelledError \
  --harbor-retry-error NetworkConnectionError \
  --agent-timeout-multiplier "$AGENT_TIMEOUT_MULT" \
  --docker-prune-between \
  --min-free-gb "$MIN_FREE_GB" \
  ${AGENT_IDS:+--agent $AGENT_IDS}

rc=$?
status "workstation overnight finished rc=$rc log=$LOG"
exit "$rc"
