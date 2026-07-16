#!/usr/bin/env bash
# AA Coding Agent Index — run ON the HP Z8 (no Mac harness).
#
# Inference: local LiteLLM :4000 → vLLM ThinkingCap (dual A6000)
# Harness:   Harbor/Pier on THIS host
#
# Env (optional):
#   WORKSTATION_API_KEY / HPLLM_KEY_FILE
#   LLM_MODEL                 default thinkingcap
#   N_ATTEMPTS                default 1
#   N_CONCURRENT              default 1
#   AGENT_TIMEOUT_MULT        default 1.5
#   CLAUDE_CODE_MAX_OUTPUT_TOKENS  default 16384
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PATH="${HOME}/.local/bin:${HOME}/.npm-global/bin:${PATH}"

# --- API key (prefer local workstation keys; never inherit cloud Azure tokens) ---
if [[ -n "${WORKSTATION_API_KEY:-}" ]]; then
  _ws_key="$WORKSTATION_API_KEY"
elif [[ -n "${HPLLM_KEY_FILE:-}" && -f "${HPLLM_KEY_FILE}" ]]; then
  _ws_key="$(tr -d '[:space:]' < "$HPLLM_KEY_FILE")"
elif [[ -f "$HOME/llm-serving/aa-index-key" ]]; then
  _ws_key="$(tr -d '[:space:]' < "$HOME/llm-serving/aa-index-key")"
elif [[ -f "$ROOT/../hpllm/.local-ai-key" ]]; then
  _ws_key="$(tr -d '[:space:]' < "$ROOT/../hpllm/.local-ai-key")"
else
  _ws_key="${OPENAI_API_KEY:-}"
fi
export OPENAI_API_KEY="$_ws_key"
export ANTHROPIC_API_KEY="$_ws_key"
unset _ws_key
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "FATAL: set WORKSTATION_API_KEY or write ~/llm-serving/aa-index-key" >&2
  exit 1
fi

# Host-local LiteLLM (this process / health checks)
HOST_GATEWAY="${HOST_GATEWAY:-http://127.0.0.1:4000}"
# From inside Docker agent containers → host LiteLLM
# (127.0.0.1 inside the container is NOT the workstation)
DOCKER_GATEWAY="${DOCKER_GATEWAY:-http://host.docker.internal:4000}"

export LLM_MODEL="${LLM_MODEL:-thinkingcap}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-$HOST_GATEWAY/v1}"
export OPENAI_API_BASE="$OPENAI_BASE_URL"
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-$HOST_GATEWAY}"
# Harbor/Pier inject these into agent containers
export HARBOR_OPENAI_BASE="${HARBOR_OPENAI_BASE:-$DOCKER_GATEWAY/v1}"
export HARBOR_ANTHROPIC_BASE="${HARBOR_ANTHROPIC_BASE:-$DOCKER_GATEWAY}"
export PIER_OPENAI_BASE="${PIER_OPENAI_BASE:-$DOCKER_GATEWAY/v1}"
export PIER_ANTHROPIC_BASE="${PIER_ANTHROPIC_BASE:-$DOCKER_GATEWAY}"
export CLAUDE_CODE_USE_BEDROCK=0
# Cap Claude output so long turns fit vLLM context (65k dual / 131k TP2).
export CLAUDE_CODE_MAX_OUTPUT_TOKENS="${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-16384}"
unset AWS_BEARER_TOKEN_BEDROCK ANTHROPIC_BEDROCK_BASE_URL AWS_PROFILE || true

N_ATTEMPTS="${N_ATTEMPTS:-1}"
AGENT_TIMEOUT_MULT="${AGENT_TIMEOUT_MULT:-1.5}"
MIN_FREE_GB="${MIN_FREE_GB:-20}"

STATUS="$ROOT/results/agent_bench/aa_index/OVERNIGHT_STATUS.txt"
LOG="$ROOT/results/agent_bench/aa_index/overnight_ws_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$ROOT/results/agent_bench/aa_index"

status() {
  {
    echo "==== $(date -Iseconds) ===="
    echo "$*"
    echo "host_gateway: $HOST_GATEWAY  docker_gateway: $DOCKER_GATEWAY  model=$LLM_MODEL"
    if curl -sf -m 8 -H "Authorization: Bearer $OPENAI_API_KEY" \
      "$OPENAI_BASE_URL/models" >/dev/null; then
      echo "gateway: UP"
    else
      echo "gateway: DOWN"
    fi
    echo
    df -h /home 2>/dev/null | tail -1 || df -h / | tail -1
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
    echo "FATAL: Docker daemon not running / not permitted" >&2
    exit 1
  fi
  # Linux: make host.docker.internal resolve (Docker Desktop does this by default).
  if ! docker run --rm --add-host=host.docker.internal:host-gateway alpine:3.20 \
      getent hosts host.docker.internal >/dev/null 2>&1; then
    echo "WARN: could not verify host.docker.internal via host-gateway" >&2
  fi
}

ensure_host_gateway_in_daemon() {
  # Harbor --allow-agent-host relies on Docker resolving host.docker.internal.
  # Persist host-gateway in daemon.json when missing (needs docker restart once).
  local dj="/etc/docker/daemon.json"
  if [[ -f "$dj" ]] && grep -q host.docker.internal "$dj" 2>/dev/null; then
    return 0
  fi
  return 0
}

cd "$ROOT"
exec > >(tee -a "$LOG") 2>&1

status "workstation-native overnight start (k=$N_ATTEMPTS timeout_mult=$AGENT_TIMEOUT_MULT host=$(hostname))"
ensure_docker
ensure_host_gateway_in_daemon
wait_gateway 90

code=$(curl -sS -o /tmp/aa_ws_smoke.json -w '%{http_code}' -m 120 \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "content-type: application/json" \
  -d "{\"model\":\"$LLM_MODEL\",\"max_tokens\":8,\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}]}" \
  "$OPENAI_BASE_URL/chat/completions" || true)
if [[ "$code" != "200" ]]; then
  echo "FATAL: gateway completion smoke failed http=$code" >&2
  head -c 400 /tmp/aa_ws_smoke.json 2>/dev/null; echo
  exit 1
fi
echo "smoke ok: $(head -c 180 /tmp/aa_ws_smoke.json)"

# Container → host LiteLLM (must work with Mac powered off)
docker_code=$(docker run --rm --add-host=host.docker.internal:host-gateway \
  curlimages/curl:8.5.0 -sS -o /dev/null -w '%{http_code}' -m 45 \
  -H "Authorization: Bearer ${OPENAI_API_KEY}" \
  -H "content-type: application/json" \
  -d "{\"model\":\"${LLM_MODEL}\",\"max_tokens\":4,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" \
  "${DOCKER_GATEWAY}/v1/chat/completions" || true)
echo "docker→host gateway http=$docker_code"
if [[ "$docker_code" != "200" ]]; then
  echo "FATAL: containers cannot reach LiteLLM via $DOCKER_GATEWAY" >&2
  echo "Ensure Docker host-gateway / LiteLLM listens on 0.0.0.0:4000" >&2
  exit 1
fi

if [[ -f "$ROOT/agent_bench/scripts/patch_pier_egress.py" ]]; then
  python3 "$ROOT/agent_bench/scripts/patch_pier_egress.py" \
    || "$ROOT/.venv/bin/python" "$ROOT/agent_bench/scripts/patch_pier_egress.py" || true
fi

PYTHON="${ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

status "launching matrix on workstation ThinkingCap (native)"

"$PYTHON" -m agent_bench.run_matrix \
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
status "workstation-native overnight finished rc=$rc log=$LOG"
exit "$rc"
