#!/usr/bin/env bash
# AA Coding Agent Index — run ON the HP Z8 (no Mac harness).
#
# Inference (k3s llm-serving):
#   LiteLLM hostNetwork :4000
#     → Service vllm-int4 (2× GPU pods, MooncakeStoreConnector cross-pod KV)
# Harness: Harbor/Pier on this host or in Deployment aa-index-runner
#          (docker.sock); trial sandboxes are Docker, not k3s pods
# Concurrency: N_CONCURRENT=2 (one heavy trial per A6000 via LiteLLM LB)
#
# After the first full matrix pass, retries Harbor technical failures
# (timeout / 403 / network / context / UnknownApiError / …) until only
# content failures (reward=0, no exception) remain.
#
# Env (optional):
#   WORKSTATION_API_KEY / HPLLM_KEY_FILE
#   LLM_MODEL                 default thinkingcap
#   N_ATTEMPTS                default 1
#   N_CONCURRENT              default 2
#   AGENT_TIMEOUT_MULT        default 1.0 (official Harbor / TB)
#   CLAUDE_CODE_MAX_OUTPUT_TOKENS  default 8192  (fits dual-replica 65k)
#   MAX_TECH_ROUNDS           default 12
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUNNER_HOME="${RUNNER_HOME:-$HOME/aa-index-runner-home}"
export RUNNER_HOME
export UV_TOOL_DIR="${UV_TOOL_DIR:-$RUNNER_HOME/.local/share/uv/tools}"
export UV_TOOL_BIN_DIR="${UV_TOOL_BIN_DIR:-$RUNNER_HOME/.local/bin}"
export PATH="${UV_TOOL_BIN_DIR}:${HOME}/.local/bin:${HOME}/.npm-global/bin:${PATH}"

# Corp egress — Harbor pulls / agent installs. No sudo/kinit (expire overnight).
# Host process uses localhost; containers get host.docker.internal via
# agent_bench/docker/host-gateway.compose.yaml + run_harbor --ae.
export HTTP_PROXY="${HTTP_PROXY:-http://localhost:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://localhost:3128}"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,.svc,.cluster.local,.corpintra.net,cmtcdeu89976740.rd.corpintra.net}"
export no_proxy="$NO_PROXY"
export HARBOR_HTTP_PROXY="${HARBOR_HTTP_PROXY:-http://host.docker.internal:3128}"

# After claude-code TB 2.0 finishes, set in BENCH_TP2_128K.env (or here):
#   export AA_TB_REMAP_TO_21=1
#   export AA_TB_LEGACY_AGENTS=claude-code
# Then restart aa-ws so remaining agents use TB 2.1.
# Pass-through is automatic via environment (remap_tb_suite in run_matrix).

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

HOST_GATEWAY="${HOST_GATEWAY:-http://127.0.0.1:4000}"
DOCKER_GATEWAY="${DOCKER_GATEWAY:-http://host.docker.internal:4000}"

export LLM_MODEL="${LLM_MODEL:-thinkingcap}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-$HOST_GATEWAY/v1}"
export OPENAI_API_BASE="$OPENAI_BASE_URL"
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-$HOST_GATEWAY}"
export HARBOR_OPENAI_BASE="${HARBOR_OPENAI_BASE:-$DOCKER_GATEWAY/v1}"
export HARBOR_ANTHROPIC_BASE="${HARBOR_ANTHROPIC_BASE:-$DOCKER_GATEWAY}"
export PIER_OPENAI_BASE="${PIER_OPENAI_BASE:-$DOCKER_GATEWAY/v1}"
export PIER_ANTHROPIC_BASE="${PIER_ANTHROPIC_BASE:-$DOCKER_GATEWAY}"
export CLAUDE_CODE_USE_BEDROCK=0
# Dual-replica MAX_MODEL_LEN=65536: keep output budget small so 2 concurrent
# long agent turns still fit (input + max_out < 65k).
export CLAUDE_CODE_MAX_OUTPUT_TOKENS="${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-8192}"
unset AWS_BEARER_TOKEN_BEDROCK ANTHROPIC_BEDROCK_BASE_URL AWS_PROFILE || true

N_ATTEMPTS="${N_ATTEMPTS:-1}"
N_CONCURRENT="${N_CONCURRENT:-2}"
AGENT_TIMEOUT_MULT="${AGENT_TIMEOUT_MULT:-1.0}"
MIN_FREE_GB="${MIN_FREE_GB:-20}"
MAX_TECH_ROUNDS="${MAX_TECH_ROUNDS:-12}"

STATUS="$ROOT/results/agent_bench/aa_index/OVERNIGHT_STATUS.txt"
LOG="$ROOT/results/agent_bench/aa_index/overnight_ws_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$ROOT/results/agent_bench/aa_index"

PYTHON="${ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

status() {
  {
    echo "==== $(date -Iseconds) ===="
    echo "$*"
    echo "host_gateway: $HOST_GATEWAY  docker_gateway: $DOCKER_GATEWAY  model=$LLM_MODEL"
    echo "n_concurrent=$N_CONCURRENT max_out=$CLAUDE_CODE_MAX_OUTPUT_TOKENS"
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
}

count_tech() {
  "$PYTHON" -m agent_bench.tech_failures --root "$ROOT/results/agent_bench/aa_index" 2>/dev/null \
    | awk -F'[= ]' '/^tech=/{print $2; exit}'
}

run_matrix() {
  local label="$1"
  status "matrix pass: $label (n_concurrent=$N_CONCURRENT)"
  # Do NOT pass --skip-unavailable: Harbor/Pier install agents inside
  # containers. Host CLI detection would drop every agent on a clean Z8.
  "$PYTHON" -m agent_bench.run_matrix \
    --matrix \
    --profile aa-index \
    --model "$LLM_MODEL" \
    --n-attempts "$N_ATTEMPTS" \
    --n-concurrent "$N_CONCURRENT" \
    --suite-major \
    --suite-order terminal-bench-v2 swe-atlas-qna deepswe \
    --exclude-deepswe-touched \
    --resume-harbor \
    --harbor-retry-error UnknownApiError \
    --harbor-retry-error AgentTimeoutError \
    --harbor-retry-error CancelledError \
    --harbor-retry-error NetworkConnectionError \
    --harbor-retry-error NonZeroAgentExitCodeError \
    --harbor-retry-error ContextWindowExceededError \
    --harbor-retry-error RateLimitError \
    --harbor-retry-error ApiRateLimitError \
    --harbor-retry-error TimeoutError \
    --harbor-retry-error RuntimeError \
    --harbor-retry-error ValueError \
    --harbor-retry-error AgentSetupTimeoutError \
    --harbor-retry-error EnvironmentStartTimeoutError \
    --harbor-retry-error VerifierTimeoutError \
    --agent-timeout-multiplier "$AGENT_TIMEOUT_MULT" \
    ${TB_FULL_89:+--tb-full-89} \
    ${TB_FORCE_FRESH:+--tb-force-fresh} \
    --min-free-gb "$MIN_FREE_GB" \
    ${AGENT_IDS:+--agent $AGENT_IDS}
}
# NOTE: intentionally NO --docker-prune-between. prune -af deleted SWE-Atlas
# images and caused tech×124. Disk pressure: delete old logs / unused datasets
# via watch_aa_index_workstation.sh instead.

cd "$ROOT"
exec > >(tee -a "$LOG") 2>&1

status "workstation-native overnight start (k=$N_ATTEMPTS n=$N_CONCURRENT timeout_mult=$AGENT_TIMEOUT_MULT host=$(hostname))"
ensure_docker
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

docker_code=$(docker run --rm --add-host=host.docker.internal:host-gateway \
  curlimages/curl:8.5.0 -sS -o /dev/null -w '%{http_code}' -m 45 \
  -H "Authorization: Bearer ${OPENAI_API_KEY}" \
  -H "content-type: application/json" \
  -d "{\"model\":\"${LLM_MODEL}\",\"max_tokens\":4,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" \
  "${DOCKER_GATEWAY}/v1/chat/completions" || true)
echo "docker→host gateway http=$docker_code"
if [[ "$docker_code" != "200" ]]; then
  echo "FATAL: containers cannot reach LiteLLM via $DOCKER_GATEWAY" >&2
  exit 1
fi

if [[ -f "$ROOT/agent_bench/scripts/patch_pier_egress.py" ]]; then
  "$PYTHON" "$ROOT/agent_bench/scripts/patch_pier_egress.py" \
    || python3 "$ROOT/agent_bench/scripts/patch_pier_egress.py" || true
fi
if [[ -f "$ROOT/agent_bench/scripts/patch_harbor_claude_npm.py" ]]; then
  "$PYTHON" "$ROOT/agent_bench/scripts/patch_harbor_claude_npm.py" \
    || python3 "$ROOT/agent_bench/scripts/patch_harbor_claude_npm.py" || true
fi
if [[ -f "$ROOT/agent_bench/scripts/patch_harbor_claude_webfetch.py" ]]; then
  "$PYTHON" "$ROOT/agent_bench/scripts/patch_harbor_claude_webfetch.py" \
    || python3 "$ROOT/agent_bench/scripts/patch_harbor_claude_webfetch.py" || true
fi

# --- Pass 1: full matrix ---
run_matrix "initial"
rc=$?
if [[ "$rc" -ne 0 ]]; then
  status "WARN: matrix pass initial exited rc=$rc (will still scan/retry tech fails)"
fi

# --- Pass 2..N: retry only technical Harbor failures until clean ---
round=1
while (( round <= MAX_TECH_ROUNDS )); do
  "$PYTHON" -m agent_bench.tech_failures --root "$ROOT/results/agent_bench/aa_index" | tee -a "$STATUS"
  tech="$(count_tech)"
  tech="${tech:-0}"
  # Guard: empty result tree after a failed matrix is NOT success.
  trial_n=$("$PYTHON" -m agent_bench.tech_failures --json --root "$ROOT/results/agent_bench/aa_index" \
    2>/dev/null | "$PYTHON" -c "import sys,json; d=json.load(sys.stdin); print(sum(d.get('counts',{}).values()))" || echo 0)
  status "tech-failure check round=$round tech=$tech trials=$trial_n"
  if [[ "${trial_n:-0}" -eq 0 ]]; then
    status "FATAL: no trial results yet — matrix did not produce work; aborting retry loop"
    rc=1
    break
  fi
  if [[ "$tech" -eq 0 ]]; then
    status "OK: zero technical failures — only content pass/fail remain"
    break
  fi
  if (( round == MAX_TECH_ROUNDS )); then
    status "STOP: still $tech technical failures after $MAX_TECH_ROUNDS rounds — see tech_failures report"
    "$PYTHON" -m agent_bench.tech_failures --json --root "$ROOT/results/agent_bench/aa_index" \
      > "$ROOT/results/agent_bench/aa_index/tech_failures_final.json" || true
    rc=2
    break
  fi
  run_matrix "tech-retry-$round"
  rc=$?
  round=$((round + 1))
done

"$PYTHON" -m agent_bench.tech_failures --json --root "$ROOT/results/agent_bench/aa_index" \
  > "$ROOT/results/agent_bench/aa_index/tech_failures_final.json" || true
status "workstation-native overnight finished rc=$rc log=$LOG tech_report=tech_failures_final.json"
exit "$rc"
