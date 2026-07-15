#!/usr/bin/env bash
# Launch Artificial Analysis Coding Agent Index v1.1 on ThinkingCap.
# Suites: DeepSWE (Pier) + Terminal-Bench v2 + SWE-Atlas-QnA (Harbor), 3 attempts.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PATH="${HOME}/.local/bin:${HOME}/.npm-global/bin:${PATH}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local}"
# Claude Code requires sk-ant-* prefix; Kevlar accepts any key.
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-sk-ant-local}"
if [[ "$ANTHROPIC_API_KEY" == "local" ]]; then
  export ANTHROPIC_API_KEY="sk-ant-local"
fi
export LLM_MODEL="${LLM_MODEL:-t-prazak/ThinkingCap-Qwen3.6-27B-MLX-4bit}"
# Host Bedrock tokens make Pier/Claude Code skip local ANTHROPIC_BASE_URL.
unset AWS_BEARER_TOKEN_BEDROCK ANTHROPIC_BEDROCK_BASE_URL AWS_PROFILE || true
export CLAUDE_CODE_USE_BEDROCK=0
export PIER_ANTHROPIC_BASE="${PIER_ANTHROPIC_BASE:-http://host.docker.internal:8080}"
export PIER_OPENAI_BASE="${PIER_OPENAI_BASE:-http://host.docker.internal:8091/v1}"
export HARBOR_ANTHROPIC_BASE="${HARBOR_ANTHROPIC_BASE:-http://host.docker.internal:8080}"
export HARBOR_OPENAI_BASE="${HARBOR_OPENAI_BASE:-http://host.docker.internal:8091/v1}"

cd "$ROOT"
mkdir -p results/agent_bench/aa_index

if ! curl -sf --max-time 5 http://127.0.0.1:8080/health >/dev/null; then
  echo "FATAL: Kevlar not up on :8080" >&2
  exit 1
fi
if ! curl -sf --max-time 5 http://127.0.0.1:8091/health >/dev/null; then
  echo "FATAL: OpenAI shim not up on :8091" >&2
  exit 1
fi
# Pier Squid: allow :8080/:8091 + prefer IPv4 for host.docker.internal
"$ROOT/.venv/bin/python" "$ROOT/agent_bench/scripts/patch_pier_egress.py" \
  || python3.12 "$ROOT/agent_bench/scripts/patch_pier_egress.py"
command -v pier >/dev/null || { echo "FATAL: pier missing (uv tool install --python 3.12 datacurve-pier)" >&2; exit 1; }
command -v harbor >/dev/null || { echo "FATAL: harbor missing (uv tool install --python 3.12 harbor)" >&2; exit 1; }

LOG="results/agent_bench/aa_index/run_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG"
exec > >(tee -a "$LOG") 2>&1

.venv/bin/python -m agent_bench.run_matrix \
  --matrix \
  --profile aa-index \
  --skip-unavailable \
  --n-concurrent "${N_CONCURRENT:-1}" \
  ${AGENT_IDS:+--agent $AGENT_IDS}
