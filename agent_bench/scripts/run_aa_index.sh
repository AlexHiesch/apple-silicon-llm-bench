#!/usr/bin/env bash
# Launch Artificial Analysis Coding Agent Index v1.1 on ThinkingCap.
# Suites: DeepSWE (Pier) + Terminal-Bench v2 + SWE-Atlas-QnA (Harbor), 3 attempts.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PATH="${HOME}/.local/bin:${HOME}/.npm-global/bin:${PATH}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-local}"
export LLM_MODEL="${LLM_MODEL:-t-prazak/ThinkingCap-Qwen3.6-27B-MLX-4bit}"

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
