#!/usr/bin/env bash
# TEMPORARY launcher for AA Index on the Z8 while serving is TP=2 @ 128k.
#
# Serving (temporary):
#   bash ~/llm-serving/k8s/activate-vllm-bench-tp2-128k.sh
# Revert serving to prod TP=1 dual-replica:
#   bash ~/llm-serving/k8s/revert-vllm-prod-tp1.sh
#
# This wrapper only sets runner knobs; it does not change k8s.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_FILE="${AA_BENCH_ENV:-$ROOT/results/agent_bench/aa_index/BENCH_TP2_128K.env}"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi
export N_CONCURRENT="${N_CONCURRENT:-2}"
export CLAUDE_CODE_MAX_OUTPUT_TOKENS="${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-32768}"
export AGENT_TIMEOUT_MULT="${AGENT_TIMEOUT_MULT:-1.5}"
echo "TEMP MODE: N_CONCURRENT=$N_CONCURRENT CLAUDE_CODE_MAX_OUTPUT_TOKENS=$CLAUDE_CODE_MAX_OUTPUT_TOKENS"
exec bash "$ROOT/agent_bench/scripts/run_aa_index_workstation.sh"
