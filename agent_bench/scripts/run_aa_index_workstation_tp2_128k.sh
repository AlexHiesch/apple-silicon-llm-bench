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
# Dual-node: N=4 (~2/replica) was OK until TQ+MTP+enforce-eager slowed decode
# and tech timeouts piled up (api_retry). N=2 for tech-retry / overnight stability.
# N=8 historically = timeout storm.
export N_CONCURRENT="${N_CONCURRENT:-2}"
# 16k out leaves ~112k input under 128k context (32k out overflowed at ~98k input).
export CLAUDE_CODE_MAX_OUTPUT_TOKENS="${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-16384}"
# Official Harbor / Terminal Bench default (no --agent-timeout-multiplier).
export AGENT_TIMEOUT_MULT="${AGENT_TIMEOUT_MULT:-1.0}"
export TB_FULL_89="${TB_FULL_89:-0}"
echo "TEMP MODE: N_CONCURRENT=$N_CONCURRENT CLAUDE_CODE_MAX_OUTPUT_TOKENS=$CLAUDE_CODE_MAX_OUTPUT_TOKENS timeout_mult=$AGENT_TIMEOUT_MULT tb_full_89=$TB_FULL_89"
exec bash "$ROOT/agent_bench/scripts/run_aa_index_workstation.sh"
