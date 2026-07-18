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
# Dual-node default is 4 (~2 per TP2 replica). N=8 saturated GPUs and spiked
# AgentTimeouts; keep 4 unless A/B shows clean/h gains.
export N_CONCURRENT="${N_CONCURRENT:-4}"
# 16k out leaves ~112k input under 128k context (32k out overflowed at ~98k input).
export CLAUDE_CODE_MAX_OUTPUT_TOKENS="${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-16384}"
# 1.5x is the floor that kept all observed TB passes under the agent cap
# (1.25x would have AgentTimeout'd 2 real passes). AgentTimeoutError stays
# tech → resume_until_content; never scored as content_fail.
export AGENT_TIMEOUT_MULT="${AGENT_TIMEOUT_MULT:-1.5}"
echo "TEMP MODE: N_CONCURRENT=$N_CONCURRENT CLAUDE_CODE_MAX_OUTPUT_TOKENS=$CLAUDE_CODE_MAX_OUTPUT_TOKENS timeout_mult=$AGENT_TIMEOUT_MULT"
exec bash "$ROOT/agent_bench/scripts/run_aa_index_workstation.sh"
