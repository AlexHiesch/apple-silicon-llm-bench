#!/usr/bin/env bash
# Flip AA Index TB suite remap to 2.1 for all agents except legacy (claude-code).
# Safe to run only AFTER the live claude-code × terminal-bench-v2 job has finished
# (or you accept abandoning in-flight TB 2.0 trials for other agents).
#
# Usage (on x40):
#   bash agent_bench/scripts/enable_tb21_after_claude.sh
#   # then restart aa-ws (this script can do it with RESTART_AA_WS=1)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_FILE="${AA_BENCH_ENV:-$ROOT/results/agent_bench/aa_index/BENCH_TP2_128K.env}"
mkdir -p "$(dirname "$ENV_FILE")"
touch "$ENV_FILE"

set_kv() {
  local key="$1" val="$2"
  if grep -q "^export ${key}=" "$ENV_FILE" 2>/dev/null; then
    # portable in-place replace
    local tmp
    tmp="$(mktemp)"
    sed "s|^export ${key}=.*|export ${key}=${val}|" "$ENV_FILE" >"$tmp"
    mv "$tmp" "$ENV_FILE"
  else
    printf 'export %s=%s\n' "$key" "$val" >>"$ENV_FILE"
  fi
}

set_kv AA_TB_REMAP_TO_21 1
set_kv AA_TB_LEGACY_AGENTS claude-code

echo "Wrote remap knobs to $ENV_FILE:"
grep -E 'AA_TB_' "$ENV_FILE" || true

# Optional sanity: plan-only must show remaps
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  echo "Plan-only check (first 40 lines)…"
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  AA_TB_REMAP_TO_21=1 AA_TB_LEGACY_AGENTS=claude-code \
    "$ROOT/.venv/bin/python" -m agent_bench.run_matrix --matrix --profile aa-index \
    --model thinkingcap --suite-major \
    --suite-order terminal-bench-v2 swe-atlas-qna deepswe --plan-only 2>&1 | head -40
fi

if [[ "${RESTART_AA_WS:-0}" == "1" ]]; then
  echo "Restarting tmux aa-ws…"
  tmux has-session -t aa-ws 2>/dev/null && tmux kill-session -t aa-ws || true
  tmux new-session -d -s aa-ws -c "$ROOT" -- \
    bash agent_bench/scripts/run_aa_index_workstation_tp2_128k.sh
  tmux has-session -t aa-ws 2>/dev/null && echo "aa-ws restarted" || echo "FATAL: aa-ws failed"
else
  echo "Remap enabled in env file. Restart aa-ws when ready:"
  echo "  RESTART_AA_WS=1 bash agent_bench/scripts/enable_tb21_after_claude.sh"
  echo "  # or: tmux kill-session -t aa-ws; tmux new-session -d -s aa-ws -c $ROOT -- bash agent_bench/scripts/run_aa_index_workstation_tp2_128k.sh"
fi
