#!/usr/bin/env bash
# Overnight AA Coding Agent Index — resume after disk-full / Docker restart.
#
# Strategy:
#   1. Bring ThinkingCap stack up (Kevlar + shim); Kevlar --no-ssd-cache + backoff
#   2. Harbor suites first — resume incomplete jobs (keep finished trials)
#   3. On Harbor resume, retry UnknownApiError trials (422/stream after Kevlar fixes)
#   4. DeepSWE with --exclude-deepswe-touched (keep existing trials; finish rest @k=3)
#   5. Docker prune + disk guard between jobs
#
# Recoverable from last run:
#   - DeepSWE claude-code: 43 tasks × 1 attempt (real LLM for ~33); keep under aa_index/deepswe/
#   - Harbor TB claude-code: ~28/267 trials in terminal-bench-v2__claude-code__20260716_001019
#   - Datasets intact under results/agent_bench/datasets/
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
KEVLAR_ROOT="${KEVLAR_ROOT:-$ROOT/../Kevlar}"
export PATH="${HOME}/.local/bin:${HOME}/.npm-global/bin:${PATH}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-sk-ant-local}"
[[ "$ANTHROPIC_API_KEY" == "local" ]] && export ANTHROPIC_API_KEY="sk-ant-local"
export LLM_MODEL="${LLM_MODEL:-t-prazak/ThinkingCap-Qwen3.6-27B-MLX-4bit}"
unset AWS_BEARER_TOKEN_BEDROCK ANTHROPIC_BEDROCK_BASE_URL AWS_PROFILE || true
export CLAUDE_CODE_USE_BEDROCK=0
export PIER_ANTHROPIC_BASE="${PIER_ANTHROPIC_BASE:-http://host.docker.internal:8080}"
export PIER_OPENAI_BASE="${PIER_OPENAI_BASE:-http://host.docker.internal:8091/v1}"
export HARBOR_ANTHROPIC_BASE="${HARBOR_ANTHROPIC_BASE:-http://host.docker.internal:8080}"
export HARBOR_OPENAI_BASE="${HARBOR_OPENAI_BASE:-http://host.docker.internal:8091/v1}"

MIN_FREE_GB="${MIN_FREE_GB:-40}"
STATUS="$ROOT/results/agent_bench/aa_index/OVERNIGHT_STATUS.txt"
LOG="$ROOT/results/agent_bench/aa_index/overnight_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$ROOT/results/agent_bench/aa_index"

status() {
  {
    echo "==== $(date -Iseconds) ===="
    echo "$*"
    df -h /System/Volumes/Data 2>/dev/null | tail -1 || df -h / | tail -1
    curl -sf -m 2 http://127.0.0.1:8080/health || echo "kevlar: DOWN"
    curl -sf -m 2 http://127.0.0.1:8091/health || echo "shim: DOWN"
    echo
  } | tee -a "$STATUS"
}

wait_health() {
  local url="$1" name="$2" tries="${3:-90}"
  for i in $(seq 1 "$tries"); do
    if curl -sf -m 3 "$url" >/dev/null 2>&1; then
      echo "$name up"
      return 0
    fi
    sleep 5
  done
  echo "FATAL: $name not healthy at $url" >&2
  return 1
}

ensure_docker() {
  if ! docker info >/dev/null 2>&1; then
    echo "FATAL: Docker daemon not running" >&2
    exit 1
  fi
}

ensure_shim() {
  if curl -sf -m 3 http://127.0.0.1:8091/health >/dev/null 2>&1; then
    return 0
  fi
  local sess=thinkingcap-shim
  tmux -f /exec-daemon/tmux.portal.conf has-session -t "=$sess" 2>/dev/null \
    || tmux -f /exec-daemon/tmux.portal.conf new-session -d -s "$sess" -c "$ROOT" -- "${SHELL:-zsh}" -l
  tmux -f /exec-daemon/tmux.portal.conf send-keys -t "$sess:0.0" C-c
  sleep 1
  tmux -f /exec-daemon/tmux.portal.conf send-keys -t "$sess:0.0" \
    "cd '$ROOT' && .venv/bin/python -u -m agent_bench.openai_anthropic_shim --port 8091 --upstream http://127.0.0.1:8080" C-m
}

ensure_kevlar() {
  if curl -sf -m 3 http://127.0.0.1:8080/health >/dev/null 2>&1; then
    return 0
  fi
  local sess=kevlar-thinkingcap
  # Disable SSD KV cache by default: background Metal save raced inference (SIGABRT popups).
  local kevlar_flags="${KEVLAR_FLAGS:---no-ssd-cache}"
  tmux -f /exec-daemon/tmux.portal.conf has-session -t "=$sess" 2>/dev/null \
    || tmux -f /exec-daemon/tmux.portal.conf new-session -d -s "$sess" -c "$KEVLAR_ROOT" -- "${SHELL:-zsh}" -l
  tmux -f /exec-daemon/tmux.portal.conf send-keys -t "$sess:0.0" C-c
  sleep 1
  # Restart with exponential backoff on abort (134); avoid popup spam.
  tmux -f /exec-daemon/tmux.portal.conf send-keys -t "$sess:0.0" \
    "cd '$KEVLAR_ROOT' && backoff=5; while true; do .venv/bin/kevlar serve --port 8080 --model '$LLM_MODEL' $kevlar_flags; rc=\$?; echo \"[watchdog] kevlar exited \$rc at \$(date)\"; if [ \$rc -eq 134 ] || [ \$rc -gt 128 ]; then backoff=\$((backoff < 120 ? backoff * 2 : 120)); else backoff=5; fi; echo \"[watchdog] restart in \${backoff}s\"; sleep \$backoff; done" C-m
}

cd "$ROOT"
exec > >(tee -a "$LOG") 2>&1

status "overnight start"
ensure_docker
ensure_kevlar
ensure_shim
wait_health http://127.0.0.1:8080/health Kevlar 120
wait_health http://127.0.0.1:8091/health shim 30

python3 "$ROOT/agent_bench/scripts/patch_pier_egress.py" \
  || "$ROOT/.venv/bin/python" "$ROOT/agent_bench/scripts/patch_pier_egress.py"

# Drop garbage Harbor result stubs from Docker-down (0.5s exit_1)
python3 - <<'PY'
from pathlib import Path
root = Path("results/agent_bench/aa_index")
moved = 0
trash = root / "_failed_docker_down"
trash.mkdir(exist_ok=True)
for suite in ("terminal-bench-v2", "swe-atlas-qna"):
    sdir = root / suite
    if not sdir.is_dir():
        continue
    for agent in sdir.iterdir():
        if not agent.is_dir():
            continue
        for log in agent.glob("*.log"):
            text = log.read_text(errors="ignore")
            if "Docker daemon is not running" in text:
                dest = trash / f"{suite}__{agent.name}"
                dest.mkdir(parents=True, exist_ok=True)
                for f in agent.iterdir():
                    f.rename(dest / f.name)
                moved += 1
                break
print(f"archived docker-down stubs: {moved}")
PY

status "launching matrix (Harbor first, then DeepSWE resume)"

.venv/bin/python -m agent_bench.run_matrix \
  --matrix \
  --profile aa-index \
  --skip-unavailable \
  --n-concurrent "${N_CONCURRENT:-1}" \
  --suite-major \
  --suite-order terminal-bench-v2 swe-atlas-qna deepswe \
  --exclude-deepswe-touched \
  --resume-harbor \
  --harbor-retry-error UnknownApiError \
  --docker-prune-between \
  --min-free-gb "$MIN_FREE_GB" \
  ${AGENT_IDS:+--agent $AGENT_IDS}

rc=$?
status "overnight finished rc=$rc log=$LOG"
exit "$rc"
