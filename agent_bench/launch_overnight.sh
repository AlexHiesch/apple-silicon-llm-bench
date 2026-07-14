#!/usr/bin/env bash
# Launch full overnight ThinkingCap harness for blog results.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p results/agent_bench/overnight logs

STAMP=$(date +%Y%m%d_%H%M%S)
LOG="results/agent_bench/overnight/overnight_${STAMP}.log"
STATUS="results/agent_bench/overnight/STATUS.txt"
MODEL="t-prazak/ThinkingCap-Qwen3.6-27B-MLX-4bit"
KEVLAR_PORT=8080
SHIM_PORT=8091
KEVLAR_BASE="http://127.0.0.1:${KEVLAR_PORT}"
SHIM_BASE="http://127.0.0.1:${SHIM_PORT}"
CLAUDE_SETTINGS="/Users/HIESCHA/.claude/settings.qwen36.json"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

echo "started=$(date -Iseconds)" > "$STATUS"
echo "log=$LOG" >> "$STATUS"
echo "model=$MODEL" >> "$STATUS"
echo "kevlar_port=$KEVLAR_PORT" >> "$STATUS"
echo "shim_port=$SHIM_PORT" >> "$STATUS"

PY="$ROOT/.venv/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python3)"

kevlar_healthy() {
  if curl -sf --max-time 20 "$KEVLAR_BASE/v1/status" 2>/dev/null | grep -q '"model_loaded":true'; then
    return 0
  fi
  pgrep -f "kevlar serve" >/dev/null 2>&1 && lsof -tiTCP:"$KEVLAR_PORT" -sTCP:LISTEN >/dev/null 2>&1
}

shim_healthy() {
  curl -sf --max-time 3 "$SHIM_BASE/health" >/dev/null 2>&1
}

ensure_kevlar() {
  if kevlar_healthy; then
    log "Kevlar ThinkingCap on :$KEVLAR_PORT"
    return 0
  fi
  log "Starting Kevlar on :$KEVLAR_PORT ..."
  if command -v tmux >/dev/null 2>&1; then
    tmux -f /exec-daemon/tmux.portal.conf has-session -t thinkingcap-kevlar 2>/dev/null \
      || tmux -f /exec-daemon/tmux.portal.conf new-session -d -s thinkingcap-kevlar \
        "caffeinate -ims kevlar serve --model '$MODEL' --port $KEVLAR_PORT 2>&1 | tee /tmp/llm-model.log"
  else
    nohup caffeinate -ims kevlar serve --model "$MODEL" --port "$KEVLAR_PORT" \
      > /tmp/llm-model.log 2>&1 &
  fi
}

wait_kevlar() {
  local deadline=$((SECONDS + 3600))
  log "Waiting up to 60m for Kevlar $KEVLAR_BASE/v1/status ..."
  while (( SECONDS < deadline )); do
    if kevlar_healthy; then
      log "Kevlar healthy."
      curl -s --max-time 5 "$KEVLAR_BASE/v1/status" | head -c 400 >> "$LOG" || true
      echo >> "$LOG"
      return 0
    fi
    if ! pgrep -f "kevlar serve" >/dev/null 2>&1; then
      log "WARNING: kevlar not running — ensure_kevlar"
      ensure_kevlar
    fi
    sleep 15
  done
  log "FATAL: Kevlar never became healthy"
  echo "status=failed_kevlar" >> "$STATUS"
  return 1
}

ensure_shim() {
  if shim_healthy; then
    log "OpenAI shim already on :$SHIM_PORT"
    return 0
  fi
  log "Starting OpenAI→Anthropic shim on :$SHIM_PORT ..."
  if command -v tmux >/dev/null 2>&1; then
    tmux -f /exec-daemon/tmux.portal.conf kill-session -t thinkingcap-shim 2>/dev/null || true
    tmux -f /exec-daemon/tmux.portal.conf new-session -d -s thinkingcap-shim \
      "cd '$ROOT' && PYTHONPATH='$ROOT' caffeinate -i $PY -m agent_bench.openai_anthropic_shim \
        --port $SHIM_PORT --upstream $KEVLAR_BASE --model '$MODEL' \
        2>&1 | tee results/agent_bench/overnight/logs/shim_tmux.log"
  else
    mkdir -p results/agent_bench/overnight/logs
    PYTHONPATH="$ROOT" nohup "$PY" -m agent_bench.openai_anthropic_shim \
      --port "$SHIM_PORT" --upstream "$KEVLAR_BASE" --model "$MODEL" \
      > results/agent_bench/overnight/logs/shim_nohup.log 2>&1 &
  fi
  local deadline=$((SECONDS + 60))
  while (( SECONDS < deadline )); do
    if shim_healthy; then
      log "Shim healthy on :$SHIM_PORT"
      return 0
    fi
    sleep 1
  done
  log "FATAL: shim failed to start"
  echo "status=failed_shim" >> "$STATUS"
  return 1
}

smoke_tests() {
  log "Smoke Anthropic /v1/messages (Kevlar) ..."
  local code
  code=$(curl -s -o /tmp/tc_smoke.json -w "%{http_code}" --max-time 180 \
    -X POST "$KEVLAR_BASE/v1/messages" \
    -H "content-type: application/json" \
    -H "x-api-key: local" \
    -H "anthropic-version: 2023-06-01" \
    -d '{"model":"'"$MODEL"'","max_tokens":32,"messages":[{"role":"user","content":"Reply with exactly: OK"}]}' || echo 000)
  log "Anthropic smoke HTTP $code"
  head -c 300 /tmp/tc_smoke.json >> "$LOG" 2>/dev/null || true
  echo >> "$LOG"

  log "Smoke OpenAI /v1/chat/completions via shim :$SHIM_PORT ..."
  code=$(curl -s -o /tmp/tc_smoke_oai.json -w "%{http_code}" --max-time 180 \
    -X POST "$SHIM_BASE/v1/chat/completions" \
    -H "content-type: application/json" \
    -H "authorization: Bearer local" \
    -d '{"model":"'"$MODEL"'","max_tokens":32,"messages":[{"role":"user","content":"Reply with exactly: OK"}]}' || echo 000)
  log "Shim OpenAI smoke HTTP $code"
  head -c 300 /tmp/tc_smoke_oai.json >> "$LOG" 2>/dev/null || true
  echo >> "$LOG"

  if [[ -x "$(command -v claude)" && -f "$CLAUDE_SETTINGS" ]]; then
    log "Smoke claude --settings (45s cap) ..."
    timeout 45 claude -p "Reply with exactly: OK" \
      --settings "$CLAUDE_SETTINGS" \
      --dangerously-skip-permissions 2>&1 | head -c 200 >> "$LOG" || log "Claude smoke timed out or failed (non-fatal)"
    echo >> "$LOG"
  fi
}

# Kill stale orchestrator / goose only — never kevlar (tmux session managed externally)
pkill -f "run_overnight_full.py" 2>/dev/null || true
pkill -f "goose run" 2>/dev/null || true

ensure_kevlar
wait_kevlar || exit 1
ensure_shim || exit 1
smoke_tests

log "==== Starting overnight orchestrator ===="
echo "status=running" >> "$STATUS"
echo "orchestrator_start=$(date -Iseconds)" >> "$STATUS"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local}"
export OPENAI_BASE_URL="$SHIM_BASE/v1"
export OPENAI_API_BASE="$SHIM_BASE/v1"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-local}"
export ANTHROPIC_BASE_URL="$KEVLAR_BASE"
export ANTHROPIC_MODEL="$MODEL"
export CLAUDE_CODE_USE_BEDROCK=0
export CLAUDE_CODE_SKIP_BEDROCK_AUTH=1

caffeinate -i "$PY" -u "$ROOT/agent_bench/run_overnight_full.py" 2>&1 | tee -a "$LOG"
EC=${PIPESTATUS[0]}

echo "orchestrator_end=$(date -Iseconds)" >> "$STATUS"
echo "exit_code=$EC" >> "$STATUS"
if [[ $EC -eq 0 ]]; then
  echo "status=completed" >> "$STATUS"
  log "Overnight completed OK"
else
  echo "status=partial_or_failed" >> "$STATUS"
  log "Overnight exited with $EC (partial results in summary.json)"
fi
exit 0
