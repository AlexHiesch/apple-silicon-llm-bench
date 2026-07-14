#!/usr/bin/env bash
# Launch full overnight ThinkingCap harness for blog results.
# Keeps Mac awake, ensures Kevlar is healthy, then runs the orchestrator.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p results/agent_bench/overnight logs

STAMP=$(date +%Y%m%d_%H%M%S)
LOG="results/agent_bench/overnight/overnight_${STAMP}.log"
STATUS="results/agent_bench/overnight/STATUS.txt"
MODEL="t-prazak/ThinkingCap-Qwen3.6-27B-MLX-4bit"
PORT=8080
BASE="http://127.0.0.1:${PORT}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

echo "started=$(date -Iseconds)" > "$STATUS"
echo "log=$LOG" >> "$STATUS"
echo "model=$MODEL" >> "$STATUS"
echo "port=$PORT" >> "$STATUS"

# Prefer project venv python
PY="$ROOT/.venv/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python3)"

kevlar_healthy() {
  curl -sf --max-time 5 "$BASE/v1/status" 2>/dev/null | grep -q '"model_loaded":true'
}

ensure_model() {
  if kevlar_healthy; then
    log "ThinkingCap already up on :$PORT (Kevlar /v1/status)"
    return 0
  fi
  log "Starting Kevlar ThinkingCap on :$PORT ..."
  # Kill stale listeners on port if any
  if lsof -tiTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    log "Killing stale process on :$PORT"
    lsof -tiTCP:"$PORT" -sTCP:LISTEN | xargs kill -9 2>/dev/null || true
    sleep 2
  fi
  # tmux session if available, else nohup
  if command -v tmux >/dev/null 2>&1; then
    tmux has-session -t thinkingcap-kevlar 2>/dev/null && tmux kill-session -t thinkingcap-kevlar 2>/dev/null || true
    tmux new-session -d -s thinkingcap-kevlar \
      "caffeinate -ims kevlar serve --model '$MODEL' --port $PORT 2>&1 | tee /tmp/llm-model.log"
  else
    nohup caffeinate -ims kevlar serve --model "$MODEL" --port "$PORT" \
      > /tmp/llm-model.log 2>&1 &
  fi
}

wait_healthy() {
  local deadline=$((SECONDS + 3600))
  log "Waiting up to 60m for $BASE/v1/status (model_loaded) ..."
  while (( SECONDS < deadline )); do
    if kevlar_healthy; then
      log "Model healthy."
      curl -s --max-time 5 "$BASE/v1/status" | head -c 400 >> "$LOG" || true
      echo >> "$LOG"
      return 0
    fi
    # Still loading?
    if ! pgrep -f "kevlar serve" >/dev/null 2>&1; then
      log "WARNING: kevlar process not found — restarting"
      ensure_model
    fi
    sleep 15
  done
  log "FATAL: model never became healthy"
  echo "status=failed_model" >> "$STATUS"
  return 1
}

smoke_chat() {
  log "Smoke Anthropic /v1/messages ..."
  local code
  code=$(curl -s -o /tmp/tc_smoke.json -w "%{http_code}" --max-time 180 \
    -X POST "$BASE/v1/messages" \
    -H "content-type: application/json" \
    -H "x-api-key: local" \
    -H "anthropic-version: 2023-06-01" \
    -d '{"model":"'"$MODEL"'","max_tokens":32,"messages":[{"role":"user","content":"Reply with exactly: OK"}]}' || echo 000)
  log "Anthropic smoke HTTP $code"
  head -c 300 /tmp/tc_smoke.json >> "$LOG" 2>/dev/null || true
  echo >> "$LOG"

  log "Smoke OpenAI /v1/chat/completions (optional — Kevlar may be Anthropic-only) ..."
  code=$(curl -s -o /tmp/tc_smoke_oai.json -w "%{http_code}" --max-time 30 \
    -X POST "$BASE/v1/chat/completions" \
    -H "content-type: application/json" \
    -H "authorization: Bearer local" \
    -d '{"model":"'"$MODEL"'","max_tokens":32,"messages":[{"role":"user","content":"Reply with exactly: OK"}]}' || echo 000)
  log "OpenAI smoke HTTP $code (404 expected on pure Kevlar)"
  head -c 300 /tmp/tc_smoke_oai.json >> "$LOG" 2>/dev/null || true
  echo >> "$LOG"
}

ensure_model
wait_healthy || exit 1
smoke_chat

log "==== Starting overnight orchestrator ===="
echo "status=running" >> "$STATUS"
echo "orchestrator_start=$(date -Iseconds)" >> "$STATUS"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local}"
export OPENAI_BASE_URL="$BASE/v1"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-local}"
export ANTHROPIC_BASE_URL="$BASE"
export ANTHROPIC_MODEL="$MODEL"
export CLAUDE_CODE_USE_BEDROCK=0
export CLAUDE_CODE_SKIP_BEDROCK_AUTH=1

# Keep Mac awake for the entire orchestration
caffeinate -i "$PY" -u "$ROOT/agent_bench/run_overnight_full.py" 2>&1 | tee -a "$LOG"
EC=${PIPESTATUS[0]}

echo "orchestrator_end=$(date -Iseconds)" >> "$STATUS"
echo "exit_code=$EC" >> "$STATUS"
if [[ $EC -eq 0 ]]; then
  echo "status=completed" >> "$STATUS"
  log "Overnight completed OK"
else
  echo "status=partial_or_failed" >> "$STATUS"
  log "Overnight exited with $EC (partial results may still be in summary.json)"
fi
# Blog run: partial results are useful — do not fail the launcher on orchestrator exit 2
exit 0
