#!/usr/bin/env bash
# Re-smoke agent CLIs against local ThinkingCap (Kevlar :8080 + shim :8091).
# Pass = workspace contains hello_tc.py
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FIX="$ROOT/agent_bench/fixtures"
BASE="${SMOKE_ROOT:-/tmp/tc-agent-resmoke}"
PROMPT="In this empty project, create a file named hello_tc.py that prints 'ThinkingCap-OK' and nothing else. Then exit."
MODEL="t-prazak/ThinkingCap-Qwen3.6-27B-MLX-4bit"
SHIM="http://127.0.0.1:8091/v1"
KEVLAR="http://127.0.0.1:8080"
CLAUDE_SETTINGS="$HOME/.claude/settings.qwen36.json"
REPORT="$BASE/REPORT.txt"
mkdir -p "$BASE" "$FIX"
: > "$REPORT"

export OPENAI_API_KEY=local
export OPENAI_BASE_URL="$SHIM"
export OPENAI_API_BASE="$SHIM"
export ANTHROPIC_API_KEY=local
export ANTHROPIC_BASE_URL="$KEVLAR"
export CI=1
export NO_COLOR=1

pass() { echo "PASS $1" | tee -a "$REPORT"; }
fail() { echo "FAIL $1 — $2" | tee -a "$REPORT"; }

run_one() {
  local name="$1"; shift
  local ws="$BASE/$name"
  rm -rf "$ws"; mkdir -p "$ws"
  echo "" | tee -a "$REPORT"
  echo "=== $name ===" | tee -a "$REPORT"
  echo "\$ $*" | tee -a "$REPORT"
  (
    cd "$ws"
    # 4 minute soft wall via background+wait
    "$@" >"$ws/run.log" 2>&1 &
    local pid=$!
    local i=0
    while kill -0 "$pid" 2>/dev/null; do
      i=$((i+1))
      if [[ $i -gt 240 ]]; then
        kill -9 "$pid" 2>/dev/null || true
        echo "killed after 240s" >>"$ws/run.log"
        break
      fi
      sleep 1
    done
    wait "$pid" 2>/dev/null || true
  )
  if [[ -f "$ws/hello_tc.py" ]]; then
    pass "$name ($(tr -d '\n' <"$ws/hello_tc.py" | head -c 80))"
  else
    fail "$name" "no hello_tc.py; tail: $(tail -5 "$ws/run.log" | tr '\n' ' ' | head -c 200)"
  fi
}

# 1 Claude (already known good — keep)
if command -v claude >/dev/null; then
  run_one claude-code claude -p "$PROMPT" --settings "$CLAUDE_SETTINGS" \
    --dangerously-skip-permissions --bare --max-turns 8
fi

# 2 Aider
if command -v aider >/dev/null; then
  run_one aider aider --model "openai/$MODEL" --openai-api-base "$SHIM" \
    --openai-api-key local --yes --no-git --no-stream --message "$PROMPT"
fi

FIX="$ROOT/agent_bench/fixtures"
# Isolated XDG config dirs so MCP/Azure defaults do not load
setup_xdg() {
  local app="$1" src="$2"
  local dir="$BASE/xdg-$app"
  rm -rf "$dir"
  mkdir -p "$dir/$app"
  cp "$src" "$dir/$app/config.json" 2>/dev/null || cp "$src" "$dir/$app/$app.json"
  # opencode looks at config.json + opencode.json; mimo at mimocode.json; kilo at kilo.json
  case "$app" in
    opencode)
      cp "$src" "$dir/opencode/config.json"
      cp "$src" "$dir/opencode/opencode.json"
      ;;
    mimocode)
      mkdir -p "$dir/mimocode"
      cp "$src" "$dir/mimocode/mimocode.json"
      ;;
    kilo)
      mkdir -p "$dir/kilo"
      cp "$src" "$dir/kilo/kilo.json"
      ;;
  esac
  echo "$dir"
}

# 3 OpenCode — MCP-free fixture via XDG_CONFIG_HOME
if command -v opencode >/dev/null; then
  OC_XDG="$(setup_xdg opencode "$FIX/opencode-thinkingcap.json")"
  run_one opencode env XDG_CONFIG_HOME="$OC_XDG" OPENCODE_CONFIG="$FIX/opencode-thinkingcap.json" \
    opencode run --pure --model local/thinkingcap --auto "$PROMPT"
fi

# 4 Mimo
if command -v mimo >/dev/null; then
  MIMO_XDG="$(setup_xdg mimocode "$FIX/mimo-thinkingcap.json")"
  run_one mimo env XDG_CONFIG_HOME="$MIMO_XDG" \
    mimo run --pure --model local/thinkingcap --auto "$PROMPT"
fi

# 5 Kilo
if command -v kilocode >/dev/null || command -v kilo >/dev/null; then
  KILO_BIN="$(command -v kilocode || command -v kilo)"
  KILO_XDG="$(setup_xdg kilo "$FIX/kilo-thinkingcap.json")"
  run_one kilocode env XDG_CONFIG_HOME="$KILO_XDG" \
    "$KILO_BIN" run --pure -m local/thinkingcap --auto "$PROMPT"
fi

# 6 Antigravity (agy) — Anthropic env → Kevlar; no invalid --settings
if command -v agy >/dev/null; then
  run_one antigravity env ANTHROPIC_BASE_URL="$KEVLAR" ANTHROPIC_API_KEY=local \
    agy --print "$PROMPT" --dangerously-skip-permissions --print-timeout 3m
fi

# 7 Codex — thinkingcap profile, ignore azure user config
if command -v codex >/dev/null; then
  run_one codex env OPENAI_API_KEY=local \
    codex --ignore-user-config -p thinkingcap exec --skip-git-repo-check \
    --sandbox danger-full-access -c 'approval_policy="never"' "$PROMPT"
fi

# 8 Goose — force openai host to shim; short max turns; no profile extensions from Azure
if command -v goose >/dev/null; then
  run_one goose env \
    GOOSE_PROVIDER=openai \
    GOOSE_MODEL="$MODEL" \
    OPENAI_HOST="$SHIM" \
    OPENAI_BASE_PATH=chat/completions \
    OPENAI_API_KEY=local \
    OPENAI_TIMEOUT=600 \
    GOOSE_MODE=auto \
    goose run -t "$PROMPT" --no-session --no-profile --max-turns 12 --quiet \
      --with-builtin developer
fi

# 9 Hermes — thinkingcap provider, yolo, safe-mode (fewer tools)
if command -v hermes >/dev/null; then
  run_one hermes env OPENAI_API_KEY=local \
    hermes -z "$PROMPT" -m "$MODEL" --provider thinkingcap --yolo --safe-mode
fi

# 10 Cursor — needs login unless key present; try once
CURSOR="$(command -v cursor-agent || true)"
if [[ -z "$CURSOR" ]]; then
  CURSOR="$HOME/Library/Application Support/Cursor/User/globalStorage/anysphere.cursor-agent-worker/agent-cli/.local/share/cursor-agent/versions/2026.07.09-a3815c0/cursor-agent"
fi
if [[ -x "$CURSOR" ]]; then
  run_one cursor-cli env OPENAI_API_KEY=local OPENAI_BASE_URL="$SHIM" \
    "$CURSOR" -p "$PROMPT" --model "$MODEL"
fi
echo "" | tee -a "$REPORT"
echo "==== SUMMARY ====" | tee -a "$REPORT"
grep -E '^(PASS|FAIL) ' "$REPORT" | tee -a "$REPORT"
echo "Full report: $REPORT"
