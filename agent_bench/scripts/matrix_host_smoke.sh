#!/usr/bin/env bash
# Host-native micro-smoke for ThinkingCap matrix newcomers (Jul 2026 shortlist).
# Pass = hello_tc.py contains ThinkingCap-OK.
# Requires Kevlar :8080 + openai_anthropic_shim :8091 on the host.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FIX="$ROOT/agent_bench/fixtures"
OUT="${RESULTS_DIR:-$ROOT/results/agent_bench}/matrix_host_smoke"
MODEL="${LLM_MODEL:-t-prazak/ThinkingCap-Qwen3.6-27B-MLX-4bit}"
SHIM="${OPENAI_BASE_URL:-http://127.0.0.1:8091/v1}"
KEVLAR="${ANTHROPIC_BASE_URL:-http://127.0.0.1:8080}"
KEVLAR_BIN="${KEVLAR_BIN:-$HOME/Projects/Work/Kevlar/.venv/bin/kevlar}"
WALL="${AGENT_WALL_SEC:-360}"
WALL_OPENCLAW="${AGENT_WALL_SEC_OPENCLAW:-600}"
ONLY="${ONLY_CLIS:-}"
PROMPT="${AGENT_PROMPT:-In this empty project, create a file named hello_tc.py that prints 'ThinkingCap-OK' and nothing else. Then exit.}"
BASH_PROMPT="Use the bash/shell tool exactly once with this command and then stop:
printf \"%s\\n\" \"print('ThinkingCap-OK')\" > hello_tc.py
Do not invent absolute paths. Do not use the Write tool."

export PATH="${HOME}/.npm-global/bin:${HOME}/.local/bin:${PATH}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local}"
export OPENAI_BASE_URL="$SHIM"
export OPENAI_API_BASE="$SHIM"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-local}"
export ANTHROPIC_BASE_URL="$KEVLAR"
export CI=1 NO_COLOR=1 OPENHANDS_SUPPRESS_BANNER=1

mkdir -p "$OUT"
REPORT="$OUT/REPORT.txt"
: > "$REPORT"

echo "matrix_host_smoke model=$MODEL shim=$SHIM kevlar=$KEVLAR" | tee -a "$REPORT"

want() {
  [[ -z "$ONLY" ]] && return 0
  [[ ",$ONLY," == *",$1,"* ]]
}

pass() { echo "PASS $1" | tee -a "$REPORT"; }
fail() { echo "FAIL $1 — $2" | tee -a "$REPORT"; }
skip() { echo "SKIP $1 — $2" | tee -a "$REPORT"; }

wait_artifact() {
  local pid=$1 ws=$2
  local i=0
  while kill -0 "$pid" 2>/dev/null; do
    i=$((i + 1))
    if grep -q 'ThinkingCap-OK' "$ws/hello_tc.py" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      break
    fi
    if (( i > WALL )); then
      kill -9 "$pid" 2>/dev/null || true
      echo "killed after ${WALL}s" >>"$ws/run.log"
      break
    fi
    sleep 1
  done
  wait "$pid" 2>/dev/null || true
}

artifact_ok() {
  local ws=$1
  [[ -f "$ws/hello_tc.py" ]] && grep -q 'ThinkingCap-OK' "$ws/hello_tc.py"
}

ensure_kevlar() {
  if curl -sf --max-time 5 "$KEVLAR/v1/status" >/dev/null; then
    echo "kevlar ok $KEVLAR" | tee -a "$REPORT"
    return 0
  fi
  if [[ ! -x "$KEVLAR_BIN" ]]; then
    echo "WARN: kevlar not reachable and KEVLAR_BIN missing ($KEVLAR_BIN)" | tee -a "$REPORT"
    return 1
  fi
  local sess=thinkingcap-kevlar
  tmux -f /exec-daemon/tmux.portal.conf has-session -t "=$sess" 2>/dev/null || \
    tmux -f /exec-daemon/tmux.portal.conf new-session -d -s "$sess" -c "$(dirname "$KEVLAR_BIN")/.." -- "${SHELL:-zsh}" -l
  tmux -f /exec-daemon/tmux.portal.conf send-keys -t "$sess:0.0" C-c
  sleep 1
  tmux -f /exec-daemon/tmux.portal.conf send-keys -t "$sess:0.0" \
    "$KEVLAR_BIN serve --port 8080 --model $MODEL" C-m
  local i=0
  while (( i < 40 )); do
    i=$((i + 1))
    if curl -sf --max-time 5 "$KEVLAR/v1/status" >/dev/null; then
      echo "kevlar started $KEVLAR" | tee -a "$REPORT"
      return 0
    fi
    sleep 3
  done
  echo "WARN: kevlar still not reachable" | tee -a "$REPORT"
  return 1
}

setup_pi_home() {
  # $1 = destination home (e.g. $OUT/pi-home)
  local home="$1"
  mkdir -p "$home/.pi/agent"
  cp "$FIX/pi-local-models-thinkingcap.mjs" "$home/.pi/local-models.mjs"
  cat >"$home/.pi/agent/settings.json" <<JSON
{
  "defaultProvider": "local-ai",
  "defaultModel": "thinkingcap",
  "enabledModels": ["local-ai/thinkingcap", "local/thinkingcap"],
  "extensions": ["$home/.pi/local-models.mjs"]
}
JSON
  echo "$home"
}

if ! curl -sf --max-time 10 "$SHIM/models" >/dev/null; then
  echo "FATAL: cannot reach shim $SHIM" | tee -a "$REPORT"
  exit 1
fi
ensure_kevlar || true

# --- Cline (act mode; isolated --data-dir; base URL is .../v1 not .../v1/v1) ---
if want cline; then
  if command -v cline >/dev/null; then
    ws="$OUT/workspaces/cline"
    rm -rf "$ws"; mkdir -p "$ws"
    cline auth --provider openai -k local -m "$MODEL" -b "$SHIM" \
      --data-dir "$ws/.cline-data" -c "$ws" >>"$ws/auth.log" 2>&1 || true
    echo "" | tee -a "$REPORT"
    echo "=== cline ===" | tee -a "$REPORT"
    (
      cd "$ws"
      cline --data-dir "$ws/.cline-data" -c "$ws" --auto-approve true "$PROMPT" \
        </dev/null >"$ws/run.log" 2>&1 &
      wait_artifact $! "$ws"
    )
    if artifact_ok "$ws"; then
      pass "cline ($(tr -d '\n' <"$ws/hello_tc.py" | head -c 80))"
    else
      fail cline "$(tail -3 "$ws/run.log" 2>/dev/null | tr '\n' ' ' | head -c 220)"
    fi
  else
    skip cline "binary not on PATH"
  fi
fi

# --- OpenSquilla (global ~/.opensquilla/config.toml → shim) ---
if want opensquilla; then
  if command -v opensquilla >/dev/null; then
    ws="$OUT/workspaces/opensquilla"
    rm -rf "$ws"; mkdir -p "$ws"
    echo "" | tee -a "$REPORT"
    echo "=== opensquilla ===" | tee -a "$REPORT"
    (
      cd "$ws"
      opensquilla agent -m "$PROMPT" --workspace "$ws" --workspace-lockdown \
        --timeout "$WALL" --max-iterations 25 \
        </dev/null >"$ws/run.log" 2>&1 &
      wait_artifact $! "$ws"
    )
    if artifact_ok "$ws"; then
      pass "opensquilla ($(tr -d '\n' <"$ws/hello_tc.py" | head -c 80))"
    else
      fail opensquilla "$(tail -3 "$ws/run.log" 2>/dev/null | tr '\n' ' ' | head -c 220)"
    fi
  else
    skip opensquilla "binary not on PATH"
  fi
fi

# --- OpenHands (must run with cwd = workspace; settings + env override) ---
if want openhands; then
  if command -v openhands >/dev/null; then
    ws="$OUT/workspaces/openhands"
    rm -rf "$ws"; mkdir -p "$ws"
    echo "" | tee -a "$REPORT"
    echo "=== openhands ===" | tee -a "$REPORT"
    (
      cd "$ws"
      env LLM_API_KEY=local LLM_MODEL="openai/$MODEL" LLM_BASE_URL="$SHIM" \
        openhands --headless --override-with-envs --always-approve -t "$PROMPT" \
        </dev/null >"$ws/run.log" 2>&1 &
      wait_artifact $! "$ws"
    )
    if artifact_ok "$ws"; then
      pass "openhands ($(tr -d '\n' <"$ws/hello_tc.py" | head -c 80))"
    else
      fail openhands "$(tail -5 "$ws/run.log" 2>/dev/null | tr '\n' ' ' | head -c 220)"
    fi
  else
    skip openhands "binary not on PATH"
  fi
fi

# --- Pi (isolated HOME + local-ai/thinkingcap → shim) ---
if want pi; then
  if command -v pi >/dev/null; then
    ws="$OUT/workspaces/pi"
    rm -rf "$ws"; mkdir -p "$ws"
    PI_HOME="$(setup_pi_home "$OUT/pi-home")"
    echo "" | tee -a "$REPORT"
    echo "=== pi ===" | tee -a "$REPORT"
    (
      cd "$ws"
      env HOME="$PI_HOME" OPENAI_API_KEY=local OPENAI_BASE_URL="$SHIM" ANTHROPIC_BASE_URL="$KEVLAR" \
        pi -p -e "$FIX/pi-local-models-thinkingcap.mjs" --provider local-ai --model thinkingcap "$BASH_PROMPT" \
        </dev/null >"$ws/run.log" 2>&1 &
      wait_artifact $! "$ws"
    )
    if artifact_ok "$ws"; then
      pass "pi ($(tr -d '\n' <"$ws/hello_tc.py" | head -c 80))"
    else
      fail pi "$(tail -5 "$ws/run.log" 2>/dev/null | tr '\n' ' ' | head -c 220)"
    fi
  else
    skip pi "binary not on PATH"
  fi
fi

# --- Oh-my-pi / omp (Pi fork; same isolated HOME pattern) ---
if want oh-my-pi; then
  if command -v omp >/dev/null; then
    ws="$OUT/workspaces/oh-my-pi"
    rm -rf "$ws"; mkdir -p "$ws"
    OMP_HOME="$(setup_pi_home "$OUT/omp-home")"
    echo "" | tee -a "$REPORT"
    echo "=== oh-my-pi ===" | tee -a "$REPORT"
    (
      cd "$ws"
      env HOME="$OMP_HOME" OPENAI_API_KEY=local OPENAI_BASE_URL="$SHIM" \
        omp -p -e "$FIX/pi-local-models-thinkingcap.mjs" --model local-ai/thinkingcap "$BASH_PROMPT" \
        </dev/null >"$ws/run.log" 2>&1 &
      wait_artifact $! "$ws"
    )
    if artifact_ok "$ws"; then
      pass "oh-my-pi ($(tr -d '\n' <"$ws/hello_tc.py" | head -c 80))"
    else
      fail oh-my-pi "$(tail -5 "$ws/run.log" 2>/dev/null | tr '\n' ' ' | head -c 220)"
    fi
  else
    skip oh-my-pi "omp not on PATH"
  fi
fi

# --- Peezy (needs git repo; OpenAI shim) ---
if want peezy; then
  if command -v peezy >/dev/null; then
    ws="$OUT/workspaces/peezy"
    rm -rf "$ws"; mkdir -p "$ws"
    git -C "$ws" init -q
    echo "" | tee -a "$REPORT"
    echo "=== peezy ===" | tee -a "$REPORT"
    (
      cd "$ws"
      peezy --print --base-url "$SHIM" --model "$MODEL" --provider openai \
        --approval never --sandbox workspace-write --skip-git-repo-check "$PROMPT" \
        </dev/null >"$ws/run.log" 2>&1 &
      wait_artifact $! "$ws"
    )
    if artifact_ok "$ws"; then
      pass "peezy ($(tr -d '\n' <"$ws/hello_tc.py" | head -c 80))"
    else
      fail peezy "$(tail -3 "$ws/run.log" 2>/dev/null | tr '\n' ' ' | head -c 220)"
    fi
  else
    skip peezy "binary not on PATH"
  fi
fi

# --- Command Code (vendor model catalog only — no custom OpenAI base URL in CLI) ---
if want command-code; then
  if command -v cmd >/dev/null; then
    skip command-code "CLI uses vendor catalog only; custom ThinkingCap id rejected (needs upstream BYOK path)"
  else
    skip command-code "cmd not on PATH"
  fi
fi

# --- OpenClaw (local embedded agent; patch openclaw.json → shim) ---
if want openclaw; then
  if command -v openclaw >/dev/null; then
    ws="$OUT/workspaces/openclaw"
    rm -rf "$ws"; mkdir -p "$ws"
    openclaw config set gateway.mode local >/dev/null 2>&1 || true
    openclaw config set agents.defaults.model.primary "openai/$MODEL" >/dev/null 2>&1 || true
    openclaw config patch --stdin >/dev/null 2>&1 <<JSON || true
{"models":{"providers":{"openai":{"baseUrl":"$SHIM","apiKey":"local"}}}}
JSON
    echo "" | tee -a "$REPORT"
    echo "=== openclaw ===" | tee -a "$REPORT"
    oc_ws="${OPENCLAW_WORKSPACE:-$HOME/.openclaw/workspace}"
    rm -f "$oc_ws/hello_tc.py" 2>/dev/null || true
    (
      cd "$ws"
      openclaw agent --local --session-id "tc-matrix-$(date +%s)" -m "$PROMPT" \
        </dev/null >"$ws/run.log" 2>&1 &
      pid=$!
      i=0
      while kill -0 "$pid" 2>/dev/null; do
        i=$((i + 1))
        if artifact_ok "$ws" || [[ -f "$oc_ws/hello_tc.py" ]] && grep -q ThinkingCap-OK "$oc_ws/hello_tc.py" 2>/dev/null; then
          kill "$pid" 2>/dev/null || true
          break
        fi
        if (( i > WALL_OPENCLAW )); then
          kill -9 "$pid" 2>/dev/null || true
          break
        fi
        sleep 1
      done
      wait "$pid" 2>/dev/null || true
    )
    if artifact_ok "$ws"; then
      pass "openclaw ($(tr -d '\n' <"$ws/hello_tc.py" | head -c 80))"
    elif [[ -f "$oc_ws/hello_tc.py" ]] && grep -q ThinkingCap-OK "$oc_ws/hello_tc.py"; then
      pass "openclaw (artifact in $oc_ws)"
    else
      fail openclaw "$(tail -5 "$ws/run.log" 2>/dev/null | tr '\n' ' ' | head -c 220)"
    fi
  else
    skip openclaw "binary not on PATH"
  fi
fi

# --- Poolside (no standalone pool CLI on npm; pi-poolside is an extension) ---
if want poolside; then
  if command -v pool >/dev/null; then
    skip poolside "pool binary present but smoke wiring TBD"
  else
    skip poolside "pool binary not found (npm poolside is workflow automation, not agent CLI)"
  fi
fi

echo "" | tee -a "$REPORT"
echo "==== SUMMARY ====" | tee -a "$REPORT"
grep -E '^(PASS|FAIL|SKIP) ' "$REPORT" | tee -a "$REPORT"

python3 - <<PY
import json, os, re
from pathlib import Path
out = Path("${OUT}")
report = out / "REPORT.txt"
if not report.exists():
    raise SystemExit(0)
rows, seen = [], set()
for line in report.read_text().splitlines():
    m = re.match(r'^(PASS|FAIL|SKIP) (\S+)(?:\s+(?:—\s*)?(.*))?$', line)
    if not m:
        continue
    key = (m.group(1), m.group(2))
    if key in seen:
        continue
    seen.add(key)
    rows.append({"status": m.group(1).lower(), "name": m.group(2), "detail": (m.group(3) or "").strip()})
(out / "summary.json").write_text(json.dumps({
    "results": rows,
    "pass": sum(1 for r in rows if r["status"] == "pass"),
    "fail": sum(1 for r in rows if r["status"] == "fail"),
    "skip": sum(1 for r in rows if r["status"] == "skip"),
}, indent=2))
print(f"wrote {out / 'summary.json'}")
PY
