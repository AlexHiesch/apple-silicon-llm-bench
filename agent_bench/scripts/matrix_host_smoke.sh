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
      # also stop children of agent wrappers (e.g. peezy → Codex)
      pkill -P "$pid" 2>/dev/null || true
      break
    fi
    if (( i > WALL )); then
      kill -9 "$pid" 2>/dev/null || true
      pkill -9 -P "$pid" 2>/dev/null || true
      echo "killed after ${WALL}s" >>"$ws/run.log"
      break
    fi
    sleep 1
  done
  wait "$pid" 2>/dev/null || true
  # Codex/Peezy sometimes flush the sandbox after the parent exits
  local j=0
  while (( j < 15 )); do
    grep -q 'ThinkingCap-OK' "$ws/hello_tc.py" 2>/dev/null && return 0
    sleep 1
    j=$((j + 1))
  done
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
  mkdir -p "$home/.pi/agent" "$home/.omp/agent"
  cp "$FIX/pi-local-models-thinkingcap.mjs" "$home/.pi/local-models.mjs"
  cp "$FIX/omp-models-thinkingcap.yml" "$home/.omp/agent/models.yml"
  # rewrite base URLs for this run's SHIM/KEVLAR
  python3 - "$home/.omp/agent/models.yml" "$SHIM" "$KEVLAR" <<'PY'
import sys
from pathlib import Path
p, shim, kevlar = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
text = p.read_text()
text = text.replace("http://127.0.0.1:8091/v1", shim.rstrip("/"))
text = text.replace("http://127.0.0.1:8080", kevlar.rstrip("/"))
p.write_text(text)
PY
  cat >"$home/.pi/agent/settings.json" <<JSON
{
  "defaultProvider": "local",
  "defaultModel": "thinkingcap",
  "enabledModels": ["local/thinkingcap", "local-ai/thinkingcap"],
  "extensions": ["$home/.pi/local-models.mjs"],
  "defaultThinkingLevel": "off"
}
JSON
  echo "$home"
}

# Peezy embeds Codex SDK which expects vendor/.../codex/codex under
# @openai/codex-darwin-arm64. Incomplete npm installs leave that empty;
# symlink Homebrew's Codex binary when available.
ensure_peezy_codex() {
  local vendor_root arch_root brew_bin
  vendor_root="$(npm root -g 2>/dev/null)/@p0systems/peezy/node_modules/@openai/codex-darwin-arm64/vendor"
  [[ -d "$vendor_root" ]] || vendor_root="$HOME/.npm-global/lib/node_modules/@p0systems/peezy/node_modules/@openai/codex-darwin-arm64/vendor"
  arch_root="$vendor_root/aarch64-apple-darwin"
  brew_bin="/opt/homebrew/lib/node_modules/@openai/codex/node_modules/@openai/codex-darwin-arm64/vendor/aarch64-apple-darwin/bin/codex"
  if [[ -x "${arch_root}/codex/codex" ]]; then
    return 0
  fi
  if [[ -x "$brew_bin" ]]; then
    mkdir -p "${arch_root}/codex"
    ln -sfn "$brew_bin" "${arch_root}/codex/codex"
    echo "peezy: linked Codex binary → ${arch_root}/codex/codex" | tee -a "$REPORT"
    return 0
  fi
  echo "WARN: peezy Codex vendor binary missing (install brew codex or reinstall @openai/codex)" | tee -a "$REPORT"
  return 1
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

# --- Pi (Anthropic → Kevlar; --thinking off; fixture extension) ---
# Note: OpenAI-completions path against the shim often stalls on tool turns.
# Anthropic Messages via Kevlar with thinking off is the reliable local route.
if want pi; then
  if command -v pi >/dev/null; then
    ws="$OUT/workspaces/pi"
    rm -rf "$ws"; mkdir -p "$ws"
    PI_HOME="$(setup_pi_home "$OUT/pi-home")"
    echo "" | tee -a "$REPORT"
    echo "=== pi ===" | tee -a "$REPORT"
    (
      cd "$ws"
      env HOME="$PI_HOME" OPENAI_API_KEY=local OPENAI_BASE_URL="$SHIM" \
        ANTHROPIC_API_KEY=local ANTHROPIC_BASE_URL="$KEVLAR" \
        stdbuf -oL -eL pi -p -ne -e "$FIX/pi-local-models-thinkingcap.mjs" \
          --thinking off --provider local --model thinkingcap --verbose \
          "$BASH_PROMPT" \
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

# --- Oh-my-pi / omp (Anthropic → Kevlar; slim tool allowlist) ---
# Default omp advertises ~20 tools (~22k prompt tokens) which starves/stalls
# local MLX prefills. Mirror Pi: fixture + --tools=bash,write,read,edit.
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
        ANTHROPIC_API_KEY=local ANTHROPIC_BASE_URL="$KEVLAR" \
        stdbuf -oL -eL omp -p --no-extensions -e "$FIX/pi-local-models-thinkingcap.mjs" \
          --thinking=off --provider local --model thinkingcap --auto-approve \
          --no-skills --no-rules --no-lsp --no-pty \
          --tools=bash,write,read,edit \
          "$BASH_PROMPT" \
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

# --- Peezy (Codex SDK → openai provider with responses wireApi → shim) ---
# Requires a complete @openai/codex-darwin-arm64 vendor binary (see ensure_peezy_codex).
# Do NOT use --provider local (wireApi=chat — Codex rejects it).
if want peezy; then
  if command -v peezy >/dev/null; then
    ensure_peezy_codex || true
    ws="$OUT/workspaces/peezy"
    rm -rf "$ws"; mkdir -p "$ws"
    git -C "$ws" init -q
    echo "" | tee -a "$REPORT"
    echo "=== peezy ===" | tee -a "$REPORT"
    (
      cd "$ws"
      env OPENAI_API_KEY=local \
        peezy --print --verbose --provider openai --base-url "$SHIM" --model "$MODEL" \
          --approval never --sandbox danger-full-access --skip-git-repo-check \
          "$BASH_PROMPT" \
        </dev/null >"$ws/run.log" 2>&1 &
      wait_artifact $! "$ws"
    )
    if artifact_ok "$ws"; then
      pass "peezy ($(tr -d '\n' <"$ws/hello_tc.py" | head -c 80))"
    else
      fail peezy "$(tail -5 "$ws/run.log" 2>/dev/null | tr '\n' ' ' | head -c 220)"
    fi
  else
    skip peezy "binary not on PATH"
  fi
fi

# --- Command Code (COMMAND_CODE_API_KEY + sandbox → shim /alpha/generate) ---
if want command-code; then
  if command -v cmd >/dev/null; then
    ws="$OUT/workspaces/command-code"
    rm -rf "$ws"; mkdir -p "$ws"
    echo "" | tee -a "$REPORT"
    echo "=== command-code ===" | tee -a "$REPORT"
    # Catalog rejects ThinkingCap id — use any OSS catalog model; shim remaps to ThinkingCap.
    CMD_MODEL="${CMD_MODEL:-Qwen/Qwen3.6-Plus}"
    (
      cd "$ws"
      env COMMAND_CODE_API_KEY="${COMMAND_CODE_API_KEY:-local}" \
        COMMANDCODE_SANDBOX=true \
        COMMANDCODE_API_URL="${COMMANDCODE_API_URL:-http://127.0.0.1:8091}" \
        cmd -p --skip-onboarding --yolo --trust --max-turns 25 -m "$CMD_MODEL" \
          "$BASH_PROMPT" \
        </dev/null >"$ws/run.log" 2>&1 &
      wait_artifact $! "$ws"
    )
    if artifact_ok "$ws"; then
      pass "command-code ($(tr -d '\n' <"$ws/hello_tc.py" | head -c 80))"
    else
      fail command-code "$(tail -5 "$ws/run.log" 2>/dev/null | tr '\n' ' ' | head -c 220)"
    fi
  else
    skip command-code "cmd not on PATH"
  fi
fi

# --- OpenClaw (embedded agent → shim; slim tools + raised idle timeout) ---
# Full default tool catalogs + AGENTS/SOUL bootstrap ≈ 15–20k tokens and can
# ABRT Kevlar Metal. Restrict tools, point workspace at the smoke dir, raise
# models.providers.openai.timeoutSeconds (also lifts LLM idle watchdog).
if want openclaw; then
  if command -v openclaw >/dev/null; then
    ws="$OUT/workspaces/openclaw"
    rm -rf "$ws"; mkdir -p "$ws"
    oc_ws="$ws"
    oc_prev_ws="$(openclaw config get agents.defaults.workspace 2>/dev/null | tr -d '"' || true)"
    [[ -z "$oc_prev_ws" ]] && oc_prev_ws="$HOME/.openclaw/workspace"
    # Tiny bootstrap stubs so injection stays cheap even if files are recreated
    for f in AGENTS.md SOUL.md TOOLS.md IDENTITY.md USER.md HEARTBEAT.md BOOTSTRAP.md; do
      printf '# %s\n' "$f" >"$ws/$f"
    done
    openclaw config set gateway.mode local >/dev/null 2>&1 || true
    openclaw config set agents.defaults.model.primary "openai/$MODEL" >/dev/null 2>&1 || true
    openclaw config set agents.defaults.workspace "$ws" >/dev/null 2>&1 || true
    openclaw config set agents.defaults.bootstrapMaxChars 120 --strict-json >/dev/null 2>&1 || true
    openclaw config set agents.defaults.bootstrapTotalMaxChars 500 --strict-json >/dev/null 2>&1 || true
    openclaw config set agents.defaults.bootstrapPromptTruncationWarning off >/dev/null 2>&1 || true
    # Clear prior exec/host pins that hang embedded --local without a gateway,
    # then allow only filesystem tools so the model cannot pick exec→gateway.
    openclaw config unset tools.exec >/dev/null 2>&1 || true
    openclaw config unset tools.allow >/dev/null 2>&1 || true
    openclaw config patch --stdin >/dev/null 2>&1 <<JSON || true
{
  "models": {
    "providers": {
      "openai": {
        "baseUrl": "$SHIM",
        "apiKey": "local",
        "api": "openai-completions",
        "timeoutSeconds": 600
      }
    }
  },
  "tools": {
    "profile": "coding",
    "allow": ["write", "edit", "read"]
  }
}
JSON
    echo "" | tee -a "$REPORT"
    echo "=== openclaw ===" | tee -a "$REPORT"
    rm -f "$ws/hello_tc.py" 2>/dev/null || true
    (
      cd "$ws"
      openclaw agent --local --thinking off --timeout "$WALL_OPENCLAW" \
        --session-id "tc-matrix-$(date +%s)" \
        -m "Use the write tool exactly once: path=hello_tc.py content=print('ThinkingCap-OK'). Do not use exec/bash. Then stop." \
        </dev/null >"$ws/run.log" 2>&1 &
      pid=$!
      i=0
      while kill -0 "$pid" 2>/dev/null; do
        i=$((i + 1))
        if artifact_ok "$ws"; then
          kill "$pid" 2>/dev/null || true
          break
        fi
        if (( i > WALL_OPENCLAW )); then
          kill -9 "$pid" 2>/dev/null || true
          echo "killed after ${WALL_OPENCLAW}s" >>"$ws/run.log"
          break
        fi
        sleep 1
      done
      wait "$pid" 2>/dev/null || true
    )
    # Restore caller's workspace path
    openclaw config set agents.defaults.workspace "$oc_prev_ws" >/dev/null 2>&1 || true
    if artifact_ok "$ws"; then
      pass "openclaw ($(tr -d '\n' <"$ws/hello_tc.py" | head -c 80))"
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
