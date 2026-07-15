#!/usr/bin/env bash
# Run agent CLI micro-smokes inside Docker against host ThinkingCap.
# Pass = hello_tc.py exists. Does not touch host-native CLI sessions.
set -uo pipefail

MODEL="${LLM_MODEL:-t-prazak/ThinkingCap-Qwen3.6-27B-MLX-4bit}"
SHIM="${OPENAI_BASE_URL:-http://host.docker.internal:8091/v1}"
KEVLAR="${ANTHROPIC_BASE_URL:-http://host.docker.internal:8080}"
PROMPT="${AGENT_PROMPT:-In this empty project, create a file named hello_tc.py that prints 'ThinkingCap-OK' and nothing else. Then exit.}"
OUT="${RESULTS_DIR:-/results}/docker_cli_smoke"
ONLY="${ONLY_CLIS:-}"
WALL="${AGENT_WALL_SEC:-480}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-local}"
export OPENAI_BASE_URL="$SHIM"
export OPENAI_API_BASE="$SHIM"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-local}"
export ANTHROPIC_BASE_URL="$KEVLAR"
export CI=1 NO_COLOR=1 DISPLAY= HEADLESS=1
export GOOSE_PROVIDER=openai
export GOOSE_MODEL="$MODEL"
export OPENAI_HOST="$SHIM"
export OPENAI_BASE_PATH=chat/completions

mkdir -p "$OUT"
REPORT="$OUT/REPORT.txt"
: > "$REPORT"

echo "docker_cli_smoke model=$MODEL shim=$SHIM kevlar=$KEVLAR" | tee -a "$REPORT"

# Patch fixture base URLs for docker host gateway into $HOME configs
python3 - <<'PY'
import json, os
from pathlib import Path
home = Path(os.path.expanduser("~"))
for name in ("opencode-thinkingcap.json", "kilo-thinkingcap.json", "mimo-thinkingcap.json"):
    p = Path("/fixtures") / name
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    for prov in (d.get("provider") or {}).values():
        opts = prov.setdefault("options", {})
        opts["baseURL"] = "http://host.docker.internal:8091/v1"
        opts["apiKey"] = "local"
    if "opencode" in name:
        d.setdefault("permission", {})["external_directory"] = "deny"
        (home / ".config/opencode").mkdir(parents=True, exist_ok=True)
        (home / ".config/opencode/config.json").write_text(json.dumps(d, indent=2))
        (home / ".config/opencode/opencode.json").write_text(json.dumps(d, indent=2))
print("fixtures patched for host.docker.internal")
PY

want() {
  [[ -z "$ONLY" ]] && return 0
  [[ ",$ONLY," == *",$1,"* ]]
}

pass() { echo "PASS $1" | tee -a "$REPORT"; }
fail() { echo "FAIL $1 — $2" | tee -a "$REPORT"; }
skip() { echo "SKIP $1 — $2" | tee -a "$REPORT"; }

run_one() {
  local name="$1"; shift
  want "$name" || return 0
  if ! command -v "$1" >/dev/null 2>&1 && [[ "$1" != env ]]; then
    # allow "env" wrapper; otherwise check later
    :
  fi
  local bin="$1"
  if [[ "$bin" != env ]] && ! command -v "$bin" >/dev/null 2>&1; then
    skip "$name" "binary not installed in image"
    return 0
  fi
  local ws="$OUT/workspaces/$name"
  rm -rf "$ws"; mkdir -p "$ws"
  echo "" | tee -a "$REPORT"
  echo "=== $name ===" | tee -a "$REPORT"
  echo "\$ $*" | tee -a "$REPORT"
  (
    cd "$ws"
    "$@" </dev/null >"$ws/run.log" 2>&1 &
    local pid=$!
    local i=0
    while kill -0 "$pid" 2>/dev/null; do
      i=$((i + 1))
      if (( i > WALL )); then
        kill -9 "$pid" 2>/dev/null || true
        echo "killed after ${WALL}s" >>"$ws/run.log"
        break
      fi
      sleep 1
    done
    wait "$pid" 2>/dev/null || true
  )
  if [[ -f "$ws/hello_tc.py" ]]; then
    pass "$name ($(tr -d '\n' <"$ws/hello_tc.py" | head -c 80))"
  else
    fail "$name" "no hello_tc.py; $(tail -3 "$ws/run.log" 2>/dev/null | tr '\n' ' ' | head -c 220)"
  fi
}

# Reachability first
echo "-- reachability --" | tee -a "$REPORT"
if curl -sf --max-time 10 "$SHIM/models" >/dev/null; then
  echo "shim ok $SHIM" | tee -a "$REPORT"
else
  echo "FATAL: cannot reach shim $SHIM" | tee -a "$REPORT"
  exit 1
fi
if curl -sf --max-time 10 "$KEVLAR/v1/status" >/dev/null; then
  echo "kevlar ok $KEVLAR" | tee -a "$REPORT"
else
  echo "WARN: kevlar status not reachable (Anthropic CLIs may fail)" | tee -a "$REPORT"
fi

# Claude Code (Anthropic → Kevlar) — must not run as root
if command -v claude >/dev/null; then
  mkdir -p "$HOME/.claude"
  cat > "$HOME/.claude/settings.tc.json" <<EOF
{"env":{"ANTHROPIC_BASE_URL":"$KEVLAR","ANTHROPIC_API_KEY":"local","CLAUDE_CODE_USE_BEDROCK":"0"},"model":"$MODEL","availableModels":["$MODEL"]}
EOF
  if [[ "$(id -u)" -eq 0 ]]; then
    skip claude-code "refuses root; rebuild image as non-root user"
  else
    run_one claude-code claude -p "$PROMPT" --settings "$HOME/.claude/settings.tc.json" \
      --dangerously-skip-permissions --bare --max-turns 8
  fi
else
  skip claude-code "not in image"
fi

# Aider
if command -v aider >/dev/null; then
  run_one aider aider --model "openai/$MODEL" --openai-api-base "$SHIM" \
    --openai-api-key local --yes --no-git --no-stream --message "$PROMPT"
else
  skip aider "not in image"
fi

# OpenCode — ThinkingCap invents absolute paths; seed file + force shell write + deny external_directory
if command -v opencode >/dev/null; then
  if want opencode; then
    _oc_ws="$OUT/workspaces/opencode"
    rm -rf "$_oc_ws"; mkdir -p "$_oc_ws"
    printf 'PLACEHOLDER\n' > "$_oc_ws/hello_tc.py"
    echo "" | tee -a "$REPORT"
    echo "=== opencode ===" | tee -a "$REPORT"
    echo "\$ opencode run --pure --dir $_oc_ws (shell overwrite hello_tc.py)" | tee -a "$REPORT"
    (
      cd "$_oc_ws"
      opencode run --pure --dir "$_oc_ws" --model local/thinkingcap --auto \
        "Use the bash/shell tool exactly once with this command and then stop:
printf \"%s\\n\" \"print(\\\"ThinkingCap-OK\\\")\" > hello_tc.py
Do not invent absolute paths. Do not use the Write tool." \
        >"$_oc_ws/run.log" 2>&1 &
      pid=$!
      i=0
      while kill -0 "$pid" 2>/dev/null; do
        i=$((i + 1))
        if grep -q 'ThinkingCap-OK' "$_oc_ws/hello_tc.py" 2>/dev/null; then
          kill "$pid" 2>/dev/null || true
          break
        fi
        if (( i > WALL )); then
          kill -9 "$pid" 2>/dev/null || true
          echo "killed after ${WALL}s" >>"$_oc_ws/run.log"
          break
        fi
        sleep 1
      done
      wait "$pid" 2>/dev/null || true
    )
    if grep -q 'ThinkingCap-OK' "$_oc_ws/hello_tc.py" 2>/dev/null; then
      pass "opencode ($(tr -d '\n' <"$_oc_ws/hello_tc.py" | head -c 80))"
    else
      # remove placeholder so pass criteria stays honest if model never wrote
      if grep -q PLACEHOLDER "$_oc_ws/hello_tc.py" 2>/dev/null; then
        rm -f "$_oc_ws/hello_tc.py"
      fi
      fail opencode "no hello_tc.py; $(tail -3 "$_oc_ws/run.log" 2>/dev/null | tr '\n' ' ' | head -c 220)"
    fi
  fi
else
  skip opencode "not in image"
fi

# Goose (headless — no macOS popup). Explicit shell instruction; no --quiet so tools log.
if command -v goose >/dev/null; then
  run_one goose goose run \
    -t "Use the developer shell tools. In the current directory, create hello_tc.py containing exactly: print('ThinkingCap-OK') then stop. Do not only describe the file — write it with a tool." \
    --no-session --no-profile --max-turns 16 \
    --with-builtin developer
else
  skip goose "not in image"
fi

# Hermes — ThinkingCap chat provider (ignore-user-config so fixture wins)
if command -v hermes >/dev/null; then
  run_one hermes hermes -z "$PROMPT" -m "$MODEL" --provider thinkingcap --yolo --ignore-user-config
else
  skip hermes "not in image"
fi

# Codex — Responses SSE required; ThinkingCap needs an explicit exec_command prompt
# (generic prompts cause the model to echo Codex's huge tool schemas as text).
if command -v codex >/dev/null; then
  run_one codex env OPENAI_API_KEY=local \
    codex exec --skip-git-repo-check \
    --sandbox danger-full-access -c 'approval_policy="never"' \
    -c 'model_provider="thinkingcap"' \
    -c "model=\"$MODEL\"" \
    -c 'model_providers.thinkingcap={name="ThinkingCap", base_url="http://host.docker.internal:8091/v1", env_key="OPENAI_API_KEY"}' \
    -c 'model_reasoning_effort="none"' \
    "Call the exec_command tool now with cmd set to exactly:
printf \"%s\\n\" \"print(\\\"ThinkingCap-OK\\\")\" > hello_tc.py
Do not describe tools. Do not output JSON schemas. One tool call only."
else
  skip codex "not in image"
fi

# Mac-only / auth-walled — host fallback: bash agent_bench/scripts/host_skip_smoke.sh
skip mimo-code "host-only arm64 — run host_skip_smoke.sh"
skip kilocode "prefer host_skip_smoke.sh (or add @kilocode/cli to image)"
skip antigravity "macOS agy — run host_skip_smoke.sh"
skip cursor-cli "Cursor login on host — run host_skip_smoke.sh"

echo "" | tee -a "$REPORT"
echo "==== SUMMARY ====" | tee -a "$REPORT"
grep -E '^(PASS|FAIL|SKIP) ' "$REPORT" | tee -a "$REPORT"
# machine-readable
python3 - <<'PY'
import json, re
from pathlib import Path
report = Path("/results/docker_cli_smoke/REPORT.txt")
rows = []
seen = set()
for line in report.read_text().splitlines():
    m = re.match(r'^(PASS|FAIL|SKIP) (\S+)(?:\s+(?:—\s*)?(.*))?$', line)
    if not m:
        continue
    key = (m.group(1), m.group(2))
    if key in seen:
        continue
    seen.add(key)
    rows.append({"status": m.group(1).lower(), "name": m.group(2), "detail": (m.group(3) or "").strip()})
Path("/results/docker_cli_smoke/summary.json").write_text(json.dumps({
    "results": rows,
    "pass": sum(1 for r in rows if r["status"]=="pass"),
    "fail": sum(1 for r in rows if r["status"]=="fail"),
    "skip": sum(1 for r in rows if r["status"]=="skip"),
}, indent=2))
print("wrote /results/docker_cli_smoke/summary.json")
PY
