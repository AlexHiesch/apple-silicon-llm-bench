#!/usr/bin/env bash
# Host-native micro-smoke for CLIs that Docker skips (macOS / auth).
# Pass = hello_tc.py prints ThinkingCap-OK.
# Targets: kilocode, mimo-code, antigravity (agy), cursor-cli
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FIX="$ROOT/agent_bench/fixtures"
OUT="${RESULTS_DIR:-$ROOT/results/agent_bench}/host_skip_smoke"
MODEL="${LLM_MODEL:-t-prazak/ThinkingCap-Qwen3.6-27B-MLX-4bit}"
SHIM="${OPENAI_BASE_URL:-http://127.0.0.1:8091/v1}"
KEVLAR="${ANTHROPIC_BASE_URL:-http://127.0.0.1:8080}"
WALL="${AGENT_WALL_SEC:-360}"
ONLY="${ONLY_CLIS:-}"
PROMPT="${AGENT_PROMPT:-In this empty project, create a file named hello_tc.py that prints 'ThinkingCap-OK' and nothing else. Then exit.}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-local}"
export OPENAI_BASE_URL="$SHIM"
export OPENAI_API_BASE="$SHIM"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-local}"
export ANTHROPIC_BASE_URL="$KEVLAR"
export CI=1 NO_COLOR=1

mkdir -p "$OUT"
REPORT="$OUT/REPORT.txt"
: > "$REPORT"

echo "host_skip_smoke model=$MODEL shim=$SHIM kevlar=$KEVLAR" | tee -a "$REPORT"

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

setup_xdg_json() {
  # $1 app dir name under XDG; $2 fixture path; $3 filenames to write
  local app="$1" src="$2"; shift 2
  local dir="$OUT/xdg-$app"
  rm -rf "$dir"
  mkdir -p "$dir/$app"
  python3 - "$src" "$dir/$app" "$@" <<'PY'
import json, sys
from pathlib import Path
src, dest = Path(sys.argv[1]), Path(sys.argv[2])
names = sys.argv[3:]
d = json.loads(src.read_text())
for prov in (d.get("provider") or {}).values():
    if isinstance(prov, dict):
        opts = prov.setdefault("options", {})
        opts["baseURL"] = "http://127.0.0.1:8091/v1"
        opts["apiKey"] = "local"
d.setdefault("permission", {})["external_directory"] = "deny"
dest.mkdir(parents=True, exist_ok=True)
text = json.dumps(d, indent=2)
for name in names:
    (dest / name).write_text(text)
PY
  echo "$dir"
}

# Reachability
if ! curl -sf --max-time 10 "$SHIM/models" >/dev/null; then
  echo "FATAL: cannot reach shim $SHIM" | tee -a "$REPORT"
  exit 1
fi
if curl -sf --max-time 10 "$KEVLAR/v1/status" >/dev/null; then
  echo "kevlar ok $KEVLAR" | tee -a "$REPORT"
else
  echo "WARN: kevlar status not reachable" | tee -a "$REPORT"
fi

# --- Kilo Code (OpenCode-family; invents absolute paths → seed + shell write) ---
if want kilocode; then
  KILO_BIN="$(command -v kilocode || command -v kilo || true)"
  if [[ -n "$KILO_BIN" ]]; then
    ws="$OUT/workspaces/kilocode"
    rm -rf "$ws"; mkdir -p "$ws"
    printf 'PLACEHOLDER\n' > "$ws/hello_tc.py"
    XDG="$(setup_xdg_json kilo "$FIX/kilo-thinkingcap.json" kilo.json config.json)"
    echo "" | tee -a "$REPORT"
    echo "=== kilocode ===" | tee -a "$REPORT"
    echo "\$ $KILO_BIN run --pure --dir $ws ..." | tee -a "$REPORT"
    (
      cd "$ws"
      env XDG_CONFIG_HOME="$XDG" \
        "$KILO_BIN" run --pure --dir "$ws" --model local/thinkingcap --auto \
        "Use the bash/shell tool exactly once with this command and then stop:
printf \"%s\\n\" \"print(\\\"ThinkingCap-OK\\\")\" > hello_tc.py
Do not invent absolute paths. Do not use the Write tool." \
        </dev/null >"$ws/run.log" 2>&1 &
      wait_artifact $! "$ws"
    )
    if grep -q 'ThinkingCap-OK' "$ws/hello_tc.py" 2>/dev/null; then
      pass "kilocode ($(tr -d '\n' <"$ws/hello_tc.py" | head -c 80))"
    else
      grep -q PLACEHOLDER "$ws/hello_tc.py" 2>/dev/null && rm -f "$ws/hello_tc.py"
      fail kilocode "no hello_tc.py; $(tail -3 "$ws/run.log" 2>/dev/null | tr '\n' ' ' | head -c 220)"
    fi
  else
    skip kilocode "binary not on PATH"
  fi
fi

# --- Mimo (same family as OpenCode) ---
if want mimo-code; then
  if command -v mimo >/dev/null; then
    ws="$OUT/workspaces/mimo-code"
    rm -rf "$ws"; mkdir -p "$ws"
    printf 'PLACEHOLDER\n' > "$ws/hello_tc.py"
    XDG="$(setup_xdg_json mimocode "$FIX/mimo-thinkingcap.json" mimocode.json config.json)"
    echo "" | tee -a "$REPORT"
    echo "=== mimo-code ===" | tee -a "$REPORT"
    echo "\$ mimo run --pure --dir $ws ..." | tee -a "$REPORT"
    (
      cd "$ws"
      env XDG_CONFIG_HOME="$XDG" \
        mimo run --pure --dir "$ws" --model local/thinkingcap \
        --dangerously-skip-permissions \
        "Use the bash/shell tool exactly once with this command and then stop:
printf \"%s\\n\" \"print(\\\"ThinkingCap-OK\\\")\" > hello_tc.py
Do not invent absolute paths. Do not use the Write tool." \
        </dev/null >"$ws/run.log" 2>&1 &
      wait_artifact $! "$ws"
    )
    if grep -q 'ThinkingCap-OK' "$ws/hello_tc.py" 2>/dev/null; then
      pass "mimo-code ($(tr -d '\n' <"$ws/hello_tc.py" | head -c 80))"
    else
      grep -q PLACEHOLDER "$ws/hello_tc.py" 2>/dev/null && rm -f "$ws/hello_tc.py"
      fail mimo-code "no hello_tc.py; $(tail -3 "$ws/run.log" 2>/dev/null | tr '\n' ' ' | head -c 220)"
    fi
  else
    skip mimo-code "binary not on PATH"
  fi
fi

# --- Antigravity (agy) → Kevlar Anthropic ---
if want antigravity; then
  if command -v agy >/dev/null; then
    ws="$OUT/workspaces/antigravity"
    rm -rf "$ws"; mkdir -p "$ws"
    echo "" | tee -a "$REPORT"
    echo "=== antigravity ===" | tee -a "$REPORT"
    echo "\$ agy --print ... --model $MODEL" | tee -a "$REPORT"
    (
      cd "$ws"
      env ANTHROPIC_BASE_URL="$KEVLAR" ANTHROPIC_API_KEY=local \
        agy --print "$PROMPT" --model "$MODEL" \
        --dangerously-skip-permissions --print-timeout 5m \
        </dev/null >"$ws/run.log" 2>&1 &
      wait_artifact $! "$ws"
    )
    if [[ -f "$ws/hello_tc.py" ]] && grep -q 'ThinkingCap-OK' "$ws/hello_tc.py"; then
      pass "antigravity ($(tr -d '\n' <"$ws/hello_tc.py" | head -c 80))"
    else
      fail antigravity "no hello_tc.py; $(tail -5 "$ws/run.log" 2>/dev/null | tr '\n' ' ' | head -c 220)"
    fi
  else
    skip antigravity "agy not on PATH"
  fi
fi

# --- Cursor CLI ---
# Two supported modes:
#   1) Catalog (default): cursor-agent login → Cursor cloud model (not ThinkingCap).
#   2) BYOK ThinkingCap: Cursor Settings Override OpenAI Base URL → public HTTPS
#      tunnel to :8091 (see agent_bench/scripts/cursor_byok_thinkingcap.sh). Then set
#      CURSOR_BYOK_MODEL to the custom model id and CURSOR_BYOK=1.
# Bedrock BYOK is also supported by Cursor, but this stack is OpenAI-shaped via the shim.
if want cursor-cli; then
  CURSOR="$(command -v cursor-agent || true)"
  if [[ -z "$CURSOR" ]]; then
    for c in \
      "$HOME/Library/Application Support/Cursor/User/globalStorage/anysphere.cursor-agent-worker/agent-cli/.local/share/cursor-agent/versions/"*/cursor-agent
    do
      [[ -x "$c" ]] && CURSOR="$c" && break
    done
  fi
  if [[ -n "${CURSOR:-}" && -x "$CURSOR" ]]; then
    status_out="$("$CURSOR" status 2>&1 || true)"
    echo "$status_out" | tee -a "$REPORT" >/dev/null
    if echo "$status_out" | grep -qi 'logged in'; then
      :
    elif [[ -n "${CURSOR_API_KEY:-}" ]]; then
      :
    elif [[ "${CURSOR_FORCE_SMOKE:-}" == "1" ]]; then
      echo "WARN: status not logged in, CURSOR_FORCE_SMOKE=1 — trying anyway" | tee -a "$REPORT"
    else
      skip cursor-cli "not logged in — run: cursor-agent login (or set CURSOR_API_KEY / CURSOR_FORCE_SMOKE=1)"
      status_out=""
    fi
    if echo "$status_out" | grep -qi 'logged in' || [[ -n "${CURSOR_API_KEY:-}" || "${CURSOR_FORCE_SMOKE:-}" == "1" ]]; then
      ws="$OUT/workspaces/cursor-cli"
      rm -rf "$ws"; mkdir -p "$ws"
      echo "" | tee -a "$REPORT"
      echo "=== cursor-cli ===" | tee -a "$REPORT"
      cursor_model="auto"
      cursor_note="catalog"
      if [[ "${CURSOR_BYOK:-}" == "1" ]]; then
        cursor_model="${CURSOR_BYOK_MODEL:-$MODEL}"
        cursor_note="BYOK OpenAI override → ThinkingCap"
        echo "BYOK mode: --model $cursor_model (requires IDE Override OpenAI Base URL → tunnel /v1)" | tee -a "$REPORT"
      fi
      echo "\$ cursor-agent -p ... --model $cursor_model --force  # $cursor_note" | tee -a "$REPORT"
      (
        cd "$ws"
        if [[ "$cursor_model" == "auto" ]]; then
          "$CURSOR" -p "$PROMPT" --force \
            </dev/null >"$ws/run.log" 2>&1 &
        else
          "$CURSOR" -p "$PROMPT" --model "$cursor_model" --force \
            </dev/null >"$ws/run.log" 2>&1 &
        fi
        wait_artifact $! "$ws"
      )
      if [[ -f "$ws/hello_tc.py" ]] && grep -q 'ThinkingCap-OK' "$ws/hello_tc.py"; then
        pass "cursor-cli/$cursor_note ($(tr -d '\n' <"$ws/hello_tc.py" | head -c 80))"
      else
        fail cursor-cli "no hello_tc.py ($cursor_note); $(tail -5 "$ws/run.log" 2>/dev/null | tr '\n' ' ' | head -c 220)"
      fi
    fi
  else
    skip cursor-cli "cursor-agent binary not found"
  fi
fi

echo "" | tee -a "$REPORT"
echo "==== SUMMARY ====" | tee -a "$REPORT"
grep -E '^(PASS|FAIL|SKIP) ' "$REPORT" | tee -a "$REPORT"

python3 - <<'PY'
import json, re
from pathlib import Path
import os
report = Path(os.environ.get("RESULTS_DIR", "")) 
# fall back: script OUT is not in env — hardcode relative from known layout
candidates = [
    Path("/Users/HIESCHA/Projects/Work/llm-bench/results/agent_bench/host_skip_smoke/REPORT.txt"),
    Path("results/agent_bench/host_skip_smoke/REPORT.txt"),
]
report = next((p for p in candidates if p.exists()), None)
if not report:
    raise SystemExit(0)
rows, seen = [], set()
for line in report.read_text().splitlines():
    m = re.match(r'^(PASS|FAIL|SKIP) (\S+)(?:\s+(?:—\s*)?(.*))?$', line)
    if not m: continue
    key = (m.group(1), m.group(2))
    if key in seen: continue
    seen.add(key)
    rows.append({"status": m.group(1).lower(), "name": m.group(2), "detail": (m.group(3) or "").strip()})
out = report.parent / "summary.json"
out.write_text(json.dumps({
    "results": rows,
    "pass": sum(1 for r in rows if r["status"]=="pass"),
    "fail": sum(1 for r in rows if r["status"]=="fail"),
    "skip": sum(1 for r in rows if r["status"]=="skip"),
}, indent=2))
print(f"wrote {out}")
PY
