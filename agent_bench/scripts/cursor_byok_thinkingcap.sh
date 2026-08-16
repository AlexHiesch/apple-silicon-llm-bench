#!/usr/bin/env bash
# Point Cursor's custom OpenAI endpoint (Override OpenAI Base URL) at local ThinkingCap.
#
# Cursor BYOK is not a direct localhost client: Chat/Agent build prompts on Cursor's
# servers, then the servers call your OpenAI-compatible base URL. So:
#   1) Kevlar :8080 + OpenAI shim :8091 must be up
#   2) Expose :8091 as a public HTTPS URL (cloudflared quick tunnel works)
#   3) Cursor Settings → Models → OpenAI API Key + Override OpenAI Base URL = $TUNNEL/v1
#   4) Add custom model id exactly as served by the shim
#   5) Select that model in Chat/Agent
#
# Bedrock is a separate Cursor BYOK path (Settings → Bedrock). This stack is
# OpenAI-shaped via the shim; use OpenAI override, not Bedrock, for ThinkingCap.
#
# Usage:
#   bash agent_bench/scripts/cursor_byok_thinkingcap.sh
#   bash agent_bench/scripts/cursor_byok_thinkingcap.sh --write-settings   # sets openAIBaseUrl in Cursor state
#   TUNNEL_URL=https://….trycloudflare.com bash …   # reuse an existing tunnel

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${RESULTS_DIR:-$ROOT/results/agent_bench}/cursor_byok"
MODEL="${MODEL:-t-prazak/ThinkingCap-Qwen3.6-27B-MLX-4bit}"
SHIM="${SHIM:-http://127.0.0.1:8091}"
SHIM_V1="${SHIM%/}/v1"
WRITE_SETTINGS=0
KEEP_TUNNEL=1

for arg in "$@"; do
  case "$arg" in
    --write-settings) WRITE_SETTINGS=1 ;;
    --no-keep-tunnel) KEEP_TUNNEL=0 ;;
    -h|--help)
      sed -n '1,20p' "$0"
      exit 0
      ;;
  esac
done

mkdir -p "$OUT"
REPORT="$OUT/REPORT.txt"
: >"$REPORT"
log() { echo "$*" | tee -a "$REPORT"; }

log "cursor_byok_thinkingcap model=$MODEL shim=$SHIM_V1"
log "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

if ! curl -fsS -m 5 "$SHIM_V1/models" >/dev/null; then
  log "FAIL shim — start: PYTHONPATH=. .venv/bin/python -m agent_bench.openai_anthropic_shim --port 8091 --upstream http://127.0.0.1:8080"
  exit 1
fi
log "PASS shim reachable"

# Probe that ThinkingCap answers OpenAI chat completions (what Cursor BYOK calls).
probe="$(curl -fsS -m 120 "$SHIM_V1/chat/completions" \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer local' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly ThinkingCap-OK\"}],\"max_tokens\":32}" \
  | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"])' 2>/dev/null || true)"
if [[ "$probe" == *ThinkingCap-OK* ]]; then
  log "PASS shim chat → ThinkingCap ($probe)"
else
  log "FAIL shim chat probe: ${probe:-empty}"
  exit 1
fi

TUNNEL_URL="${TUNNEL_URL:-}"
CF_PID=""
if [[ -z "$TUNNEL_URL" ]]; then
  if ! command -v cloudflared >/dev/null 2>&1; then
    log "FAIL cloudflared not installed (brew install cloudflared) — or set TUNNEL_URL="
    exit 1
  fi
  CF_LOG="$OUT/cloudflared.log"
  : >"$CF_LOG"
  # Prefer HTTP/2 edge protocol — QUIC quick tunnels sometimes drop immediately here.
  cloudflared tunnel --url "$SHIM" --no-autoupdate --protocol http2 >"$CF_LOG" 2>&1 &
  CF_PID=$!
  log "cloudflared pid=$CF_PID (log $CF_LOG)"
  for _ in $(seq 1 60); do
    TUNNEL_URL="$(rg -o 'https://[a-zA-Z0-9.-]+\.trycloudflare\.com' "$CF_LOG" 2>/dev/null | head -1 || true)"
    if [[ -n "$TUNNEL_URL" ]] && rg -q 'Registered tunnel connection|Connection registered' "$CF_LOG" 2>/dev/null; then
      break
    fi
    if ! kill -0 "$CF_PID" 2>/dev/null; then
      log "FAIL cloudflared exited early — see $CF_LOG"
      exit 1
    fi
    sleep 1
  done
  if [[ -z "$TUNNEL_URL" ]]; then
    log "FAIL no trycloudflare URL in $CF_LOG"
    kill "$CF_PID" 2>/dev/null || true
    exit 1
  fi
fi

BASE="${TUNNEL_URL%/}/v1"
log "tunnel=$TUNNEL_URL"
log "openai_base_url=$BASE"

# Public reachability (DNS for new quick tunnels can lag). Never block forever.
# Only probe when dig returns an A record — bare curl DNS can hang on NXDOMAIN/resolvers.
ok=0
for i in $(seq 1 15); do
  if [[ -n "${CF_PID:-}" ]] && ! kill -0 "$CF_PID" 2>/dev/null; then
    log "WARN cloudflared died during probe — see $OUT/cloudflared.log"
    break
  fi
  host="${TUNNEL_URL#https://}"
  ip="$(dig +time=1 +tries=1 +short "$host" A 2>/dev/null | head -1 || true)"
  if [[ -n "$ip" ]]; then
    if curl -fsS --connect-timeout 3 -m 10 --resolve "$host:443:$ip" "$BASE/models" >/dev/null 2>&1; then
      ok=1
      log "PASS public tunnel /v1/models (attempt $i)"
      break
    fi
  fi
  sleep 1
done
if [[ "$ok" != "1" ]]; then
  log "WARN public tunnel not verified from this host yet — paste $BASE into Cursor once the tunnel is up"
fi

if [[ "$WRITE_SETTINGS" == "1" ]]; then
  python3 - "$BASE" "$MODEL" <<'PY' | tee -a "$REPORT"
import json, sqlite3, sys
from pathlib import Path

base_url, model = sys.argv[1], sys.argv[2]
db = Path.home() / "Library/Application Support/Cursor/User/globalStorage/state.vscdb"
key = "src.vs.platform.reactivestorage.browser.reactiveStorageServiceImpl.persistentStorage.applicationUser"
con = sqlite3.connect(str(db))
row = con.execute("SELECT value FROM ItemTable WHERE key=?", (key,)).fetchone()
if not row:
    print("FAIL Cursor state.vscdb missing reactiveStorage key")
    raise SystemExit(1)
obj = json.loads(row[0])
obj["openAIBaseUrl"] = base_url
obj["useOpenAIKey"] = True
models = list(obj.get("aiSettings", {}).get("userAddedModels") or [])
if model not in models:
    models.append(model)
obj.setdefault("aiSettings", {})["userAddedModels"] = models
# Keep a short alias without slashes for Cursor model-name sanitization quirks.
alias = "thinkingcap-qwen36-27b"
if alias not in models:
    models.append(alias)
    obj["aiSettings"]["userAddedModels"] = models
con.execute("UPDATE ItemTable SET value=? WHERE key=?", (json.dumps(obj, separators=(",", ":")), key))
con.commit()
con.close()
print(f"PASS wrote openAIBaseUrl={base_url} useOpenAIKey=true userAddedModels+=[{model}, {alias}]")
print("NOTE: still set OpenAI API Key in Cursor Settings → Models (any non-empty value, e.g. local)")
print("NOTE: restart Cursor or reload window so settings are picked up")
PY
fi

log ""
log "==== Cursor IDE steps (OpenAI custom endpoint) ===="
log "1. Cursor Settings → Models"
log "2. OpenAI API Key = local  (or any non-empty placeholder; shim does not validate)"
log "3. Enable Override OpenAI Base URL = $BASE"
log "4. Add model: $MODEL"
log "5. In Chat/Agent, select that model and ask it to create hello_tc.py printing ThinkingCap-OK"
log ""
log "Bedrock: Cursor also supports AWS Bedrock BYOK in Settings. This ThinkingCap stack is"
log "OpenAI-compatible via :8091 — use OpenAI override, not Bedrock, unless you front a Bedrock-shaped proxy."
log ""
log "agent-cli-local note: hidden --base-url / --enable-bedrock flags exist for true localhost,"
log "but stock cursor-agent needs the IDE LocalAgentClient runtime. Prefer IDE Override OpenAI Base URL."
log ""
log "tunnel_url=$TUNNEL_URL" >"$OUT/tunnel.env"
log "openai_base_url=$BASE" >>"$OUT/tunnel.env"
log "model=$MODEL" >>"$OUT/tunnel.env"
log "wrote $OUT/tunnel.env"

if [[ -n "$CF_PID" && "$KEEP_TUNNEL" == "1" ]]; then
  log "KEEPING cloudflared pid=$CF_PID — stop with: kill $CF_PID"
elif [[ -n "$CF_PID" ]]; then
  kill "$CF_PID" 2>/dev/null || true
  log "stopped cloudflared"
fi

log "done report=$REPORT"
