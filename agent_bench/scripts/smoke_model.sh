#!/usr/bin/env bash
# Smoke-test from inside a clean container against an OpenAI-compatible endpoint.
set -euo pipefail

BASE_URL="${OPENAI_BASE_URL:-http://host.docker.internal:8080/v1}"
MODEL="${LLM_MODEL:-t-prazak/ThinkingCap-Qwen3.6-27B-MLX-4bit}"
OUT_DIR="${RESULTS_DIR:-/results}"

echo "== agent_bench sandbox smoke =="
echo "BASE_URL=$BASE_URL"
echo "MODEL=$MODEL"
mkdir -p "$OUT_DIR" 2>/dev/null || true

echo "-- GET /models --"
if ! curl -sf --max-time 5 "${BASE_URL}/models" -o /tmp/models.json; then
  echo "FAIL: cannot reach model server at ${BASE_URL}/models"
  echo "For real runs, start host ThinkingCap:"
  echo "  python -m mlx_lm.server --model $MODEL --port 8080"
  echo "For docker-only smoke:"
  echo "  docker compose -f agent_bench/docker-compose.yml --profile smoke up --build --abort-on-container-exit"
  exit 1
fi

echo "models ok:"
python3 -c 'import json; d=json.load(open("/tmp/models.json")); print(" ",[m.get("id") for m in d.get("data",[])])'

echo "-- POST /chat/completions --"
curl -sf --max-time 60 "${BASE_URL}/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${OPENAI_API_KEY:-local}" \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: pong\"}],\"max_tokens\":16,\"temperature\":0}" \
  -o /tmp/chat.json

python3 - <<'PY'
import json
data = json.load(open("/tmp/chat.json"))
content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
print("assistant:", repr(content)[:200])
if not content:
    raise SystemExit("FAIL: empty completion")
print("PASS: sandbox reached model endpoint")
PY

cp /tmp/chat.json "${OUT_DIR}/sandbox_smoke.json" 2>/dev/null || true
