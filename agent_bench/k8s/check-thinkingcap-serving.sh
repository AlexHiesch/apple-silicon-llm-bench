#!/usr/bin/env bash
# Smoke ThinkingCap via LiteLLM; auto-revert MTP/TQ deploy if content broken.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NS=llm-serving
KEY=$(kubectl -n "$NS" exec deploy/litellm -- printenv LITELLM_MASTER_KEY)
export KEY

python3 - <<'PY'
import json,urllib.request,os,sys
key=os.environ["KEY"]
body={
  "model":"thinkingcap",
  "max_tokens":512,
  "temperature":0,
  "messages":[{"role":"user","content":"Reply with exactly the word PONG and nothing else."}],
}
req=urllib.request.Request(
  "http://127.0.0.1:4000/v1/chat/completions",
  data=json.dumps(body).encode(),
  headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},
)
with urllib.request.urlopen(req, timeout=180) as r:
  data=json.loads(r.read())
msg=(data.get("choices") or [{}])[0].get("message") or {}
content=msg.get("content") or ""
reasoning=msg.get("reasoning_content") or ""
print("content=", repr(content)[:120], "reasoning_prefix=", repr(reasoning[:80]))
# Fail if no usable content, or obvious garbage loop
if not str(content).strip():
  print("HEALTH_FAIL empty content")
  sys.exit(2)
if "empty, empty" in reasoning or str(content).count("\n")>40:
  print("HEALTH_FAIL garbage output")
  sys.exit(2)
if "PONG" not in str(content).upper() and len(str(content).strip())<2:
  print("HEALTH_FAIL unexpected content")
  sys.exit(2)
print("HEALTH_OK")
PY
rc=$?

if [[ $rc -ne 0 ]]; then
  echo "INTERVENE: serving unhealthy — revert MTP/TQ if present"
  if kubectl -n "$NS" get deploy/vllm-int4 -o jsonpath='{.spec.template.spec.containers[0].args}' | grep -q turboquant; then
    bash "$ROOT/agent_bench/k8s/revert-thinkingcap-mtp.sh"
  else
    echo "already on baseline; check LiteLLM/vLLM manually"
  fi
  exit $rc
fi

# AA liveness
if ! pgrep -u "$USER" -f agent_bench.run_matrix >/dev/null; then
  echo "INTERVENE: matrix dead — overnight_babysit"
  bash "$ROOT/agent_bench/scripts/overnight_babysit.sh"
fi
exit 0
