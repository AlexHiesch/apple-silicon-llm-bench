#!/usr/bin/env bash
# Enable MTP + TurboQuant KV on dual TP2@128k ThinkingCap INT4 pods.
# Rolling: x39 first (x40 keeps serving), then x40.
# Waits for Ready + GPU warmup, then content smoke (max_tokens=512).
# On smoke failure after warmup → automatic revert.
# Skips DFlash (incompatible with TurboQuant).
#
# vLLM 0.24: TQ×MTP with FULL cudagraph emits degenerate loops
# (upstream vllm#40831 / #40880). Recipes use --enforce-eager so both
# stay correct until a fixed vLLM image is available.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NS=llm-serving
WARMUP_SEC="${MTP_WARMUP_SEC:-180}"
MTP_X40="$ROOT/agent_bench/k8s/vllm-int4.bench-tp2-128k.mtp.yaml"
MTP_X39="$ROOT/agent_bench/k8s/vllm-int4.bench-tp2-128k-x39.mtp.yaml"

echo "[$(date -Iseconds)] enable MTP + turboquant_4bit_nc + enforce-eager (vllm#40831 workaround)"

[[ -f "$MTP_X40" && -f "$MTP_X39" ]] || { echo "missing .mtp.yaml recipes"; exit 1; }
cp -f "$MTP_X40" "$ROOT/agent_bench/k8s/vllm-int4.bench-tp2-128k.yaml"
cp -f "$MTP_X39" "$ROOT/agent_bench/k8s/vllm-int4.bench-tp2-128k-x39.yaml"
if [[ -d "$HOME/llm-serving/k8s" ]]; then
  cp -f "$MTP_X40" "$HOME/llm-serving/k8s/vllm-int4.bench-tp2-128k.yaml"
  cp -f "$MTP_X39" "$HOME/llm-serving/k8s/vllm-int4.bench-tp2-128k-x39.yaml"
fi

wait_ready() {
  local deploy=$1
  echo "  wait $deploy ready (progressDeadline up to 30m)…"
  kubectl -n "$NS" rollout status "deploy/$deploy" --timeout=1800s
  kubectl -n "$NS" get pod -l "app=$deploy" -o wide
  # probes can flip Ready before compile/cudagraph fully settled
  echo "  warmup ${WARMUP_SEC}s after Ready…"
  sleep "$WARMUP_SEC"
}

smoke_content() {
  echo "  content smoke (LiteLLM + direct vLLM; max_tokens=512)…"
  KEY=$(kubectl -n "$NS" exec deploy/litellm -- printenv LITELLM_MASTER_KEY)
  export KEY
  python3 - <<'PY'
import json, urllib.request, os, sys, subprocess

def post(url, body, headers, timeout=300):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

key = os.environ["KEY"]
prompt = "Reply with exactly the word PONG and nothing else."
base_body = {
    "model": "thinkingcap",
    "max_tokens": 512,
    "temperature": 0,
    "messages": [{"role": "user", "content": prompt}],
}

def ok_content(content):
    c = (content or "").strip()
    if not c:
        return False
    if c.count("\n") > 40:
        return False
    return True

# 1) LiteLLM OpenAI
try:
    data = post(
        "http://127.0.0.1:4000/v1/chat/completions",
        base_body,
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    msg = (data.get("choices") or [{}])[0].get("message") or {}
    c, r = msg.get("content"), msg.get("reasoning_content")
    print("litellm content=", repr(c)[:120], "reason=", repr((r or "")[:60]))
    litellm_ok = ok_content(c)
except Exception as e:
    print("litellm FAIL", e)
    litellm_ok = False

# 2) LiteLLM with thinking off (request-level)
try:
    body2 = dict(base_body)
    body2["chat_template_kwargs"] = {"enable_thinking": False}
    data = post(
        "http://127.0.0.1:4000/v1/chat/completions",
        body2,
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    msg = (data.get("choices") or [{}])[0].get("message") or {}
    c = msg.get("content")
    print("litellm_nothink content=", repr(c)[:120])
    litellm_nothink_ok = ok_content(c)
except Exception as e:
    print("litellm_nothink FAIL", e)
    litellm_nothink_ok = False

# 3) Direct vLLM (bypass LiteLLM)
ip = subprocess.check_output(
    ["kubectl", "-n", "llm-serving", "get", "svc", "vllm-int4", "-o", "jsonpath={.spec.clusterIP}"],
    text=True,
).strip()
try:
    body3 = {
        "model": "thinkingcap-qwen3.6-27b",
        "max_tokens": 512,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
    }
    data = post(f"http://{ip}:8000/v1/chat/completions", body3, {"Content-Type": "application/json"})
    msg = (data.get("choices") or [{}])[0].get("message") or {}
    c = msg.get("content")
    print("direct content=", repr(c)[:120], "reason=", repr((msg.get("reasoning_content") or "")[:60]))
    direct_ok = ok_content(c)
except Exception as e:
    print("direct FAIL", e)
    direct_ok = False

# Pass if any agent-usable path returns content. Prefer litellm default.
if litellm_ok or litellm_nothink_ok or direct_ok:
    print("SMOKE_OK", {"litellm": litellm_ok, "litellm_nothink": litellm_nothink_ok, "direct": direct_ok})
    # Hard fail only if LiteLLM default path is empty AND direct also empty
    # (agents use LiteLLM with thinking on — need litellm_ok for overnight)
    if not litellm_ok and not direct_ok:
        sys.exit(2)
    if not litellm_ok:
        print("WARN: LiteLLM default thinking path empty; agents may suffer")
        # still fail for AA — Claude uses thinking
        sys.exit(3)
    sys.exit(0)
print("SMOKE_FAIL all paths empty")
sys.exit(1)
PY
}

if kubectl -n "$NS" get deploy/vllm-int4-x39 >/dev/null 2>&1; then
  echo "  apply x39 (MTP + TQ)"
  kubectl -n "$NS" apply -f "$ROOT/agent_bench/k8s/vllm-int4.bench-tp2-128k-x39.mtp.yaml"
  wait_ready vllm-int4-x39
fi

echo "  apply x40 (MTP + TQ)"
kubectl -n "$NS" apply -f "$ROOT/agent_bench/k8s/vllm-int4.bench-tp2-128k.mtp.yaml"
wait_ready vllm-int4

echo "  verify live args"
for dep in vllm-int4 vllm-int4-x39; do
  kubectl -n "$NS" get deploy "$dep" >/dev/null 2>&1 || continue
  echo -n "  $dep: "
  kubectl -n "$NS" get deploy "$dep" -o jsonpath='{.spec.template.spec.containers[0].args}' \
    | tr ',' '\n' | grep -E 'speculative|turboquant|enforce-eager|kv-cache|batched' || echo "(flags missing!)"
done

if ! smoke_content; then
  echo "INTERVENE: smoke failed after warmup — reverting"
  bash "$ROOT/agent_bench/k8s/revert-thinkingcap-mtp.sh"
  exit 1
fi

mkdir -p "$ROOT/results/agent_bench/aa_index"
{
  echo "MTP + TurboQuant + enforce-eager enabled $(date -Iseconds)"
  echo "kv-cache-dtype: turboquant_4bit_nc"
  echo "speculative-config: method=mtp num_speculative_tokens=1"
  echo "max-num-batched-tokens: 4096"
  echo "enforce-eager: yes (workaround vllm#40831/#40880 TQ×MTP×cudagraph)"
  echo "warmup_sec: $WARMUP_SEC"
  echo "skipped: DFlash"
} | tee "$ROOT/results/agent_bench/aa_index/MTP_ENABLED.txt"
rm -f "$ROOT/results/agent_bench/aa_index/MTP_REVERTED.txt"

echo "[$(date -Iseconds)] MTP + TurboQuant enable done (smoke OK)"
