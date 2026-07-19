#!/usr/bin/env bash
# Enable MTP + TurboQuant KV on dual TP2@128k ThinkingCap INT4 pods.
# Rolling: x39 first (x40 keeps serving), then x40.
# Skips FP8 weights / DFlash (Ampere + 128k overnight path; DFlash ≠ TurboQuant).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NS=llm-serving

echo "[$(date -Iseconds)] enable MTP + turboquant_4bit_nc on vllm-int4 (+ x39 if present)"

if [[ -d "$HOME/llm-serving/k8s" ]]; then
  cp -f "$ROOT/agent_bench/k8s/vllm-int4.bench-tp2-128k.yaml" \
    "$HOME/llm-serving/k8s/vllm-int4.bench-tp2-128k.yaml"
  cp -f "$ROOT/agent_bench/k8s/vllm-int4.bench-tp2-128k-x39.yaml" \
    "$HOME/llm-serving/k8s/vllm-int4.bench-tp2-128k-x39.yaml"
fi

wait_ready() {
  local deploy=$1
  echo "  wait $deploy ready…"
  kubectl -n "$NS" rollout status "deploy/$deploy" --timeout=900s
  kubectl -n "$NS" get pod -l "app=$deploy" -o wide
}

if kubectl -n "$NS" get deploy/vllm-int4-x39 >/dev/null 2>&1; then
  echo "  apply x39 (MTP + TQ)"
  kubectl -n "$NS" apply -f "$ROOT/agent_bench/k8s/vllm-int4.bench-tp2-128k-x39.yaml"
  wait_ready vllm-int4-x39
fi

echo "  apply x40 (MTP + TQ)"
kubectl -n "$NS" apply -f "$ROOT/agent_bench/k8s/vllm-int4.bench-tp2-128k.yaml"
wait_ready vllm-int4

echo "  verify live args"
for dep in vllm-int4 vllm-int4-x39; do
  kubectl -n "$NS" get deploy "$dep" >/dev/null 2>&1 || continue
  echo -n "  $dep: "
  kubectl -n "$NS" get deploy "$dep" -o jsonpath='{.spec.template.spec.containers[0].args}' \
    | tr ',' '\n' | grep -E 'speculative|turboquant|default-chat|kv-cache' || echo "(flags missing!)"
done

mkdir -p "$ROOT/results/agent_bench/aa_index"
{
  echo "MTP + TurboQuant enabled $(date -Iseconds)"
  echo "kv-cache-dtype: turboquant_4bit_nc"
  echo "speculative-config: method=mtp num_speculative_tokens=1"
  echo "default-chat-template-kwargs: enable_thinking+preserve_thinking"
  echo "skipped: FP8 model switch, DFlash"
} | tee "$ROOT/results/agent_bench/aa_index/MTP_ENABLED.txt"

echo "[$(date -Iseconds)] MTP + TurboQuant enable done"
