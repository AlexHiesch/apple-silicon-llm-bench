#!/usr/bin/env bash
# Revert MTP/TurboQuant → baseline TP2@128k (no speculative / no TQ).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NS=llm-serving
BASE_X40="$ROOT/agent_bench/k8s/vllm-int4.bench-tp2-128k.baseline.yaml"
BASE_X39="$ROOT/agent_bench/k8s/vllm-int4.bench-tp2-128k-x39.baseline.yaml"
echo "[$(date -Iseconds)] REVERT MTP/TQ → baseline TP2@128k"
[[ -f "$BASE_X40" && -f "$BASE_X39" ]] || { echo "missing baseline yaml"; exit 1; }
# also restore active manifests to baseline so next apply is clean
cp -f "$BASE_X40" "$ROOT/agent_bench/k8s/vllm-int4.bench-tp2-128k.yaml"
cp -f "$BASE_X39" "$ROOT/agent_bench/k8s/vllm-int4.bench-tp2-128k-x39.yaml"
if [[ -d "$HOME/llm-serving/k8s" ]]; then
  cp -f "$BASE_X40" "$HOME/llm-serving/k8s/vllm-int4.bench-tp2-128k.yaml"
  cp -f "$BASE_X39" "$HOME/llm-serving/k8s/vllm-int4.bench-tp2-128k-x39.yaml"
fi
if kubectl -n "$NS" get deploy/vllm-int4-x39 >/dev/null 2>&1; then
  kubectl -n "$NS" apply -f "$BASE_X39"
  kubectl -n "$NS" rollout status deploy/vllm-int4-x39 --timeout=1800s
fi
kubectl -n "$NS" apply -f "$BASE_X40"
kubectl -n "$NS" rollout status deploy/vllm-int4 --timeout=1800s
rm -f "$ROOT/results/agent_bench/aa_index/MTP_ENABLED.txt"
echo "REVERTED $(date -Iseconds)" | tee "$ROOT/results/agent_bench/aa_index/MTP_REVERTED.txt"
