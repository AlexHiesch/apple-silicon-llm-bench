#!/usr/bin/env bash
# Revert MTP/TurboQuant/chat-template extras → last known-good TP2@128k args.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NS=llm-serving
echo "[$(date -Iseconds)] REVERT MTP/TQ → baseline TP2@128k"
if [[ -d "$HOME/llm-serving/k8s" ]]; then
  cp -f "$ROOT/agent_bench/k8s/vllm-int4.bench-tp2-128k.yaml" "$HOME/llm-serving/k8s/"
  cp -f "$ROOT/agent_bench/k8s/vllm-int4.bench-tp2-128k-x39.yaml" "$HOME/llm-serving/k8s/"
fi
if kubectl -n "$NS" get deploy/vllm-int4-x39 >/dev/null 2>&1; then
  kubectl -n "$NS" apply -f "$ROOT/agent_bench/k8s/vllm-int4.bench-tp2-128k-x39.yaml"
  kubectl -n "$NS" rollout status deploy/vllm-int4-x39 --timeout=900s
fi
kubectl -n "$NS" apply -f "$ROOT/agent_bench/k8s/vllm-int4.bench-tp2-128k.yaml"
kubectl -n "$NS" rollout status deploy/vllm-int4 --timeout=900s
rm -f "$ROOT/results/agent_bench/aa_index/MTP_ENABLED.txt"
echo "REVERTED $(date -Iseconds)" | tee "$ROOT/results/agent_bench/aa_index/MTP_REVERTED.txt"
