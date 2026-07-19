#!/usr/bin/env bash
# Restore ConfigMap MODEL_PATH to josefprusa INT4 AutoRound (LiteLLM name unchanged).
set -euo pipefail
NS=llm-serving
export KUBECONFIG="${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}"
PATH_INT4=/models/huggingface/hub/models--josefprusa--ThinkingCap-Qwen3.6-27B-int4-AutoRound-v1
echo "[$(date -Iseconds)] restore INT4 configmap → $PATH_INT4"
kubectl -n "$NS" create configmap vllm-int4-config \
  --from-literal=MODEL_PATH="$PATH_INT4" \
  --from-literal=SERVED_MODEL_NAME=thinkingcap-qwen3.6-27b \
  -o yaml --dry-run=client | kubectl apply -f -
echo "done"
