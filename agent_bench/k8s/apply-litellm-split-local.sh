#!/usr/bin/env bash
# Pin LiteLLM on x40 to local vLLM only; deploy hostNetwork LiteLLM on x39.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
NS=llm-serving
export KUBECONFIG="${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}"
STAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="${BACKUP_DIR:-$HOME/llm-serving/k8s-backups}"
mkdir -p "$BACKUP_DIR"

echo "== backup litellm-config =="
kubectl -n "$NS" get cm litellm-config -o yaml >"$BACKUP_DIR/litellm-config.pre-split-$STAMP.yaml" 2>/dev/null || true

echo "== x40 litellm → vllm-int4 only =="
kubectl -n "$NS" create configmap litellm-config \
  --from-file=config.yaml="$ROOT/litellm-config.bench-x40-only.yaml" \
  -o yaml --dry-run=client | kubectl apply -f -
kubectl -n "$NS" rollout restart deploy/litellm
kubectl -n "$NS" rollout status deploy/litellm --timeout=180s

echo "== x39 litellm hostNetwork :4000 → vllm-int4-x39 only =="
kubectl -n "$NS" apply -f "$ROOT/litellm-x39-host.yaml"
kubectl -n "$NS" rollout status deploy/litellm-x39 --timeout=180s

echo "== pods =="
kubectl -n "$NS" get pods -l 'app in (litellm,litellm-x39,vllm-int4,vllm-int4-x39)' -o wide

echo "Done. x40 Harbor → 127.0.0.1:4000 (x40 vLLM). x39 Harbor → 127.0.0.1:4000 on x39."
