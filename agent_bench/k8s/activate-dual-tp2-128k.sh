#!/usr/bin/env bash
# Activate dual-node TEMP bench: TP2@128k on x40 (existing) + x39 (new).
# Safe for results: does not delete Harbor job dirs. Runner restart is separate.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
NS=llm-serving
OBS=llm-observability
BACKUP_DIR="${HOME}/llm-serving/k8s-backups"
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$BACKUP_DIR"

echo "== backup litellm + prometheus =="
kubectl -n "$NS" get cm litellm-config -o yaml >"$BACKUP_DIR/litellm-config.pre-dual-$STAMP.yaml"
kubectl -n "$OBS" get cm prometheus-config -o yaml >"$BACKUP_DIR/prometheus-config.pre-dual-$STAMP.yaml"

echo "== deploy vllm-int4-x39 (TP2@128k) =="
kubectl -n "$NS" apply -f "$ROOT/vllm-int4.bench-tp2-128k-x39.yaml"
echo "Waiting for x39 vLLM Ready (model load can take several minutes)…"
kubectl -n "$NS" rollout status deploy/vllm-int4-x39 --timeout=1200s

echo "== LiteLLM dual backend =="
kubectl -n "$NS" create configmap litellm-config \
  --from-file=config.yaml="$ROOT/litellm-config.bench-dual-tp2.yaml" \
  -o yaml --dry-run=client | kubectl apply -f -
kubectl -n "$NS" rollout restart deploy/litellm
kubectl -n "$NS" rollout status deploy/litellm --timeout=180s

echo "== Prometheus scrape labels (node_short / bench_node) =="
kubectl -n "$OBS" create configmap prometheus-config \
  --from-file=prometheus.yml="$ROOT/prometheus.yml.bench-dual" \
  -o yaml --dry-run=client | kubectl apply -f -
kubectl -n "$OBS" rollout restart deploy/prometheus
kubectl -n "$OBS" rollout status deploy/prometheus --timeout=180s

echo "== label x40 vLLM pod for dashboards =="
POD40=$(kubectl -n "$NS" get pods -l app=vllm-int4,llm-bench-node!=x39 -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
if [[ -z "${POD40:-}" ]]; then
  POD40=$(kubectl -n "$NS" get pods -l app=vllm-int4 -o jsonpath='{.items[0].metadata.name}')
fi
kubectl -n "$NS" label pod "$POD40" llm-bench-node=x40 --overwrite || true

echo "== status =="
kubectl -n "$NS" get deploy,pods,svc -l 'app in (vllm-int4,vllm-int4-x39)' -o wide
kubectl -n "$NS" get pods -l app=litellm -o wide
echo "Done. Next: patch-grafana-dual-dashboard.sh + smoke + runner n=4 restart."
