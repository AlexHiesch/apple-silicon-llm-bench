#!/usr/bin/env bash
# Remove x39 TP2 replica and restore single-backend LiteLLM (x40 Service only).
# Does NOT revert x40 TP2@128k → use ~/llm-serving/k8s/revert-vllm-prod-tp1.sh for that.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
NS=llm-serving
OBS=llm-observability

kubectl -n "$NS" delete deploy/vllm-int4-x39 svc/vllm-int4-x39 --ignore-not-found

# Restore single-backend litellm (x40 only, timeout 1800)
cat >"/tmp/litellm-single.yaml" <<'EOF'
model_list:
  - model_name: thinkingcap
    litellm_params:
      model: openai/thinkingcap-qwen3.6-27b
      api_base: http://vllm-int4.llm-serving.svc.cluster.local:8000/v1
      api_key: sk-k8s-local-vllm

litellm_settings:
  request_timeout: 1800
  set_verbose: false
  json_logs: true
  drop_params: true
  callbacks: ["prometheus"]
  require_auth_for_metrics_endpoint: false

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: os.environ/DATABASE_URL
  disable_error_logs: true
  allow_requests_on_db_unavailable: true
EOF
kubectl -n "$NS" create configmap litellm-config \
  --from-file=config.yaml=/tmp/litellm-single.yaml \
  -o yaml --dry-run=client | kubectl apply -f -
kubectl -n "$NS" rollout restart deploy/litellm

echo "x39 vLLM removed; LiteLLM back to x40-only."
