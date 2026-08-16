#!/usr/bin/env bash
# Apply dual-node LiteLLM routing (least-busy + session_affinity) without
# touching vLLM deployments or Harbor job dirs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
NS=llm-serving
BACKUP_DIR="${HOME}/llm-serving/k8s-backups"
STAMP=$(date +%Y%m%d_%H%M%S)
CFG="$ROOT/litellm-config.bench-dual-tp2.yaml"
mkdir -p "$BACKUP_DIR"

if [[ ! -f "$CFG" ]]; then
  echo "FATAL: missing $CFG" >&2
  exit 1
fi

# Guardrails: never ship key-affinity or shuffle by accident.
if grep -qE '^\s*routing_strategy:\s*simple-shuffle' "$CFG"; then
  echo "FATAL: config still uses simple-shuffle" >&2
  exit 1
fi
if grep -qE '^\s*-\s*deployment_affinity\s*$' "$CFG"; then
  echo "FATAL: deployment_affinity would pin shared API key to one node" >&2
  exit 1
fi
if ! grep -qE '^\s*routing_strategy:\s*least-busy\s*$' "$CFG"; then
  echo "FATAL: expected routing_strategy: least-busy" >&2
  exit 1
fi
if ! grep -qE '^\s*-\s*session_affinity\s*$' "$CFG"; then
  echo "FATAL: expected session_affinity in optional_pre_call_checks" >&2
  exit 1
fi

echo "== backup current litellm-config =="
kubectl -n "$NS" get cm litellm-config -o yaml >"$BACKUP_DIR/litellm-config.pre-routing-$STAMP.yaml"

echo "== apply $CFG =="
kubectl -n "$NS" create configmap litellm-config \
  --from-file=config.yaml="$CFG" \
  -o yaml --dry-run=client | kubectl apply -f -

echo "== rollout litellm =="
kubectl -n "$NS" rollout restart deploy/litellm
kubectl -n "$NS" rollout status deploy/litellm --timeout=180s

echo "== live config snippet =="
kubectl -n "$NS" get cm litellm-config -o jsonpath='{.data.config\.yaml}' | sed -n '1,45p'
echo
echo "Done. Smoke: bash $ROOT/smoke-litellm-session-affinity.sh"
