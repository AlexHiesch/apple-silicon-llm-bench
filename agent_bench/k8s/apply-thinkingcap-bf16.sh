#!/usr/bin/env bash
# Serve bottlecapai ThinkingCap BF16 from ~/llm-model-cache (both nodes).
# Args: [max_model_len=65536]
# Requires BF16 cache present on x40 + x39 under ~/llm-model-cache/huggingface/hub/...
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NS=llm-serving
MAX_LEN="${1:-65536}"
export KUBECONFIG="${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}"
CACHE_HOST="$HOME/llm-model-cache"
SNAP_REL="huggingface/hub/models--bottlecapai--ThinkingCap-Qwen3.6-27B"
SNAP=$(ls -d "$CACHE_HOST/$SNAP_REL/snapshots"/*/ 2>/dev/null | head -1)
[[ -n "$SNAP" ]] || { echo "missing BF16 snapshot under $CACHE_HOST/$SNAP_REL"; exit 1; }
SNAP_NAME=$(basename "${SNAP%/}")
MODEL_PATH="/models-home/$SNAP_REL/snapshots/$SNAP_NAME"

echo "[$(date -Iseconds)] apply BF16 TQ max_model_len=$MAX_LEN path=$MODEL_PATH"

# Point ConfigMap at BF16 (served name unchanged → LiteLLM keeps working)
kubectl -n "$NS" create configmap vllm-int4-config \
  --from-literal=MODEL_PATH="$MODEL_PATH" \
  --from-literal=SERVED_MODEL_NAME=thinkingcap-qwen3.6-27b \
  -o yaml --dry-run=client | kubectl apply -f -

gen_deploy() {
  local src=$1 dst=$2
  python3 - <<PY
from pathlib import Path
src = Path("$src")
dst = Path("$dst")
text = src.read_text()
import re
text = re.sub(r'(llm-bench/mode: ")[^"]+"', r'\1temporary-aa-index-tp2-128k-bf16-tq"', text, count=1)
text = text.replace('- "131072"', f'- "{MAX_LEN}"', 1)
# insert home-cache volume mount
old = """          volumeMounts:
            - mountPath: /models
              name: model-cache
              readOnly: true
      volumes:
        - name: model-cache
          hostPath:
            path: /opt/models
            type: Directory"""
new = """          volumeMounts:
            - mountPath: /models
              name: model-cache
              readOnly: true
            - mountPath: /models-home
              name: model-home
              readOnly: true
      volumes:
        - name: model-cache
          hostPath:
            path: /opt/models
            type: Directory
        - name: model-home
          hostPath:
            path: $CACHE_HOST
            type: Directory"""
if old not in text:
    raise SystemExit("volumeMounts block not found in "+src.name)
dst.write_text(text.replace(old, new))
print("wrote", dst)
PY
}

TMP="$ROOT/agent_bench/k8s/_generated"
mkdir -p "$TMP"
gen_deploy "$ROOT/agent_bench/k8s/vllm-int4.bench-tp2-128k.tq-only.yaml" "$TMP/vllm-bf16-x40.yaml"
gen_deploy "$ROOT/agent_bench/k8s/vllm-int4.bench-tp2-128k-x39.tq-only.yaml" "$TMP/vllm-bf16-x39.yaml"

if kubectl -n "$NS" get deploy/vllm-int4-x39 >/dev/null 2>&1; then
  kubectl -n "$NS" apply -f "$TMP/vllm-bf16-x39.yaml"
  kubectl -n "$NS" rollout status deploy/vllm-int4-x39 --timeout=2400s
fi
kubectl -n "$NS" apply -f "$TMP/vllm-bf16-x40.yaml"
kubectl -n "$NS" rollout status deploy/vllm-int4 --timeout=2400s

echo "  warmup 240s (BF16 load)…"
sleep 240
echo "BF16 TQ max_len=$MAX_LEN @ $(date -Iseconds)" | tee "$ROOT/results/agent_bench/aa_index/SERVING_MODE.txt"
echo "[$(date -Iseconds)] BF16 apply done"
