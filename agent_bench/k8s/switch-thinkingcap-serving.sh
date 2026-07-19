#!/usr/bin/env bash
# Switch dual TP2@128k ThinkingCap INT4 serving mode.
# Modes: mtp-eager | tq-only | baseline
# Usage: KUBECONFIG=/etc/rancher/k3s/k3s.yaml bash switch-thinkingcap-serving.sh tq-only
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NS=llm-serving
MODE="${1:-}"
WARMUP_SEC="${WARMUP_SEC:-180}"
export KUBECONFIG="${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}"

case "$MODE" in
  mtp-eager|mtp)
    X40="$ROOT/agent_bench/k8s/vllm-int4.bench-tp2-128k.mtp-eager.yaml"
    X39="$ROOT/agent_bench/k8s/vllm-int4.bench-tp2-128k-x39.mtp-eager.yaml"
    MARKER_MSG="MTP + TurboQuant + enforce-eager"
    ;;
  tq-only|tq)
    X40="$ROOT/agent_bench/k8s/vllm-int4.bench-tp2-128k.tq-only.yaml"
    X39="$ROOT/agent_bench/k8s/vllm-int4.bench-tp2-128k-x39.tq-only.yaml"
    MARKER_MSG="TurboQuant only (no MTP, no enforce-eager)"
    ;;
  baseline|base)
    X40="$ROOT/agent_bench/k8s/vllm-int4.bench-tp2-128k.baseline.yaml"
    X39="$ROOT/agent_bench/k8s/vllm-int4.bench-tp2-128k-x39.baseline.yaml"
    MARKER_MSG="baseline TP2@128k (no TQ, no MTP)"
    ;;
  *)
    echo "usage: $0 {mtp-eager|tq-only|baseline}"
    exit 2
    ;;
esac

[[ -f "$X40" && -f "$X39" ]] || { echo "missing recipe yaml"; exit 1; }

echo "[$(date -Iseconds)] switch → $MODE ($MARKER_MSG)"

# Keep active manifests in sync for humans / other scripts.
cp -f "$X40" "$ROOT/agent_bench/k8s/vllm-int4.bench-tp2-128k.yaml"
cp -f "$X39" "$ROOT/agent_bench/k8s/vllm-int4.bench-tp2-128k-x39.yaml"
if [[ -d "$HOME/llm-serving/k8s" ]]; then
  cp -f "$X40" "$HOME/llm-serving/k8s/vllm-int4.bench-tp2-128k.yaml"
  cp -f "$X39" "$HOME/llm-serving/k8s/vllm-int4.bench-tp2-128k-x39.yaml"
fi

wait_ready() {
  local deploy=$1
  echo "  wait $deploy…"
  kubectl -n "$NS" rollout status "deploy/$deploy" --timeout=1800s
  kubectl -n "$NS" get pod -l "app=$deploy" -o wide
}

if kubectl -n "$NS" get deploy/vllm-int4-x39 >/dev/null 2>&1; then
  kubectl -n "$NS" apply -f "$X39"
  wait_ready vllm-int4-x39
fi
kubectl -n "$NS" apply -f "$X40"
wait_ready vllm-int4

echo "  warmup ${WARMUP_SEC}s…"
sleep "$WARMUP_SEC"

echo "  live flags:"
for dep in vllm-int4 vllm-int4-x39; do
  kubectl -n "$NS" get deploy "$dep" >/dev/null 2>&1 || continue
  echo -n "  $dep: "
  kubectl -n "$NS" get deploy "$dep" -o jsonpath='{.spec.template.spec.containers[0].args}' \
    | tr ',' '\n' | grep -E 'speculative|turboquant|enforce-eager|kv-cache|batched' \
    || echo "(none of speculative/tq/eager)"
done

mkdir -p "$ROOT/results/agent_bench/aa_index"
{
  echo "$MARKER_MSG @ $(date -Iseconds)"
  echo "mode=$MODE"
  echo "warmup_sec=$WARMUP_SEC"
} | tee "$ROOT/results/agent_bench/aa_index/SERVING_MODE.txt"

if [[ "$MODE" == mtp-eager || "$MODE" == mtp ]]; then
  cp -f "$ROOT/results/agent_bench/aa_index/SERVING_MODE.txt" \
    "$ROOT/results/agent_bench/aa_index/MTP_ENABLED.txt"
  rm -f "$ROOT/results/agent_bench/aa_index/MTP_REVERTED.txt"
else
  rm -f "$ROOT/results/agent_bench/aa_index/MTP_ENABLED.txt"
fi

echo "[$(date -Iseconds)] switch $MODE done"
