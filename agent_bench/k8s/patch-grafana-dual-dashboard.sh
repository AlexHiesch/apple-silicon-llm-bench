#!/usr/bin/env bash
# Merge dual-bench dashboard into grafana-dashboards-hpllm ConfigMap.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
OBS=llm-observability
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

kubectl -n "$OBS" get cm grafana-dashboards-hpllm -o json >"$TMP/cm.json"
python3 - <<'PY' "$TMP/cm.json" "$ROOT/grafana-dual-bench-dashboard.json" "$TMP/out.json"
import json, sys
cm_path, dash_path, out_path = sys.argv[1:4]
cm = json.load(open(cm_path))
dash = json.load(open(dash_path))
cm["data"]["dual-bench-dashboard.json"] = json.dumps(dash, indent=2)
# strip runtime fields for apply
for k in ("resourceVersion", "uid", "creationTimestamp", "managedFields"):
    cm.get("metadata", {}).pop(k, None)
json.dump(cm, open(out_path, "w"), indent=2)
print("patched dual-bench-dashboard.json into configmap")
PY
kubectl -n "$OBS" apply -f "$TMP/out.json"
kubectl -n "$OBS" rollout restart deploy/grafana
kubectl -n "$OBS" rollout status deploy/grafana --timeout=120s
echo "Grafana dashboard: /d/hpllm-dual-bench/hpllm-dual-node-bench-x40-x39"
