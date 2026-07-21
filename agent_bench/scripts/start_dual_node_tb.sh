#!/usr/bin/env bash
# Master: stop old runs, split LiteLLM, bootstrap x39, start dual runners + monitor.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
X40="${X40_HOST:-hiescha@cmtcdeu89976740.rd.corpintra.net}"
X39="${X39_HOST:-hiescha@cmtcdeu89976739.rd.corpintra.net}"
PORT="${SSH_PORT:-42022}"
export KUBECONFIG="${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}"

STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$ROOT/results/agent_bench/aa_index/dual_start_${STAMP}.log"
mkdir -p "$ROOT/results/agent_bench/aa_index"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "=== 1) stop benchmarks on x40 ==="
bash "$ROOT/agent_bench/scripts/stop_aa_benchmarks.sh" 2>&1 | tee -a "$LOG"

log "=== 2) apply split LiteLLM routing ==="
bash "$ROOT/agent_bench/k8s/apply-litellm-split-local.sh" 2>&1 | tee -a "$LOG"

log "=== 3) sync repo → x39 ==="
ssh -p "$PORT" "$X39" "mkdir -p ~/Projects/Work/llm-bench/results/agent_bench/aa_index"
rsync -az \
  --exclude '.git' --exclude '.venv' --exclude 'results/agent_bench/aa_index/terminal-bench-v2' \
  -e "ssh -p $PORT" \
  "$ROOT/" "$X39:~/Projects/Work/llm-bench/" 2>&1 | tee -a "$LOG"

log "=== 4) sync TB datasets → x39 (incremental) ==="
rsync -az \
  -e "ssh -p $PORT" \
  "$ROOT/results/agent_bench/datasets/" \
  "$X39:~/Projects/Work/llm-bench/results/agent_bench/datasets/" 2>&1 | tee -a "$LOG"

log "=== 5) x39 docker + bootstrap ==="
ssh -p "$PORT" "$X39" "bash -s" <<'REMOTE' 2>&1 | tee -a "$LOG"
set -euo pipefail
sudo chgrp docker /var/run/docker.sock 2>/dev/null || true
sudo chmod 660 /var/run/docker.sock 2>/dev/null || true
sudo systemctl restart docker.socket docker.service
docker ps >/dev/null
echo "docker OK"

REPO=~/Projects/Work/llm-bench
cd "$REPO"
export UV_TOOL_DIR="$HOME/aa-index-runner-home/.local/share/uv/tools"
export UV_TOOL_BIN_DIR="$HOME/aa-index-runner-home/.local/bin"
export PATH="$UV_TOOL_BIN_DIR:$PATH"
mkdir -p "$HOME/aa-index-runner-home"
if ! command -v harbor >/dev/null; then
  uv tool install harbor --python 3.12
fi
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
  .venv/bin/pip install -q pyyaml
fi
# corp CA for harbor docker
mkdir -p agent_bench/certs
test -f agent_bench/certs/docker-ca-bundle.pem || cp -f ../llm-bench/agent_bench/certs/docker-ca-bundle.pem agent_bench/certs/ 2>/dev/null || true
chmod +x agent_bench/scripts/*.sh agent_bench/k8s/*.sh 2>/dev/null || true
echo "x39 bootstrap OK"
REMOTE

log "=== 6) stop x39 if anything running ==="
ssh -p "$PORT" "$X39" "bash ~/Projects/Work/llm-bench/agent_bench/scripts/stop_aa_benchmarks.sh" 2>&1 | tee -a "$LOG" || true

log "=== 7) smoke LiteLLM local on both nodes ==="
KEY=$(tr -d '[:space:]' < "$HOME/llm-serving/aa-index-key")
for host in x40 x39; do
  if [[ "$host" == "x40" ]]; then
    H="$X40"
  else
    H="$X39"
  fi
  code=$(ssh -p "$PORT" "$H" "curl -s -o /dev/null -w '%{http_code}' -m 15 -H 'Authorization: Bearer $KEY' http://127.0.0.1:4000/v1/models" || echo 000)
  log "$host litellm /v1/models HTTP $code"
  [[ "$code" == "200" ]] || { log "FATAL: $host LiteLLM smoke failed"; exit 1; }
done

log "=== 8) start dual runners in tmux ==="
tmux kill-session -t dual-x40 2>/dev/null || true
tmux kill-session -t dual-monitor 2>/dev/null || true
tmux new-session -d -s dual-x40 -c "$ROOT" -- \
  bash agent_bench/scripts/dual_node_tb_loop.sh x40
tmux new-session -d -s dual-monitor -c "$ROOT" -- \
  bash agent_bench/scripts/dual_node_monitor.sh

ssh -p "$PORT" "$X39" "tmux kill-session -t dual-x39 2>/dev/null || true; tmux new-session -d -s dual-x39 -c ~/Projects/Work/llm-bench -- bash agent_bench/scripts/dual_node_tb_loop.sh x39"

log "=== started. monitor: tmux attach -t dual-monitor ==="
bash "$ROOT/agent_bench/scripts/dual_node_merge_tb.py" --print | tee -a "$LOG"
