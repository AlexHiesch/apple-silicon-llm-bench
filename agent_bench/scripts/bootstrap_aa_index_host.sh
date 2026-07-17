#!/usr/bin/env bash
# One-shot host bootstrap for AA Index on the Z8 (no sudo, no kinit refresh).
# Run while your session/ticket is still valid. Overnight path must not apt/sudo.
set -euo pipefail

export HTTP_PROXY="${HTTP_PROXY:-http://localhost:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://localhost:3128}"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,.svc,.cluster.local,.corpintra.net,cmtcdeu89976740.rd.corpintra.net}"
export no_proxy="$NO_PROXY"

REPO="${REPO:-$HOME/Projects/Work/llm-bench}"
RUNNER_HOME="${RUNNER_HOME:-$HOME/aa-index-runner-home}"
mkdir -p "$RUNNER_HOME/.local/bin" "$REPO"

export PATH="$RUNNER_HOME/.local/bin:$HOME/.local/bin:$PATH"
export UV_TOOL_DIR="${UV_TOOL_DIR:-$RUNNER_HOME/.local/share/uv/tools}"
export UV_TOOL_BIN_DIR="${UV_TOOL_BIN_DIR:-$RUNNER_HOME/.local/bin}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$RUNNER_HOME/.local/share/uv/python}"

if [[ ! -x "$RUNNER_HOME/.local/bin/uv" && ! -x "$HOME/.local/bin/uv" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$RUNNER_HOME/.local" sh
fi
UV="$(command -v uv)"

# Install tools into runner-home so the k8s pod hostPath sees them.
UV_TOOL_DIR="$UV_TOOL_DIR" UV_TOOL_BIN_DIR="$UV_TOOL_BIN_DIR" \
  "$UV" tool install --python 3.12 harbor
UV_TOOL_DIR="$UV_TOOL_DIR" UV_TOOL_BIN_DIR="$UV_TOOL_BIN_DIR" \
  "$UV" tool install --python 3.12 datacurve-pier || true

cd "$REPO"
if [[ ! -d .venv ]]; then
  "$UV" venv .venv
fi
"$UV" pip install -p .venv -r requirements.txt pyyaml

# Harbor compose networks land on 172.19+/16. Stock px-proxy (salt) only allows
# 172.17/18 — agent apt/curl then gets "Connection failed [IP: 172.17.0.1 3128]".
# User drop-in (no sudo): allow all RFC1918 docker bridges.
mkdir -p "$HOME/.config/systemd/user/px-proxy.service.d"
cat > "$HOME/.config/systemd/user/px-proxy.service.d/harbor-docker.conf" <<'EOF'
[Unit]
StartLimitIntervalSec=0

[Service]
EnvironmentFile=
ExecStart=
ExecStart=/usr/bin/px-proxy --hostonly --threads 40 --idle 300 --socktimeout 600 --gateway --allow=127.0.0.0/8,172.16.0.0/12,192.168.0.0/16 --log --pac=http://browsercfg.edc.corpintra.net:8899/linux/proxy.pac --noproxy 127.0.0.0/8,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,100.64.0.0/10
EOF
systemctl --user daemon-reload
systemctl --user reset-failed px-proxy 2>/dev/null || true
systemctl --user restart px-proxy
systemctl --user is-active px-proxy >/dev/null
# After a fresh `kinit`, restart px again — stale upstream CONNECT tunnels
# otherwise time out (ghcr.io / npm / github all fail until restart).

# Warm Docker pulls that Harbor/smoke will need (daemon already has corp proxy).
docker pull alpine:3.20 >/dev/null
docker pull curlimages/curl:8.5.0 >/dev/null || true

# Sanity: gateway + docker→host LiteLLM
KEY_FILE="${HPLLM_KEY_FILE:-$HOME/litellm-portal/state/mvp-test-key}"
KEY="$(tr -d '[:space:]' < "$KEY_FILE")"
curl -sf -m 15 -H "Authorization: Bearer $KEY" http://127.0.0.1:4000/v1/models >/dev/null
docker run --rm --add-host=host.docker.internal:host-gateway \
  curlimages/curl:8.5.0 -sf -m 30 -o /dev/null \
  -H "Authorization: Bearer $KEY" \
  http://host.docker.internal:4000/v1/models

echo "bootstrap OK"
echo "  harbor: $(command -v harbor)"
echo "  pier:   $(command -v pier || echo missing)"
echo "  venv:   $REPO/.venv"
echo "  runner: $RUNNER_HOME"
