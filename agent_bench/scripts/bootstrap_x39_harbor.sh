#!/usr/bin/env bash
# One-shot x39 Harbor bootstrap: px-proxy ACL, docker compose plugin, datasets check.
set -euo pipefail
X40_HOST="${X40_HOST:-hiescha@cmtcdeu89976740.rd.corpintra.net}"
PORT="${SSH_PORT:-42022}"
REPO="${REPO:-$HOME/Projects/Work/llm-bench}"

# Harbor compose networks use 172.17–172.31; stock px-proxy only allows 172.17/18.
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
echo "px-proxy restarted (harbor-docker ACL)"

mkdir -p ~/.docker/cli-plugins
if ! docker compose version >/dev/null 2>&1; then
  echo "install docker compose plugin from x40..."
  scp -P "$PORT" "$X40_HOST:~/.docker/cli-plugins/docker-compose" \
    ~/.docker/cli-plugins/docker-compose
  chmod +x ~/.docker/cli-plugins/docker-compose
fi
docker compose version

TB="$REPO/results/agent_bench/datasets/terminal-bench-2.0/terminal-bench/task.toml"
if [[ ! -f "${TB/task.toml/query-optimize/task.toml}" ]]; then
  echo "sync datasets from x40..."
  mkdir -p "$REPO/results/agent_bench/datasets"
  rsync -az -e "ssh -p $PORT" \
    "$X40_HOST:$REPO/results/agent_bench/datasets/" \
    "$REPO/results/agent_bench/datasets/"
fi

test -f "$REPO/results/agent_bench/datasets/terminal-bench-2.0/terminal-bench/query-optimize/task.toml"
echo "x39 harbor bootstrap OK"
