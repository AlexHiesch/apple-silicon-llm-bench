#!/usr/bin/env bash
# One-shot x39 Harbor bootstrap: docker compose plugin + datasets check.
set -euo pipefail
X40_HOST="${X40_HOST:-hiescha@cmtcdeu89976740.rd.corpintra.net}"
PORT="${SSH_PORT:-42022}"
REPO="${REPO:-$HOME/Projects/Work/llm-bench}"

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
