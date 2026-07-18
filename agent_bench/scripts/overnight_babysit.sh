#!/usr/bin/env bash
# One-shot bootstrap for overnight AA Index babysitting on the Z8.
# Starts krenew + aa-watch (+ aa-ws if missing). Safe to re-run.
set -uo pipefail

REPO="${REPO:-$HOME/Projects/Work/llm-bench}"
ROOT="$REPO/results/agent_bench/aa_index"
mkdir -p "$ROOT"

echo "[$(date -Iseconds)] overnight_babysit bootstrap"

# Kerberos keep-alive (ticket must already be valid from interactive kinit)
if command -v krenew >/dev/null 2>&1 && klist -s 2>/dev/null; then
  kinit -R 2>/dev/null || true
  if ! pgrep -u "$USER" -x krenew >/dev/null 2>&1; then
    echo "  start krenew -K 30"
    nohup krenew -K 30 -i >/dev/null 2>&1 &
    disown || true
  else
    echo "  krenew already running"
  fi
  klist | head -5
else
  echo "  WARN: no ticket / no krenew — corp egress may fail overnight"
fi

# Ensure SWE-Atlas images list + prefetch session if needed
list=/tmp/swe-atlas-images.txt
grep -rh '^docker_image' "$REPO/results/agent_bench/datasets/SWE-Atlas/data/qa" --include='task.toml' \
  | sed -E 's/.*= "([^"]+)".*/\1/' | sort -u >"$list" || true
missing=0
while read -r img; do
  [[ -z "$img" ]] && continue
  docker image inspect "$img" >/dev/null 2>&1 || missing=$((missing + 1))
done <"$list"
echo "  swe-atlas images missing=$missing / $(wc -l <"$list")"
if [[ "$missing" -gt 0 ]] && ! tmux has-session -t swe-pull 2>/dev/null; then
  echo "  start swe-pull"
  tmux new-session -d -s swe-pull -- bash -lc "
    export HTTP_PROXY=http://127.0.0.1:3128 HTTPS_PROXY=http://127.0.0.1:3128
    while read img; do
      [[ -z \"\$img\" ]] && continue
      docker image inspect \"\$img\" >/dev/null 2>&1 && continue
      echo PULL \$img; docker pull \"\$img\" || echo FAIL \$img
    done < $list
    echo DONE
  "
fi

# aa-watch
if tmux has-session -t aa-watch 2>/dev/null; then
  tmux kill-session -t aa-watch || true
  sleep 1
fi
echo "  start aa-watch"
tmux new-session -d -s aa-watch -c "$REPO" -- bash -lc \
  "bash agent_bench/scripts/watch_aa_index_workstation.sh"

# aa-ws if missing
if ! tmux has-session -t aa-ws 2>/dev/null; then
  launcher=agent_bench/scripts/run_aa_index_workstation_tp2_128k.sh
  [[ -x "$REPO/$launcher" ]] || launcher=agent_bench/scripts/run_aa_index_workstation.sh
  echo "  start aa-ws via $launcher"
  tmux new-session -d -s aa-ws -c "$REPO" -- bash -lc \
    "bash $launcher 2>&1 | tee -a $ROOT/overnight_ws_\$(date +%Y%m%d_%H%M%S).log"
else
  echo "  aa-ws already running — leave it (no interrupt)"
fi

echo "  sessions:"
tmux ls 2>/dev/null || true
echo "[$(date -Iseconds)] babysit ready"
