#!/usr/bin/env bash
# Overnight guardian on the Z8. Complements aa-watch.
# Never docker prune -af / image prune. Logs → NIGHT_GUARDIAN.log
set -uo pipefail

REPO="${REPO:-$HOME/Projects/Work/llm-bench}"
ROOT="$REPO/results/agent_bench/aa_index"
LOG="$ROOT/NIGHT_GUARDIAN.log"
INTERVAL="${GUARDIAN_INTERVAL:-120}"
MIN_FREE_HARD="${MIN_FREE_GB_HARD:-20}"

mkdir -p "$ROOT"
exec >>"$LOG" 2>&1

ts() { date -Iseconds; }
say() { echo "[$(ts)] $*"; }

free_gib() { df -BG --output=avail / | tail -1 | tr -dc '0-9'; }

ensure_krenew() {
  klist -s 2>/dev/null || { say "ALERT: no kerberos ticket"; return 0; }
  local left
  left=$(klist 2>/dev/null | awk '/krbtgt/{print $3,$4; exit}' | python3 -c '
import sys,datetime,time
s=sys.stdin.read().strip()
for fmt in ("%m/%d/%Y %H:%M:%S","%Y-%m-%d %H:%M:%S"):
  try:
    exp=datetime.datetime.strptime(s, fmt).timestamp(); print(int(exp-time.time())); break
  except Exception:
    pass
else:
  print(-1)
' 2>/dev/null || echo -1)
  if [[ "$left" -ge 0 && "$left" -lt 5400 ]]; then
    say "INTERVENE: kinit -R (left ${left}s)"
    kinit -R 2>&1 | while read -r l; do say "  $l"; done || true
  fi
  if ! pgrep -u "$USER" -x krenew >/dev/null 2>&1; then
    say "INTERVENE: start krenew -K 30"
    nohup krenew -K 30 -i >/dev/null 2>&1 &
    disown || true
  fi
}

restart_aa_ws() {
  say "INTERVENE: restart aa-ws"
  tmux has-session -t aa-ws 2>/dev/null && tmux kill-session -t aa-ws || true
  pkill -u "$USER" -f 'agent_bench.run_matrix' 2>/dev/null || true
  pkill -u "$USER" -f 'uv/tools/harbor/bin/python .*harbor ' 2>/dev/null || true
  sleep 3
  local launcher=agent_bench/scripts/run_aa_index_workstation_tp2_128k.sh
  [[ -x "$REPO/$launcher" ]] || launcher=agent_bench/scripts/run_aa_index_workstation.sh
  tmux new-session -d -s aa-ws -c "$REPO" -- bash -lc \
    "bash $launcher 2>&1 | tee -a $ROOT/overnight_ws_guardian_\$(date +%Y%m%d_%H%M%S).log"
  sleep 4
  tmux has-session -t aa-ws 2>/dev/null && say "aa-ws up" || say "FATAL aa-ws start failed"
}

cleanup_orphan_trial_containers() {
  # After aa-ws restarts, old trial containers can linger and steal resources.
  # Keep only containers whose trial key matches a dir without result.json.
  python3 - <<'PY' 2>/dev/null || return 0
import subprocess
from pathlib import Path
root = Path("/home/hiescha/Projects/Work/llm-bench/results/agent_bench/aa_index")
active = set()
for suite in root.iterdir():
    if not suite.is_dir() or suite.name.startswith("_"):
        continue
    for agent in suite.iterdir():
        if not agent.is_dir() or agent.name.startswith("_"):
            continue
        # newest job dir only
        jobs = [p for p in agent.iterdir() if p.is_dir() and not p.name.startswith("_")]
        if not jobs:
            continue
        job = max(jobs, key=lambda p: p.stat().st_mtime)
        for d in job.iterdir():
            if not d.is_dir() or d.name.startswith("_") or d.name == "artifacts":
                continue
            if (d / "result.json").is_file():
                continue
            active.add(d.name.lower())
if not active:
    raise SystemExit(0)
out = subprocess.check_output(["docker", "ps", "--format", "{{.ID}}\t{{.Names}}"], text=True)
for line in out.splitlines():
    if "local-registry" in line:
        continue
    cid, name = line.split("\t", 1)
    parts = name.split("__")
    if len(parts) < 2:
        continue
    key = f"{parts[0]}__{parts[1]}".lower()
    if "env-main" not in name:
        continue
    if key not in active:
        print(f"orphan {name}")
        subprocess.run(["docker", "stop", cid], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["docker", "rm", cid], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
PY
}

disk_emergency() {
  local free
  free=$(free_gib)
  [[ -n "$free" && "$free" -lt "$MIN_FREE_HARD" ]] || return 0
  say "INTERVENE: disk HARD free=${free}G — junk only (NO docker prune)"
  find "$ROOT" -maxdepth 3 -type d \( -name '_broken_*' -o -name '_failed_*' -o -name '_smoke*' -o -name '_stuck_*' -o -name '_partial_*' \) \
    2>/dev/null | while read -r d; do
      say "  rm -rf $d"; rm -rf "$d" 2>/dev/null || true
    done
  find "$ROOT" -maxdepth 2 -type f \( -name 'overnight_*.log' -o -name '*.resume_*.log' \) -mtime +1 -delete 2>/dev/null || true
  # exited trial containers (explicit rm)
  docker ps -aq --filter status=exited 2>/dev/null | while read -r id; do
    name=$(docker inspect -f '{{.Name}}' "$id" 2>/dev/null | sed 's#^/##')
    [[ "$name" == *registry* ]] && continue
    say "  rm exited $name"; docker rm "$id" >/dev/null 2>&1 || true
  done
  free=$(free_gib)
  if [[ "$free" -lt "$MIN_FREE_HARD" ]]; then
    for d in \
      "$HOME/finetune-output/orgchart-thinkingcap" \
      "$HOME/sglang-bench/.venv" \
      "$HOME/litellm-v191-test/.venv" \
      "$HOME/llama-build" \
      "$HOME/finetune-venv"
    do
      [[ -e "$d" ]] || continue
      say "  EMERGENCY rm -rf $d ($(du -sh "$d" 2>/dev/null | awk '{print $1}'))"
      rm -rf "$d" 2>/dev/null || true
      free=$(free_gib)
      [[ "$free" -ge "$MIN_FREE_HARD" ]] && break
    done
  fi
  say "  free now ${free}G"
}

say "night_guardian start interval=${INTERVAL}s (no docker prune)"
ensure_krenew
idle=0
loop_i=0

while true; do
  ensure_krenew
  disk_emergency
  loop_i=$((loop_i + 1))
  # every ~10 min (5 * 120s)
  if [[ $((loop_i % 5)) -eq 0 ]]; then
    orphans=$(cleanup_orphan_trial_containers | wc -l | tr -d ' ')
    [[ "${orphans:-0}" -gt 0 ]] && say "INTERVENE: removed ${orphans} orphan trial containers"
  fi

  local_krb=n; klist -s 2>/dev/null && local_krb=y
  local_aa=n; tmux has-session -t aa-ws 2>/dev/null && local_aa=y
  local_watch=n; tmux has-session -t aa-watch 2>/dev/null && local_watch=y
  local_matrix=n; pgrep -u "$USER" -f 'agent_bench.run_matrix' >/dev/null && local_matrix=y
  local_harbor=n; pgrep -u "$USER" -f 'harbor (run|job resume)' >/dev/null && local_harbor=y
  px=$(curl -sS -o /dev/null -w '%{http_code}' -x http://127.0.0.1:3128 --connect-timeout 12 -I https://github.com 2>/dev/null || echo 000)
  free=$(free_gib)
  gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr '\n' ',' || echo '?')
  swe=$(docker images 'ghcr.io/scaleapi/swe-atlas' -q 2>/dev/null | wc -l)
  newest=$(find "$ROOT" \( -name 'claude-code.txt' -o -name 'trial.log' -o -name result.json \) \
    ! -path '*/_failed*' ! -path '*/_broken*' ! -path '*/_smoke*' \
    -printf '%T@\n' 2>/dev/null | sort -nr | head -1)
  age=9999
  if [[ -n "${newest:-}" ]]; then
    age=$(python3 -c "import time; print(int((time.time()-float('$newest'))//60))")
  fi
  pane=$(tmux capture-pane -t aa-ws -p -S -6 2>/dev/null | tr '\n' '|' | tail -c 200)

  say "snap krb=$local_krb aa=$local_aa watch=$local_watch matrix=$local_matrix harbor=$local_harbor px=$px free=${free}G gpu=$gpu swe=$swe age_min=$age | $pane"

  if [[ "$px" != "200" ]]; then
    say "INTERVENE: restart px-proxy (px=$px)"
    systemctl --user reset-failed px-proxy 2>/dev/null || true
    systemctl --user restart px-proxy 2>/dev/null || true
  fi

  if [[ "$local_watch" != "y" ]]; then
    say "INTERVENE: restart aa-watch"
    tmux new-session -d -s aa-watch -c "$REPO" -- bash -lc \
      "bash agent_bench/scripts/watch_aa_index_workstation.sh"
  fi

  if [[ "$local_aa" != "y" || "$local_matrix" != "y" ]]; then
    # allow short bootstrap window
    if pgrep -u "$USER" -f 'run_aa_index_workstation' >/dev/null 2>&1 && [[ "$local_aa" == "y" ]]; then
      say "aa-ws bootstrapping — wait"
    else
      restart_aa_ws
      idle=0
    fi
    sleep "$INTERVAL"
    continue
  fi

  if [[ "${swe:-0}" -lt 11 ]] && ! tmux has-session -t swe-pull 2>/dev/null; then
    say "WARN: swe images=$swe — kick swe-pull"
    list=/tmp/swe-atlas-images.txt
    grep -rh '^docker_image' "$REPO/results/agent_bench/datasets/SWE-Atlas/data/qa" --include='task.toml' \
      | sed -E 's/.*= "([^"]+)".*/\1/' | sort -u >"$list" || true
    tmux new-session -d -s swe-pull -- bash -lc "
      export HTTP_PROXY=http://127.0.0.1:3128 HTTPS_PROXY=http://127.0.0.1:3128
      while read img; do
        [[ -z \"\$img\" ]] && continue
        docker image inspect \"\$img\" >/dev/null 2>&1 && continue
        echo PULL \$img; docker pull \"\$img\" || echo FAIL \$img
      done < $list; echo DONE
    "
  fi

  gpu_busy=0
  echo "$gpu" | tr ',' '\n' | while read -r g; do
    [[ "${g:-0}" -gt 5 ]] && exit 1
    true
  done && gpu_busy=0 || gpu_busy=1

  if [[ "$local_harbor" == "y" || "$gpu_busy" -eq 1 ]]; then
    idle=0
  elif [[ "$age" -ge 15 ]]; then
    idle=$((idle + 1))
    say "stale age_min=$age idle=$idle"
    if [[ "$idle" -ge 4 ]]; then
      restart_aa_ws
      idle=0
    fi
  else
    idle=0
  fi

  sleep "$INTERVAL"
done
