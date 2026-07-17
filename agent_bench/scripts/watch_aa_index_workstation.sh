#!/usr/bin/env bash
# Tight watchdog for AA Index overnight on the Z8.
# Detects stuck/infra failure modes and intervenes without sudo/kinit.
set -uo pipefail

REPO="${REPO:-$HOME/Projects/Work/llm-bench}"
ROOT="$REPO/results/agent_bench/aa_index"
LOG="$ROOT/WATCHDOG.log"
STATUS="$ROOT/OVERNIGHT_STATUS.txt"
INTERVAL="${WATCH_INTERVAL:-90}"
STALE_RESULT_MIN="${STALE_RESULT_MIN:-12}"   # no new result.json → suspect
STALE_GPU_IDLE_MIN="${STALE_GPU_IDLE_MIN:-8}"
PX_FAIL_STREAK_RESTART=2

mkdir -p "$ROOT"
exec >>"$LOG" 2>&1

ts() { date -Iseconds; }
say() { echo "[$(ts)] $*"; }

px_ok() {
  local code
  code=$(curl -sS -o /dev/null -w '%{http_code}' -x http://127.0.0.1:3128 \
    --connect-timeout 12 -I https://registry.npmjs.org/ 2>/dev/null || echo 000)
  [[ "$code" == "200" ]]
}

ghcr_connect_ok() {
  # 401/405 = CONNECT worked; 000/timeout = broken
  local code
  code=$(curl -sS -o /dev/null -w '%{http_code}' -x http://127.0.0.1:3128 \
    --connect-timeout 15 -I https://ghcr.io/v2/ 2>/dev/null || echo 000)
  [[ "$code" != "000" && "$code" != "" ]]
}

restart_px() {
  say "INTERVENE: restart px-proxy"
  systemctl --user reset-failed px-proxy 2>/dev/null || true
  systemctl --user restart px-proxy
  sleep 3
  systemctl --user is-active px-proxy >/dev/null && say "px active" || say "WARN px not active"
}

aa_alive() {
  tmux has-session -t aa-ws 2>/dev/null
}

harbor_alive() {
  pgrep -u "$USER" -f 'harbor (run|job resume)' >/dev/null 2>&1
}

matrix_alive() {
  pgrep -u "$USER" -f 'agent_bench.run_matrix' >/dev/null 2>&1
}

newest_result_age_min() {
  local f
  f=$(find "$ROOT" \( -name result.json -o -name trial.log -o -name 'claude-code.txt' \) \
    ! -path '*/_failed*' ! -path '*/_broken*' ! -path '*/_smoke*' \
    -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | awk '{print $1}')
  if [[ -z "${f:-}" ]]; then echo 9999; return; fi
  local now age
  now=$(date +%s)
  age=$(python3 -c "print(int(($now - float('$f')) // 60))")
  echo "$age"
}

gpu_busy() {
  # any GPU util > 5%
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
    | awk '{if ($1+0 > 5) found=1} END{exit !found}'
}

count_zombies() {
  # harbor env containers older than 3h (exclude local-registry)
  docker ps --format '{{.Names}}\t{{.Status}}' 2>/dev/null \
    | grep -v local-registry \
    | grep -cE 'Up [3-9] hours|Up [0-9]+ days' || true
}

prune_zombies() {
  local n
  n=$(count_zombies)
  if [[ "${n:-0}" -lt 1 ]]; then return 0; fi
  say "INTERVENE: stop ${n} stale containers (>3h)"
  docker ps --format '{{.ID}} {{.Names}} {{.Status}}' \
    | grep -v local-registry \
    | grep -E 'Up [3-9] hours|Up [0-9]+ days' \
    | while read -r id name status; do
        say "  stop $name ($status)"
        docker stop "$id" >/dev/null 2>&1 || true
      done
  docker container prune -f >/dev/null 2>&1 || true
}

recent_ghcr_fail_storm() {
  # ≥5 compose/ghcr exceptions in last 10 minutes under aa_index
  local n
  n=$(find "$ROOT" -name exception.txt -mmin -10 ! -path '*/_failed*' 2>/dev/null \
    | xargs grep -l 'ghcr.io\|context deadline exceeded\|main Pulling' 2>/dev/null \
    | wc -l)
  echo "${n:-0}"
}

restart_aa_ws() {
  say "INTERVENE: restart tmux aa-ws"
  tmux has-session -t aa-ws 2>/dev/null && tmux kill-session -t aa-ws || true
  # careful kill — match harbor/matrix only
  ps -u "$USER" -o pid=,args= \
    | awk '/agent_bench[.]run_matrix|uv\/tools\/harbor\/bin\/python .*harbor /{print $1}' \
    | while read -r p; do kill "$p" 2>/dev/null || true; done
  sleep 3
  tmux new-session -d -s aa-ws -c "$REPO" -- bash -lc \
    "bash agent_bench/scripts/run_aa_index_workstation.sh 2>&1 | tee -a $ROOT/overnight_ws_restart_\$(date +%Y%m%d_%H%M%S).log"
  sleep 4
  tmux has-session -t aa-ws 2>/dev/null && say "aa-ws restarted" || say "FATAL aa-ws failed to start"
}

ensure_swe_pull() {
  # keep prefetch going if images remain
  local list=/tmp/swe-atlas-images.txt
  if [[ ! -f "$list" ]]; then
    grep -rh '^docker_image' "$REPO/results/agent_bench/datasets/SWE-Atlas/data/qa" --include='task.toml' \
      | sed -E 's/.*= "([^"]+)".*/\1/' | sort -u >"$list" || true
  fi
  local missing=0
  while read -r img; do
    [[ -z "$img" ]] && continue
    docker image inspect "$img" >/dev/null 2>&1 || missing=$((missing + 1))
  done <"$list"
  if [[ "$missing" -eq 0 ]]; then
    return 0
  fi
  if tmux has-session -t swe-pull 2>/dev/null; then
    return 0
  fi
  say "INTERVENE: restart swe-pull ($missing images missing)"
  tmux new-session -d -s swe-pull -- bash -lc "
    ok=0; fail=0; skip=0
    while read img; do
      [[ -z \"\$img\" ]] && continue
      if docker image inspect \"\$img\" >/dev/null 2>&1; then skip=\$((skip+1)); continue; fi
      echo PULL \$img
      if docker pull \"\$img\"; then ok=\$((ok+1)); else fail=\$((fail+1)); echo FAIL \$img; fi
    done < $list
    echo DONE ok=\$ok fail=\$fail skip=\$skip
  "
}

snapshot() {
  local age tech_line gpu docker_n
  age=$(newest_result_age_min)
  tech_line=$("$REPO/.venv/bin/python" -m agent_bench.tech_failures --root "$ROOT" 2>/dev/null | head -1 || echo "?")
  gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr '\n' ',' || echo "?")
  docker_n=$(docker ps -q 2>/dev/null | wc -l)
  say "snap age_min=$age gpu%=$gpu docker=$docker_n aa=$(aa_alive && echo y || echo n) harbor=$(harbor_alive && echo y || echo n) matrix=$(matrix_alive && echo y || echo n) | $tech_line"
}

px_fail=0
idle_rounds=0

say "watchdog start interval=${INTERVAL}s repo=$REPO"

while true; do
  snapshot

  # 1) proxy health
  if px_ok && ghcr_connect_ok; then
    px_fail=0
  else
    px_fail=$((px_fail + 1))
    say "px unhealthy streak=$px_fail"
    if [[ "$px_fail" -ge "$PX_FAIL_STREAK_RESTART" ]]; then
      restart_px
      px_fail=0
      sleep 5
    fi
  fi

  # 2) ghcr fail storm → bounce px (often fixes CONNECT)
  storm=$(recent_ghcr_fail_storm)
  if [[ "${storm:-0}" -ge 5 ]]; then
    say "ghcr fail storm last10m=$storm"
    restart_px
    # archive nothing automatically — resume-harbor retries RuntimeError
  fi

  # 3) zombies
  prune_zombies

  # 4) swe image prefetch
  ensure_swe_pull

  # 5) aa-ws / harbor liveness
  if ! aa_alive || ! matrix_alive; then
    say "aa-ws or matrix dead"
    restart_aa_ws
    idle_rounds=0
    sleep "$INTERVAL"
    continue
  fi

  # 6) progress watchdog: no new artifacts AND no harbor AND gpu idle
  age=$(newest_result_age_min)
  if harbor_alive || gpu_busy; then
    idle_rounds=0
  else
    # harbor may be between trials briefly
    if [[ "$age" -ge "$STALE_RESULT_MIN" ]]; then
      idle_rounds=$((idle_rounds + 1))
      say "no harbor+gpu, stale results ${age}m (idle_rounds=$idle_rounds)"
      if [[ "$idle_rounds" -ge 3 ]]; then
        say "stuck suspected — restart aa-ws"
        restart_aa_ws
        idle_rounds=0
      fi
    else
      idle_rounds=0
    fi
  fi

  # 7) harbor running but results stale AND gpu idle for long — could be hung agent install
  if harbor_alive && [[ "$age" -ge 25 ]] && ! gpu_busy; then
    say "WARN harbor alive but no results ${age}m and GPU idle — leave unless storm"
  fi

  sleep "$INTERVAL"
done
