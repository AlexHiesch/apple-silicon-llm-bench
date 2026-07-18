#!/usr/bin/env bash
# Tight watchdog for AA Index overnight on the Z8.
# Keeps Kerberos + px-proxy + SWE images + aa-ws alive.
# NEVER docker image prune -a / system prune -af (wipes swe-atlas images).
set -uo pipefail

REPO="${REPO:-$HOME/Projects/Work/llm-bench}"
ROOT="$REPO/results/agent_bench/aa_index"
LOG="$ROOT/WATCHDOG.log"
INTERVAL="${WATCH_INTERVAL:-90}"
STALE_RESULT_MIN="${STALE_RESULT_MIN:-12}"
PX_FAIL_STREAK_RESTART=2
MIN_FREE_GB_SOFT="${MIN_FREE_GB_SOFT:-40}"
MIN_FREE_GB_HARD="${MIN_FREE_GB_HARD:-15}"

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

ensure_krenew() {
  # Ticket renew-until ~1 week after fresh kinit; keep TGT alive overnight.
  if ! command -v krenew >/dev/null 2>&1; then
    return 0
  fi
  if ! klist -s 2>/dev/null; then
    say "WARN: no kerberos ticket (klist -s failed) — corp proxy may die"
    return 0
  fi
  # Renew if < 90 minutes left
  local exp_epoch now left
  exp_epoch=$(klist 2>/dev/null | awk '/krbtgt/{print $3,$4; exit}' | python3 -c '
import sys,datetime
s=sys.stdin.read().strip()
for fmt in ("%m/%d/%Y %H:%M:%S","%Y-%m-%d %H:%M:%S"):
  try:
    print(int(datetime.datetime.strptime(s, fmt).timestamp())); break
  except Exception:
    pass
' 2>/dev/null || echo 0)
  now=$(date +%s)
  left=$((exp_epoch - now))
  if [[ "$exp_epoch" -gt 0 && "$left" -lt 5400 ]]; then
    say "INTERVENE: kinit -R (ticket left ${left}s)"
    kinit -R 2>&1 | while read -r line; do say "  kinit: $line"; done || true
  fi
  if ! pgrep -u "$USER" -x krenew >/dev/null 2>&1; then
    say "INTERVENE: start krenew -K 30"
    # -K 30: check every 30 min; renew before expiry
    nohup krenew -K 30 -i >/dev/null 2>&1 &
    disown || true
  fi
}

aa_alive() { tmux has-session -t aa-ws 2>/dev/null; }
harbor_alive() { pgrep -u "$USER" -f 'harbor (run|job resume)' >/dev/null 2>&1; }
matrix_alive() { pgrep -u "$USER" -f 'agent_bench.run_matrix' >/dev/null 2>&1; }

newest_result_age_min() {
  local f
  f=$(find "$ROOT" \( -name result.json -o -name trial.log -o -name 'claude-code.txt' \) \
    ! -path '*/_failed*' ! -path '*/_broken*' ! -path '*/_smoke*' \
    -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | awk '{print $1}')
  if [[ -z "${f:-}" ]]; then echo 9999; return; fi
  local now
  now=$(date +%s)
  python3 -c "print(int(($now - float('$f')) // 60))"
}

gpu_busy() {
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
    | awk '{if ($1+0 > 5) found=1} END{exit !found}'
}

free_gib() {
  df -BG --output=avail / | tail -1 | tr -dc '0-9'
}

count_zombies() {
  docker ps --format '{{.Names}}\t{{.Status}}' 2>/dev/null \
    | grep -v local-registry \
    | grep -cE 'Up [3-9] hours|Up [0-9]+ days' || true
}

prune_zombies() {
  # Stop stale Harbor trial containers only — NEVER docker prune / image delete.
  local n
  n=$(count_zombies)
  if [[ "${n:-0}" -lt 1 ]]; then return 0; fi
  say "INTERVENE: stop ${n} stale containers (>3h) — no prune"
  docker ps --format '{{.ID}} {{.Names}} {{.Status}}' \
    | grep -v local-registry \
    | grep -E 'Up [3-9] hours|Up [0-9]+ days' \
    | while read -r id name status; do
        say "  stop $name ($status)"
        docker stop "$id" >/dev/null 2>&1 || true
        docker rm "$id" >/dev/null 2>&1 || true
      done
}

disk_hygiene() {
  # Free space WITHOUT any docker prune (keeps SWE-Atlas + TB images).
  # Soft: old logs / smoke / failed trees. Hard: known large unused dirs.
  local free
  free=$(free_gib)
  [[ -z "$free" ]] && return 0
  if [[ "$free" -ge "$MIN_FREE_GB_SOFT" ]]; then
    return 0
  fi
  say "INTERVENE: disk soft pressure free=${free}GiB — clean logs/junk (NO docker prune)"
  find "$ROOT" -maxdepth 2 -type f \( -name 'overnight_*.log' -o -name '*.resume_*.log' -o -name 'aa_watch_*.log' -o -name 'NIGHT_*.log' \) \
    -mtime +2 -delete 2>/dev/null || true
  find "$REPO/results/agent_bench" -maxdepth 1 -type f -name 'plan_*.json' -mtime +3 -delete 2>/dev/null || true
  # Stopped trial containers (by name pattern) — explicit rm, not prune
  docker ps -aq --filter status=exited 2>/dev/null | while read -r id; do
    local name
    name=$(docker inspect -f '{{.Name}}' "$id" 2>/dev/null | sed 's#^/##')
    case "$name" in
      local-registry|*"registry"*) continue ;;
    esac
    say "  rm exited container $name"
    docker rm "$id" >/dev/null 2>&1 || true
  done
  # failed/broken/smoke/stuck trees
  find "$ROOT" -maxdepth 3 -type d \( -name '_broken_*' -o -name '_failed_*' -o -name '_partial_*' -o -name '_smoke*' -o -name '_stuck_*' \) \
    2>/dev/null | head -40 | while read -r d; do
      say "  rm -rf $d"
      rm -rf "$d" 2>/dev/null || true
    done
  free=$(free_gib)
  say "  free after soft clean: ${free}GiB"
  if [[ "$free" -lt "$MIN_FREE_GB_HARD" ]]; then
    say "WARN: HARD disk pressure ${free}GiB — remove large unused non-bench dirs"
    # Safe-ish emergency candidates (not model weights in active serving path)
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
      [[ "$free" -ge "$MIN_FREE_GB_HARD" ]] && break
    done
    say "  free after hard clean: ${free}GiB"
  fi
}

recent_ghcr_fail_storm() {
  local n
  n=$(find "$ROOT" -name exception.txt -mmin -10 ! -path '*/_failed*' 2>/dev/null \
    | xargs grep -l 'ghcr.io\|context deadline exceeded\|main Pulling\|Service Unavailable' 2>/dev/null \
    | wc -l)
  echo "${n:-0}"
}

recent_lock_mismatch_storm() {
  local n
  n=$(find "$ROOT" -name '*.resume_*.log' -mmin -15 2>/dev/null \
    | xargs grep -l 'Existing trial config does not match\|does not match the resolved job lock' 2>/dev/null \
    | wc -l)
  echo "${n:-0}"
}

restart_aa_ws() {
  say "INTERVENE: restart tmux aa-ws"
  tmux has-session -t aa-ws 2>/dev/null && tmux kill-session -t aa-ws || true
  ps -u "$USER" -o pid=,args= \
    | awk '/agent_bench[.]run_matrix|uv\/tools\/harbor\/bin\/python .*harbor /{print $1}' \
    | while read -r p; do kill "$p" 2>/dev/null || true; done
  sleep 3
  local launcher="agent_bench/scripts/run_aa_index_workstation.sh"
  if [[ -f "$ROOT/BENCH_TP2_128K.env" || -f "$ROOT/TEMP_MODE_ACTIVE.txt" ]]; then
    if [[ -x "$REPO/agent_bench/scripts/run_aa_index_workstation_tp2_128k.sh" ]]; then
      launcher="agent_bench/scripts/run_aa_index_workstation_tp2_128k.sh"
      say "using TEMP TP2@128k launcher"
    fi
  fi
  # Always prefer TP2 launcher if present (overnight mode).
  if [[ -x "$REPO/agent_bench/scripts/run_aa_index_workstation_tp2_128k.sh" ]]; then
    launcher="agent_bench/scripts/run_aa_index_workstation_tp2_128k.sh"
  fi
  tmux new-session -d -s aa-ws -c "$REPO" -- bash -lc \
    "bash $launcher 2>&1 | tee -a $ROOT/overnight_ws_restart_\$(date +%Y%m%d_%H%M%S).log"
  sleep 4
  tmux has-session -t aa-ws 2>/dev/null && say "aa-ws restarted" || say "FATAL aa-ws failed to start"
}

ensure_swe_pull() {
  local list=/tmp/swe-atlas-images.txt
  if [[ ! -f "$list" ]]; then
    grep -rh '^docker_image' "$REPO/results/agent_bench/datasets/SWE-Atlas/data/qa" --include='task.toml' \
      | sed -E 's/.*= "([^"]+)".*/\1/' | sort -u >"$list" || true
  fi
  local missing=0 present=0
  while read -r img; do
    [[ -z "$img" ]] && continue
    if docker image inspect "$img" >/dev/null 2>&1; then
      present=$((present + 1))
    else
      missing=$((missing + 1))
    fi
  done <"$list"
  if [[ "$missing" -eq 0 ]]; then
    return 0
  fi
  if tmux has-session -t swe-pull 2>/dev/null; then
    return 0
  fi
  say "INTERVENE: start swe-pull (present=$present missing=$missing)"
  tmux new-session -d -s swe-pull -- bash -lc "
    export HTTP_PROXY=http://127.0.0.1:3128 HTTPS_PROXY=http://127.0.0.1:3128
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

# Only archive a job if the *newest* resume log shows config mismatch AND that
# job path is named in the log (never guess the first TB job on disk).
unstick_lock_mismatch() {
  local newest job dest
  newest=$(find "$ROOT" -name '*.resume_*.log' -mmin -10 2>/dev/null | xargs ls -t 2>/dev/null | head -1)
  [[ -z "${newest:-}" ]] && return 0
  if ! grep -q 'Existing trial config does not match planned job config\|does not match the resolved job lock' "$newest" 2>/dev/null; then
    return 0
  fi
  # Extract -p <jobpath> from the resume command line in the log.
  job=$(grep -oE '\-p[[:space:]]+/[^[:space:]]+' "$newest" 2>/dev/null | awk '{print $2}' | head -1)
  [[ -z "$job" || ! -d "$job" ]] && return 0
  # Require repeated failures on same job (same path in ≥3 recent logs).
  local hits
  hits=$(find "$ROOT" -name '*.resume_*.log' -mmin -20 2>/dev/null \
    | xargs grep -l "Existing trial config does not match\|$job" 2>/dev/null | wc -l)
  if [[ "${hits:-0}" -lt 5 ]]; then
    return 0
  fi
  if [[ ! -f "$job/result.json" ]]; then
    return 0
  fi
  dest="$(dirname "$job")/_stuck_resume_$(basename "$job")"
  if [[ -e "$dest" ]]; then
    return 0
  fi
  say "INTERVENE: archive lock-mismatched job $job → $dest (hits=$hits)"
  mv "$job" "$dest" 2>/dev/null || true
  restart_aa_ws
}

gateway_ok() {
  local code
  code=$(curl -sS -o /dev/null -w '%{http_code}' -m 8 http://127.0.0.1:4000/health 2>/dev/null || echo 000)
  # LiteLLM may 401 without key — any HTTP response means up
  [[ "$code" != "000" && "$code" != "" ]]
}

snapshot() {
  local age tech_line gpu docker_n free
  age=$(newest_result_age_min)
  tech_line=$("$REPO/.venv/bin/python" -m agent_bench.tech_failures --root "$ROOT" 2>/dev/null | head -1 || echo "?")
  gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr '\n' ',' || echo "?")
  docker_n=$(docker ps -q 2>/dev/null | wc -l)
  free=$(free_gib)
  say "snap age_min=$age free=${free}G gpu%=$gpu docker=$docker_n aa=$(aa_alive && echo y || echo n) harbor=$(harbor_alive && echo y || echo n) matrix=$(matrix_alive && echo y || echo n) krb=$(klist -s 2>/dev/null && echo y || echo n) | $tech_line"
}

px_fail=0
idle_rounds=0

say "watchdog start interval=${INTERVAL}s repo=$REPO (no image prune)"
ensure_krenew

while true; do
  ensure_krenew
  snapshot

  # 0) LiteLLM gateway
  if ! gateway_ok; then
    say "WARN: LiteLLM :4000 not responding"
  fi

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

  # 2) ghcr fail storm → bounce px
  storm=$(recent_ghcr_fail_storm)
  if [[ "${storm:-0}" -ge 5 ]]; then
    say "ghcr fail storm last10m=$storm"
    restart_px
    ensure_swe_pull
  fi

  # 3) lock mismatch thrash
  unstick_lock_mismatch

  # 4) zombies (containers only)
  prune_zombies

  # 5) swe image prefetch
  ensure_swe_pull

  # 6) disk hygiene without killing images
  disk_hygiene

  # 7) aa-ws / harbor liveness
  runner_bootstrapping=0
  if aa_alive && ! matrix_alive; then
    if pgrep -u "$USER" -f 'run_aa_index_workstation' >/dev/null 2>&1; then
      runner_bootstrapping=1
    fi
  fi
  if ! aa_alive || { ! matrix_alive && [[ "$runner_bootstrapping" -eq 0 ]]; }; then
    say "aa-ws or matrix dead"
    restart_aa_ws
    idle_rounds=0
    sleep "$INTERVAL"
    continue
  fi
  if [[ "$runner_bootstrapping" -eq 1 ]]; then
    say "aa-ws bootstrapping (pre-matrix); skip restart"
  fi

  # 8) progress watchdog
  age=$(newest_result_age_min)
  if harbor_alive || gpu_busy; then
    idle_rounds=0
  else
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

  if harbor_alive && [[ "$age" -ge 40 ]] && ! gpu_busy; then
    say "WARN harbor alive but no results ${age}m and GPU idle — possible hung install"
  fi

  sleep "$INTERVAL"
done
