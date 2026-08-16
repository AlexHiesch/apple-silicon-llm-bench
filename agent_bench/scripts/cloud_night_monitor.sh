#!/usr/bin/env bash
# Cloud-side overnight poller for Z8 AA Index.
set -uo pipefail

HOST="${Z8_SSH:-hiescha@cmtcdeu89976740.rd.corpintra.net}"
PORT="${Z8_PORT:-42022}"
REPO_LOCAL="${REPO_LOCAL:-/Users/HIESCHA/Projects/Work/llm-bench}"
LOG="$REPO_LOCAL/results/agent_bench/aa_index/CLOUD_NIGHT_MONITOR.log"
PROBE="$REPO_LOCAL/agent_bench/scripts/z8_health_probe.sh"
INTERVAL="${MONITOR_INTERVAL:-180}"

mkdir -p "$(dirname "$LOG")"
say() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

say "cloud night monitor start host=$HOST interval=${INTERVAL}s"
fail_ssh=0

# Keep probe in sync on Z8
scp -o BatchMode=yes -o ConnectTimeout=20 -P "$PORT" "$PROBE" \
  "$HOST:/home/hiescha/Projects/Work/llm-bench/agent_bench/scripts/z8_health_probe.sh" \
  >>"$LOG" 2>&1 || say "WARN: initial scp probe failed"

while true; do
  out=$(ssh -o BatchMode=yes -o ConnectTimeout=25 -p "$PORT" "$HOST" \
    "bash /home/hiescha/Projects/Work/llm-bench/agent_bench/scripts/z8_health_probe.sh" 2>&1) || true

  if [[ -z "$out" ]] || echo "$out" | grep -Eq 'Connection refused|Could not resolve|Permission denied|Connection timed out|No such file'; then
    fail_ssh=$((fail_ssh + 1))
    say "SSH_FAIL streak=$fail_ssh :: ${out:0:240}"
    if [[ "$fail_ssh" -eq 2 ]]; then
      scp -o BatchMode=yes -P "$PORT" "$PROBE" \
        "$HOST:/home/hiescha/Projects/Work/llm-bench/agent_bench/scripts/z8_health_probe.sh" \
        >>"$LOG" 2>&1 || true
    fi
    sleep 60
    continue
  fi
  fail_ssh=0
  say "$out"

  need_babysit=0
  echo "$out" | grep -q 'aa=n' && need_babysit=1
  echo "$out" | grep -q 'matrix=n' && need_babysit=1
  echo "$out" | grep -q 'guard=n' && need_babysit=1
  echo "$out" | grep -q 'watch=n' && need_babysit=1
  echo "$out" | grep -q 'krb=DEAD' && need_babysit=1
  echo "$out" | grep -q 'px=000' && need_babysit=1

  if [[ "$need_babysit" -eq 1 ]]; then
    say "INTERVENE: babysit bootstrap"
    ssh -o BatchMode=yes -o ConnectTimeout=25 -p "$PORT" "$HOST" \
      'bash /home/hiescha/Projects/Work/llm-bench/agent_bench/scripts/overnight_babysit.sh' \
      >>"$LOG" 2>&1 || true
  fi

  free_n=$(echo "$out" | sed -n 's/.*free=\([0-9]*\)G.*/\1/p' | head -1)
  if [[ -n "${free_n:-}" && "$free_n" -lt 20 ]]; then
    say "INTERVENE: disk hard free=${free_n} — purge junk (no prune)"
    ssh -o BatchMode=yes -p "$PORT" "$HOST" \
      'ROOT=/home/hiescha/Projects/Work/llm-bench/results/agent_bench/aa_index
       find "$ROOT" -maxdepth 3 -type d \( -name "_broken_*" -o -name "_failed_*" -o -name "_smoke*" -o -name "_stuck_*" \) -exec rm -rf {} + 2>/dev/null || true
       for d in "$HOME/finetune-output/orgchart-thinkingcap" "$HOME/sglang-bench/.venv" "$HOME/litellm-v191-test/.venv" "$HOME/llama-build"; do
         [[ -e "$d" ]] || continue
         echo "rm -rf $d"; rm -rf "$d" || true
         free=$(df -BG --output=avail / | tail -1 | tr -dc "0-9")
         [[ "$free" -ge 25 ]] && break
       done
       df -h / | tail -1' >>"$LOG" 2>&1 || true
  fi

  sleep "$INTERVAL"
done
