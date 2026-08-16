#!/usr/bin/env bash
# Run ON the Z8 (via ssh). Prints one health snapshot.
set -uo pipefail
REPO=/home/hiescha/Projects/Work/llm-bench
ROOT=$REPO/results/agent_bench/aa_index
krb=DEAD; klist -s && krb=OK
krenew=DEAD; pgrep -x krenew >/dev/null && krenew=OK
px=$(curl -sS -o /dev/null -w '%{http_code}' -x http://127.0.0.1:3128 --connect-timeout 12 -I https://github.com 2>/dev/null || echo 000)
free=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
aa=n; tmux has-session -t aa-ws 2>/dev/null && aa=y
watch=n; tmux has-session -t aa-watch 2>/dev/null && watch=y
guard=n; tmux has-session -t aa-guard 2>/dev/null && guard=y
matrix=n; pgrep -u "$USER" -f 'agent_bench.run_matrix' >/dev/null && matrix=y
harbor=n; pgrep -u "$USER" -f 'harbor (run|job resume)' >/dev/null && harbor=y
gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr '\n' '/' || echo '?')
swe=$(docker images ghcr.io/scaleapi/swe-atlas -q 2>/dev/null | wc -l | tr -d ' ')
dock=$(docker ps -q 2>/dev/null | wc -l | tr -d ' ')
pane=$(tmux capture-pane -t aa-ws -p -S -10 2>/dev/null | tr '\n' ' ' | sed 's/  */ /g' | tail -c 280)
python3 - <<'PY'
import json
from pathlib import Path
root=Path("/home/hiescha/Projects/Work/llm-bench/results/agent_bench/aa_index")
job=root/"terminal-bench-v2/claude-code/terminal-bench-v2__claude-code__20260717_154234"
if job.is_dir() and (job/"result.json").is_file():
    j=json.loads((job/"result.json").read_text())
    s=j.get("stats") or {}
    print(
        "claude_job completed={} err={} run={} pend={}".format(
            s.get("n_completed_trials"),
            s.get("n_errored_trials"),
            s.get("n_running_trials"),
            s.get("n_pending_trials"),
        )
    )
logs=sorted(root.glob("overnight_ws*.log"), key=lambda p: p.stat().st_mtime)
if logs:
    for ln in logs[-1].read_text(errors="replace").splitlines():
        if "/75]" in ln or "tech_stagnant" in ln or "Ensure SWE" in ln:
            print("log:", ln[:200])
PY
echo "HEALTH krb=$krb krenew=$krenew px=$px free=${free}G aa=$aa watch=$watch guard=$guard matrix=$matrix harbor=$harbor gpu=$gpu swe=$swe dock=$dock"
echo "PANE $pane"
echo "WATCH $(tail -3 "$ROOT/WATCHDOG.log" 2>/dev/null | tr '\n' ';')"
echo "GUARD $(tail -2 "$ROOT/NIGHT_GUARDIAN.log" 2>/dev/null | tr '\n' ';')"
