#!/usr/bin/env bash
# AFK overnight A/B for ThinkingCap serving + tool-call stability.
# Runs on x40. Keeps aa-ws alive; serving rollouts may tech-fail in-flight trials
# (Harbor resume_until_content retries those).
#
# Never docker prune -af.
set -uo pipefail

REPO="${REPO:-$HOME/Projects/Work/llm-bench}"
ROOT="$REPO/results/agent_bench/aa_index"
LOG="$ROOT/OVERNIGHT_AB.log"
STAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="$ROOT/overnight_ab_$STAMP"
N_SMOKE="${N_SMOKE:-24}"
N_SMOKE_SAVE="$N_SMOKE"
WARMUP_SEC="${WARMUP_SEC:-180}"
export KUBECONFIG="${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUTDIR" "$ROOT"
exec >>"$LOG" 2>&1

ts() { date -Iseconds; }
say() { echo "[$(ts)] $*"; }

cd "$REPO"
say "=== overnight A/B start out=$OUTDIR n_smoke=$N_SMOKE ==="
echo "$STAMP" >"$ROOT/OVERNIGHT_AB_ACTIVE.txt"

smoke() {
  local label=$1
  say "tool-smoke label=$label n=$N_SMOKE"
  .venv/bin/python agent_bench/scripts/smoke_tool_calls.py \
    --n "$N_SMOKE" --label "$label" \
    --out "$OUTDIR/smoke_${label}.json" \
    --key-file "$HOME/llm-serving/aa-index-key" || true
}

rate_of() {
  local f=$1
  .venv/bin/python - <<PY
import json
from pathlib import Path
p=Path("$f")
if not p.exists():
  print("0")
else:
  print(json.loads(p.read_text())["summary"]["tool_use_rate"])
PY
}

content_smoke() {
  say "content smoke (PONG; try thinking + nothink)"
  KEY=$(kubectl -n llm-serving exec deploy/litellm -- printenv LITELLM_MASTER_KEY 2>/dev/null || true)
  [[ -n "$KEY" ]] || KEY=$(cat "$HOME/llm-serving/aa-index-key")
  .venv/bin/python - <<PY || return 1
import json, urllib.request
key="""$KEY"""

def once(extra):
    body={"model":"thinkingcap","max_tokens":64,"temperature":0,
          "messages":[{"role":"user","content":"Reply with exactly the word PONG and nothing else."}]}
    body.update(extra)
    req=urllib.request.Request(
      "http://127.0.0.1:4000/v1/chat/completions",
      data=json.dumps(body).encode(),
      headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
      data=json.loads(r.read())
    msg=(data.get("choices") or [{}])[0].get("message") or {}
    c=(msg.get("content") or "") or ""
    reason=(msg.get("reasoning_content") or "") or ""
    print("extra", extra, "content=",repr(c)[:80], "reason=",repr(reason)[:60])
    blob=(c+" "+reason).upper()
    return "PONG" in blob and blob.count("\n")<20

ok=False
try:
    ok = once({})
except Exception as e:
    print("thinking path FAIL", e)
try:
    ok = once({"chat_template_kwargs":{"enable_thinking":False}}) or ok
except Exception as e:
    print("nothink path FAIL", e)
raise SystemExit(0 if ok else 1)
PY
}

ensure_aa_ws() {
  if pgrep -u "$USER" -f 'agent_bench.run_matrix' >/dev/null 2>&1; then
    say "aa-ws/matrix alive"
    return 0
  fi
  say "INTERVENE: restart aa-ws"
  tmux has-session -t aa-ws 2>/dev/null && tmux kill-session -t aa-ws || true
  sleep 2
  tmux new-session -d -s aa-ws -c "$REPO" -- bash -lc \
    "bash agent_bench/scripts/run_aa_index_workstation_tp2_128k.sh 2>&1 | tee -a $ROOT/overnight_ws_ab_\$(date +%Y%m%d_%H%M%S).log"
  sleep 5
}

mark_short_fails() {
  say "mark short content_fails (out_tok<500, 1-ish turn) as AgentTimeout for Harbor retry"
  .venv/bin/python - <<'PY'
import json, shutil
from datetime import datetime, timezone
from pathlib import Path
from agent_bench.tech_failures import classify_result

job = Path("results/agent_bench/aa_index/terminal-bench-v2/claude-code")
jobs = sorted([p for p in job.glob("terminal-bench-v2__claude-code__*") if p.is_dir()], key=lambda p: p.stat().st_mtime)
if not jobs:
    print("no job"); raise SystemExit(0)
j = jobs[-1]
arch = j / f"_retry_shortfail_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
n = 0
for trial in j.iterdir():
    if not trial.is_dir() or trial.name.startswith("_"):
        continue
    rj = trial / "result.json"
    if not rj.exists():
        continue
    r = json.loads(rj.read_text())
    if classify_result(r) != "content_fail":
        continue
    out = (r.get("agent_result") or {}).get("n_output_tokens") or 0
    # short suspicious fails (XML/tool format)
    if out is None or out > 500:
        continue
    arch.mkdir(parents=True, exist_ok=True)
    if not (arch / trial.name).exists():
        shutil.copytree(trial, arch / trial.name)
    r["exception_info"] = {
        "exception_type": "AgentTimeoutError",
        "exception_message": "overnight_ab: remediating short content_fail (likely tool-format glitch)",
        "exception_traceback": None,
        "occurred_at": datetime.now(timezone.utc).isoformat(),
    }
    rj.write_text(json.dumps(r, indent=2) + "\n")
    n += 1
    print("marked", trial.name, "out_tok", out)
print("marked_n", n, "job", j.name)
PY
}

# --- background: rsync BF16 weights to x39 for later experiment ---
BF16_SRC="$HOME/llm-model-cache/huggingface/hub/models--bottlecapai--ThinkingCap-Qwen3.6-27B"
if [[ -d "$BF16_SRC" ]]; then
  say "start BF16 rsync → x39 (background)"
  nohup rsync -a --info=progress2 "$BF16_SRC/" \
    "hiescha@cmtcdeu89976739:llm-model-cache/huggingface/hub/models--bottlecapai--ThinkingCap-Qwen3.6-27B/" \
    >"$OUTDIR/rsync_bf16_x39.log" 2>&1 &
  echo $! >"$OUTDIR/rsync_bf16_x39.pid"
fi

# ========== Phase 0: baseline smoke on CURRENT serving (mtp-eager) ==========
smoke "00_current_live"
content_smoke && say "content_smoke OK" || say "WARN content_smoke FAIL"

# ========== Phase 1: TQ-only (hypothesis: MTP breaks tools) ==========
say "PHASE tq-only"
WARMUP_SEC="$WARMUP_SEC" bash agent_bench/k8s/switch-thinkingcap-serving.sh tq-only
smoke "01_tq_only"
content_smoke && say "tq-only content OK" || say "WARN tq-only content FAIL"
ensure_aa_ws
mark_short_fails

# ========== Phase 2: baseline no TQ/MTP ==========
say "PHASE baseline"
WARMUP_SEC="$WARMUP_SEC" bash agent_bench/k8s/switch-thinkingcap-serving.sh baseline
smoke "02_baseline"
content_smoke && say "baseline content OK" || say "WARN baseline content FAIL"
ensure_aa_ws

# ========== Decide INT4 winner ==========
R0=$(rate_of "$OUTDIR/smoke_00_current_live.json")
R1=$(rate_of "$OUTDIR/smoke_01_tq_only.json")
R2=$(rate_of "$OUTDIR/smoke_02_baseline.json")
say "rates: mtp_live=$R0 tq_only=$R1 baseline=$R2"

WINNER=$(.venv/bin/python - <<PY
rates={"tq-only": float("$R1"), "mtp-eager": float("$R0"), "baseline": float("$R2")}
# Prefer tq-only on ties with mtp (faster graphs); baseline only if clearly best
best=max(rates, key=rates.get)
# If tq-only within 0.05 of best and >= mtp, pick tq-only (speed+stability)
if rates["tq-only"] + 1e-9 >= rates["mtp-eager"] and rates["tq-only"] + 0.05 >= rates[best]:
    best="tq-only"
print(best)
print(rates, file=__import__("sys").stderr)
PY
)
say "WINNER_INT4=$WINNER"
WARMUP_SEC="$WARMUP_SEC" bash agent_bench/k8s/switch-thinkingcap-serving.sh "$WINNER"
smoke "03_winner_${WINNER}"
content_smoke || say "WARN winner content fail — still continuing"
ensure_aa_ws
mark_short_fails

# ========== Phase 3: BF16 if rsync done ==========
try_bf16() {
  local snap
  snap=$(ls -d "$HOME/llm-model-cache/huggingface/hub/models--bottlecapai--ThinkingCap-Qwen3.6-27B/snapshots"/*/ 2>/dev/null | head -1)
  [[ -n "$snap" ]] || { say "no local BF16 snapshot"; return 1; }
  # wait up to ~2h for rsync
  local pid
  pid=$(cat "$OUTDIR/rsync_bf16_x39.pid" 2>/dev/null || true)
  if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
    say "wait BF16 rsync pid=$pid (max 2h)"
    for _ in $(seq 1 120); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 60
    done
  fi
  if ! ssh -o BatchMode=yes -o ConnectTimeout=10 hiescha@cmtcdeu89976739 \
      "test -e llm-model-cache/huggingface/hub/models--bottlecapai--ThinkingCap-Qwen3.6-27B/snapshots/*/model.safetensors.index.json"; then
    say "BF16 not ready on x39 — skip BF16 phase"
    return 1
  fi
  say "PHASE bf16-tq (max_model_len=65536 first)"
  bash agent_bench/k8s/apply-thinkingcap-bf16.sh 65536 || {
    say "BF16 apply failed — restore INT4 winner $WINNER"
    WARMUP_SEC=120 bash agent_bench/k8s/switch-thinkingcap-serving.sh "$WINNER"
    return 1
  }
  smoke "04_bf16_64k"
  if ! content_smoke; then
    say "BF16 content smoke FAIL — revert to $WINNER"
    bash agent_bench/k8s/restore-thinkingcap-int4-config.sh || true
    WARMUP_SEC=120 bash agent_bench/k8s/switch-thinkingcap-serving.sh "$WINNER"
    return 1
  fi
  local rb
  rb=$(rate_of "$OUTDIR/smoke_04_bf16_64k.json")
  say "BF16 tool rate=$rb vs winner INT4"
  # Keep BF16 only if clearly better (+10pp) else revert for throughput/context
  .venv/bin/python - <<PY
win=float("$(rate_of "$OUTDIR/smoke_03_winner_${WINNER}.json")")
bf=float("$rb")
print("compare win", win, "bf16", bf)
raise SystemExit(0 if bf >= win + 0.10 else 1)
PY
  local keep=$?
  if [[ $keep -ne 0 ]]; then
    say "BF16 not clearly better — revert INT4 $WINNER"
    bash agent_bench/k8s/restore-thinkingcap-int4-config.sh || true
    WARMUP_SEC=120 bash agent_bench/k8s/switch-thinkingcap-serving.sh "$WINNER"
  else
    say "KEEP BF16 (tool rate win)"
    echo "bf16-tq-64k @ $(date -Iseconds)" | tee "$ROOT/SERVING_MODE.txt"
  fi
}

try_bf16 || say "BF16 phase skipped/failed"

# ========== Guard loop until morning ==========
say "enter guard loop (mark short fails + ensure aa-ws)"
for hour in $(seq 1 16); do
  say "guard tick $hour/16"
  ensure_aa_ws
  mark_short_fails
  # periodic tool pulse (5)
  N_SMOKE=5
  smoke "guard_${hour}" || true
  N_SMOKE="${N_SMOKE_SAVE:-24}"
  # stop around 08:00 local if long enough
  hour_now=$(date +%H)
  if [[ "$hour_now" =~ ^0[7-9]$ || "$hour_now" =~ ^1[0-2]$ ]] && [[ $hour -ge 4 ]]; then
    say "morning window — stop guard loop"
    break
  fi
  sleep 1800
done

{
  echo "overnight_ab finished $(date -Iseconds)"
  echo "outdir=$OUTDIR"
  echo "winner_int4=$WINNER"
  echo "serving_mode=$(cat "$ROOT/SERVING_MODE.txt" 2>/dev/null | head -1)"
  ls -la "$OUTDIR"
} | tee "$OUTDIR/DONE.txt" | tee "$ROOT/OVERNIGHT_AB_DONE.txt"
rm -f "$ROOT/OVERNIGHT_AB_ACTIVE.txt"
say "=== overnight A/B done ==="
