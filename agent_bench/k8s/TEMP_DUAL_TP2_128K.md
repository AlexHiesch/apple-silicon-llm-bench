# Temporary dual-node AA Index serving (TP=2 @ 128k × 2)

**Status:** temporary — x40 + x39 each run one TP=2 @ 128k replica.  
**Runner:** stays on x40 (x39 has no Docker; limited sudo).  
**Results:** Harbor job dirs on x40 are kept; matrix uses `--resume-harbor` + tech retry-until-content.

## Topology

| Node | Role |
|------|------|
| `cmtcdeu89976740` (x40) | `deploy/vllm-int4` TP=2 @ 128k, LiteLLM hostNetwork `:4000`, Harbor/`aa-ws` |
| `cmtcdeu89976739` (x39) | `deploy/vllm-int4-x39` TP=2 @ 128k, idle GPUs → now serving |

LiteLLM `thinkingcap` → **least-busy** across:
- `http://vllm-int4.llm-serving.svc:8000/v1` (x40, `model_info.id=thinkingcap-x40`)
- `http://vllm-int4-x39.llm-serving.svc:8000/v1` (x39, `model_info.id=thinkingcap-x39`)

Session stickiness: `session_affinity` (TTL 3h). Claude Code already sends
`x-claude-code-session-id`; LiteLLM maps that into `metadata.session_id`, so
follow-up turns of one trial stay on the same vLLM replica (prefix cache).
Do **not** enable `deployment_affinity` — one shared API key would pin all
trials to a single node.

## Runner knobs

| Knob | Value |
|------|-------|
| `N_CONCURRENT` | `4` (~2 per node; N=8 caused timeout storm) |
| vLLM `--max-num-seqs` | `8` (headroom above N=4) |
| `CLAUDE_CODE_MAX_OUTPUT_TOKENS` | `16384` |
| `AGENT_TIMEOUT_MULT` | `1.5` |
| Mooncake | off (GPU prefix cache already ~95% hits; KV usage low) |

### MTP + TurboQuant (tried 2026-07-19, reverted)

Tried josefprusa recipe (`mtp` n=1 + `turboquant_4bit_nc` + `enable_thinking` chat kwargs).
First boot needed `--max-num-batched-tokens 4096` (MTP default 2048 &lt; TQ/Mamba block ~3088).
After enable, short/agent-facing smokes looked unhealthy during the bounce; **rolled back** to this baseline via `revert-thinkingcap-mtp.sh`. Baseline re-verified (`max_tokens≥256` → content `PONG`; Harbor trials advancing again).

Scripts:
- `enable-thinkingcap-mtp.sh` — opt-in re-enable (x39→x40)
- `revert-thinkingcap-mtp.sh` — back to baseline
- `check-thinkingcap-serving.sh` — smoke + auto-revert if TQ flags present and content broken

**Still skipped:** FP8 weights, DFlash (≠ TurboQuant).

Harbor `job resume` has **no** `-n` flag. Resume rebuilds the lock from
`config.json` (Harbor default `n_concurrent_trials=4`) and errors if that
disagrees with existing `lock.json`. `run_harbor.set_job_n_concurrent` patches
**both** files to match `--n-concurrent` / `N_CONCURRENT` before each resume.

LiteLLM routing for this mode: `least-busy` + `session_affinity` (see
`litellm-config.bench-dual-tp2.yaml`). Re-apply with
`bash agent_bench/k8s/apply-litellm-dual-routing.sh`.

### Timeout rationale (no false content fails)

- `AgentTimeoutError` is in `TECH_EXCEPTION_TYPES` → scored **tech**, retried by
  `resume_until_content`. A tighter multiplier can never turn a timeout into a
  `content_fail`.
- Empirically on TB/claude-code (job `…20260717_154234`): **1.5x** left all
  observed passes under the agent cap; **1.25x** would have timed out 2 real
  passes (`largest-eigenval`, `extract-elf`) as tech retries.
- Env `2.0` was wasteful: stuck tasks already die at `base×1.5` (e.g. caffe
  1200→1800s). Align runner + docs on **1.5**.
- Resume reuses per-trial `agent_timeout_multiplier` from the job lock (already
  1.5 on the active job); the env knob applies to **new** `harbor run` jobs.

## Activate / deactivate

```bash
# on x40
cd ~/Projects/Work/llm-bench
bash agent_bench/k8s/activate-dual-tp2-128k.sh
# optional later: bash agent_bench/k8s/enable-thinkingcap-mtp.sh  # MTP+TQ (validate with check-thinkingcap-serving.sh)
bash agent_bench/k8s/apply-litellm-dual-routing.sh   # least-busy + session_affinity
bash agent_bench/k8s/smoke-litellm-session-affinity.sh
bash agent_bench/k8s/patch-grafana-dual-dashboard.sh
bash agent_bench/k8s/check-thinkingcap-serving.sh    # smoke; auto-revert if broken

# update runner env + restart aa-ws (results preserved via resume)
cat > results/agent_bench/aa_index/BENCH_TP2_128K.env <<'EOF'
export N_CONCURRENT=4
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=16384
export AGENT_TIMEOUT_MULT=1.5
EOF
tmux kill-session -t aa-ws 2>/dev/null || true
tmux new-session -d -s aa-ws -c ~/Projects/Work/llm-bench -- \
  bash agent_bench/scripts/run_aa_index_workstation_tp2_128k.sh

# remove x39 only (keep x40 TP2)
bash agent_bench/k8s/deactivate-dual-tp2-x39.sh

# full revert to prod TP1 dual-replica
bash ~/llm-serving/k8s/revert-vllm-prod-tp1.sh
```

## Metrics

- Grafana: `http://cmtcdeu89976740.rd.corpintra.net:30300/d/hpllm-dual-bench/`
- Also: HPLLM GPU/DCGM (`node_short=x40|x39`), vLLM/Mooncake, Overview
- Prometheus: `:30090` — job `vllm-int4` scrapes both pods with `node_short` + `bench_node`

## Why not Harbor trials on x39 / k3s (without Docker)

Harbor 0.18 env types include `docker`, `gke`, `openshift`, `ack`, cloud
sandboxes (`e2b`, `modal`, …) — **no generic Kubernetes / k3s backend**.

| Option | Blocker on this cluster |
|--------|-------------------------|
| `--env docker` on x39 | No Docker package; limited sudo (no `apt install docker`) |
| `--env gke` | GCP auth + GKE API; still runs tasks via **DinD** in pods |
| `--env openshift` | Needs `oc` + privileged SCC, not plain k3s |
| `--env ack` | Alibaba OpenKruise SandboxClaim CRDs |

So x39 stays **inference-only**. Trial sandboxes stay on x40 Docker; LiteLLM
spreads decode across both vLLM replicas. A future upstream generic k8s env
(or installing Docker on x39) would be required to move Harbor trials off x40.
