# Temporary dual-node AA Index serving (TP=2 @ 128k × 2)

**Status:** temporary — x40 + x39 each run one TP=2 @ 128k replica.  
**Runner:** stays on x40 (x39 has no Docker; limited sudo).  
**Results:** Harbor job dirs on x40 are kept; matrix uses `--resume-harbor` + tech retry-until-content.

## Topology

| Node | Role |
|------|------|
| `cmtcdeu89976740` (x40) | `deploy/vllm-int4` TP=2 @ 128k, LiteLLM hostNetwork `:4000`, Harbor/`aa-ws` |
| `cmtcdeu89976739` (x39) | `deploy/vllm-int4-x39` TP=2 @ 128k, idle GPUs → now serving |

LiteLLM `thinkingcap` → simple-shuffle across:
- `http://vllm-int4.llm-serving.svc:8000/v1` (x40)
- `http://vllm-int4-x39.llm-serving.svc:8000/v1` (x39)

## Runner knobs

| Knob | Value |
|------|-------|
| `N_CONCURRENT` | `4` (~2 per node) |
| `CLAUDE_CODE_MAX_OUTPUT_TOKENS` | `16384` |
| `AGENT_TIMEOUT_MULT` | `2.0` |
| Mooncake | off (GPU prefix cache already ~95% hits; KV usage low) |

Harbor `job resume` has **no** `-n` flag — concurrency lives in the job's
`lock.json` (`n_concurrent_trials`). `run_harbor.resume_*` patches that field
to match `--n-concurrent` / `N_CONCURRENT` before each resume so dual-node
bumps stick on existing jobs.

## Activate / deactivate

```bash
# on x40
cd ~/Projects/Work/llm-bench
bash agent_bench/k8s/activate-dual-tp2-128k.sh
bash agent_bench/k8s/patch-grafana-dual-dashboard.sh

# update runner env + restart aa-ws (results preserved via resume)
cat > results/agent_bench/aa_index/BENCH_TP2_128K.env <<'EOF'
export N_CONCURRENT=4
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=16384
export AGENT_TIMEOUT_MULT=2.0
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

## Why not Harbor on x39

No Docker package; limited sudo has no `apt install docker`. Inference scale-out is the available lever without host package changes.
