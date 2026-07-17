# Temporary AA Index serving mode (TP=2 @ 128k)

**Status:** temporary for overnight AA Coding Agent Index only.  
**Prod default remains:** `vllm-int4` TP=1, 2 replicas, `max-model-len=65536`, Mooncake.

## Active settings (bench)

| Knob | Value |
|------|-------|
| Topology | 1× pod, `--tensor-parallel-size 2`, 2 GPUs on `cmtcdeu89976740` |
| `max-model-len` | `131072` |
| Mooncake | off |
| `CLAUDE_CODE_MAX_OUTPUT_TOKENS` | `16384` (started at 32768; dialed down after `~98k+32k>128k` on schemelike) |
| `N_CONCURRENT` | `2` |
| `AGENT_TIMEOUT_MULT` | `2.0` |
| LiteLLM `request_timeout` | `1800` (was 600) |
| LiteLLM `nodeSelector` | pinned to Z8 (`cmtcdeu89976740`) for `hostNetwork :4000` |

## Activate / revert (on Z8)

```bash
# activate temporary TP2@128k
bash ~/llm-serving/k8s/activate-vllm-bench-tp2-128k.sh

# runner
tmux new-session -d -s aa-ws -c ~/Projects/Work/llm-bench -- \
  bash agent_bench/scripts/run_aa_index_workstation_tp2_128k.sh

# revert serving to prod TP=1 dual-replica
bash ~/llm-serving/k8s/revert-vllm-prod-tp1.sh
# then restore LiteLLM timeout 1800→600 from ~/llm-serving/k8s-backups/litellm-config.prod-*.yaml
# and remove litellm nodeSelector if undesired for multi-node
```

Manifests live on the workstation under `~/llm-serving/k8s/` (not applied from this repo by default).
