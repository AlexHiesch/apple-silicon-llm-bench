# Temporary AA Index serving mode (TP=2 @ 128k)

**Status:** temporary for overnight AA Coding Agent Index only.  
**Prod default remains:** `vllm-int4` TP=1, 2 replicas, `max-model-len=65536`, Mooncake.

> **Dual-node update:** see [`TEMP_DUAL_TP2_128K.md`](./TEMP_DUAL_TP2_128K.md) — second TP2@128k on `cmtcdeu89976739` + `N_CONCURRENT=8`.

## Active settings (bench)

| Knob | Value |
|------|-------|
| Topology | 2× pods TP=2 @ 128k (`x40` + `x39`), Mooncake off |
| `max-model-len` | `131072` |
| Mooncake | off |
| `CLAUDE_CODE_MAX_OUTPUT_TOKENS` | `16384` (started at 32768; dialed down after `~98k+32k>128k` on schemelike) |
| `N_CONCURRENT` | `8` (dual-node; was 4) |
| `AGENT_TIMEOUT_MULT` | `1.5` (was 2.0; see dual doc) |
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
