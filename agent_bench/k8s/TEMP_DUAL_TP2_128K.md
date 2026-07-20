# Temporary dual-node AA Index serving (TP=2 @ 128k × 2)

**Status:** temporary — x40 + x39 each run one TP=2 @ 128k replica.  
**Runner:** still on x40 for now. x39 has Docker 29.3 installed (2026-07-19) but
`/var/run/docker.sock` is still `root:root` (needs `SocketGroup=docker` applied) —
Harbor trials on x39 blocked until that one sudo fix.  
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
| `N_CONCURRENT` | `2` (was 4; lowered under TQ+MTP+eager after tech/api_retry pile-up; N=8 = timeout storm) |
| vLLM `--max-num-seqs` | `8` (headroom above N=4) |
| `CLAUDE_CODE_MAX_OUTPUT_TOKENS` | `16384` |
| `AGENT_TIMEOUT_MULT` | `1.0` (official Harbor / TB; was temporary `1.5`) |
| Mooncake | off (GPU prefix cache already ~95% hits; KV usage low) |

### MTP + TurboQuant

**On** (josefprusa INT4 recipe), via `enable-thinkingcap-mtp.sh`:
- `--kv-cache-dtype turboquant_4bit_nc`
- `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`
- `--max-num-batched-tokens 4096` (required: MTP default 2048 &lt; TQ/Mamba block ~3088)
- `--enforce-eager` — **required on vLLM 0.24**: TQ×MTP with FULL cudagraph produces degenerate token loops (`empty, empty…`, empty tool args). Upstream: [vllm#40831](https://github.com/vllm-project/vllm/issues/40831), [vllm#40880](https://github.com/vllm-project/vllm/issues/40880). TQ alone and MTP alone are fine; only the combo + cudagraph breaks. Fix lands in newer vLLM; until then eager keeps both features correct (some decode TPS cost).

Isolation (x39 A/B, after Ready + ≥90s warmup, `max_tokens≥512`):

| Config | thinking PONG | tools |
|--------|---------------|-------|
| baseline | OK | OK |
| TQ only | OK | OK |
| MTP only | OK | OK |
| TQ+MTP (cudagraph) | FAIL loops | FAIL |
| TQ+MTP + `--enforce-eager` | OK | OK |

Enable script waits for Ready + **warmup** (default 180s — GPU load is not the same as Ready), then smokes with `max_tokens=512`. Short budgets with thinking-on can look like “empty content” even on baseline. On smoke failure → auto-revert.

**DFlash** still skipped (≠ TurboQuant; custom vLLM).

Scripts: `enable-thinkingcap-mtp.sh`, `revert-thinkingcap-mtp.sh`, `check-thinkingcap-serving.sh`

Harbor `job resume` has **no** `-n` flag. Resume rebuilds the lock from
`config.json` (Harbor default `n_concurrent_trials=4`) and errors if that
disagrees with existing `lock.json`. `run_harbor.set_job_n_concurrent` patches
**both** files to match `--n-concurrent` / `N_CONCURRENT` before each resume.

LiteLLM routing for this mode: `least-busy` + `session_affinity` (see
`litellm-config.bench-dual-tp2.yaml`). Re-apply with
`bash agent_bench/k8s/apply-litellm-dual-routing.sh`.

### Timeout rationale

- Official benchmark: Harbor default / TB leaderboard use
  ``agent_timeout_multiplier=1.0`` (no ``--agent-timeout-multiplier``).
- Temporary ``1.5`` gave extra agent wall time; passes with
  ``agent_duration > task [agent] timeout_sec`` are invalid at 1.0 and are
  re-queued via ``agent_bench/scripts/requeue_inflated_timeout_trials.py``.
- ``AgentTimeoutError`` stays **tech** → ``resume_until_content`` retries.
- Resume copies multiplier from job ``config.json`` / ``lock.json``; patch with
  ``run_harbor.set_job_agent_timeout_multiplier`` when changing the env knob.

## Overnight serving A/B (AFK)

Autonomous loop on x40 (`tmux aa-ab`):

```bash
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
bash agent_bench/scripts/overnight_ab_serving.sh
# log: results/agent_bench/aa_index/OVERNIGHT_AB.log
```

Sequence: tool-smoke on live MTP → switch **tq-only** → **baseline** → pick INT4 winner → optional **BF16** (rsync to x39, 64k first) → guard loop marking short content_fails for Harbor retry.

Helpers: `switch-thinkingcap-serving.sh {mtp-eager|tq-only|baseline}`, `smoke_tool_calls.py`, `apply-thinkingcap-bf16.sh`.

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
export AGENT_TIMEOUT_MULT=1.0
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

## Harbor trials on x39 (Docker now installed)

Docker **29.3.1** is on x39 (`hiescha` ∈ group `docker`). Unit already has
`SocketGroup=docker`, but the live socket was left as `root:root` 0660 →
`permission denied` for non-root.

**One-shot for a sudo colleague:**

```bash
# on cmtcdeu89976739
sudo chgrp docker /var/run/docker.sock
sudo chmod 660 /var/run/docker.sock
# durable: restart socket so systemd recreates with SocketGroup=docker
sudo systemctl restart docker.socket docker.service
# verify as hiescha (new SSH login if needed):
docker ps
```

Also confirm `/etc/group` lists `hiescha` under `docker` (`getent group docker`).
`id` already shows gid 986; empty member list in `getent` is a smell.

After sock works: install Harbor (`uv tool install harbor`), sync `llm-bench`,
point trials at LiteLLM on x40 (`host.docker.internal` / x40 IP + NO_PROXY),
then optionally split matrix runners (x40 + x39) — GPUs on x39 stay on
`vllm-int4-x39` (inference); Harbor only needs CPU/disk for sandboxes.

k3s still has **no** generic Harbor env — Docker remains the path.
