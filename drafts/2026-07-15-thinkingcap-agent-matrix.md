<!-- DRAFT — local only. Do not publish / post / submit anywhere. -->

# ThinkingCap overnight: one local 27B vs a matrix of agent CLIs

**Status:** DRAFT (2026-07-15). Not published.

Machine: Apple M3 Max, 64 GB. Model: `t-prazak/ThinkingCap-Qwen3.6-27B-MLX-4bit`. Window: ~00:39 → 06:22 CEST (~5h 45m), Mac asleep-proofed with `caffeinate`.

The question was simple and, as far as I can tell, unanswered in public: can you point a pile of coding-agent CLIs at the same locally loaded ThinkingCap weights overnight, measure who actually ships a file, and also get speed numbers — without touching cloud quotas?

Short answer: partially. Two CLIs wrote the artifact. The speed harness finished clean. Most of the “all agents / all evals” ambition hit the usual local-agent tax — auth walls, MCP dumps, Azure redirects, and one process that ate the port the others needed.

## Setup

ThinkingCap stays on the host (MLX). Serving stack:

| Port | Process | Role |
|------|---------|------|
| `:8080` | Kevlar | Anthropic-compatible `/v1/messages` (what Claude Code wants) |
| `:8091` | thin OpenAI→Anthropic shim | `/v1/chat/completions` + SSE for agents that only speak OpenAI |

Same weights. Two wire formats. Orchestrator: `agent_bench/run_overnight_full.py` under tmux `overnight-tc`.

Pass rule for agents was not “exit 0”. It was: did you create `hello_tc.py` that prints `ThinkingCap-OK`?

## Agent matrix (same task, local ThinkingCap)

| CLI | Result | Time | Note |
|-----|--------|------|------|
| Claude Code (`--bare`) | **pass** | 8s | Wrote `hello_tc.py` |
| Aider (`--no-stream` → shim) | **pass** | 12s | Same artifact |
| Mimo | fail | 10m | SSE read timed out after tooling |
| Goose | stuck | 45m | killed — no log progress |
| OpenCode | fail | 2s | opaque “Unexpected server error” |
| Hermes | fail | 2s | unknown provider `openai` |
| Kilo Code | fail | 6s | no artifact |
| Antigravity (`agy`) | fail | 2s | exit 2 |
| Codex | fail | 12s | still called Azure OpenAI (`dev-cloudlab-aoai…`), 404 |
| Cursor agent CLI | fail | 2s | needs `agent login` / `CURSOR_API_KEY` |
| Pier / Harbor install | fail | — | `uv tool install` failed |

Artifacts that landed:

```python
print('ThinkingCap-OK')
```

Twice. Claude Code and Aider. That is the honest agent result for this night: **2/10 installed CLIs** completed the micro-task against local ThinkingCap under automated headless conditions.

### What burned time before numbers

Claude Code without `--bare` shoved ~40k tokens of MCP tool schemas into the 27B prefill. Prefill alone was ~157s for one turn. `--bare` dropped that to a usable agent. If you blog “local Claude Code on ThinkingCap” without mentioning MCP weight, you are lying by omission.

Codex ignored the local base URL and kept its cloud Azure endpoint. Cursor CLI refused unauthenticated. Those are product constraints, not model quality.

## Speed results (the part that actually measured)

`benchmark.py --group TC` finished. HTML/CSV: `results/bench_20260715_062244.{html,csv}`. Peak RSS ~15 GB.

Medians that matter on this box:

| Backend | Prompt | Decode | Warm TTFT | Notes |
|---------|--------|--------|-----------|-------|
| mlx-lm 4bit | short | **20.1 t/s** | ~208 ms | baseline |
| mlx-lm 4bit | code (~751 tok) | **21.6 t/s** | ~211 ms | |
| vllm-mlx 4bit | short | **22.9 t/s** | ~366 ms | fastest decode here |
| oMLX 4bit ssd-paged | short | **20.6 t/s** | ~509 ms | |
| mlx-lm 4bit | context-32k | **20.7 t/s** | ~317 ms warm | cold TTFT ~172 s |
| mlx-lm 4bit | context-64k | **18.0 t/s** | ~475 ms warm | cold TTFT ~397 s |
| mlx-lm 4bit **think** | short | 19.0 t/s | **~48 s TTFT** | thinking tax is latency, not just tokens |

So on M3 Max 64GB: ThinkingCap-27B-4bit is a ~20–23 tok/s local coding model with usable 64k warm latency after the first cold prefill. Enabling “think” roughly adds 50 seconds before the first visible token on a short prompt. That is the ThinkingCap trade you feel in an agent loop.

## Quality harnesses this night

Orchestrator reported `ok=7 failed=16`. Several “ok” checks were scaffolding (smoke, plan-only). Quality side was messy once the speed bench spun its own `mlx_lm.server` and Kevlar/shim path degraded:

- HumanEval: ran for the 3h hard timeout; progress showed 0/80… because responses never returned usable completions under the dying/hung shim.
- BFCL: missing dataset path under `/tmp/gorilla/...` — data not staged.
- Context-bench overnight pass: 0/10 with connection errors to `:8091` (server gone / 502), not a model score.
- Thinking-token efficiency script: reported 0% across the board with 0 think tokens — instrumentation failed against the shim, not a real ThinkingCap scorecard.
- HellaSwag: intentionally skipped overnight (`skipped_ram`) so a second full MLX load would not fight the resident server.

Read that as: **agent smoke + speed bench are the publishable substance from this run**. Do not quote the overnight quality percentages as ThinkingCap IQ until a clean second pass with a single stable OpenAI endpoint and staged datasets.

Older ThinkingCap quality JSON under `results/*thinkingcap*` from earlier sessions still exists for a proper follow-up post.

## What is actually new

Not “27B is fast on Apple Silicon” — everyone has that chart by now.

What I had not seen written up:

1. **One ThinkingCap MLX process** serving both Anthropic-shaped and OpenAI-shaped agent CLIs overnight.
2. **Artifact-gated** agent scoring (file on disk), not vibes.
3. Documented **CLI failure taxonomy** on a lab-locked Mac: MCP bloat, auth walls, cloud endpoint gravity, SSE timeouts, install gaps.
4. Side-by-side decode numbers for ThinkingCap across mlx-lm / vllm-mlx / oMLX plus 32k/64k and think-mode TTFT on the same night.

## If I re-ran tomorrow

1. Keep Kevlar up for the whole agent matrix; run the speed group **after**, or on a second machine — do not let `benchmark.py` steal `:8090` / RAM mid-eval.
2. Stage BFCL/HumanEval data before the orchestrator starts.
3. Hard-block cloud base URLs in Codex/Goose configs for the run profile.
4. Default every Claude-family CLI to `--bare` (or a settings file with MCP empty).
5. Cap agent wall time at ~10 minutes; 45 minutes of silence is just heat.

## Numbers to paste if you only take one table

**Agent microbench (local ThinkingCap, headless):** Claude Code ✓ 8s · Aider ✓ 12s · 8 other CLIs ✗  

**Decode (mlx-lm 4bit, warm):** ~20–22 t/s · **vllm-mlx:** ~23 t/s · **64k context warm TTFT:** ~0.5 s · **think mode short TTFT:** ~48 s  

**Peak RSS:** ~15 GB on 64 GB unified memory.

---

Raw trail: `results/agent_bench/overnight/summary.json`, `steps.jsonl`, `overnight_20260715_003905.log`, `results/bench_20260715_062244.html`.

Repo WIP: https://github.com/AlexHiesch/apple-silicon-llm-bench/pull/1
