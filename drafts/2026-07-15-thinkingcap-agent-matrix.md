<!-- DRAFT — local only. Do not publish / post / submit anywhere. -->

# ThinkingCap overnight: one local 27B vs a matrix of agent CLIs

**Status:** DRAFT (2026-07-15, revised after Docker re-smoke). Not published.

Machine: Apple M3 Max, 64 GB. Model: `t-prazak/ThinkingCap-Qwen3.6-27B-MLX-4bit`. Overnight window: ~00:39 → 06:22 CEST (~5h 45m), Mac asleep-proofed with `caffeinate`. Docker CLI scorecard re-run later the same morning.

The question was simple and, as far as I can tell, unanswered in public: can you point a pile of coding-agent CLIs at the same locally loaded ThinkingCap weights overnight, measure who actually ships a file, and also get speed numbers — without touching cloud quotas?

Short answer after the Docker pass: **six Linux agent CLIs write the artifact against the same host ThinkingCap**. Four more stay SKIP (mac-only or auth). The overnight native run was harsher (2/10) because Goose/OpenCode fought the desktop session and Codex/Hermes hit config gaps — that is why agent smokes now prefer Docker.

## Setup

ThinkingCap stays on the host (MLX). Serving stack:

| Port | Process | Role |
|------|---------|------|
| `:8080` | Kevlar | Anthropic-compatible `/v1/messages` (Claude Code) |
| `:8091` | OpenAI→Anthropic shim | `/v1/chat/completions` (+ SSE) and `/v1/responses` (+ SSE `response.completed` for Codex) |

Same weights. Two wire formats. Orchestrator: `agent_bench/run_overnight_full.py`. Prefer agent micro-smokes via:

```bash
docker compose -f agent_bench/docker-compose.yml --profile cli run --rm --user 1000:1000 -e AGENT_WALL_SEC=480 cli-smoke
```

Pass rule is not exit 0. It is: did you create `hello_tc.py` that prints `ThinkingCap-OK`?

## Docker agent scorecard (headless Linux → host ThinkingCap)

| CLI | Result | Note |
|-----|--------|------|
| Claude Code (`--bare`) | **pass** | Anthropic → Kevlar |
| Aider | **pass** | Chat Completions → shim |
| OpenCode | **pass** | Needs `--dir` + `external_directory: deny` + shell overwrite — model invents `/Users/...` paths otherwise |
| Goose | **pass** | Headless Linux binary (avoids macOS GUI popup) |
| Hermes | **pass** | `--provider thinkingcap --ignore-user-config` |
| Codex | **pass** | Needs Responses SSE + explicit `exec_command` prompt (generic prompt → dumps tool JSON as text) |
| Mimo / Kilo | **skip** | not packaged in this Linux image |
| Antigravity (`agy`) | **skip** | macOS binary |
| Cursor agent CLI | **skip** | requires Cursor login / `CURSOR_API_KEY` |

Artifact that landed (representative):

```python
print('ThinkingCap-OK')
```

Honest Docker result: **6 PASS / 0 FAIL / 4 SKIP** among the targets that were in scope. Pier/Harbor full suites remain plan-only wrappers (not run).

### What burned time before those greens

Claude Code without `--bare` shoved ~40k tokens of MCP tool schemas into the 27B prefill. Prefill alone was ~157s for one turn. `--bare` dropped that to a usable agent.

OpenCode’s model freely invents absolute paths (`/home/user`, `/Users/sabeer/...`). Deny `external_directory` and force a one-shot shell write.

Codex on ThinkingCap: first the shim had to emit Responses SSE ending in `response.completed`. Then a generic “write a file” prompt caused the 27B to echo Codex’s huge tool inventory as assistant text (~8k tokens) instead of emitting a `function_call`. Pinning `exec_command` with an exact `printf` fixed it.

Native overnight Goose/OpenCode could popup or hang on the Mac desktop — keep those smokes in Docker.

## Overnight native matrix (earlier, for the record)

| CLI | Result | Note |
|-----|--------|------|
| Claude Code (`--bare`) | pass | 8s |
| Aider | pass | 12s |
| Mimo / Goose / OpenCode / Hermes / Kilo / agy / Codex / Cursor | fail or stuck | auth, MCP, Azure gravity, GUI, provider config |

That night’s publishable agent story was **2/10 installed host CLIs**. The Docker re-pass is the scorecard to cite going forward.

## Speed results (the part that actually measured)

`benchmark.py --group TC` finished. HTML/CSV: `results/bench_20260715_062244.{html,csv}`. Peak RSS ~15 GB.

| Backend | Prompt | Decode | Warm TTFT | Notes |
|---------|--------|--------|-----------|-------|
| mlx-lm 4bit | short | **20.1 t/s** | ~208 ms | baseline |
| mlx-lm 4bit | code (~751 tok) | **21.6 t/s** | ~211 ms | |
| vllm-mlx 4bit | short | **22.9 t/s** | ~366 ms | fastest decode here |
| oMLX 4bit ssd-paged | short | **20.6 t/s** | ~509 ms | |
| mlx-lm 4bit | context-32k | **20.7 t/s** | ~317 ms warm | cold TTFT ~172 s |
| mlx-lm 4bit | context-64k | **18.0 t/s** | ~475 ms warm | cold TTFT ~397 s |
| mlx-lm 4bit **think** | short | 19.0 t/s | **~48 s TTFT** | thinking tax is latency |

On M3 Max 64GB: ThinkingCap-27B-4bit is a ~20–23 tok/s local coding model with usable 64k warm latency after the first cold prefill. “Think” adds ~50s before the first visible token on a short prompt.

## Quality harnesses that night

Orchestrator reported `ok=7 failed=16`. Agent smoke + speed were real. Quality percentages are **not** publishable ThinkingCap IQ from that window: the speed bench spun its own `mlx_lm.server`, Kevlar/shim degraded mid-eval, BFCL data was not staged, HumanEval hit a 3h wall with no usable completions, context-bench saw 502s to `:8091`.

## What is actually new

1. **One ThinkingCap MLX process** serving Anthropic-shaped and OpenAI-shaped agent CLIs (including Codex Responses SSE).
2. **Artifact-gated** agent scoring (file on disk), not vibes.
3. Documented **CLI failure taxonomy** and the Docker mitigation for GUI/auth noise.
4. Side-by-side decode numbers for ThinkingCap across mlx-lm / vllm-mlx / oMLX plus 32k/64k and think-mode TTFT on the same night.

## If I re-ran tomorrow

1. Keep Kevlar/shim up for the whole agent matrix; run the speed group after, or on a second machine.
2. Stage BFCL/HumanEval data before quality.
3. Prefer `docker compose … --profile cli` for agent CLIs.
4. Default Claude-family CLIs to `--bare`.
5. Cap agent wall time; silent Goose for 45 minutes is just heat.

## Numbers to paste if you only take one table

**Docker agent microbench (local ThinkingCap):** Claude ✓ · Aider ✓ · OpenCode ✓ · Goose ✓ · Hermes ✓ · Codex ✓ · 4 skip  

**Decode (mlx-lm 4bit, warm):** ~20–22 t/s · **vllm-mlx:** ~23 t/s · **64k warm TTFT:** ~0.5 s · **think short TTFT:** ~48 s  

**Peak RSS:** ~15 GB on 64 GB unified memory.

---

Raw trail: `results/agent_bench/docker_cli_smoke/{REPORT.txt,summary.json}`, `results/agent_bench/overnight/summary.json`, `results/bench_20260715_062244.html`.

Repo WIP: https://github.com/AlexHiesch/apple-silicon-llm-bench/pull/1
