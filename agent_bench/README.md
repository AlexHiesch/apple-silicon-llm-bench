# Agent CLI Benchmark Suite

Fair-compare coding agent CLIs (Claude Code, Codex, Cursor CLI, OpenCode, …)
on standardized suites (DeepSWE, Terminal-Bench, SWE-Atlas, Letta Context-Bench, …).

## Default model for all harnesses

**`t-prazak/ThinkingCap-Qwen3.6-27B-MLX-4bit`**

Every BYOK / OpenAI-compatible agent and every Pier/Harbor/Letta harness defaults
to this downloaded ThinkingCap checkpoint so harness quality is the variable,
not the underlying LLM. Override with `--model` only when intentionally changing
the comparison.

Shared config:

| File | Role |
|------|------|
| [`../harness_model.py`](../harness_model.py) | `DEFAULT_MODEL`, `agent_env()` |
| [`models.yaml`](models.yaml) | Model + base URL + routing hints |
| [`agent_clis.yaml`](agent_clis.yaml) | ~40 CLIs, each with `default_model` |
| [`benchmarks.yaml`](benchmarks.yaml) | Suites, profiles, composites |

## Quick start

```bash
# Unit tests (no GPU / no Docker required)
python -m pytest agent_bench/tests -q

# Serve ThinkingCap on host (Apple Silicon / MLX)
python -m mlx_lm.server --model t-prazak/ThinkingCap-Qwen3.6-27B-MLX-4bit --port 8080

# Readiness dashboard
python -m agent_bench detect
python -m agent_bench --list

# Emit smoke plan (agents × Tier-1 suites @ ThinkingCap)
python -m agent_bench run --profile smoke --plan-only
```

## Clean Docker setups

MLX ThinkingCap stays on the **host**. Containers talk to it via `host.docker.internal`.

```bash
# Host: Kevlar Anthropic :8080 + OpenAI shim :8091
#   kevlar serve --model t-prazak/ThinkingCap-Qwen3.6-27B-MLX-4bit --port 8080
#   python -m agent_bench.openai_anthropic_shim --port 8091 --upstream http://127.0.0.1:8080

# Agent CLI micro-smokes in Linux (does NOT touch host-native CLIs / GUI popups)
docker compose -f agent_bench/docker-compose.yml --profile cli build cli-smoke
docker compose -f agent_bench/docker-compose.yml --profile cli run --rm --user 1000:1000 cli-smoke
# → results/agent_bench/docker_cli_smoke/REPORT.txt

# Docker-only networking smoke (mock LLM, no GPU)
docker compose -f agent_bench/docker-compose.yml --profile smoke up --build --abort-on-container-exit

# Minimal sandbox curl against host OpenAI shim
docker compose -f agent_bench/docker-compose.yml --profile host run --rm sandbox
```

**Why Docker for CLIs:** native Goose/OpenCode/etc. can popup on macOS or fight your interactive sessions. The `cli` profile runs headless Linux binaries as user `bench` (uid 1000).

**Latest Docker micro-smoke scorecard** (`hello_tc.py` artifact gate → host ThinkingCap):

| Status | CLIs |
|--------|------|
| PASS (6) | Claude Code, Aider, OpenCode, Goose, Hermes, Codex |
| Host fallback (4) | Kilo, Mimo, Antigravity (`agy`), Cursor CLI — `bash agent_bench/scripts/host_skip_smoke.sh` |

**Host skip smoke** (macOS-native; Cursor needs `cursor-agent login`):

```bash
# Requires Kevlar :8080 + shim :8091; Cursor already logged in on this Mac
bash agent_bench/scripts/host_skip_smoke.sh
# → results/agent_bench/host_skip_smoke/REPORT.txt
```

Latest host result: **4/4 PASS** (Kilo/Mimo/agy against local ThinkingCap; Cursor CLI after login — uses Cursor catalog models, not ThinkingCap BYOK).

Shim notes: Codex needs `/v1/responses` SSE ending in `response.completed`. OpenCode/Kilo/Mimo need `--dir` + deny `external_directory` (model invents absolute paths). Keep Kevlar up for the whole matrix — a dead `:8080` shows up as shim `502`.

## Profiles

| Profile | Scope |
|---------|-------|
| `smoke` | Tier 1 × 3 tasks (default) |
| `coding-core` | Tier 1 + Letta FS + SWE-bench Verified (50) |
| `aa-index` | Full AA Coding Agent Index v1.1 |
| `extended` / `full` | Broader / research matrices |

## Status

Registries + detect + plan orchestrator ship first. Pier/Harbor/Letta wrappers
(`run_pier.py`, `run_harbor.py`, `run_letta.py`) land next; until then runs are
`--plan-only`.
