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
# Serve ThinkingCap (once)
python -m mlx_lm.server --model t-prazak/ThinkingCap-Qwen3.6-27B-MLX-4bit --port 8080

# Readiness dashboard
python -m agent_bench detect
# or
python -m agent_bench --list

# Emit smoke plan (all selected agents × Tier-1 suites @ ThinkingCap)
python -m agent_bench run --profile smoke --plan-only
python -m agent_bench run --agent opencode aider goose --suite deepswe --plan-only
```

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
