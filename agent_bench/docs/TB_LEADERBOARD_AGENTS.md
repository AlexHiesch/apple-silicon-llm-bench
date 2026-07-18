# Terminal-Bench 2.0 leaderboard agents vs our matrix

Inventory of agents that appear on the TB 2.0 / Snorkel / tbench.ai leaderboards
(model×scaffold rows), mapped to `agent_clis.yaml` + Harbor.

Sources: [tbench.ai](https://www.tbench.ai/), [Snorkel TB 2.0](https://snorkel.ai/leaderboard/terminal-bench-2-0/),
TB paper (Claude Code, Codex CLI, Gemini CLI, OpenHands, Mini-SWE-Agent, Terminus 2).

## A — In our matrix (Harbor-mapped, `matrix: include`)

| Leaderboard scaffold | Our `id` | Harbor `--agent` |
|----------------------|----------|------------------|
| Claude Code | `claude-code` | `claude-code` |
| Codex CLI | `codex` | `codex` |
| OpenHands | `openhands` | `openhands` |
| OpenCode | `opencode` | `opencode` |
| Cursor CLI | `cursor-cli` | `cursor-cli` |
| Aider | `aider` | `aider` |
| Goose | `goose` | `goose` |
| Cline | `cline` | `cline-cli` |
| Terminus 2 | `terminus-2` | `terminus-2` |
| Mini-SWE-Agent | `mini-swe-agent` | `mini-swe-agent` |

Plus other matrix agents not typically on the public TB top rows (hermes, pi, mimo, …).

## B — Harbor-ready (supported upstream; now wired)

| Scaffold | Status |
|----------|--------|
| `terminus-2` | Added to map + matrix; needs `--ak api_base=…` (handled in `run_harbor.agent_kwarg_flags`) |
| `mini-swe-agent` | Map + matrix include; uses `OPENAI_BASE_URL` / `OPENAI_API_BASE` from `--ae` |

## C — Present in Harbor, not primary matrix

| Scaffold | Our status |
|----------|------------|
| Gemini CLI | `gemini-cli` deprecated → use `antigravity` |
| Copilot CLI | `copilot-cli` `matrix: out` |
| Qwen Code | `qwen-code` `matrix: out` |
| Kimi CLI | `kimi-cli` `matrix: out` |

## D — Closed / cloud / not runnable locally

| Name | Why out |
|------|---------|
| Devin | Cloud proprietary API |
| Jules | Google cloud, no local CLI |
| Warp / Maestro | Meta-terminal / orchestrator (`bench_enabled: false`) |
| Windsurf / Augment | IDE-only |

## TB suite version note

- Current aa-ws may finish **TB 2.0** for `claude-code`.
- With `AA_TB_REMAP_TO_21=1` + `AA_TB_LEGACY_AGENTS=claude-code`, other agents run **TB 2.1** while claude-code stays on 2.0 until its job completes.
- After that, drop `claude-code` from legacy (or clear the env) so everyone uses 2.1.
