# Terminal-Bench 2.0 leaderboard agents vs our matrix

Inventory of unique **agent scaffolds** on
[tbench.ai TB 2.0](https://www.tbench.ai/leaderboard/terminal-bench/2.0)
(142 model×agent rows → ~40 unique harnesses), mapped to `agent_clis.yaml` + Harbor.

## A — In our matrix (`matrix: include`, Harbor-mapped)

| Leaderboard scaffold | Our `id` | Harbor `--agent` |
|----------------------|----------|------------------|
| Claude Code | `claude-code` | `claude-code` |
| Codex CLI / Simple Codex | `codex` | `codex` |
| OpenHands | `openhands` | `openhands` |
| OpenCode | `opencode` | `opencode` |
| Cursor CLI | `cursor-cli` | `cursor-cli` |
| Aider | `aider` | `aider` |
| Goose | `goose` | `goose` |
| Cline | `cline` | `cline-cli` |
| Terminus 2 | `terminus-2` | `terminus-2` |
| Mini-SWE-Agent | `mini-swe-agent` | `mini-swe-agent` |
| Gemini CLI (via Antigravity) | `antigravity` | `antigravity-cli` |

Plus other matrix agents not typically on public TB top rows (hermes, pi, mimo, …).

## B — ThinkingCap host-smoke OK; Harbor wrapper still needed

Host smokes (2026-07-18) via SSH tunnel → x40 LiteLLM `:4000` / `thinkingcap`:

| Scaffold | Install | Host smoke | Harbor |
|----------|---------|------------|--------|
| **Grok CLI** (TB: Superagent / `@vibe-kit/grok-cli`) | `npm i -g @vibe-kit/grok-cli` | **PASS** (`GROK_BASE_URL` + `GROK_API_KEY` → PONG) | no built-in; keep `grok-build` `matrix: out` until wrapper |
| **little-coder** | `npm i -g little-coder` | **PASS** (pi-based `-p` → PONG) | no built-in; add wrapper or map via `pi` carefully |
| Factory **Droid** | `npm i -g droid` / `curl …/cli \| sh` | installed; `droid exec` still wants Factory login / `FACTORY_API_KEY` | not in Harbor `AgentName` |
| **Junie CLI** | JetBrains install.sh | binary installs; custom `OpenAICompletion` model JSON TBD; hung on first task smoke | not in Harbor |
| **Letta Code** | `npm i -g @letta-ai/letta-code` | `connect openai-compatible` / headless currently tied to Letta Cloud `LETTA_API_KEY` | not in Harbor |
| **clnkr** | brew/go | install failed in smoke env | not in Harbor |
| **Mux** / **Deep Agents** | public CLIs | OpenAI-compatible possible | Harbor import-path / `langgraph` |

## C — Harbor-present, not primary matrix

| Scaffold | Our status |
|----------|------------|
| Gemini CLI | `gemini-cli` deprecated → `antigravity` |
| Copilot CLI | `copilot-cli` `matrix: out` |
| Qwen Code | `qwen-code` `matrix: out` |
| Kimi CLI | `kimi-cli` `matrix: out` |
| LangGraph / Deep Agents | Harbor `langgraph` — optional later |

## D — Closed / no public CLI (skip)

NexAU-AHE, LemonHarness, Capy, Polaris, WOZCODE, TongAgents, SageAgent, CodeBrain-1.5,
Codelia, MAYA-V2, spoox-o-m, Ante, IndusAGI, Crux, Meta-Harness / Terminus-KIRA (research
import only), II-Agent (heavy platform), hookele, Abacus AI Desktop, Simplai, CAMEL-AI
(research), Harness Agent, cchuter, Dakou, Bash Agent, Warp (needs public HTTPS URL),
Devin, Jules, Windsurf, Augment.

## TB suite version note

- Live `aa-ws` finishes **TB 2.0** for `claude-code` (remap **off** until flip).
- After that job completes:
  ```bash
  bash agent_bench/scripts/enable_tb21_after_claude.sh
  # or RESTART_AA_WS=1 bash agent_bench/scripts/enable_tb21_after_claude.sh
  ```
  Sets `AA_TB_REMAP_TO_21=1` + `AA_TB_LEGACY_AGENTS=claude-code` so **other agents → TB 2.1**.
- Dataset on x40: `results/agent_bench/datasets/terminal-bench-2.1/terminal-bench` (89 tasks).
