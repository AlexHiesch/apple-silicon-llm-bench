"""Shared default model for llm-bench quality evals and agent CLI harnesses.

Fair-comparison backend for agent CLI benchmarks: every BYOK / OpenAI-compatible
agent and Pier mini-swe-agent control run should point here unless overridden.
"""

from __future__ import annotations

# Downloaded MLX 4-bit ThinkingCap (HF hub cache):
#   ~/.cache/huggingface/hub/models--t-prazak--ThinkingCap-Qwen3.6-27B-MLX-4bit
DEFAULT_MODEL = "t-prazak/ThinkingCap-Qwen3.6-27B-MLX-4bit"
DEFAULT_MODEL_SHORT = "ThinkingCap-Qwen3.6-27B-MLX-4bit"
DEFAULT_MODEL_ALIAS = "thinkingcap"  # LiteLLM / gateway alias when configured

# Local OpenAI-compatible server (mlx_lm.server / Kevlar / LiteLLM)
DEFAULT_BASE_URL = "http://localhost:8080/v1"
DEFAULT_API_KEY = "local"

# Anthropic-compatible proxy (Claude Code → local ThinkingCap via Kevlar/LiteLLM)
DEFAULT_ANTHROPIC_BASE_URL = "http://localhost:8080"


def agent_env(extra: dict | None = None) -> dict[str, str]:
    """Env vars that point agent CLIs at the shared ThinkingCap backend."""
    env = {
        "OPENAI_BASE_URL": DEFAULT_BASE_URL,
        "OPENAI_API_BASE": DEFAULT_BASE_URL,
        "OPENAI_API_KEY": DEFAULT_API_KEY,
        "ANTHROPIC_BASE_URL": DEFAULT_ANTHROPIC_BASE_URL,
        "ANTHROPIC_API_KEY": DEFAULT_API_KEY,
        "LLM_MODEL": DEFAULT_MODEL,
        "MODEL": DEFAULT_MODEL,
        "AGENT_BENCH_MODEL": DEFAULT_MODEL,
    }
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env
