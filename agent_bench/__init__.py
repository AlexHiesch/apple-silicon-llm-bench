"""agent_bench — coding agent CLI benchmark suite.

Default model for all harnesses: ThinkingCap-Qwen3.6-27B-MLX-4bit
(see repo-root harness_model.py).
"""

from harness_model import (
    DEFAULT_API_KEY,
    DEFAULT_ANTHROPIC_BASE_URL,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_MODEL_ALIAS,
    DEFAULT_MODEL_SHORT,
    agent_env,
)

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_MODEL_SHORT",
    "DEFAULT_MODEL_ALIAS",
    "DEFAULT_BASE_URL",
    "DEFAULT_API_KEY",
    "DEFAULT_ANTHROPIC_BASE_URL",
    "agent_env",
]
