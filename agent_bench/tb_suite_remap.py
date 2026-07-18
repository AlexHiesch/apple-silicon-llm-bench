"""Terminal-Bench suite remapping for AA Index (2.0 → 2.1).

Env:
  AA_TB_REMAP_TO_21=1
  AA_TB_LEGACY_AGENTS=claude-code   # comma-separated, stay on 2.0
"""
from __future__ import annotations

import os

TB_V2 = "terminal-bench-v2"
TB_V21 = "terminal-bench-v2-1"


def remap_enabled() -> bool:
    return os.environ.get("AA_TB_REMAP_TO_21", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def legacy_tb_agents() -> set[str]:
    raw = os.environ.get("AA_TB_LEGACY_AGENTS", "claude-code")
    return {a.strip() for a in raw.split(",") if a.strip()}


def remap_tb_suite(suite_id: str, agent_id: str) -> str:
    if suite_id != TB_V2 or not remap_enabled():
        return suite_id
    if agent_id in legacy_tb_agents():
        return TB_V2
    return TB_V21
