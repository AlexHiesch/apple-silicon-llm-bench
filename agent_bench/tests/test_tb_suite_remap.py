"""Tests for Terminal-Bench 2.0 → 2.1 suite remapping."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent_bench.tb_suite_remap import (  # noqa: E402
    TB_V2,
    TB_V21,
    legacy_tb_agents,
    remap_enabled,
    remap_tb_suite,
)


def test_remap_off_by_default(monkeypatch):
    monkeypatch.delenv("AA_TB_REMAP_TO_21", raising=False)
    assert remap_enabled() is False
    assert remap_tb_suite(TB_V2, "codex") == TB_V2
    assert remap_tb_suite(TB_V2, "claude-code") == TB_V2


def test_remap_truthy_values(monkeypatch):
    for val in ("1", "true", "YES", "on"):
        monkeypatch.setenv("AA_TB_REMAP_TO_21", val)
        assert remap_enabled() is True


def test_remap_rewrites_non_legacy(monkeypatch):
    monkeypatch.setenv("AA_TB_REMAP_TO_21", "1")
    monkeypatch.delenv("AA_TB_LEGACY_AGENTS", raising=False)
    assert remap_tb_suite(TB_V2, "codex") == TB_V21
    assert remap_tb_suite(TB_V2, "terminus-2") == TB_V21
    assert remap_tb_suite(TB_V2, "mini-swe-agent") == TB_V21


def test_remap_keeps_legacy_claude_code(monkeypatch):
    monkeypatch.setenv("AA_TB_REMAP_TO_21", "1")
    monkeypatch.delenv("AA_TB_LEGACY_AGENTS", raising=False)
    assert legacy_tb_agents() == {"claude-code"}
    assert remap_tb_suite(TB_V2, "claude-code") == TB_V2


def test_remap_custom_legacy_list(monkeypatch):
    monkeypatch.setenv("AA_TB_REMAP_TO_21", "1")
    monkeypatch.setenv("AA_TB_LEGACY_AGENTS", "claude-code,codex")
    assert legacy_tb_agents() == {"claude-code", "codex"}
    assert remap_tb_suite(TB_V2, "codex") == TB_V2
    assert remap_tb_suite(TB_V2, "opencode") == TB_V21


def test_remap_ignores_other_suites(monkeypatch):
    monkeypatch.setenv("AA_TB_REMAP_TO_21", "1")
    assert remap_tb_suite("swe-atlas-qna", "codex") == "swe-atlas-qna"
    assert remap_tb_suite(TB_V21, "codex") == TB_V21
    assert remap_tb_suite("deepswe", "claude-code") == "deepswe"
