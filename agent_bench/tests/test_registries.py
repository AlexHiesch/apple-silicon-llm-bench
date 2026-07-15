"""Unit tests for agent_bench registries and planning (no Pier/Harbor required)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from harness_model import DEFAULT_MODEL, agent_env  # noqa: E402
from agent_bench.detect import detect_agent, list_readiness, load_yaml  # noqa: E402
from agent_bench.run_matrix import (  # noqa: E402
    plan_runs,
    resolve_model,
    select_agents,
    select_suites,
)

AGENTS = ROOT / "agent_bench" / "agent_clis.yaml"
BENCH = ROOT / "agent_bench" / "benchmarks.yaml"
MODELS = ROOT / "agent_bench" / "models.yaml"


def test_default_model_is_thinkingcap():
    assert "ThinkingCap" in DEFAULT_MODEL
    assert DEFAULT_MODEL == "t-prazak/ThinkingCap-Qwen3.6-27B-MLX-4bit"
    models = load_yaml(MODELS)
    assert models["default"]["model"] == DEFAULT_MODEL


def test_every_enabled_agent_defaults_to_thinkingcap():
    cfg = load_yaml(AGENTS)
    assert cfg["default_model"] == DEFAULT_MODEL
    enabled = [a for a in cfg["agents"] if a.get("bench_enabled")]
    assert len(enabled) >= 20
    for agent in enabled:
        assert agent.get("default_model") == DEFAULT_MODEL, agent["id"]


def test_every_benchmark_defaults_to_thinkingcap():
    cfg = load_yaml(BENCH)
    assert cfg["default_model"] == DEFAULT_MODEL
    for suite in cfg["benchmarks"]:
        assert suite.get("default_model") == DEFAULT_MODEL, suite["id"]


def test_agent_ids_unique():
    agents = load_yaml(AGENTS)["agents"]
    ids = [a["id"] for a in agents]
    assert len(ids) == len(set(ids))


def test_benchmark_ids_unique():
    suites = load_yaml(BENCH)["benchmarks"]
    ids = [s["id"] for s in suites]
    assert len(ids) == len(set(ids))


def test_profiles_reference_known_suites():
    cfg = load_yaml(BENCH)
    known = {s["id"] for s in cfg["benchmarks"]}
    for name, profile in cfg["profiles"].items():
        suites = profile.get("suites", [])
        if suites == "all":
            continue
        missing = set(suites) - known
        assert not missing, f"profile {name} unknown suites: {missing}"


def test_smoke_profile_selects_tier1():
    cfg = load_yaml(BENCH)
    suites = select_suites(cfg, "smoke", None)
    assert {s["id"] for s in suites} == {"deepswe", "terminal-bench-v2", "swe-atlas-qna"}


def test_thinkingcap_matrix_include_set():
    cfg = load_yaml(AGENTS)
    meta = cfg["thinkingcap_matrix"]
    include = set(meta["include"])
    agents = {a["id"]: a for a in cfg["agents"]}
    assert include == {a["id"] for a in cfg["agents"] if a.get("matrix") == "include"}
    for aid in include:
        assert agents[aid]["bench_enabled"] is True
        assert agents[aid]["default_model"] == DEFAULT_MODEL
    for aid in meta["skip"]:
        assert agents[aid]["matrix"] == "skip"
        assert agents[aid]["bench_enabled"] is False
    # Roo EOL / Ito out; Zoo Code is watch-only
    assert agents["roo-code"]["matrix"] == "skip"
    assert agents["ito"]["matrix"] == "skip"
    assert agents["zoocode"]["bench_enabled"] is False


def test_select_agents_matrix_only():
    agents_cfg = load_yaml(AGENTS)
    agents = select_agents(agents_cfg, None, skip_unavailable=False, matrix_only=True)
    assert {a["id"] for a in agents} == set(agents_cfg["thinkingcap_matrix"]["include"])
    assert all(a.get("matrix") == "include" for a in agents)


def test_resolve_model_override():
    assert resolve_model(None) == DEFAULT_MODEL
    assert resolve_model("other/model") == "other/model"


def test_agent_env_points_at_local_thinkingcap():
    env = agent_env()
    assert env["LLM_MODEL"] == DEFAULT_MODEL
    assert "8080" in env["OPENAI_BASE_URL"]
    assert "8080" in env["ANTHROPIC_BASE_URL"]


def test_detect_agent_shape():
    agent = {
        "id": "opencode",
        "binary": "opencode",
        "bench_enabled": True,
        "harness_tier": "A",
        "default_model": DEFAULT_MODEL,
        "prereqs": [],
    }
    result = detect_agent(agent)
    assert result["id"] == "opencode"
    assert result["default_model"] == DEFAULT_MODEL
    assert "binary_ok" in result
    assert "ready" in result


def test_list_readiness_includes_docker_flag():
    data = list_readiness()
    assert data["default_model"] == DEFAULT_MODEL
    assert "docker_ok" in data
    assert isinstance(data["docker_ok"], bool)
    assert data["enabled_count"] >= data["ready_count"]


def test_yaml_files_parse():
    for path in (AGENTS, BENCH, MODELS):
        with path.open() as f:
            assert yaml.safe_load(f) is not None
