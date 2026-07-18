"""Harbor model-arg + safe docker prune helpers."""

from __future__ import annotations

import json
from pathlib import Path

from agent_bench.run_harbor import harbor_model_arg, patch_job_model_names
from agent_bench.run_matrix import _docker_prune
from agent_bench.tech_failures import TECH_EXCEPTION_TYPES


def test_thinkingcap_gets_openai_prefix(monkeypatch):
    monkeypatch.delenv("HARBOR_MODEL_AS_IS", raising=False)
    assert harbor_model_arg("thinkingcap") == "openai/thinkingcap"
    assert harbor_model_arg("ThinkingCap") == "openai/thinkingcap"
    assert (
        harbor_model_arg("t-prazak/ThinkingCap-Qwen3.6-27B-MLX-4bit")
        == "openai/thinkingcap"
    )


def test_explicit_provider_passthrough(monkeypatch):
    monkeypatch.delenv("HARBOR_MODEL_AS_IS", raising=False)
    assert harbor_model_arg("anthropic/claude-sonnet-4") == "anthropic/claude-sonnet-4"


def test_model_as_is(monkeypatch):
    monkeypatch.setenv("HARBOR_MODEL_AS_IS", "1")
    assert harbor_model_arg("thinkingcap") == "thinkingcap"


def test_patch_job_model_names(tmp_path: Path):
    job = tmp_path / "job"
    job.mkdir()
    cfg = {
        "agents": [
            {
                "name": "opencode",
                "model_name": "thinkingcap",
                "env": {"LLM_MODEL": "thinkingcap", "MODEL": "thinkingcap"},
            },
            {
                "name": "claude-code",
                "model_name": "thinkingcap",
                "env": {"LLM_MODEL": "thinkingcap"},
            },
        ]
    }
    (job / "config.json").write_text(json.dumps(cfg))
    (job / "lock.json").write_text(json.dumps(cfg))
    n = patch_job_model_names(job, "openai/thinkingcap")
    assert n == 2  # opencode in config + lock only
    for name in ("config.json", "lock.json"):
        data = json.loads((job / name).read_text())
        by = {a["name"]: a for a in data["agents"]}
        assert by["opencode"]["model_name"] == "openai/thinkingcap"
        assert by["claude-code"]["model_name"] == "thinkingcap"
        assert by["opencode"]["env"]["LLM_MODEL"] == "thinkingcap"


def test_valueerror_is_tech():
    assert "ValueError" in TECH_EXCEPTION_TYPES


def test_docker_prune_keeps_tagged_images(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))

        class R:
            returncode = 0

        return R()

    monkeypatch.setattr("agent_bench.run_matrix.subprocess.run", fake_run)
    monkeypatch.setattr("agent_bench.run_matrix._data_free_gib", lambda: 100.0)
    _docker_prune()
    joined = [" ".join(c) for c in calls]
    assert any("container prune" in j for j in joined)
    assert not any("prune -af" in j or "prune -a" in j for j in joined)
