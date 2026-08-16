"""Tests for TB force-retry / credit skip behaviour."""

from __future__ import annotations

import json
from pathlib import Path

from agent_bench.run_harbor import (
    discover_clean_task_names,
    parse_force_retry_tasks,
    tb_full_ordered_include_names,
)


def _write_trial(job: Path, task: str, *, reward: float, exc: str | None = None) -> None:
    trial = job / f"{task}__trial"
    trial.mkdir(parents=True)
    result = {
        "task_name": task,
        "verifier_result": {"rewards": {"reward": reward}},
        "exception_info": (
            {"exception_type": exc, "exception_message": "t", "exception_traceback": None}
            if exc
            else None
        ),
        "config": {"agent_timeout_multiplier": 1.5},
    }
    (trial / "result.json").write_text(json.dumps(result) + "\n")


def test_parse_force_retry_tasks():
    assert parse_force_retry_tasks("") == set()
    assert parse_force_retry_tasks("a, b  c") == {"a", "b", "c"}


def test_force_retry_uncredits_content_fail(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("AGENT_FORCE_RETRY_TASKS", raising=False)
    out = tmp_path / "claude-code"
    job = out / "terminal-bench-v2__claude-code__t"
    job.mkdir(parents=True)
    _write_trial(job, "query-optimize", reward=0.0)
    _write_trial(job, "hello-world", reward=1.0)

    ds = tmp_path / "ds"
    ds.mkdir()
    for name in ("query-optimize", "hello-world", "other-task"):
        d = ds / name
        d.mkdir()
        (d / "task.toml").write_text(
            "[metadata]\nexpert_time_estimate_min = 5\ndifficulty = easy\n"
            "[agent]\ntimeout_sec = 100\n"
        )

    clean = discover_clean_task_names(out)
    assert "query-optimize" in clean
    assert "hello-world" in clean

    names = tb_full_ordered_include_names(
        out,
        ds,
        force_retry={"query-optimize"},
    )
    assert "query-optimize" in names
    assert "hello-world" not in names
    assert "other-task" in names
