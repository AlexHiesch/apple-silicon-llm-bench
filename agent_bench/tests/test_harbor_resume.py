"""Harbor incomplete-job resume helpers."""

from __future__ import annotations

import json
from pathlib import Path

from agent_bench import run_harbor


def _write_job(tmp: Path, *, n_attempts: int, n_tasks: int, n_results: int) -> Path:
    ds = tmp / "dataset"
    ds.mkdir()
    for i in range(n_tasks):
        t = ds / f"task-{i}"
        t.mkdir()
        (t / "task.toml").write_text("[task]\nname = 'x'\n")
    job = tmp / "jobs" / "job-1"
    job.mkdir(parents=True)
    (job / "config.json").write_text(
        json.dumps(
            {
                "job_name": "job-1",
                "n_attempts": n_attempts,
                "datasets": [{"path": str(ds)}],
            }
        )
    )
    for i in range(n_results):
        trial = job / f"task-{i % n_tasks}__id{i}"
        trial.mkdir()
        (trial / "result.json").write_text("{}")
    # incomplete trial without result
    if n_results < n_tasks * n_attempts:
        (job / "pending__xyz").mkdir()
    return job


def test_job_is_complete_false_when_missing_results(tmp_path: Path):
    job = _write_job(tmp_path, n_attempts=3, n_tasks=2, n_results=1)
    assert run_harbor.job_is_complete(job) is False


def test_find_resumable_job(tmp_path: Path):
    job = _write_job(tmp_path, n_attempts=3, n_tasks=2, n_results=1)
    out = job.parent
    assert run_harbor.find_resumable_job(out) == job


def test_already_complete_skips_resume(tmp_path: Path):
    ds = tmp_path / "dataset"
    ds.mkdir()
    for i in range(2):
        t = ds / f"task-{i}"
        t.mkdir()
        (t / "task.toml").write_text("[task]\n")
    job = tmp_path / "jobs" / "done"
    job.mkdir(parents=True)
    (job / "config.json").write_text(
        json.dumps({"n_attempts": 1, "datasets": [{"path": str(ds)}]})
    )
    for i in range(2):
        trial = job / f"task-{i}__a"
        trial.mkdir()
        (trial / "result.json").write_text("{}")
    assert run_harbor.job_is_complete(job) is True
    assert run_harbor.find_resumable_job(job.parent) is None
