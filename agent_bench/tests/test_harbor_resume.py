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


def test_job_has_exception_types(tmp_path: Path):
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
    clean = job / "task-0__a"
    clean.mkdir()
    (clean / "result.json").write_text("{}")
    bad = job / "task-1__b"
    bad.mkdir()
    (bad / "result.json").write_text(
        json.dumps({"exception_info": {"exception_type": "RuntimeError"}})
    )
    assert run_harbor.job_is_complete(job) is True
    assert run_harbor.job_has_exception_types(job, ["RuntimeError"]) is True
    assert run_harbor.job_has_exception_types(job, ["AgentTimeoutError"]) is False


def test_tech_exception_types_cover_harbor_rate_limit_aliases():
    from agent_bench.tech_failures import TECH_EXCEPTION_TYPES

    # Harbor installed agents raise ApiRateLimitError; keep both names retryable.
    assert "RateLimitError" in TECH_EXCEPTION_TYPES
    assert "ApiRateLimitError" in TECH_EXCEPTION_TYPES
    assert "EnvironmentStartTimeoutError" in TECH_EXCEPTION_TYPES
    assert "VerifierTimeoutError" in TECH_EXCEPTION_TYPES


def test_count_job_exception_types(tmp_path: Path):
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
    clean = job / "task-0__a"
    clean.mkdir()
    (clean / "result.json").write_text(
        json.dumps({"verifier_result": {"rewards": {"reward": 1.0}}})
    )
    bad = job / "task-1__b"
    bad.mkdir()
    (bad / "result.json").write_text(
        json.dumps({"exception_info": {"exception_type": "ApiRateLimitError"}})
    )
    assert run_harbor.count_job_exception_types(job, ["ApiRateLimitError"]) == 1
    assert run_harbor.count_job_exception_types(job, ["AgentTimeoutError"]) == 0


def test_resume_until_content_stops_when_no_tech(tmp_path: Path, monkeypatch):
    ds = tmp_path / "dataset"
    ds.mkdir()
    t = ds / "task-0"
    t.mkdir()
    (t / "task.toml").write_text("[task]\n")
    job = tmp_path / "jobs" / "done"
    job.mkdir(parents=True)
    (job / "config.json").write_text(
        json.dumps({"n_attempts": 1, "datasets": [{"path": str(ds)}]})
    )
    trial = job / "task-0__a"
    trial.mkdir()
    (trial / "result.json").write_text(
        json.dumps({"verifier_result": {"rewards": {"reward": 0.0}}})
    )

    calls = {"n": 0}

    def fake_resume(job_path, filter_error_types=None, n_concurrent=None):
        calls["n"] += 1
        return {"status": "ok", "complete": True, "job_path": str(job_path)}

    monkeypatch.setattr(run_harbor, "resume_job", fake_resume)
    result = run_harbor.resume_until_content(
        job, filter_error_types=["ApiRateLimitError"], max_rounds=3
    )
    assert result["tech_remaining"] == 0
    assert calls["n"] == 0  # already content-only; no Harbor call


def test_set_job_n_concurrent_patches_lock(tmp_path: Path):
    job = tmp_path / "job"
    job.mkdir()
    (job / "lock.json").write_text(json.dumps({"n_concurrent_trials": 2}))
    (job / "config.json").write_text(
        json.dumps({"n_concurrent_trials": 2, "job_name": "x"})
    )
    prev = run_harbor.set_job_n_concurrent(job, 4)
    assert prev == 2
    assert json.loads((job / "lock.json").read_text())["n_concurrent_trials"] == 4
    assert json.loads((job / "config.json").read_text())["n_concurrent_trials"] == 4
    # idempotent
    assert run_harbor.set_job_n_concurrent(job, 4) == 4


def test_set_job_n_concurrent_writes_config_when_key_missing(tmp_path: Path):
    """Harbor resume rebuilds the lock from config (default n=4).

    Lock-only bumps to 8 leave config at default 4 → FileExistsError.
    """
    job = tmp_path / "job"
    job.mkdir()
    (job / "lock.json").write_text(json.dumps({"n_concurrent_trials": 4}))
    (job / "config.json").write_text(json.dumps({"job_name": "x", "agents": []}))
    prev = run_harbor.set_job_n_concurrent(job, 8)
    assert prev == 4
    assert json.loads((job / "lock.json").read_text())["n_concurrent_trials"] == 8
    assert json.loads((job / "config.json").read_text())["n_concurrent_trials"] == 8
