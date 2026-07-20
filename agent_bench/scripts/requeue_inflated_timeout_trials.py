#!/usr/bin/env python3
"""Re-queue Harbor trials that ran with a non-official agent timeout multiplier.

Official Terminal Bench / Harbor default is ``agent_timeout_multiplier=1.0``.
Trials finished as *pass* while agent wall time exceeded the task's base
``[agent] timeout_sec`` only succeeded because of the inflated multiplier
(e.g. 1.5x). Mark them ``AgentTimeoutError`` so ``harbor job resume
--filter-error-type AgentTimeoutError`` re-runs them at the patched job
multiplier.

Usage:
  python agent_bench/scripts/requeue_inflated_timeout_trials.py --apply
  python agent_bench/scripts/requeue_inflated_timeout_trials.py --job PATH --from-mult 1.5
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from agent_bench.run_harbor import set_job_agent_timeout_multiplier
from agent_bench.tech_failures import classify_result

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JOB = (
    ROOT
    / "results/agent_bench/aa_index/terminal-bench-v2/claude-code"
)


def _parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def agent_duration_sec(result: dict) -> float | None:
    ae = result.get("agent_execution") or {}
    started = _parse_ts(ae.get("started_at"))
    finished = _parse_ts(ae.get("finished_at"))
    if started and finished:
        return (finished - started).total_seconds()
    return None


def task_base_agent_timeout(dataset_root: Path, task: str) -> float | None:
    toml = dataset_root / task / "task.toml"
    if not toml.is_file():
        return None
    in_agent = False
    for line in toml.read_text().splitlines():
        line = line.strip()
        if line == "[agent]":
            in_agent = True
            continue
        if line.startswith("[") and line.endswith("]"):
            in_agent = False
        if in_agent and line.startswith("timeout_sec"):
            m = re.search(r"=\s*([0-9.]+)", line)
            if m:
                return float(m.group(1))
    return None


def trial_multiplier(result: dict) -> float | None:
    cfg = result.get("config") or {}
    mult = cfg.get("agent_timeout_multiplier")
    return float(mult) if mult is not None else None


def find_latest_job(jobs_dir: Path) -> Path | None:
    jobs = sorted(
        (
            p
            for p in jobs_dir.glob("terminal-bench-v2__claude-code__*")
            if p.is_dir() and not p.name.startswith("_")
        ),
        key=lambda p: p.stat().st_mtime,
    )
    return jobs[-1] if jobs else None


def affected_passes(
    job: Path,
    dataset_root: Path,
    *,
    from_mult: float,
    to_mult: float,
) -> list[dict]:
    rows: list[dict] = []
    for trial in sorted(job.iterdir()):
        if not trial.is_dir() or trial.name.startswith("_"):
            continue
        rj = trial / "result.json"
        if not rj.is_file():
            continue
        result = json.loads(rj.read_text())
        mult = trial_multiplier(result)
        if mult is None or abs(mult - from_mult) > 1e-9:
            continue
        if classify_result(result) != "pass":
            continue
        task = result.get("task_name") or trial.name.split("__")[0]
        base = task_base_agent_timeout(dataset_root, task)
        dur = agent_duration_sec(result)
        if base is None or dur is None:
            continue
        # Invalid official pass if wall time exceeds base×to_mult cap.
        if dur > base * to_mult + 1.0:
            rows.append(
                {
                    "task": task,
                    "trial": trial.name,
                    "duration_sec": round(dur, 1),
                    "base_sec": base,
                    "ratio": round(dur / base, 3),
                    "from_mult": mult,
                }
            )
    return rows


def mark_for_retry(job: Path, trial_names: list[str], *, reason: str, apply: bool) -> int:
    if not trial_names:
        return 0
    arch = job / f"_retry_timeout_mult_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    n = 0
    for name in trial_names:
        trial = job / name
        rj = trial / "result.json"
        if not rj.is_file():
            continue
        result = json.loads(rj.read_text())
        if classify_result(result) != "pass":
            continue
        print(
            f"  {'MARK' if apply else 'DRY'} {name} "
            f"task={result.get('task_name')} reason={reason}",
        )
        if not apply:
            n += 1
            continue
        arch.mkdir(parents=True, exist_ok=True)
        if not (arch / name).exists():
            shutil.copytree(trial, arch / name)
        result["exception_info"] = {
            "exception_type": "AgentTimeoutError",
            "exception_message": reason,
            "exception_traceback": None,
            "occurred_at": datetime.now(timezone.utc).isoformat(),
        }
        rj.write_text(json.dumps(result, indent=2) + "\n")
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job", type=Path, default=None, help="Harbor job dir")
    ap.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "results/agent_bench/datasets/terminal-bench-2.0/terminal-bench",
    )
    ap.add_argument("--from-mult", type=float, default=1.5)
    ap.add_argument("--to-mult", type=float, default=1.0)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--patch-job-mult", action="store_true", default=True)
    ap.add_argument("--no-patch-job-mult", dest="patch_job_mult", action="store_false")
    args = ap.parse_args()

    jobs_dir = DEFAULT_JOB
    job = args.job or find_latest_job(jobs_dir)
    if job is None or not job.is_dir():
        raise SystemExit(f"No job dir under {jobs_dir}")

    rows = affected_passes(
        job, args.dataset, from_mult=args.from_mult, to_mult=args.to_mult
    )
    print(f"job={job.name}")
    print(f"from_mult={args.from_mult} to_mult={args.to_mult}")
    print(f"affected_passes={len(rows)}")
    for row in rows:
        print(
            f"  {row['task']:40s} dur={row['duration_sec']:7.1f}s "
            f"base={row['base_sec']:6.0f}s ratio={row['ratio']}"
        )

    if args.patch_job_mult:
        if args.apply:
            prev = set_job_agent_timeout_multiplier(job, args.to_mult)
            print(f"patched job agent_timeout_multiplier {prev} -> {args.to_mult}")
        else:
            print(f"DRY would patch job agent_timeout_multiplier -> {args.to_mult}")

    reason = (
        f"requeue_inflated_timeout: pass exceeded base×{args.to_mult} "
        f"(ran at {args.from_mult}x)"
    )
    marked = mark_for_retry(
        job, [r["trial"] for r in rows], reason=reason, apply=args.apply
    )
    print(f"marked_for_retry={marked}")


if __name__ == "__main__":
    main()
