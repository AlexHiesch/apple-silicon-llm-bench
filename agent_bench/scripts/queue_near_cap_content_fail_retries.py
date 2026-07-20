#!/usr/bin/env python3
"""Queue near-cap content_fails for another attempt at a higher timeout mult.

Most content_fails finish well under the agent cap (wrong answer, not time).
A few stop near the wall — those are worth one more try at AGENT_TIMEOUT_MULT.

Default set (from 1.5× TB Claude analysis):
  - query-optimize              (~95.7% of base×1.5, end_turn)
  - large-scale-text-editing    soft ~80–90%
  - adaptive-rejection-sampler  soft ~80–90%
  - torch-pipeline-parallelism  soft ~80–90%

Steps:
  1. Mark latest clean content_fail result.json as AgentTimeoutError so
     credit_any_clean no longer skips the task.
  2. Optionally wait for Harbor to go idle, then start an include-only job
     at the target multiplier (default 2.5).

Usage:
  # dry-run mark
  python agent_bench/scripts/queue_near_cap_content_fail_retries.py

  # mark + start when harbor free
  python agent_bench/scripts/queue_near_cap_content_fail_retries.py --apply --start
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results/agent_bench/aa_index/terminal-bench-v2/claude-code"

DEFAULT_TASKS = (
    "query-optimize",
    "large-scale-text-editing",
    "adaptive-rejection-sampler",
    "torch-pipeline-parallelism",
)

REASON = (
    "near-cap content_fail requeue for higher agent_timeout_multiplier "
    "(wall time was close to base×mult; more time may help)"
)


def _trial_reward(result: dict) -> float | None:
    vr = result.get("verifier_result") or {}
    rewards = vr.get("rewards") if isinstance(vr, dict) else None
    if isinstance(rewards, dict) and "reward" in rewards:
        try:
            return float(rewards["reward"])
        except (TypeError, ValueError):
            return None
    if "reward" in result:
        try:
            return float(result["reward"])
        except (TypeError, ValueError):
            return None
    return None


def _exception_type(result: dict) -> str | None:
    ei = result.get("exception_info") or {}
    if isinstance(ei, dict):
        return ei.get("exception_type")
    return ei if isinstance(ei, str) else None


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def find_latest_content_fail(out: Path, task: str) -> Path | None:
    """Newest result.json for task that is a clean content_fail (reward≠1, no exc)."""
    hits: list[Path] = []
    if not out.is_dir():
        return None
    for job in out.iterdir():
        if not job.is_dir():
            continue
        for trial in job.iterdir():
            if not trial.is_dir():
                continue
            rj = trial / "result.json"
            if not rj.is_file():
                continue
            try:
                r = json.loads(rj.read_text())
            except Exception:
                continue
            name = r.get("task_name") or trial.name.split("__")[0]
            if name != task:
                continue
            if _exception_type(r):
                continue
            reward = _trial_reward(r)
            if reward is None:
                continue
            if abs(reward - 1.0) < 1e-9:
                continue
            hits.append(rj)
    if not hits:
        return None
    return max(hits, key=_mtime)


def mark_result(rj: Path, *, apply: bool, reason: str) -> bool:
    result = json.loads(rj.read_text())
    task = result.get("task_name") or rj.parent.name.split("__")[0]
    print(f"  {'MARK' if apply else 'DRY'} {task}  {rj}")
    if not apply:
        return True
    arch = rj.parent.parent / f"_retry_near_cap_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    arch.mkdir(parents=True, exist_ok=True)
    dest = arch / rj.parent.name
    if not dest.exists():
        shutil.copytree(rj.parent, dest)
    result["exception_info"] = {
        "exception_type": "AgentTimeoutError",
        "exception_message": reason,
        "exception_traceback": None,
        "occurred_at": datetime.now(timezone.utc).isoformat(),
    }
    rj.write_text(json.dumps(result, indent=2) + "\n")
    return True


def harbor_busy() -> bool:
    try:
        out = subprocess.check_output(
            ["pgrep", "-af", "harbor (run|job)"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    lines = [
        ln
        for ln in out.splitlines()
        if "harbor" in ln and "pgrep" not in ln and "queue_near_cap" not in ln
    ]
    return bool(lines)


def wait_harbor_idle(*, timeout_sec: float, poll_sec: float = 30.0) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if not harbor_busy():
            return True
        print(f"  waiting for harbor idle… ({int(deadline - time.time())}s left)", flush=True)
        time.sleep(poll_sec)
    return not harbor_busy()


def start_include_only_job(
    tasks: list[str],
    *,
    mult: float,
    n_concurrent: int,
) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["AGENT_TIMEOUT_MULT"] = str(mult)
    env["AGENT_FORCE_RETRY_TASKS"] = ",".join(tasks)
    cmd = [
        sys.executable,
        "-m",
        "agent_bench.run_harbor",
        "--agent",
        "claude-code",
        "--suite",
        "terminal-bench-v2",
        "--n-attempts",
        "1",
        "--n-concurrent",
        str(n_concurrent),
        "--agent-timeout-multiplier",
        str(mult),
        "--tb-force-fresh",
        "--tb-include-only",
        ",".join(tasks),
    ]
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT), env=env)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Harbor jobs dir for claude-code × TB",
    )
    ap.add_argument(
        "--tasks",
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated task names to requeue",
    )
    ap.add_argument("--apply", action="store_true", help="Write AgentTimeoutError stamps")
    ap.add_argument(
        "--start",
        action="store_true",
        help="After mark, start include-only Harbor job at --mult",
    )
    ap.add_argument("--mult", type=float, default=float(os.environ.get("AGENT_TIMEOUT_MULT", "2.5")))
    ap.add_argument(
        "--n-concurrent",
        type=int,
        default=int(os.environ.get("N_CONCURRENT", "2")),
    )
    ap.add_argument(
        "--wait-sec",
        type=float,
        default=float(os.environ.get("NEAR_CAP_WAIT_SEC", "86400")),
        help="Max seconds to wait for harbor idle before --start",
    )
    ap.add_argument(
        "--no-wait",
        action="store_true",
        help="Do not wait; fail if harbor already running",
    )
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.replace(",", " ").split() if t.strip()]
    if not tasks:
        raise SystemExit("no tasks")

    print(f"out={args.out}")
    print(f"tasks={tasks}")
    marked: list[str] = []
    missing: list[str] = []
    for task in tasks:
        rj = find_latest_content_fail(args.out, task)
        if rj is None:
            print(f"  MISS {task} (no clean content_fail found)")
            missing.append(task)
            continue
        mark_result(rj, apply=args.apply, reason=REASON)
        marked.append(task)

    if missing and not marked:
        raise SystemExit("no matching content_fails to mark")

    to_run = marked or tasks
    if not args.start:
        print(
            "\nNext: re-run with --apply --start  "
            f"(or export AGENT_FORCE_RETRY_TASKS={','.join(to_run)})"
        )
        return

    if not args.apply:
        raise SystemExit("--start requires --apply")

    if args.no_wait:
        if harbor_busy():
            raise SystemExit("harbor already running; refuse --no-wait start")
    elif not wait_harbor_idle(timeout_sec=args.wait_sec):
        raise SystemExit("timed out waiting for harbor idle")

    rc = start_include_only_job(to_run, mult=args.mult, n_concurrent=args.n_concurrent)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
