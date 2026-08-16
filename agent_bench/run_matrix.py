#!/usr/bin/env python3
"""Orchestrator for agent CLI × benchmark matrix.

All harnesses default to ThinkingCap-Qwen3.6-27B-MLX-4bit (harness_model.py).
AA Coding Agent Index (`--profile aa-index`) runs DeepSWE (Pier) + Terminal-Bench
v2 + SWE-Atlas-QnA (Harbor), 3 attempts each.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import yaml

from harness_model import DEFAULT_BASE_URL, DEFAULT_MODEL, agent_env

from .detect import list_readiness, print_report
from . import run_harbor, run_pier
from .tb_suite_remap import remap_tb_suite

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent / "results" / "agent_bench"
AGENTS_YAML = ROOT / "agent_clis.yaml"
BENCH_YAML = ROOT / "benchmarks.yaml"


def load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def resolve_model(cli_model: str | None) -> str:
    return cli_model or DEFAULT_MODEL


def _data_free_gib() -> float | None:
    """Free GiB on the macOS Data volume (where Docker/results live)."""
    target = Path("/System/Volumes/Data")
    if not target.exists():
        target = Path("/")
    try:
        usage = shutil.disk_usage(target)
        return usage.free / (1024**3)
    except OSError:
        return None


def _docker_prune() -> None:
    # Never use `prune -a`: that deletes unused *tagged* images, including the
    # prefetched ghcr.io/scaleapi/swe-atlas:* set. SWE-Atlas then fails every
    # trial with RuntimeError (ghcr pull / Service Unavailable) and the matrix
    # thrash-resumes tech×124 forever.
    print("  (disk) docker container/network/dangling prune (keep tagged images) …", flush=True)
    for cmd in (
        ["docker", "container", "prune", "-f"],
        ["docker", "network", "prune", "-f"],
        ["docker", "image", "prune", "-f"],  # dangling only
        ["docker", "builder", "prune", "-f"],
    ):
        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    free = _data_free_gib()
    if free is not None:
        print(f"  (disk) free after prune: {free:.1f} GiB", flush=True)


def select_agents(
    cfg: dict,
    ids: list[str] | None,
    skip_unavailable: bool,
    *,
    matrix_only: bool = False,
) -> list[dict]:
    agents = list(cfg.get("agents", []))
    if matrix_only:
        agents = [a for a in agents if a.get("matrix") == "include"]
    else:
        agents = [a for a in agents if a.get("bench_enabled", False)]
    if ids:
        want = set(ids)
        agents = [a for a in agents if a["id"] in want]
    if skip_unavailable:
        ready = {r["id"] for r in list_readiness()["agents"] if r["ready"]}
        agents = [a for a in agents if a["id"] in ready]
    return agents


def select_suites(bench_cfg: dict, profile: str | None, suite: str | None) -> list[dict]:
    suites = bench_cfg.get("benchmarks", [])
    if suite:
        return [s for s in suites if s["id"] == suite]
    if profile:
        profiles = bench_cfg.get("profiles", {})
        if profile not in profiles:
            raise SystemExit(f"Unknown profile: {profile}. "
                             f"Choose from: {', '.join(profiles)}")
        ids = profiles[profile].get("suites", [])
        if ids == "all":
            return suites
        idset = set(ids)
        return [s for s in suites if s["id"] in idset]
    smoke = bench_cfg.get("profiles", {}).get("smoke", {})
    idset = set(smoke.get("suites", []))
    return [s for s in suites if s["id"] in idset] or suites[:3]


def profile_attempts(bench_cfg: dict, profile: str | None) -> int:
    if not profile:
        return 1
    return int(bench_cfg.get("profiles", {}).get(profile, {}).get("repeats", 1))


def plan_runs(
    agents: list[dict],
    suites: list[dict],
    model: str,
    *,
    suite_major: bool = False,
) -> list[dict]:
    runs = []
    loops = (
        ((suite, agent) for suite in suites for agent in agents)
        if suite_major
        else ((suite, agent) for agent in agents for suite in suites)
    )
    for suite, agent in loops:
        requested = suite["id"]
        remapped = remap_tb_suite(requested, agent["id"])
        # When remap rewrites TB 2.0 → 2.1, pull harness/meta from the
        # remapped suite catalog entry if present; else keep requested.
        harness = suite.get("harness")
        if remapped != requested:
            harness = "harbor"
        runs.append({
            "agent_id": agent["id"],
            "suite": remapped,
            "suite_requested": requested,
            "harness": harness,
            # Orchestrator --model / DEFAULT_MODEL wins so workstation alias
            # (thinkingcap) is not overridden by agent YAML MLX ids.
            "model": model,
            "base_url": DEFAULT_BASE_URL,
            "status": "planned",
        })
    return runs


def write_plan(runs: list[dict], model: str, profile: str | None) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = RESULTS / f"plan_{stamp}.json"
    payload = {
        "timestamp": stamp,
        "profile": profile or "smoke",
        "default_model": model,
        "agent_env": agent_env(),
        "runs": runs,
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def execute_run(
    run: dict,
    *,
    n_attempts: int,
    n_concurrent: int,
    exclude_deepswe_touched: bool = False,
    resume_harbor: bool = False,
    harbor_filter_errors: list[str] | None = None,
    agent_timeout_multiplier: float = 1.0,
    tb_full_ordered: bool = False,
    tb_force_fresh: bool = False,
) -> dict:
    suite = run["suite"]
    agent_id = run["agent_id"]
    model = run["model"]
    harness = run.get("harness")
    if suite == "deepswe" or harness == "pier":
        exclude = (
            run_pier.discover_touched_task_names(agent_id)
            if exclude_deepswe_touched
            else None
        )
        if exclude:
            print(f"  (resume) excluding {len(exclude)} already-touched DeepSWE tasks", flush=True)
        return run_pier.run_suite(
            agent_id=agent_id,
            model=model,
            n_attempts=n_attempts,
            n_concurrent=n_concurrent,
            exclude_task_names=exclude,
        )
    if suite in ("terminal-bench-v2", "terminal-bench-v2-1", "swe-atlas-qna") or harness == "harbor":
        return run_harbor.run_suite(
            agent_id=agent_id,
            suite_id=suite,
            model=model,
            n_attempts=n_attempts,
            n_concurrent=n_concurrent,
            resume=resume_harbor,
            filter_error_types=harbor_filter_errors,
            agent_timeout_multiplier=agent_timeout_multiplier,
            tb_full_ordered=tb_full_ordered,
            tb_force_fresh=tb_force_fresh,
        )
    return {
        "status": "skipped",
        "reason": f"no wrapper for harness={harness} suite={suite}",
        "agent_id": agent_id,
        "suite": suite,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="agent_bench matrix orchestrator")
    parser.add_argument("--list", action="store_true", help="Show agents + readiness")
    parser.add_argument("--agent", nargs="+", metavar="ID", help="Agent IDs to run")
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Only curated ThinkingCap shortlist (agents with matrix: include)",
    )
    parser.add_argument("--suite", metavar="ID", help="Single benchmark suite")
    parser.add_argument("--profile", choices=["smoke", "coding-core", "aa-index",
                                              "extended", "full"],
                        help="Run profile (default: smoke)")
    parser.add_argument("--model", help=f"Override model (default: {DEFAULT_MODEL})")
    parser.add_argument("--skip-unavailable", action="store_true")
    parser.add_argument("--plan-only", action="store_true",
                        help="Emit run plan JSON without invoking Pier/Harbor")
    parser.add_argument("--n-concurrent", type=int, default=1,
                        help="Concurrent trials inside Pier/Harbor (default: 1 for local MLX)")
    parser.add_argument(
        "--n-attempts",
        type=int,
        default=None,
        metavar="K",
        help="Override profile repeats/attempts per task (default: from profile)",
    )
    parser.add_argument(
        "--suite-order",
        nargs="+",
        metavar="ID",
        help="Override suite execution order (subset of selected suites)",
    )
    parser.add_argument(
        "--exclude-deepswe-touched",
        action="store_true",
        help="Skip DeepSWE tasks that already have ≥1 trial result (resume)",
    )
    parser.add_argument(
        "--resume-harbor",
        action="store_true",
        help="Resume incomplete Harbor jobs instead of starting fresh",
    )
    parser.add_argument(
        "--harbor-retry-error",
        action="append",
        default=[],
        metavar="TYPE",
        help="On Harbor resume, drop+retry trials with this exception "
             "(e.g. UnknownApiError; repeatable)",
    )
    parser.add_argument(
        "--agent-timeout-multiplier",
        type=float,
        default=1.0,
        help="Harbor agent walltime multiplier for local slow MLX (default: 1.0)",
    )
    parser.add_argument(
        "--docker-prune-between",
        action="store_true",
        help="docker container/network/dangling prune between jobs (keeps tagged images)",
    )
    parser.add_argument(
        "--suite-major",
        action="store_true",
        help="Run all agents of suite A before suite B (Harbor-first overnight)",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=0,
        help="Abort if Data volume free space drops below this many GiB (0=off)",
    )
    parser.add_argument(
        "--tb-full-89",
        action="store_true",
        help="Terminal Bench: all tasks short-first (--include-task-name order)",
    )
    parser.add_argument(
        "--tb-force-fresh",
        action="store_true",
        help="With --tb-full-89, start a new ordered job (ignore partial resume)",
    )
    args = parser.parse_args(argv)

    if args.list:
        print_report(matrix_only=args.matrix)
        return 0

    model = resolve_model(args.model)
    agents_cfg = load_yaml(AGENTS_YAML)
    bench_cfg = load_yaml(BENCH_YAML)
    profile = args.profile or "smoke"
    agents = select_agents(
        agents_cfg,
        args.agent,
        args.skip_unavailable,
        matrix_only=args.matrix,
    )
    suites = select_suites(bench_cfg, profile, args.suite)
    n_attempts = args.n_attempts if args.n_attempts is not None else profile_attempts(bench_cfg, profile)

    if not agents:
        print("No agents selected (check --agent / --skip-unavailable).")
        return 1
    if not suites:
        print("No suites selected.")
        return 1

    if args.suite_order:
        selected = list(suites)
        order = {sid: i for i, sid in enumerate(args.suite_order)}
        suites = sorted(
            [s for s in selected if s["id"] in order],
            key=lambda s: order[s["id"]],
        )
        suites += [s for s in selected if s["id"] not in order]

    runs = plan_runs(agents, suites, model, suite_major=args.suite_major)
    print(f"Model (all harnesses): {model}")
    print(f"Profile: {profile} (repeats/attempts={n_attempts})")
    print(f"Agents ({len(agents)}): {', '.join(a['id'] for a in agents)}")
    print(f"Suites ({len(suites)}): {', '.join(s['id'] for s in suites)}")
    print(f"Planned runs: {len(runs)}")
    remapped = [
        r for r in runs
        if r.get("suite_requested") and r["suite_requested"] != r["suite"]
    ]
    if remapped:
        print(f"TB suite remap ({len(remapped)} runs):")
        for r in remapped:
            print(
                f"  {r['agent_id']}: {r['suite_requested']} → {r['suite']}",
                flush=True,
            )
    if args.exclude_deepswe_touched:
        print("Resume: excluding already-touched DeepSWE tasks")
    # Tech failures must never stick: default to retrying every known tech
    # exception type until trials resolve to pass or content_fail.
    from agent_bench.tech_failures import TECH_EXCEPTION_TYPES

    harbor_retry_errors = list(args.harbor_retry_error)
    if args.resume_harbor and not harbor_retry_errors:
        harbor_retry_errors = sorted(TECH_EXCEPTION_TYPES)
    if args.resume_harbor:
        print("Resume: Harbor incomplete jobs (job resume)")
        print(
            "  retry until content (tech excluded from score): "
            + ", ".join(harbor_retry_errors)
        )
    if args.agent_timeout_multiplier != 1.0:
        print(f"Harbor agent timeout multiplier: {args.agent_timeout_multiplier}x")
    tb_full_ordered = args.tb_full_89 or os.environ.get("TB_FULL_89", "").strip() in (
        "1", "true", "yes",
    )
    tb_force_fresh = args.tb_force_fresh or os.environ.get("TB_FORCE_FRESH", "").strip() in (
        "1", "true", "yes",
    )
    if tb_full_ordered:
        print("Terminal Bench: full dataset, short-first task order")
    if args.min_free_gb:
        print(f"Disk guard: abort below {args.min_free_gb} GiB free")

    plan_path = write_plan(runs, model, profile)
    print(f"Plan written: {plan_path}")

    if args.plan_only:
        print("\n(--plan-only) Pier/Harbor not invoked.")
        return 0

    print(f"\nExecuting {len(runs)} runs (n_concurrent={args.n_concurrent})…")
    print(f"Ensure ThinkingCap: Kevlar {DEFAULT_BASE_URL} + OpenAI shim :8091")
    results: list[dict] = []
    for i, run in enumerate(runs, 1):
        if args.min_free_gb:
            free_gb = _data_free_gib()
            if free_gb is not None and free_gb < args.min_free_gb:
                print(
                    f"\nABORT: only {free_gb:.1f} GiB free "
                    f"(min {args.min_free_gb}). Stopping before "
                    f"{run['agent_id']} × {run['suite']}.",
                    flush=True,
                )
                results.append({
                    "status": "aborted_disk",
                    "agent_id": run["agent_id"],
                    "suite": run["suite"],
                    "free_gib": free_gb,
                })
                break
        print(f"\n[{i}/{len(runs)}] {run['agent_id']} × {run['suite']} …", flush=True)
        try:
            result = execute_run(
                run,
                n_attempts=n_attempts,
                n_concurrent=args.n_concurrent,
                exclude_deepswe_touched=args.exclude_deepswe_touched,
                resume_harbor=args.resume_harbor,
                harbor_filter_errors=harbor_retry_errors or None,
                agent_timeout_multiplier=args.agent_timeout_multiplier,
                tb_full_ordered=tb_full_ordered,
                tb_force_fresh=tb_force_fresh,
            )
        except Exception as e:
            result = {
                "status": "error",
                "error": str(e),
                "agent_id": run["agent_id"],
                "suite": run["suite"],
            }
        results.append(result)
        print(f"  → {result.get('status')} ({result.get('elapsed_s', '?')}s)"
              f" {result.get('reason') or result.get('log') or ''}", flush=True)
        if args.docker_prune_between:
            _docker_prune()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    summary_path = RESULTS / f"aa_index_summary_{stamp}.json"
    summary = {
        "timestamp": stamp,
        "profile": profile,
        "model": model,
        "n_attempts": n_attempts,
        "plan": str(plan_path),
        "results": results,
        "ok": sum(1 for r in results if r.get("status") == "ok"),
        "skipped": sum(1 for r in results if r.get("status") == "skipped"),
        "failed": sum(1 for r in results if r.get("status") not in ("ok", "skipped")),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary: {summary_path}")
    print(f"ok={summary['ok']} skipped={summary['skipped']} failed={summary['failed']}")
    return 0 if summary["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
