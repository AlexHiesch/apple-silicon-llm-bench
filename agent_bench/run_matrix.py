#!/usr/bin/env python3
"""Orchestrator for agent CLI × benchmark matrix.

All harnesses default to ThinkingCap-Qwen3.6-27B-MLX-4bit (harness_model.py).
Wrappers for Pier/Harbor/Letta are stubs until installed; --list / detect work now.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from harness_model import DEFAULT_BASE_URL, DEFAULT_MODEL, agent_env

from .detect import list_readiness, print_report

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent / "results" / "agent_bench"
AGENTS_YAML = ROOT / "agent_clis.yaml"
BENCH_YAML = ROOT / "benchmarks.yaml"


def load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def resolve_model(cli_model: str | None) -> str:
    return cli_model or DEFAULT_MODEL


def select_agents(cfg: dict, ids: list[str] | None, skip_unavailable: bool) -> list[dict]:
    agents = [a for a in cfg.get("agents", []) if a.get("bench_enabled", False)]
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
    # default: smoke suite ids
    smoke = bench_cfg.get("profiles", {}).get("smoke", {})
    idset = set(smoke.get("suites", []))
    return [s for s in suites if s["id"] in idset] or suites[:3]


def plan_runs(agents: list[dict], suites: list[dict], model: str) -> list[dict]:
    runs = []
    for agent in agents:
        for suite in suites:
            runs.append({
                "agent_id": agent["id"],
                "suite": suite["id"],
                "harness": suite.get("harness"),
                "model": agent.get("default_model") or suite.get("default_model") or model,
                "base_url": DEFAULT_BASE_URL,
                "status": "planned",
            })
    return runs


def write_plan(runs: list[dict], model: str) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = RESULTS / f"plan_{stamp}.json"
    payload = {
        "timestamp": stamp,
        "default_model": model,
        "agent_env": agent_env(),
        "note": "Harness wrappers not yet executed — plan only. "
                "All runs target ThinkingCap-Qwen3.6-27B-MLX-4bit.",
        "runs": runs,
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="agent_bench matrix orchestrator")
    parser.add_argument("--list", action="store_true", help="Show agents + readiness")
    parser.add_argument("--agent", nargs="+", metavar="ID", help="Agent IDs to run")
    parser.add_argument("--suite", metavar="ID", help="Single benchmark suite")
    parser.add_argument("--profile", choices=["smoke", "coding-core", "aa-index",
                                              "extended", "full"],
                        help="Run profile (default: smoke)")
    parser.add_argument("--model", help=f"Override model (default: {DEFAULT_MODEL})")
    parser.add_argument("--skip-unavailable", action="store_true")
    parser.add_argument("--plan-only", action="store_true",
                        help="Emit run plan JSON without invoking Pier/Harbor")
    args = parser.parse_args(argv)

    if args.list:
        print_report()
        return 0

    model = resolve_model(args.model)
    agents_cfg = load_yaml(AGENTS_YAML)
    bench_cfg = load_yaml(BENCH_YAML)
    agents = select_agents(agents_cfg, args.agent, args.skip_unavailable)
    suites = select_suites(bench_cfg, args.profile or "smoke", args.suite)

    if not agents:
        print("No agents selected (check --agent / --skip-unavailable).")
        return 1
    if not suites:
        print("No suites selected.")
        return 1

    runs = plan_runs(agents, suites, model)
    print(f"Model (all harnesses): {model}")
    print(f"Agents ({len(agents)}): {', '.join(a['id'] for a in agents)}")
    print(f"Suites ({len(suites)}): {', '.join(s['id'] for s in suites)}")
    print(f"Planned runs: {len(runs)}")

    plan_path = write_plan(runs, model)
    print(f"Plan written: {plan_path}")

    if args.plan_only or True:  # wrappers land in follow-up PRs
        print("\nPier/Harbor/Letta wrappers not invoked yet (--plan-only).")
        print("Next: implement run_pier.py / run_harbor.py / run_letta.py.")
        print(f"Ensure ThinkingCap is served at {DEFAULT_BASE_URL}:")
        print(f"  python -m mlx_lm.server --model {model} --port 8080")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
