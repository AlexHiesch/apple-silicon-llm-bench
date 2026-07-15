#!/usr/bin/env python3
"""Prerequisite detection for agent CLI benchmarks."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import yaml

from harness_model import DEFAULT_BASE_URL, DEFAULT_MODEL, agent_env

ROOT = Path(__file__).resolve().parent
AGENTS_YAML = ROOT / "agent_clis.yaml"
MODELS_YAML = ROOT / "models.yaml"


def load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def check_binary(name: str | None) -> bool:
    if not name:
        return True  # library agents (mini-swe-agent)
    return shutil.which(name) is not None


def check_docker() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


def check_model_server(base_url: str = DEFAULT_BASE_URL) -> bool:
    try:
        import urllib.request

        req = urllib.request.Request(f"{base_url.rstrip('/')}/models", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def detect_agent(agent: dict) -> dict:
    binary = agent.get("binary")
    prereqs = agent.get("prereqs") or []
    missing_env = [e for e in prereqs if e and not __import__("os").environ.get(e)]
    return {
        "id": agent["id"],
        "group": agent.get("group"),
        "matrix": agent.get("matrix", "out"),
        "bench_enabled": agent.get("bench_enabled", False),
        "harness_tier": agent.get("harness_tier"),
        "default_model": agent.get("default_model", DEFAULT_MODEL),
        "binary_ok": check_binary(binary),
        "binary": binary,
        "missing_env": missing_env,
        "ready": (
            agent.get("bench_enabled", False)
            and check_binary(binary)
            and not missing_env
        ),
        "skip_reason": agent.get("skip_reason") or agent.get("deprecated"),
    }


def list_readiness() -> dict:
    agents_cfg = load_yaml(AGENTS_YAML)
    models_cfg = load_yaml(MODELS_YAML)
    agents = [detect_agent(a) for a in agents_cfg.get("agents", [])]
    return {
        "default_model": agents_cfg.get("default_model", DEFAULT_MODEL),
        "default_base_url": agents_cfg.get("default_base_url", DEFAULT_BASE_URL),
        "models": models_cfg.get("default"),
        "docker_ok": check_docker(),
        "model_server_ok": check_model_server(),
        "agent_env_sample": agent_env(),
        "agents": agents,
        "ready_count": sum(1 for a in agents if a["ready"]),
        "enabled_count": sum(1 for a in agents if a["bench_enabled"]),
    }


def print_report(data: dict | None = None, *, matrix_only: bool = False) -> None:
    data = data or list_readiness()
    print("=" * 72)
    print("agent_bench readiness" + (" (ThinkingCap matrix)" if matrix_only else ""))
    print("=" * 72)
    print(f"default_model:  {data['default_model']}")
    print(f"default_base:   {data['default_base_url']}")
    print(f"docker:         {'OK' if data['docker_ok'] else 'MISSING'}")
    print(f"model server:   {'OK' if data['model_server_ok'] else 'DOWN'} ({data['default_base_url']})")
    agents = data["agents"]
    if matrix_only:
        agents = [a for a in agents if a.get("matrix") == "include"]
        print(f"matrix include: {len(agents)}")
    else:
        print(f"agents ready:   {data['ready_count']} / {data['enabled_count']} enabled "
              f"({len(data['agents'])} registered)")
    print()
    print(f"{'ID':<18} {'Mx':<7} {'Tier':<4} {'Bin':<6} {'Ready':<6} {'Model'}")
    print("-" * 72)
    for a in agents:
        if not matrix_only and not a["bench_enabled"]:
            continue
        bin_ok = "OK" if a["binary_ok"] else "MISS"
        ready = "YES" if a["ready"] else "no"
        model = (a["default_model"] or "").split("/")[-1][:32]
        mx = (a.get("matrix") or "-")[:7]
        print(f"{a['id']:<18} {mx:<7} {str(a.get('harness_tier') or '-'):<4} {bin_ok:<6} {ready:<6} {model}")
    print()
    print("All enabled agents default to ThinkingCap-Qwen3.6-27B-MLX-4bit.")
    print("Curated shortlist: python -m agent_bench.run_matrix --list --matrix")
    print("Start server:  python -m mlx_lm.server --model "
          f"{data['default_model']} --port 8080")


if __name__ == "__main__":
    print_report()
