"""Pier wrapper for DeepSWE (AA Coding Agent Index)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
RESULTS = REPO / "results" / "agent_bench"
MAP_PATH = ROOT / "agent_harbor_map.yaml"
DEFAULT_TASKS = REPO / "results" / "agent_bench" / "datasets" / "deep-swe" / "tasks"

HOST_SHIM = os.environ.get("PIER_OPENAI_BASE", "http://host.docker.internal:8091/v1")
HOST_KEVLAR = os.environ.get("PIER_ANTHROPIC_BASE", "http://host.docker.internal:8080")
MODEL = os.environ.get(
    "LLM_MODEL",
    "t-prazak/ThinkingCap-Qwen3.6-27B-MLX-4bit",
)


def load_map() -> dict:
    with MAP_PATH.open() as f:
        return yaml.safe_load(f) or {}


def pier_agent_name(agent_id: str) -> str | None:
    mapped = load_map().get("pier", {}).get(agent_id)
    return mapped or None


def ensure_pier() -> str:
    path = shutil.which("pier") or str(Path.home() / ".local" / "bin" / "pier")
    if not Path(path).exists():
        raise FileNotFoundError(
            "pier not found. Install: uv tool install --python 3.12 datacurve-pier"
        )
    return path


def resolve_tasks_path(explicit: Path | None = None) -> Path:
    env = os.environ.get("DEEPSWE_TASKS")
    candidates = [
        explicit,
        Path(env) if env else None,
        DEFAULT_TASKS,
        REPO / "deep-swe" / "tasks",
        Path.home() / "datasets" / "deep-swe" / "tasks",
    ]
    for c in candidates:
        if c and c.is_dir():
            return c
    raise FileNotFoundError(
        "DeepSWE tasks not found. Clone: git clone https://github.com/datacurve-ai/deep-swe "
        f"into {DEFAULT_TASKS.parent}"
    )


def agent_env_flags(model: str) -> list[str]:
    pairs = {
        "OPENAI_API_KEY": "local",
        "OPENAI_BASE_URL": HOST_SHIM,
        "OPENAI_API_BASE": HOST_SHIM,
        "ANTHROPIC_API_KEY": "local",
        "ANTHROPIC_BASE_URL": HOST_KEVLAR,
        "LLM_MODEL": model,
        "MODEL": model,
    }
    flags: list[str] = []
    for k, v in pairs.items():
        flags.extend(["--ae", f"{k}={v}"])
    return flags


def run_suite(
    *,
    agent_id: str,
    model: str = MODEL,
    n_attempts: int = 3,
    n_concurrent: int = 1,
    tasks_path: Path | None = None,
    jobs_dir: Path | None = None,
    yes: bool = True,
) -> dict:
    pier_agent = pier_agent_name(agent_id)
    if not pier_agent:
        return {
            "status": "skipped",
            "reason": f"no Pier agent mapping for {agent_id}",
            "agent_id": agent_id,
            "suite": "deepswe",
        }

    pier = ensure_pier()
    tasks = resolve_tasks_path(tasks_path)
    out = jobs_dir or (RESULTS / "aa_index" / "deepswe" / agent_id)
    out.mkdir(parents=True, exist_ok=True)
    job_name = f"deepswe__{agent_id}__{time.strftime('%Y%m%d_%H%M%S')}"

    model_arg = model if "/" in model else f"openai/{model}"
    if model.startswith("t-prazak/"):
        model_arg = f"openai/{model}"

    cmd = [
        pier, "run",
        "-p", str(tasks),
        "-a", pier_agent,
        "-m", model_arg,
        "-k", str(n_attempts),
        "-n", str(n_concurrent),
        "-o", str(out),
        "--job-name", job_name,
        "--env", "docker",
        *agent_env_flags(model),
    ]
    if yes:
        cmd.append("-y")

    log_path = out / f"{job_name}.log"
    t0 = time.time()
    with log_path.open("w") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    elapsed = round(time.time() - t0, 1)
    result = {
        "status": "ok" if proc.returncode == 0 else "exit_" + str(proc.returncode),
        "agent_id": agent_id,
        "pier_agent": pier_agent,
        "suite": "deepswe",
        "tasks": str(tasks),
        "model": model_arg,
        "n_attempts": n_attempts,
        "elapsed_s": elapsed,
        "jobs_dir": str(out),
        "log": str(log_path),
        "cmd": cmd,
    }
    (out / f"{job_name}.result.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agent", required=True)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--n-attempts", type=int, default=3)
    p.add_argument("--n-concurrent", type=int, default=1)
    p.add_argument("--tasks", type=Path, default=None)
    args = p.parse_args()
    print(json.dumps(run_suite(
        agent_id=args.agent,
        model=args.model,
        n_attempts=args.n_attempts,
        n_concurrent=args.n_concurrent,
        tasks_path=args.tasks,
    ), indent=2))
