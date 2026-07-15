"""Harbor wrapper for Terminal-Bench v2 and SWE-Atlas-QnA."""

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

# Inside Docker containers, host ThinkingCap is reached via host.docker.internal
HOST_SHIM = os.environ.get("HARBOR_OPENAI_BASE", "http://host.docker.internal:8091/v1")
HOST_KEVLAR = os.environ.get("HARBOR_ANTHROPIC_BASE", "http://host.docker.internal:8080")
MODEL = os.environ.get(
    "LLM_MODEL",
    "t-prazak/ThinkingCap-Qwen3.6-27B-MLX-4bit",
)

SUITE_DATASETS = {
    "terminal-bench-v2": {
        "registry": "terminal-bench@2.0",
        "local": RESULTS / "datasets" / "terminal-bench-2.0" / "terminal-bench",
    },
    "swe-atlas-qna": {
        "registry": None,  # not on default Harbor registry; use Scale GitHub export
        "local": RESULTS / "datasets" / "SWE-Atlas" / "data" / "qa",
    },
}


def resolve_dataset(suite_id: str) -> tuple[list[str], str]:
    """Return (cli_flags, label) for harbor run dataset selection."""
    conf = SUITE_DATASETS.get(suite_id)
    if not conf:
        raise KeyError(suite_id)
    local = Path(conf["local"])
    # Harbor export layout may nest as <out>/<dataset>/<tasks…>
    if local.is_dir() and any(local.iterdir()):
        return ["-p", str(local)], str(local)
    # alternate: terminal-bench downloaded under …/terminal-bench-2.0 with nested name
    parent = local.parent
    if parent.is_dir():
        kids = [p for p in parent.iterdir() if p.is_dir()]
        if len(kids) == 1 and any(kids[0].iterdir()):
            return ["-p", str(kids[0])], str(kids[0])
    if conf.get("registry"):
        return ["-d", conf["registry"]], conf["registry"]
    raise FileNotFoundError(
        f"Dataset for {suite_id} not found at {local}. "
        "Clone or download first (see agent_bench/README.md)."
    )


def load_map() -> dict:
    with MAP_PATH.open() as f:
        return yaml.safe_load(f) or {}


def harbor_agent_name(agent_id: str) -> str | None:
    mapped = load_map().get("harbor", {}).get(agent_id)
    return mapped or None


def ensure_harbor() -> str:
    path = shutil.which("harbor") or str(Path.home() / ".local" / "bin" / "harbor")
    if not Path(path).exists():
        raise FileNotFoundError(
            "harbor not found. Install: uv tool install --python 3.12 harbor"
        )
    return path


def agent_env_flags(model: str) -> list[str]:
    """Pass OpenAI + Anthropic routes into the agent container."""
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
    suite_id: str,
    model: str = MODEL,
    n_attempts: int = 3,
    n_concurrent: int = 1,
    jobs_dir: Path | None = None,
    yes: bool = True,
) -> dict:
    harbor_agent = harbor_agent_name(agent_id)
    if not harbor_agent:
        return {
            "status": "skipped",
            "reason": f"no Harbor agent mapping for {agent_id}",
            "agent_id": agent_id,
            "suite": suite_id,
        }
    try:
        ds_flags, ds_label = resolve_dataset(suite_id)
    except FileNotFoundError as e:
        return {
            "status": "skipped",
            "reason": str(e),
            "agent_id": agent_id,
            "suite": suite_id,
        }
    except KeyError:
        return {
            "status": "skipped",
            "reason": f"suite {suite_id} is not a Harbor AA suite",
            "agent_id": agent_id,
            "suite": suite_id,
        }

    harbor = ensure_harbor()
    out = jobs_dir or (RESULTS / "aa_index" / suite_id / agent_id)
    out.mkdir(parents=True, exist_ok=True)
    job_name = f"{suite_id}__{agent_id}__{time.strftime('%Y%m%d_%H%M%S')}"

    # Prefer OpenAI-compatible model id for BYOK agents; Harbor prepends provider.
    model_arg = model if "/" in model and not model.startswith("t-prazak/") else f"openai/{model}"

    cmd = [
        harbor, "run",
        *ds_flags,
        "-a", harbor_agent,
        "-m", model_arg,
        "-k", str(n_attempts),
        "-n", str(n_concurrent),
        "-o", str(out),
        "--job-name", job_name,
        "--env", "docker",
        "--allow-agent-host", "host.docker.internal",
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
        "harbor_agent": harbor_agent,
        "suite": suite_id,
        "dataset": ds_label,
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
    p.add_argument("--suite", required=True, choices=list(SUITE_DATASETS))
    p.add_argument("--model", default=MODEL)
    p.add_argument("--n-attempts", type=int, default=3)
    p.add_argument("--n-concurrent", type=int, default=1)
    args = p.parse_args()
    print(json.dumps(run_suite(
        agent_id=args.agent,
        suite_id=args.suite,
        model=args.model,
        n_attempts=args.n_attempts,
        n_concurrent=args.n_concurrent,
    ), indent=2))
