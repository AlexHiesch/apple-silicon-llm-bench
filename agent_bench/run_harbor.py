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
CERT_BUNDLE = ROOT / "certs" / "docker-ca-bundle.pem"

# Inside Docker containers, host ThinkingCap is reached via host.docker.internal
HOST_SHIM = os.environ.get(
    "HARBOR_OPENAI_BASE",
    "http://host.docker.internal:8091/v1",
)
HOST_KEVLAR = os.environ.get(
    "HARBOR_ANTHROPIC_BASE",
    "http://host.docker.internal:8080",
)
MODEL = os.environ.get(
    "LLM_MODEL",
    "t-prazak/ThinkingCap-Qwen3.6-27B-MLX-4bit",
)

_BEDROCK_ENV_CLEAR = (
    "AWS_BEARER_TOKEN_BEDROCK",
    "ANTHROPIC_BEDROCK_BASE_URL",
    "CLAUDE_CODE_USE_BEDROCK",
    "AWS_PROFILE",
)


def clean_runner_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in _BEDROCK_ENV_CLEAR:
        env.pop(key, None)
    env["CLAUDE_CODE_USE_BEDROCK"] = "0"
    env["ANTHROPIC_API_KEY"] = env.get("ANTHROPIC_API_KEY") or "sk-ant-local"
    if env["ANTHROPIC_API_KEY"] == "local":
        env["ANTHROPIC_API_KEY"] = "sk-ant-local"
    return env

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
    openai_key = os.environ.get("OPENAI_API_KEY") or "local"
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY") or "sk-ant-local"
    if anthropic_key == "local":
        anthropic_key = "sk-ant-local"
    pairs = {
        "OPENAI_API_KEY": openai_key,
        "OPENAI_BASE_URL": HOST_SHIM,
        "OPENAI_API_BASE": HOST_SHIM,
        "ANTHROPIC_API_KEY": anthropic_key,
        "ANTHROPIC_BASE_URL": HOST_KEVLAR,
        "CLAUDE_CODE_USE_BEDROCK": "0",
        "AWS_BEARER_TOKEN_BEDROCK": "",
        "LLM_MODEL": model,
        "MODEL": model,
    }
    # Corporate MITM (Netskope) breaks npm/curl agent installs inside Docker
    # unless the interception CA is trusted.
    if CERT_BUNDLE.is_file():
        ca = "/etc/harbor-corp-ca/docker-ca-bundle.pem"
        pairs.update({
            "SSL_CERT_FILE": ca,
            "CURL_CA_BUNDLE": ca,
            "REQUESTS_CA_BUNDLE": ca,
            "NODE_EXTRA_CA_CERTS": ca,
            "GIT_SSL_CAINFO": ca,
        })
    flags: list[str] = []
    for k, v in pairs.items():
        flags.extend(["--ae", f"{k}={v}"])
    return flags


def corp_ca_mount_flags() -> list[str]:
    if not CERT_BUNDLE.is_file():
        return []
    mounts = [{
        "type": "bind",
        "source": str(CERT_BUNDLE.resolve()),
        "target": "/etc/harbor-corp-ca/docker-ca-bundle.pem",
        "read_only": True,
    }]
    return ["--mounts", json.dumps(mounts)]


def trial_exception_type(result: dict) -> str | None:
    ei = result.get("exception_info") or {}
    if isinstance(ei, dict):
        return ei.get("exception_type")
    return ei


def job_clean_count(job: Path) -> int:
    n = 0
    for d in _trial_dirs(job):
        rj = d / "result.json"
        if not rj.is_file():
            continue
        try:
            r = json.loads(rj.read_text())
        except Exception:
            continue
        if not trial_exception_type(r):
            n += 1
    return n


def job_is_technical_junk(job: Path) -> bool:
    """Complete job with zero clean trials — only setup/network failures."""
    if not job_is_complete(job):
        return False
    if job_clean_count(job) > 0:
        return False
    tech = {
        "NetworkConnectionError",
        "CancelledError",
        "UnknownApiError",
        "AgentTimeoutError",
        "NonZeroAgentExitCodeError",
    }
    total = 0
    tech_n = 0
    for d in _trial_dirs(job):
        rj = d / "result.json"
        if not rj.is_file():
            continue
        total += 1
        try:
            r = json.loads(rj.read_text())
        except Exception:
            continue
        exc = trial_exception_type(r)
        if exc in tech:
            tech_n += 1
    return total > 0 and tech_n == total

def _dataset_task_count(ds_path: Path) -> int:
    if not ds_path.is_dir():
        return 0
    return sum(1 for p in ds_path.iterdir() if p.is_dir() and (p / "task.toml").exists())


def _trial_dirs(job: Path) -> list[Path]:
    return [
        d for d in job.iterdir()
        if d.is_dir() and not d.name.startswith(".") and d.name not in ("artifacts",)
    ]


def job_is_complete(job: Path) -> bool:
    """True when all expected Harbor trials have result.json."""
    cfg_path = job / "config.json"
    if not cfg_path.is_file():
        return False
    cfg = json.loads(cfg_path.read_text())
    n_attempts = int(cfg.get("n_attempts") or 1)
    ds_path = Path((cfg.get("datasets") or [{}])[0].get("path") or "")
    n_tasks = _dataset_task_count(ds_path)
    trials = _trial_dirs(job)
    if any(not (d / "result.json").exists() for d in trials):
        return False
    if n_tasks and len(trials) < n_tasks * n_attempts:
        return False
    return bool(trials)


def find_latest_job(out: Path) -> Path | None:
    jobs = [
        p for p in out.iterdir()
        if p.is_dir()
        and (p / "config.json").is_file()
        and not p.name.startswith("_")
    ]
    if not jobs:
        return None
    return max(jobs, key=lambda p: p.stat().st_mtime)


def find_resumable_job(out: Path) -> Path | None:
    """Newest incomplete Harbor job dir under agent×suite output."""
    jobs = [
        p for p in out.iterdir()
        if p.is_dir()
        and (p / "config.json").is_file()
        and not p.name.startswith("_")
    ]
    for job in sorted(jobs, key=lambda p: p.stat().st_mtime, reverse=True):
        if not job_is_complete(job):
            return job
    return None


def discover_clean_task_names(out: Path) -> list[str]:
    """Task names with a finished clean trial under this agent×suite dir.

    Includes archived `_partial*` / `_broken*` dirs so we can start a fresh
    Harbor job without redoing intel-usable results after a lock mismatch.
    """
    names: set[str] = set()
    if not out.is_dir():
        return []
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
            ei = r.get("exception_info") or {}
            exc = ei.get("exception_type") if isinstance(ei, dict) else ei
            if exc:
                continue
            task = r.get("task_name") or trial.name.split("__")[0]
            if task:
                names.add(task)
    return sorted(names)

def resume_job(
    job_path: Path,
    *,
    filter_error_types: list[str] | None = None,
) -> dict:
    """Resume an interrupted Harbor job (keeps finished trials)."""
    harbor = ensure_harbor()
    job_path = Path(job_path)
    job_name = job_path.name
    out = job_path.parent
    cmd = [harbor, "job", "resume", "-p", str(job_path)]
    for err in filter_error_types or []:
        cmd.extend(["--filter-error-type", err])

    log_path = out / f"{job_name}.resume_{time.strftime('%Y%m%d_%H%M%S')}.log"
    t0 = time.time()
    with log_path.open("w") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=clean_runner_env(),
        )
    elapsed = round(time.time() - t0, 1)
    result = {
        "status": "ok" if proc.returncode == 0 else "exit_" + str(proc.returncode),
        "mode": "resume",
        "job_path": str(job_path),
        "filter_error_types": list(filter_error_types or []),
        "elapsed_s": elapsed,
        "jobs_dir": str(out),
        "log": str(log_path),
        "cmd": cmd,
        "complete": job_is_complete(job_path),
    }
    (out / f"{job_name}.resume.result.json").write_text(json.dumps(result, indent=2))
    return result


def run_suite(
    *,
    agent_id: str,
    suite_id: str,
    model: str = MODEL,
    n_attempts: int = 3,
    n_concurrent: int = 1,
    jobs_dir: Path | None = None,
    yes: bool = True,
    resume: bool = False,
    filter_error_types: list[str] | None = None,
    agent_timeout_multiplier: float = 1.0,
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

    if resume:
        latest = find_latest_job(out)
        if latest and job_is_complete(latest):
            return {
                "status": "ok",
                "mode": "already_complete",
                "agent_id": agent_id,
                "suite": suite_id,
                "job_path": str(latest),
                "jobs_dir": str(out),
                "elapsed_s": 0,
            }
        resumable = find_resumable_job(out)
        if resumable:
            print(f"  (resume) Harbor job {resumable.name}", flush=True)
            result = resume_job(
                resumable,
                filter_error_types=filter_error_types,
            )
            result.update({
                "agent_id": agent_id,
                "harbor_agent": harbor_agent,
                "suite": suite_id,
                "dataset": ds_label,
                "n_attempts": n_attempts,
            })
            if result.get("status") == "ok" or result.get("complete"):
                return result
            # Common after n_attempts/config patches: lock.json mismatch.
            log_text = ""
            try:
                log_text = Path(result["log"]).read_text(errors="ignore")
            except Exception:
                pass
            if "lock.json" in log_text or "FileExistsError" in log_text:
                print(
                    "  (resume) lock mismatch — archiving job and starting fresh "
                    "with clean-task excludes",
                    flush=True,
                )
                broken = out / f"_broken_lock_{resumable.name}"
                if not broken.exists():
                    resumable.rename(broken)
            else:
                return result
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
    if agent_timeout_multiplier and agent_timeout_multiplier != 1.0:
        cmd.extend(["--agent-timeout-multiplier", str(agent_timeout_multiplier)])
    # Skip already-clean intel-usable tasks from prior partial jobs.
    clean = discover_clean_task_names(out)
    if clean:
        print(f"  (fresh) excluding {len(clean)} clean tasks from prior runs", flush=True)
        for name in clean:
            cmd.extend(["--exclude-task-name", name])
    if yes:
        cmd.append("-y")

    log_path = out / f"{job_name}.log"
    t0 = time.time()
    with log_path.open("w") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=clean_runner_env(),
        )
    elapsed = round(time.time() - t0, 1)
    result = {
        "status": "ok" if proc.returncode == 0 else "exit_" + str(proc.returncode),
        "mode": "run",
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
    p.add_argument("--resume", action="store_true")
    p.add_argument("--agent-timeout-multiplier", type=float, default=1.0)
    p.add_argument(
        "--filter-error-type",
        action="append",
        default=[],
        help="On resume, drop+retry trials with this exception (repeatable)",
    )
    args = p.parse_args()
    print(json.dumps(run_suite(
        agent_id=args.agent,
        suite_id=args.suite,
        model=args.model,
        n_attempts=args.n_attempts,
        n_concurrent=args.n_concurrent,
        resume=args.resume,
        filter_error_types=args.filter_error_type or None,
        agent_timeout_multiplier=args.agent_timeout_multiplier,
    ), indent=2))
