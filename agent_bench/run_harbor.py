"""Harbor wrapper for Terminal-Bench v2 and SWE-Atlas-QnA."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import yaml

from agent_bench.tech_failures import TECH_EXCEPTION_TYPES
from agent_bench.tb_task_order import tasks_sorted_by_duration

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
RESULTS = REPO / "results" / "agent_bench"
# Keep resuming a job until every trial is pass or content_fail (no tech).
MAX_TECH_RESUME_ROUNDS = int(os.environ.get("HARBOR_MAX_TECH_RESUME_ROUNDS", "50"))
MAP_PATH = ROOT / "agent_harbor_map.yaml"
CERT_BUNDLE = ROOT / "certs" / "docker-ca-bundle.pem"
HOST_GATEWAY_COMPOSE = ROOT / "docker" / "host-gateway.compose.yaml"

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
    "terminal-bench-v2-1": {
        "registry": "terminal-bench/terminal-bench-2-1",
        "local": RESULTS / "datasets" / "terminal-bench-2.1" / "terminal-bench",
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


def harbor_model_arg(model: str, harbor_agent: str | None = None) -> str:
    """Normalize ``-m`` for Harbor agents + LiteLLM ThinkingCap.

    Many Harbor agents require ``provider/model`` (opencode, hermes, mimo, …)
    and raise ValueError otherwise. Claude Code strips a provider prefix before
    calling the API, so ``openai/thinkingcap`` still sends bare ``thinkingcap``
    to LiteLLM (whose key allowlist rejects unknown model ids).
    """
    if "/" in model and not model.startswith("t-prazak/"):
        return model
    if os.environ.get("HARBOR_MODEL_AS_IS") == "1":
        return model
    bare = model
    if model.startswith("t-prazak/"):
        bare = model.split("/", 1)[-1]
        # MLX ids → LiteLLM alias used on the workstation
        if "ThinkingCap" in bare or "thinkingcap" in bare.lower():
            bare = "thinkingcap"
    if bare in {"thinkingcap", "ThinkingCap"}:
        bare = "thinkingcap"
        # Prefer openai/* so OpenAI-compatible agents wire OPENAI_BASE_URL.
        return f"openai/{bare}"
    if "/" not in bare:
        return f"openai/{bare}"
    return bare


def ensure_swe_atlas_images(ds_path: Path) -> dict:
    """Prefetch missing SWE-Atlas docker images; return presence stats."""
    if not ds_path.is_dir():
        return {"present": 0, "missing": 0, "images": []}
    images: list[str] = []
    for toml in ds_path.glob("*/task.toml"):
        for line in toml.read_text(errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("docker_image"):
                # docker_image = "ghcr.io/..."
                if '"' in line:
                    images.append(line.split('"', 2)[1])
                break
    images = sorted(set(images))
    missing: list[str] = []
    for img in images:
        chk = subprocess.run(
            ["docker", "image", "inspect", img],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if chk.returncode != 0:
            missing.append(img)
    for img in missing:
        print(f"  (images) pulling {img}", flush=True)
        subprocess.run(["docker", "pull", img], check=False)
    still = [
        img
        for img in images
        if subprocess.run(
            ["docker", "image", "inspect", img],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        != 0
    ]
    present = len(images) - len(still)
    print(
        f"  (images) swe-atlas docker images present={present}/{len(images)} "
        f"missing={len(still)}",
        flush=True,
    )
    return {"present": present, "missing": len(still), "images": images, "still": still}


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
    # Claude Code defaults to max_output=32000. With workstation vLLM at
    # MAX_MODEL_LEN=65536, long agent turns hit ContextWindowExceeded
    # (input + 32000 > 65536). Cap output so prompt+completion fits.
    max_out = os.environ.get("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "16384")
    # Corp Z8: containers must use host proxy (NOT localhost). px-proxy
    # listens on 0.0.0.0:3128 and allows docker bridges 172.17/18.
    docker_proxy = os.environ.get(
        "HARBOR_HTTP_PROXY",
        "http://host.docker.internal:3128",
    )
    no_proxy = os.environ.get(
        "HARBOR_NO_PROXY",
        "localhost,127.0.0.1,host.docker.internal,"
        "10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,"
        ".svc,.cluster.local,.corpintra.net",
    )
    pairs = {
        "OPENAI_API_KEY": openai_key,
        "OPENAI_BASE_URL": HOST_SHIM,
        "OPENAI_API_BASE": HOST_SHIM,
        "ANTHROPIC_API_KEY": anthropic_key,
        "ANTHROPIC_BASE_URL": HOST_KEVLAR,
        "CLAUDE_CODE_USE_BEDROCK": "0",
        "AWS_BEARER_TOKEN_BEDROCK": "",
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": max_out,
        # WebFetch / subagents pin Haiku+Sonnet tiers; LiteLLM keys often
        # allow only `thinkingcap`. Remap every tier to the bench model.
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        "ANTHROPIC_SMALL_FAST_MODEL": model,
        "CLAUDE_CODE_SUBAGENT_MODEL": model,
        "LLM_MODEL": model,
        "MODEL": model,
        "HTTP_PROXY": docker_proxy,
        "HTTPS_PROXY": docker_proxy,
        "http_proxy": docker_proxy,
        "https_proxy": docker_proxy,
        "NO_PROXY": no_proxy,
        "no_proxy": no_proxy,
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


def agent_kwarg_flags(harbor_agent: str) -> list[str]:
    """Agent-specific Harbor ``--ak`` flags.

    Terminus 2 takes ``api_base`` as a constructor kwarg (LiteLLM endpoint).
    mini-swe-agent reads OPENAI_BASE_URL / OPENAI_API_BASE from ``--ae`` instead.
    """
    if harbor_agent == "terminus-2":
        return ["--ak", f"api_base={HOST_SHIM}"]
    return []


def corp_ca_mount_flags() -> list[str]:
    """Bind-mount corp CA dir into the agent container.

    IMPORTANT: agent_env_flags() points SSL_CERT_FILE etc. at
    /etc/harbor-corp-ca/docker-ca-bundle.pem — without this mount curl exits 77
    ("error setting certificate file") and every trial becomes NetworkConnectionError.
    """
    if not CERT_BUNDLE.is_file():
        return []
    # Mount the directory (more reliable than a single-file bind on Docker Desktop).
    mounts = [{
        "type": "bind",
        "source": str(CERT_BUNDLE.parent.resolve()),
        "target": "/etc/harbor-corp-ca",
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


def job_has_exception_types(job: Path, types: list[str] | set[str] | None) -> bool:
    """True if any finished trial has an exception_type in *types*."""
    return count_job_exception_types(job, types) > 0


def count_job_exception_types(job: Path, types: list[str] | set[str] | None) -> int:
    """How many finished trials have an exception_type in *types*."""
    if not types:
        return 0
    want = set(types)
    n = 0
    for d in _trial_dirs(job):
        rj = d / "result.json"
        if not rj.is_file():
            continue
        try:
            r = json.loads(rj.read_text())
        except Exception:
            continue
        if trial_exception_type(r) in want:
            n += 1
    return n


def job_is_technical_junk(job: Path) -> bool:
    """Complete job with zero clean trials — only setup/network failures."""
    if not job_is_complete(job):
        return False
    if job_clean_count(job) > 0:
        return False
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
        if exc in TECH_EXCEPTION_TYPES:
            tech_n += 1
    return total > 0 and tech_n == total


def resume_until_content(
    job: Path,
    *,
    filter_error_types: list[str] | None = None,
    max_rounds: int | None = None,
    n_concurrent: int | None = None,
) -> dict:
    """Resume *job* until no filterable tech exceptions remain (or cap).

    AA Index scoring is clean-only: tech failures must not stick. Harbor's
    ``job resume --filter-error-type`` drops matching trials and re-runs them;
    we loop that until every trial is pass or content_fail.
    """
    types = list(filter_error_types or sorted(TECH_EXCEPTION_TYPES))
    limit = MAX_TECH_RESUME_ROUNDS if max_rounds is None else max_rounds
    result: dict = {"status": "ok", "mode": "resume", "job_path": str(job)}
    stagnant = 0
    prev_tech = None
    for round_i in range(1, limit + 1):
        tech_n = count_job_exception_types(job, types)
        if tech_n == 0 and job_is_complete(job):
            result["tech_resume_rounds"] = round_i - 1
            result["tech_remaining"] = 0
            result["complete"] = True
            return result
        if tech_n == 0 and not job_is_complete(job):
            # Incomplete / pending trials — one more plain resume.
            print(
                f"  (resume) incomplete job {job.name} — continuing",
                flush=True,
            )
        else:
            print(
                f"  (resume) tech×{tech_n} still open on {job.name} "
                f"— round {round_i}/{limit}",
                flush=True,
            )
        result = resume_job(
            job, filter_error_types=types, n_concurrent=n_concurrent
        )
        result["tech_resume_rounds"] = round_i
        after = count_job_exception_types(job, types)
        result["tech_remaining"] = after
        log_text = ""
        try:
            log_text = Path(result["log"]).read_text(errors="ignore")
        except Exception:
            pass
        if (
            "FileExistsError" in log_text
            or "does not match the resolved job lock" in log_text
            or "Existing trial config does not match planned job config" in log_text
        ):
            print(
                "  (resume) ERROR: lock/config mismatch — "
                "archive job or revert model_name/n_concurrent drift; "
                "aborting resume loop",
                flush=True,
            )
            result["status"] = "lock_mismatch"
            result["complete"] = False
            return result
        if after == 0 and job_is_complete(job):
            result["complete"] = True
            return result
        if prev_tech is not None and after >= prev_tech and after > 0:
            stagnant += 1
        else:
            stagnant = 0
        prev_tech = after
        # Infra may be hard-stuck (same tech count every round). Keep trying
        # but bail after several zero-progress rounds so the matrix can move.
        if stagnant >= 5:
            print(
                f"  (resume) WARN: tech count stuck at {after} for "
                f"{stagnant} rounds — leaving for a later pass",
                flush=True,
            )
            result["status"] = "tech_stagnant"
            return result
        # Failed resume with no progress on an incomplete job: don't burn
        # the full round budget on the same error.
        if (
            result.get("status", "").startswith("exit_")
            and after == 0
            and not job_is_complete(job)
        ):
            stagnant += 1
            if stagnant >= 3:
                print(
                    "  (resume) WARN: incomplete job resume failing — "
                    "leaving for a later pass",
                    flush=True,
                )
                result["status"] = "resume_failed"
                return result
    result["status"] = "tech_budget_exhausted"
    result["tech_remaining"] = count_job_exception_types(job, types)
    print(
        f"  (resume) WARN: hit {limit} tech-resume rounds "
        f"({result['tech_remaining']} tech left)",
        flush=True,
    )
    return result

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
    ds_cfg = (cfg.get("datasets") or [{}])[0]
    ds_path = Path(ds_cfg.get("path") or "")
    excludes = set(ds_cfg.get("exclude_task_names") or [])
    n_tasks = _dataset_task_count(ds_path)
    if n_tasks and excludes:
        n_tasks = max(0, n_tasks - len(excludes))
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


def trial_exception_message(result: dict) -> str:
    ei = result.get("exception_info") or {}
    if isinstance(ei, dict):
        return str(ei.get("exception_message") or "")
    return ""


def job_is_missing_ca_mount_junk(job: Path) -> bool:
    """True when trials fail because SSL_CERT_FILE points at an unmounted CA path."""
    seen = 0
    ca_fail = 0
    for d in _trial_dirs(job):
        rj = d / "result.json"
        if not rj.is_file():
            continue
        try:
            r = json.loads(rj.read_text())
        except Exception:
            continue
        seen += 1
        msg = trial_exception_message(r)
        if (
            "error setting certificate file" in msg
            or "/etc/harbor-corp-ca/" in msg
        ):
            ca_fail += 1
    return seen > 0 and ca_fail == seen


def find_resumable_job(out: Path) -> Path | None:
    """Newest incomplete Harbor job dir under agent×suite output."""
    jobs = [
        p for p in out.iterdir()
        if p.is_dir()
        and (p / "config.json").is_file()
        and not p.name.startswith("_")
    ]
    for job in sorted(jobs, key=lambda p: p.stat().st_mtime, reverse=True):
        if job_is_complete(job):
            continue
        # Old jobs set SSL_CERT_FILE without --mounts; resume would keep failing.
        if job_is_missing_ca_mount_junk(job):
            continue
        return job
    return None


def _trial_reward(result: dict) -> float | None:
    vr = result.get("verifier_result") or {}
    if isinstance(vr, dict):
        rw = (vr.get("rewards") or {}).get("reward")
        if isinstance(rw, (int, float)):
            return float(rw)
    return None


def discover_clean_task_names(out: Path) -> list[str]:
    """Task names with a finished clean trial under this agent×suite dir.

    Includes archived `_partial*` / `_broken*` / `_archived*` dirs so we can
    start a fresh Harbor job without redoing intel-usable results.

    Trials with verifier reward==1 that were later stamped AgentTimeoutError
    (timeout-multiplier requeue) still count as clean passes.
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
            reward = _trial_reward(r)
            if exc:
                if not (reward == 1.0 and exc == "AgentTimeoutError"):
                    continue
            task = r.get("task_name") or trial.name.split("__")[0]
            if task:
                names.add(task)
    return sorted(names)


def restore_timeout_marked_passes(out: Path) -> int:
    """Clear AgentTimeoutError stamps on reward==1 trials (requeue artifact)."""
    n = 0
    if not out.is_dir():
        return 0
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
            if not isinstance(ei, dict):
                continue
            if ei.get("exception_type") != "AgentTimeoutError":
                continue
            if _trial_reward(r) != 1.0:
                continue
            r["exception_info"] = None
            rj.write_text(json.dumps(r, indent=2) + "\n")
            n += 1
    return n


def trial_agent_timeout_multiplier(result: dict) -> float:
    cfg = result.get("config") or {}
    mult = cfg.get("agent_timeout_multiplier")
    if mult is None:
        return 1.0
    return float(mult)


def discover_official_clean_task_names(
    out: Path,
    *,
    max_agent_timeout_multiplier: float = 1.0,
) -> list[str]:
    """Clean-finished tasks at or below the official agent timeout multiplier."""
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
            if trial_agent_timeout_multiplier(r) > max_agent_timeout_multiplier + 1e-9:
                continue
            task = r.get("task_name") or trial.name.split("__")[0]
            if task:
                names.add(task)
    return sorted(names)


def observed_agent_durations_sec(out: Path) -> dict[str, float]:
    """Median agent wall time per task from prior trials (for ordering)."""
    from datetime import datetime

    def parse_ts(s: str | None) -> datetime | None:
        if not s:
            return None
        return datetime.fromisoformat(s.replace("Z", "+00:00"))

    buckets: dict[str, list[float]] = {}
    if not out.is_dir():
        return {}
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
            ae = r.get("agent_execution") or {}
            a = parse_ts(ae.get("started_at"))
            b = parse_ts(ae.get("finished_at"))
            if not a or not b:
                continue
            task = r.get("task_name") or trial.name.split("__")[0]
            buckets.setdefault(task, []).append((b - a).total_seconds())
    out_d: dict[str, float] = {}
    for task, vals in buckets.items():
        vals.sort()
        out_d[task] = vals[len(vals) // 2]
    return out_d


def tb_full_ordered_include_names(
    out: Path,
    dataset_root: Path,
    *,
    skip_official_done: bool = True,
    max_agent_timeout_multiplier: float = 1.0,
    credit_any_clean: bool = True,
) -> list[str]:
    """TB tasks short-first; skip already-clean (pass/content_fail) when asked.

    ``credit_any_clean`` (default True): skip any historical clean trial
    regardless of agent_timeout_multiplier — use when local mult is raised
    above TB-official 1.0 and prior passes should not be re-run.
    """
    observed = observed_agent_durations_sec(out)
    ordered = tasks_sorted_by_duration(dataset_root, observed_sec=observed)
    if not skip_official_done:
        return ordered
    if credit_any_clean:
        done = set(discover_clean_task_names(out))
    else:
        done = set(
            discover_official_clean_task_names(
                out, max_agent_timeout_multiplier=max_agent_timeout_multiplier
            )
        )
    return [t for t in ordered if t not in done]

def reclaim_root_owned_trial_dirs(job_path: Path) -> int:
    """chown root-owned crash leftovers so Harbor can rmtree incomplete trials.

    After a hard reboot, Docker-created agent/sessions/debug dirs are often
    owned by root; Harbor's resume then dies with PermissionError on 'debug'.
    Uses a privileged alpine container (no host sudo required).
    """
    job_path = Path(job_path)
    try:
        probe = subprocess.run(
            ["find", str(job_path), "-user", "root"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return 0
    paths = [ln for ln in (probe.stdout or "").splitlines() if ln.strip()]
    if not paths:
        return 0
    uid = os.getuid()
    gid = os.getgid()
    subprocess.run(
        [
            "docker", "run", "--rm",
            "-v", f"{job_path.resolve()}:/job",
            "alpine:3.20",
            "chown", "-R", f"{uid}:{gid}", "/job",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return len(paths)


# Harbor agents that raise if ``-m`` / model_name lacks ``provider/…``.
# Claude Code / OpenHands strip a prefix and must keep bare thinkingcap in
# existing trial configs (resume rejects model_name drift).
_HARBOR_AGENTS_NEED_SLASH_MODEL = frozenset({
    "aider",
    "antigravity-cli",
    "cursor-cli",
    "goose",
    "hermes",
    "mimo",
    "mini-swe-agent",
    "openclaw",
    "opencode",
    "pi",
})


def patch_job_model_names(job_path: Path, model_arg: str) -> int:
    """Rewrite bare ``thinkingcap`` → ``openai/thinkingcap`` only for slash-agents.

    Agent *env* ``LLM_MODEL`` / ``MODEL`` stay bare (LiteLLM allowlist). Never
    rewrite claude-code / openhands jobs — resume then fails with
    ``Existing trial config does not match planned job config``.
    """
    job_path = Path(job_path)
    n = 0
    for name in ("config.json", "lock.json"):
        path = job_path / name
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        changed = False
        for agent in data.get("agents") or []:
            if not isinstance(agent, dict):
                continue
            if agent.get("name") not in _HARBOR_AGENTS_NEED_SLASH_MODEL:
                continue
            cur = agent.get("model_name")
            if cur in {"thinkingcap", "ThinkingCap"} and cur != model_arg:
                agent["model_name"] = model_arg
                changed = True
                n += 1
        if changed:
            path.write_text(json.dumps(data, indent=2) + "\n")
    return n


def _patch_agent_timeout_multiplier_obj(data: object, mult: float) -> int:
    """Recursively set ``agent_timeout_multiplier`` in nested Harbor JSON."""
    n = 0
    if isinstance(data, dict):
        if "agent_timeout_multiplier" in data:
            old = data.get("agent_timeout_multiplier")
            if old != mult:
                data["agent_timeout_multiplier"] = mult
                n += 1
        for v in data.values():
            n += _patch_agent_timeout_multiplier_obj(v, mult)
    elif isinstance(data, list):
        for item in data:
            n += _patch_agent_timeout_multiplier_obj(item, mult)
    return n


def set_job_agent_timeout_multiplier(job_path: Path, mult: float) -> float | None:
    """Align job + lock + per-trial configs with Harbor's timeout multiplier.

    Harbor ``job resume`` copies multiplier from job config/lock into new
    trials; stale 1.5x entries would keep giving extra agent time after the
    runner env is reset to the official 1.0.
    """
    job_path = Path(job_path)
    prev: float | None = None
    for name in ("config.json", "lock.json"):
        path = job_path / name
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        old = data.get("agent_timeout_multiplier")
        if prev is None and isinstance(old, (int, float)):
            prev = float(old)
        _patch_agent_timeout_multiplier_obj(data, mult)
        path.write_text(json.dumps(data, indent=2) + "\n")
    for trial in job_path.iterdir():
        if not trial.is_dir() or trial.name.startswith("_"):
            continue
        for name in ("config.json", "lock.json"):
            path = trial / name
            if not path.is_file():
                continue
            try:
                data = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            _patch_agent_timeout_multiplier_obj(data, mult)
            path.write_text(json.dumps(data, indent=2) + "\n")
    return prev


def set_job_n_concurrent(job_path: Path, n_concurrent: int) -> int | None:
    """Bump Harbor job concurrency before resume.

    Harbor ``job resume`` has no ``-n`` flag. It rebuilds the job lock from
    ``config.json`` (default ``n_concurrent_trials=4``) and refuses to run if
    that disagrees with existing ``lock.json``. Both files must be patched to
    the same value — lock-only bumps (e.g. 4→8) raise ``FileExistsError``.
    """
    if n_concurrent < 1:
        raise ValueError(f"n_concurrent must be >= 1, got {n_concurrent}")
    job_path = Path(job_path)
    prev: int | None = None
    for name in ("lock.json", "config.json"):
        path = job_path / name
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        old = data.get("n_concurrent_trials")
        if prev is None and isinstance(old, int):
            prev = old
        if old == n_concurrent:
            continue
        data["n_concurrent_trials"] = n_concurrent
        path.write_text(json.dumps(data, indent=2) + "\n")
    return prev


def resume_job(
    job_path: Path,
    *,
    filter_error_types: list[str] | None = None,
    n_concurrent: int | None = None,
) -> dict:
    """Resume an interrupted Harbor job (keeps finished trials)."""
    harbor = ensure_harbor()
    job_path = Path(job_path)
    job_name = job_path.name
    out = job_path.parent
    reclaimed = reclaim_root_owned_trial_dirs(job_path)
    # SWE-Atlas / slash-model agents: fix bare thinkingcap before Harbor re-runs.
    model_patches = patch_job_model_names(
        job_path, harbor_model_arg("thinkingcap")
    )
    if model_patches:
        print(
            f"  (resume) patched model_name → openai/thinkingcap "
            f"({model_patches} agent entries)",
            flush=True,
        )
    if "swe-atlas" in str(job_path):
        ensure_swe_atlas_images(SUITE_DATASETS["swe-atlas-qna"]["local"])
    conc_prev = None
    if n_concurrent is not None:
        conc_prev = set_job_n_concurrent(job_path, n_concurrent)
        if conc_prev is not None and conc_prev != n_concurrent:
            print(
                f"  (resume) n_concurrent_trials {conc_prev} → {n_concurrent}",
                flush=True,
            )
    cmd = [harbor, "job", "resume", "-p", str(job_path)]
    for err in filter_error_types or []:
        cmd.extend(["--filter-error-type", err])

    log_path = out / f"{job_name}.resume_{time.strftime('%Y%m%d_%H%M%S')}.log"
    t0 = time.time()
    with log_path.open("w") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        if reclaimed:
            log.write(f"# reclaimed {reclaimed} root-owned paths under job\n\n")
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
        "reclaimed_root_paths": reclaimed,
        "n_concurrent": n_concurrent,
        "n_concurrent_prev": conc_prev,
        "elapsed_s": elapsed,
        "jobs_dir": str(out),
        "log": str(log_path),
        "cmd": cmd,
        "complete": job_is_complete(job_path),
    }
    (out / f"{job_name}.resume.result.json").write_text(json.dumps(result, indent=2))
    return result


def find_latest_tb_full_job(out: Path) -> Path | None:
    jobs = [
        p for p in out.iterdir()
        if p.is_dir()
        and (p / "config.json").is_file()
        and not p.name.startswith("_")
    ]
    full = []
    for job in jobs:
        try:
            cfg = json.loads((job / "config.json").read_text())
        except Exception:
            continue
        if cfg.get("tb_full_ordered"):
            full.append(job)
    if not full:
        return None
    return max(full, key=lambda p: p.stat().st_mtime)


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
    tb_full_ordered: bool = False,
    tb_skip_official_done: bool = True,
    tb_force_fresh: bool = False,
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

    dataset_root: Path | None = None
    tb_includes: list[str] = []
    if tb_full_ordered and suite_id in ("terminal-bench-v2", "terminal-bench-v2-1"):
        dataset_root = Path(SUITE_DATASETS[suite_id]["local"])
        tb_includes = tb_full_ordered_include_names(
            out,
            dataset_root,
            skip_official_done=tb_skip_official_done,
            max_agent_timeout_multiplier=agent_timeout_multiplier,
        )
        if not tb_includes and tb_skip_official_done:
            return {
                "status": "ok",
                "mode": "already_complete",
                "agent_id": agent_id,
                "suite": suite_id,
                "reason": "all tasks official-clean",
                "jobs_dir": str(out),
                "elapsed_s": 0,
            }

    if resume and tb_full_ordered and not tb_force_fresh:
        full_job = find_latest_tb_full_job(out)
        if full_job and not job_is_complete(full_job):
            print(f"  (resume) TB full-89 job {full_job.name}", flush=True)
            types = list(filter_error_types or sorted(TECH_EXCEPTION_TYPES))
            result = resume_until_content(
                full_job,
                filter_error_types=types,
                n_concurrent=n_concurrent,
            )
            result.update({
                "agent_id": agent_id,
                "harbor_agent": harbor_agent,
                "suite": suite_id,
                "dataset": ds_label,
                "n_attempts": n_attempts,
                "tb_full_ordered": True,
            })
            return result

    if resume and not tb_force_fresh:
        # Default: every known tech exception is retryable. Content-only jobs
        # (pass / content_fail) are the only ones we treat as done.
        filter_error_types = list(filter_error_types or sorted(TECH_EXCEPTION_TYPES))
        latest = find_latest_job(out)
        # Complete jobs with only content results → skip. Complete jobs that
        # still have filterable tech exceptions must be resumed (Harbor drops
        # those trials). Previously we short-circuited and never retried
        # RuntimeError-only SWE-Atlas/TB jobs.
        if (
            latest
            and job_is_complete(latest)
            and not job_has_exception_types(latest, filter_error_types)
        ):
            return {
                "status": "ok",
                "mode": "already_complete",
                "agent_id": agent_id,
                "suite": suite_id,
                "job_path": str(latest),
                "jobs_dir": str(out),
                "elapsed_s": 0,
            }
        resumable = None
        if (
            latest
            and job_is_complete(latest)
            and job_has_exception_types(latest, filter_error_types)
        ):
            resumable = latest
        else:
            resumable = find_resumable_job(out)
        if resumable:
            print(f"  (resume) Harbor job {resumable.name}", flush=True)
            types = list(filter_error_types or sorted(TECH_EXCEPTION_TYPES))
            result = resume_until_content(
                resumable,
                filter_error_types=types,
                n_concurrent=n_concurrent,
            )
            result.update({
                "agent_id": agent_id,
                "harbor_agent": harbor_agent,
                "suite": suite_id,
                "dataset": ds_label,
                "n_attempts": n_attempts,
            })
            if result.get("status") in (
                "ok",
                "tech_stagnant",
                "tech_budget_exhausted",
                "lock_mismatch",
                "resume_failed",
            ) or result.get("complete"):
                # tech_stagnant/budget/lock_mismatch: still return — overnight
                # watchdog can revisit; do not start a duplicate fresh job.
                if result.get("status") in (
                    "tech_stagnant",
                    "tech_budget_exhausted",
                    "lock_mismatch",
                    "resume_failed",
                ):
                    return result
                if result.get("tech_remaining", 0) == 0 or result.get("complete"):
                    return result
            # Common after n_attempts/config patches: lock.json mismatch.
            log_text = ""
            try:
                log_text = Path(result["log"]).read_text(errors="ignore")
            except Exception:
                pass
            if "PermissionError" in log_text and "debug" in log_text:
                print(
                    "  (resume) root-owned crash leftovers — reclaiming and retrying",
                    flush=True,
                )
                reclaim_root_owned_trial_dirs(resumable)
                result = resume_until_content(
                    resumable,
                    filter_error_types=types,
                    n_concurrent=n_concurrent,
                )
                result.update({
                    "agent_id": agent_id,
                    "harbor_agent": harbor_agent,
                    "suite": suite_id,
                    "dataset": ds_label,
                    "n_attempts": n_attempts,
                })
                if (
                    result.get("status") in (
                        "ok",
                        "tech_stagnant",
                        "tech_budget_exhausted",
                        "lock_mismatch",
                        "resume_failed",
                    )
                    or result.get("complete")
                ):
                    return result
                try:
                    log_text = Path(result["log"]).read_text(errors="ignore")
                except Exception:
                    log_text = ""
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

    model_arg = harbor_model_arg(model, harbor_agent)
    if suite_id == "swe-atlas-qna":
        conf = SUITE_DATASETS["swe-atlas-qna"]
        ensure_swe_atlas_images(Path(conf["local"]))

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
        *agent_kwarg_flags(harbor_agent),
        *corp_ca_mount_flags(),
    ]
    # Linux Docker does not define host.docker.internal by default.
    if HOST_GATEWAY_COMPOSE.is_file():
        cmd.extend(["--extra-docker-compose", str(HOST_GATEWAY_COMPOSE)])
    if agent_timeout_multiplier and agent_timeout_multiplier != 1.0:
        cmd.extend(["--agent-timeout-multiplier", str(agent_timeout_multiplier)])
    if tb_includes:
        print(
            f"  (fresh) TB full ordered: {len(tb_includes)} tasks "
            f"(short-first, official-done skipped={tb_skip_official_done})",
            flush=True,
        )
        for name in tb_includes:
            cmd.extend(["--include-task-name", name])
    elif not tb_full_ordered:
        # Legacy partial jobs: skip already-clean tasks from prior partial runs.
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
        "tb_full_ordered": bool(tb_includes),
        "tb_include_count": len(tb_includes),
    }
    (out / f"{job_name}.result.json").write_text(json.dumps(result, indent=2))
    if tb_includes and proc.returncode == 0:
        cfg_path = out / job_name / "config.json"
        if cfg_path.is_file():
            try:
                cfg = json.loads(cfg_path.read_text())
                cfg["tb_full_ordered"] = True
                cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
            except Exception:
                pass
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
    p.add_argument(
        "--tb-full-ordered",
        action="store_true",
        help="Terminal Bench: run tasks short-first via --include-task-name",
    )
    p.add_argument(
        "--tb-force-fresh",
        action="store_true",
        help="With --tb-full-ordered, start a new job instead of resuming partial",
    )
    p.add_argument(
        "--tb-redo-official-done",
        action="store_true",
        help="With --tb-full-ordered, include tasks already official-clean",
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
        tb_full_ordered=args.tb_full_ordered,
        tb_skip_official_done=not args.tb_redo_official_done,
        tb_force_fresh=args.tb_force_fresh,
    ), indent=2))
