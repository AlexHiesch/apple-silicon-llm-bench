#!/usr/bin/env python3
"""
Overnight AFK runner: quality harnesses + agent CLIs against ThinkingCap
via Kevlar (:8080 Anthropic) and OpenAI shim (:8091).

Usage (from llm-bench, under caffeinate preferred):
  PYTHONPATH=. .venv/bin/python -u agent_bench/run_overnight_full.py
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "agent_bench" / "overnight"
WORK = RESULTS / "workspaces"
LOGS = RESULTS / "logs"
VENV = ROOT / ".venv" / "bin" / "python"
CLAUDE_SETTINGS = Path.home() / ".claude" / "settings.qwen36.json"

MODEL = "t-prazak/ThinkingCap-Qwen3.6-27B-MLX-4bit"
KEVLAR_BASE = "http://127.0.0.1:8080"
KEVLAR_STATUS = f"{KEVLAR_BASE}/v1/status"
SHIM_HOST = "127.0.0.1"
SHIM_PORT = 8091
SHIM_BASE = f"http://{SHIM_HOST}:{SHIM_PORT}/v1"
QUALITY_API = f"{SHIM_BASE}/chat/completions"

STUCK_SEC = 45 * 60
STEP_TIMEOUT = 3 * 60 * 60
POLL_SEC = 2

_shim_proc: subprocess.Popen | None = None

QUALITY_STEPS = {
    "eval_bfcl", "eval_bfcl_prompting", "eval_humaneval", "eval_coding",
    "eval_context_degradation", "eval_context_bench", "eval_thinking_tokens",
    "eval_hellaswag", "benchmark_group_TC",
}


def ensure_dirs() -> None:
    for p in (RESULTS, WORK, LOGS):
        p.mkdir(parents=True, exist_ok=True)


def common_env() -> dict:
    env = os.environ.copy()
    env.update({
        "OPENAI_BASE_URL": SHIM_BASE,
        "OPENAI_API_BASE": SHIM_BASE,
        "OPENAI_API_KEY": "local",
        "ANTHROPIC_BASE_URL": KEVLAR_BASE,
        "ANTHROPIC_API_KEY": "local",
        "CLAUDE_CODE_USE_BEDROCK": "0",
        "LLM_MODEL": MODEL,
        "MODEL": MODEL,
        "AGENT_BENCH_MODEL": MODEL,
        "GOOSE_PROVIDER": "openai",
        "GOOSE_MODEL": MODEL,
        "OPENAI_HOST": SHIM_BASE,
        "OPENAI_BASE_PATH": "chat/completions",
        "OPENAI_TIMEOUT": "600",
        "CI": "1",
        "NO_COLOR": "1",
    })
    return env


def kevlar_ready() -> bool:
    try:
        with urllib.request.urlopen(KEVLAR_STATUS, timeout=20) as r:
            data = json.loads(r.read().decode())
            model = data.get("model") or ""
            return (
                data.get("status") == "ok"
                and data.get("model_loaded")
                and ("ThinkingCap" in model or "thinkingcap" in model.lower())
            )
    except Exception:
        pass
    try:
        import socket
        with socket.create_connection(("127.0.0.1", 8080), timeout=2):
            pass
        out = subprocess.run(["pgrep", "-f", "kevlar serve"], capture_output=True)
        return out.returncode == 0
    except Exception:
        return False


def wait_kevlar(timeout: int = 3600) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if kevlar_ready():
            print(f"[ok] Kevlar ThinkingCap ready at {KEVLAR_BASE}", flush=True)
            return True
        print("[wait] Kevlar ...", flush=True)
        time.sleep(15)
    return False


def shim_ready() -> bool:
    try:
        with urllib.request.urlopen(f"http://{SHIM_HOST}:{SHIM_PORT}/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def start_shim() -> bool:
    global _shim_proc
    if shim_ready():
        print(f"[ok] OpenAI shim already on :{SHIM_PORT}", flush=True)
        return True

    log_path = LOGS / "openai_shim.log"
    cmd = [
        str(VENV), "-m", "agent_bench.openai_anthropic_shim",
        "--port", str(SHIM_PORT),
        "--upstream", KEVLAR_BASE,
        "--model", MODEL,
    ]
    with log_path.open("w") as log:
        _shim_proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    t0 = time.time()
    while time.time() - t0 < 60:
        if shim_ready():
            print(f"[ok] OpenAI shim started on :{SHIM_PORT} (pid {_shim_proc.pid})", flush=True)
            return True
        if _shim_proc.poll() is not None:
            print("[fatal] shim exited early", flush=True)
            return False
        time.sleep(1)
    return False


def stop_shim() -> None:
    global _shim_proc
    if _shim_proc and _shim_proc.poll() is None:
        _shim_proc.terminate()
        try:
            _shim_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _shim_proc.kill()
    _shim_proc = None


def run_step(
    name: str,
    cmd: list[str],
    cwd: Path | None = None,
    timeout: int = STEP_TIMEOUT,
    artifact: Path | None = None,
) -> dict:
    ensure_dirs()
    log_path = LOGS / f"{name}.log"
    print(f"\n{'='*70}\n[{name}] START: {' '.join(cmd)}\n{'='*70}", flush=True)
    t0 = time.time()
    status = "ok"
    err = ""
    with log_path.open("w") as log:
        log.write(f"$ {' '.join(cmd)}\n\n")
        log.flush()
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd or ROOT),
                env=common_env(),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            last_size = 0
            last_progress = time.time()
            rc = None
            while True:
                rc = proc.poll()
                now = time.time()
                size = log_path.stat().st_size if log_path.exists() else 0
                if size > last_size:
                    last_size = size
                    last_progress = now
                if rc is not None:
                    break
                if now - t0 > timeout:
                    os.killpg(proc.pid, signal.SIGKILL)
                    status = "timeout"
                    err = f"hard timeout {timeout}s"
                    break
                if now - last_progress > STUCK_SEC:
                    os.killpg(proc.pid, signal.SIGKILL)
                    status = "stuck"
                    err = f"no log progress for {STUCK_SEC}s"
                    break
                time.sleep(POLL_SEC)
            else:
                pass
            if status not in ("timeout", "stuck"):
                if rc == 0:
                    if artifact is not None and not artifact.exists():
                        status = "no_artifact"
                        err = f"missing {artifact.name}"
                    else:
                        status = "ok"
                else:
                    status = f"exit_{rc}"
        except Exception as e:
            status = "error"
            err = str(e)

    elapsed = round(time.time() - t0, 1)
    result = {
        "name": name,
        "cmd": cmd,
        "status": status,
        "elapsed_s": elapsed,
        "error": err,
        "log": str(log_path),
    }
    if artifact is not None:
        result["artifact"] = str(artifact)
        result["artifact_exists"] = artifact.exists()
    print(f"[{name}] DONE status={status} elapsed={elapsed}s", flush=True)
    (RESULTS / "steps.jsonl").open("a").write(json.dumps(result) + "\n")
    return result


def smoke_chat(name: str = "api_smoke") -> dict:
    ensure_dirs()
    script = RESULTS / "_smoke_chat.py"
    script.write_text(
        f"""
import json, urllib.request
model = "{MODEL}"
# Anthropic direct (Kevlar)
req = urllib.request.Request(
    "{KEVLAR_BASE}/v1/messages",
    data=json.dumps({{"model": model, "max_tokens": 32,
        "messages": [{{"role":"user","content":"Reply with exactly: pong"}}]}}).encode(),
    headers={{"Content-Type":"application/json","x-api-key":"local","anthropic-version":"2023-06-01"}},
)
with urllib.request.urlopen(req, timeout=180) as r:
    data = json.load(r)
text = "".join(b.get("text","") for b in data.get("content",[]) if b.get("type")=="text")
print("anthropic:", text[:120])
assert "pong" in text.lower(), text
# OpenAI via shim
req2 = urllib.request.Request(
    "{QUALITY_API}",
    data=json.dumps({{"model": model, "messages": [{{"role":"user","content":"Reply with exactly: pong"}}],
        "max_tokens": 32, "temperature": 0}}).encode(),
    headers={{"Content-Type":"application/json","Authorization":"Bearer local"}},
)
with urllib.request.urlopen(req2, timeout=180) as r:
    oai = json.load(r)
content = oai["choices"][0]["message"]["content"]
print("shim_openai:", repr(content))
assert content and "pong" in content.lower(), content
print("PASS")
"""
    )
    return run_step(name, [str(VENV), str(script)], timeout=300)


def quality_suite() -> list[dict]:
    out: list[dict] = []
    api = QUALITY_API
    wrappers = [
        ("eval_bfcl", f"import eval_bfcl as e; e.API_BASE='{api}'; e.main()"),
        ("eval_bfcl_prompting", f"import eval_bfcl_prompting as e; e.API_BASE='{api}'; e.main()"),
        ("eval_humaneval", f"import eval_humaneval as e; e.API_BASE='{api}'; e.main()"),
        ("eval_coding", f"import eval_coding as e; e.API_BASE='{api}'; e.MODEL='{MODEL}'; e.main()"),
        ("eval_context_degradation", f"import eval_context_degradation as e; e.API_BASE='{api}'; e.MODEL='{MODEL}'; e.main()"),
        ("eval_context_bench", f"import eval_context_bench as e; e.API_BASE='{api}'; e.MODEL='{MODEL}'; e.main()"),
    ]
    for name, code in wrappers:
        out.append(run_step(name, [str(VENV), "-c", code], timeout=STEP_TIMEOUT))

    thinking_code = f"""
import eval_thinking_tokens as e
e.API_BASE = '{api}'
e.wait_for_server = lambda timeout=300: True
e.start_server = lambda model_path: None
e.main()
"""
    out.append(run_step("eval_thinking_tokens", [str(VENV), "-c", thinking_code], timeout=STEP_TIMEOUT))

    out.append({
        "name": "eval_hellaswag",
        "cmd": [],
        "status": "skipped_ram",
        "elapsed_s": 0,
        "error": "direct MLX load would evict Kevlar; skipped overnight",
        "log": "",
    })
    print("[eval_hellaswag] SKIP (would load 27B into RAM alongside Kevlar)", flush=True)
    return out


def agent_cli_suite() -> list[dict]:
    out: list[dict] = []
    prompt = (
        "In this empty project, create a file named hello_tc.py that prints "
        "'ThinkingCap-OK' and nothing else. Then exit."
    )
    agents: list[tuple[str, list[str]]] = []

    if shutil.which("claude") and CLAUDE_SETTINGS.exists():
        # --bare: skip MCP/plugins (40k+ tool tokens starve local 27B prefill)
        agents.append(("claude-code", [
            "claude", "-p", prompt,
            "--settings", str(CLAUDE_SETTINGS),
            "--dangerously-skip-permissions",
            "--bare",
            "--max-turns", "8",
        ]))

    if shutil.which("opencode"):
        agents.append(("opencode", [
            "opencode", "run", prompt,
            "--model", "local/thinkingcap",
        ]))

    if shutil.which("mimo"):
        agents.append(("mimo-code", [
            "mimo", "run", prompt,
            "--model", "local/thinkingcap",
        ]))

    if shutil.which("aider"):
        agents.append(("aider", [
            "aider",
            "--model", f"openai/{MODEL}",
            "--openai-api-base", SHIM_BASE,
            "--openai-api-key", "local",
            "--yes", "--no-git", "--no-stream",
            "--message", prompt,
        ]))

    if shutil.which("goose"):
        agents.append(("goose", ["goose", "run", "-t", prompt]))

    if shutil.which("hermes"):
        agents.append(("hermes", [
            "hermes", "-z", prompt, "-m", MODEL, "--provider", "openai", "--yolo",
        ]))

    if shutil.which("kilocode") or shutil.which("kilo"):
        bin_ = shutil.which("kilocode") or shutil.which("kilo")
        agents.append(("kilocode", [bin_, "run", prompt]))

    if shutil.which("agy"):
        agents.append(("antigravity", [
            "agy", "-p", prompt,
            "--settings", str(CLAUDE_SETTINGS),
            "--dangerously-skip-permissions",
        ]))

    if shutil.which("codex"):
        agents.append(("codex", [
            "codex", "exec", "--skip-git-repo-check",
            "-c", f'model="{MODEL}"',
            prompt,
        ]))

    cursor = shutil.which("cursor-agent")
    if cursor:
        agents.append(("cursor-cli", [cursor, "-p", prompt, "--model", MODEL]))

    for name, cmd in agents:
        ws = WORK / name
        if ws.exists():
            shutil.rmtree(ws, ignore_errors=True)
        ws.mkdir(parents=True)
        (ws / "README.md").write_text(f"# overnight workspace for {name}\n")
        out.append(run_step(
            f"agent_{name}", cmd, cwd=ws, timeout=STEP_TIMEOUT,
            artifact=ws / "hello_tc.py",
        ))
    return out


def pier_harbor_attempt() -> list[dict]:
    out: list[dict] = []
    if not shutil.which("pier"):
        out.append(run_step("install_pier", ["uv", "tool", "install", "datacurve-pier"], timeout=1800))
    if not shutil.which("harbor"):
        out.append(run_step("install_harbor", ["uv", "tool", "install", "harbor"], timeout=1800))
    pier = shutil.which("pier") or str(Path.home() / ".local" / "bin" / "pier")
    harbor = shutil.which("harbor") or str(Path.home() / ".local" / "bin" / "harbor")
    if Path(pier).exists():
        out.append(run_step("pier_deepswe_smoke", [
            pier, "run", "-p", "deep-swe/tasks", "--agent", "mini-swe-agent",
            "--model", MODEL, "--env", "docker", "--n-tasks", "3", "--sample-seed", "0",
        ], timeout=STEP_TIMEOUT))
    if Path(harbor).exists():
        out.append(run_step("harbor_tb2_smoke", [
            harbor, "run", "-d", "terminal-bench@2.0", "--agent", "oracle",
            "--n-concurrent", "1",
        ], timeout=STEP_TIMEOUT))
    return out


def speed_bench_tc() -> list[dict]:
    return [run_step("benchmark_group_TC", [
        str(VENV), str(ROOT / "benchmark.py"),
        "--group", "TC", "--skip-unavailable", "--no-think", "--report-html",
    ], timeout=STEP_TIMEOUT)]


def write_summary(results: list[dict]) -> None:
    path = RESULTS / "summary.json"
    path.write_text(json.dumps({
        "model": MODEL,
        "kevlar_base": KEVLAR_BASE,
        "shim_base": SHIM_BASE,
        "quality_api": QUALITY_API,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
        "ok": sum(1 for r in results if r.get("status") == "ok"),
        "failed": sum(1 for r in results if r.get("status") != "ok"),
    }, indent=2))
    print(f"\nSummary written: {path}", flush=True)


def exit_code(results: list[dict]) -> int:
    smoke_ok = any(r.get("name") == "api_smoke" and r.get("status") == "ok" for r in results)
    quality_ok = any(r.get("status") == "ok" and r.get("name") in QUALITY_STEPS for r in results)
    agent_ok = any(r.get("status") == "ok" and str(r.get("name", "")).startswith("agent_") for r in results)
    if smoke_ok and (quality_ok or agent_ok):
        return 0
    return 2


def main() -> int:
    ensure_dirs()
    results: list[dict] = []
    code = 2
    print("OVERNIGHT FULL RUN — ThinkingCap-Qwen3.6-27B-MLX-4bit", flush=True)
    print(f"kevlar={KEVLAR_BASE} shim={SHIM_BASE} quality={QUALITY_API}", flush=True)
    try:
        if not wait_kevlar():
            print("FATAL: Kevlar not ready", flush=True)
            write_summary(results)
            return 2
        if not start_shim():
            print("FATAL: OpenAI shim failed to start", flush=True)
            write_summary(results)
            return 2

        results.append(smoke_chat())
        results.extend(agent_cli_suite())
        results.extend(pier_harbor_attempt())
        results.extend(quality_suite())
        results.extend(speed_bench_tc())
        results.append(run_step("agent_bench_plan_full", [
            str(VENV), "-m", "agent_bench", "run", "--profile", "full", "--plan-only",
        ], timeout=300))
        code = exit_code(results)
    except Exception as e:
        print(f"FATAL orchestrator error: {e}", flush=True)
        results.append({
            "name": "orchestrator",
            "cmd": [],
            "status": "error",
            "elapsed_s": 0,
            "error": str(e),
            "log": "",
        })
        code = 2
    finally:
        write_summary(results)

    failed = [r for r in results if r.get("status") != "ok"]
    print(f"\nDONE ok={len(results)-len(failed)} failed={len(failed)} exit={code}", flush=True)
    for r in failed:
        print(f"  FAIL {r['name']}: {r['status']} {r.get('error')}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
