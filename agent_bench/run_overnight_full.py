#!/usr/bin/env python3
"""
Overnight AFK runner: all quality harnesses + installed agent CLIs
pointed at local ThinkingCap-Qwen3.6-27B-MLX-4bit via Kevlar (:8080).

Usage (from llm-bench, already under caffeinate preferred):
  PYTHONPATH=. .venv/bin/python -u agent_bench/run_overnight_full.py \
    2>&1 | tee results/agent_bench/overnight_$(date +%Y%m%d_%H%M%S).log
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "agent_bench" / "overnight"
WORK = RESULTS / "workspaces"
LOGS = RESULTS / "logs"
VENV = ROOT / ".venv" / "bin" / "python"

MODEL = "t-prazak/ThinkingCap-Qwen3.6-27B-MLX-4bit"
BASE = "http://localhost:8080"
OPENAI_BASE = f"{BASE}/v1"
STATUS_URL = f"{BASE}/v1/status"
# OpenAI chat evals use mlx_lm.server (Kevlar is Anthropic-only on :8080)
QUALITY_PORT = 8090
QUALITY_API = f"http://localhost:{QUALITY_PORT}/v1/chat/completions"

STUCK_SEC = 45 * 60  # no log progress → kill step
STEP_TIMEOUT = 3 * 60 * 60  # hard cap per step

QUALITY_STEPS = {
    "eval_bfcl", "eval_bfcl_prompting", "eval_humaneval", "eval_coding",
    "eval_context_degradation", "eval_context_bench", "eval_thinking_tokens",
    "eval_hellaswag", "benchmark_group_TC",
}


def ensure_dirs():
    for p in (RESULTS, WORK, LOGS):
        p.mkdir(parents=True, exist_ok=True)


def common_env() -> dict:
    env = os.environ.copy()
    env.update({
        "OPENAI_BASE_URL": OPENAI_BASE,
        "OPENAI_API_BASE": OPENAI_BASE,
        "OPENAI_API_KEY": "local",
        "ANTHROPIC_BASE_URL": BASE,
        "ANTHROPIC_API_KEY": "local",
        "CLAUDE_CODE_USE_BEDROCK": "0",
        "LLM_MODEL": MODEL,
        "MODEL": MODEL,
        "AGENT_BENCH_MODEL": MODEL,
        # goose overnight overrides (do not use Azure)
        "GOOSE_PROVIDER": "openai",
        "GOOSE_MODEL": MODEL,
        "OPENAI_HOST": OPENAI_BASE,
        "OPENAI_BASE_PATH": "chat/completions",
        "OPENAI_TIMEOUT": "600",
        # disable interactive prompts
        "CI": "1",
        "NO_COLOR": "1",
    })
    return env


def model_ready() -> bool:
    """Kevlar exposes /v1/status (not /v1/models or OpenAI chat)."""
    try:
        import urllib.request
        with urllib.request.urlopen(STATUS_URL, timeout=5) as r:
            data = json.loads(r.read().decode())
            model = data.get("model") or ""
            return data.get("status") == "ok" and data.get("model_loaded") and (
                "ThinkingCap" in model or "thinkingcap" in model.lower()
            )
    except Exception:
        return False


def wait_model(timeout=3600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if model_ready():
            print(f"[ok] ThinkingCap ready at {BASE} (Kevlar /v1/status)", flush=True)
            try:
                import urllib.request
                body = urllib.request.urlopen(STATUS_URL, timeout=5).read().decode()
                print(f"[check] /v1/status → {body[:300]}", flush=True)
            except Exception as e:
                print(f"[warn] status check: {e}", flush=True)
            return True
        print("[wait] Kevlar model server ...", flush=True)
        time.sleep(15)
    return False


def kevlar_unload() -> bool:
    try:
        import urllib.request
        req = urllib.request.Request(f"{BASE}/v1/model/unload", method="POST")
        with urllib.request.urlopen(req, timeout=120) as r:
            print(f"[ok] Kevlar unload: {r.read().decode()[:200]}", flush=True)
            return True
    except Exception as e:
        print(f"[warn] Kevlar unload failed: {e}", flush=True)
        return False


def start_mlx_quality_server() -> bool:
    """Free RAM from Kevlar, then mlx_lm OpenAI server for quality harnesses."""
    kevlar_unload()
    subprocess.run(["pkill", "-9", "-f", "mlx_lm.server"], capture_output=True)
    time.sleep(5)
    log_path = LOGS / "mlx_quality_server.log"
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            [str(VENV), "-m", "mlx_lm.server", "--model", MODEL, "--port", str(QUALITY_PORT),
             "--chat-template-args", '{"enable_thinking":false}', "--log-level", "WARNING"],
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    t0 = time.time()
    while time.time() - t0 < 300:
        try:
            import urllib.request
            with urllib.request.urlopen(f"http://localhost:{QUALITY_PORT}/v1/models", timeout=5) as r:
                if r.status == 200:
                    print(f"[ok] mlx_lm quality server on :{QUALITY_PORT} (pid {proc.pid})", flush=True)
                    return True
        except Exception:
            pass
        if proc.poll() is not None:
            print("[fatal] mlx_lm server exited early", flush=True)
            return False
        time.sleep(3)
    return False


def stop_mlx_quality_server():
    subprocess.run(["pkill", "-9", "-f", "mlx_lm.server"], capture_output=True)
    time.sleep(3)


def run_step(name: str, cmd: list[str], cwd: Path | None = None, timeout: int = STEP_TIMEOUT) -> dict:
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
            while True:
                rc = proc.poll()
                now = time.time()
                size = log_path.stat().st_size if log_path.exists() else 0
                if size > last_size:
                    last_size = size
                    last_progress = now
                if rc is not None:
                    status = "ok" if rc == 0 else f"exit_{rc}"
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
                time.sleep(10)
        except Exception as e:
            status = "error"
            err = str(e)
    elapsed = round(time.time() - t0, 1)
    result = {"name": name, "cmd": cmd, "status": status, "elapsed_s": elapsed, "error": err, "log": str(log_path)}
    print(f"[{name}] DONE status={status} elapsed={elapsed}s", flush=True)
    (RESULTS / "steps.jsonl").open("a").write(json.dumps(result) + "\n")
    return result


def smoke_chat(name: str = "api_smoke") -> dict:
    """Verify Kevlar Anthropic /v1/messages (+ optional OpenAI if available)."""
    ensure_dirs()
    script = RESULTS / "_smoke_chat.py"
    script.write_text(
        f"""
import json, urllib.request, sys
model = "{MODEL}"
# Anthropic messages (Kevlar primary API)
req2 = urllib.request.Request(
    "{BASE}/v1/messages",
    data=json.dumps({{"model": model, "max_tokens": 32, "messages": [{{"role":"user","content":"Reply with exactly: pong"}}]}}).encode(),
    headers={{"Content-Type":"application/json","x-api-key":"local","anthropic-version":"2023-06-01"}},
)
with urllib.request.urlopen(req2, timeout=180) as r:
    data2 = json.load(r)
print("anthropic:", json.dumps(data2)[:300])
text = "".join(b.get("text","") for b in data2.get("content",[]) if b.get("type")=="text")
assert "pong" in text.lower(), f"unexpected: {{text!r}}"
# OpenAI chat (optional — Kevlar may not expose this)
try:
    req = urllib.request.Request(
        "{OPENAI_BASE}/chat/completions",
        data=json.dumps({{"model": model, "messages": [{{"role":"user","content":"Reply with exactly: pong"}}], "max_tokens": 16, "temperature": 0}}).encode(),
        headers={{"Content-Type":"application/json","Authorization":"Bearer local"}},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    print("openai:", repr(data["choices"][0]["message"]["content"]))
except Exception as e:
    print("openai_skip:", e)
print("PASS")
"""
    )
    return run_step(name, [str(VENV), str(script)], timeout=300)


def quality_suite() -> list[dict]:
    out = []
    if not start_mlx_quality_server():
        out.append({
            "name": "mlx_quality_server",
            "cmd": ["mlx_lm.server"],
            "status": "error",
            "elapsed_s": 0,
            "error": f"failed to start mlx_lm on :{QUALITY_PORT}",
            "log": str(LOGS / "mlx_quality_server.log"),
        })
        return out

    api = QUALITY_API
    server_wrappers = [
        ("eval_bfcl", f"import eval_bfcl as e; e.API_BASE='{api}'; e.main()"),
        ("eval_bfcl_prompting", f"import eval_bfcl_prompting as e; e.API_BASE='{api}'; e.main()"),
        ("eval_humaneval", f"import eval_humaneval as e; e.API_BASE='{api}'; e.main()"),
        ("eval_coding", f"import eval_coding as e; e.API_BASE='{api}'; e.MODEL='{MODEL}'; e.main()"),
        ("eval_context_degradation", f"import eval_context_degradation as e; e.API_BASE='{api}'; e.MODEL='{MODEL}'; e.main()"),
        ("eval_context_bench", f"import eval_context_bench as e; e.API_BASE='{api}'; e.MODEL='{MODEL}'; e.main()"),
    ]
    for name, code in server_wrappers:
        out.append(run_step(name, [str(VENV), "-c", code], timeout=STEP_TIMEOUT))

    stop_mlx_quality_server()
    out.append(run_step("eval_thinking_tokens", [str(VENV), "-c", "import eval_thinking_tokens as e; e.main()"], timeout=STEP_TIMEOUT))
    out.append(run_step("eval_hellaswag", [str(VENV), "-c", "import eval_hellaswag as e; e.main()"], timeout=STEP_TIMEOUT))
    return out


def agent_cli_suite() -> list[dict]:
    """Headless prompts against ThinkingCap for every installed CLI we can coerce."""
    out = []
    prompt = (
        "In this empty project, create a file named hello_tc.py that prints "
        "'ThinkingCap-OK' and nothing else. Then exit."
    )
    agents = []

    if shutil.which("claude"):
        agents.append(("claude-code", [
            "claude", "-p", prompt,
            "--dangerously-skip-permissions",
            "--model", MODEL,
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
            "--openai-api-base", OPENAI_BASE,
            "--openai-api-key", "local",
            "--yes", "--no-git",
            "--message", prompt,
        ]))

    if shutil.which("goose"):
        # Force openai local; recipe via stdin
        agents.append(("goose", [
            "goose", "run", "-t", prompt,
            # if unsupported flags fail, log captures it
        ]))

    if shutil.which("hermes"):
        agents.append(("hermes", [
            "hermes", "-z", prompt, "-m", MODEL, "--provider", "openai", "--yolo",
        ]))

    if shutil.which("kilocode") or shutil.which("kilo"):
        bin_ = shutil.which("kilocode") or shutil.which("kilo")
        agents.append(("kilocode", [bin_, "run", prompt]))

    if shutil.which("agy"):
        agents.append(("antigravity", [
            "agy", "-p", prompt, "--dangerously-skip-permissions",
            "--model", MODEL,
        ]))

    if shutil.which("codex"):
        # Codex prefers OpenAI; force OSS/local if possible
        agents.append(("codex", [
            "codex", "exec", "--skip-git-repo-check",
            "-c", f'model="{MODEL}"',
            prompt,
        ]))

    # Cursor: not logged in — try creative OpenAI/Bedrock-shaped env anyway
    cursor = shutil.which("cursor-agent")
    if cursor:
        agents.append(("cursor-cli", [
            cursor, "-p", prompt,
            "--model", MODEL,
        ]))

    for name, cmd in agents:
        ws = WORK / name
        if ws.exists():
            shutil.rmtree(ws, ignore_errors=True)
        ws.mkdir(parents=True)
        (ws / "README.md").write_text(f"# overnight workspace for {name}\n")
        out.append(run_step(f"agent_{name}", cmd, cwd=ws, timeout=STEP_TIMEOUT))
    return out


def pier_harbor_attempt() -> list[dict]:
    out = []
    # Try install Pier + Harbor for full suite later tonight
    if not shutil.which("pier"):
        out.append(run_step("install_pier", [
            "uv", "tool", "install", "datacurve-pier",
        ], timeout=1800))
    if not shutil.which("harbor"):
        out.append(run_step("install_harbor", [
            "uv", "tool", "install", "harbor",
        ], timeout=1800))
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
    # Group TC ThinkingCap speed harness if present
    return [run_step("benchmark_group_TC", [
        str(VENV), str(ROOT / "benchmark.py"),
        "--group", "TC", "--skip-unavailable", "--no-think", "--report-html",
    ], timeout=STEP_TIMEOUT)]


def write_summary(results: list[dict]):
    path = RESULTS / "summary.json"
    path.write_text(json.dumps({
        "model": MODEL,
        "base_url": OPENAI_BASE,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
        "ok": sum(1 for r in results if r["status"] == "ok"),
        "failed": sum(1 for r in results if r["status"] != "ok"),
    }, indent=2))
    print(f"\nSummary written: {path}", flush=True)


def exit_code(results: list[dict]) -> int:
    smoke_ok = any(r.get("name") == "api_smoke" and r.get("status") == "ok" for r in results)
    quality_ok = any(r.get("status") == "ok" and r.get("name") in QUALITY_STEPS for r in results)
    agent_ok = any(r.get("status") == "ok" and str(r.get("name", "")).startswith("agent_") for r in results)
    if smoke_ok and (quality_ok or agent_ok):
        return 0
    if not smoke_ok:
        return 2
    return 2


def main() -> int:
    ensure_dirs()
    results: list[dict] = []
    code = 2
    print("OVERNIGHT FULL RUN — ThinkingCap-Qwen3.6-27B-MLX-4bit", flush=True)
    print(f"model={MODEL} kevlar={BASE} quality_api={QUALITY_API}", flush=True)
    try:
        if not wait_model():
            print("FATAL: Kevlar model server not ready", flush=True)
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
