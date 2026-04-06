#!/usr/bin/env python3
"""
Re-run missing Gemma 4 tests:
  - K_Q4_4 / K_OLL_CTX_1  — Ollama gemma4:26b (was blocked by 0.19.0 service)
  - L_Q4_3 / L_OLL_CTX_1  — Ollama gemma4:31b
  - L_MXFP4_1              — mlx-vlm 31B mxfp4 (was "server not ready after 600s")

Root cause fix: /opt/homebrew/opt/ollama symlink updated to 0.20.0; L_MXFP4_1
gets server_timeout: 1200 in config.yaml.

Usage:
  cd ~/Projects/Work/llm-bench
  nohup caffeinate -i python3 -u run_gemma4_missing.py >> /tmp/gemma4_missing.log 2>&1 &
  echo "PID: $!"
  tail -f /tmp/gemma4_missing.log
"""
import csv, os, subprocess, sys, time, types
from pathlib import Path

BENCH = Path(__file__).parent / "benchmark.py"
sys.path.insert(0, str(Path(__file__).parent))

HARDWARE  = {"name": "Apple M3 Max", "memory_gb": 64, "platform": "darwin-arm64"}
HF_CACHE  = Path.home() / ".cache" / "huggingface" / "hub"
DONE_FILE = Path(__file__).parent / ".gemma4_missing_done"
LOG_FILE  = Path("/tmp/gemma4_missing.log")


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"\n[{ts}] {msg}", flush=True)


def check_single_instance():
    import psutil
    my_pid = os.getpid()
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmd = " ".join(proc.info['cmdline'] or [])
            if proc.info['pid'] != my_pid and (
                "run_gemma4_missing.py" in cmd or "run_gemma4_bench.py" in cmd or
                ("benchmark.py" in cmd and proc.info['pid'] != my_pid)
            ):
                print(f"  ERROR: competing process (PID {proc.info['pid']}): {cmd[:80]}")
                sys.exit(1)
        except Exception:
            pass


def ollama_pull(tag: str, max_retries: int = 3):
    """Pull an Ollama model, retrying on failure."""
    # Check if already present
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if tag.split(":")[0] in result.stdout.lower() and tag in result.stdout:
        print(f"  ✔ {tag} already in ollama list", flush=True)
        return True

    for attempt in range(1, max_retries + 1):
        log(f"Pulling {tag} (attempt {attempt}/{max_retries})")
        result = subprocess.run(["ollama", "pull", tag])
        if result.returncode == 0:
            print(f"  ✔ {tag} pulled", flush=True)
            return True
        if attempt < max_retries:
            print(f"  ⚠ Pull attempt {attempt} failed — retrying in 30s", flush=True)
            time.sleep(30)
    print(f"  ✘ All {max_retries} pull attempts failed for {tag}", flush=True)
    return False


def run_tests(*test_ids: str):
    log(f"Running: {', '.join(test_ids)}")
    subprocess.run([
        sys.executable, "-u", str(BENCH),
        "--test", *test_ids,
        "--skip-unavailable", "--no-think", "--report-html",
    ])


def build_combined_html():
    log("Rebuilding combined HTML report")
    import benchmark as bm
    results = []
    for csv_path in sorted((BENCH.parent / "results").glob("bench_*.csv")):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                r = types.SimpleNamespace(
                    test_id=row["test_id"], test_name=row["test_name"],
                    backend=row["backend"], fmt=row["fmt"], quant=row["quant"],
                    kv_cache=row["kv_cache"], prompt_type=row["prompt_type"],
                    ttft_ms=float(row["ttft_ms"] or 0),
                    decode_tps=float(row["decode_tps"] or 0),
                    prefill_tps=float(row["prefill_tps"] or 0),
                    total_time_s=float(row["total_time_s"] or 0),
                    completion_tokens=int(row["completion_tokens"] or 0),
                    prompt_tokens=int(row["prompt_tokens"] or 0),
                    peak_mem_mb=float(row["peak_mem_mb"] or 0),
                    cold_ttft_ms=float(row["cold_ttft_ms"] or 0),
                )
                results.append(r)
    out = BENCH.parent / "results" / "complete_results.html"
    bm.save_results_html(results, HARDWARE, out)
    print(f"  ✔ HTML: {out}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

if DONE_FILE.exists():
    print(f"  Skipping: {DONE_FILE} exists — delete it to re-run", flush=True)
    sys.exit(0)

check_single_instance()

log("Starting Gemma 4 missing-tests re-run")
start_ts = time.strftime("%Y-%m-%d %H:%M:%S")
print(f"  Started: {start_ts}", flush=True)
print(f"  Missing: K_Q4_4, K_OLL_CTX_1, L_Q4_3, L_OLL_CTX_1, L_MXFP4_1", flush=True)

# ── Pull Ollama models ────────────────────────────────────────────────────────

log("Pulling Ollama models")
k26b_ok = ollama_pull("gemma4:26b")
l31b_ok = ollama_pull("gemma4:31b")

# ── Ollama Group K ────────────────────────────────────────────────────────────

if k26b_ok:
    log("Step 1: K_Q4_4 — Gemma4-26B-A4B Ollama Q4_K_M baseline")
    run_tests("K_Q4_4")
else:
    log("Step 1: K_Q4_4 — SKIPPED (gemma4:26b pull failed)")

# ── L_MXFP4_1 (mlx-vlm, not Ollama) ─────────────────────────────────────────

log("Step 2: L_MXFP4_1 — Gemma4-31B mlx-vlm mxfp4 (server_timeout=1200s)")
run_tests("L_MXFP4_1")

# ── Ollama Group L ────────────────────────────────────────────────────────────

if l31b_ok:
    log("Step 3: L_Q4_3 — Gemma4-31B Ollama Q4_K_M baseline")
    run_tests("L_Q4_3")
else:
    log("Step 3: L_Q4_3 — SKIPPED (gemma4:31b pull failed)")

# ── Context tests ─────────────────────────────────────────────────────────────

if k26b_ok:
    log("Step 4: K_OLL_CTX_1 — Gemma4-26B-A4B Ollama ctx-32k/64k/128k")
    run_tests("K_OLL_CTX_1")
else:
    log("Step 4: K_OLL_CTX_1 — SKIPPED (gemma4:26b pull failed)")

if l31b_ok:
    log("Step 5: L_OLL_CTX_1 — Gemma4-31B Ollama ctx-32k/64k/128k")
    run_tests("L_OLL_CTX_1")
else:
    log("Step 5: L_OLL_CTX_1 — SKIPPED (gemma4:31b pull failed)")

# ── Rebuild HTML ───────────────────────────────────────────────────────────────

build_combined_html()

# ── Done ───────────────────────────────────────────────────────────────────────

end_ts = time.strftime("%Y-%m-%d %H:%M:%S")
DONE_FILE.write_text(f"Completed: {start_ts} → {end_ts}\n")
log(f"ALL DONE.  {end_ts}")
print(f"""
  Results: ~/Projects/Work/llm-bench/results/complete_results.html
  Done marker: {DONE_FILE}
""", flush=True)
