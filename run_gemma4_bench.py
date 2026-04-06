#!/usr/bin/env python3
"""
Gemma 4 benchmark — Groups K (26B A4B MoE) and L (31B Dense).

Overnight-safe runner: downloads models, runs all mlx-vlm tests, rebuilds HTML.
Uses caffeinate internally to prevent macOS sleep between steps.

Disabled (documented, re-enable when deps ship):
  K_Q4_3 (kv2)         — verify mlx-vlm gemma4 kv2 support first
  K_Q4_4, K_OLL_CTX_1  — Ollama gemma4 requires 0.21.0+ (current: 0.20.0)
  L_Q4_3, L_OLL_CTX_1  — same Ollama version constraint
  K_TQ4/35/3_1          — mlx-lm 0.31.1 has no gemma4 module

Execution order + downloads:
  [Download] gemma-4-26b-a4b-it-4bit   (~13 GB)
  Step 1:  K_Q4_1     mlx-vlm 4bit baseline
  Step 2:  K_Q4_2     mlx-vlm kv4 baseline
  [Download] gemma-4-26b-a4b-it-mxfp4  (~13 GB)
  Step 3:  K_MXFP4_1  mlx-vlm mxfp4 (MX Microscaling FP4)
  [Download] gemma-4-26b-a4b-it-nvfp4  (~13 GB)
  Step 4:  K_NVFP4_1  mlx-vlm nvfp4
  Step 5:  K_CTX_1    mlx-vlm ctx-32k/64k/128k
  Step 6:  K_CTX_2    mlx-vlm kv4 ctx (~200–400s TTFT at 128k expected)
  [Download] gemma-4-31b-it-4bit        (~17 GB)
  Step 7:  L_Q4_1     Gemma4-31B mlx-vlm 4bit baseline
  Step 8:  L_Q4_2     Gemma4-31B mlx-vlm kv4 baseline
  [Download] gemma-4-31b-it-mxfp4       (~17 GB)
  Step 9:  L_MXFP4_1  Gemma4-31B mlx-vlm mxfp4
  Step 10: L_CTX_1    Gemma4-31B mlx-vlm ctx-32k/64k/128k
  Step 11: L_CTX_2    Gemma4-31B mlx-vlm kv4 ctx-32k/64k/128k
  Step 12: HTML rebuild

Usage (run from terminal, not Claude Code):
  cd ~/Projects/Work/llm-bench
  nohup caffeinate -i python3 -u run_gemma4_bench.py >> /tmp/gemma4_bench.log 2>&1 &
  echo "PID: $!"
  tail -f /tmp/gemma4_bench.log
"""
import csv, os, subprocess, sys, time, types
from pathlib import Path

BENCH = Path(__file__).parent / "benchmark.py"
sys.path.insert(0, str(Path(__file__).parent))

HARDWARE  = {"name": "Apple M3 Max", "memory_gb": 64, "platform": "darwin-arm64"}
HF_CACHE  = Path.home() / ".cache" / "huggingface" / "hub"
MLX_VLM_PYTHON = "/Users/HIESCHA/.local/pipx/venvs/mlx-vlm/bin/python"
DONE_FILE = Path(__file__).parent / ".gemma4_bench_done"
LOG_FILE  = Path("/tmp/gemma4_bench.log")


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"\n[{ts}] {msg}"
    print(line, flush=True)


def hf_model_cached(repo_id: str) -> bool:
    """Return True if the HF model is in cache and complete (>1 GB)."""
    dir_name = f"models--{repo_id.replace('/', '--')}"
    model_dir = HF_CACHE / dir_name
    if not model_dir.exists():
        return False
    total = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
    return total > 1_000_000_000


def ensure_hf_model(repo_id: str, max_retries: int = 3):
    """Download model to HF cache. Retries up to max_retries times."""
    if hf_model_cached(repo_id):
        # report size
        dir_name = f"models--{repo_id.replace('/', '--')}"
        total = sum(f.stat().st_size for f in (HF_CACHE / dir_name).rglob("*") if f.is_file())
        print(f"  ✔ {repo_id} already cached ({total/1e9:.1f} GB)", flush=True)
        return

    for attempt in range(1, max_retries + 1):
        log(f"Downloading {repo_id} (attempt {attempt}/{max_retries})")
        result = subprocess.run([
            MLX_VLM_PYTHON, "-c",
            f"from huggingface_hub import snapshot_download; "
            f"snapshot_download('{repo_id}')"
        ])
        if result.returncode == 0 and hf_model_cached(repo_id):
            dir_name = f"models--{repo_id.replace('/', '--')}"
            total = sum(f.stat().st_size for f in (HF_CACHE / dir_name).rglob("*") if f.is_file())
            print(f"  ✔ Downloaded {repo_id} ({total/1e9:.1f} GB)", flush=True)
            return
        if attempt < max_retries:
            print(f"  ⚠ Download attempt {attempt} failed — retrying in 60s", flush=True)
            time.sleep(60)
        else:
            print(f"  ✘ All {max_retries} download attempts failed for {repo_id}", flush=True)
            print(f"    Tests requiring this model will be skipped.", flush=True)


def check_single_instance():
    import psutil
    my_pid = os.getpid()
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmd = " ".join(proc.info['cmdline'] or [])
            if proc.info['pid'] != my_pid and (
                "run_gemma4_bench.py" in cmd or
                ("benchmark.py" in cmd and proc.info['pid'] != my_pid)
            ):
                print(f"  ERROR: competing process (PID {proc.info['pid']}): {cmd[:80]}")
                sys.exit(1)
        except Exception:
            pass


def check_prerequisites():
    log("Checking prerequisites")
    try:
        out = subprocess.check_output(["pmset", "-g"], text=True)
        powermode = next((l for l in out.splitlines() if "powermode" in l), "")
        print(f"  {'✔' if '2' in powermode else '⚠'} {powermode.strip() or 'powermode unknown'}", flush=True)
    except Exception:
        pass
    try:
        out = subprocess.check_output(["pmset", "-g", "batt"], text=True)
        batt_line = next((l for l in out.splitlines() if "InternalBattery" in l), "")
        print(f"  {batt_line.strip()}", flush=True)
    except Exception:
        pass
    result = subprocess.run(
        [MLX_VLM_PYTHON, "-c", "import mlx_vlm.models.gemma4; print('ok')"],
        capture_output=True, text=True
    )
    if "ok" in result.stdout:
        print("  ✔ mlx-vlm gemma4 architecture: ready", flush=True)
    else:
        print(f"  ✘ mlx-vlm missing gemma4 — run: pipx upgrade mlx-vlm", flush=True)
        sys.exit(1)


def run_tests(*test_ids: str):
    log(f"Running: {', '.join(test_ids)}")
    subprocess.run([
        sys.executable, "-u", str(BENCH),
        "--test", *test_ids,
        "--skip-unavailable", "--no-think", "--report-html",
    ])


def build_combined_html():
    log("Building combined HTML report")
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
    print(f"  ✔ HTML report: {out}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

if DONE_FILE.exists():
    print(f"  Skipping: {DONE_FILE} exists — delete it to re-run", flush=True)
    sys.exit(0)

check_single_instance()
check_prerequisites()

log("Starting Gemma 4 benchmark — Groups K + L")
start_ts = time.strftime("%Y-%m-%d %H:%M:%S")
print(f"  Started: {start_ts}", flush=True)
print(f"  Log:     {LOG_FILE}", flush=True)
print(f"  Total downloads: ~73 GB  |  Benchmark runtime: ~3.5 h", flush=True)

# ── Group K: Gemma 4 26B A4B (MoE) ───────────────────────────────────────────

ensure_hf_model("mlx-community/gemma-4-26b-a4b-it-4bit")

log("Step 1: K_Q4_1 — Gemma4-26B-A4B mlx-vlm 4bit baseline")
run_tests("K_Q4_1")

log("Step 2: K_Q4_2 — Gemma4-26B-A4B mlx-vlm kv4 baseline")
run_tests("K_Q4_2")

# Step 3 (Ollama K_Q4_4): disabled — Ollama 0.21.0+ required for gemma4
log("Step 3: K_Q4_4 — SKIPPED (Ollama 0.21.0+ required for gemma4)")

ensure_hf_model("mlx-community/gemma-4-26b-a4b-it-mxfp4")
log("Step 4: K_MXFP4_1 — Gemma4-26B-A4B mlx-vlm mxfp4 (MX Microscaling FP4)")
run_tests("K_MXFP4_1")

ensure_hf_model("mlx-community/gemma-4-26b-a4b-it-nvfp4")
log("Step 5: K_NVFP4_1 — Gemma4-26B-A4B mlx-vlm nvfp4")
run_tests("K_NVFP4_1")

log("Step 6: K_CTX_1 — Gemma4-26B-A4B mlx-vlm ctx-32k/64k/128k")
run_tests("K_CTX_1")

log("Step 7: K_CTX_2 — Gemma4-26B-A4B mlx-vlm kv4 ctx (slow TTFT at 128k expected)")
run_tests("K_CTX_2")

# Step 8 (Ollama K_OLL_CTX_1): disabled
log("Step 8: K_OLL_CTX_1 — SKIPPED (Ollama 0.21.0+ required for gemma4)")

# ── Group L: Gemma 4 31B Dense ────────────────────────────────────────────────

ensure_hf_model("mlx-community/gemma-4-31b-it-4bit")

log("Step 9: L_Q4_1 — Gemma4-31B mlx-vlm 4bit baseline")
run_tests("L_Q4_1")

log("Step 10: L_Q4_2 — Gemma4-31B mlx-vlm kv4 baseline")
run_tests("L_Q4_2")

# Step 11 (Ollama L_Q4_3): disabled
log("Step 11: L_Q4_3 — SKIPPED (Ollama 0.21.0+ required for gemma4)")

ensure_hf_model("mlx-community/gemma-4-31b-it-mxfp4")
log("Step 12: L_MXFP4_1 — Gemma4-31B mlx-vlm mxfp4")
run_tests("L_MXFP4_1")

log("Step 13: L_CTX_1 — Gemma4-31B mlx-vlm ctx-32k/64k/128k")
run_tests("L_CTX_1")

log("Step 14: L_CTX_2 — Gemma4-31B mlx-vlm kv4 ctx-32k/64k/128k")
run_tests("L_CTX_2")

# Step 15 (Ollama L_OLL_CTX_1): disabled
log("Step 15: L_OLL_CTX_1 — SKIPPED (Ollama 0.21.0+ required for gemma4)")

# ── Rebuild HTML ───────────────────────────────────────────────────────────────

build_combined_html()

# ── Done ───────────────────────────────────────────────────────────────────────

end_ts = time.strftime("%Y-%m-%d %H:%M:%S")
DONE_FILE.write_text(f"Completed: {start_ts} → {end_ts}\n")
log(f"ALL DONE — Groups K + L complete.  {end_ts}")
print("""
  Enabled tests run:   K_Q4_1/2, K_MXFP4_1, K_NVFP4_1, K_CTX_1/2
                       L_Q4_1/2, L_MXFP4_1, L_CTX_1/2
  Skipped (Ollama):    K_Q4_4, K_OLL_CTX_1, L_Q4_3, L_OLL_CTX_1
                       → re-enable after: brew upgrade ollama (needs 0.21.0+)
  Disabled (arch):     K_TQ4/35/3_1 (mlx-lm), K_Q4_3 (kv2 verify)
  Results:             ~/Projects/Work/llm-bench/results/complete_results.html
  Done marker:         ~/Projects/Work/llm-bench/.gemma4_bench_done
""", flush=True)
