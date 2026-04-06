#!/usr/bin/env python3
"""
Run the remaining gaps vs the document's complete matrix:
  1. F_TQ4_1 + F_TQ35_1  — TurboQuant baselines re-run (clean AC power confirmation)
  2. A_OLL_CTX_INT4       — Ollama 0.19 INT4 context (fastest format, 134 t/s claim)
  3. Group I              — vllm-mlx (21–87% over llama.cpp claim)
  4. Group J              — LM Studio (reference baseline from document)
  5. Rebuild complete_results.html

Usage:  python3 -u run_remaining_gaps.py
Run only after charger is connected and no other benchmarks are running.
"""
import csv, subprocess, sys, time, types
from pathlib import Path

BENCH = Path(__file__).parent / "benchmark.py"
sys.path.insert(0, str(Path(__file__).parent))

HARDWARE = {"name": "Apple M3 Max", "memory_gb": 64, "platform": "darwin-arm64"}


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"\n[{ts}] {msg}", flush=True)


def check_single_instance():
    """Abort if another benchmark or this script is already running."""
    import psutil, os
    my_pid = os.getpid()
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmd = " ".join(proc.info['cmdline'] or [])
            if proc.info['pid'] != my_pid and ("benchmark.py" in cmd or "run_" in cmd.split("/")[-1]):
                print(f"  ERROR: competing process detected (PID {proc.info['pid']}): {cmd[:80]}")
                print("  Kill it first, then re-run this script.")
                sys.exit(1)
        except Exception:
            pass


def run_tests(*test_ids):
    log(f"Running tests: {', '.join(test_ids)}")
    subprocess.run([
        sys.executable, "-u", str(BENCH),
        "--test", *test_ids,
        "--skip-unavailable", "--no-think", "--report-html",
    ])


def run_group(group: str):
    log(f"Running group: {group}")
    subprocess.run([
        sys.executable, "-u", str(BENCH),
        "--group", group,
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
    print(f"  Combined report: {out}", flush=True)


# ── Guard: abort if anything else is already running ─────────────────────────
check_single_instance()

# ── Step 1: TurboQuant baselines re-run (precautionary AC-power confirmation) ─
log("Step 1: TurboQuant baselines re-run (F_TQ4_1 + F_TQ35_1)")
run_tests("F_TQ4_1", "F_TQ35_1")

# ── Step 2: Ollama 0.19 INT4 context ──────────────────────────────────────────
log("Step 2: Ollama 0.19 INT4 context (A_OLL_CTX_INT4)")
run_tests("A_OLL_CTX_INT4")

# ── Step 3: vllm-mlx ──────────────────────────────────────────────────────────
log("Step 3: Group I — vllm-mlx")
run_group("I — vllm-mlx")

# ── Step 4: LM Studio ─────────────────────────────────────────────────────────
log("Step 4: Group J — LM Studio")
run_group("J — LM Studio")

# ── Step 5: Rebuild combined HTML ─────────────────────────────────────────────
build_combined_html()
log("All done. Open results/complete_results.html for the full picture.")
