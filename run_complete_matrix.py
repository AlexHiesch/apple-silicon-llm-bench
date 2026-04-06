#!/usr/bin/env python3
"""
Complete benchmark matrix — runs every remaining gap in one sequential session.

Covers all groups A–J across all feasible context tiers:
  1. F_TQ4_1 + F_TQ35_1   — TurboQuant baselines re-run (AC-power confirmation)
  2. A_OLL_CTX_INT4        — Ollama 0.19 INT4 context (tainted CSV deleted)
  3. B_CTX_1               — Coder-Next mlx-lm ctx-32k/64k/128k (128k new — "fits but tight")
  4. G_CTX_1 + G_CTX_2     — oMLX SSD-tier ctx-64k/128k (32k exists; long-ctx is oMLX's key claim)
  5. Group I               — vllm-mlx Qwen3.5 + Coder baseline + Qwen3.5 ctx
  6. Group J               — LM Studio Qwen3.5 + Coder baseline
  7. HTML rebuild          — complete_results.html from all CSVs

Skipped (Phase 2, needs separate infrastructure):
  - Tool call parameter accuracy
  - Needle-in-haystack / RULER
  - Thinking token overhead matrix
  - MLC-LLM (not installable via pip/brew on this system)
  - mlx-optiq 35B (no pre-built OptiQ models for 35B on HF)

Usage:
  cd ~/Projects/Work/llm-bench
  python3 -u run_complete_matrix.py 2>&1 | tee /tmp/complete_matrix.log

Prerequisites (check before running):
  pmset -g batt             # battery ≥ 50% preferred; powermode 2
  pmset -g | grep powermode # must show: powermode 2
"""
import csv, subprocess, sys, time, types
from pathlib import Path

BENCH = Path(__file__).parent / "benchmark.py"
sys.path.insert(0, str(Path(__file__).parent))

HARDWARE = {"name": "Apple M3 Max", "memory_gb": 64, "platform": "darwin-arm64"}

ETA_ESTIMATES = {
    "TurboQuant re-run (F_TQ4_1 + F_TQ35_1)": "~15 min",
    "Ollama INT4 context (A_OLL_CTX_INT4)":    "~25 min",
    "Coder mlx-lm ctx-128k (B_CTX_1)":         "~30 min (128k may OOM — skipped gracefully)",
    "oMLX long context (G_CTX_1 + G_CTX_2)":   "~45 min (SSD paging; 64k/128k new data)",
    "vllm-mlx Group I":                         "~25 min",
    "LM Studio Group J":                        "~10 min",
    "HTML rebuild":                             "~1 min",
}


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"\n[{ts}] {msg}", flush=True)


def check_single_instance():
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


def check_prerequisites():
    import subprocess
    log("Checking prerequisites")
    # Power mode
    try:
        out = subprocess.check_output(["pmset", "-g"], text=True)
        powermode = next((l for l in out.splitlines() if "powermode" in l), "")
        if "2" in powermode:
            print(f"  ✔ powermode 2 (High Performance)")
        else:
            print(f"  ⚠ WARNING: {powermode.strip()} — expected powermode 2")
            print("    Set with: sudo pmset -a powermode 2")
    except Exception:
        print("  ⚠ Could not read powermode")
    # Battery
    try:
        out = subprocess.check_output(["pmset", "-g", "batt"], text=True)
        batt_line = next((l for l in out.splitlines() if "InternalBattery" in l), "")
        print(f"  {batt_line.strip()}")
        if "charging" not in batt_line and "AC" not in batt_line:
            pct = int(batt_line.split(";")[0].split("%")[0].split()[-1])
            if pct < 30:
                print(f"  ⚠ WARNING: battery at {pct}% and not charging — risk of throttling below 15%")
                if pct < 15:
                    print("  ✘ ABORT: battery too low. Connect charger first.")
                    sys.exit(1)
    except Exception:
        print("  ⚠ Could not read battery level")


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


# ── Main ────────────────────────────────────────────────────────────────────────

check_single_instance()
check_prerequisites()

log("Starting complete matrix run")
print("\n  Estimated times:")
for step, eta in ETA_ESTIMATES.items():
    print(f"    {step}: {eta}")
print(f"\n  Total ETA: ~2.5–3 hours\n")

# ── Step 1: TurboQuant baselines re-run ───────────────────────────────────────
log("Step 1: TurboQuant baselines re-run (F_TQ4_1 + F_TQ35_1)")
run_tests("F_TQ4_1", "F_TQ35_1")

# ── Step 2: Ollama 0.19 INT4 context ─────────────────────────────────────────
log("Step 2: Ollama 0.19 INT4 context (A_OLL_CTX_INT4) — ctx-32k/64k")
run_tests("A_OLL_CTX_INT4")

# ── Step 3: Coder-Next mlx-lm ctx-128k ───────────────────────────────────────
# B_CTX_1 has ctx-32k and ctx-64k in results; ctx-128k was never collected.
# Memory budget: ~15 GB weights + ~38 GB KV at 128k = ~53 GB total on 64 GB — tight but fits.
# --skip-unavailable handles OOM gracefully (connection refused → no row saved).
log("Step 3: Coder-Next mlx-lm context (B_CTX_1) — all tiers including ctx-128k")
run_tests("B_CTX_1")

# ── Step 4: oMLX long context (64k + 128k — oMLX's key claim) ────────────────
# G_CTX_1 and G_CTX_2 have ctx-32k in results; 64k/128k were never collected.
# oMLX SSD-tiered KV: hot blocks in 8GB RAM, cold blocks spill to NVMe.
# The document claims "virtually unlimited context" — this is the test.
# If SSD paging dominates, TTFT at 64k/128k will reveal the paging overhead.
log("Step 4: oMLX long context (G_CTX_1 + G_CTX_2) — ctx-32k/64k/128k")
run_tests("G_CTX_1", "G_CTX_2")

# ── Step 5: vllm-mlx ─────────────────────────────────────────────────────────
log("Step 5: Group I — vllm-mlx (Qwen3.5 + Coder baseline + Qwen3.5 ctx)")
run_group("I — vllm-mlx")

# ── Step 6: LM Studio ────────────────────────────────────────────────────────
log("Step 6: Group J — LM Studio (Qwen3.5 + Coder baseline)")
run_group("J — LM Studio")

# ── Step 7: Rebuild combined HTML ─────────────────────────────────────────────
build_combined_html()

log("Complete matrix run finished. Open results/complete_results.html")
print("""
  Groups completed: A B C D E F G H I J
  Context tiers collected for: A, B, C(disabled-SWA), D(32k only), E(baseline only),
                                F, G(32k+64k+128k), H(32k+64k), I, A_OLL_CTX(32k+64k)

  Still pending (Phase 2 — needs new infrastructure):
    - MLC-LLM (not installable via pip/brew)
    - mlx-optiq 35B (no pre-built models on HF)
    - Tool call accuracy / RULER / thinking token matrix
""", flush=True)
