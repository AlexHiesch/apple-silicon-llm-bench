#!/usr/bin/env python3
"""
Install TurboQuant, then run Groups F (TurboQuant), G (oMLX), H (llama cache-reuse),
and A_OLL_CTX (Ollama 0.19 prefix cache), then regenerate combined HTML report.

Usage:  python3 -u run_turboquant_omlx.py
"""
import csv, subprocess, sys, time, types
from pathlib import Path

HF_PY = "/opt/homebrew/Cellar/mlx-lm/0.31.1/libexec/bin/python"
BENCH  = Path(__file__).parent / "benchmark.py"
sys.path.insert(0, str(Path(__file__).parent))

HARDWARE = {"name": "Apple M3 Max", "memory_gb": 64, "platform": "darwin-arm64"}

TURBOQUANT_REPO = "https://github.com/sharpner/turboquant-mlx"
TURBOQUANT_DIR  = Path("/tmp/turboquant-mlx")


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"\n[{ts}] {msg}", flush=True)


def install_turboquant() -> bool:
    """Clone and pip-install sharpner/turboquant-mlx into mlx-lm's Python env."""
    log("Installing TurboQuant (sharpner V2 — Metal-accelerated)")

    if not TURBOQUANT_DIR.exists():
        r = subprocess.run(["git", "clone", TURBOQUANT_REPO, str(TURBOQUANT_DIR)])
        if r.returncode != 0:
            print(f"  ERROR: git clone failed (exit {r.returncode})", flush=True)
            return False
    else:
        print(f"  Repo already at {TURBOQUANT_DIR}, pulling latest ...", flush=True)
        subprocess.run(["git", "-C", str(TURBOQUANT_DIR), "pull"])

    # The repo has no pyproject.toml/setup.py — create a minimal one so pip can install it
    setup_py = TURBOQUANT_DIR / "setup.py"
    if not setup_py.exists() and not (TURBOQUANT_DIR / "pyproject.toml").exists():
        print("  Creating minimal setup.py (repo has no build config) ...", flush=True)
        setup_py.write_text(
            "from setuptools import setup, find_packages\n"
            "setup(name='turboquant-mlx', version='0.1.0', packages=find_packages())\n"
        )

    # Install requirements first (mlx, mlx-lm, numpy)
    req = TURBOQUANT_DIR / "requirements.txt"
    if req.exists():
        subprocess.run([HF_PY, "-m", "pip", "install", "-r", str(req)], check=False)

    r = subprocess.run([HF_PY, "-m", "pip", "install", "-e", str(TURBOQUANT_DIR)])
    if r.returncode != 0:
        print(f"  ERROR: pip install failed (exit {r.returncode})", flush=True)
        return False

    # Verify import
    r = subprocess.run([HF_PY, "-c", "import turboquant; print('  OK:', turboquant.__file__)"],
                       capture_output=False)
    if r.returncode != 0:
        print("  ERROR: 'import turboquant' failed after install", flush=True)
        return False

    print("  TurboQuant installed successfully", flush=True)
    return True


def run_bench_group(group: str):
    log(f"Benchmarking group: {group}")
    subprocess.run([
        sys.executable, "-u", str(BENCH),
        "--group", group,
        "--skip-unavailable", "--no-think", "--report-html",
    ])


def run_bench_tests(*test_ids: str):
    log(f"Benchmarking tests: {', '.join(test_ids)}")
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
    print(f"  Combined report: {out}", flush=True)


# ── Phase 1: Install TurboQuant ───────────────────────────────────────────────

if not install_turboquant():
    print("\nTurboQuant install failed — Group F will run in fallback mode (standard KV cache).",
          flush=True)
    print("Continuing with Groups G, H, and A_OLL_CTX ...\n", flush=True)

# ── Phase 2: Run new groups ───────────────────────────────────────────────────

log("Phase 2: Group F — TurboQuant")
run_bench_group("F — TurboQuant")

log("Phase 3: Group G — oMLX (SSD KV Cache)")
run_bench_group("G — oMLX (SSD KV Cache)")

log("Phase 4: Group H — llama cache-reuse")
run_bench_group("H — llama cache-reuse")

log("Phase 5: Ollama 0.19 prefix-cache context tests (A_OLL_CTX)")
run_bench_tests("A_OLL_CTX")

# ── Phase 6: Regenerate combined report ──────────────────────────────────────

build_combined_html()
log("All done.")
