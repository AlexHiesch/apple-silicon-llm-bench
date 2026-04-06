#!/usr/bin/env python3
"""
Re-run benchmarks affected by mlx-vlm 0.4.4 fixes.

mlx-vlm 0.4.4 (released 2026-04-04) fixes:
  1. Gemma 4 E4B/E2B chunked prefill (per_layer_projection shape mismatch)
     → M_CTX_1, N_CTX_1 were disabled; now should work
  2. KV-shared model fixes for chunked prefill
     → K_CTX_2, L_CTX_2 had "No tokens received"; may now work
  3. PromptCacheState (prefix caching for mlx-vlm)
     → Not directly benchmarkable via config.yaml (needs multi-turn), but
       the underlying prefill improvements may affect single-request TTFT too

This script:
  - Activates the .venv with mlx-vlm 0.4.4 on PATH
  - Re-enables disabled tests
  - Runs M_CTX_1, N_CTX_1, K_CTX_2, L_CTX_2
  - Rebuilds the HTML report
"""

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

BENCH_DIR = Path(__file__).parent
VENV = BENCH_DIR / ".venv"
CONFIG = BENCH_DIR / "config.yaml"
BENCHMARK = BENCH_DIR / "benchmark.py"

# Tests to re-run
TESTS = ["M_CTX_1", "N_CTX_1", "K_CTX_2", "L_CTX_2"]

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def check_single_instance():
    import psutil
    me = os.getpid()
    parent = os.getppid()
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cl = " ".join(p.info["cmdline"] or [])
            pid = p.info["pid"]
            if "run_vlm044_rerun" in cl and pid != me and pid != parent:
                print(f"ERROR: Another instance running (PID {pid})")
                sys.exit(1)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

def patch_config_enable(test_id: str):
    """Temporarily enable a disabled test in config.yaml."""
    text = CONFIG.read_text()
    # Pattern: id: M_CTX_1 ... enabled: false  # comment
    import re
    # Find the test block and enable it
    lines = text.split('\n')
    in_test = False
    patched = False
    for i, line in enumerate(lines):
        if f'id: {test_id}' in line:
            in_test = True
        elif in_test and line.strip().startswith('- id:'):
            in_test = False
        elif in_test and 'enabled: false' in line:
            lines[i] = line.replace('enabled: false', 'enabled: true  # TEMP re-enabled for 0.4.4 rerun')
            patched = True
            in_test = False
    if patched:
        CONFIG.write_text('\n'.join(lines))
        log(f"  Enabled {test_id} in config.yaml")
    else:
        log(f"  {test_id} already enabled or not found")

def restore_config():
    """Restore any TEMP re-enabled tests back to disabled."""
    text = CONFIG.read_text()
    text = text.replace('enabled: true  # TEMP re-enabled for 0.4.4 rerun',
                        'enabled: false  # mlx-vlm 0.4.3 bug (re-tested on 0.4.4)')
    CONFIG.write_text(text)
    log("Restored config.yaml (disabled temp-enabled tests)")

def run_benchmark(test_ids: list[str]):
    """Run benchmark.py with venv on PATH for mlx-vlm 0.4.4."""
    env = os.environ.copy()
    # Prepend venv bin to PATH so mlx_vlm.server resolves to 0.4.4
    env["PATH"] = str(VENV / "bin") + ":" + env.get("PATH", "")
    # Verify
    result = subprocess.run(
        [str(VENV / "bin" / "python3"), "-c", "import mlx_vlm; print(f'mlx-vlm {mlx_vlm.__version__}')"],
        capture_output=True, text=True, env=env
    )
    log(f"Using: {result.stdout.strip()}")

    cmd = [
        sys.executable, str(BENCHMARK),
        "--test", *test_ids,
        "--skip-unavailable",
        "--report-html",
    ]
    log(f"Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, env=env)
    return proc.returncode

def rebuild_html():
    """Rebuild the complete HTML report."""
    env = os.environ.copy()
    env["PATH"] = str(VENV / "bin") + ":" + env.get("PATH", "")
    cmd = [sys.executable, str(BENCHMARK), "--report-html", "--test", "NONE_EXIST"]
    # This will fail to find tests but trigger HTML rebuild from existing CSV
    subprocess.run(cmd, env=env, capture_output=True)

def main():
    check_single_instance()
    start = time.time()

    log("=" * 60)
    log("mlx-vlm 0.4.4 re-run: testing fixes for E4B/E2B chunked prefill + kv4 context")
    log(f"Tests: {', '.join(TESTS)}")
    log("=" * 60)

    # Step 1: Enable disabled tests
    log("\nStep 1: Enabling disabled tests in config.yaml")
    for tid in TESTS:
        patch_config_enable(tid)

    try:
        # Step 2: Run benchmarks
        log(f"\nStep 2: Running {len(TESTS)} test configurations")
        rc = run_benchmark(TESTS)
        log(f"Benchmark exit code: {rc}")
    finally:
        # Step 3: Always restore config
        log("\nStep 3: Restoring config.yaml")
        restore_config()

    elapsed = time.time() - start
    log(f"\nDone in {elapsed/60:.1f} minutes")
    log(f"Results in: {BENCH_DIR / 'results'}")

if __name__ == "__main__":
    main()
