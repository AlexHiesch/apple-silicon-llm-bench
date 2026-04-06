#!/usr/bin/env python3
"""
Full context benchmark: run A/B groups, download C/D/E models, run C/D/E groups,
then regenerate a combined HTML report covering all results.

Usage:  python3 -u run_full_context_bench.py
"""
import csv, subprocess, sys, time, types
from pathlib import Path

HF_PY = "/opt/homebrew/Cellar/mlx-lm/0.31.1/libexec/bin/python"
BENCH  = Path(__file__).parent / "benchmark.py"
sys.path.insert(0, str(Path(__file__).parent))

HARDWARE = {"name": "Apple M3 Max", "memory_gb": 64, "platform": "darwin-arm64"}

# ── helpers ──────────────────────────────────────────────────────────────────

def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"\n[{ts}] {msg}", flush=True)

def run_bench(groups: list[str] | None = None, test_ids: list[str] | None = None,
              extra: list[str] | None = None):
    cmd = [sys.executable, "-u", str(BENCH),
           "--skip-unavailable", "--no-think", "--report-html"]
    if groups:
        for g in groups:
            cmd += ["--group", g]     # NOTE: --group only accepts one value; run separately
    if test_ids:
        cmd += ["--test"] + test_ids
    if extra:
        cmd += extra
    subprocess.run(cmd)

def run_bench_group(group: str):
    log(f"Running benchmark — group: {group}")
    subprocess.run([
        sys.executable, "-u", str(BENCH),
        "--group", group,
        "--skip-unavailable", "--no-think", "--report-html",
    ])

def download_model(repo_id: str) -> bool:
    log(f"Downloading {repo_id}")
    t0 = time.time()
    r = subprocess.run([
        HF_PY, "-c",
        f"from huggingface_hub import snapshot_download; "
        f"snapshot_download('{repo_id}', repo_type='model')"
    ])
    elapsed = time.time() - t0
    if r.returncode != 0:
        print(f"  ERROR: download failed (exit {r.returncode})", flush=True)
        return False
    print(f"  Downloaded in {elapsed/60:.1f} min", flush=True)
    return True

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

# ── phase 1: existing models (A + B groups) ──────────────────────────────────

log("Phase 1: Groups A + B (cached models — starting immediately)")
for g in ["A — Qwen3.5-35B-A3B (MoE)", "B — Qwen3-Coder-Next (Dense)"]:
    run_bench_group(g)

# ── phase 2: new models — download then bench ────────────────────────────────

NEW_MODELS = [
    ("mlx-community/gemma-3-27b-it-4bit",         "C — Gemma 3 27B-IT"),
    ("mlx-community/Qwen3-32B-4bit",              "D — Qwen3-32B (Dense)"),
    ("mlx-community/Llama-3.3-70B-Instruct-4bit", "E — Llama 3.3 70B"),
]

for repo, group in NEW_MODELS:
    log(f"Phase 2: {group}")
    if download_model(repo):
        run_bench_group(group)
    else:
        print(f"  Skipping benchmark for '{group}' (download failed)", flush=True)

# ── phase 3: combined report ─────────────────────────────────────────────────

build_combined_html()
log("All done.")
