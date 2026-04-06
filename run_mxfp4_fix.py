#!/usr/bin/env python3
"""
Fix L_MXFP4_1: re-download gemma-4-31b-it-mxfp4 (was 0 safetensors, only metadata)
and retry with server_timeout: 1200 in config.yaml.

Usage:
  cd ~/Projects/Work/llm-bench
  nohup python3 -u run_mxfp4_fix.py >> /tmp/mxfp4_fix.log 2>&1 &
  echo "PID: $!" && tail -f /tmp/mxfp4_fix.log
"""
import csv, os, subprocess, sys, time, types
from pathlib import Path

BENCH = Path(__file__).parent / "benchmark.py"
sys.path.insert(0, str(Path(__file__).parent))
MLX_VLM_PYTHON = "/Users/HIESCHA/.local/pipx/venvs/mlx-vlm/bin/python"
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
HARDWARE = {"name": "Apple M3 Max", "memory_gb": 64, "platform": "darwin-arm64"}
REPO = "mlx-community/gemma-4-31b-it-mxfp4"


def log(msg):
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def model_complete(repo_id):
    dir_name = f"models--{repo_id.replace('/', '--')}"
    model_dir = HF_CACHE / dir_name
    safetensors = list(model_dir.rglob("*.safetensors")) if model_dir.exists() else []
    total = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file()) if model_dir.exists() else 0
    print(f"  {repo_id}: {len(safetensors)} safetensors, {total/1e9:.1f} GB", flush=True)
    return len(safetensors) > 0 and total > 10_000_000_000  # >10GB = has weights


log(f"Checking {REPO}")
if model_complete(REPO):
    print("  Already complete — skipping download", flush=True)
else:
    for attempt in range(1, 4):
        log(f"Downloading {REPO} (attempt {attempt}/3)")
        result = subprocess.run([
            MLX_VLM_PYTHON, "-c",
            f"from huggingface_hub import snapshot_download; snapshot_download('{REPO}')"
        ], env={**os.environ, "HF_HUB_DISABLE_XET": "1"})
        if result.returncode == 0 and model_complete(REPO):
            log("Download complete")
            break
        if attempt < 3:
            print(f"  ⚠ attempt {attempt} failed — retry in 60s", flush=True)
            time.sleep(60)
    else:
        print("  ✘ download failed after 3 attempts — aborting", flush=True)
        sys.exit(1)

log("Running L_MXFP4_1 (server_timeout=1200s in config.yaml)")
subprocess.run([
    sys.executable, "-u", str(BENCH),
    "--test", "L_MXFP4_1",
    "--skip-unavailable", "--no-think", "--report-html",
])

log("Rebuilding combined HTML")
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
print(f"  ✔ {out}", flush=True)

log("DONE")
