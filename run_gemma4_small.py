#!/usr/bin/env python3
"""
Gemma 4 small-model additions — Groups M (E4B) and N (E2B) + L_NVFP4_1.

New tests added to config.yaml by this session (2026-04-04):
  L_NVFP4_1  — Gemma4-31B mlx-vlm nvfp4
  M_Q4_1/2   — Gemma4-E4B mlx-vlm 4bit / kv4 baselines
  M_MXFP4_1  — Gemma4-E4B mlx-vlm mxfp4
  M_NVFP4_1  — Gemma4-E4B mlx-vlm nvfp4
  M_Q4_3     — Gemma4-E4B Ollama Q4_K_M (gemma4:4b)
  M_CTX_1    — Gemma4-E4B ctx 32k/64k/128k/256k
  M_OLL_CTX_1 — Gemma4-E4B Ollama ctx 32k/64k/128k
  N_Q4_1/2   — Gemma4-E2B mlx-vlm 4bit / kv4 baselines
  N_NVFP4_1  — Gemma4-E2B mlx-vlm nvfp4
  N_Q4_3     — Gemma4-E2B Ollama Q4_K_M (gemma4:2b)
  N_CTX_1    — Gemma4-E2B ctx 32k/64k/128k/256k
  N_OLL_CTX_1 — Gemma4-E2B Ollama ctx 32k/64k/128k

Prerequisites:
  - run_mxfp4_fix.py must have completed (L_MXFP4_1 done)
  - AC power + caffeinate

Usage:
  cd ~/Projects/Work/llm-bench
  nohup caffeinate -i python3 -u run_gemma4_small.py >> /tmp/gemma4_small.log 2>&1 &
  PY_PID=$!; caffeinate -i -w $PY_PID &
  echo "PID: $PY_PID" && tail -f /tmp/gemma4_small.log
"""
import csv, os, subprocess, sys, time, types
from pathlib import Path

BENCH = Path(__file__).parent / "benchmark.py"
sys.path.insert(0, str(Path(__file__).parent))

HARDWARE    = {"name": "Apple M3 Max", "memory_gb": 64, "platform": "darwin-arm64"}
HF_CACHE    = Path.home() / ".cache" / "huggingface" / "hub"
DONE_FILE   = Path(__file__).parent / ".gemma4_small_done"
MLX_VLM_PY  = "/Users/HIESCHA/.local/pipx/venvs/mlx-vlm/bin/python"


def log(msg: str):
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def check_single_instance():
    import psutil
    my_pid = os.getpid()
    blocking = ["run_mxfp4_fix.py", "run_gemma4_missing.py", "run_gemma4_bench.py",
                "run_gemma4_small.py"]
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = " ".join(proc.info["cmdline"] or [])
            if proc.info["pid"] == my_pid:
                continue
            for b in blocking:
                if b in cmd:
                    print(f"  ERROR: competing process (PID {proc.info['pid']}): {cmd[:80]}")
                    sys.exit(1)
        except Exception:
            pass


def model_complete(repo_id: str, min_gb: float = 0.5) -> bool:
    dir_name = f"models--{repo_id.replace('/', '--')}"
    model_dir = HF_CACHE / dir_name
    if not model_dir.exists():
        return False
    safetensors = list(model_dir.rglob("*.safetensors"))
    total = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
    print(f"  {repo_id}: {len(safetensors)} safetensors, {total/1e9:.1f} GB", flush=True)
    return len(safetensors) > 0 and total > min_gb * 1e9


def download_model(repo_id: str, min_gb: float = 0.5, attempts: int = 3):
    if model_complete(repo_id, min_gb):
        print(f"  Already complete — skipping", flush=True)
        return True
    for attempt in range(1, attempts + 1):
        log(f"Downloading {repo_id} (attempt {attempt}/{attempts})")
        result = subprocess.run([
            MLX_VLM_PY, "-c",
            f"from huggingface_hub import snapshot_download; snapshot_download('{repo_id}')"
        ], env={**os.environ, "HF_HUB_DISABLE_XET": "1"})
        if result.returncode == 0 and model_complete(repo_id, min_gb):
            log("Download complete")
            return True
        if attempt < attempts:
            print(f"  ⚠ attempt {attempt} failed — retry in 60s", flush=True)
            time.sleep(60)
    print(f"  ✘ download failed after {attempts} attempts", flush=True)
    return False


def ollama_available(tag: str) -> bool:
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    return tag in result.stdout


def ollama_pull(tag: str, max_retries: int = 3) -> bool:
    if ollama_available(tag):
        print(f"  ✔ {tag} already in ollama list", flush=True)
        return True
    for attempt in range(1, max_retries + 1):
        log(f"Pulling {tag} (attempt {attempt}/{max_retries})")
        result = subprocess.run(["ollama", "pull", tag])
        if result.returncode == 0 and ollama_available(tag):
            print(f"  ✔ {tag} pulled", flush=True)
            return True
        if attempt < max_retries:
            print(f"  ⚠ attempt {attempt} failed — retrying in 30s", flush=True)
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
    print(f"  Skipping: {DONE_FILE} exists — delete to re-run", flush=True)
    sys.exit(0)

check_single_instance()
log("Starting Gemma 4 small-model benchmark (Groups M, N + L_NVFP4_1)")
start_ts = time.strftime("%Y-%m-%d %H:%M:%S")
print(f"  Started: {start_ts}", flush=True)

# ── Pull Ollama small models ──────────────────────────────────────────────────

log("Pulling Ollama small models")
m4b_ok = ollama_pull("gemma4:4b")
n2b_ok = ollama_pull("gemma4:2b")

# ── Download MLX models ───────────────────────────────────────────────────────

log("Checking / downloading MLX models")
l_nvfp4_ok = download_model("mlx-community/gemma-4-31b-it-nvfp4", min_gb=10)

# E4B models (~2GB each)
download_model("mlx-community/gemma-4-e4b-it-4bit", min_gb=1)
download_model("mlx-community/gemma-4-e4b-it-mxfp4", min_gb=1)
download_model("mlx-community/gemma-4-e4b-it-nvfp4", min_gb=1)

# E2B models (~1GB each)
download_model("mlx-community/gemma-4-e2b-it-4bit", min_gb=0.5)
download_model("mlx-community/gemma-4-e2b-it-nvfp4", min_gb=0.5)

# ── L_NVFP4_1 — 31B nvfp4 ────────────────────────────────────────────────────

if l_nvfp4_ok:
    log("Step 1: L_NVFP4_1 — Gemma4-31B nvfp4 baseline")
    run_tests("L_NVFP4_1")
else:
    log("Step 1: L_NVFP4_1 — SKIPPED (download failed)")

# ── Group M — E4B baselines ───────────────────────────────────────────────────

log("Step 2: M_Q4_1 — Gemma4-E4B 4bit baseline")
run_tests("M_Q4_1")

log("Step 3: M_Q4_2 — Gemma4-E4B kv4 baseline")
run_tests("M_Q4_2")

log("Step 4: M_MXFP4_1 — Gemma4-E4B mxfp4 baseline")
run_tests("M_MXFP4_1")

log("Step 5: M_NVFP4_1 — Gemma4-E4B nvfp4 baseline")
run_tests("M_NVFP4_1")

if m4b_ok:
    log("Step 6: M_Q4_3 — Gemma4-E4B Ollama Q4_K_M baseline")
    run_tests("M_Q4_3")
else:
    log("Step 6: M_Q4_3 — SKIPPED (gemma4:4b pull failed)")

# ── Group M — E4B context ─────────────────────────────────────────────────────

log("Step 7: M_CTX_1 — Gemma4-E4B ctx 32k/64k/128k/256k")
run_tests("M_CTX_1")

if m4b_ok:
    log("Step 8: M_OLL_CTX_1 — Gemma4-E4B Ollama ctx")
    run_tests("M_OLL_CTX_1")
else:
    log("Step 8: M_OLL_CTX_1 — SKIPPED (gemma4:4b pull failed)")

# ── Group N — E2B baselines ───────────────────────────────────────────────────

log("Step 9: N_Q4_1 — Gemma4-E2B 4bit baseline")
run_tests("N_Q4_1")

log("Step 10: N_Q4_2 — Gemma4-E2B kv4 baseline")
run_tests("N_Q4_2")

log("Step 11: N_NVFP4_1 — Gemma4-E2B nvfp4 baseline")
run_tests("N_NVFP4_1")

if n2b_ok:
    log("Step 12: N_Q4_3 — Gemma4-E2B Ollama Q4_K_M baseline")
    run_tests("N_Q4_3")
else:
    log("Step 12: N_Q4_3 — SKIPPED (gemma4:2b pull failed)")

# ── Group N — E2B context ─────────────────────────────────────────────────────

log("Step 13: N_CTX_1 — Gemma4-E2B ctx 32k/64k/128k/256k")
run_tests("N_CTX_1")

if n2b_ok:
    log("Step 14: N_OLL_CTX_1 — Gemma4-E2B Ollama ctx")
    run_tests("N_OLL_CTX_1")
else:
    log("Step 14: N_OLL_CTX_1 — SKIPPED (gemma4:2b pull failed)")

# ── Rebuild HTML ───────────────────────────────────────────────────────────────

build_combined_html()

# ── Done ───────────────────────────────────────────────────────────────────────

end_ts = time.strftime("%Y-%m-%d %H:%M:%S")
DONE_FILE.write_text(f"Completed: {start_ts} → {end_ts}\n")
log(f"ALL DONE.  {end_ts}")
print(f"""
  Results: ~/Projects/Work/llm-bench/results/complete_results.html
  Done marker: {DONE_FILE}
  New groups: L_NVFP4_1, M (E4B), N (E2B)
""", flush=True)
