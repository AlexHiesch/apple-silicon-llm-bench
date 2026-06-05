#!/usr/bin/env python3
"""
Gemma 4 12B benchmark — Group U (Dense, Unified/Encoder-Free).

Overnight-safe runner: downloads models, runs all tests, rebuilds HTML.
Uses caffeinate internally to prevent macOS sleep between steps.

Model: google/gemma-4-12B-it (12B dense, unified encoder-free architecture)
  - No separate vision/audio encoders — raw patches projected into LLM backbone
  - Hybrid attention: sliding window (1024) + global
  - 256K native context, 48 layers
  - Supports thinking mode ("ultrathink")

Execution order + downloads:
  [Download] gemma-4-12B-it-4bit    (~6 GB)
  Step 1:  U_Q4_1      mlx-vlm 4bit baseline
  Step 2:  U_Q4_2      mlx-vlm kv4 baseline
  [Download] gemma-4-12B-it-8bit    (~12 GB)
  Step 3:  U_Q8_1      mlx-vlm 8bit
  [Download] gemma-4-12B-it-mxfp4   (~6 GB)
  Step 4:  U_MXFP4_1   mlx-vlm mxfp4
  [Download] gemma-4-12B-it-nvfp4   (~6 GB)
  Step 5:  U_NVFP4_1   mlx-vlm nvfp4
  Step 6:  U_Q4_3      Ollama gemma4:12b
  Step 7:  U_CTX_1     mlx-vlm ctx-32k/64k/128k
  Step 8:  U_CTX_2     mlx-vlm kv4 ctx-32k/64k/128k
  Step 9:  U_OLL_CTX_1 Ollama ctx-32k/64k/128k
  Step 10: U_LC_Q4KM   llama-server Q4_K_M
  Step 11: U_LC_UDQ4   llama-server UD-Q4_K_XL
  Step 12: U_LC_UDQ2   llama-server UD-Q2_K_XL
  Step 13: U_LC_UDQ6   llama-server UD-Q6_K_XL
  Step 14: U_VM_1      vllm-mlx 4bit
  Step 15: U_OX_1      oMLX ssd-paged baseline
  Step 16: U_OX_CTX_1  oMLX ctx-32k
  Step 17: U_DMR_1     Docker Model Runner
  Step 18: U_LMS_1     LM Studio
  Step 19: U_SPEC_1    speculative (E2B draft, UD-Q4_K_XL)
  Step 20: U_SPEC_2    speculative (E2B draft, Q4_K_M)
  Step 21: U_THINK_1   ultrathink mlx-vlm 4bit
  Step 22: U_THINK_2   ultrathink mlx-vlm 8bit
  Step 23: U_THINK_3   ultrathink Ollama
  Step 24: U_THINK_4   ultrathink llama-server
  Step 25: HTML rebuild

  Disabled (need external deps):
    U_DF_1    dflash (no z-lab drafter for 12B yet)
    U_MTP_1/2 MTP self-draft (GGUF unverified)
    U_TQ4_1   TurboQuant (mlx-lm lacks gemma4)
    U_TQ35_1  TurboQuant 3.5bit (same blocker)

Usage (run from terminal, not Claude Code):
  cd ~/Projects/Work/llm-bench
  nohup caffeinate -i python3 -u run_gemma4_12b_bench.py >> /tmp/gemma4_12b_bench.log 2>&1 &
  echo "PID: $!"
  tail -f /tmp/gemma4_12b_bench.log
"""
import csv, os, subprocess, sys, time, types
from pathlib import Path

BENCH = Path(__file__).parent / "benchmark.py"
sys.path.insert(0, str(Path(__file__).parent))

HARDWARE  = {"name": "Apple M3 Max", "memory_gb": 64, "platform": "darwin-arm64"}
HF_CACHE  = Path.home() / ".cache" / "huggingface" / "hub"
GGUF_DIR  = Path.home() / ".cache" / "llmfit" / "models"
MLX_VLM_PYTHON = "/Users/HIESCHA/.local/share/uv/tools/mlx-vlm/bin/python"
DONE_FILE = Path(__file__).parent / ".gemma4_12b_bench_done"
LOG_FILE  = Path("/tmp/gemma4_12b_bench.log")

HF_GGUF_REPO = "unsloth/gemma-4-12b-it-GGUF"
GGUF_FILES = {
    "Q4_K_M":     "gemma-4-12b-it-Q4_K_M.gguf",
    "UD-Q4_K_XL": "gemma-4-12b-it-UD-Q4_K_XL.gguf",
    "UD-Q2_K_XL": "gemma-4-12b-it-UD-Q2_K_XL.gguf",
    "UD-Q6_K_XL": "gemma-4-12b-it-UD-Q6_K_XL.gguf",
}


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"\n[{ts}] {msg}"
    print(line, flush=True)


def hf_model_cached(repo_id: str) -> bool:
    dir_name = f"models--{repo_id.replace('/', '--')}"
    model_dir = HF_CACHE / dir_name
    if not model_dir.exists():
        return False
    total = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
    return total > 1_000_000_000


def ensure_hf_model(repo_id: str, max_retries: int = 3):
    if hf_model_cached(repo_id):
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


def ensure_gguf(quant: str):
    filename = GGUF_FILES[quant]
    target = GGUF_DIR / filename
    if target.exists() and target.stat().st_size > 1_000_000_000:
        print(f"  ✔ {filename} already exists ({target.stat().st_size/1e9:.1f} GB)", flush=True)
        return True

    log(f"Downloading GGUF: {HF_GGUF_REPO} / {filename}")
    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([
        "hf", "download", HF_GGUF_REPO,
        filename,
        "--local-dir", str(GGUF_DIR),
    ])
    if result.returncode == 0 and target.exists():
        print(f"  ✔ Downloaded {filename} ({target.stat().st_size/1e9:.1f} GB)", flush=True)
        return True
    print(f"  ✘ Failed to download {filename}", flush=True)
    return False


def purge_gguf(quant: str):
    filename = GGUF_FILES[quant]
    target = GGUF_DIR / filename
    if target.exists():
        size_gb = target.stat().st_size / 1e9
        target.unlink()
        print(f"  🗑 Purged {filename} ({size_gb:.1f} GB)", flush=True)


def ensure_ollama_model(tag: str):
    log(f"Pulling Ollama model: {tag}")
    result = subprocess.run(["ollama", "pull", tag])
    if result.returncode == 0:
        print(f"  ✔ Ollama model ready: {tag}", flush=True)
    else:
        print(f"  ⚠ Ollama pull failed for {tag} — test may be skipped", flush=True)


def check_single_instance():
    import psutil
    my_pid = os.getpid()
    for proc in psutil.process_iter(['pid', 'cmdline', 'name']):
        try:
            name = proc.info.get('name', '')
            if 'python' not in name.lower():
                continue
            cmd = " ".join(proc.info['cmdline'] or [])
            if proc.info['pid'] != my_pid and (
                "run_gemma4_12b_bench.py" in cmd or
                ("benchmark.py" in cmd and "--test" in cmd)
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


def run_tests_think(*test_ids: str):
    """Run tests WITHOUT --no-think (thinking mode enabled via no_think: false in config)."""
    log(f"Running (ultrathink): {', '.join(test_ids)}")
    subprocess.run([
        sys.executable, "-u", str(BENCH),
        "--test", *test_ids,
        "--skip-unavailable", "--report-html",
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
                    model=row.get("model", ""),
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

log("Starting Gemma 4 12B benchmark — Group U (full rerun, all backends)")
start_ts = time.strftime("%Y-%m-%d %H:%M:%S")
print(f"  Started: {start_ts}", flush=True)
print(f"  Log:     {LOG_FILE}", flush=True)
print(f"  Backends: mlx-vlm 0.6.1, Ollama 0.30.5, llama-server b9430, oMLX 0.4.1, vllm-mlx, LM Studio", flush=True)
print(f"  Total downloads: ~30 GB HF + ~30 GB GGUF  |  Runtime: ~3-4 h", flush=True)

# ── mlx-vlm baselines ───────────────────────────────────────────────────────

ensure_hf_model("mlx-community/gemma-4-12B-it-4bit")

log("Step 1: U_Q4_1 — mlx-vlm 4bit baseline")
run_tests("U_Q4_1")

log("Step 2: U_Q4_2 — mlx-vlm kv4 baseline")
run_tests("U_Q4_2")

ensure_hf_model("mlx-community/gemma-4-12B-it-8bit")
log("Step 3: U_Q8_1 — mlx-vlm 8bit")
run_tests("U_Q8_1")

ensure_hf_model("mlx-community/gemma-4-12B-it-mxfp4")
log("Step 4: U_MXFP4_1 — mlx-vlm mxfp4")
run_tests("U_MXFP4_1")

ensure_hf_model("mlx-community/gemma-4-12B-it-nvfp4")
log("Step 5: U_NVFP4_1 — mlx-vlm nvfp4")
run_tests("U_NVFP4_1")

# ── Ollama ───────────────────────────────────────────────────────────────────

ensure_ollama_model("gemma4:12b")

log("Step 6: U_Q4_3 — Ollama gemma4:12b")
run_tests("U_Q4_3")

# ── Context scaling (mlx-vlm) ────────────────────────────────────────────────

log("Step 7: U_CTX_1 — mlx-vlm context scaling 32k/64k/128k")
run_tests("U_CTX_1")

log("Step 8: U_CTX_2 — mlx-vlm kv4 context scaling 32k/64k/128k")
run_tests("U_CTX_2")

# ── Context scaling (Ollama) ─────────────────────────────────────────────────

log("Step 9: U_OLL_CTX_1 — Ollama context scaling 32k/64k/128k")
run_tests("U_OLL_CTX_1")

# ── llama-server GGUF quants (download→bench→purge) ──────────────────────────

for step, (quant, test_id) in enumerate([
    ("Q4_K_M",     "U_LC_Q4KM"),
    ("UD-Q4_K_XL", "U_LC_UDQ4"),
    ("UD-Q2_K_XL", "U_LC_UDQ2"),
    ("UD-Q6_K_XL", "U_LC_UDQ6"),
], start=10):
    if ensure_gguf(quant):
        log(f"Step {step}: {test_id} — llama-server {quant}")
        run_tests(test_id)
        purge_gguf(quant)
    else:
        log(f"Step {step}: {test_id} — SKIPPED (download failed)")

# ── vllm-mlx ────────────────────────────────────────────────────────────────

log("Step 14: U_VM_1 — vllm-mlx 4bit")
run_tests("U_VM_1")

# ── oMLX ─────────────────────────────────────────────────────────────────────

log("Step 15: U_OX_1 — oMLX ssd-paged baseline")
run_tests("U_OX_1")

log("Step 16: U_OX_CTX_1 — oMLX context scaling 32k")
run_tests("U_OX_CTX_1")

# ── LM Studio ───────────────────────────────────────────────────────────────

log("Step 17: U_LMS_1 — Gemma4-12B LM Studio")
run_tests("U_LMS_1")

# ── Speculative Decoding (E2B as draft) ─────────────────────────────────────

e2b_draft = GGUF_DIR / "gemma-4-E2B-it-Q4_K_M.gguf"
if not e2b_draft.exists():
    log("Downloading E2B draft model for speculative decoding")
    subprocess.run([
        "hf", "download", "unsloth/gemma-4-e2b-it-GGUF",
        "gemma-4-E2B-it-Q4_K_M.gguf",
        "--local-dir", str(GGUF_DIR),
    ])

if ensure_gguf("UD-Q4_K_XL"):
    log("Step 18: U_SPEC_1 — speculative (E2B draft, UD-Q4_K_XL)")
    run_tests("U_SPEC_1")
else:
    log("Step 18: U_SPEC_1 — SKIPPED (main model GGUF not available)")

if ensure_gguf("Q4_K_M"):
    log("Step 19: U_SPEC_2 — speculative (E2B draft, Q4_K_M)")
    run_tests("U_SPEC_2")
else:
    log("Step 19: U_SPEC_2 — SKIPPED (main model GGUF not available)")

# ── Ultrathink (thinking mode) ──────────────────────────────────────────────

log("Step 20: U_THINK_1 — ultrathink mlx-vlm 4bit")
run_tests_think("U_THINK_1")

log("Step 21: U_THINK_2 — ultrathink mlx-vlm 8bit")
run_tests_think("U_THINK_2")

log("Step 22: U_THINK_3 — ultrathink Ollama")
run_tests_think("U_THINK_3")

if ensure_gguf("UD-Q4_K_XL"):
    log("Step 23: U_THINK_4 — ultrathink llama-server")
    run_tests_think("U_THINK_4")
else:
    log("Step 23: U_THINK_4 — SKIPPED (download failed)")

# Purge remaining GGUFs
for q in GGUF_FILES:
    purge_gguf(q)

# ── Rebuild HTML ─────────────────────────────────────────────────────────────

build_combined_html()

# ── Done ─────────────────────────────────────────────────────────────────────

end_ts = time.strftime("%Y-%m-%d %H:%M:%S")
DONE_FILE.write_text(f"Completed: {start_ts} → {end_ts}\n")
log(f"ALL DONE — Group U (Gemma 4 12B) complete.  {end_ts}")
print(f"""
  Tests run: U_Q4_1/2, U_Q8_1, U_MXFP4_1, U_NVFP4_1
             U_Q4_3, U_CTX_1/2, U_OLL_CTX_1
             U_LC_Q4KM, U_LC_UDQ4, U_LC_UDQ2, U_LC_UDQ6
             U_VM_1, U_OX_1, U_OX_CTX_1, U_LMS_1
             U_SPEC_1/2, U_THINK_1/2/3/4
  Disabled:  U_DF_1, U_MTP_1/2, U_TQ4_1, U_TQ35_1, U_DMR_1
  Results:   ~/Projects/Work/llm-bench/results/complete_results.html
  Done:      {DONE_FILE}
""", flush=True)
