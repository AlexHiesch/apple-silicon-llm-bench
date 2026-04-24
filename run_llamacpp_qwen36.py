#!/usr/bin/env python3
"""
llama.cpp benchmarks for Qwen3.6-27B — Unsloth Dynamic GGUF quants.

Downloads each GGUF one at a time, runs the benchmark, then deletes it
before downloading the next to conserve disk space (~18 GB peak).

Quants tested:
  Q4_K_M       16.8 GB  — standard 4-bit, direct comparison to Ollama Q4_K_M
  UD-Q4_K_XL   17.6 GB  — Unsloth Dynamic 4-bit (selective high-precision layers)
  UD-Q6_K_XL   25.6 GB  — Unsloth Dynamic 6-bit
  UD-Q2_K_XL   11.8 GB  — Unsloth Dynamic 2-bit (smallest usable)

Usage:
  cd ~/Projects/Work/llm-bench
  python3 -u run_llamacpp_qwen36.py 2>&1 | tee /tmp/llamacpp_qwen36.log
"""
import subprocess, sys, time
from pathlib import Path

BENCH = Path(__file__).parent / "benchmark.py"
CACHE = Path.home() / ".cache" / "llmfit" / "models"
HF_REPO = "unsloth/Qwen3.6-27B-GGUF"

# (test_id, hf_filename, local_filename)
TESTS = [
    ("P_LC_Q4KM", "Qwen3.6-27B-Q4_K_M.gguf",    "Qwen3.6-27B-Q4_K_M.gguf"),
    ("P_LC_UDQ4", "Qwen3.6-27B-UD-Q4_K_XL.gguf", "Qwen3.6-27B-UD-Q4_K_XL.gguf"),
    ("P_LC_UDQ6", "Qwen3.6-27B-UD-Q6_K_XL.gguf", "Qwen3.6-27B-UD-Q6_K_XL.gguf"),
    ("P_LC_UDQ2", "Qwen3.6-27B-UD-Q2_K_XL.gguf", "Qwen3.6-27B-UD-Q2_K_XL.gguf"),
]


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"\n[{ts}] {msg}", flush=True)


def download(hf_filename, local_path):
    """Download a single GGUF file from HuggingFace."""
    from huggingface_hub import hf_hub_download
    log(f"Downloading {hf_filename} ({HF_REPO})")
    cached = hf_hub_download(repo_id=HF_REPO, filename=hf_filename)
    # Symlink to expected location
    expected = CACHE / local_path
    if not expected.exists():
        expected.symlink_to(cached)
    log(f"Downloaded: {expected} ({expected.stat().st_size / 1e9:.1f} GB)")
    return expected


def bench(test_id):
    """Run benchmark for a single test ID."""
    log(f"Benchmarking {test_id}")
    result = subprocess.run([
        sys.executable, "-u", str(BENCH),
        "--test", test_id,
        "--no-think", "--report-html",
    ])
    return result.returncode == 0


def purge(local_path):
    """Delete GGUF file (and HF cache blob) to free disk space."""
    p = CACHE / local_path
    if p.exists() or p.is_symlink():
        if p.is_symlink():
            target = p.resolve()
            size_gb = target.stat().st_size / 1e9 if target.exists() else 0
            p.unlink()
            if target.exists():
                target.unlink()
        else:
            size_gb = p.stat().st_size / 1e9
            p.unlink()
        log(f"Purged {p.name} ({size_gb:.1f} GB freed)")
    else:
        log(f"Nothing to purge: {p}")


def main():
    log("=== llama.cpp Qwen3.6-27B benchmark (download → bench → purge) ===")
    CACHE.mkdir(parents=True, exist_ok=True)

    results = {}
    for test_id, hf_filename, local_filename in TESTS:
        log(f"── {test_id}: {hf_filename} ──")
        try:
            download(hf_filename, local_filename)
            ok = bench(test_id)
            results[test_id] = "✔" if ok else "✘ (bench failed)"
        except Exception as e:
            log(f"ERROR: {e}")
            results[test_id] = f"✘ ({e})"
        finally:
            purge(local_filename)

    log("=== Summary ===")
    for tid, status in results.items():
        print(f"  {tid}: {status}", flush=True)

    # Rebuild combined HTML report
    log("Rebuilding combined HTML report")
    subprocess.run([sys.executable, str(BENCH.parent / "build_report.py")])
    log("Done.")


if __name__ == "__main__":
    main()
