#!/usr/bin/env python3
"""
llama.cpp benchmarks for ALL model groups — Unsloth GGUF quants.

Downloads each GGUF one at a time, runs the benchmark, then deletes it
before downloading the next to conserve disk space (~22 GB peak).

Groups (run order: smallest first):
  N — Gemma4-E2B    (4 tests, ~13 GB cumulative download)
  M — Gemma4-E4B    (4 tests, ~20 GB cumulative download)
  K — Gemma4-26B    (3 tests, ~45 GB cumulative download)
  L — Gemma4-31B    (3 tests, ~48 GB cumulative download)
  D — Qwen3-32B     (3 tests, ~53 GB cumulative download)
  O — Qwen3.6-35B   (3 tests, ~57 GB cumulative download)
  A — Qwen3.5-35B   (3 tests, ~56 GB cumulative download)
  P — Qwen3.6-27B   (4 tests, ~72 GB cumulative download)

Usage:
  cd ~/Projects/Work/llm-bench
  python3 -u run_llamacpp_all.py 2>&1 | tee /tmp/llamacpp_all.log

  # Run only specific groups:
  python3 -u run_llamacpp_all.py --groups N M

  # Skip specific groups:
  python3 -u run_llamacpp_all.py --skip A O

  # Preview without downloading:
  python3 -u run_llamacpp_all.py --dry-run

  # Long unattended run (prevent sleep):
  nohup caffeinate -i python3 -u run_llamacpp_all.py 2>&1 | tee /tmp/llamacpp_all.log &
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

BENCH = Path(__file__).parent / "benchmark.py"
CACHE = Path.home() / ".cache" / "llmfit" / "models"

# ── Group definitions: HF repo + (test_id, gguf_filename) ──────────────────
GROUPS = {
    "N": {
        "label": "Gemma4-E2B (2B Efficient)",
        "hf_repo": "unsloth/gemma-4-e2b-it-GGUF",
        "tests": [
            ("N_LC_Q4KM", "gemma-4-e2b-it-Q4_K_M.gguf"),
            ("N_LC_UDQ4", "gemma-4-e2b-it-UD-Q4_K_XL.gguf"),
            ("N_LC_UDQ2", "gemma-4-e2b-it-UD-Q2_K_XL.gguf"),
            ("N_LC_UDQ6", "gemma-4-e2b-it-UD-Q6_K_XL.gguf"),
        ],
    },
    "M": {
        "label": "Gemma4-E4B (4B Efficient)",
        "hf_repo": "unsloth/gemma-4-e4b-it-GGUF",
        "tests": [
            ("M_LC_Q4KM", "gemma-4-e4b-it-Q4_K_M.gguf"),
            ("M_LC_UDQ4", "gemma-4-e4b-it-UD-Q4_K_XL.gguf"),
            ("M_LC_UDQ2", "gemma-4-e4b-it-UD-Q2_K_XL.gguf"),
            ("M_LC_UDQ6", "gemma-4-e4b-it-UD-Q6_K_XL.gguf"),
        ],
    },
    "K": {
        "label": "Gemma4-26B-A4B (MoE)",
        "hf_repo": "unsloth/gemma-4-26b-a4b-it-GGUF",
        "tests": [
            ("K_LC_Q4KM", "gemma-4-26b-a4b-it-Q4_K_M.gguf"),
            ("K_LC_UDQ4", "gemma-4-26b-a4b-it-UD-Q4_K_XL.gguf"),
            ("K_LC_UDQ2", "gemma-4-26b-a4b-it-UD-Q2_K_XL.gguf"),
        ],
    },
    "L": {
        "label": "Gemma4-31B Dense",
        "hf_repo": "unsloth/gemma-4-31b-it-GGUF",
        "tests": [
            ("L_LC_Q4KM", "gemma-4-31b-it-Q4_K_M.gguf"),
            ("L_LC_UDQ4", "gemma-4-31b-it-UD-Q4_K_XL.gguf"),
            ("L_LC_UDQ2", "gemma-4-31b-it-UD-Q2_K_XL.gguf"),
        ],
    },
    "D": {
        "label": "Qwen3-32B (Dense)",
        "hf_repo": "unsloth/Qwen3-32B-GGUF",
        "tests": [
            ("D_LC_Q4KM", "Qwen3-32B-Q4_K_M.gguf"),
            ("D_LC_UDQ4", "Qwen3-32B-UD-Q4_K_XL.gguf"),
            ("D_LC_UDQ2", "Qwen3-32B-UD-Q2_K_XL.gguf"),
        ],
    },
    "O": {
        "label": "Qwen3.6-35B-A3B (MoE)",
        "hf_repo": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "tests": [
            ("O_LC_Q4KM", "Qwen3.6-35B-A3B-Q4_K_M.gguf"),
            ("O_LC_UDQ4", "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"),
            ("O_LC_UDQ2", "Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf"),
        ],
    },
    "A": {
        "label": "Qwen3.5-35B-A3B (MoE)",
        "hf_repo": "unsloth/Qwen3.5-35B-A3B-GGUF",
        "tests": [
            ("A_LC_Q4KM", "Qwen3.5-35B-A3B-Q4_K_M.gguf"),
            ("A_LC_UDQ4", "Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"),
            ("A_LC_UDQ2", "Qwen3.5-35B-A3B-UD-Q2_K_XL.gguf"),
        ],
    },
    "P": {
        "label": "Qwen3.6-27B (Dense, VLM)",
        "hf_repo": "unsloth/Qwen3.6-27B-GGUF",
        "tests": [
            ("P_LC_Q4KM", "Qwen3.6-27B-Q4_K_M.gguf"),
            ("P_LC_UDQ4", "Qwen3.6-27B-UD-Q4_K_XL.gguf"),
            ("P_LC_UDQ2", "Qwen3.6-27B-UD-Q2_K_XL.gguf"),
            ("P_LC_UDQ6", "Qwen3.6-27B-UD-Q6_K_XL.gguf"),
        ],
    },
}

GROUP_ORDER = ["N", "M", "K", "L", "D", "O", "A", "P"]


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"\n[{ts}] {msg}", flush=True)


def download(hf_repo, hf_filename):
    """Download a single GGUF file from HuggingFace. Returns local path."""
    log(f"Downloading {hf_filename} from {hf_repo}")
    subprocess.run([
        sys.executable, "-m", "huggingface_hub", "download",
        hf_repo, hf_filename,
        "--local-dir", str(CACHE),
    ], check=True)
    dest = CACHE / hf_filename
    if not dest.exists():
        candidates = list(CACHE.rglob(hf_filename))
        if candidates:
            dest = candidates[0]
        else:
            raise FileNotFoundError(f"Could not find {hf_filename} after download")
    log(f"Downloaded: {dest.name} ({dest.stat().st_size / 1e9:.1f} GB)")
    return dest


def bench(test_id):
    """Run benchmark for a single test ID. Returns True on success."""
    log(f"Benchmarking {test_id}")
    result = subprocess.run([
        sys.executable, "-u", str(BENCH),
        "--test", test_id,
        "--no-think", "--report-html",
    ])
    return result.returncode == 0


def purge(filename):
    """Delete GGUF file to free disk space."""
    p = CACHE / filename
    if p.exists():
        size_gb = p.stat().st_size / 1e9
        p.unlink()
        log(f"Purged {p.name} ({size_gb:.1f} GB freed)")


def main():
    parser = argparse.ArgumentParser(description="llama.cpp benchmarks for all model groups")
    parser.add_argument("--groups", nargs="+", metavar="G",
                        help="Run only these groups (e.g., --groups N M K)")
    parser.add_argument("--skip", nargs="+", metavar="G",
                        help="Skip these groups (e.g., --skip A O)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without downloading")
    args = parser.parse_args()

    # Determine which groups to run
    groups = args.groups if args.groups else GROUP_ORDER
    if args.skip:
        groups = [g for g in groups if g not in args.skip]

    # Validate group names
    for g in groups:
        if g not in GROUPS:
            print(f"ERROR: Unknown group '{g}'. Valid: {', '.join(GROUP_ORDER)}")
            sys.exit(1)

    total_tests = sum(len(GROUPS[g]["tests"]) for g in groups)
    log(f"=== llama.cpp benchmark: {len(groups)} groups, {total_tests} tests ===")
    for g in groups:
        info = GROUPS[g]
        print(f"  {g}: {info['label']} — {len(info['tests'])} tests ({info['hf_repo']})")

    if args.dry_run:
        log("Dry run — listing all downloads:")
        for g in groups:
            info = GROUPS[g]
            for test_id, filename in info["tests"]:
                print(f"  {test_id}: {info['hf_repo']}/{filename} → {CACHE / filename}")
        log("Dry run complete. No files downloaded.")
        return

    CACHE.mkdir(parents=True, exist_ok=True)

    results = {}
    t0 = time.time()

    for g in groups:
        info = GROUPS[g]
        log(f"═══ Group {g}: {info['label']} ═══")
        for test_id, filename in info["tests"]:
            log(f"── {test_id}: {filename} ──")
            try:
                download(info["hf_repo"], filename)
                ok = bench(test_id)
                results[test_id] = "✔" if ok else "✘ (bench failed)"
            except Exception as e:
                log(f"ERROR: {e}")
                results[test_id] = f"✘ ({e})"
            finally:
                purge(filename)

    elapsed = time.time() - t0
    log(f"=== Summary ({elapsed / 60:.0f} min total) ===")
    for tid, status in results.items():
        print(f"  {tid}: {status}", flush=True)

    # Rebuild combined HTML report
    log("Rebuilding combined HTML report")
    subprocess.run([sys.executable, str(BENCH.parent / "build_report.py")])
    log("Done.")


if __name__ == "__main__":
    main()
