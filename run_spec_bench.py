#!/usr/bin/env python3
"""
llama.cpp speculative decoding benchmarks (Group Q).

Downloads Gemma4-E2B as a persistent draft model, then runs each 26B variant
as main model with E2B as draft. Purges 26B GGUFs after each test; purges E2B
at the very end.

Usage:
  cd ~/Projects/Work/llm-bench
  python3 -u run_spec_bench.py 2>&1 | tee /tmp/spec_bench.log

  # Also include group N (E2B standalone) and K (26B standalone):
  python3 -u run_spec_bench.py --all

  # Preview without downloading:
  python3 -u run_spec_bench.py --dry-run

  # Long unattended run:
  nohup caffeinate -i python3 -u run_spec_bench.py --all 2>&1 | tee /tmp/spec_bench.log &
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

BENCH = Path(__file__).parent / "benchmark.py"
CACHE = Path.home() / ".cache" / "llmfit" / "models"

E2B_REPO = "unsloth/gemma-4-E2B-it-GGUF"
K26B_REPO = "unsloth/gemma-4-26B-A4B-it-GGUF"

# Draft model (kept persistent throughout spec bench run)
DRAFT_FILE = "gemma-4-E2B-it-Q4_K_M.gguf"

# Speculative decoding test pairs: (test_id, main_gguf, main_repo)
SPEC_TESTS = [
    ("Q_LC_SPEC1", "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",  K26B_REPO),
    ("Q_LC_SPEC2", "gemma-4-26B-A4B-it-UD-Q2_K_XL.gguf", K26B_REPO),
]

# Group N: E2B standalone tests (run before spec tests to reuse download)
N_TESTS = [
    ("N_LC_Q4KM", "gemma-4-E2B-it-Q4_K_M.gguf",      E2B_REPO),
    ("N_LC_UDQ4", "gemma-4-E2B-it-UD-Q4_K_XL.gguf",   E2B_REPO),
    ("N_LC_UDQ2", "gemma-4-E2B-it-UD-Q2_K_XL.gguf",   E2B_REPO),
    ("N_LC_UDQ6", "gemma-4-E2B-it-UD-Q6_K_XL.gguf",   E2B_REPO),
]

# Group K: 26B standalone tests
K_TESTS = [
    ("K_LC_Q4KM", "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",  K26B_REPO),
    ("K_LC_UDQ4", "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf", K26B_REPO),
    ("K_LC_UDQ2", "gemma-4-26B-A4B-it-UD-Q2_K_XL.gguf", K26B_REPO),
]


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"\n[{ts}] {msg}", flush=True)


def download(hf_repo, hf_filename):
    """Download a GGUF from HuggingFace. Returns local path."""
    from huggingface_hub import hf_hub_download
    dest = CACHE / hf_filename
    if dest.exists() and dest.stat().st_size > 1_000_000:
        log(f"Already cached: {hf_filename} ({dest.stat().st_size / 1e9:.1f} GB)")
        return dest
    log(f"Downloading {hf_filename} from {hf_repo}")
    cached = hf_hub_download(repo_id=hf_repo, filename=hf_filename)
    if not dest.exists():
        dest.symlink_to(cached)
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
    """Delete GGUF file (and HF cache blob) to free disk."""
    p = CACHE / filename
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
        log(f"Purged {filename} ({size_gb:.1f} GB freed)")


def run_standalone_tests(tests, dry_run=False):
    """Run a list of (test_id, filename, repo) tests, purging each GGUF after use."""
    results = {}
    for test_id, filename, repo in tests:
        log(f"── {test_id}: {filename} ──")
        if dry_run:
            print(f"  DRY: download {repo}/{filename}, bench {test_id}, purge")
            continue
        try:
            download(repo, filename)
            ok = bench(test_id)
            results[test_id] = "✔" if ok else "✘ (bench failed)"
        except Exception as e:
            log(f"ERROR: {e}")
            results[test_id] = f"✘ ({e})"
        finally:
            purge(filename)
    return results


def main():
    parser = argparse.ArgumentParser(description="llama.cpp speculative decoding benchmarks")
    parser.add_argument("--all", action="store_true",
                        help="Also run Group N (E2B standalone) and Group K (26B standalone)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without downloading")
    args = parser.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    results = {}
    t0 = time.time()

    # ── Group N: E2B standalone (optional) ──────────────────────────────────
    if args.all:
        log("═══ Group N: Gemma4-E2B standalone ═══")
        if args.dry_run:
            for test_id, filename, repo in N_TESTS:
                print(f"  DRY: download {repo}/{filename}, bench {test_id}, purge")
        else:
            results.update(run_standalone_tests(N_TESTS))

    # ── Speculative decoding: download E2B draft once, keep throughout ──────
    log("═══ Group Q: Speculative Decoding (26B main + E2B draft) ═══")
    if args.dry_run:
        print(f"  DRY: download E2B draft: {E2B_REPO}/{DRAFT_FILE} (KEEP)")
        for test_id, main_file, main_repo in SPEC_TESTS:
            print(f"  DRY: download {main_repo}/{main_file}")
            print(f"       bench {test_id} (draft={DRAFT_FILE})")
            print(f"       purge main GGUF")
        print(f"  DRY: purge draft {DRAFT_FILE}")
    else:
        # Download draft model (keep until all spec tests are done)
        try:
            download(E2B_REPO, DRAFT_FILE)
        except Exception as e:
            log(f"FATAL: Failed to download draft model: {e}")
            sys.exit(1)

        for test_id, main_file, main_repo in SPEC_TESTS:
            log(f"── {test_id}: {main_file} ──")
            try:
                download(main_repo, main_file)
                ok = bench(test_id)
                results[test_id] = "✔" if ok else "✘ (bench failed)"
            except Exception as e:
                log(f"ERROR: {e}")
                results[test_id] = f"✘ ({e})"
            finally:
                purge(main_file)

        # Purge draft model after all spec tests
        purge(DRAFT_FILE)

    # ── Group K: 26B standalone (optional) ──────────────────────────────────
    if args.all:
        log("═══ Group K: Gemma4-26B standalone ═══")
        if not args.dry_run:
            results.update(run_standalone_tests(K_TESTS))
        else:
            for test_id, filename, repo in K_TESTS:
                print(f"  DRY: download {repo}/{filename}, bench {test_id}, purge")

    elapsed = time.time() - t0
    if not args.dry_run:
        log(f"=== Summary ({elapsed / 60:.0f} min total) ===")
        for tid, status in results.items():
            print(f"  {tid}: {status}", flush=True)

        log("Rebuilding combined HTML report")
        subprocess.run([sys.executable, str(BENCH.parent / "build_report.py")])

    log("Done.")


if __name__ == "__main__":
    main()
