#!/usr/bin/env python3
"""
MTP (Multi-Token Prediction) speculative decoding benchmarks.

Qwen3.6 models ship with trained MTP heads enabling self-speculative decoding
(no separate draft model needed). Requires llama.cpp from the mtp-clean branch.

Usage:
  cd ~/Projects/Work/llm-bench
  python3 -u run_mtp_bench.py 2>&1 | tee /tmp/mtp_bench.log

  # Preview without downloading:
  python3 -u run_mtp_bench.py --dry-run

  # Long unattended run:
  nohup caffeinate -i python3 -u run_mtp_bench.py 2>&1 | tee /tmp/mtp_bench.log &
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

BENCH = Path(__file__).parent / "benchmark.py"
REPORT = Path(__file__).parent / "build_report.py"
CACHE = Path.home() / ".cache" / "llmfit" / "models"
VENDOR = Path(__file__).parent / "vendor"
MTP_DIR = VENDOR / "llama-mtp"
MTP_BINARY = MTP_DIR / "build" / "bin" / "llama-server"
HIESCH_EU = Path.home() / "Projects" / "Private" / "hiesch.eu"

# Models to benchmark (ordered smallest first for disk safety)
MTP_TESTS = [
    # (test_id, hf_repo, hf_filename, size_gb)
    ("R_MTP_P2", "unsloth/Qwen3.6-27B-MTP-GGUF", "Qwen3.6-27B-UD-Q2_K_XL.gguf", 11.0),
    ("R_MTP_P1", "unsloth/Qwen3.6-27B-MTP-GGUF", "Qwen3.6-27B-UD-Q4_K_XL.gguf", 17.9),
    ("R_MTP_O1", "unsloth/Qwen3.6-35B-A3B-MTP-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf", 22.9),
]

# Old files safe to purge for disk space
PURGEABLE = [
    CACHE / "Qwen3-Coder-Next-Q4_K_M.gguf",
]


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"\n[{ts}] {msg}", flush=True)


def disk_free_gb():
    usage = shutil.disk_usage("/")
    return usage.free / (1024**3)


def check_ac_power():
    """Return True if on AC power (or can't determine)."""
    try:
        out = subprocess.check_output(["pmset", "-g", "batt"], text=True)
        return "AC Power" in out
    except Exception:
        return True


def free_disk_space(dry_run=False):
    """Purge known-safe large files to free disk."""
    for p in PURGEABLE:
        if p.exists():
            size_gb = p.stat().st_size / 1e9
            if dry_run:
                print(f"  DRY: would purge {p.name} ({size_gb:.1f} GB)")
            else:
                log(f"Purging {p.name} ({size_gb:.1f} GB) to free disk")
                p.unlink()


def build_mtp_binary():
    """Build llama.cpp from mtp-clean branch. Returns True on success."""
    if MTP_BINARY.exists():
        log(f"MTP binary already exists: {MTP_BINARY}")
        return True

    log("Building llama.cpp MTP branch...")

    if not MTP_DIR.exists():
        log("Cloning am17an/llama.cpp mtp-clean branch...")
        result = subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/am17an/llama.cpp.git",
             "-b", "mtp-clean", str(MTP_DIR)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            log(f"Clone failed: {result.stderr}")
            return False

    build_dir = MTP_DIR / "build"
    ncpu = os.cpu_count() or 4

    log("Running cmake configure...")
    result = subprocess.run(
        ["cmake", "-B", str(build_dir), "-S", str(MTP_DIR),
         "-DCMAKE_BUILD_TYPE=Release", "-DGGML_METAL=ON", "-DBUILD_SHARED_LIBS=OFF"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        log(f"CMake configure failed: {result.stderr[-500:]}")
        return False

    log(f"Building with {ncpu} jobs...")
    result = subprocess.run(
        ["cmake", "--build", str(build_dir), "--config", "Release",
         "-j", str(ncpu), "--target", "llama-server"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        log(f"Build failed: {result.stderr[-500:]}")
        return False

    if MTP_BINARY.exists():
        log(f"Build successful: {MTP_BINARY}")
        return True
    else:
        log("Build completed but binary not found!")
        return False


def verify_mtp_support():
    """Check that the binary actually supports --spec-type mtp."""
    try:
        out = subprocess.check_output(
            [str(MTP_BINARY), "--help"], stderr=subprocess.STDOUT, text=True
        )
        if "draft-mtp" in out:
            log("MTP binary verified: --spec-type draft-mtp supported")
            return True
        else:
            log("WARNING: Binary built but 'mtp' not in --help output")
            return False
    except Exception as e:
        log(f"Verify failed: {e}")
        return False


def download(hf_repo, hf_filename):
    """Download a GGUF from HuggingFace. Returns local path."""
    from huggingface_hub import hf_hub_download

    dest = CACHE / hf_filename
    if dest.exists() and dest.stat().st_size > 1_000_000:
        log(f"Already cached: {hf_filename} ({dest.stat().st_size / 1e9:.1f} GB)")
        return dest

    log(f"Downloading {hf_filename} from {hf_repo} ...")
    local = hf_hub_download(repo_id=hf_repo, filename=hf_filename)
    if not dest.exists():
        dest.symlink_to(local)
    log(f"Downloaded: {dest.name} ({dest.stat().st_size / 1e9:.1f} GB)")
    return dest


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


def bench(test_id):
    """Run benchmark for a single test ID. Returns True on success."""
    log(f"Benchmarking {test_id} ...")
    result = subprocess.run([
        sys.executable, "-u", str(BENCH),
        "--test", test_id,
        "--no-think", "--report-html",
    ])
    return result.returncode == 0


def publish():
    """Rebuild HTML report and push to both repos."""
    project_dir = BENCH.parent

    log("Rebuilding HTML report...")
    subprocess.run([sys.executable, str(REPORT)], cwd=str(project_dir))

    # Push to llm-bench repo
    log("Git: committing results to llm-bench...")
    subprocess.run(["git", "add", "results/", "config.yaml", "benchmark.py",
                    "run_mtp_bench.py"], cwd=str(project_dir))
    ts = time.strftime("%Y-%m-%d %H:%M")
    subprocess.run(["git", "commit", "-m",
                    f"Add MTP speculative decoding benchmarks ({ts})"],
                   cwd=str(project_dir))
    subprocess.run(["git", "push", "origin", "main"], cwd=str(project_dir))

    # Copy to hiesch.eu and push
    if HIESCH_EU.exists():
        src = project_dir / "results" / "complete_results.html"
        dst = HIESCH_EU / "public" / "bench" / "index.html"
        if src.exists() and dst.parent.exists():
            log("Copying results to hiesch.eu...")
            shutil.copy2(str(src), str(dst))
            subprocess.run(["git", "add", "."], cwd=str(HIESCH_EU))
            subprocess.run(["git", "commit", "-m",
                            f"Update bench results — MTP benchmarks ({ts})"],
                           cwd=str(HIESCH_EU))
            subprocess.run(["git", "push"], cwd=str(HIESCH_EU))
            log("Published to hiesch.eu")
    else:
        log(f"WARNING: {HIESCH_EU} not found, skipping website publish")


def main():
    parser = argparse.ArgumentParser(description="MTP speculative decoding benchmarks")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without downloading or running")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip building MTP binary (assume already built)")
    parser.add_argument("--no-publish", action="store_true",
                        help="Skip git commit/push after benchmarks")
    args = parser.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    results = {}
    t0 = time.time()

    # ── Pre-flight checks ──────────────────────────────────────────────────
    log("═══ Pre-flight checks ═══")
    free = disk_free_gb()
    log(f"Disk: {free:.1f} GB free")

    if not check_ac_power():
        log("WARNING: Not on AC power — performance may be throttled!")

    # ── Free disk space ────────────────────────────────────────────────────
    if free < 25:
        log("Disk tight — purging known-safe large files...")
        free_disk_space(dry_run=args.dry_run)
        if not args.dry_run:
            free = disk_free_gb()
            log(f"After purge: {free:.1f} GB free")

    # ── Build MTP binary ───────────────────────────────────────────────────
    if args.dry_run:
        print(f"  DRY: would build MTP binary at {MTP_BINARY}")
    elif not args.skip_build:
        if not build_mtp_binary():
            log("FATAL: Cannot build MTP binary")
            sys.exit(1)
        if not verify_mtp_support():
            log("FATAL: MTP binary does not support --spec-type mtp")
            sys.exit(1)
    else:
        if not MTP_BINARY.exists():
            log("FATAL: --skip-build but binary not found")
            sys.exit(1)

    # ── Run MTP benchmarks ─────────────────────────────────────────────────
    log("═══ Group R: MTP Speculative Decoding ═══")

    for test_id, hf_repo, hf_filename, size_gb in MTP_TESTS:
        log(f"── {test_id}: {hf_filename} ({size_gb:.0f} GB) ──")

        if args.dry_run:
            print(f"  DRY: download {hf_repo}/{hf_filename}")
            print(f"       bench {test_id}")
            print(f"       purge {hf_filename}")
            continue

        # Check disk
        free = disk_free_gb()
        needed = size_gb + 5
        if free < needed:
            log(f"SKIP {test_id}: only {free:.1f} GB free, need {needed:.0f} GB")
            results[test_id] = f"⊘ skipped (disk: {free:.0f} GB < {needed:.0f} GB)"
            continue

        try:
            download(hf_repo, hf_filename)
            ok = bench(test_id)
            results[test_id] = "✔" if ok else "✘ (bench failed)"
        except Exception as e:
            log(f"ERROR: {e}")
            results[test_id] = f"✘ ({e})"
        finally:
            purge(hf_filename)
            free = disk_free_gb()
            log(f"Disk after purge: {free:.1f} GB free")

    # ── Publish ────────────────────────────────────────────────────────────
    if not args.dry_run and not args.no_publish:
        publish()

    # ── Summary ────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    if not args.dry_run:
        log(f"═══ Summary ({elapsed / 60:.0f} min total) ═══")
        for tid, status in results.items():
            print(f"  {tid}: {status}", flush=True)

    log("Done.")


if __name__ == "__main__":
    main()
