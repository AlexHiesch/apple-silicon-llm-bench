#!/usr/bin/env python3
"""Download new model groups (C/D/E) then run their benchmarks."""
import subprocess, sys, time
from pathlib import Path

HF_PY = "/opt/homebrew/Cellar/mlx-lm/0.31.1/libexec/bin/python"
BENCH  = str(Path(__file__).parent / "benchmark.py")

MODELS = [
    ("mlx-community/gemma-3-27b-it-4bit",             "C — Gemma 3 27B-IT"),
    ("mlx-community/Qwen3-32B-4bit",                  "D — Qwen3-32B (Dense)"),
    ("mlx-community/Llama-3.3-70B-Instruct-4bit",     "E — Llama 3.3 70B"),
]

def download(repo_id: str):
    print(f"\n{'='*60}")
    print(f"  Downloading: {repo_id}")
    print(f"{'='*60}")
    t0 = time.time()
    code = subprocess.run([
        HF_PY, "-c",
        f"from huggingface_hub import snapshot_download; "
        f"snapshot_download('{repo_id}', repo_type='model')"
    ]).returncode
    elapsed = time.time() - t0
    if code != 0:
        print(f"  ERROR downloading {repo_id} (exit {code})")
        return False
    print(f"  Done in {elapsed/60:.1f} min")
    return True

def run_group(group: str):
    print(f"\n{'='*60}")
    print(f"  Benchmarking group: {group}")
    print(f"{'='*60}")
    subprocess.run([
        sys.executable, "-u", BENCH,
        "--group", group,
        "--skip-unavailable", "--no-think", "--report-html",
    ])

for repo, group in MODELS:
    if download(repo):
        run_group(group)
    else:
        print(f"  Skipping benchmark for {group} (download failed)")

print("\nAll groups done.")
