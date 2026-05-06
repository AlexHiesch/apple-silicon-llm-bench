#!/usr/bin/env python3
"""Patch backend_version column into all existing CSVs based on known version history.

Run once to backfill version info. Safe to re-run (idempotent).
"""

import csv
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

# Version mapping: (date_prefix, backend) → version
# Dates are YYYYMMDD from CSV filename; we match the first qualifying rule.
VERSION_RULES = [
    # 2026-04-24/25 llama-server runs used b8920
    (lambda d, b: d >= "20260424" and b == "llama-server", "b8920"),
    # 2026-04-23 Ollama runs: Ollama 0.21.0
    (lambda d, b: d >= "20260423" and b == "ollama", "0.21.0"),
    # 2026-04-23 mlx-lm: 0.31.3
    (lambda d, b: d >= "20260423" and b == "mlx-lm", "0.31.3"),
    (lambda d, b: d >= "20260423" and b == "mlx-lm-turboquant", "0.31.3"),
    (lambda d, b: d >= "20260423" and b == "vllm-mlx", "0.31.3"),
    # backend_retest 2026-04-06: llama-server was b8670
    (lambda d, b: "20260406" in d and b == "llama-server", "b8670"),
    # backend_retest 2026-04-06: mlx-lm-0.31.2 — already versioned in name
    (lambda d, b: b == "mlx-lm-0.31.2", "0.31.2"),
    (lambda d, b: b == "mlx-lm-0.31.3", "0.31.3"),
    (lambda d, b: b == "mlx-vlm-0.4.3", "0.4.3"),
    (lambda d, b: b == "mlx-vlm-0.4.4", "0.4.4"),
    (lambda d, b: b == "omlx-0.3.4", "0.3.4"),
    # 2026-04-03 Ollama: 0.20.0 (gemma4 support added)
    (lambda d, b: d >= "20260403" and d < "20260423" and b == "ollama", "0.20.0"),
    # 2026-04-01/02 early runs
    (lambda d, b: d >= "20260401" and d < "20260403" and b == "ollama", "0.19.0"),
    (lambda d, b: d >= "20260401" and d < "20260406" and b == "mlx-lm", "0.31.2"),
    (lambda d, b: d >= "20260401" and d < "20260406" and b == "mlx-lm-turboquant", "0.31.2"),
    (lambda d, b: d >= "20260401" and d < "20260406" and b == "mlx-vlm", "0.4.3"),
    (lambda d, b: d >= "20260401" and d < "20260406" and b == "llama-server", "b5220"),
    (lambda d, b: d >= "20260401" and d < "20260406" and b == "vllm-mlx", "0.1"),
    (lambda d, b: b == "omlx", "0.3.4"),
    (lambda d, b: b == "lm-studio", ""),
    (lambda d, b: b == "docker-model-runner", ""),
    # tq_bench 2026-04-05/06 with mlx-vlm-0.4.4 already has version in name
]

# Today's run (2026-05-06) — will be applied to any CSV with date >= 20260506
TODAY_VERSIONS = {
    "ollama": "0.23.1",
    "llama-server": "b9020",
    "mlx-lm": "0.31.3",
    "mlx-lm-turboquant": "0.31.3",
    "mlx-vlm": "0.4.4",
    "vllm-mlx": "0.31.3",
    "omlx": "0.3.4",
}


def extract_date(filename: str) -> str:
    """Extract YYYYMMDD from filename like bench_20260402_005026.csv."""
    import re
    m = re.search(r"(\d{8})", filename)
    return m.group(1) if m else ""


def get_version(date: str, backend: str) -> str:
    """Determine backend version from date and backend name."""
    # Today's run takes priority
    if date >= "20260506":
        return TODAY_VERSIONS.get(backend, "")

    for rule_fn, version in VERSION_RULES:
        if rule_fn(date, backend):
            return version
    return ""


def patch_csv(csv_path: Path) -> tuple[int, int]:
    """Add backend_version column to CSV. Returns (total_rows, patched_rows)."""
    date = extract_date(csv_path.name)

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return 0, 0
        has_version = "backend_version" in reader.fieldnames
        rows = list(reader)

    if not rows:
        return 0, 0

    patched = 0
    for r in rows:
        backend = r.get("backend", "")
        existing_ver = r.get("backend_version", "")
        if not existing_ver:
            ver = get_version(date, backend)
            r["backend_version"] = ver
            if ver:
                patched += 1
        else:
            r["backend_version"] = existing_ver

    # Write back with backend_version column
    fieldnames = list(reader.fieldnames)
    if "backend_version" not in fieldnames:
        # Insert after peak_cpu_pct, before tool_call_valid
        if "peak_cpu_pct" in fieldnames:
            idx = fieldnames.index("peak_cpu_pct") + 1
            fieldnames.insert(idx, "backend_version")
        else:
            fieldnames.append("backend_version")

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    return len(rows), patched


def main():
    csvs = sorted(RESULTS_DIR.glob("*.csv"))
    print(f"Patching {len(csvs)} CSV files with backend_version...")

    total_rows = 0
    total_patched = 0
    for csv_path in csvs:
        rows, patched = patch_csv(csv_path)
        total_rows += rows
        total_patched += patched
        if patched:
            print(f"  {csv_path.name}: {patched}/{rows} rows patched")

    print(f"\nDone: {total_patched}/{total_rows} rows patched across {len(csvs)} files")


if __name__ == "__main__":
    main()
