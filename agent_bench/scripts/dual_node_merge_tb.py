#!/usr/bin/env python3
"""Merge TB claude-code results from x40 + x39 job dirs; report + retry hints."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from agent_bench.run_harbor import discover_clean_task_names, tb_full_ordered_include_names, SUITE_DATASETS
from agent_bench.tech_failures import classify_result, TECH_EXCEPTION_TYPES

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_X40 = ROOT / "results/agent_bench/aa_index/terminal-bench-v2/claude-code-x40"
DEFAULT_X39 = ROOT / "results/agent_bench/aa_index/terminal-bench-v2/claude-code-x39"
LEGACY_X40 = ROOT / "results/agent_bench/aa_index/terminal-bench-v2/claude-code"
SPLIT = ROOT / "results/agent_bench/aa_index/dual_node_tb_split.json"


def _reward(r: dict) -> float | None:
    vr = r.get("verifier_result") or {}
    rw = vr.get("rewards") if isinstance(vr, dict) else None
    if isinstance(rw, dict) and "reward" in rw:
        try:
            return float(rw["reward"])
        except (TypeError, ValueError):
            return None
    if "reward" in r:
        try:
            return float(r["reward"])
        except (TypeError, ValueError):
            return None
    return None


def _exc(r: dict) -> str | None:
    ei = r.get("exception_info") or {}
    if isinstance(ei, dict):
        return ei.get("exception_type")
    return str(ei) if ei else None


def scan_jobs_dir(out: Path, *, node: str) -> dict[str, dict]:
    rank = {"pass": 0, "content_fail": 1, "tech": 2, "other": 3}
    best: dict[str, dict] = {}
    if not out.is_dir():
        return best
    for job in out.iterdir():
        if not job.is_dir():
            continue
        for trial in job.iterdir():
            if not trial.is_dir() or trial.name.startswith("_"):
                continue
            rj = trial / "result.json"
            if not rj.is_file():
                continue
            try:
                r = json.loads(rj.read_text())
            except Exception:
                continue
            task = r.get("task_name") or trial.name.split("__")[0]
            c = classify_result(r)
            rnk = rank.get(c, 9)
            row = {
                "task": task,
                "class": c,
                "reward": _reward(r),
                "exc": _exc(r),
                "node": node,
                "job": job.name,
                "trial": trial.name,
                "mtime": rj.stat().st_mtime,
            }
            prev = best.get(task)
            if prev is None or rnk < rank.get(prev["class"], 9):
                best[task] = row
            elif rnk == rank.get(prev["class"], 9) and row["mtime"] > prev["mtime"]:
                best[task] = row
    return best


def merged_report(x40: Path, x39: Path) -> dict:
    ds = Path(SUITE_DATASETS["terminal-bench-v2"]["local"])
    all_tasks = sorted(p.name for p in ds.iterdir() if (p / "task.toml").is_file())
    b40 = scan_jobs_dir(x40, node="x40")
    b39 = scan_jobs_dir(x39, node="x39")
    b40_legacy = scan_jobs_dir(LEGACY_X40, node="x40-legacy")
    # merge legacy into b40 candidates per task
    for task, row in b40_legacy.items():
        if task not in b40:
            b40[task] = row
        else:
            rank = {"pass": 0, "content_fail": 1, "tech": 2, "other": 3}
            if rank.get(row["class"], 9) < rank.get(b40[task]["class"], 9):
                b40[task] = row
    rank = {"pass": 0, "content_fail": 1, "tech": 2, "other": 3}
    merged: dict[str, dict] = {}
    for task in all_tasks:
        cands = [b40.get(task), b39.get(task)]
        cands = [c for c in cands if c]
        if not cands:
            merged[task] = {"task": task, "class": "never", "node": "—"}
            continue
        merged[task] = min(cands, key=lambda c: rank.get(c["class"], 9))

    counts = Counter(m["class"] for m in merged.values())
    split = json.loads(SPLIT.read_text()) if SPLIT.is_file() else {}
    remaining = tb_full_ordered_include_names(x40, ds)
    # also credit x39 + legacy cleans
    done = (
        set(discover_clean_task_names(x40))
        | set(discover_clean_task_names(x39))
        | set(discover_clean_task_names(LEGACY_X40))
    )
    # simpler: tasks not pass/content_fail in merged
    open_tasks = [t for t, m in merged.items() if m["class"] not in ("pass", "content_fail")]

    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "counts": dict(counts),
        "pass_rate": round(counts.get("pass", 0) / len(all_tasks), 4),
        "merged": {t: merged[t] for t in all_tasks},
        "open_tasks": open_tasks,
        "split": split,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--x40", type=Path, default=DEFAULT_X40)
    ap.add_argument("--x39", type=Path, default=DEFAULT_X39)
    ap.add_argument("--json-out", type=Path, default=ROOT / "results/agent_bench/aa_index/dual_node_merged.json")
    ap.add_argument("--print", action="store_true")
    args = ap.parse_args()
    rep = merged_report(args.x40, args.x39)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(rep, indent=2) + "\n")
    if args.print:
        c = rep["counts"]
        print(
            f"pass={c.get('pass',0)} content_fail={c.get('content_fail',0)} "
            f"tech={c.get('tech',0)} never={c.get('never',0)} other={c.get('other',0)} "
            f"open={len(rep['open_tasks'])}"
        )
        for t in rep["open_tasks"]:
            m = rep["merged"][t]
            print(f"  {t}: {m['class']} ({m.get('node','?')} exc={m.get('exc','—')})")


if __name__ == "__main__":
    main()
