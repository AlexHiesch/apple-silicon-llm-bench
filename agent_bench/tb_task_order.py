"""Terminal Bench task ordering helpers (duration / difficulty)."""
from __future__ import annotations

import re
from pathlib import Path

_DIFF_RANK = {"easy": 0, "medium": 1, "hard": 2, "unknown": 3}


def _grab_float(text: str, key: str) -> float | None:
    m = re.search(rf"^{re.escape(key)}\s*=\s*([0-9.]+)", text, re.MULTILINE)
    return float(m.group(1)) if m else None


def _grab_str(text: str, key: str) -> str | None:
    m = re.search(rf'^{re.escape(key)}\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return m.group(1) if m else None


def task_duration_minutes(task_dir: Path) -> tuple[float, str, float | None, float | None]:
    """Return (sort_key_minutes, difficulty, expert_min, junior_min)."""
    toml = task_dir / "task.toml"
    if not toml.is_file():
        return (9999.0, "unknown", None, None)
    text = toml.read_text()
    expert = _grab_float(text, "expert_time_estimate_min")
    junior = _grab_float(text, "junior_time_estimate_min")
    est_sec = _grab_float(text, "estimated_duration_sec")
    agent_sec = None
    in_agent = False
    for line in text.splitlines():
        line = line.strip()
        if line == "[agent]":
            in_agent = True
            continue
        if line.startswith("[") and line.endswith("]"):
            in_agent = False
        if in_agent and line.startswith("timeout_sec"):
            m = re.search(r"=\s*([0-9.]+)", line)
            if m:
                agent_sec = float(m.group(1))
    diff = _grab_str(text, "difficulty") or "unknown"
    if expert is not None:
        key = expert
    elif junior is not None:
        key = junior
    elif est_sec is not None:
        key = est_sec / 60.0
    elif agent_sec is not None:
        key = agent_sec / 60.0
    else:
        key = 9999.0
    return (key, diff, expert, junior)


def list_tb_tasks(dataset_root: Path) -> list[str]:
    return sorted(
        p.name for p in Path(dataset_root).iterdir() if p.is_dir() and (p / "task.toml").is_file()
    )


def tasks_sorted_by_duration(
    dataset_root: Path,
    *,
    observed_sec: dict[str, float] | None = None,
) -> list[str]:
    """Short / easy tasks first (expert estimate, then difficulty, then name).

    ``observed_sec`` is ignored for ordering — partial tech retries must not
    float hard tasks to the front.
    """
    _ = observed_sec
    rows: list[tuple[float, int, str, str]] = []
    for name in list_tb_tasks(dataset_root):
        key, diff, expert, junior = task_duration_minutes(dataset_root / name)
        rows.append((key, _DIFF_RANK.get(diff, 3), name, diff))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    return [r[2] for r in rows]


def format_task_rank_table(dataset_root: Path, limit: int | None = None) -> str:
    lines = ["task,duration_min,difficulty,expert_min,junior_min"]
    order = tasks_sorted_by_duration(dataset_root)
    if limit:
        order = order[:limit]
    for name in order:
        key, diff, expert, junior = task_duration_minutes(dataset_root / name)
        lines.append(f"{name},{key:.1f},{diff},{expert},{junior}")
    return "\n".join(lines) + "\n"
