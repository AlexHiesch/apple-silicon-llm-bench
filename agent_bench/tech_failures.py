"""Count AA Index trials that failed for technical (non-content) reasons.

Content failure = no exception, reward 0 (model/harness too weak).
Technical failure = exception like timeout, 403, network, context window —
these must be retried until gone.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

AA_INDEX = Path(__file__).resolve().parent.parent / "results" / "agent_bench" / "aa_index"

TECH_EXCEPTION_TYPES = frozenset({
    "UnknownApiError",
    "AgentTimeoutError",
    "CancelledError",
    "NetworkConnectionError",
    "NonZeroAgentExitCodeError",
    "ContextWindowExceededError",
    "RateLimitError",
    "ApiRateLimitError",  # Harbor's installed-agent classification (≠ RateLimitError)
    "TimeoutError",
    "AgentSetupTimeoutError",
    "EnvironmentStartTimeoutError",
    "VerifierTimeoutError",
    "RuntimeError",  # Harbor wraps docker/compose infra failures as RuntimeError
})

# Message snippets that mean infra/API — even if exception type is odd.
TECH_MESSAGE_MARKERS = (
    "not allowed to access model",
    "ContextWindowExceeded",
    "maximum context length",
    "error setting certificate file",
    "curl: (77)",
    "SSLCertVerificationError",
    "Connection refused",
    "Connection reset",
    "ECONNREFUSED",
    "ECONNRESET",
    "401",
    "403",
    "502 Bad Gateway",
    "503 Service Unavailable",
    "Gateway Timeout",
    "timed out",
    "Timeout",
    "Rate limit",
    "Docker compose command failed",
    "unknown flag: --project-name",
    "docker compose",
)


def _trial_dirs_under(root: Path) -> list[Path]:
    out: list[Path] = []
    if not root.is_dir():
        return out
    for p in root.rglob("result.json"):
        rel_parts = p.relative_to(root).parts
        # Skip smoke / failed junk, but KEEP lock-mismatch archives that hold
        # clean (pass/content_fail) trials — otherwise a resume archive zeroes
        # the scoreboard.
        skip = False
        for part in rel_parts:
            if not part.startswith("_"):
                continue
            if part.startswith("_broken_lock") or part.startswith("_partial"):
                continue
            skip = True
            break
        if skip:
            continue
        if "artifacts" in set(rel_parts):
            continue
        out.append(p.parent)
    return out


def classify_result(result: dict) -> str:
    """Return 'tech' | 'content_fail' | 'pass' | 'pending' | 'other'."""
    ei = result.get("exception_info")
    if ei:
        et = ei.get("exception_type") if isinstance(ei, dict) else str(ei)
        msg = ""
        if isinstance(ei, dict):
            msg = str(ei.get("exception_message") or "")
        if et in TECH_EXCEPTION_TYPES:
            return "tech"
        if any(m.lower() in msg.lower() for m in TECH_MESSAGE_MARKERS):
            return "tech"
        # Unknown exception with HTTP/auth smell
        if re.search(r"\b(401|403|502|503|504)\b", msg):
            return "tech"
        return "other"
    rew = ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    if rew in (1, 1.0, True):
        return "pass"
    if rew in (0, 0.0, False):
        return "content_fail"
    return "other"


def scan_aa_index(root: Path | None = None) -> dict:
    root = root or AA_INDEX
    counts: Counter[str] = Counter()
    tech_examples: list[dict] = []
    by_type: Counter[str] = Counter()

    for trial in _trial_dirs_under(root):
        rj = trial / "result.json"
        if not rj.is_file():
            counts["pending"] += 1
            continue
        try:
            result = json.loads(rj.read_text())
        except Exception:
            counts["tech"] += 1
            by_type["UnreadableResult"] += 1
            continue
        kind = classify_result(result)
        counts[kind] += 1
        if kind == "tech":
            ei = result.get("exception_info") or {}
            et = ei.get("exception_type") if isinstance(ei, dict) else str(ei)
            by_type[str(et or "unknown")] += 1
            if len(tech_examples) < 30:
                tech_examples.append({
                    "trial": str(trial.relative_to(root)),
                    "exception_type": et,
                })

    return {
        "root": str(root),
        "counts": dict(counts),
        "tech_by_type": dict(by_type),
        "tech_examples": tech_examples,
        "tech_total": int(counts.get("tech", 0)),
    }


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    report = scan_aa_index(args.root)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        c = report["counts"]
        print(
            f"tech={report['tech_total']} pass={c.get('pass', 0)} "
            f"content_fail={c.get('content_fail', 0)} "
            f"pending={c.get('pending', 0)} other={c.get('other', 0)}"
        )
        if report["tech_by_type"]:
            print("tech_by_type:", report["tech_by_type"])
        for ex in report["tech_examples"][:10]:
            print(f"  {ex['exception_type']}: {ex['trial']}")


if __name__ == "__main__":
    main()
