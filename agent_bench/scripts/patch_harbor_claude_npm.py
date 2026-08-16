#!/usr/bin/env python3
"""Restore Harbor Claude install to bootstrap.sh (corp Z8).

Earlier we tried forcing apt nodejs+npm because downloads.claude.ai failed
when px-proxy blocked Harbor compose nets (172.19+/16). With the px-proxy
allow widened to 172.16.0.0/12, bootstrap completes in ~2 min — while
`apt install nodejs npm` exceeds Harbor's 360s agent-setup timeout.

This script is idempotent: if a previous llm-bench npm patch is present,
revert it; otherwise leave Harbor stock install alone.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

MARKER = "# llm-bench: prefer npm over downloads.claude.ai bootstrap"


def harbor_claude_code_paths() -> list[Path]:
    home = Path.home()
    runner = Path(os.environ.get("RUNNER_HOME", home / "aa-index-runner-home"))
    tool_dir = Path(
        os.environ.get(
            "UV_TOOL_DIR",
            runner / ".local" / "share" / "uv" / "tools",
        )
    )
    candidates = [
        tool_dir
        / "harbor/lib/python3.12/site-packages/harbor/agents/installed/claude_code.py",
        home
        / ".local/share/uv/tools/harbor/lib/python3.12/site-packages"
        / "harbor/agents/installed/claude_code.py",
    ]
    return [p for p in candidates if p.is_file()]


def patch_text(src: str) -> str:
    """Revert prior npm-forcing patch; no-op on stock Harbor."""
    out = src
    if MARKER in out:
        out = out.replace(f"        {MARKER}\n", "")
        out = out.replace(
            "apt-get update && apt-get install -y curl procps nodejs npm;",
            "apt-get update && apt-get install -y curl procps;",
        )
        out = out.replace(
            '"if command -v npm &> /dev/null; then"\n'
            '                f"  npm install -g @anthropic-ai/claude-code',
            '"if command -v apk &> /dev/null; then"\n'
            '                f"  npm install -g @anthropic-ai/claude-code',
        )
        out = out.replace(
            "if command -v npm &> /dev/null; then"
            "  npm install -g @anthropic-ai/claude-code",
            "if command -v apk &> /dev/null; then"
            "  npm install -g @anthropic-ai/claude-code",
        )
    # Also clean accidental nodejs npm without marker
    if "apt-get install -y curl procps nodejs npm;" in out:
        out = out.replace(
            "apt-get update && apt-get install -y curl procps nodejs npm;",
            "apt-get update && apt-get install -y curl procps;",
            1,
        )
    return out


def main() -> int:
    paths = harbor_claude_code_paths()
    if not paths:
        print("harbor claude_code.py not found", file=sys.stderr)
        return 1
    for path in paths:
        original = path.read_text()
        updated = patch_text(original)
        if updated == original:
            print(f"stock/ok: {path}")
            continue
        path.write_text(updated)
        print(f"reverted npm-force patch: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
