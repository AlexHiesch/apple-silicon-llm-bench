#!/usr/bin/env python3
"""Patch Harbor's Claude Code install to prefer npm (corp-proxy friendly).

Default Harbor path on Debian/Ubuntu:
  apt install curl procps → curl https://downloads.claude.ai/.../bootstrap.sh

Corp px-proxy often aborts CONNECT to downloads.claude.ai, while
registry.npmjs.org works. Prefer nodejs/npm via apt, then npm install -g.
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
    if MARKER in src:
        return src
    out = src
    out = out.replace(
        "apt-get update && apt-get install -y curl procps;",
        "apt-get update && apt-get install -y curl procps nodejs npm;",
        1,
    )
    # After apt/apk, prefer npm whenever present (not Alpine-only).
    out = out.replace(
        "if command -v apk &> /dev/null; then"
        "  npm install -g @anthropic-ai/claude-code",
        "if command -v npm &> /dev/null; then"
        "  npm install -g @anthropic-ai/claude-code",
        1,
    )
    # Also handle the split-string form in Harbor source.
    out = out.replace(
        '"if command -v apk &> /dev/null; then"\n'
        '                f"  npm install -g @anthropic-ai/claude-code',
        '"if command -v npm &> /dev/null; then"\n'
        '                f"  npm install -g @anthropic-ai/claude-code',
        1,
    )
    if out == src:
        return out
    # Mark so we don't re-apply / can detect success.
    return out.replace(
        "async def install(self, environment: BaseEnvironment) -> None:",
        f"async def install(self, environment: BaseEnvironment) -> None:\n"
        f"        {MARKER}",
        1,
    )


def main() -> int:
    paths = harbor_claude_code_paths()
    if not paths:
        print("harbor claude_code.py not found", file=sys.stderr)
        return 1
    for path in paths:
        original = path.read_text()
        updated = patch_text(original)
        if updated == original:
            print(f"already patched or no match: {path}")
            continue
        path.write_text(updated)
        print(f"patched: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
