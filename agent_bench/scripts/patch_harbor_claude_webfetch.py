#!/usr/bin/env python3
"""Harbor Claude Code: skip WebFetch preflight + pin SMALL_FAST model.

Anthropic's WebFetch preflight (`claude.ai/api/web/domain_info`) returns
`can_fetch: false` for some public domains (e.g. ard.de, tagesschau.de,
spiegel.de). Claude Code then fails with "unable to fetch" even when curl
works. Enterprise networks that block claude.ai hit the same path.

Also: when routing through LiteLLM ThinkingCap, WebFetch summarization uses
the Haiku tier. Harbor already remaps ANTHROPIC_DEFAULT_HAIKU_MODEL but not
ANTHROPIC_SMALL_FAST_MODEL (higher priority for the haiku tier).

Idempotent patch of the installed harbor claude_code.py.

IMPORTANT: never put a shell ``#`` comment mid-``&&`` chain inside
``setup_command`` — ``#`` comments out the rest of that shell line (including
``printf``), which surfaces as ``NonZeroAgentExitCodeError`` / exit 2 on every
trial setup.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Appears inside setup_command shell text (safe — not a `#` comment).
MARKER_SETTINGS = "skipWebFetchPreflight"
# Python-only comment in harbor's claude_code.py (not executed as shell).
MARKER_SMALL = "# llm-bench: pin ANTHROPIC_SMALL_FAST_MODEL"

OLD_ALIAS = (
    '            env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = env["ANTHROPIC_MODEL"]\n'
    '            env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = env["ANTHROPIC_MODEL"]\n'
    '            env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = env["ANTHROPIC_MODEL"]\n'
    '            env["CLAUDE_CODE_SUBAGENT_MODEL"] = env["ANTHROPIC_MODEL"]\n'
)
NEW_ALIAS = (
    '            env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = env["ANTHROPIC_MODEL"]\n'
    '            env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = env["ANTHROPIC_MODEL"]\n'
    '            env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = env["ANTHROPIC_MODEL"]\n'
    f"            {MARKER_SMALL}\n"
    '            env["ANTHROPIC_SMALL_FAST_MODEL"] = env["ANTHROPIC_MODEL"]\n'
    '            env["CLAUDE_CODE_SUBAGENT_MODEL"] = env["ANTHROPIC_MODEL"]\n'
)

OLD_SETUP = (
    '        setup_command = (\n'
    '            "mkdir -p $CLAUDE_CONFIG_DIR/debug $CLAUDE_CONFIG_DIR/projects/-app "\n'
    '            "$CLAUDE_CONFIG_DIR/shell-snapshots $CLAUDE_CONFIG_DIR/statsig "\n'
    '            "$CLAUDE_CONFIG_DIR/todos $CLAUDE_CONFIG_DIR/skills && "\n'
    '            "if [ -d ~/.claude/skills ]; then "\n'
    '            "cp -r ~/.claude/skills/. $CLAUDE_CONFIG_DIR/skills/ 2>/dev/null || true; "\n'
    '            "fi"\n'
    "        )\n"
)

# Broken v1: inline shell `#` comment ate printf → exit 2 on every trial.
BROKEN_SETUP = (
    '        setup_command = (\n'
    '            "mkdir -p $CLAUDE_CONFIG_DIR/debug $CLAUDE_CONFIG_DIR/projects/-app "\n'
    '            "$CLAUDE_CONFIG_DIR/shell-snapshots $CLAUDE_CONFIG_DIR/statsig "\n'
    '            "$CLAUDE_CONFIG_DIR/todos $CLAUDE_CONFIG_DIR/skills && "\n'
    '            "# llm-bench: skipWebFetchPreflight settings "\n'
    '            "printf \'%s\\n\' \'{\\"skipWebFetchPreflight\\":true}\' '
    '>$CLAUDE_CONFIG_DIR/settings.json && "\n'
    '            "if [ -d ~/.claude/skills ]; then "\n'
    '            "cp -r ~/.claude/skills/. $CLAUDE_CONFIG_DIR/skills/ 2>/dev/null || true; "\n'
    '            "fi"\n'
    "        )\n"
)

NEW_SETUP = (
    '        setup_command = (\n'
    '            "mkdir -p $CLAUDE_CONFIG_DIR/debug $CLAUDE_CONFIG_DIR/projects/-app "\n'
    '            "$CLAUDE_CONFIG_DIR/shell-snapshots $CLAUDE_CONFIG_DIR/statsig "\n'
    '            "$CLAUDE_CONFIG_DIR/todos $CLAUDE_CONFIG_DIR/skills && "\n'
    '            "printf \'%s\\n\' \'{\\"skipWebFetchPreflight\\":true}\' '
    '>$CLAUDE_CONFIG_DIR/settings.json && "\n'
    '            "if [ -d ~/.claude/skills ]; then "\n'
    '            "cp -r ~/.claude/skills/. $CLAUDE_CONFIG_DIR/skills/ 2>/dev/null || true; "\n'
    '            "fi"\n'
    "        )\n"
)


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
    out = src
    if MARKER_SMALL not in out and OLD_ALIAS in out:
        out = out.replace(OLD_ALIAS, NEW_ALIAS)
    # Repair broken v1 first (marker string differed).
    if BROKEN_SETUP in out:
        out = out.replace(BROKEN_SETUP, NEW_SETUP)
    elif '">$CLAUDE_CONFIG_DIR/settings.json' not in out and OLD_SETUP in out:
        out = out.replace(OLD_SETUP, NEW_SETUP)
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
            has_s = MARKER_SETTINGS in original and "settings.json" in original
            has_broken = "# llm-bench: skipWebFetchPreflight settings" in original
            has_f = MARKER_SMALL in original
            print(
                f"ok/noop: {path} (settings={has_s} broken_v1={has_broken} "
                f"small_fast={has_f})"
            )
            continue
        path.write_text(updated)
        print(f"patched: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
