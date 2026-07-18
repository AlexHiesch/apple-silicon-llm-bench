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
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

MARKER_SETTINGS = "# llm-bench: skipWebFetchPreflight settings"
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
NEW_SETUP = (
    '        setup_command = (\n'
    '            "mkdir -p $CLAUDE_CONFIG_DIR/debug $CLAUDE_CONFIG_DIR/projects/-app "\n'
    '            "$CLAUDE_CONFIG_DIR/shell-snapshots $CLAUDE_CONFIG_DIR/statsig "\n'
    '            "$CLAUDE_CONFIG_DIR/todos $CLAUDE_CONFIG_DIR/skills && "\n'
    f'            "{MARKER_SETTINGS} "\n'
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
    if MARKER_SETTINGS not in out and OLD_SETUP in out:
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
            has_s = MARKER_SETTINGS in original
            has_f = MARKER_SMALL in original
            print(f"ok/noop: {path} (settings={has_s} small_fast={has_f})")
            continue
        path.write_text(updated)
        print(f"patched: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
