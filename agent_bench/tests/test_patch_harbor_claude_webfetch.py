"""Unit tests for Harbor Claude WebFetch patch (no Harbor install required)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "agent_bench" / "scripts" / "patch_harbor_claude_webfetch.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "patch_harbor_claude_webfetch", SCRIPT
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_patch_inserts_small_fast_and_settings():
    mod = _load()
    src = (
        "class X:\n"
        "    def run(self):\n"
        + mod.OLD_ALIAS
        + "\n"
        + mod.OLD_SETUP
        + "        return 1\n"
    )
    out = mod.patch_text(src)
    assert mod.MARKER_SMALL in out
    assert mod.MARKER_SETTINGS in out
    assert 'env["ANTHROPIC_SMALL_FAST_MODEL"]' in out
    assert "skipWebFetchPreflight" in out
    assert "settings.json" in out
    # Must not inject a shell `#` mid-&& chain (comments out printf).
    assert "# llm-bench: skipWebFetchPreflight settings" not in out
    assert mod.OLD_ALIAS not in out
    assert mod.OLD_SETUP not in out


def test_patch_repairs_broken_v1_comment():
    mod = _load()
    src = mod.BROKEN_SETUP
    out = mod.patch_text(src)
    assert out == mod.NEW_SETUP
    assert "# llm-bench: skipWebFetchPreflight settings" not in out
    assert "skipWebFetchPreflight" in out


def test_patch_idempotent():
    mod = _load()
    src = mod.NEW_ALIAS + "\n" + mod.NEW_SETUP
    assert mod.patch_text(src) == src
