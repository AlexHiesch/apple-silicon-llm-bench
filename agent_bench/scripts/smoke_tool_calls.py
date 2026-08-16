#!/usr/bin/env python3
"""Tool-call smoke battery against LiteLLM Anthropic /v1/messages.

Scores how often ThinkingCap returns a real tool_use block vs text/XML junk.
Exit 0 always (report is the product); use --min-rate to fail CI-style.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path


def post_messages(url: str, key: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def classify(content: list | None) -> str:
    if not content:
        return "empty"
    types = [c.get("type") for c in content if isinstance(c, dict)]
    if "tool_use" in types:
        return "tool_use"
    texts = [
        c.get("text") or ""
        for c in content
        if isinstance(c, dict) and c.get("type") == "text"
    ]
    blob = "\n".join(texts)
    if "<Bash>" in blob or "<Read" in blob or "invoke>" in blob.lower():
        return "xml_text"
    if "Bash</" in blob or "</anth" in blob:
        return "garbled"
    if blob.strip():
        return "text_only"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--url", default=os.environ.get("LITELLM_URL", "http://127.0.0.1:4000/v1/messages"))
    ap.add_argument("--model", default=os.environ.get("SMOKE_MODEL", "thinkingcap"))
    ap.add_argument("--key-file", default="")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--label", default="smoke")
    ap.add_argument("--out", default="")
    ap.add_argument("--min-rate", type=float, default=-1.0)
    args = ap.parse_args()

    key = os.environ.get("LITELLM_KEY") or os.environ.get("ANTHROPIC_API_KEY") or ""
    if not key and args.key_file:
        key = Path(args.key_file).read_text().strip()
    if not key:
        for cand in (
            Path.home() / "llm-serving" / "aa-index-key",
            Path("results/agent_bench/aa_index/aa-index-key"),
        ):
            if cand.is_file():
                key = cand.read_text().strip()
                break
    if not key:
        print("ERROR: no API key", file=sys.stderr)
        return 2

    # Minimal set + optional Claude-Code-shaped stress (many tools, agent voice).
    tools = [
        {
            "name": "Bash",
            "description": "Run a shell command and return stdout/stderr.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["command"],
            },
        }
    ]
    if os.environ.get("SMOKE_CLAUDE_LIKE", "1") not in ("0", "false", "no"):
        for name, desc in (
            ("Read", "Read a file from disk"),
            ("Write", "Write a file to disk"),
            ("Edit", "Edit a file in place"),
            ("Glob", "Find files by glob"),
            ("Grep", "Search file contents"),
            ("WebFetch", "Fetch a URL"),
            ("WebSearch", "Web search"),
            ("Task", "Launch a sub-agent"),
        ):
            tools.append(
                {
                    "name": name,
                    "description": desc,
                    "input_schema": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}, "pattern": {"type": "string"}},
                        "additionalProperties": True,
                    },
                }
            )
    prompts = [
        "Use the Bash tool to run: echo hello. Do not describe the tool in text; call it.",
        "Call Bash with command `pwd`. No prose — tool call only.",
        "You must invoke Bash to run `uname -a`. Prefer a tool_use block.",
        "Tool-call required: Bash command=`date -Iseconds`.",
        "Execute via Bash tool: `echo TOOL_OK`. Do not answer in plain text.",
        # Stress: agent-style instruction that previously produced XML-as-text.
        (
            "You are Claude Code in /app. Start by examining files. "
            "Call tools via the API tool_use mechanism only — never emit "
            "angle-bracket XML like <Bash> in assistant text. "
            "First action: Bash command=`ls -la /app/`."
        ),
    ]

    counts: Counter[str] = Counter()
    rows: list[dict] = []
    t0 = time.time()
    for i in range(args.n):
        prompt = prompts[i % len(prompts)]
        body = {
            "model": args.model,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "tools": tools,
            "messages": [{"role": "user", "content": prompt}],
        }
        row: dict = {"i": i, "prompt": prompt}
        try:
            data = post_messages(args.url, key, body, args.timeout)
            content = data.get("content")
            kind = classify(content if isinstance(content, list) else None)
            row.update(
                {
                    "ok": True,
                    "kind": kind,
                    "stop_reason": data.get("stop_reason"),
                    "usage": data.get("usage"),
                    "content_snip": json.dumps(content)[:400] if content is not None else None,
                }
            )
        except Exception as e:
            kind = "error"
            row.update({"ok": False, "kind": kind, "error": f"{type(e).__name__}: {e}"})
        counts[kind] += 1
        rows.append(row)
        print(f"[{args.label}] {i+1}/{args.n} {kind}", flush=True)

    n_ok_tool = counts["tool_use"]
    rate = n_ok_tool / args.n if args.n else 0.0
    summary = {
        "label": args.label,
        "n": args.n,
        "tool_use": n_ok_tool,
        "tool_use_rate": round(rate, 4),
        "counts": dict(counts),
        "elapsed_s": round(time.time() - t0, 1),
        "model": args.model,
        "url": args.url,
        "temperature": args.temperature,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n")
        print(f"wrote {out}", flush=True)

    if args.min_rate >= 0 and rate < args.min_rate:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
