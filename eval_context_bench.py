#!/usr/bin/env python3
"""
Context-Bench (Standalone) — Multi-hop file retrieval benchmark.
Based on Letta's filesystem-agent benchmark dataset.
Tests agentic context engineering: model must use tools to navigate files
and answer multi-hop questions about synthetic people records.

Uses OpenAI-compatible API with function calling at localhost:8090.
"""

import json
import time
import re
import requests
from pathlib import Path
from typing import Optional

from harness_model import DEFAULT_MODEL

API_BASE = "http://localhost:8090/v1/chat/completions"
MODEL = DEFAULT_MODEL
FILES_DIR = Path("letta-evals/letta-leaderboard/filesystem-agent/files")
DATASET = Path("letta-evals/letta-leaderboard/filesystem-agent/datasets/filesystem_code.jsonl")
MAX_TURNS = 15
MAX_SAMPLES = 10

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file in the data directory. Available files: people.txt, pets.txt, vehicles.txt, credit_cards.txt, bank_accounts.txt, addresses.txt, employments.txt, internet_accounts.txt, insurance_policies.txt, medical_records.txt",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the file to read (e.g., 'people.txt')"
                    }
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "grep_file",
            "description": "Search for a pattern in a file and return matching lines. Case-insensitive.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the file to search"
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Text pattern to search for"
                    }
                },
                "required": ["filename", "pattern"]
            }
        }
    }
]


def execute_tool(name: str, args: dict) -> str:
    if name == "read_file":
        filepath = FILES_DIR / args["filename"]
        if not filepath.exists():
            return f"Error: File '{args['filename']}' not found."
        content = filepath.read_text()
        if len(content) > 8000:
            return content[:8000] + "\n... [truncated]"
        return content
    elif name == "grep_file":
        filepath = FILES_DIR / args["filename"]
        if not filepath.exists():
            return f"Error: File '{args['filename']}' not found."
        pattern = args["pattern"].lower()
        lines = filepath.read_text().splitlines()
        matches = [l for l in lines if pattern in l.lower()]
        if not matches:
            return f"No matches found for '{args['pattern']}' in {args['filename']}"
        result = "\n".join(matches[:50])
        if len(matches) > 50:
            result += f"\n... [{len(matches)-50} more matches]"
        return result
    return f"Error: Unknown tool '{name}'"


def run_agent(question: str, max_turns: int = MAX_TURNS) -> tuple[str, int, float]:
    """Run agent loop. Returns (answer, tool_calls_count, elapsed_seconds)."""
    messages = [
        {"role": "system", "content": "You are a data analyst. Use the available tools to read and search files to answer questions. Be concise in your final answer."},
        {"role": "user", "content": question}
    ]
    tool_calls_count = 0
    t0 = time.time()

    for turn in range(max_turns):
        payload = {
            "model": MODEL,
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 0,
        }
        if turn < max_turns - 1:
            payload["tools"] = TOOLS
        else:
            messages.append({"role": "user", "content": "Answer now with what you know. One line only."})
            payload["max_tokens"] = 512
        try:
            resp = requests.post(API_BASE, json=payload, timeout=300)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return f"ERROR: {e}", tool_calls_count, time.time() - t0

        choice = data["choices"][0]
        msg = choice["message"]

        if turn == max_turns - 1:
            answer = msg.get("content", "")
            return answer, tool_calls_count, time.time() - t0

        if choice.get("finish_reason") == "tool_calls" or msg.get("tool_calls"):
            messages.append(msg)
            for tc in msg["tool_calls"]:
                tool_calls_count += 1
                fn_name = tc["function"]["name"]
                try:
                    fn_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    fn_args = {}
                result = execute_tool(fn_name, fn_args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result
                })
        else:
            answer = msg.get("content", "")
            return answer, tool_calls_count, time.time() - t0

    return "ERROR: max turns reached", tool_calls_count, time.time() - t0


def check_answer(response: str, ground_truth: str) -> bool:
    if not response or not ground_truth:
        return False
    resp_lower = response.lower().strip()
    gt_lower = ground_truth.lower().strip()
    if gt_lower in resp_lower:
        return True
    gt_parts = gt_lower.split()
    if len(gt_parts) > 1 and all(p in resp_lower for p in gt_parts):
        return True
    return False


def main():
    samples = []
    with open(DATASET) as f:
        for line in f:
            samples.append(json.loads(line))

    samples = samples[:MAX_SAMPLES]

    results = []
    print(f"{'#':>3} | {'Pass':>4} | {'Tools':>5} | {'Time':>6} | {'Difficulty':<6} | Answer (first 50 chars)")
    print("-" * 95)

    for i, sample in enumerate(samples, 1):
        question = sample["input"].replace("{pwd}", str(FILES_DIR.resolve()))
        ground_truth = sample["ground_truth"]
        difficulty = sample.get("agent_args", {}).get("extra", {}).get("difficulty", "?")

        answer, tools_used, elapsed = run_agent(question)
        passed = check_answer(answer, ground_truth)

        status = "PASS" if passed else "FAIL"
        short_ans = answer.replace("\n", " ")[:50] if answer else ""
        print(f"{i:>3} | {status:>4} | {tools_used:>5} | {elapsed:>5.1f}s | {difficulty:<6} | {short_ans}")

        results.append({
            "index": i,
            "passed": passed,
            "tools_used": tools_used,
            "elapsed_s": round(elapsed, 1),
            "difficulty": difficulty,
            "ground_truth": ground_truth,
            "answer": answer[:500] if answer else None,
        })

    passed_count = sum(1 for r in results if r["passed"])
    total = len(results)
    avg_tools = sum(r["tools_used"] for r in results) / total if total else 0
    total_time = sum(r["elapsed_s"] for r in results)

    print(f"\n{'='*95}")
    print(f"CONTEXT-BENCH SCORE: {passed_count}/{total} ({100*passed_count/total:.0f}%)")
    print(f"Avg tool calls/question: {avg_tools:.1f}")
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")

    by_diff = {}
    for r in results:
        by_diff.setdefault(r["difficulty"], []).append(r["passed"])
    print("\nBy difficulty:")
    for diff, passes in sorted(by_diff.items()):
        pct = 100 * sum(passes) / len(passes)
        print(f"  {diff:<10}: {sum(passes)}/{len(passes)} ({pct:.0f}%)")

    out_path = Path("results/context_bench.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
