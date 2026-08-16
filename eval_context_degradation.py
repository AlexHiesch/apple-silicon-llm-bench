#!/usr/bin/env python3
"""
Context Degradation Test — Needle in a Haystack (NIAH) variant.
Tests whether the model can retrieve a specific fact embedded at various
positions within contexts of increasing length.

Uses OpenAI-compatible API at localhost:8090.
"""

import json
import time
import random
import string
import requests
from pathlib import Path

from harness_model import DEFAULT_MODEL

API_BASE = "http://localhost:8090/v1/chat/completions"
MODEL = DEFAULT_MODEL

FILLER_PARAGRAPH = (
    "The history of computing is deeply intertwined with advances in mathematics "
    "and engineering. From the earliest mechanical calculators to modern silicon "
    "processors, each generation built upon the insights of the previous one. "
    "Researchers at universities and corporate labs collaborated to push the "
    "boundaries of what was computationally feasible, often driven by practical "
    "needs in science, business, and defense. The development of programming "
    "languages evolved in parallel, from machine code to assembly to high-level "
    "languages that abstracted away hardware details. Operating systems emerged "
    "to manage increasingly complex hardware configurations and provide standard "
    "interfaces for application developers. Networking protocols connected "
    "individual machines into global communications infrastructure.\n\n"
)

NEEDLES = [
    ("The secret project code name is AURORA-7.", "What is the secret project code name?", "AURORA-7"),
    ("The quarterly revenue target is $4.2 million.", "What is the quarterly revenue target?", "4.2 million"),
    ("The password for the test server is mango-sunset-42.", "What is the password for the test server?", "mango-sunset-42"),
    ("The next team offsite is scheduled for March 15th in Portland.", "When and where is the next team offsite?", "March 15"),
    ("The maximum allowed latency for the API is 200 milliseconds.", "What is the maximum allowed latency for the API?", "200"),
]

CONTEXT_LENGTHS = [2000, 4000, 8000, 16000, 32000]
POSITIONS = ["start", "middle", "end"]


def build_context(needle_text: str, target_tokens: int, position: str) -> str:
    filler_tokens_approx = len(FILLER_PARAGRAPH.split()) * 1.3
    num_paragraphs = int(target_tokens / filler_tokens_approx) + 1
    paragraphs = [FILLER_PARAGRAPH] * num_paragraphs

    if position == "start":
        insert_idx = 1
    elif position == "middle":
        insert_idx = len(paragraphs) // 2
    else:
        insert_idx = len(paragraphs) - 2

    paragraphs.insert(insert_idx, f"\n{needle_text}\n\n")
    return "".join(paragraphs)


def query_model(context: str, question: str, max_tokens: int = 1024) -> tuple[str, float]:
    prompt = f"Read the following document carefully:\n\n{context}\n\nBased ONLY on the document above, answer this question concisely: {question}"
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    t0 = time.time()
    resp = requests.post(API_BASE, json=payload, timeout=300)
    elapsed = time.time() - t0
    data = resp.json()
    content = data["choices"][0]["message"].get("content", "")
    return content, elapsed


def check_answer(response: str, expected: str) -> bool:
    return expected.lower() in response.lower()


def main():
    results = []
    print(f"{'Context':>8} | {'Position':>8} | {'Pass':>4} | {'Time':>6} | Answer")
    print("-" * 80)

    for ctx_len in CONTEXT_LENGTHS:
        for position in POSITIONS:
            needle_text, question, expected = random.choice(NEEDLES)
            context = build_context(needle_text, ctx_len, position)
            actual_tokens = len(context.split()) * 1.3

            try:
                response, elapsed = query_model(context, question)
                passed = check_answer(response, expected)
            except Exception as e:
                response = f"ERROR: {e}"
                elapsed = 0
                passed = False

            status = "PASS" if passed else "FAIL"
            short_resp = response.replace("\n", " ")[:60]
            print(f"{ctx_len:>8} | {position:>8} | {status:>4} | {elapsed:>5.1f}s | {short_resp}")
            results.append({
                "context_tokens": ctx_len,
                "position": position,
                "needle": needle_text,
                "question": question,
                "expected": expected,
                "response": response,
                "passed": passed,
                "elapsed_s": round(elapsed, 1),
            })

    passed_count = sum(1 for r in results if r["passed"])
    total = len(results)
    print(f"\n{'='*80}")
    print(f"TOTAL: {passed_count}/{total} passed ({100*passed_count/total:.0f}%)")

    by_length = {}
    for r in results:
        by_length.setdefault(r["context_tokens"], []).append(r["passed"])
    print("\nBy context length:")
    for length, passes in sorted(by_length.items()):
        pct = 100 * sum(passes) / len(passes)
        print(f"  {length:>6} tokens: {sum(passes)}/{len(passes)} ({pct:.0f}%)")

    out_path = Path("results/context_degradation.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
