#!/usr/bin/env python3
"""
Applied Coding Benchmark — 15 practical coding tasks.
Tests code generation quality via execution. No external judge needed.
Uses OpenAI-compatible API at localhost:8090.
"""

import json
import time
import re
import subprocess
import tempfile
import textwrap
import requests
from pathlib import Path

from harness_model import DEFAULT_MODEL

API_BASE = "http://localhost:8090/v1/chat/completions"
MODEL = DEFAULT_MODEL

TASKS = [
    {
        "id": "fizzbuzz",
        "prompt": "Write a Python function `fizzbuzz(n)` that returns a list of strings for numbers 1 to n. For multiples of 3 use 'Fizz', multiples of 5 use 'Buzz', multiples of both use 'FizzBuzz', otherwise the number as string.",
        "test": "assert fizzbuzz(15)[-1] == 'FizzBuzz'\nassert fizzbuzz(5) == ['1', '2', 'Fizz', '4', 'Buzz']\nassert len(fizzbuzz(100)) == 100",
    },
    {
        "id": "palindrome",
        "prompt": "Write a Python function `is_palindrome(s)` that checks if a string is a palindrome, ignoring case and non-alphanumeric characters.",
        "test": "assert is_palindrome('A man, a plan, a canal: Panama') == True\nassert is_palindrome('race a car') == False\nassert is_palindrome('') == True",
    },
    {
        "id": "two_sum",
        "prompt": "Write a Python function `two_sum(nums, target)` that returns the indices of two numbers that add up to target. Assume exactly one solution exists.",
        "test": "assert sorted(two_sum([2,7,11,15], 9)) == [0,1]\nassert sorted(two_sum([3,2,4], 6)) == [1,2]",
    },
    {
        "id": "flatten",
        "prompt": "Write a Python function `flatten(lst)` that flattens a nested list of arbitrary depth. E.g., flatten([1,[2,[3,4]],5]) returns [1,2,3,4,5].",
        "test": "assert flatten([1,[2,[3,4]],5]) == [1,2,3,4,5]\nassert flatten([[1,2],[3,[4,[5]]]]) == [1,2,3,4,5]\nassert flatten([]) == []",
    },
    {
        "id": "roman_to_int",
        "prompt": "Write a Python function `roman_to_int(s)` that converts a Roman numeral string to an integer. Handle subtraction cases (IV=4, IX=9, etc.).",
        "test": "assert roman_to_int('III') == 3\nassert roman_to_int('LVIII') == 58\nassert roman_to_int('MCMXCIV') == 1994",
    },
    {
        "id": "merge_intervals",
        "prompt": "Write a Python function `merge_intervals(intervals)` that merges overlapping intervals. Input: list of [start, end] pairs. Return merged list sorted by start.",
        "test": "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\nassert merge_intervals([[1,4],[4,5]]) == [[1,5]]",
    },
    {
        "id": "valid_parens",
        "prompt": "Write a Python function `is_valid_parens(s)` that checks if a string of brackets '()[]{}' is valid (properly opened and closed in order).",
        "test": "assert is_valid_parens('()[]{}') == True\nassert is_valid_parens('(]') == False\nassert is_valid_parens('([)]') == False\nassert is_valid_parens('{[]}') == True",
    },
    {
        "id": "lru_cache",
        "prompt": "Write a Python class `LRUCache` with `__init__(self, capacity)`, `get(self, key)` returning -1 if not found, and `put(self, key, value)` that evicts the least recently used item when at capacity.",
        "test": "c = LRUCache(2)\nc.put(1, 1)\nc.put(2, 2)\nassert c.get(1) == 1\nc.put(3, 3)\nassert c.get(2) == -1\nc.put(4, 4)\nassert c.get(1) == -1\nassert c.get(3) == 3",
    },
    {
        "id": "binary_search",
        "prompt": "Write a Python function `binary_search(arr, target)` that returns the index of target in a sorted array, or -1 if not found.",
        "test": "assert binary_search([1,3,5,7,9], 5) == 2\nassert binary_search([1,3,5,7,9], 4) == -1\nassert binary_search([], 1) == -1",
    },
    {
        "id": "word_frequency",
        "prompt": "Write a Python function `word_freq(text)` that returns a dict of word frequencies. Convert to lowercase and split on whitespace. Ignore punctuation attached to words.",
        "test": "r = word_freq('Hello world hello World!')\nassert r.get('hello', 0) == 2\nassert r.get('world', 0) == 2\nassert 'Hello' not in r",
    },
    {
        "id": "matrix_spiral",
        "prompt": "Write a Python function `spiral_order(matrix)` that returns elements of a 2D matrix in spiral order (clockwise from top-left).",
        "test": "assert spiral_order([[1,2,3],[4,5,6],[7,8,9]]) == [1,2,3,6,9,8,7,4,5]\nassert spiral_order([[1,2],[3,4]]) == [1,2,4,3]",
    },
    {
        "id": "longest_common_prefix",
        "prompt": "Write a Python function `longest_common_prefix(strs)` that returns the longest common prefix among a list of strings. Return '' if none.",
        "test": "assert longest_common_prefix(['flower','flow','flight']) == 'fl'\nassert longest_common_prefix(['dog','racecar','car']) == ''\nassert longest_common_prefix(['a']) == 'a'",
    },
    {
        "id": "group_anagrams",
        "prompt": "Write a Python function `group_anagrams(strs)` that groups anagrams together. Return a list of lists (order doesn't matter within or between groups).",
        "test": "result = group_anagrams(['eat','tea','tan','ate','nat','bat'])\nresult_sorted = sorted([sorted(g) for g in result])\nassert result_sorted == [['ate','eat','tea'],['bat'],['nat','tan']]",
    },
    {
        "id": "json_flatten",
        "prompt": "Write a Python function `flatten_json(obj, prefix='')` that flattens a nested JSON dict using dot notation for keys. E.g., {'a': {'b': 1}} becomes {'a.b': 1}. Only flatten dicts, not lists.",
        "test": "assert flatten_json({'a': {'b': 1, 'c': {'d': 2}}}) == {'a.b': 1, 'a.c.d': 2}\nassert flatten_json({'x': [1,2]}) == {'x': [1,2]}",
    },
    {
        "id": "rate_limiter",
        "prompt": "Write a Python class `RateLimiter` with `__init__(self, max_calls, period_seconds)` and `allow(self, timestamp)` that returns True if a call at the given timestamp (float seconds) is within the rate limit. Use a sliding window.",
        "test": "rl = RateLimiter(3, 1.0)\nassert rl.allow(0.0) == True\nassert rl.allow(0.3) == True\nassert rl.allow(0.6) == True\nassert rl.allow(0.9) == False\nassert rl.allow(1.1) == True",
    },
]


def extract_code(response: str) -> str:
    blocks = re.findall(r"```(?:python)?\n?(.*?)```", response, re.DOTALL)
    if blocks:
        return "\n".join(blocks)
    lines = response.split("\n")
    code_lines = [l for l in lines if not l.startswith("Here") and not l.startswith("This")]
    return "\n".join(code_lines)


def run_code_test(code: str, test: str, timeout: int = 10) -> tuple[bool, str]:
    full_code = code + "\n\n" + test
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        f.flush()
        try:
            result = subprocess.run(
                ["python3", f.name],
                capture_output=True, text=True, timeout=timeout
            )
            if result.returncode == 0:
                return True, ""
            return False, result.stderr[-300:]
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT"
        except Exception as e:
            return False, str(e)


def query_model(prompt: str, max_tokens: int = 4096) -> tuple[str, float]:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a Python coding assistant. Write clean, correct code. Only output the code, no explanations."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    t0 = time.time()
    resp = requests.post(API_BASE, json=payload, timeout=300)
    elapsed = time.time() - t0
    data = resp.json()
    content = data["choices"][0]["message"].get("content", "")
    return content, elapsed


def main():
    results = []
    print(f"{'#':>2} | {'Task':<25} | {'Pass':>4} | {'Time':>6} | Error")
    print("-" * 80)

    for i, task in enumerate(TASKS, 1):
        response, elapsed = query_model(task["prompt"])
        code = extract_code(response)
        passed, error = run_code_test(code, task["test"])

        status = "PASS" if passed else "FAIL"
        err_short = error.split("\n")[-1][:40] if error else ""
        print(f"{i:>2} | {task['id']:<25} | {status:>4} | {elapsed:>5.1f}s | {err_short}")

        results.append({
            "id": task["id"],
            "passed": passed,
            "elapsed_s": round(elapsed, 1),
            "error": error[:200] if error else None,
            "response_length": len(response),
        })

    passed_count = sum(1 for r in results if r["passed"])
    total = len(results)
    print(f"\n{'='*80}")
    print(f"CODING SCORE: {passed_count}/{total} ({100*passed_count/total:.0f}%)")
    print(f"Total time: {sum(r['elapsed_s'] for r in results):.0f}s")

    out_path = Path("results/coding_eval.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
