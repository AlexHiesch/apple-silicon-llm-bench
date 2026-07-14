#!/usr/bin/env python3
"""
HumanEval evaluation via chat API.
Generates Python function completions and evaluates with official test cases.
"""

import json
import time
import signal
import requests
from pathlib import Path
from human_eval.data import read_problems

API_BASE = "http://localhost:8090/v1/chat/completions"
MAX_SAMPLES = 164
TIMEOUT = 10.0


class TimeoutError(Exception):
    pass


def run_with_timeout(code: str, timeout: float) -> bool:
    def handler(signum, frame):
        raise TimeoutError()

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        exec_globals = {}
        exec(code, exec_globals)
        return True
    except TimeoutError:
        return False
    except Exception:
        return False
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def generate_completion(model: str, prompt: str) -> str:
    system = (
        "You are a Python coding assistant. Complete the function below. "
        "Return ONLY the function body (the code that goes after the function signature). "
        "Do not include the function signature, docstring, imports, or explanation. "
        "Do not wrap in markdown code blocks."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Complete this function:\n\n{prompt}"},
        ],
        "max_tokens": 1024,
        "temperature": 0,
        "stop": ["\nclass ", "\ndef ", "\n#", "\nif __name__"],
    }
    resp = requests.post(API_BASE, json=payload, timeout=120)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]

    if "```python" in content:
        content = content.split("```python")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]

    return content


def run_eval(model: str, problems: dict):
    correct = 0
    total = 0
    errors = 0
    t0 = time.time()

    for i, (task_id, problem) in enumerate(problems.items()):
        if i >= MAX_SAMPLES:
            break

        prompt = problem["prompt"]
        test = problem["test"]
        entry_point = problem["entry_point"]

        try:
            completion = generate_completion(model, prompt)
            # Indent completion if it's not already indented (model returns body without indent)
            lines = completion.split("\n")
            if lines and not lines[0].startswith("    ") and not lines[0].startswith("\t"):
                completion = "\n".join("    " + l if l.strip() else l for l in lines)
            full_code = prompt + completion + "\n" + test + f"\ncheck({entry_point})\n"
            passed = run_with_timeout(full_code, TIMEOUT)
        except Exception:
            passed = False
            errors += 1

        if passed:
            correct += 1
        total += 1

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{model.split('/')[-1][:20]}] {i+1}/{min(MAX_SAMPLES, len(problems))} — "
                  f"{correct}/{total} ({100*correct/total:.0f}%) — {elapsed:.0f}s",
                  flush=True)

    elapsed = time.time() - t0
    return {
        "model": model,
        "correct": correct,
        "total": total,
        "accuracy": round(100 * correct / total, 1),
        "errors": errors,
        "elapsed_s": round(elapsed, 1),
    }


def main():
    problems = read_problems()
    print(f"HumanEval: {len(problems)} problems, running {min(MAX_SAMPLES, len(problems))}\n")

    from harness_model import DEFAULT_MODEL
    models = [DEFAULT_MODEL]

    all_results = []
    for model in models:
        print(f"=== {model} ===")
        result = run_eval(model, problems)
        all_results.append(result)
        print(f"  SCORE: {result['correct']}/{result['total']} "
              f"({result['accuracy']}%) in {result['elapsed_s']}s "
              f"({result['errors']} errors)\n")

    print("=" * 70)
    print(f"{'Model':<45} {'pass@1':>8} {'Time':>8}")
    print("-" * 70)
    for r in all_results:
        name = r["model"].split("/")[-1][:40]
        print(f"{name:<45} {r['accuracy']:>6.1f}% {r['elapsed_s']:>6.0f}s")

    print("\nheyneo reference (Q4_K_M): 50.6%")

    out_path = Path("results/quality/humaneval_comparison.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
