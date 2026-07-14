#!/usr/bin/env python3
"""
BFCL evaluation via PROMPTING (no native tool_calls).
Same methodology as heyneo: model generates function call as text, we parse it.
This makes results comparable to heyneo's 63% Q4_K_M score.
"""

import json
import time
import re
import requests
from pathlib import Path

API_BASE = "http://localhost:8090/v1/chat/completions"
DATA_DIR = Path("/tmp/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data")
QUESTIONS_FILE = DATA_DIR / "BFCL_v4_simple_python.json"
ANSWERS_FILE = DATA_DIR / "possible_answer" / "BFCL_v4_simple_python.json"
MAX_SAMPLES = 100


def load_data():
    questions = [json.loads(l) for l in QUESTIONS_FILE.open()]
    answers = {json.loads(l)["id"]: json.loads(l)["ground_truth"] for l in ANSWERS_FILE.open()}
    return questions, answers


def format_functions_as_text(functions: list) -> str:
    """Format function definitions as text prompt (no native tools)."""
    lines = ["You have access to the following functions:\n"]
    for f in functions:
        params = f.get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])
        param_strs = []
        for name, prop in props.items():
            req = " (required)" if name in required else " (optional)"
            desc = prop.get("description", "")
            param_strs.append(f"    - {name} ({prop.get('type', 'any')}){req}: {desc}")
        lines.append(f"Function: {f['name']}")
        lines.append(f"  Description: {f.get('description', '')}")
        lines.append(f"  Parameters:")
        lines.extend(param_strs)
        lines.append("")
    lines.append(
        "To call a function, respond with a JSON object in this exact format:\n"
        '[{"name": "function_name", "arguments": {"param1": value1, "param2": value2}}]\n\n'
        "Respond ONLY with the JSON array. No explanation, no markdown."
    )
    return "\n".join(lines)


def call_model(model: str, messages: list, functions: list) -> str:
    """Call model with prompting (NO tools parameter)."""
    func_prompt = format_functions_as_text(functions)
    system_msg = {"role": "system", "content": func_prompt}
    full_messages = [system_msg] + messages

    payload = {
        "model": model,
        "messages": full_messages,
        "max_tokens": 512,
        "temperature": 0,
    }
    resp = requests.post(API_BASE, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def parse_function_calls(response: str) -> list[dict]:
    """Parse model's text response into function call dicts."""
    response = response.strip()

    # Remove markdown code fences
    if "```json" in response:
        response = response.split("```json")[1].split("```")[0].strip()
    elif "```" in response:
        response = response.split("```")[1].split("```")[0].strip()

    # Try direct JSON parse
    try:
        parsed = json.loads(response)
        if isinstance(parsed, list):
            return [{item["name"]: item.get("arguments", {})} for item in parsed]
        elif isinstance(parsed, dict) and "name" in parsed:
            return [{parsed["name"]: parsed.get("arguments", {})}]
    except json.JSONDecodeError:
        pass

    # Try finding JSON array in response
    match = re.search(r'\[.*\]', response, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return [{item["name"]: item.get("arguments", {})} for item in parsed]
        except json.JSONDecodeError:
            pass

    # Try finding JSON object
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if "name" in parsed:
                return [{parsed["name"]: parsed.get("arguments", {})}]
        except json.JSONDecodeError:
            pass

    return []


def check_answer(predicted: list[dict], ground_truth: list[dict]) -> bool:
    if len(predicted) != len(ground_truth):
        return False

    for pred, gt in zip(predicted, ground_truth):
        if set(pred.keys()) != set(gt.keys()):
            return False
        for func_name in gt:
            if func_name not in pred:
                return False
            pred_args = pred[func_name]
            gt_args = gt[func_name]
            for param, acceptable_values in gt_args.items():
                if param not in pred_args:
                    if "" in acceptable_values or None in acceptable_values:
                        continue
                    return False
                pred_val = pred_args[param]
                matched = False
                for av in acceptable_values:
                    if av == "" and param not in pred_args:
                        matched = True
                        break
                    if type(av) == type(pred_val) and av == pred_val:
                        matched = True
                        break
                    if str(av).lower().strip() == str(pred_val).lower().strip():
                        matched = True
                        break
                if not matched:
                    return False
    return True


def run_eval(model: str, questions: list, answers: dict, max_samples: int = MAX_SAMPLES):
    results = []
    correct = 0
    total = 0
    parse_failures = 0
    t0 = time.time()

    for i, q in enumerate(questions[:max_samples]):
        qid = q["id"]
        messages = q["question"][0]
        functions = q["function"]
        gt = answers.get(qid, [])

        try:
            response = call_model(model, messages, functions)
            predicted = parse_function_calls(response)
            if not predicted:
                parse_failures += 1
            passed = check_answer(predicted, gt)
        except Exception as e:
            predicted = []
            passed = False
            parse_failures += 1

        if passed:
            correct += 1
        total += 1

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{model.split('/')[-1][:20]}] {i+1}/{max_samples} — "
                  f"{correct}/{total} ({100*correct/total:.0f}%) — "
                  f"parse_fail={parse_failures} — {elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    return {
        "model": model,
        "correct": correct,
        "total": total,
        "accuracy": round(100 * correct / total, 1),
        "parse_failures": parse_failures,
        "elapsed_s": round(elapsed, 1),
        "method": "prompting (text generation, no native tool_calls)",
    }


def main():
    questions, answers = load_data()
    print(f"BFCL Prompting Eval — {len(questions)} available, running {MAX_SAMPLES}")
    print("Method: model generates function call as text (same as heyneo)\n")

    from harness_model import DEFAULT_MODEL
    models = [DEFAULT_MODEL]

    all_results = []
    for model in models:
        print(f"=== {model} ===")
        result = run_eval(model, questions, answers)
        all_results.append(result)
        print(f"  SCORE: {result['correct']}/{result['total']} ({result['accuracy']}%) "
              f"in {result['elapsed_s']}s (parse_failures={result['parse_failures']})\n")

    print("=" * 70)
    print(f"{'Model':<45} {'Score':>8} {'Parse Fail':>10} {'Time':>8}")
    print("-" * 70)
    for r in all_results:
        name = r["model"].split("/")[-1][:40]
        print(f"{name:<45} {r['accuracy']:>6.1f}% {r['parse_failures']:>10} {r['elapsed_s']:>6.0f}s")

    print(f"\nMethod: {all_results[0]['method']}")
    print("heyneo reference (Q4_K_M): 63.0%")

    out_path = Path("results/quality/bfcl_prompting_comparison.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
