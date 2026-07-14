#!/usr/bin/env python3
"""
Minimal BFCL (Berkeley Function Calling Leaderboard) evaluation.
Tests function calling against localhost:8090 using BFCL v4 simple_python dataset.
Default model: ThinkingCap-Qwen3.6-27B-MLX-4bit (see harness_model.py).
"""

import json
import time
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


def bfcl_func_to_openai_tool(func):
    params = func.get("parameters", {})
    properties = {}
    for name, prop in params.get("properties", {}).items():
        p = dict(prop)
        if p.get("type") == "dict":
            p["type"] = "object"
        properties[name] = p
    return {
        "type": "function",
        "function": {
            "name": func["name"],
            "description": func.get("description", ""),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": params.get("required", []),
            }
        }
    }


def call_model(model: str, messages: list, tools: list) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "max_tokens": 512,
        "temperature": 0,
    }
    resp = requests.post(API_BASE, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def extract_tool_calls(response: dict) -> list[dict]:
    msg = response["choices"][0]["message"]
    if not msg.get("tool_calls"):
        content = msg.get("content", "")
        try:
            import re
            matches = re.findall(r'<tool_call>\s*({.*?})\s*</tool_call>', content, re.DOTALL)
            if matches:
                calls = []
                for m in matches:
                    parsed = json.loads(m)
                    calls.append({parsed["name"]: parsed.get("arguments", {})})
                return calls
        except:
            pass
        return []

    calls = []
    for tc in msg["tool_calls"]:
        fn_name = tc["function"]["name"]
        try:
            fn_args = json.loads(tc["function"]["arguments"])
        except json.JSONDecodeError:
            fn_args = {}
        calls.append({fn_name: fn_args})
    return calls


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
    t0 = time.time()

    for i, q in enumerate(questions[:max_samples]):
        qid = q["id"]
        messages = q["question"][0]
        tools = [bfcl_func_to_openai_tool(f) for f in q["function"]]
        gt = answers.get(qid, [])

        try:
            response = call_model(model, messages, tools)
            predicted = extract_tool_calls(response)
            passed = check_answer(predicted, gt)
        except Exception as e:
            predicted = []
            passed = False

        if passed:
            correct += 1
        total += 1

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{model.split('/')[-1][:20]}] {i+1}/{max_samples} — {correct}/{total} ({100*correct/total:.0f}%) — {elapsed:.0f}s")

    elapsed = time.time() - t0
    return {
        "model": model,
        "correct": correct,
        "total": total,
        "accuracy": round(100 * correct / total, 1),
        "elapsed_s": round(elapsed, 1),
    }


def main():
    questions, answers = load_data()
    print(f"Loaded {len(questions)} BFCL simple_python questions")
    print(f"Running {MAX_SAMPLES} samples per model\n")

    from harness_model import DEFAULT_MODEL
    models = [DEFAULT_MODEL]

    all_results = []
    for model in models:
        print(f"=== {model} ===")
        result = run_eval(model, questions, answers)
        all_results.append(result)
        print(f"  SCORE: {result['correct']}/{result['total']} ({result['accuracy']}%) in {result['elapsed_s']}s\n")

    print("=" * 60)
    print(f"{'Model':<45} {'Score':>8} {'Time':>8}")
    print("-" * 60)
    for r in all_results:
        name = r["model"].split("/")[-1][:40]
        print(f"{name:<45} {r['accuracy']:>6.1f}% {r['elapsed_s']:>6.0f}s")

    out_path = Path("results/quality/bfcl_comparison.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
