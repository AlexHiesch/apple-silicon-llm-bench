#!/usr/bin/env python3
"""
ThinkingCap-Qwen3.6-27B thinking token efficiency benchmark.

Mirrors BottlecapAI's methodology:
- Thinking ENABLED (no --chat-template-args disable)
- Sampling: temperature=1.0, top_p=0.95, top_k=20
- Tracks: accuracy, thinking_tokens, total_tokens per sample
- Multiple benchmarks: GSM8K, ARC-Challenge, MMLU-Pro, HumanEval (subset)

Default model: ThinkingCap-Qwen3.6-27B-MLX-4bit (see harness_model.py).
"""

import json
import time
import re
import random
import signal
import subprocess
import sys
import os
import requests
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

API_BASE = "http://localhost:8090/v1/chat/completions"
RESULTS_DIR = Path("results/quality/thinking_tokens")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

from harness_model import DEFAULT_MODEL

MODELS = [DEFAULT_MODEL]

SEED = 42
NUM_SEEDS = 1  # BottlecapAI uses 5; we use 1 for time (can increase later)

# Benchmark sizes — tuned for practical runtime on Apple Silicon (~30 tok/s)
# With thinking enabled, each sample takes 30-120s → keep total manageable
GSM8K_SAMPLES = 100
ARC_CHALLENGE_SAMPLES = 100
MMLU_PRO_SAMPLES = 100
HUMANEVAL_SAMPLES = 80


@dataclass
class SampleResult:
    benchmark: str
    sample_id: str
    model: str
    correct: bool
    thinking_tokens: int
    completion_tokens: int
    total_tokens: int
    answer: str
    ground_truth: str
    elapsed_s: float


def wait_for_server(timeout=300):
    """Wait until mlx_lm server is responsive."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get("http://localhost:8090/v1/models", timeout=5)
            if r.status_code == 200:
                return True
        except:
            pass
        time.sleep(3)
    return False


def start_server(model_path: str):
    """Start mlx_lm.server with thinking ENABLED."""
    print(f"\n{'='*70}")
    print(f"Starting server for: {model_path}")
    print(f"{'='*70}")

    # Kill ALL existing mlx_lm servers aggressively
    subprocess.run(["pkill", "-9", "-f", "mlx_lm.server"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "mlx_lm"], capture_output=True)
    time.sleep(5)

    # Verify port is free
    import socket
    for _ in range(10):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("", 8090))
            s.close()
            break
        except OSError:
            time.sleep(2)

    cmd = [
        sys.executable, "-m", "mlx_lm.server",
        "--model", model_path,
        "--port", "8090",
        "--log-level", "WARNING",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    print(f"  Waiting for server (PID {proc.pid})...", flush=True)
    if not wait_for_server():
        proc.kill()
        raise RuntimeError(f"Server failed to start for {model_path}")

    # Quick sanity check: send a tiny request to confirm model is loaded
    try:
        test_resp = requests.post(API_BASE, json={
            "model": model_path,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10,
        }, timeout=60)
        test_resp.raise_for_status()
        print(f"  Server ready! (model loaded, inference confirmed)", flush=True)
    except Exception as e:
        print(f"  Server ready! (warning: sanity check failed: {e})", flush=True)
    return proc


def stop_server(proc):
    """Stop mlx_lm server."""
    if proc:
        proc.terminate()
        proc.wait(timeout=30)
    os.system("pkill -f 'mlx_lm.server' 2>/dev/null")
    time.sleep(3)


def call_model(model: str, messages: list, max_tokens: int = 15000) -> dict:
    """Call model with thinking enabled, return full response with token counts."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 0.95,
    }

    # Retry once on timeout
    for attempt in range(2):
        try:
            resp = requests.post(API_BASE, json=payload, timeout=600)
            resp.raise_for_status()
            break
        except requests.exceptions.Timeout:
            if attempt == 0:
                print(f"      [timeout, retrying with max_tokens={max_tokens//2}]", flush=True)
                payload["max_tokens"] = max_tokens // 2
            else:
                raise
    data = resp.json()

    choice = data["choices"][0]
    message = choice["message"]
    usage = data.get("usage", {})

    content = message.get("content", "") or ""
    thinking = ""

    # mlx_lm.server puts thinking in "reasoning" field
    if "reasoning" in message and message["reasoning"]:
        thinking = message["reasoning"]
    elif "reasoning_content" in message and message["reasoning_content"]:
        thinking = message["reasoning_content"]

    # Fallback: extract from <think> tags in content
    if not thinking:
        think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if think_match:
            thinking = think_match.group(1)
            content = content[think_match.end():].strip()

    # Strip leading newlines from content (mlx_lm adds \n\n before content)
    content = content.lstrip("\n")

    # Token counting: completion_tokens from API includes thinking + content
    completion_tokens = usage.get("completion_tokens", 0)
    # Estimate thinking vs content tokens by character ratio (best we can do)
    total_chars = len(thinking) + len(content)
    if thinking and total_chars > 0:
        thinking_tokens = int(completion_tokens * len(thinking) / total_chars)
    else:
        thinking_tokens = 0
    content_tokens = completion_tokens - thinking_tokens

    return {
        "content": content,
        "thinking": thinking,
        "thinking_tokens": thinking_tokens,
        "content_tokens": content_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": usage.get("total_tokens", 0),
        "prompt_tokens": usage.get("prompt_tokens", 0),
    }


# ============================================================
# GSM8K
# ============================================================

def load_gsm8k(max_samples=GSM8K_SAMPLES):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    random.seed(SEED)
    indices = random.sample(range(len(ds)), min(max_samples, len(ds)))
    samples = []
    for i in indices:
        row = ds[i]
        answer_text = row["answer"].split("####")[-1].strip()
        samples.append({
            "id": f"gsm8k_{i}",
            "question": row["question"],
            "answer": answer_text,
        })
    return samples


def extract_number(text: str) -> Optional[str]:
    """Extract final number from model response."""
    # Look for boxed answer first
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return boxed[-1].replace(",", "").strip()

    # Look for "the answer is X" pattern
    answer_match = re.search(r"(?:the answer is|answer:|final answer:?)\s*\$?([0-9,.-]+)", text, re.I)
    if answer_match:
        return answer_match.group(1).replace(",", "").replace("$", "").strip()

    # Last number in text
    numbers = re.findall(r"-?[0-9,]+\.?[0-9]*", text)
    if numbers:
        return numbers[-1].replace(",", "").strip()
    return None


def eval_gsm8k(model: str, samples: list) -> list[SampleResult]:
    results = []
    correct = 0
    t0 = time.time()

    for i, sample in enumerate(samples):
        st = time.time()
        messages = [
            {"role": "user", "content": f"Solve this math problem step by step. Give your final numerical answer after '####'.\n\n{sample['question']}"}
        ]

        try:
            resp = call_model(model, messages, max_tokens=4096)
            predicted = extract_number(resp["content"])
            gt = sample["answer"].replace(",", "").strip()
            is_correct = predicted is not None and predicted == gt
        except Exception as e:
            resp = {"content": "", "thinking": "", "thinking_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            predicted = None
            is_correct = False

        if is_correct:
            correct += 1

        results.append(SampleResult(
            benchmark="gsm8k",
            sample_id=sample["id"],
            model=model,
            correct=is_correct,
            thinking_tokens=resp["thinking_tokens"],
            completion_tokens=resp["completion_tokens"],
            total_tokens=resp["total_tokens"],
            answer=str(predicted),
            ground_truth=sample["answer"],
            elapsed_s=round(time.time() - st, 2),
        ))

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            avg_think = sum(r.thinking_tokens for r in results[-10:]) / 10
            print(f"    GSM8K {i+1}/{len(samples)} — {correct}/{i+1} ({100*correct/(i+1):.0f}%) — "
                  f"avg_think={avg_think:.0f} tok — {elapsed:.0f}s", flush=True)

    return results


# ============================================================
# ARC-Challenge
# ============================================================

def load_arc_challenge(max_samples=ARC_CHALLENGE_SAMPLES):
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    random.seed(SEED)
    indices = random.sample(range(len(ds)), min(max_samples, len(ds)))
    samples = []
    for i in indices:
        row = ds[i]
        choices = row["choices"]
        labels = choices["label"]
        texts = choices["text"]
        options = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
        samples.append({
            "id": f"arc_{i}",
            "question": row["question"],
            "options": options,
            "labels": labels,
            "answer": row["answerKey"],
        })
    return samples


def eval_arc_challenge(model: str, samples: list) -> list[SampleResult]:
    results = []
    correct = 0
    t0 = time.time()

    for i, sample in enumerate(samples):
        st = time.time()
        messages = [
            {"role": "user", "content": f"Answer the following multiple choice question. Reply with ONLY the letter of the correct answer.\n\n{sample['question']}\n\n{sample['options']}"}
        ]

        try:
            resp = call_model(model, messages, max_tokens=2048)
            content = resp["content"].strip()
            # Extract letter answer
            predicted = None
            for label in sample["labels"]:
                if re.search(rf'\b{label}\b', content[:20]):
                    predicted = label
                    break
            if not predicted and content and content[0].upper() in sample["labels"]:
                predicted = content[0].upper()
            is_correct = predicted == sample["answer"]
        except Exception as e:
            resp = {"content": "", "thinking": "", "thinking_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            predicted = None
            is_correct = False

        if is_correct:
            correct += 1

        results.append(SampleResult(
            benchmark="arc_challenge",
            sample_id=sample["id"],
            model=model,
            correct=is_correct,
            thinking_tokens=resp["thinking_tokens"],
            completion_tokens=resp["completion_tokens"],
            total_tokens=resp["total_tokens"],
            answer=str(predicted),
            ground_truth=sample["answer"],
            elapsed_s=round(time.time() - st, 2),
        ))

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            avg_think = sum(r.thinking_tokens for r in results[-10:]) / 10
            print(f"    ARC-C {i+1}/{len(samples)} — {correct}/{i+1} ({100*correct/(i+1):.0f}%) — "
                  f"avg_think={avg_think:.0f} tok — {elapsed:.0f}s", flush=True)

    return results


# ============================================================
# MMLU-Pro (subset)
# ============================================================

def load_mmlu_pro(max_samples=MMLU_PRO_SAMPLES):
    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    random.seed(SEED)
    indices = random.sample(range(len(ds)), min(max_samples, len(ds)))
    samples = []
    for i in indices:
        row = ds[i]
        options_list = row["options"]
        labels = [chr(65 + j) for j in range(len(options_list))]
        options = "\n".join(f"{l}. {t}" for l, t in zip(labels, options_list))
        samples.append({
            "id": f"mmlu_pro_{i}",
            "question": row["question"],
            "options": options,
            "labels": labels,
            "answer": row["answer"],
        })
    return samples


def eval_mmlu_pro(model: str, samples: list) -> list[SampleResult]:
    results = []
    correct = 0
    t0 = time.time()

    for i, sample in enumerate(samples):
        st = time.time()
        messages = [
            {"role": "user", "content": f"Answer the following multiple choice question. Reply with ONLY the letter of the correct answer.\n\n{sample['question']}\n\n{sample['options']}"}
        ]

        try:
            resp = call_model(model, messages, max_tokens=4096)
            content = resp["content"].strip()
            predicted = None
            for label in sample["labels"]:
                if re.search(rf'\b{label}\b', content[:20]):
                    predicted = label
                    break
            if not predicted and content and content[0].upper() in sample["labels"]:
                predicted = content[0].upper()
            is_correct = predicted == sample["answer"]
        except Exception as e:
            resp = {"content": "", "thinking": "", "thinking_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            predicted = None
            is_correct = False

        if is_correct:
            correct += 1

        results.append(SampleResult(
            benchmark="mmlu_pro",
            sample_id=sample["id"],
            model=model,
            correct=is_correct,
            thinking_tokens=resp["thinking_tokens"],
            completion_tokens=resp["completion_tokens"],
            total_tokens=resp["total_tokens"],
            answer=str(predicted),
            ground_truth=sample["answer"],
            elapsed_s=round(time.time() - st, 2),
        ))

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            avg_think = sum(r.thinking_tokens for r in results[-10:]) / 10
            print(f"    MMLU-Pro {i+1}/{len(samples)} — {correct}/{i+1} ({100*correct/(i+1):.0f}%) — "
                  f"avg_think={avg_think:.0f} tok — {elapsed:.0f}s", flush=True)

    return results


# ============================================================
# HumanEval (code generation)
# ============================================================

def load_humaneval():
    from human_eval.data import read_problems
    problems = read_problems()
    return list(problems.items())[:HUMANEVAL_SAMPLES]


def run_code_with_timeout(code: str, timeout: float = 10.0) -> bool:
    def handler(signum, frame):
        raise TimeoutError()
    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        exec_globals = {}
        exec(code, exec_globals)
        return True
    except:
        return False
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def eval_humaneval(model: str, problems: list) -> list[SampleResult]:
    results = []
    correct = 0
    t0 = time.time()

    for i, (task_id, problem) in enumerate(problems):
        st = time.time()
        prompt = problem["prompt"]
        test = problem["test"]
        entry_point = problem["entry_point"]

        messages = [
            {"role": "system", "content": "You are a Python coding assistant. Complete the function below. Return ONLY the function body. Do not include the function signature, docstring, imports, or explanation. Do not wrap in markdown code blocks."},
            {"role": "user", "content": f"Complete this function:\n\n{prompt}"},
        ]

        try:
            resp = call_model(model, messages, max_tokens=4096)
            completion = resp["content"]

            if "```python" in completion:
                completion = completion.split("```python")[1].split("```")[0]
            elif "```" in completion:
                completion = completion.split("```")[1].split("```")[0]

            lines = completion.split("\n")
            if lines and not lines[0].startswith("    ") and not lines[0].startswith("\t"):
                completion = "\n".join("    " + l if l.strip() else l for l in lines)

            full_code = prompt + completion + "\n" + test + f"\ncheck({entry_point})\n"
            is_correct = run_code_with_timeout(full_code)
        except Exception as e:
            resp = {"content": "", "thinking": "", "thinking_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            is_correct = False

        if is_correct:
            correct += 1

        results.append(SampleResult(
            benchmark="humaneval",
            sample_id=task_id,
            model=model,
            correct=is_correct,
            thinking_tokens=resp["thinking_tokens"],
            completion_tokens=resp["completion_tokens"],
            total_tokens=resp["total_tokens"],
            answer=resp["content"][:200],
            ground_truth=entry_point,
            elapsed_s=round(time.time() - st, 2),
        ))

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            avg_think = sum(r.thinking_tokens for r in results[-10:]) / 10
            print(f"    HumanEval {i+1}/{len(problems)} — {correct}/{i+1} ({100*correct/(i+1):.0f}%) — "
                  f"avg_think={avg_think:.0f} tok — {elapsed:.0f}s", flush=True)

    return results


# ============================================================
# Main
# ============================================================

def summarize_results(results: list[SampleResult], benchmark: str, model: str) -> dict:
    bench_results = [r for r in results if r.benchmark == benchmark and r.model == model]
    if not bench_results:
        return {}
    correct = sum(1 for r in bench_results if r.correct)
    total = len(bench_results)
    thinking_tokens = [r.thinking_tokens for r in bench_results]
    completion_tokens = [r.completion_tokens for r in bench_results]

    return {
        "benchmark": benchmark,
        "model": model,
        "accuracy": round(100 * correct / total, 1),
        "correct": correct,
        "total": total,
        "avg_thinking_tokens": round(sum(thinking_tokens) / len(thinking_tokens), 1),
        "avg_completion_tokens": round(sum(completion_tokens) / len(completion_tokens), 1),
        "total_thinking_tokens": sum(thinking_tokens),
        "total_completion_tokens": sum(completion_tokens),
    }


def main():
    print("=" * 70)
    print("ThinkingCap-Qwen3.6-27B — Thinking Token Efficiency Benchmark")
    print("=" * 70)
    print(f"Methodology: thinking ENABLED, temp=1.0, top_p=0.95")
    print(f"Benchmarks: GSM8K({GSM8K_SAMPLES}), ARC-Challenge({ARC_CHALLENGE_SAMPLES}), "
          f"MMLU-Pro({MMLU_PRO_SAMPLES}), HumanEval({HUMANEVAL_SAMPLES})")
    print(f"Model: {', '.join(m.split('/')[-1] for m in MODELS)}")
    print()

    # Pre-load all datasets
    print("Loading datasets...")
    gsm8k_samples = load_gsm8k()
    arc_samples = load_arc_challenge()
    mmlu_samples = load_mmlu_pro()
    humaneval_problems = load_humaneval()
    print(f"  GSM8K: {len(gsm8k_samples)}, ARC-C: {len(arc_samples)}, "
          f"MMLU-Pro: {len(mmlu_samples)}, HumanEval: {len(humaneval_problems)}")
    print()

    all_results = []

    for model in MODELS:
        model_short = model.split("/")[-1]
        proc = start_server(model)

        try:
            print(f"\n--- {model_short}: GSM8K ---")
            results = eval_gsm8k(model, gsm8k_samples)
            all_results.extend(results)
            s = summarize_results(all_results, "gsm8k", model)
            print(f"  => {s['accuracy']}% | avg_think={s['avg_thinking_tokens']:.0f} tok")

            print(f"\n--- {model_short}: ARC-Challenge ---")
            results = eval_arc_challenge(model, arc_samples)
            all_results.extend(results)
            s = summarize_results(all_results, "arc_challenge", model)
            print(f"  => {s['accuracy']}% | avg_think={s['avg_thinking_tokens']:.0f} tok")

            print(f"\n--- {model_short}: MMLU-Pro ---")
            results = eval_mmlu_pro(model, mmlu_samples)
            all_results.extend(results)
            s = summarize_results(all_results, "mmlu_pro", model)
            print(f"  => {s['accuracy']}% | avg_think={s['avg_thinking_tokens']:.0f} tok")

            print(f"\n--- {model_short}: HumanEval ---")
            results = eval_humaneval(model, humaneval_problems)
            all_results.extend(results)
            s = summarize_results(all_results, "humaneval", model)
            print(f"  => {s['accuracy']}% | avg_think={s['avg_thinking_tokens']:.0f} tok")

        finally:
            stop_server(proc)

        # Save intermediate results after each model
        interim_path = RESULTS_DIR / f"thinking_tokens_{model_short}.json"
        model_results = [asdict(r) for r in all_results if r.model == model]
        interim_path.write_text(json.dumps(model_results, indent=2))
        print(f"\n  Intermediate results saved: {interim_path}")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS — Thinking Token Efficiency")
    print("=" * 70)

    benchmarks = ["gsm8k", "arc_challenge", "mmlu_pro", "humaneval"]
    summary_table = []

    for bench in benchmarks:
        row = {"benchmark": bench}
        for model in MODELS:
            s = summarize_results(all_results, bench, model)
            row["accuracy"] = s.get("accuracy", 0)
            row["avg_think_tokens"] = s.get("avg_thinking_tokens", 0)
            row["avg_completion_tokens"] = s.get("avg_completion_tokens", 0)
        summary_table.append(row)

    print(f"\n{'Benchmark':<15} {'Acc':>7} {'Think Tok':>10} {'Compl Tok':>10}")
    print("-" * 45)
    for row in summary_table:
        print(f"{row['benchmark']:<15} {row.get('accuracy',0):>6.1f}% "
              f"{row.get('avg_think_tokens',0):>9.0f} {row.get('avg_completion_tokens',0):>9.0f}")

    # Save final
    final_path = RESULTS_DIR / "thinking_token_comparison.json"
    final_data = {
        "metadata": {
            "methodology": "thinking enabled, temp=1.0, top_p=0.95",
            "models": MODELS,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "samples": {
                "gsm8k": GSM8K_SAMPLES,
                "arc_challenge": ARC_CHALLENGE_SAMPLES,
                "mmlu_pro": MMLU_PRO_SAMPLES,
                "humaneval": HUMANEVAL_SAMPLES,
            },
        },
        "summary": summary_table,
        "raw_results": [asdict(r) for r in all_results],
    }
    final_path.write_text(json.dumps(final_data, indent=2))
    print(f"\nFull results saved: {final_path}")


if __name__ == "__main__":
    main()
