#!/usr/bin/env python3
"""
HellaSwag evaluation via log-likelihood scoring (same methodology as lm-eval / heyneo).
Loads model directly with MLX, computes per-token log-probabilities for each continuation,
picks the highest-scoring option.

Must run AFTER server-based benchmarks (can't have 2x 27B in RAM).
"""

import json
import time
import random
import mlx.core as mx
import mlx.nn as nn
import mlx_lm
from pathlib import Path
from datasets import load_dataset

MAX_SAMPLES = 200
SEED = 42


def preprocess(text):
    text = text.strip()
    text = text.replace(" [title]", ". ")
    import re
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def load_hellaswag(max_samples=MAX_SAMPLES):
    ds = load_dataset("Rowan/hellaswag", split="validation")
    random.seed(SEED)
    indices = random.sample(range(len(ds)), min(max_samples, len(ds)))
    samples = []
    for i in indices:
        doc = ds[i]
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        samples.append({
            "query": preprocess(doc["activity_label"] + ": " + ctx),
            "choices": [preprocess(e) for e in doc["endings"]],
            "label": int(doc["label"]),
        })
    return samples


def compute_loglikelihood(model, tokenizer, context: str, continuation: str) -> float:
    ctx_tokens = tokenizer.encode(context)
    full_tokens = tokenizer.encode(context + continuation)
    cont_start = len(ctx_tokens)

    if cont_start >= len(full_tokens):
        return float("-inf")

    input_ids = mx.array([full_tokens[:-1]])
    logits = model(input_ids)
    mx.eval(logits)

    log_probs = nn.log_softmax(logits[0], axis=-1)

    target_tokens = full_tokens[cont_start:]
    token_log_probs = []
    for i, token_id in enumerate(target_tokens):
        pos = cont_start - 1 + i
        if pos < log_probs.shape[0]:
            token_log_probs.append(log_probs[pos, token_id].item())

    if not token_log_probs:
        return float("-inf")

    # Normalize by byte length (same as lm-eval acc_norm)
    byte_len = len(continuation.encode())
    return sum(token_log_probs) / byte_len


def run_eval(model_path: str, samples: list):
    print(f"  Loading model: {model_path}")
    model, tokenizer = mlx_lm.load(model_path)
    mx.eval(model.parameters())

    correct = 0
    total = 0
    t0 = time.time()

    for i, sample in enumerate(samples):
        ctx = sample["query"]
        endings = sample["choices"]
        gt_idx = sample["label"]

        scores = []
        for ending in endings:
            ll = compute_loglikelihood(model, tokenizer, ctx, " " + ending)
            scores.append(ll)

        predicted = scores.index(max(scores))
        if predicted == gt_idx:
            correct += 1
        total += 1

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"    {i+1}/{len(samples)} — {correct}/{total} ({100*correct/total:.0f}%) — {elapsed:.0f}s")

    elapsed = time.time() - t0

    del model
    mx.metal.clear_cache()

    return {
        "model": model_path,
        "correct": correct,
        "total": total,
        "accuracy": round(100 * correct / total, 1),
        "elapsed_s": round(elapsed, 1),
        "method": "log-likelihood (per-token avg)",
    }


def main():
    samples = load_hellaswag()
    print(f"HellaSwag log-likelihood eval — {len(samples)} samples\n")

    from harness_model import DEFAULT_MODEL
    models = [DEFAULT_MODEL]

    all_results = []
    for model_path in models:
        print(f"=== {model_path} ===")
        result = run_eval(model_path, samples)
        all_results.append(result)
        print(f"  SCORE: {result['correct']}/{result['total']} ({result['accuracy']}%) in {result['elapsed_s']}s\n")

    print("=" * 70)
    print(f"{'Model':<45} {'Score':>8} {'Time':>8}")
    print("-" * 70)
    for r in all_results:
        name = r["model"].split("/")[-1][:40]
        print(f"{name:<45} {r['accuracy']:>6.1f}% {r['elapsed_s']:>6.0f}s")

    print(f"\nMethod: {all_results[0]['method']}")
    print("(Same methodology as lm-eval / heyneo — NOT prompting-based)")

    out_path = Path("results/quality/hellaswag_comparison.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
