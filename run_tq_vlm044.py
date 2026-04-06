#!/usr/bin/env python3
"""
Comprehensive TurboQuant KV compression benchmark via mlx-vlm 0.4.4.

mlx-vlm 0.4.4 includes TurboQuant as a first-class --kv-quant-scheme flag,
with fused Metal kernels (7→1 dispatch). Tests TQ 3.5-bit KV against default
KV across all locally-cached models that fit in 64 GB unified memory.

Models tested (all locally cached, no network needed):
  - Gemma 4 26B A4B (MoE) — 4bit, mxfp4, nvfp4
  - Gemma 4 31B (dense) — 4bit, mxfp4, nvfp4
  - Gemma 4 E4B — 4bit, mxfp4, nvfp4
  - Gemma 4 E2B — 4bit, nvfp4
  - Qwen3.5-35B-A3B (MoE) — 4bit
  - Qwen3-32B (dense) — 4bit
  - Gemma 3-27B — 4bit

Each model is tested at context-32k with:
  a) default KV cache (baseline)
  b) TurboQuant 3.5-bit KV

Larger models (MoE, small dense) also test context-64k and context-128k
where memory permits.

Usage:
  cd /Users/HIESCHA/Projects/Work/llm-bench
  PATH=".venv/bin:$PATH" python3 -u run_tq_vlm044.py 2>&1 | tee /tmp/tq-vlm044.log
"""

import os
import signal
import subprocess
import sys
import time
import json
import csv
from datetime import datetime
from pathlib import Path

BENCH_DIR = Path(__file__).parent
VENV_BIN = BENCH_DIR / ".venv" / "bin"
RESULTS_DIR = BENCH_DIR / "results"
PORT = 8091
HOST = "127.0.0.1"

HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

# ── Model registry ──────────────────────────────────────────────────
# Each entry: local snapshot path, display name, context tiers to test
# Memory budget: 64 GB total, ~58 GB usable for model+KV

def repo_id_from_path(path: str) -> str:
    """Extract HF repo ID from local cache path, e.g. 'mlx-community/gemma-4-e2b-it-4bit'."""
    # Path contains: models--mlx-community--gemma-4-e2b-it-4bit/snapshots/...
    import re
    m = re.search(r'models--([^/]+)--([^/]+)', path)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return path  # fallback

MODELS = {
    # ── Gemma 4 MoE (26B total, 4B active) ──
    "gemma4-26b-4bit": {
        "path": str(HF_CACHE / "models--mlx-community--gemma-4-26b-a4b-it-4bit" / "snapshots" / "b86b3e222c60ae7c652380cf516cb9c55c954fea"),
        "name": "Gemma4-26B-A4B 4bit",
        "group": "gemma4-26b",
        "quant": "4bit",
        "contexts": ["context-32k", "context-64k", "context-128k"],
    },
    "gemma4-26b-mxfp4": {
        "path": str(HF_CACHE / "models--mlx-community--gemma-4-26b-a4b-it-mxfp4" / "snapshots" / "3e402e6595e6543303a8fc7539e6df1393706590"),
        "name": "Gemma4-26B-A4B mxfp4",
        "group": "gemma4-26b",
        "quant": "mxfp4",
        "contexts": ["context-32k", "context-64k"],
    },
    "gemma4-26b-nvfp4": {
        "path": str(HF_CACHE / "models--mlx-community--gemma-4-26b-a4b-it-nvfp4" / "snapshots" / "1ccf21a18f50c9e5ef101f838a79408f308f104a"),
        "name": "Gemma4-26B-A4B nvfp4",
        "group": "gemma4-26b",
        "quant": "nvfp4",
        "contexts": ["context-32k", "context-64k"],
    },

    # ── Gemma 4 Dense (31B) ──
    "gemma4-31b-4bit": {
        "path": str(HF_CACHE / "models--mlx-community--gemma-4-31b-it-4bit" / "snapshots" / "535c5606372deb5d5ab7e29280f111ef2a8e084e"),
        "name": "Gemma4-31B 4bit",
        "group": "gemma4-31b",
        "quant": "4bit",
        "contexts": ["context-32k", "context-64k"],
    },
    "gemma4-31b-mxfp4": {
        "path": str(HF_CACHE / "models--mlx-community--gemma-4-31b-it-mxfp4" / "snapshots" / "93085b8389728558a91ce61fde64b24540ca52b1"),
        "name": "Gemma4-31B mxfp4",
        "group": "gemma4-31b",
        "quant": "mxfp4",
        "contexts": ["context-32k"],
    },
    "gemma4-31b-nvfp4": {
        "path": str(HF_CACHE / "models--mlx-community--gemma-4-31b-it-nvfp4" / "snapshots" / "e2467a5897572245e8fde71940fe90da97c1e2bc"),
        "name": "Gemma4-31B nvfp4",
        "group": "gemma4-31b",
        "quant": "nvfp4",
        "contexts": ["context-32k"],
    },

    # ── Gemma 4 E4B (efficient 4B) ──
    "gemma4-e4b-4bit": {
        "path": str(HF_CACHE / "models--mlx-community--gemma-4-e4b-it-4bit" / "snapshots" / "62b0e4e2d06c2f3baeeb0f8b7b18d7308c7786fc"),
        "name": "Gemma4-E4B 4bit",
        "group": "gemma4-e4b",
        "quant": "4bit",
        "contexts": ["context-32k", "context-64k", "context-128k"],
    },
    "gemma4-e4b-mxfp4": {
        "path": str(HF_CACHE / "models--mlx-community--gemma-4-e4b-it-mxfp4" / "snapshots" / "2db95479b743bf0c2a44061a157790501bc181d5"),
        "name": "Gemma4-E4B mxfp4",
        "group": "gemma4-e4b",
        "quant": "mxfp4",
        "contexts": ["context-32k", "context-64k", "context-128k"],
    },
    "gemma4-e4b-nvfp4": {
        "path": str(HF_CACHE / "models--mlx-community--gemma-4-e4b-it-nvfp4" / "snapshots" / "b7c7eedb1c4f23ddad16b78254cc043dd7e5016e"),
        "name": "Gemma4-E4B nvfp4",
        "group": "gemma4-e4b",
        "quant": "nvfp4",
        "contexts": ["context-32k", "context-64k", "context-128k"],
    },

    # ── Gemma 4 E2B (efficient 2B — ultra-low memory) ──
    "gemma4-e2b-4bit": {
        "path": str(HF_CACHE / "models--mlx-community--gemma-4-e2b-it-4bit" / "snapshots" / "76b6a5af250fa029339a757deeb93716baa8ead0"),
        "name": "Gemma4-E2B 4bit",
        "group": "gemma4-e2b",
        "quant": "4bit",
        "contexts": ["context-32k", "context-64k", "context-128k"],
    },

    # ── Qwen3.5-35B-A3B (MoE, 3B active) ──
    "qwen35-a3b-4bit": {
        "path": str(HF_CACHE / "models--mlx-community--Qwen3.5-35B-A3B-4bit" / "snapshots" / "1e20fd8d42056f870933bf98ca6211024744f7ec"),
        "name": "Qwen3.5-35B-A3B 4bit",
        "group": "qwen35",
        "quant": "4bit",
        "contexts": ["context-32k", "context-64k", "context-128k"],
    },

    # ── Qwen3-32B (dense) ──
    "qwen3-32b-4bit": {
        "path": str(HF_CACHE / "models--mlx-community--Qwen3-32B-4bit" / "snapshots" / "bcaaf7f538adf166c1080a2befdb4f6019f66639"),
        "name": "Qwen3-32B 4bit",
        "group": "qwen3-32b",
        "quant": "4bit",
        "contexts": ["context-32k"],  # dense 32B will OOM at 64k+
    },

    # ── Gemma 3-27B ──
    "gemma3-27b-4bit": {
        "path": str(HF_CACHE / "models--mlx-community--gemma-3-27b-it-4bit" / "snapshots" / "83acee3d10064661a7a39ead3732dddb16fe15bb"),
        "name": "Gemma3-27B 4bit",
        "group": "gemma3",
        "quant": "4bit",
        "contexts": ["context-32k"],  # dense 27B will OOM at 64k+
    },
}

# KV cache modes to test per model
KV_MODES = [
    {"label": "default", "args": []},
    {"label": "tq3.5bit", "args": ["--kv-bits", "3.5", "--kv-quant-scheme", "turboquant"]},
]

# ── Context prompt generation ───────────────────────────────────────

SYSTEM_PROMPT = "You are a senior Python developer. Respond concisely."
CODE_BLOCK = '''
def process_batch_{n}(items: list, config: dict) -> dict:
    """Process a batch of items with the given configuration."""
    results = {{}}
    for item in items:
        key = item.get("id", "unknown")
        value = item.get("value", 0)
        if config.get("normalize"):
            value = value / max(abs(value), 1e-8)
        if config.get("filter_threshold") is not None:
            if abs(value) < config["filter_threshold"]:
                continue
        results[key] = {{"processed": value, "batch": {n}, "timestamp": None}}
    return results
'''

CONTEXT_TIERS = {
    "context-32k": 32_000,
    "context-64k": 64_000,
    "context-128k": 128_000,
}

def build_prompt(target_tokens: int) -> list:
    """Build a prompt targeting approximately target_tokens."""
    user_content = "Refactor the following code to improve performance:\n\n"
    n = 0
    while len(user_content) // 4 < target_tokens - 200:
        user_content += CODE_BLOCK.format(n=n)
        n += 1
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ── Logging ─────────────────────────────────────────────────────────

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Server management ───────────────────────────────────────────────

def kill_port():
    """Kill anything on PORT."""
    subprocess.run(
        f"lsof -ti :{PORT} | xargs kill -9 2>/dev/null",
        shell=True, capture_output=True
    )
    time.sleep(2)

def start_server(model_path: str, kv_args: list) -> subprocess.Popen:
    """Start mlx-vlm server with optional TurboQuant."""
    kill_port()
    cmd = [
        str(VENV_BIN / "mlx_vlm.server"),
        "--model", model_path,
        "--port", str(PORT),
        "--host", HOST,
        "--max-kv-size", "131072",
    ] + kv_args
    log(f"  CMD: mlx_vlm.server --port {PORT} {' '.join(kv_args) or '(default KV)'}")
    proc = subprocess.Popen(
        cmd,
        stdout=open("/tmp/tq-server.log", "w"),
        stderr=subprocess.STDOUT,
    )
    # Wait for server ready (up to 5 min for large models)
    for i in range(300):
        try:
            r = subprocess.run(
                ["curl", "-sf", f"http://{HOST}:{PORT}/v1/models"],
                capture_output=True, timeout=2
            )
            if r.returncode == 0:
                log(f"  Server ready after {i+1}s")
                return proc
        except:
            pass
        # Check if process died
        if proc.poll() is not None:
            log(f"  Server exited with code {proc.returncode}")
            try:
                with open("/tmp/tq-server.log") as f:
                    tail = f.read()[-500:]
                log(f"  Last log: {tail}")
            except:
                pass
            return None
        time.sleep(1)
    log("  ERROR: Server timeout after 300s")
    proc.kill()
    return None

def stop_server(proc):
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except:
            proc.kill()
    kill_port()


# ── Inference ────────────────────────────────────────────────────────

def run_inference(messages, model_id="default", max_tokens=512, timeout_s=600):
    """Run a single inference request and return metrics."""
    import urllib.request
    payload = json.dumps({
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }).encode()

    req = urllib.request.Request(
        f"http://{HOST}:{PORT}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    ttft = None
    tokens = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            buf = b""
            for chunk in iter(lambda: resp.read(1), b""):
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line or line == b"data: [DONE]":
                        continue
                    if line.startswith(b"data: "):
                        try:
                            d = json.loads(line[6:])
                            delta = d.get("choices", [{}])[0].get("delta", {})
                            if delta.get("content"):
                                if ttft is None:
                                    ttft = (time.perf_counter() - t0) * 1000
                                tokens += 1
                        except:
                            pass
    except Exception as e:
        log(f"  ERROR: {e}")
        return None

    t_end = time.perf_counter()
    total = t_end - t0

    if ttft is None or tokens < 2:
        log(f"  No tokens received (ttft={ttft}, tokens={tokens})")
        return None

    decode_time = total - (ttft / 1000)
    decode_tps = (tokens - 1) / decode_time if decode_time > 0 else 0

    return {
        "ttft_ms": round(ttft, 1),
        "decode_tps": round(decode_tps, 1),
        "tokens": tokens,
        "total_s": round(total, 1),
    }


def get_rss_mb():
    """Get RSS of mlx_vlm.server process tree."""
    try:
        import psutil
        for p in psutil.process_iter(["pid", "cmdline"]):
            cl = " ".join(p.info.get("cmdline") or [])
            if "mlx_vlm.server" in cl:
                proc = psutil.Process(p.info["pid"])
                children = proc.children(recursive=True)
                total = sum(c.memory_info().rss for c in [proc] + children)
                return round(total / 1024 / 1024, 1)
    except:
        pass
    return 0


# ── Main ─────────────────────────────────────────────────────────────

def main():
    log("=" * 70)
    log("Comprehensive TurboQuant benchmark — mlx-vlm 0.4.4 native Metal")
    log(f"Models: {len(MODELS)} | KV modes: default + TQ 3.5bit")
    log(f"Runs per config: 1 warmup + 3 measured")
    log("=" * 70)

    # Verify mlx-vlm version
    result = subprocess.run(
        [str(VENV_BIN / "python3"), "-c", "import mlx_vlm; print(mlx_vlm.__version__)"],
        capture_output=True, text=True
    )
    log(f"mlx-vlm version: {result.stdout.strip()}")

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"tq_bench_{ts}.csv"

    fields = ["test_id", "test_name", "backend", "fmt", "quant", "kv_cache",
              "prompt_type", "run_num", "ttft_ms", "decode_tps", "prefill_tps",
              "completion_tokens", "prompt_tokens", "total_time_s", "model_load_s",
              "thinking_tokens", "visible_tokens", "cold_ttft_ms", "peak_mem_mb",
              "peak_cpu_pct", "tool_call_valid", "quality_pass"]

    # Resume: load prior results to skip completed configs
    rows = []
    completed_configs = set()  # (test_id, prompt_type) pairs with 3 runs
    prior_csvs = sorted(RESULTS_DIR.glob("tq_bench_*.csv"))
    if prior_csvs:
        latest = prior_csvs[-1]
        log(f"Resuming from: {latest.name}")
        with open(latest) as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
                key = (r["test_id"], r["prompt_type"])
                completed_configs.add(key)
        log(f"  Loaded {len(rows)} prior rows, {len(completed_configs)} completed configs")

    total_configs = sum(
        len(m["contexts"]) * len(KV_MODES) for m in MODELS.values()
    )
    config_num = 0
    failed_configs = []

    for model_key, model_cfg in MODELS.items():
        log(f"\n{'━' * 70}")
        log(f"  MODEL: {model_cfg['name']}")
        log(f"  Path:  ...{model_cfg['path'][-60:]}")
        log(f"  Contexts: {', '.join(model_cfg['contexts'])}")
        log(f"{'━' * 70}")

        # Verify model files exist
        model_dir = Path(model_cfg["path"])
        if not model_dir.exists():
            log(f"  SKIP: path does not exist")
            failed_configs.append((model_key, "all", "path missing"))
            continue
        safetensors = list(model_dir.glob("*.safetensors"))
        if not safetensors:
            log(f"  SKIP: no safetensors found")
            failed_configs.append((model_key, "all", "no safetensors"))
            continue
        log(f"  Found {len(safetensors)} safetensors file(s)")
        # IMPORTANT: model_id in the request must match the exact path passed to --model,
        # otherwise mlx-vlm unloads the current model and tries to re-download from HF Hub.
        model_id = model_cfg["path"]
        log(f"  Repo ID: {repo_id_from_path(model_cfg['path'])}")

        for kv_mode in KV_MODES:
            kv_label = kv_mode["label"]
            kv_args = kv_mode["args"]

            log(f"\n  ── KV mode: {kv_label} ──")

            # Check if all contexts already done for this model+kv combo
            test_id_prefix = f"TQ_{model_key}_{kv_label}"
            all_done = all(
                (test_id_prefix, ctx) in completed_configs
                for ctx in model_cfg["contexts"]
            )
            if all_done:
                log(f"  All contexts already done — skipping server start")
                config_num += len(model_cfg["contexts"])
                continue

            # Start server for this model+kv combo
            proc = start_server(model_cfg["path"], kv_args)
            if not proc:
                log(f"  SKIP: server failed to start")
                for ctx in model_cfg["contexts"]:
                    failed_configs.append((model_key, f"{kv_label}/{ctx}", "server failed"))
                continue

            for ctx_name in model_cfg["contexts"]:
                config_num += 1
                test_id = f"TQ_{model_key}_{kv_label}"

                # Skip if already completed in prior run
                if (test_id, ctx_name) in completed_configs:
                    log(f"\n  [{config_num}/{total_configs}] {model_cfg['name']} | {kv_label} | {ctx_name} — SKIP (already done)")
                    continue

                target = CONTEXT_TIERS[ctx_name]
                messages = build_prompt(target)
                prompt_tokens_est = sum(len(m["content"]) for m in messages) // 4

                log(f"\n  [{config_num}/{total_configs}] {model_cfg['name']} | {kv_label} | {ctx_name} (~{prompt_tokens_est:,} tokens)")

                # Warmup (also captures cold TTFT)
                log(f"  Warmup...")
                cold_ttft = None
                result = run_inference(messages, model_id=model_id, timeout_s=900)
                if result:
                    cold_ttft = result["ttft_ms"]
                    log(f"  Cold: TTFT={cold_ttft:,.0f}ms  dec={result['decode_tps']:.1f}t/s  tok={result['tokens']}")
                else:
                    log(f"  Warmup failed — skipping this context tier")
                    failed_configs.append((model_key, f"{kv_label}/{ctx_name}", "warmup failed"))
                    continue

                # 3 measured runs
                for run in range(1, 4):
                    log(f"  Run {run}/3...")
                    result = run_inference(messages, model_id=model_id, timeout_s=900)
                    if result is None:
                        log(f"  Run {run} failed")
                        failed_configs.append((model_key, f"{kv_label}/{ctx_name}/run{run}", "inference failed"))
                        continue

                    rss = get_rss_mb()
                    prefill_tps = round(prompt_tokens_est / (result["ttft_ms"] / 1000), 1) if result["ttft_ms"] > 0 else 0

                    log(f"  ✔ TTFT={result['ttft_ms']:,.0f}ms  dec={result['decode_tps']:.1f}t/s  "
                        f"pre={prefill_tps:,.0f}t/s  tok={result['tokens']}  "
                        f"total={result['total_s']:.1f}s  RSS={rss:,.0f}MB")

                    row = {
                        "test_id": test_id,
                        "test_name": f"{model_cfg['name']} {kv_label}",
                        "backend": "mlx-vlm-0.4.4",
                        "fmt": "MLX",
                        "quant": model_cfg["quant"],
                        "kv_cache": kv_label,
                        "prompt_type": ctx_name,
                        "run_num": run,
                        "ttft_ms": result["ttft_ms"],
                        "decode_tps": result["decode_tps"],
                        "prefill_tps": prefill_tps,
                        "completion_tokens": result["tokens"],
                        "prompt_tokens": prompt_tokens_est,
                        "total_time_s": result["total_s"],
                        "model_load_s": 0,
                        "thinking_tokens": 0,
                        "visible_tokens": result["tokens"],
                        "cold_ttft_ms": cold_ttft,
                        "peak_mem_mb": rss,
                        "peak_cpu_pct": 0,
                        "tool_call_valid": "",
                        "quality_pass": "",
                    }
                    rows.append(row)

                    # Incremental save after each run
                    with open(csv_path, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=fields)
                        writer.writeheader()
                        writer.writerows(rows)

            stop_server(proc)
            log("  Server stopped. Cooldown 10s...")
            time.sleep(10)

    # Final summary
    log(f"\n{'=' * 70}")
    log(f"COMPLETE")
    log(f"  Total rows:   {len(rows)}")
    log(f"  Configs tried: {config_num}/{total_configs}")
    log(f"  Failures:     {len(failed_configs)}")
    log(f"  CSV:          {csv_path}")
    if failed_configs:
        log(f"\n  Failed configs:")
        for model, ctx, reason in failed_configs:
            log(f"    {model} / {ctx}: {reason}")
    log(f"{'=' * 70}")


if __name__ == "__main__":
    main()
