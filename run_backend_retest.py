#!/usr/bin/env python3
"""
Backend re-test: mlx-lm 0.31.2 (from main), oMLX 0.3.4, llama-server b8670.

Tests motivated by upstream changes:

  mlx-lm 0.31.2 (from main, unreleased):
    - Gemma 4 model support (#1093, #1105) — NEW, never tested
    - Qwen 3.5 GatedDeltaNet memory leak fix (#1077) — 540→20 KB/tok prefill
    - Batch generation rewrite (#1072) — sequences stop when done
    - min_p sampling 3x faster (#1083)

  oMLX 0.3.4 (Apr 5):
    - TurboQuant overhead 43%→8% with fused Metal kernels
    - Gemma 4 support (vision, MoE, reasoning)
    - Memory fix for batching regression from 0.3.3
    - Continuous batching + TurboQuant now works

  llama-server b8670 (Apr 5):
    - --clear-idle: +20-25% throughput (clear idle slot KV from VRAM)
    - Metal matmul2d descriptor fix for macOS 26.4+
    - KV cache rotation for Q4 quality
    - Gemma 4 GGUF bool metadata fix

Test matrix:
  mlx-lm:  Gemma4-26B, Gemma4-E4B, Gemma4-E2B, Qwen3.5-35B @ 32k/64k/128k
           (+ baseline without TQ for Gemma 4, since mlx-lm doesn't have TQ)
  oMLX:    Gemma4-26B, Gemma4-E4B, Gemma4-E2B, Qwen3.5-35B, Qwen3-32B @ 32k
           (+ TurboQuant mode for supported models)
  llama:   Qwen3.5-35B (Ollama GGUF), Gemma4-26B (Ollama GGUF) @ 32k
           (+ --clear-idle comparison)

Usage:
  cd /Users/HIESCHA/Projects/Work/llm-bench
  PATH=".venv/bin:$PATH" python3 -u run_backend_retest.py 2>&1 | tee /tmp/backend-retest.log
"""

import os
import subprocess
import sys
import time
import json
import csv
import signal
from datetime import datetime
from pathlib import Path

BENCH_DIR = Path(__file__).parent
VENV_BIN = BENCH_DIR / ".venv" / "bin"
RESULTS_DIR = BENCH_DIR / "results"
HOST = "127.0.0.1"

HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

# ── Prompt generation (same as TQ benchmark) ─────────────────────────

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
    user_content = "Refactor the following code to improve performance:\n\n"
    n = 0
    while len(user_content) // 4 < target_tokens - 200:
        user_content += CODE_BLOCK.format(n=n)
        n += 1
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ── Server management ────────────────────────────────────────────────

def kill_port(port):
    subprocess.run(f"lsof -ti :{port} | xargs kill -9 2>/dev/null", shell=True, capture_output=True)
    time.sleep(2)

def wait_for_server(port, timeout=300):
    for i in range(timeout):
        try:
            r = subprocess.run(
                ["curl", "-sf", f"http://{HOST}:{port}/v1/models"],
                capture_output=True, timeout=2
            )
            if r.returncode == 0:
                return i + 1
        except:
            pass
        time.sleep(1)
    return None

def run_inference(port, messages, model_id="default", max_tokens=512, timeout_s=900):
    import urllib.request
    payload = json.dumps({
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }).encode()

    req = urllib.request.Request(
        f"http://{HOST}:{port}/v1/chat/completions",
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
                            # Count both regular content and reasoning/thinking tokens
                            # mlx-lm uses "reasoning", OpenAI uses "reasoning_content"
                            has_token = (
                                delta.get("content")
                                or delta.get("reasoning")
                                or delta.get("reasoning_content")
                                or delta.get("thinking")
                            )
                            if has_token:
                                if ttft is None:
                                    ttft = (time.perf_counter() - t0) * 1000
                                tokens += 1
                        except:
                            pass
    except Exception as e:
        log(f"    ERROR: {e}")
        return None

    t_end = time.perf_counter()
    total = t_end - t0
    if ttft is None or tokens < 2:
        log(f"    No tokens (ttft={ttft}, tokens={tokens})")
        return None

    decode_time = total - (ttft / 1000)
    decode_tps = (tokens - 1) / decode_time if decode_time > 0 else 0

    return {
        "ttft_ms": round(ttft, 1),
        "decode_tps": round(decode_tps, 1),
        "tokens": tokens,
        "total_s": round(total, 1),
    }

def get_rss_mb(match_str):
    try:
        import psutil
        for p in psutil.process_iter(["pid", "cmdline"]):
            cl = " ".join(p.info.get("cmdline") or [])
            if match_str in cl:
                proc = psutil.Process(p.info["pid"])
                children = proc.children(recursive=True)
                total = sum(c.memory_info().rss for c in [proc] + children)
                return round(total / 1024 / 1024, 1)
    except:
        pass
    return 0

# ── Test definitions ─────────────────────────────────────────────────

MLX_MODELS = {
    "gemma4-26b-4bit": {
        "path": str(HF_CACHE / "models--mlx-community--gemma-4-26b-a4b-it-4bit" / "snapshots" / "b86b3e222c60ae7c652380cf516cb9c55c954fea"),
        "name": "Gemma4-26B-A4B 4bit",
        "contexts": ["context-32k", "context-64k", "context-128k"],
    },
    "gemma4-e4b-4bit": {
        "path": str(HF_CACHE / "models--mlx-community--gemma-4-e4b-it-4bit" / "snapshots" / "62b0e4e2d06c2f3baeeb0f8b7b18d7308c7786fc"),
        "name": "Gemma4-E4B 4bit",
        "contexts": ["context-32k", "context-64k", "context-128k"],
    },
    "gemma4-e2b-4bit": {
        "path": str(HF_CACHE / "models--mlx-community--gemma-4-e2b-it-4bit" / "snapshots" / "76b6a5af250fa029339a757deeb93716baa8ead0"),
        "name": "Gemma4-E2B 4bit",
        "contexts": ["context-32k", "context-64k", "context-128k"],
    },
    "qwen35-a3b-4bit": {
        "path": str(HF_CACHE / "models--mlx-community--Qwen3.5-35B-A3B-4bit" / "snapshots" / "1e20fd8d42056f870933bf98ca6211024744f7ec"),
        "name": "Qwen3.5-35B-A3B 4bit",
        "contexts": ["context-32k", "context-64k", "context-128k"],
        "chat_template_args": '{"enable_thinking":true}',
    },
    "qwen3-coder-next-4bit": {
        "path": str(HF_CACHE / "models--mlx-community--Qwen3-Coder-Next-4bit" / "snapshots" / "7b9321eabb85ce79625cac3f61ea691e4ea984b5"),
        "name": "Qwen3-Coder-Next 4bit",
        "contexts": ["context-32k"],  # dense ~35B, will OOM at 64k+
    },
}

OLLAMA_MODELS = {
    "qwen35-ollama": {
        "model": "qwen3.5:35b-a3b",
        "name": "Qwen3.5-35B Ollama INT4",
        "contexts": ["context-32k"],
    },
    "gemma4-26b-ollama": {
        "model": "gemma4:26b",
        "name": "Gemma4-26B Ollama",
        "contexts": ["context-32k"],
    },
}

# ── Backend runners ──────────────────────────────────────────────────

def run_mlx_lm_tests(rows, fields, csv_path, completed):
    """Test Gemma 4 + Qwen 3.5 on mlx-lm 0.31.2 (from main)."""
    port = 8090
    log("\n" + "=" * 70)
    log("BACKEND: mlx-lm 0.31.2 (from main)")
    log("=" * 70)

    # Verify version
    r = subprocess.run(
        [str(VENV_BIN / "python3"), "-c", "import mlx_lm; print(mlx_lm.__version__)"],
        capture_output=True, text=True
    )
    log(f"  mlx-lm version: {r.stdout.strip()}")

    for model_key, mcfg in MLX_MODELS.items():
        model_path = mcfg["path"]
        if not Path(model_path).exists():
            log(f"  SKIP {mcfg['name']}: path missing")
            continue

        log(f"\n{'━' * 70}")
        log(f"  MODEL: {mcfg['name']}")
        log(f"{'━' * 70}")

        kill_port(port)
        cmd = [
            str(VENV_BIN / "mlx_lm.server"),
            "--model", model_path,
            "--port", str(port),
            "--host", HOST,
        ]
        if mcfg.get("chat_template_args"):
            cmd += ["--chat-template-args", mcfg["chat_template_args"]]
        log(f"  Starting mlx_lm.server...")
        proc = subprocess.Popen(cmd, stdout=open("/tmp/mlx-lm-server.log", "w"), stderr=subprocess.STDOUT)

        secs = wait_for_server(port, timeout=300)
        if secs is None:
            log(f"  Server timeout, skipping")
            proc.kill()
            continue
        log(f"  Server ready after {secs}s")

        for ctx_name in mcfg["contexts"]:
            test_id = f"MLXLM_{model_key}"
            if (test_id, ctx_name) in completed:
                log(f"  {ctx_name} — already done, skip")
                continue

            target = CONTEXT_TIERS[ctx_name]
            messages = build_prompt(target)
            prompt_est = sum(len(m["content"]) for m in messages) // 4

            log(f"\n  {mcfg['name']} | {ctx_name} (~{prompt_est:,} tokens)")

            # Warmup
            log(f"    Warmup...")
            result = run_inference(port, messages, model_id=model_path, timeout_s=900)
            if not result:
                log(f"    Warmup failed, skipping")
                continue
            cold_ttft = result["ttft_ms"]
            log(f"    Cold: TTFT={cold_ttft:,.0f}ms  dec={result['decode_tps']:.1f}t/s")

            for run in range(1, 4):
                log(f"    Run {run}/3...")
                result = run_inference(port, messages, model_id=model_path, timeout_s=900)
                if not result:
                    log(f"    Run {run} failed")
                    continue
                rss = get_rss_mb("mlx_lm.server")
                prefill_tps = round(prompt_est / (result["ttft_ms"] / 1000), 1) if result["ttft_ms"] > 0 else 0
                log(f"    ✔ TTFT={result['ttft_ms']:,.0f}ms  dec={result['decode_tps']:.1f}t/s  "
                    f"pre={prefill_tps:,.0f}t/s  RSS={rss:,.0f}MB")

                row = {
                    "test_id": test_id,
                    "test_name": f"{mcfg['name']} mlx-lm",
                    "backend": "mlx-lm-0.31.2",
                    "fmt": "MLX", "quant": "4bit",
                    "kv_cache": "default",
                    "prompt_type": ctx_name, "run_num": run,
                    "ttft_ms": result["ttft_ms"], "decode_tps": result["decode_tps"],
                    "prefill_tps": prefill_tps,
                    "completion_tokens": result["tokens"], "prompt_tokens": prompt_est,
                    "total_time_s": result["total_s"], "model_load_s": 0,
                    "thinking_tokens": 0, "visible_tokens": result["tokens"],
                    "cold_ttft_ms": cold_ttft, "peak_mem_mb": rss,
                    "peak_cpu_pct": 0, "tool_call_valid": "", "quality_pass": "",
                }
                rows.append(row)
                _save_csv(rows, fields, csv_path)

        proc.terminate()
        try:
            proc.wait(timeout=10)
        except:
            proc.kill()
        kill_port(port)
        log("  Server stopped. Cooldown 10s...")
        time.sleep(10)


def run_omlx_tests(rows, fields, csv_path, completed):
    """Test oMLX 0.3.4 with default + TurboQuant KV.

    oMLX is a multi-model server: start once, select model via request body.
    TurboQuant is toggled via the admin API per model.
    """
    port = 8092
    omlx_bin = "/opt/homebrew/opt/omlx/bin/omlx"

    log("\n" + "=" * 70)
    log("BACKEND: oMLX 0.3.4")
    log("=" * 70)

    # Model IDs match directory names under ~/.omlx/models/
    omlx_models = {
        "gemma4-26b": {
            "model_id": "gemma-4-26b-a4b-it-4bit",
            "name": "Gemma4-26B-A4B 4bit",
            "contexts": ["context-32k", "context-64k"],
        },
        "gemma4-e4b": {
            "model_id": "gemma-4-e4b-it-4bit",
            "name": "Gemma4-E4B 4bit",
            "contexts": ["context-32k", "context-64k", "context-128k"],
        },
        "gemma4-e2b": {
            "model_id": "gemma-4-e2b-it-4bit",
            "name": "Gemma4-E2B 4bit",
            "contexts": ["context-32k", "context-64k", "context-128k"],
        },
        "qwen35": {
            "model_id": "Qwen3.5-35B-A3B-4bit",
            "name": "Qwen3.5-35B-A3B 4bit",
            "contexts": ["context-32k", "context-64k"],
        },
        "qwen3-coder": {
            "model_id": "Qwen3-Coder-Next-4bit",
            "name": "Qwen3-Coder-Next 4bit",
            "contexts": ["context-32k"],
        },
    }

    kv_modes = [
        {"label": "default", "tq_enabled": False},
        {"label": "tq4bit", "tq_enabled": True, "tq_bits": 4},
    ]

    model_settings_path = Path.home() / ".omlx" / "model_settings.json"

    def set_omlx_model_settings(model_id, settings):
        """Update model settings via model_settings.json file directly.

        Bypasses admin API auth issue by writing settings to disk.
        oMLX picks up changes on next model load.
        """
        try:
            if model_settings_path.exists():
                data = json.loads(model_settings_path.read_text())
            else:
                data = {"version": 1, "models": {}}

            if model_id not in data["models"]:
                data["models"][model_id] = {}
            data["models"][model_id].update(settings)

            model_settings_path.write_text(json.dumps(data, indent=2) + "\n")
            log(f"    Settings written to model_settings.json: {settings}")
            return True
        except Exception as e:
            log(f"    Settings update failed: {e}")
            return False

    # oMLX: restart server for each KV mode to ensure model_settings.json is re-read.
    # Settings are written to ~/.omlx/model_settings.json before each server start.

    for kv_mode in kv_modes:
        kv_label = kv_mode["label"]
        tq_enabled = kv_mode["tq_enabled"]

        # Write settings for ALL models with current TQ mode before starting server
        for model_key, mcfg in omlx_models.items():
            max_ctx = max(CONTEXT_TIERS[c] for c in mcfg["contexts"]) + 10000
            settings = {
                "max_context_window": max_ctx,
                "turboquant_kv_enabled": tq_enabled,
            }
            if tq_enabled:
                settings["turboquant_kv_bits"] = kv_mode.get("tq_bits", 4)
            set_omlx_model_settings(mcfg["model_id"], settings)

        # Check if all tests for this KV mode are already done
        all_kv_done = True
        for model_key, mcfg in omlx_models.items():
            test_id = f"OMLX_{model_key}_{kv_label}"
            if not all((test_id, ctx) in completed for ctx in mcfg["contexts"]):
                all_kv_done = False
                break
        if all_kv_done:
            log(f"  All {kv_label} tests done, skip server start")
            continue

        kill_port(port)
        cmd = [
            omlx_bin, "serve",
            "--port", str(port),
            "--host", HOST,
            "--no-cache",
            "--initial-cache-blocks", "4096",
        ]
        log(f"\n  Starting oMLX server (KV mode: {kv_label})...")
        proc = subprocess.Popen(cmd, stdout=open("/tmp/omlx-server.log", "w"), stderr=subprocess.STDOUT)

        secs = wait_for_server(port, timeout=300)
        if secs is None:
            log(f"  Server timeout, skipping {kv_label} tests")
            proc.kill()
            continue
        log(f"  Server ready after {secs}s")

        for model_key, mcfg in omlx_models.items():
            test_id = f"OMLX_{model_key}_{kv_label}"
            all_done = all((test_id, ctx) in completed for ctx in mcfg["contexts"])
            if all_done:
                log(f"  {mcfg['name']} {kv_label}: all done, skip")
                continue

            log(f"\n{'━' * 70}")
            log(f"  MODEL: {mcfg['name']} | KV: {kv_label}")
            log(f"{'━' * 70}")

            for ctx_name in mcfg["contexts"]:
                if (test_id, ctx_name) in completed:
                    log(f"  {ctx_name} — already done, skip")
                    continue

                target = CONTEXT_TIERS[ctx_name]
                messages = build_prompt(target)
                prompt_est = sum(len(m["content"]) for m in messages) // 4

                log(f"\n  {mcfg['name']} | {kv_label} | {ctx_name} (~{prompt_est:,} tokens)")

                log(f"    Warmup...")
                result = run_inference(port, messages, model_id=mcfg["model_id"], timeout_s=900)
                if not result:
                    log(f"    Warmup failed, skipping")
                    continue
                cold_ttft = result["ttft_ms"]
                log(f"    Cold: TTFT={cold_ttft:,.0f}ms  dec={result['decode_tps']:.1f}t/s")

                for run in range(1, 4):
                    log(f"    Run {run}/3...")
                    result = run_inference(port, messages, model_id=mcfg["model_id"], timeout_s=900)
                    if not result:
                        log(f"    Run {run} failed")
                        continue
                    rss = get_rss_mb("omlx")
                    prefill_tps = round(prompt_est / (result["ttft_ms"] / 1000), 1) if result["ttft_ms"] > 0 else 0
                    log(f"    ✔ TTFT={result['ttft_ms']:,.0f}ms  dec={result['decode_tps']:.1f}t/s  "
                        f"pre={prefill_tps:,.0f}t/s  RSS={rss:,.0f}MB")

                    row = {
                        "test_id": test_id,
                        "test_name": f"{mcfg['name']} oMLX {kv_label}",
                        "backend": "omlx-0.3.4",
                        "fmt": "MLX", "quant": "4bit",
                        "kv_cache": kv_label,
                        "prompt_type": ctx_name, "run_num": run,
                        "ttft_ms": result["ttft_ms"], "decode_tps": result["decode_tps"],
                        "prefill_tps": prefill_tps,
                        "completion_tokens": result["tokens"], "prompt_tokens": prompt_est,
                        "total_time_s": result["total_s"], "model_load_s": 0,
                        "thinking_tokens": 0, "visible_tokens": result["tokens"],
                        "cold_ttft_ms": cold_ttft, "peak_mem_mb": rss,
                        "peak_cpu_pct": 0, "tool_call_valid": "", "quality_pass": "",
                    }
                    rows.append(row)
                    _save_csv(rows, fields, csv_path)

        proc.terminate()
        try:
            proc.wait(timeout=10)
        except:
            proc.kill()
        kill_port(port)
        log(f"  oMLX server stopped ({kv_label}). Cooldown 10s...")
        time.sleep(10)


def run_llama_tests(rows, fields, csv_path, completed):
    """Test llama-server b8670 with --clear-idle."""
    port = 8093
    llama_bin = "/opt/homebrew/bin/llama-server"

    log("\n" + "=" * 70)
    log("BACKEND: llama-server b8670")
    log("=" * 70)

    # Use Ollama's GGUF blobs
    ollama_dir = Path.home() / ".ollama" / "models"
    manifest_dir = ollama_dir / "manifests" / "registry.ollama.ai" / "library"

    def get_ollama_gguf(model_tag):
        """Resolve Ollama model tag to GGUF blob path."""
        parts = model_tag.split(":")
        name = parts[0]
        tag = parts[1] if len(parts) > 1 else "latest"
        manifest_path = manifest_dir / name / tag
        if not manifest_path.exists():
            return None
        manifest = json.loads(manifest_path.read_text())
        for layer in manifest.get("layers", []):
            if layer.get("mediaType") == "application/vnd.ollama.image.model":
                digest = layer["digest"].replace(":", "-")
                blob = ollama_dir / "blobs" / digest
                if blob.exists():
                    return str(blob)
        return None

    llama_models = [
        {
            "tag": "qwen3.5:35b-a3b",
            "name": "Qwen3.5-35B Ollama GGUF",
            "contexts": ["context-32k"],
        },
        # Gemma4-26B Ollama GGUF removed: llama.cpp b8670 tensor count mismatch
        # (expected 1014, got 658 — Ollama uses a split GGUF format)
    ]

    # Standalone GGUFs (not from Ollama)
    standalone_ggufs = [
        {
            "path": str(Path.home() / ".cache" / "llmfit" / "models" / "Qwen3-Coder-Next-Q4_K_M.gguf"),
            "tag": "qwen3-coder-next-gguf",
            "name": "Qwen3-Coder-Next GGUF Q4_K_M",
            "contexts": ["context-32k"],
        },
    ]

    clear_idle_modes = [
        {"label": "default", "args": []},
        {"label": "clear-idle", "args": ["--clear-idle"]},
    ]

    # Build unified model list: Ollama models + standalone GGUFs
    all_llama_models = []
    for mcfg in llama_models:
        gguf_path = get_ollama_gguf(mcfg["tag"])
        if gguf_path:
            all_llama_models.append({**mcfg, "gguf_path": gguf_path})
        else:
            log(f"  SKIP {mcfg['name']}: GGUF not found for {mcfg['tag']}")
    for mcfg in standalone_ggufs:
        if Path(mcfg["path"]).exists():
            all_llama_models.append({**mcfg, "gguf_path": mcfg["path"]})
        else:
            log(f"  SKIP {mcfg['name']}: GGUF not found at {mcfg['path']}")

    for mcfg in all_llama_models:
        gguf_path = mcfg["gguf_path"]

        log(f"\n{'━' * 70}")
        log(f"  MODEL: {mcfg['name']}")
        log(f"  GGUF: ...{gguf_path[-60:]}")
        log(f"{'━' * 70}")

        for ci_mode in clear_idle_modes:
            ci_label = ci_mode["label"]
            ci_args = ci_mode["args"]
            model_key = mcfg["tag"].replace(":", "-").replace(".", "")
            test_id = f"LLAMA_{model_key}_{ci_label}"

            all_done = all((test_id, ctx) in completed for ctx in mcfg["contexts"])
            if all_done:
                log(f"  {ci_label}: all done, skip")
                continue

            kill_port(port)
            cmd = [
                llama_bin,
                "--model", gguf_path,
                "--port", str(port),
                "--host", HOST,
                "--ctx-size", "32768",
                "--n-gpu-layers", "99",
                "--flash-attn", "on",
            ] + ci_args
            log(f"  Starting llama-server ({ci_label})...")
            proc = subprocess.Popen(cmd, stdout=open("/tmp/llama-server.log", "w"), stderr=subprocess.STDOUT)

            secs = wait_for_server(port, timeout=300)
            if secs is None:
                log(f"  Server timeout, skipping")
                proc.kill()
                kill_port(port)
                continue
            log(f"  Server ready after {secs}s")

            for ctx_name in mcfg["contexts"]:
                if (test_id, ctx_name) in completed:
                    log(f"  {ctx_name} — already done, skip")
                    continue

                target = CONTEXT_TIERS[ctx_name]
                messages = build_prompt(target)
                prompt_est = sum(len(m["content"]) for m in messages) // 4

                log(f"\n  {mcfg['name']} | {ci_label} | {ctx_name} (~{prompt_est:,} tokens)")

                log(f"    Warmup...")
                result = run_inference(port, messages, model_id=mcfg["tag"], timeout_s=900)
                if not result:
                    log(f"    Warmup failed, skipping")
                    continue
                cold_ttft = result["ttft_ms"]
                log(f"    Cold: TTFT={cold_ttft:,.0f}ms  dec={result['decode_tps']:.1f}t/s")

                for run in range(1, 4):
                    log(f"    Run {run}/3...")
                    result = run_inference(port, messages, model_id=mcfg["tag"], timeout_s=900)
                    if not result:
                        log(f"    Run {run} failed")
                        continue
                    rss = get_rss_mb("llama-server")
                    prefill_tps = round(prompt_est / (result["ttft_ms"] / 1000), 1) if result["ttft_ms"] > 0 else 0
                    log(f"    ✔ TTFT={result['ttft_ms']:,.0f}ms  dec={result['decode_tps']:.1f}t/s  "
                        f"pre={prefill_tps:,.0f}t/s  RSS={rss:,.0f}MB")

                    row = {
                        "test_id": test_id,
                        "test_name": f"{mcfg['name']} llama {ci_label}",
                        "backend": f"llama-b8670",
                        "fmt": "GGUF", "quant": "Q4_K_M",
                        "kv_cache": ci_label,
                        "prompt_type": ctx_name, "run_num": run,
                        "ttft_ms": result["ttft_ms"], "decode_tps": result["decode_tps"],
                        "prefill_tps": prefill_tps,
                        "completion_tokens": result["tokens"], "prompt_tokens": prompt_est,
                        "total_time_s": result["total_s"], "model_load_s": 0,
                        "thinking_tokens": 0, "visible_tokens": result["tokens"],
                        "cold_ttft_ms": cold_ttft, "peak_mem_mb": rss,
                        "peak_cpu_pct": 0, "tool_call_valid": "", "quality_pass": "",
                    }
                    rows.append(row)
                    _save_csv(rows, fields, csv_path)

            proc.terminate()
            try:
                proc.wait(timeout=10)
            except:
                proc.kill()
            kill_port(port)
            log("  Server stopped. Cooldown 10s...")
            time.sleep(10)


# ── CSV helpers ──────────────────────────────────────────────────────

FIELDS = ["test_id", "test_name", "backend", "fmt", "quant", "kv_cache",
          "prompt_type", "run_num", "ttft_ms", "decode_tps", "prefill_tps",
          "completion_tokens", "prompt_tokens", "total_time_s", "model_load_s",
          "thinking_tokens", "visible_tokens", "cold_ttft_ms", "peak_mem_mb",
          "peak_cpu_pct", "tool_call_valid", "quality_pass"]

def _save_csv(rows, fields, csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def _load_prior(results_dir):
    """Load prior results for resume."""
    rows = []
    completed = set()
    csvs = sorted(results_dir.glob("backend_retest_*.csv"))
    if csvs:
        latest = csvs[-1]
        with open(latest) as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
                completed.add((r["test_id"], r["prompt_type"]))
        return rows, completed, latest
    return rows, completed, None


# ── Main ─────────────────────────────────────────────────────────────

def main():
    log("=" * 70)
    log("Backend re-test: mlx-lm 0.31.2, oMLX 0.3.4, llama-server b8670")
    log("=" * 70)

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"backend_retest_{ts}.csv"

    rows, completed, prior = _load_prior(RESULTS_DIR)
    if prior:
        log(f"Resuming from: {prior.name} ({len(rows)} rows, {len(completed)} configs)")

    t0 = time.time()

    # Run all backends
    run_mlx_lm_tests(rows, FIELDS, csv_path, completed)
    run_omlx_tests(rows, FIELDS, csv_path, completed)
    run_llama_tests(rows, FIELDS, csv_path, completed)

    elapsed = time.time() - t0
    log(f"\n{'=' * 70}")
    log(f"COMPLETE — {len(rows)} rows in {elapsed/60:.1f} minutes")
    log(f"CSV: {csv_path}")
    log(f"{'=' * 70}")


if __name__ == "__main__":
    main()
