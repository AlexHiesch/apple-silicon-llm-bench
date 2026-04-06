# apple-silicon-llm-bench

Systematic benchmark harness for local LLM inference on Apple Silicon. Evaluates time-to-first-token (TTFT), decode throughput, and memory usage across backends, models, quantization formats, and context depths.

## Quick start

```bash
pip install -r requirements.txt
# Edit config.yaml to select backends and models
python benchmark.py
```

## What it measures

- **TTFT** (warm + cold) at realistic context depths (512 to 128k tokens)
- **Decode throughput** (tokens/second)
- **Peak RSS memory** via process-tree monitoring
- Full server lifecycle per config: start, health check, warmup, 3 measured runs, teardown

## Coverage

57 configurations across:
- **8 backends**: mlx-lm, mlx-vlm, Ollama, oMLX, vllm-mlx, llama.cpp, LM Studio, Docker Model Runner
- **7 models**: Gemma 4 (26B, E4B, E2B), Qwen3.5-35B-A3B, Qwen3-Coder-Next, Qwen3-32B, Gemma 4 31B
- **6 quantization formats**: 4-bit, 8-bit, MXFP4, NVFP4, GGUF Q4_K_M, GGUF Q8_0
- **7 KV cache strategies**: FP16, Q8, Q4, TurboQuant 3.5-bit, TurboQuant 4-bit, clear-idle, SSD offload

## Results

791 measurements in `results/`. Browse the [interactive report](https://hiesch.eu/bench/).

## Hardware

All results collected on Apple M3 Max, 64 GB unified memory, macOS 15.4.

## Blog post

[Running Local LLMs: An Easter Weekend Rabbit Hole](https://hiesch.eu/blog/running-local-llms-easter-weekend/)
