#!/usr/bin/env python3
"""Minimal OpenAI-compatible HTTP wrapper around llama-diffusion-cli."""
import argparse
import json
import subprocess
import time
import re
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

DIFFUSION_CLI = "/tmp/llama-diffusion/build/bin/llama-diffusion-cli"


def run_diffusion(model_path: str, prompt: str, max_tokens: int = 512,
                  temperature: float = 0.7, ngl: int = 99,
                  diffusion_steps: int = 128) -> dict:
    cmd = [
        DIFFUSION_CLI, "-m", model_path, "-ngl", str(ngl),
        "-p", prompt, "-n", str(max_tokens),
        "--diffusion-steps", str(diffusion_steps),
        "--diffusion-algorithm", "4",
        "--temp", str(temperature),
    ]
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    elapsed = time.perf_counter() - t0

    output = result.stdout + result.stderr
    content = ""
    tokens = max_tokens
    tps = 0.0

    tps_match = re.search(r"throughput:\s*([\d.]+)\s*tok/s\s*\((\d+)\s*tok", output)
    if tps_match:
        tps = float(tps_match.group(1))
        tokens = int(tps_match.group(2))

    lines = output.split("\n")
    capture = False
    content_lines = []
    for line in lines:
        if "diffusion step:" in line or "diffusion_" in line:
            continue
        if line.startswith("<|channel>") or capture:
            capture = True
            content_lines.append(line)

    if content_lines:
        content = "\n".join(content_lines)
    else:
        for line in reversed(lines):
            if line.strip() and "throughput:" not in line and "total time:" not in line and "diffusion" not in line.lower() and "ggml" not in line.lower() and "llama_" not in line.lower() and "print_info" not in line.lower():
                content = line
                break

    return {
        "content": content,
        "tokens": tokens,
        "elapsed": elapsed,
        "tps": tps,
    }


class DiffusionHandler(BaseHTTPRequestHandler):
    model_path = ""

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/health" or self.path == "/v1/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if "/models" in self.path:
                resp = {"object": "list", "data": [{"id": "diffusiongemma", "object": "model"}]}
            else:
                resp = {"status": "ok"}
            self.wfile.write(json.dumps(resp).encode())
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if "/v1/chat/completions" not in self.path:
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", 512)
        temperature = body.get("temperature", 0.7)

        prompt_parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                prompt_parts.append(f"<start_of_turn>system\n{content}<end_of_turn>")
            elif role == "user":
                prompt_parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
            elif role == "assistant":
                prompt_parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")
        prompt_parts.append("<start_of_turn>model\n")
        prompt = "\n".join(prompt_parts)

        result = run_diffusion(self.model_path, prompt, max_tokens, temperature)
        stream = body.get("stream", False)
        chat_id = f"diff-{int(time.time())}"
        prompt_toks = len(prompt.split())

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "diffusiongemma-26B-A4B-it-Q4_K_M",
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": result["content"]},
                    "finish_reason": None,
                }],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())

            done_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "diffusiongemma-26B-A4B-it-Q4_K_M",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": prompt_toks,
                    "completion_tokens": result["tokens"],
                    "total_tokens": prompt_toks + result["tokens"],
                },
            }
            self.wfile.write(f"data: {json.dumps(done_chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            resp = {
                "id": chat_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "diffusiongemma-26B-A4B-it-Q4_K_M",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": result["content"]},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": prompt_toks,
                    "completion_tokens": result["tokens"],
                    "total_tokens": prompt_toks + result["tokens"],
                },
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-p", "--port", type=int, default=8090)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    DiffusionHandler.model_path = args.model
    server = HTTPServer((args.host, args.port), DiffusionHandler)
    print(f"Diffusion server ready on {args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
