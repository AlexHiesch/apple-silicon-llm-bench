#!/usr/bin/env python3
"""
Thin OpenAI-compatible server wrapper around dflash MLX stream_generate().

Uses Python's http.server (same pattern as mlx_lm.server) to avoid
thread-local MLX stream issues with async frameworks.

Usage:
  python3 dflash_server.py --model mlx-community/gemma-4-26b-a4b-it-4bit \
    --draft-model z-lab/gemma-4-26B-A4B-it-DFlash --port 8090
"""
import argparse
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Lock

# Globals populated at startup
MODEL = None
DRAFT = None
TOKENIZER = None
MODEL_ID = ""
BLOCK_SIZE = None
GEN_LOCK = Lock()


class DFlashHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress request logging

    def do_GET(self):
        if self.path == "/v1/models":
            body = json.dumps({"data": [{"id": MODEL_ID, "object": "model"}]})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body.encode())
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length))
        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", 512)
        temperature = body.get("temperature", 0.0)
        stream = body.get("stream", True)

        prompt = TOKENIZER.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        if stream:
            self._stream_response(prompt, max_tokens, temperature, chat_id, created)
        else:
            self._full_response(prompt, max_tokens, temperature, chat_id, created)

    def _stream_response(self, prompt, max_tokens, temperature, chat_id, created):
        from dflash.model_mlx import stream_generate

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        total_tokens = 0
        prompt_tokens = 0

        with GEN_LOCK:
            for resp in stream_generate(
                MODEL, DRAFT, TOKENIZER, prompt,
                block_size=BLOCK_SIZE,
                max_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 0.0,
            ):
                if not resp.text:
                    continue
                total_tokens = resp.generation_tokens
                prompt_tokens = resp.prompt_tokens
                chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_ID,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": resp.text},
                        "finish_reason": resp.finish_reason,
                    }],
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()

        # Usage chunk
        usage_chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_ID,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": total_tokens,
                "total_tokens": prompt_tokens + total_tokens,
            },
        }
        self.wfile.write(f"data: {json.dumps(usage_chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _full_response(self, prompt, max_tokens, temperature, chat_id, created):
        from dflash.model_mlx import stream_generate

        text_parts = []
        total_tokens = 0
        prompt_tokens = 0

        with GEN_LOCK:
            for resp in stream_generate(
                MODEL, DRAFT, TOKENIZER, prompt,
                block_size=BLOCK_SIZE,
                max_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 0.0,
            ):
                if resp.text:
                    text_parts.append(resp.text)
                total_tokens = resp.generation_tokens
                prompt_tokens = resp.prompt_tokens

        response = {
            "id": chat_id,
            "object": "chat.completion",
            "created": created,
            "model": MODEL_ID,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "".join(text_parts)},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": total_tokens,
                "total_tokens": prompt_tokens + total_tokens,
            },
        }
        body = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    global MODEL, DRAFT, TOKENIZER, MODEL_ID, BLOCK_SIZE

    parser = argparse.ArgumentParser(description="dflash OpenAI-compat server")
    parser.add_argument("--model", required=True, help="Target model HF ID")
    parser.add_argument("--draft-model", required=True, help="DFlash draft model HF ID")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--block-size", type=int, default=None)
    args = parser.parse_args()

    from dflash.model_mlx import load, load_draft

    MODEL_ID = args.model
    BLOCK_SIZE = args.block_size
    print(f"[dflash] Loading target: {args.model}", flush=True)
    MODEL, TOKENIZER = load(args.model)
    print(f"[dflash] Loading draft: {args.draft_model}", flush=True)
    DRAFT = load_draft(args.draft_model)
    print(f"[dflash] Ready (block_size={DRAFT.config.block_size})", flush=True)

    server = HTTPServer((args.host, args.port), DFlashHandler)
    print(f"[dflash] Serving on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
