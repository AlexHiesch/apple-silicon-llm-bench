#!/usr/bin/env python3
"""Minimal OpenAI-compatible mock for Docker smoke tests (no GPU / no model weights)."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

MODEL = "t-prazak/ThinkingCap-Qwen3.6-27B-MLX-4bit"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[mock-llm] {self.address_string()} - {fmt % args}")

    def _json(self, code: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._json(200, {
                "object": "list",
                "data": [{"id": MODEL, "object": "model", "owned_by": "mock"}],
            })
            return
        self._json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        _ = self.rfile.read(length)
        if self.path.rstrip("/").endswith("/chat/completions"):
            self._json(200, {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "model": MODEL,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "pong"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            })
            return
        self._json(404, {"error": "not found"})


if __name__ == "__main__":
    HTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
