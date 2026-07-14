#!/usr/bin/env python3
"""OpenAI-compatible HTTP shim → Anthropic /v1/messages (Kevlar ThinkingCap)."""

from __future__ import annotations

import argparse
import json
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import error as urllib_error
from urllib import request as urllib_request


def strip_thinking(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[-1].strip()
    return text.strip()


def oai_messages_to_anthropic(messages: list) -> tuple[str | None, list]:
    system_parts: list[str] = []
    out: list[dict] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        if role == "system":
            system_parts.append(str(content))
        elif role == "assistant":
            out.append({"role": "assistant", "content": str(content)})
        elif role == "tool":
            out.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": m.get("tool_call_id", "tool"),
                    "content": str(content),
                }],
            })
        else:
            out.append({"role": "user", "content": str(content)})
    system = "\n\n".join(system_parts) if system_parts else None
    return system, out


def anthropic_to_oai(data: dict, model: str) -> dict:
    content_blocks = data.get("content", [])
    text = "".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")
    text = strip_thinking(text)

    tool_calls = []
    for b in content_blocks:
        if b.get("type") == "tool_use":
            tool_calls.append({
                "id": b.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                "type": "function",
                "function": {
                    "name": b["name"],
                    "arguments": json.dumps(b.get("input", {})),
                },
            })

    msg: dict = {"role": "assistant", "content": text or None}
    if tool_calls:
        msg["tool_calls"] = tool_calls
        if not text:
            msg["content"] = None

    usage = data.get("usage", {})
    in_tok = usage.get("input_tokens", 0)
    out_tok = usage.get("output_tokens", 0)
    stop = data.get("stop_reason", "end_turn")
    finish = "stop" if stop in ("end_turn", "stop_sequence") else stop

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": msg,
            "finish_reason": finish,
        }],
        "usage": {
            "prompt_tokens": in_tok,
            "completion_tokens": out_tok,
            "total_tokens": in_tok + out_tok,
        },
    }


class ShimHandler(BaseHTTPRequestHandler):
    upstream: str = "http://127.0.0.1:8080"
    default_model: str = "t-prazak/ThinkingCap-Qwen3.6-27B-MLX-4bit"

    def log_message(self, fmt: str, *args) -> None:
        print(f"[shim] {fmt % args}", flush=True)

    def _send_json(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/health", "/v1/health"):
            self._send_json(200, {"status": "ok"})
        elif path == "/v1/models":
            self._send_json(200, {
                "object": "list",
                "data": [{
                    "id": self.default_model,
                    "object": "model",
                    "owned_by": "kevlar-shim",
                }],
            })
        else:
            self._send_json(404, {"detail": "Not Found"})

    def _call_anthropic(self, body: dict) -> tuple[int, dict]:
        model = body.get("model") or self.default_model
        system, messages = oai_messages_to_anthropic(body.get("messages", []))
        anthropic_req: dict = {
            "model": model,
            "max_tokens": body.get("max_tokens", 4096),
            "messages": messages,
        }
        if system:
            anthropic_req["system"] = system
        if body.get("temperature") is not None:
            anthropic_req["temperature"] = body["temperature"]

        tools = body.get("tools")
        if tools:
            anthropic_tools = []
            for t in tools:
                fn = t.get("function", {})
                anthropic_tools.append({
                    "name": fn.get("name"),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                })
            anthropic_req["tools"] = anthropic_tools
            anthropic_req["tool_choice"] = {"type": "auto"}

        url = f"{self.upstream.rstrip('/')}/v1/messages"
        req = urllib_request.Request(
            url,
            data=json.dumps(anthropic_req).encode(),
            headers={
                "Content-Type": "application/json",
                "x-api-key": "local",
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=600) as resp:
                anthropic_data = json.loads(resp.read().decode())
            return 200, anthropic_to_oai(anthropic_data, model)
        except urllib_error.HTTPError as e:
            raw = e.read().decode()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {"error": {"message": raw, "type": "api_error"}}
            return e.code, payload
        except Exception as e:
            return 502, {"error": {"message": str(e), "type": "shim_error"}}

    def _send_sse_completion(self, oai: dict) -> None:
        """Emit a one-shot OpenAI SSE stream after a full upstream completion."""
        cid = oai.get("id", f"chatcmpl-{uuid.uuid4().hex[:12]}")
        model = oai.get("model", self.default_model)
        msg = (oai.get("choices") or [{}])[0].get("message") or {}
        content = msg.get("content") or ""
        finish = (oai.get("choices") or [{}])[0].get("finish_reason") or "stop"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        # role chunk
        role_chunk = {
            "id": cid, "object": "chat.completion.chunk", "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        self.wfile.write(f"data: {json.dumps(role_chunk)}\n\n".encode())
        # content in ~40-char pieces (helps clients that expect progressive deltas)
        step = 40
        for i in range(0, max(len(content), 1), step):
            piece = content[i:i + step] if content else ""
            chunk = {
                "id": cid, "object": "chat.completion.chunk", "model": model,
                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        # final
        done = {
            "id": cid, "object": "chat.completion.chunk", "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
            "usage": oai.get("usage"),
        }
        self.wfile.write(f"data: {json.dumps(done)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path != "/v1/chat/completions":
            self._send_json(404, {"detail": "Not Found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        want_stream = bool(body.get("stream"))

        code, payload = self._call_anthropic(body)
        if code != 200:
            self._send_json(code, payload)
            return
        if want_stream:
            self._send_sse_completion(payload)
        else:
            self._send_json(200, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI chat shim → Kevlar Anthropic API")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--upstream", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="t-prazak/ThinkingCap-Qwen3.6-27B-MLX-4bit")
    args = parser.parse_args()

    ShimHandler.upstream = args.upstream
    ShimHandler.default_model = args.model

    server = ThreadingHTTPServer((args.host, args.port), ShimHandler)
    print(
        f"[shim] listening {args.host}:{args.port} → {args.upstream}/v1/messages model={args.model}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
