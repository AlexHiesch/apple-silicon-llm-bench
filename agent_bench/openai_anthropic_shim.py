#!/usr/bin/env python3
"""OpenAI-compatible HTTP shim → Anthropic /v1/messages (Kevlar ThinkingCap).

Supports:
  POST /v1/chat/completions  (JSON + SSE stream)
  POST /v1/responses         (Codex / Responses API)
  GET  /v1/models, /health
"""

from __future__ import annotations

import argparse
import json
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import error as urllib_error
from urllib import request as urllib_request


def strip_thinking(text: str) -> str:
    if not text:
        return ""
    if "</think>" in text:
        return text.split("</think>", 1)[-1].strip()
    # bare thinking without close tag — drop if clearly thinking dump
    if text.lstrip().startswith("<think>"):
        return ""
    return text.strip()


def oai_messages_to_anthropic(messages: list) -> tuple[str | None, list]:
    system_parts: list[str] = []
    out: list[dict] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, dict):
                    if p.get("type") in ("text", "output_text", "input_text"):
                        parts.append(p.get("text") or "")
                    elif p.get("type") == "tool_result":
                        out.append({
                            "role": "user",
                            "content": [{
                                "type": "tool_result",
                                "tool_use_id": p.get("tool_use_id") or p.get("tool_call_id") or "tool",
                                "content": str(p.get("content") or ""),
                            }],
                        })
                else:
                    parts.append(str(p))
            content = "\n".join(parts)
        # Chat Completions tool_calls → anthropic tool_use
        if role == "assistant" and m.get("tool_calls"):
            blocks = []
            if content:
                blocks.append({"type": "text", "text": str(content)})
            for tc in m["tool_calls"]:
                fn = tc.get("function") or {}
                args = fn.get("arguments") or "{}"
                if isinstance(args, str):
                    try:
                        args_obj = json.loads(args)
                    except json.JSONDecodeError:
                        args_obj = {"raw": args}
                else:
                    args_obj = args
                blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                    "name": fn.get("name") or "tool",
                    "input": args_obj,
                })
            out.append({"role": "assistant", "content": blocks})
            continue
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


def normalize_tools(tools: list) -> list[dict]:
    """Chat Completions + Responses tool shapes → Anthropic tools."""
    out: list[dict] = []
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else {}
        name = fn.get("name") or t.get("name")
        if not name or not isinstance(name, str):
            continue
        description = fn.get("description") or t.get("description") or ""
        params = (
            fn.get("parameters")
            or t.get("parameters")
            or t.get("input_schema")
            or {"type": "object", "properties": {}}
        )
        out.append({
            "name": name,
            "description": description,
            "input_schema": params,
        })
    return out


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
                    "name": b.get("name") or "tool",
                    "arguments": json.dumps(b.get("input", {})),
                },
            })

    msg: dict = {"role": "assistant", "content": text if text else (None if tool_calls else "")}
    if tool_calls:
        msg["tool_calls"] = tool_calls

    usage = data.get("usage", {})
    in_tok = usage.get("input_tokens", 0)
    out_tok = usage.get("output_tokens", 0)
    stop = data.get("stop_reason", "end_turn")
    if tool_calls:
        finish = "tool_calls"
    elif stop in ("end_turn", "stop_sequence"):
        finish = "stop"
    else:
        finish = stop or "stop"

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


def oai_to_responses(oai: dict, model: str) -> dict:
    msg = (oai.get("choices") or [{}])[0].get("message") or {}
    text = msg.get("content") or ""
    output: list[dict] = []
    if text:
        output.append({
            "type": "message",
            "id": f"msg-{uuid.uuid4().hex[:8]}",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text}],
        })
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments") or "{}"
        output.append({
            "type": "function_call",
            "id": tc.get("id") or f"fc_{uuid.uuid4().hex[:8]}",
            "call_id": tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
            "name": fn.get("name") or "tool",
            "arguments": args if isinstance(args, str) else json.dumps(args),
        })
    if not output:
        # Never return a completely empty response — clients treat that as failure.
        output.append({
            "type": "message",
            "id": f"msg-{uuid.uuid4().hex[:8]}",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": ""}],
        })
    return {
        "id": f"resp-{uuid.uuid4().hex[:12]}",
        "object": "response",
        "status": "completed",
        "model": model,
        "output": output,
        "usage": {
            "input_tokens": (oai.get("usage") or {}).get("prompt_tokens", 0),
            "output_tokens": (oai.get("usage") or {}).get("completion_tokens", 0),
            "total_tokens": (oai.get("usage") or {}).get("total_tokens", 0),
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
        elif path in ("/v1/models", "/models"):
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
        if not messages:
            messages = [{"role": "user", "content": "hello"}]
        anthropic_req: dict = {
            "model": model,
            "max_tokens": int(body.get("max_tokens") or body.get("max_output_tokens") or 4096),
            "messages": messages,
        }
        if system:
            anthropic_req["system"] = system
        if body.get("temperature") is not None:
            anthropic_req["temperature"] = body["temperature"]

        tools = normalize_tools(body.get("tools") or [])
        if tools:
            anthropic_req["tools"] = tools
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
        role_chunk = {
            "id": cid, "object": "chat.completion.chunk", "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        self.wfile.write(f"data: {json.dumps(role_chunk)}\n\n".encode())
        # tool_calls (single chunk)
        if msg.get("tool_calls"):
            chunk = {
                "id": cid, "object": "chat.completion.chunk", "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"tool_calls": msg["tool_calls"]},
                    "finish_reason": None,
                }],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        step = 40
        for i in range(0, max(len(content), 1 if content else 0), step):
            piece = content[i:i + step]
            chunk = {
                "id": cid, "object": "chat.completion.chunk", "model": model,
                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        done = {
            "id": cid, "object": "chat.completion.chunk", "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
            "usage": oai.get("usage"),
        }
        self.wfile.write(f"data: {json.dumps(done)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _responses_to_chat_body(self, body: dict) -> dict:
        messages = body.get("messages")
        if not messages:
            inp = body.get("input")
            if isinstance(inp, str):
                messages = [{"role": "user", "content": inp}]
            elif isinstance(inp, list):
                messages = []
                for item in inp:
                    if isinstance(item, str):
                        messages.append({"role": "user", "content": item})
                    elif isinstance(item, dict):
                        typ = item.get("type")
                        if typ == "function_call_output":
                            messages.append({
                                "role": "tool",
                                "tool_call_id": item.get("call_id") or item.get("id") or "tool",
                                "content": item.get("output") or item.get("content") or "",
                            })
                            continue
                        if typ == "function_call":
                            messages.append({
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [{
                                    "id": item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                                    "type": "function",
                                    "function": {
                                        "name": item.get("name") or "tool",
                                        "arguments": item.get("arguments") or "{}",
                                    },
                                }],
                            })
                            continue
                        role = item.get("role") or "user"
                        content = item.get("content") or item.get("text") or ""
                        if isinstance(content, list):
                            content = "".join(
                                (c.get("text") or "") if isinstance(c, dict) else str(c)
                                for c in content
                            )
                        messages.append({"role": role, "content": str(content)})
            else:
                messages = [{"role": "user", "content": "hello"}]
        out = {
            "model": body.get("model") or self.default_model,
            "messages": messages,
            "max_tokens": body.get("max_output_tokens") or body.get("max_tokens") or 4096,
            "stream": bool(body.get("stream")),
        }
        if body.get("temperature") is not None:
            out["temperature"] = body["temperature"]
        if body.get("tools"):
            out["tools"] = body["tools"]
        return out

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")

        if path in ("/v1/responses", "/responses"):
            chat_body = self._responses_to_chat_body(body)
            code, payload = self._call_anthropic(chat_body)
            if code != 200:
                self._send_json(code, payload)
                return
            self._send_json(200, oai_to_responses(payload, chat_body.get("model") or self.default_model))
            return

        if path not in ("/v1/chat/completions", "/chat/completions"):
            self._send_json(404, {"detail": "Not Found"})
            return

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
    parser = argparse.ArgumentParser(description="OpenAI chat/responses shim → Kevlar Anthropic API")
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
