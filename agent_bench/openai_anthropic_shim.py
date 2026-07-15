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


def completion_sse_lines(oai: dict, default_model: str = "model") -> list[str]:
    """OpenAI chat.completion → SSE data lines (incl. trailing [DONE]).

    Tool-call deltas include required streaming ``index`` so clients like
    OpenClaw can assemble function calls from the stream.
    """
    cid = oai.get("id", f"chatcmpl-{uuid.uuid4().hex[:12]}")
    model = oai.get("model", default_model)
    msg = (oai.get("choices") or [{}])[0].get("message") or {}
    content = msg.get("content") or ""
    finish = (oai.get("choices") or [{}])[0].get("finish_reason") or "stop"
    lines: list[str] = []

    def emit(chunk: dict) -> None:
        lines.append(f"data: {json.dumps(chunk)}\n\n")

    emit({
        "id": cid, "object": "chat.completion.chunk", "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    })
    for i, tc in enumerate(msg.get("tool_calls") or []):
        fn = tc.get("function") or {}
        emit({
            "id": cid, "object": "chat.completion.chunk", "model": model,
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": i,
                        "id": tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": fn.get("name") or "tool",
                            "arguments": fn.get("arguments") or "{}",
                        },
                    }],
                },
                "finish_reason": None,
            }],
        })
    step = 40
    for i in range(0, max(len(content), 1 if content else 0), step):
        piece = content[i:i + step]
        emit({
            "id": cid, "object": "chat.completion.chunk", "model": model,
            "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
        })
    emit({
        "id": cid, "object": "chat.completion.chunk", "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
        "usage": oai.get("usage"),
    })
    lines.append("data: [DONE]\n\n")
    return lines


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


def cmd_alpha_events_from_oai(oai: dict) -> list[dict]:
    """Map an OpenAI chat.completion into Command Code /alpha/generate NDJSON events."""
    msg = (oai.get("choices") or [{}])[0].get("message") or {}
    usage = oai.get("usage") or {}
    events: list[dict] = []
    text = msg.get("content") or ""
    if text:
        step = 48
        for i in range(0, len(text), step):
            events.append({"type": "text-delta", "text": text[i:i + step]})
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments") or "{}"
        if isinstance(args, str):
            try:
                inp = json.loads(args)
            except json.JSONDecodeError:
                inp = {"raw": args}
        else:
            inp = args
        events.append({
            "type": "tool-call",
            "toolCallId": tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
            "toolName": fn.get("name") or "tool",
            "input": inp,
        })
    finish = (oai.get("choices") or [{}])[0].get("finish_reason") or "stop"
    events.append({
        "type": "finish",
        "finishReason": "tool-calls" if finish == "tool_calls" else finish,
        "rawFinishReason": finish,
        "totalUsage": {
            "inputTokens": usage.get("prompt_tokens", 0),
            "outputTokens": usage.get("completion_tokens", 0),
            "inputTokenDetails": {
                "cacheReadTokens": 0,
                "cacheWriteTokens": 0,
            },
        },
    })
    return events


def responses_sse_events(resp: dict) -> list[tuple[str, dict]]:
    """Minimal Responses SSE sequence ending in response.completed (Codex)."""
    rid = resp.get("id") or f"resp-{uuid.uuid4().hex[:12]}"
    events: list[tuple[str, dict]] = [
        ("response.created", {
            "type": "response.created",
            "response": {**resp, "status": "in_progress", "output": []},
        }),
        ("response.in_progress", {
            "type": "response.in_progress",
            "response": {**resp, "status": "in_progress", "output": []},
        }),
    ]
    for idx, item in enumerate(resp.get("output") or []):
        events.append(("response.output_item.added", {
            "type": "response.output_item.added",
            "output_index": idx,
            "item": {**item, "status": "in_progress"} if item.get("type") == "message" else item,
        }))
        if item.get("type") == "message":
            for part in item.get("content") or []:
                if part.get("type") != "output_text":
                    continue
                text = part.get("text") or ""
                events.append(("response.content_part.added", {
                    "type": "response.content_part.added",
                    "output_index": idx,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": ""},
                }))
                step = 40
                for i in range(0, max(len(text), 1 if text else 0), step):
                    events.append(("response.output_text.delta", {
                        "type": "response.output_text.delta",
                        "output_index": idx,
                        "content_index": 0,
                        "delta": text[i:i + step],
                    }))
                events.append(("response.output_text.done", {
                    "type": "response.output_text.done",
                    "output_index": idx,
                    "content_index": 0,
                    "text": text,
                }))
                events.append(("response.content_part.done", {
                    "type": "response.content_part.done",
                    "output_index": idx,
                    "content_index": 0,
                    "part": part,
                }))
        elif item.get("type") == "function_call":
            args = item.get("arguments") or "{}"
            events.append(("response.function_call_arguments.delta", {
                "type": "response.function_call_arguments.delta",
                "output_index": idx,
                "delta": args if isinstance(args, str) else json.dumps(args),
            }))
            events.append(("response.function_call_arguments.done", {
                "type": "response.function_call_arguments.done",
                "output_index": idx,
                "arguments": args if isinstance(args, str) else json.dumps(args),
            }))
        events.append(("response.output_item.done", {
            "type": "response.output_item.done",
            "output_index": idx,
            "item": item,
        }))
    events.append(("response.completed", {
        "type": "response.completed",
        "response": {**resp, "id": rid, "status": "completed"},
    }))
    return events


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
        elif path in (
            "/v1/models",
            "/models",
            "/provider/v1/models",  # Command Code sandbox API shape
        ):
            self._send_json(200, {
                "object": "list",
                "data": [{
                    "id": self.default_model,
                    "object": "model",
                    "owned_by": "kevlar-shim",
                }],
            })
        elif path in ("/alpha/whoami", "/whoami"):
            # Minimal stub so Command Code sandbox mode can start against this shim.
            self._send_json(200, {
                "user": {"id": "local", "userName": "local", "email": "local@localhost"},
                "org": None,
                "plan": "local",
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
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            for line in completion_sse_lines(oai, self.default_model):
                self.wfile.write(line.encode())
            self.wfile.flush()
        except BrokenPipeError:
            return

    def _send_sse_responses(self, resp: dict) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            for event, payload in responses_sse_events(resp):
                self.wfile.write(f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode())
            self.wfile.flush()
        except BrokenPipeError:
            return

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

    def _extract_cmd_generate_body(self, body: dict) -> dict:
        """Command Code posts {params:{model,messages,tools,...}, ...} to /alpha/generate."""
        params = body.get("params") if isinstance(body.get("params"), dict) else {}
        merged = {**params}
        # Flatten top-level overrides if present
        for key in ("model", "messages", "tools", "system", "max_tokens", "stream", "temperature"):
            if key in body and body[key] is not None and key not in merged:
                merged[key] = body[key]
        # Always force ThinkingCap for local sandbox routing
        merged["model"] = self.default_model
        # Cap tokens — Command Code defaults to 64k which hurts local MLX latency
        mt = int(merged.get("max_tokens") or 4096)
        merged["max_tokens"] = min(mt, 4096)
        if merged.get("system") and isinstance(merged.get("messages"), list):
            messages = list(merged["messages"])
            messages.insert(0, {"role": "system", "content": merged["system"]})
            merged["messages"] = messages
        return merged

    def _send_cmd_alpha_ndjson(self, oai: dict) -> None:
        payload = "".join(json.dumps(event) + "\n" for event in cmd_alpha_events_from_oai(oai)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)
        self.wfile.flush()

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")

        # Command Code sandbox stubs / agent generate bridge
        if path in (
            "/alpha/lifecycle-events",
            "/alpha/fingerprint/record",
            "/alpha/conversions/track",
            "/alpha/consent/set",
        ):
            self._send_json(200, {"ok": True})
            return

        if path in ("/alpha/generate", "/alpha/agent/generate"):
            chat_body = self._extract_cmd_generate_body(body)
            code, payload = self._call_anthropic(chat_body)
            if code != 200:
                self._send_json(code, payload)
                return
            self._send_cmd_alpha_ndjson(payload)
            return

        if path in ("/provider/v1/messages", "/v1/messages"):
            # Anthropic-shaped path — pass through almost as OpenAI chat via conversion
            # by packaging as chat.completions-like body.
            msgs = body.get("messages") or []
            if body.get("system"):
                msgs = [{"role": "system", "content": body["system"]}, *msgs]
            chat_body = {
                "model": self.default_model,
                "messages": msgs,
                "max_tokens": min(int(body.get("max_tokens") or 4096), 4096),
                "tools": body.get("tools") or [],
            }
            code, payload = self._call_anthropic(chat_body)
            self._send_json(code, payload if code != 200 else {
                "id": payload.get("id"),
                "type": "message",
                "role": "assistant",
                "model": self.default_model,
                "content": [{"type": "text", "text": ((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or ""}],
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": (payload.get("usage") or {}).get("prompt_tokens", 0),
                    "output_tokens": (payload.get("usage") or {}).get("completion_tokens", 0),
                },
            })
            return

        if path in ("/v1/responses", "/responses"):
            chat_body = self._responses_to_chat_body(body)
            code, payload = self._call_anthropic(chat_body)
            if code != 200:
                self._send_json(code, payload)
                return
            resp = oai_to_responses(payload, chat_body.get("model") or self.default_model)
            # Codex always streams Responses and requires response.completed
            if chat_body.get("stream") or "text/event-stream" in (self.headers.get("Accept") or ""):
                self._send_sse_responses(resp)
            else:
                self._send_json(200, resp)
            return

        if path not in (
            "/v1/chat/completions",
            "/chat/completions",
            "/provider/v1/chat/completions",  # Command Code sandbox
        ):
            self._send_json(404, {"detail": "Not Found"})
            return

        want_stream = bool(body.get("stream"))
        # Prefer configured ThinkingCap when client sends a catalog/OSS model id
        if not (body.get("model") or "").startswith("t-prazak/"):
            body = {**body, "model": self.default_model}
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
