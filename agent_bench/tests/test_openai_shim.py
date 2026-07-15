"""Shim tool-shape normalization for Chat Completions + Responses APIs."""

import json

from agent_bench.openai_anthropic_shim import (
    normalize_tools,
    anthropic_to_oai,
    oai_to_responses,
    responses_sse_events,
    completion_sse_lines,
)


def test_normalize_responses_and_chat_tools():
    tools = [
        {"type": "function", "name": "shell", "description": "run",
         "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}},
        {"type": "function", "function": {
            "name": "write", "description": "write",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        }},
        {"type": "function", "function": {}},  # dropped — no name
    ]
    out = normalize_tools(tools)
    assert [t["name"] for t in out] == ["shell", "write"]
    assert out[0]["input_schema"]["properties"]["command"]["type"] == "string"


def test_tool_use_maps_to_responses_function_call():
    anthropic = {
        "content": [{"type": "tool_use", "id": "tu1", "name": "shell", "input": {"command": "ls"}}],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }
    oai = anthropic_to_oai(anthropic, "m")
    assert oai["choices"][0]["finish_reason"] == "tool_calls"
    resp = oai_to_responses(oai, "m")
    assert any(x.get("type") == "function_call" and x.get("name") == "shell" for x in resp["output"])


def test_responses_sse_ends_with_completed():
    oai = {
        "choices": [{"message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    resp = oai_to_responses(oai, "m")
    events = responses_sse_events(resp)
    assert events[0][0] == "response.created"
    assert events[-1][0] == "response.completed"
    assert events[-1][1]["response"]["status"] == "completed"
    assert any(e[0] == "response.output_text.delta" for e in events)


def test_completion_sse_tool_calls_include_stream_index():
    oai = {
        "id": "chatcmpl-t",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "write", "arguments": "{\"path\":\"a\"}"},
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    lines = completion_sse_lines(oai, "m")
    assert lines[-1] == "data: [DONE]\n\n"
    payloads = [json.loads(line[6:]) for line in lines if line.startswith("data: {")]
    tool_deltas = [
        c["choices"][0]["delta"]["tool_calls"][0]
        for c in payloads
        if c["choices"][0].get("delta", {}).get("tool_calls")
    ]
    assert tool_deltas and tool_deltas[0]["index"] == 0
    assert tool_deltas[0]["function"]["name"] == "write"
