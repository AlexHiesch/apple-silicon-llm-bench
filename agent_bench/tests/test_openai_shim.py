"""Shim tool-shape normalization for Chat Completions + Responses APIs."""

from agent_bench.openai_anthropic_shim import normalize_tools, anthropic_to_oai, oai_to_responses


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
