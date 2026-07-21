"""Retry policy: full-budget AgentTimeout is tech but not auto-retried."""

from agent_bench.tech_failures import (
    classify_result,
    is_retryable_tech,
    RETRYABLE_TECH_TYPES,
    EXHAUSTED_TIMEOUT_TYPES,
)


def _exc(et: str) -> dict:
    return {"exception_info": {"exception_type": et, "exception_message": et}}


def test_agent_timeout_is_tech_but_not_retryable():
    r = _exc("AgentTimeoutError")
    assert classify_result(r) == "tech"
    assert not is_retryable_tech(r)
    assert "AgentTimeoutError" in EXHAUSTED_TIMEOUT_TYPES
    assert "AgentTimeoutError" not in RETRYABLE_TECH_TYPES


def test_unknown_api_is_retryable():
    r = _exc("UnknownApiError")
    assert classify_result(r) == "tech"
    assert is_retryable_tech(r)


def test_content_fail_not_retryable_tech():
    r = {"verifier_result": {"rewards": {"reward": 0.0}}}
    assert classify_result(r) == "content_fail"
    assert not is_retryable_tech(r)
