"""LLM client: retries, shared throttling, endpoint-quirk learning."""

from __future__ import annotations

import threading
import time

import pytest

from doccat.llm import LLMClient, LLMConfig, LLMError, RateLimitGate, TokenBucket


def cfg(mock, **kw):
    kw.setdefault("max_retries", 3)
    kw.setdefault("max_rate_limit_retries", 5)
    kw.setdefault("timeout", 30)
    return LLMConfig(base_url=mock.base_url, model="mock", **kw)


def test_basic_completion(mock_llm):
    with LLMClient(cfg(mock_llm)) as client:
        assert "Mock title" in client.complete("catalogue this")


def test_profile_overrides_reach_the_endpoint(mock_llm):
    with LLMClient(cfg(mock_llm)) as client:
        client.complete("hi", model="other-model", temperature=0.7, max_tokens=123)
    payload = mock_llm.requests[-1]
    assert payload["model"] == "other-model"
    assert payload["temperature"] == 0.7
    assert payload["max_tokens"] == 123


def test_thinking_is_disabled_by_default(mock_llm):
    with LLMClient(cfg(mock_llm)) as client:
        client.complete("hi")
        assert mock_llm.requests[-1]["chat_template_kwargs"] == {"enable_thinking": False}
        client.complete("hi", think=True)
        assert "chat_template_kwargs" not in mock_llm.requests[-1]


def test_429_is_retried_until_it_succeeds(mock_llm):
    mock_llm.fail_with_429 = 2
    with LLMClient(cfg(mock_llm)) as client:
        assert "Mock title" in client.complete("catalogue this")
    assert mock_llm.call_count == 3


def test_429_budget_is_finite(mock_llm):
    mock_llm.fail_with_429 = 999
    with LLMClient(cfg(mock_llm, max_rate_limit_retries=2)) as client:
        with pytest.raises(LLMError, match="overloaded"):
            client.complete("catalogue this")


def test_one_workers_429_pauses_the_others(mock_llm):
    """The gate is shared, so a throttled endpoint is not stampeded."""
    gate = RateLimitGate()
    pause = gate.penalize(retry_after=1)
    assert pause >= 1
    started = time.monotonic()
    gate.wait()
    assert time.monotonic() - started >= 0.9


def test_token_bucket_limits_the_rate():
    bucket = TokenBucket(rate=20)
    started = time.monotonic()
    for _ in range(25):
        bucket.acquire()
    assert time.monotonic() - started >= 0.2


def test_client_is_safe_from_many_threads(mock_llm):
    errors: list[Exception] = []

    with LLMClient(cfg(mock_llm)) as client:
        def worker():
            try:
                for _ in range(5):
                    client.complete("catalogue this")
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert not errors
    assert mock_llm.call_count == 40
