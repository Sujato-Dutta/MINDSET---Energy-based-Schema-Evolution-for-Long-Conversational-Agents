from __future__ import annotations

import os

import pytest

from mindset.api.backend import APIBackend, DailyQuotaExhausted, ProviderDispatchError, ProviderRequestError
from mindset.api.cache import RequestCache
from mindset.api.ledger import TokenBudgetExceeded, TokenLedger
from mindset.api.models import (
    BackendConfig,
    CompletionRequest,
    CompletionResult,
    ModelConfig,
    ProviderConfig,
    TokenUsage,
)


def test_token_budget_hard_stop(tmp_path) -> None:
    ledger = TokenLedger(tmp_path / "ledger.sqlite")
    first = ledger.reserve("model", 8, 10)
    with pytest.raises(TokenBudgetExceeded, match="hard limit"):
        ledger.reserve("model", 3, 10)
    ledger.settle(first, TokenUsage(input_tokens=5, output_tokens=2))
    second = ledger.reserve("model", 3, 10)
    ledger.release(second)


def test_cache_claim_deduplicates_and_round_trips(tmp_path) -> None:
    cache = RequestCache(tmp_path / "cache.sqlite")
    first = cache.claim("same", "worker-a")
    second = cache.claim("same", "worker-b")
    assert first.state == "claimed"
    assert second.state == "busy"
    result = CompletionResult(
        text="answer",
        provider="test",
        model_id="model-id",
        key_alias="key_01",
        usage=TokenUsage(input_tokens=2, output_tokens=1),
        latency_seconds=0.1,
        cache_key="same",
    )
    cache.store("same", first.owner, result)
    hit = cache.claim("same", "worker-c")
    assert hit.state == "hit"
    assert hit.result is not None and hit.result.cache_hit


def test_key_rotation_and_cached_prompt_not_resent(tmp_path, monkeypatch) -> None:
    config = BackendConfig(
        providers={
            "test": ProviderConfig(
                endpoint="https://invalid.local/v1/chat/completions",
                key_env="TEST_API_KEYS",
                timeout_seconds=1,
            )
        },
        models={
            "main": ModelConfig(
                provider="test", model_id="test-model", hard_token_budget=1000
            )
        },
        cache_path=str(tmp_path / "cache.sqlite"),
        ledger_path=str(tmp_path / "ledger.sqlite"),
    )
    monkeypatch.setenv("TEST_API_KEYS", "secret-one,secret-two")
    calls: list[str] = []

    def fake_post(provider, model_id, key, request, user_prompt, request_options):
        calls.append(key.alias)
        if key.alias == "key_01":
            raise ProviderRequestError("rate limit", status_code=429)
        return {
            "id": "request-1",
            "choices": [{"message": {"content": "cached answer"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }

    monkeypatch.setattr(APIBackend, "_post", staticmethod(fake_post))
    monkeypatch.setattr("mindset.api.backend.time.sleep", lambda _: None)
    backend = APIBackend(config)
    request = CompletionRequest(
        model_name="main",
        system_prompt="system",
        user_prompt="question",
        dataset_example_id="example-1",
        method="mindset",
        max_output_tokens=32,
    )
    first = backend.complete(request, max_retries=2)
    second = backend.complete(request, max_retries=2)
    assert calls == ["key_01", "key_02"]
    assert first.key_alias == "key_02"
    assert second.cache_hit
    report = backend.ledger.report()
    assert report["models"][0]["input_tokens"] == 5
    assert report["models"][0]["output_tokens"] == 2


def test_raw_keys_are_not_stored_in_ledger(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SECRET_PROVIDER_KEYS", "raw-secret-value")
    ledger = TokenLedger(tmp_path / "ledger.sqlite")
    ledger.record_key_result("provider", "key_01", TokenUsage(input_tokens=1))
    assert b"raw-secret-value" not in (tmp_path / "ledger.sqlite").read_bytes()


def test_empty_unstructured_content_is_accounted_without_blind_retry(tmp_path, monkeypatch) -> None:
    config = BackendConfig(
        providers={
            "test": ProviderConfig(
                endpoint="https://invalid.local/v1/chat/completions",
                key_env="EMPTY_TEST_KEYS",
                timeout_seconds=1,
            )
        },
        models={
            "main": ModelConfig(provider="test", model_id="model", hard_token_budget=1000)
        },
        cache_path=str(tmp_path / "cache.sqlite"),
        ledger_path=str(tmp_path / "ledger.sqlite"),
    )
    monkeypatch.setenv("EMPTY_TEST_KEYS", "secret")
    calls = 0

    def fake_post(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return {
            "choices": [{"message": {"content": None}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 8},
        }

    monkeypatch.setattr(APIBackend, "_post", staticmethod(fake_post))
    monkeypatch.setattr("mindset.api.backend.time.sleep", lambda _: None)
    backend = APIBackend(config)
    with pytest.raises(ProviderDispatchError):
        backend.complete(
            CompletionRequest(
                model_name="main",
                system_prompt="s",
                user_prompt="q",
                dataset_example_id="e",
                method="test",
                max_output_tokens=16,
            ),
            max_retries=1,
        )
    assert calls == 1
    usage = backend.ledger.report()["models"][0]
    assert usage["input_tokens"] == 4
    assert usage["output_tokens"] == 8


def test_daily_key_budget_rotates_then_stops(tmp_path, monkeypatch) -> None:
    config = BackendConfig(
        providers={"test": ProviderConfig(endpoint="https://invalid", key_env="DAILY_KEYS")},
        models={
            "main": ModelConfig(
                provider="test", model_id="model", hard_token_budget=10_000,
                daily_request_budget_per_key=1,
            )
        },
        cache_path=str(tmp_path / "cache.sqlite"),
        ledger_path=str(tmp_path / "ledger.sqlite"),
    )
    monkeypatch.setenv("DAILY_KEYS", "one,two")
    calls = []

    def fake_post(provider, model_id, key, request, user_prompt, request_options):
        calls.append(key.alias)
        return {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    monkeypatch.setattr(APIBackend, "_post", staticmethod(fake_post))
    backend = APIBackend(config)
    for index in range(2):
        backend.complete(CompletionRequest(model_name="main", system_prompt="s", user_prompt="q", dataset_example_id=str(index), method="m"), max_retries=0)
    with pytest.raises(DailyQuotaExhausted):
        backend.complete(CompletionRequest(model_name="main", system_prompt="s", user_prompt="q", dataset_example_id="third", method="m"), max_retries=0)
    assert calls == ["key_01", "key_02"]


def test_gemini_response_parsing() -> None:
    result = APIBackend._parse_response(
        {"responseId": "g1", "candidates": [{"content": {"parts": [{"text": '{"verdict":"correct"}'}]}}], "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 2}},
        provider="gemini", model_id="gemini-3.1-flash-lite", key_alias="key_01",
        cache_key="c", latency=0.1, retry_count=0,
    )
    assert result.text == '{"verdict":"correct"}'
    assert result.usage.total_tokens == 6
