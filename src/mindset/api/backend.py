from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ValidationError

from mindset.api.cache import RequestCache
from mindset.api.ledger import TokenLedger
from mindset.api.models import (
    BackendConfig,
    CompletionRequest,
    CompletionResult,
    ModelConfig,
    ProviderConfig,
    TokenUsage,
)
from mindset.api.tokenizer import TokenCounter
from mindset.utils.config import load_config, load_dotenv
from mindset.utils.hashing import stable_hash

StructuredT = TypeVar("StructuredT", bound=BaseModel)


class ProviderRequestError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None, retry_after: float = 0.0):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after

    @property
    def is_quota_or_rate_limit(self) -> bool:
        return self.status_code in {429, 402}

    @property
    def retryable(self) -> bool:
        return self.status_code is None or self.status_code in {408, 409, 429, 500, 502, 503, 504}


class ProviderDispatchError(RuntimeError):
    """A terminal dispatch failure that may still have incurred provider tokens."""

    def __init__(self, message: str, usage: TokenUsage):
        super().__init__(message)
        self.usage = usage


class DailyQuotaExhausted(RuntimeError):
    """All configured keys reached their local provider-day allowance."""


@dataclass(frozen=True)
class KeyEntry:
    alias: str
    secret: str


def _keys_from_environment(base_name: str) -> list[KeyEntry]:
    values: list[str] = []
    combined = os.getenv(base_name, "")
    values.extend(part.strip() for part in combined.split(",") if part.strip())
    singular = base_name[:-1] if base_name.endswith("S") else base_name
    if os.getenv(singular, "").strip():
        values.append(os.environ[singular].strip())
    prefix = singular + "_"
    numbered = sorted(
        [
            (name, value.strip())
            for name, value in os.environ.items()
            if name.startswith(prefix)
            and name[len(prefix) :].isdigit()
            and value.strip()
        ],
        key=lambda item: int(item[0][len(prefix) :]),
    )
    values.extend(value for _, value in numbered)
    unique = list(dict.fromkeys(values))
    return [KeyEntry(alias=f"key_{index:02d}", secret=value) for index, value in enumerate(unique, 1)]


class APIBackend:
    """One logical, cached, budgeted pool over all configured provider keys."""

    def __init__(self, config: BackendConfig, *, rng_seed: int = 0) -> None:
        self.config = config
        self.cache = RequestCache(config.cache_path)
        self.ledger = TokenLedger(config.ledger_path)
        self.rng = random.Random(rng_seed)
        self._round_robin: dict[str, int] = {}
        self._last_key_dispatch: dict[tuple[str, str], float] = {}
        self._token_counters: dict[str, TokenCounter] = {}

    @classmethod
    def from_file(cls, path: str | Path) -> "APIBackend":
        load_dotenv()
        return cls(BackendConfig.model_validate(load_config(path)))

    def complete(
        self,
        request: CompletionRequest,
        *,
        structured_model: type[StructuredT] | None = None,
        max_retries: int = 4,
    ) -> CompletionResult:
        model, provider = self._resolve(request.model_name)
        cache_key = stable_hash(
            request.cache_payload(model.provider, model.model_id, model.request_options)
        )
        owner = uuid.uuid4().hex
        claim = self.cache.claim(cache_key, owner)
        if claim.state == "hit":
            assert claim.result is not None
            self._log_result(request, claim.result, model, cache_hit=True)
            return claim.result
        if claim.state == "busy":
            cached = self.cache.wait_for_result(cache_key, timeout_seconds=provider.timeout_seconds)
            if cached is not None:
                self._log_result(request, cached, model, cache_hit=True)
                return cached
            claim = self.cache.claim(cache_key, owner)
            if claim.state != "claimed":
                raise RuntimeError("Unable to obtain cache dispatch lease")

        try:
            result = self._dispatch_with_retries(
                request,
                cache_key,
                model,
                provider,
                structured_model=structured_model,
                max_retries=max_retries,
            )
            self.cache.store(cache_key, claim.owner, result)
            self._log_result(request, result, model, cache_hit=False)
            return result
        except ProviderDispatchError as exc:
            self.cache.release(cache_key, claim.owner)
            raise
        except Exception:
            self.cache.release(cache_key, claim.owner)
            raise

    def _resolve(self, model_name: str) -> tuple[ModelConfig, ProviderConfig]:
        if model_name not in self.config.models:
            raise KeyError(f"Unknown model alias: {model_name}")
        model = self.config.models[model_name]
        if model.model_id == "CONFIGURE_ME":
            raise RuntimeError(f"Model ID for {model_name} is not configured")
        if model.provider not in self.config.providers:
            raise KeyError(f"Unknown provider: {model.provider}")
        return model, self.config.providers[model.provider]

    def _dispatch_with_retries(
        self,
        request: CompletionRequest,
        cache_key: str,
        model: ModelConfig,
        provider: ProviderConfig,
        *,
        structured_model: type[StructuredT] | None,
        max_retries: int,
    ) -> CompletionResult:
        keys = _keys_from_environment(provider.key_env)
        if not keys:
            raise RuntimeError(f"No API keys found in {provider.key_env} (raw values are not logged)")
        last_error: Exception | None = None
        total_usage = TokenUsage()
        user_prompt = request.user_prompt
        for attempt in range(max_retries + 1):
            model_reservation: str | None = None
            while True:
                try:
                    estimated_attempt = (
                        self._estimate_tokens(
                            request.system_prompt + user_prompt, model.model_id
                        )
                        + request.max_output_tokens
                    )
                    model_reservation = self.ledger.reserve(
                        request.model_name,
                        estimated_attempt,
                        model.hard_token_budget,
                    )
                    key, daily_reservation = self._select_key(
                        request.model_name, model, keys, estimated_attempt
                    )
                    break
                except ProviderRequestError as exc:
                    if model_reservation is not None:
                        self.ledger.release(model_reservation)
                        model_reservation = None
                    last_error = exc
                    # Waiting for the pool is not a provider attempt and must not
                    # consume the malformed/HTTP retry allowance.
                    time.sleep(min(300.0, max(exc.retry_after, 1.0)))
                except Exception:
                    if model_reservation is not None:
                        self.ledger.release(model_reservation)
                        model_reservation = None
                    raise
            started = time.perf_counter()
            attempt_usage = TokenUsage()
            daily_settled = False
            model_settled = False
            try:
                raw = self._post(
                    provider,
                    model.model_id,
                    key,
                    request,
                    user_prompt,
                    model.request_options,
                )
                result = self._parse_response(
                    raw,
                    provider=model.provider,
                    model_id=model.model_id,
                    key_alias=key.alias,
                    cache_key=cache_key,
                    latency=time.perf_counter() - started,
                    retry_count=attempt,
                )
                attempt_usage = result.usage
                total_usage = TokenUsage(
                    input_tokens=total_usage.input_tokens + attempt_usage.input_tokens,
                    output_tokens=total_usage.output_tokens + attempt_usage.output_tokens,
                )
                if not result.text.strip():
                    raise ValueError("Provider response content is empty")
                if structured_model is not None:
                    parsed = self._extract_json(result.text)
                    structured_model.model_validate(parsed)
                    result = result.model_copy(update={"parsed_json": parsed})
                self.ledger.settle(model_reservation, attempt_usage)
                model_settled = True
                self.ledger.record_key_result(model.provider, key.alias, attempt_usage, retry=attempt > 0)
                self.ledger.settle_daily_key(daily_reservation, attempt_usage)
                daily_settled = True
                return result.model_copy(update={"usage": total_usage})
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = exc
                if not model_settled:
                    self.ledger.settle(model_reservation, attempt_usage)
                    model_settled = True
                self.ledger.record_key_result(
                    model.provider,
                    key.alias,
                    attempt_usage,
                    failure=True,
                    retry=attempt < max_retries,
                )
                self.ledger.settle_daily_key(daily_reservation, attempt_usage)
                daily_settled = True
                if structured_model is None or attempt >= max_retries:
                    break
                if structured_model is not None:
                    user_prompt = (
                        request.user_prompt
                        + "\n\nYour previous response failed strict schema validation. Return only one valid "
                        + "JSON value matching this schema exactly:\n"
                        + json.dumps(structured_model.model_json_schema(), sort_keys=True)
                    )
            except ProviderRequestError as exc:
                last_error = exc
                if not model_settled:
                    self.ledger.settle(model_reservation, attempt_usage)
                    model_settled = True
                cooldown = 0.0
                if exc.is_quota_or_rate_limit:
                    cooldown = time.time() + max(exc.retry_after, min(300.0, 2**attempt))
                self.ledger.record_key_result(
                    model.provider,
                    key.alias,
                    failure=True,
                    retry=attempt < max_retries,
                    cooldown_until=cooldown,
                )
                self.ledger.settle_daily_key(daily_reservation, attempt_usage)
                daily_settled = True
                if not exc.retryable or attempt >= max_retries:
                    break
            finally:
                if not daily_settled:
                    self.ledger.settle_daily_key(daily_reservation, attempt_usage)
                if not model_settled:
                    self.ledger.release(model_reservation)
            delay = min(300.0, (2**attempt) + self.rng.random())
            if isinstance(last_error, ProviderRequestError):
                delay = max(delay, min(300.0, last_error.retry_after))
            time.sleep(delay)
        raise ProviderDispatchError(
            f"Provider request failed after {max_retries + 1} attempts: {last_error}",
            total_usage,
        )

    def _select_key(
        self,
        model_name: str,
        model: ModelConfig,
        keys: list[KeyEntry],
        estimated_tokens: int,
    ) -> tuple[KeyEntry, str | None]:
        now = time.time()
        available = [
            key
            for key in keys
            if self.ledger.key_cooldown(model.provider, key.alias) <= now
        ]
        if not available:
            earliest = min(keys, key=lambda item: self.ledger.key_cooldown(model.provider, item.alias))
            wait = self.ledger.key_cooldown(model.provider, earliest.alias) - now
            raise ProviderRequestError(
                f"All {model.provider} key aliases are cooling down for at least {max(wait, 0):.1f}s",
                status_code=429,
                retry_after=max(wait, 0),
            )
        start = self._round_robin.get(model_name, 0) % len(available)
        period = datetime.now(ZoneInfo(model.daily_reset_timezone)).date().isoformat()
        pacing_waits: list[float] = []
        for offset in range(len(available)):
            key = available[(start + offset) % len(available)]
            rpm = model.requests_per_minute_per_key
            if rpm is not None:
                interval = 60.0 / float(rpm)
                wait = self._last_key_dispatch.get((model.provider, key.alias), 0.0) + interval - now
                if wait > 0:
                    pacing_waits.append(wait)
                    continue
            reservation = self.ledger.reserve_daily_key(
                model_name=model_name,
                provider=model.provider,
                key_alias=key.alias,
                period=period,
                estimated_tokens=estimated_tokens,
                token_budget=model.daily_token_budget_per_key,
                request_budget=model.daily_request_budget_per_key,
            )
            if reservation != "":
                self._round_robin[model_name] = start + offset + 1
                self._last_key_dispatch[(model.provider, key.alias)] = time.time()
                return key, reservation
        if pacing_waits:
            wait = min(pacing_waits)
            raise ProviderRequestError(
                f"All {model.provider} keys are paced at {model.requests_per_minute_per_key} requests/minute",
                status_code=429,
                retry_after=wait,
            )
        raise DailyQuotaExhausted(
            f"All {len(keys)} {model_name} keys exhausted the {period} daily allowance"
        )

    @staticmethod
    def _post(
        provider: ProviderConfig,
        model_id: str,
        key: KeyEntry,
        request: CompletionRequest,
        user_prompt: str,
        request_options: dict[str, Any],
    ) -> dict[str, Any]:
        if provider.api_style == "gemini_generate_content":
            return APIBackend._post_gemini(
                provider, model_id, key, request, user_prompt, request_options
            )
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
        }
        if request.response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}
        payload.update(request_options)
        web_request = urllib.request.Request(
            provider.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {key.secret}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "mindset-memory/0.1 (+research client)",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(web_request, timeout=provider.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            retry_after = float(exc.headers.get("Retry-After", "0") or 0)
            body = exc.read().decode("utf-8", errors="replace")[:500]
            if exc.code == 429:
                retry_after = max(60.0, retry_after)
            # Groq token limits are rolling minute windows; use a full cooldown.
            # This prevents rapid retries from consuming more quota.
            # Body is provider error text only; never include request headers.
            raise ProviderRequestError(
                f"HTTP {exc.code}: {body}", status_code=exc.code, retry_after=retry_after
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ProviderRequestError(f"Transport error: {exc}") from exc

    @staticmethod
    def _post_gemini(
        provider: ProviderConfig,
        model_id: str,
        key: KeyEntry,
        request: CompletionRequest,
        user_prompt: str,
        request_options: dict[str, Any],
    ) -> dict[str, Any]:
        generation_config: dict[str, Any] = {
            "temperature": request.temperature,
            "maxOutputTokens": request.max_output_tokens,
            **request_options,
        }
        if request.response_format == "json_object":
            generation_config["responseMimeType"] = "application/json"
        payload = {
            "systemInstruction": {"parts": [{"text": request.system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": generation_config,
        }
        endpoint = provider.endpoint.format(model=model_id)
        web_request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "x-goog-api-key": key.secret,
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "mindset-memory/0.1 (+research client)",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(web_request, timeout=provider.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            retry_after = float(exc.headers.get("Retry-After", "0") or 0)
            body = exc.read().decode("utf-8", errors="replace")[:500]
            if exc.code == 429:
                retry_after = max(60.0, retry_after)
            raise ProviderRequestError(
                f"HTTP {exc.code}: {body}", status_code=exc.code, retry_after=retry_after
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ProviderRequestError(f"Transport error: {exc}") from exc

    @staticmethod
    def _parse_response(
        payload: dict[str, Any],
        *,
        provider: str,
        model_id: str,
        key_alias: str,
        cache_key: str,
        latency: float,
        retry_count: int,
    ) -> CompletionResult:
        if "candidates" in payload or "usageMetadata" in payload:
            candidates = payload.get("candidates") or []
            if not candidates:
                raise ValueError(f"Gemini response contains no candidates: {payload.get('promptFeedback')}")
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(str(part.get("text", "")) for part in parts)
            usage = payload.get("usageMetadata") or {}
            return CompletionResult(
                text=text,
                provider=provider,
                model_id=model_id,
                key_alias=key_alias,
                request_id=payload.get("responseId"),
                usage=TokenUsage(
                    input_tokens=int(usage.get("promptTokenCount", 0) or 0),
                    output_tokens=int(usage.get("candidatesTokenCount", 0) or 0),
                ),
                latency_seconds=latency,
                retry_count=retry_count,
                cache_key=cache_key,
            )
        choices = payload.get("choices") or []
        if not choices:
            raise ValueError("Provider response contains no choices")
        text = choices[0].get("message", {}).get("content")
        # Some reasoning endpoints return null content when the output cap is
        # exhausted by hidden reasoning. Preserve usage before retrying.
        if text is None:
            text = ""
        if not isinstance(text, str):
            text = str(text)
        usage = payload.get("usage") or {}
        return CompletionResult(
            text=text,
            provider=provider,
            model_id=model_id,
            key_alias=key_alias,
            request_id=payload.get("id"),
            usage=TokenUsage(
                input_tokens=int(usage.get("prompt_tokens", 0) or 0),
                output_tokens=int(usage.get("completion_tokens", 0) or 0),
            ),
            latency_seconds=latency,
            retry_count=retry_count,
            cache_key=cache_key,
        )

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | list[Any]:
        stripped = text.strip()
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.IGNORECASE)
        parsed = json.loads(stripped)
        if not isinstance(parsed, (dict, list)):
            raise ValueError("Structured response must be a JSON object or array")
        return parsed

    def _estimate_tokens(self, text: str, model_id: str | None = None) -> int:
        model_key = model_id or "__default__"
        counter = self._token_counters.get(model_key)
        if counter is None:
            counter = TokenCounter(model_id)
            self._token_counters[model_key] = counter
        return counter.count(text)

    def _log_result(
        self,
        request: CompletionRequest,
        result: CompletionResult,
        model: ModelConfig,
        *,
        cache_hit: bool,
    ) -> None:
        self.ledger.log_request(
            cache_key=result.cache_key,
            provider=model.provider,
            model_name=request.model_name,
            model_id=model.model_id,
            key_alias=result.key_alias if not cache_hit else "cache",
            request_id=result.request_id,
            input_tokens=0 if cache_hit else result.usage.input_tokens,
            output_tokens=0 if cache_hit else result.usage.output_tokens,
            latency_seconds=0.0 if cache_hit else result.latency_seconds,
            retry_count=result.retry_count,
            cache_hit=cache_hit,
            metadata={
                "dataset_example_id": request.dataset_example_id,
                "method": request.method,
                "ablation": request.ablation,
            },
        )
