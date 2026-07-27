from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    endpoint: str
    key_env: str
    timeout_seconds: float = Field(default=120.0, gt=0)
    api_style: Literal["openai_chat", "gemini_generate_content"] = "openai_chat"


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    model_id: str
    hard_token_budget: int = Field(gt=0)
    daily_token_budget_per_key: int | None = Field(default=None, gt=0)
    daily_request_budget_per_key: int | None = Field(default=None, gt=0)
    requests_per_minute_per_key: int | None = Field(default=None, gt=0)
    daily_reset_timezone: str = "UTC"
    request_options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def reject_placeholder(self) -> "ModelConfig":
        # Loading configuration is valid before credentials/model IDs are supplied;
        # dispatch checks this value and fails before network access.
        return self


class BackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    providers: dict[str, ProviderConfig]
    models: dict[str, ModelConfig]
    cache_path: str
    ledger_path: str


class CompletionRequest(BaseModel):
    """All fields needed to produce a deterministic experiment cache key."""

    model_config = ConfigDict(extra="forbid")

    model_name: str
    system_prompt: str
    user_prompt: str
    dataset_example_id: str
    method: str
    schema_version: str = "none"
    ablation: str = "none"
    code_version: str = "unknown"
    temperature: float = 0.0
    max_output_tokens: int = Field(default=512, gt=0)
    response_format: Literal["text", "json_object"] = "text"
    metadata: dict[str, Any] = Field(default_factory=dict)

    def cache_payload(
        self, provider: str, model_id: str, model_request_options: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return {
            "provider": provider,
            "model": model_id,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "decoding": {
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
                "response_format": self.response_format,
                "model_request_options": model_request_options or {},
            },
            "schema_version": self.schema_version,
            "dataset_example_id": self.dataset_example_id,
            "method": self.method,
            "ablation": self.ablation,
            "code_version": self.code_version,
        }


class TokenUsage(BaseModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class CompletionResult(BaseModel):
    text: str
    provider: str
    model_id: str
    key_alias: str
    request_id: str | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_seconds: float = Field(ge=0)
    retry_count: int = Field(default=0, ge=0)
    cache_key: str
    cache_hit: bool = False
    parsed_json: dict[str, Any] | list[Any] | None = None


class KeyRuntimeState(BaseModel):
    alias: str
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    failures: int = 0
    retries: int = 0
    cooldown_until: float = 0.0
