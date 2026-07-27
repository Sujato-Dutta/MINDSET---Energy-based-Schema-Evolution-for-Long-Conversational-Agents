"""Best-effort local token counting with an explicit conservative fallback."""

from __future__ import annotations

import math
import os
from typing import Any


class TokenCounter:
    """Count tokens locally when a compatible tokenizer is available.

    The exact provider tokenizer is preferred.  A locally cached GPT-2 tokenizer
    is used as a documented GPT-family surrogate when the provider tokenizer is
    unavailable.  UTF-8 byte length remains the fail-closed safety fallback.
    """

    def __init__(self, preferred_model: str | None = None) -> None:
        self.name = "utf8_bytes_upper_bound"
        self._tokenizer: Any | None = None
        names = []
        if preferred_model:
            names.append(preferred_model)
            if "/" not in preferred_model:
                names.append(f"openai/{preferred_model}")
        configured = os.getenv("MINDSET_TOKENIZER_NAME", "").strip()
        if configured:
            names.insert(0, configured)
        names.append("gpt2")
        try:
            from transformers import AutoTokenizer
        except Exception:
            AutoTokenizer = None
        if AutoTokenizer is not None:
            for name in dict.fromkeys(names):
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        name, local_files_only=True, use_fast=True
                    )
                except Exception:
                    continue
                self._tokenizer = tokenizer
                self.name = name if name != "gpt2" else "gpt2_local_surrogate"
                break

    def count(self, text: str) -> int:
        if self._tokenizer is not None:
            try:
                encoded = self._tokenizer(
                    text, add_special_tokens=False, return_attention_mask=False
                )
                return max(1, len(encoded["input_ids"]))
            except Exception:
                pass
        return max(1, len(text.encode("utf-8")))

    def offsets(self, text: str) -> list[tuple[int, int]] | None:
        if self._tokenizer is None:
            return None
        try:
            encoded = self._tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_offsets_mapping=True,
            )
            return [
                (int(start), int(end))
                for start, end in encoded["offset_mapping"]
                if int(end) > int(start)
            ]
        except Exception:
            return None

    def fallback_chunk_size(self, token_budget: int) -> int:
        """Return a safe character budget when offsets are unavailable."""
        return max(1, math.floor(token_budget * 4))
