from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from mindset.methods.base import ConversationEvent, MemoryHit
from mindset.utils.hashing import stable_hash


SummaryCallback = Callable[[str, str], str]


@dataclass
class MemoryBankItem:
    memory_id: str
    text: str
    source_event_ids: tuple[str, ...]
    memory_type: str
    day_key: str
    day_index: int
    memory_strength: int = 1
    last_recall_day: int = 0


class MemoryBankReproduction:
    """Paper-faithful MemoryBank adapted to the shared benchmark interface.

    The released MemoryBank implementation stores raw dialogue, daily event
    summaries, and personality summaries; retrieves with dense vectors; and
    updates raw-memory strength using the Ebbinghaus retention curve. The
    official code uses LangChain/FAISS and GPT-3.5; this adapter uses the
    repository's local SentenceTransformer matrix and an injected summarizer
    so every provider call is cached and ledger-accounted.
    """

    method_name = "memory_bank"
    reproduction_label = "official_paper_architecture_provider_model_substitution"
    paper_url = "https://ojs.aaai.org/index.php/AAAI/article/view/29946"
    official_repo = "https://github.com/zhongwanjun/MemoryBank-SiliconFriend"

    def __init__(
        self,
        encoder: Any,
        *,
        summarizer: SummaryCallback | None = None,
        summary_model: str = "gpt_oss_120b",
        top_k: int = 8,
        forgetting: bool = True,
        random_seed: int = 0,
        max_summary_prompt_chars: int = 80_000,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.encoder = encoder
        self.summarizer = summarizer
        self.summary_model = summary_model
        self.top_k = top_k
        self.forgetting = forgetting
        self.random_seed = random_seed
        self.max_summary_prompt_chars = max_summary_prompt_chars
        self.items: list[MemoryBankItem] = []
        self.all_items: list[MemoryBankItem] = []
        self.events: list[ConversationEvent] = []
        self.global_event_summary = ""
        self.global_personality = ""
        self._vectors = np.empty((0, 0), dtype=np.float32)
        self._item_index: dict[str, int] = {}
        self._current_day_index = 0
        self._retrieval_calls = 0
        self._summary_calls = 0
        self._personality_calls = 0
        self._global_summary_calls = 0
        self._global_personality_calls = 0

    def ingest(self, events: Sequence[ConversationEvent]) -> None:
        self.events = list(events)
        if not self.events:
            self.items = []
            self.all_items = []
            self._vectors = np.empty((0, 0), dtype=np.float32)
            return

        grouped: dict[str, list[ConversationEvent]] = {}
        day_order: list[str] = []
        for event in self.events:
            day_key = str(
                event.metadata.get("memory_bank_date")
                or event.timestamp
                or "day-0"
            )
            if day_key not in grouped:
                grouped[day_key] = []
                day_order.append(day_key)
            grouped[day_key].append(event)
        day_index = {key: index for index, key in enumerate(day_order)}
        self._current_day_index = max(day_index.values(), default=0)

        raw_items: list[MemoryBankItem] = []
        daily_summaries: list[MemoryBankItem] = []
        daily_event_summary_pairs: list[tuple[str, str]] = []
        daily_personalities: list[str] = []
        rng = random.Random(self.random_seed)
        for day_key in day_order:
            day_events = grouped[day_key]
            index = day_index[day_key]
            raw_ids: list[str] = []
            raw_day_items: list[MemoryBankItem] = []
            for event_index, event in enumerate(day_events):
                memory_id = "memory-bank-turn-" + stable_hash(
                    (event.event_id, day_key, event_index)
                )[:16]
                item = MemoryBankItem(
                    memory_id=memory_id,
                    text=self._render_event(day_key, event),
                    source_event_ids=(event.event_id,),
                    memory_type="memory_bank_turn",
                    day_key=day_key,
                    day_index=index,
                    last_recall_day=index,
                )
                raw_ids.append(event.event_id)
                raw_day_items.append(item)

            retained = raw_day_items
            if self.forgetting:
                retained = [
                    item
                    for item in raw_day_items
                    if rng.random() <= self.retention(item)
                ]
            raw_items.extend(retained)

            dialogue = self._render_day_dialogue(day_key, day_events)
            event_prompt = self._daily_event_prompt(dialogue)
            personality_prompt = self._daily_personality_prompt(dialogue)
            daily_summary = self._summarize("daily_event", event_prompt)
            daily_personality = self._summarize("daily_personality", personality_prompt)
            daily_personalities.append(daily_personality)
            daily_event_summary_pairs.append((day_key, daily_summary))
            if retained:
                daily_summaries.append(
                    MemoryBankItem(
                        memory_id="memory-bank-day-summary-" + stable_hash(
                            (day_key, tuple(raw_ids))
                        )[:16],
                        text=f"[{day_key}] Event summary: {daily_summary}",
                        source_event_ids=tuple(raw_ids),
                        memory_type="memory_bank_day_summary",
                        day_key=day_key,
                        day_index=index,
                    )
                )

        self.global_event_summary = self._summarize(
            "global_event_summary",
            self._global_event_prompt(daily_event_summary_pairs),
        )
        self.global_personality = self._summarize(
            "global_personality",
            self._global_personality_prompt(
                list(zip(day_order, daily_personalities, strict=True))
            ),
        )
        global_sources = tuple(event.event_id for event in self.events)
        global_items = [
            MemoryBankItem(
                memory_id="memory-bank-global-event-summary",
                text=f"Global event summary: {self.global_event_summary}",
                source_event_ids=global_sources,
                memory_type="memory_bank_global_event_summary",
                day_key=day_order[-1],
                day_index=self._current_day_index,
            ),
            MemoryBankItem(
                memory_id="memory-bank-global-personality",
                text=f"Global user portrait and response strategy: {self.global_personality}",
                source_event_ids=global_sources,
                memory_type="memory_bank_global_personality",
                day_key=day_order[-1],
                day_index=self._current_day_index,
            ),
        ]
        self.all_items = [*raw_items, *daily_summaries, *global_items]
        self.items = list(self.all_items)
        self._vectors = self._encode([item.text for item in self.items])
        self._item_index = {item.memory_id: index for index, item in enumerate(self.items)}

    def retrieve(self, query: str, top_k: int = 8) -> list[MemoryHit]:
        if top_k <= 0 or not self.items:
            return []
        self._retrieval_calls += 1
        query_vector = self._encode([query])[0]
        scores = self._vectors @ query_vector
        global_types = {
            "memory_bank_global_event_summary",
            "memory_bank_global_personality",
        }
        ranked = sorted(
            (
                index for index, item in enumerate(self.items)
                if item.memory_type not in global_types
            ),
            key=lambda index: (-float(scores[index]), self.items[index].memory_id),
        )
        globals_ = [
            index for index, item in enumerate(self.items)
            if item.memory_type in global_types
        ]
        selected = ranked[: max(0, top_k - len(globals_))] + globals_[:top_k]
        selected = selected[:top_k]
        for index in selected:
            item = self.items[index]
            if item.memory_type == "memory_bank_turn":
                item.memory_strength += 1
                item.last_recall_day = self._current_day_index
        return [
            MemoryHit(
                memory_id=self.items[index].memory_id,
                text=self.items[index].text,
                score=float(scores[index]),
                source_event_ids=self.items[index].source_event_ids,
                memory_type=self.items[index].memory_type,
                metadata={
                    "paper": "MemoryBank",
                    "reproduction_label": self.reproduction_label,
                    "paper_url": self.paper_url,
                    "official_repo": self.official_repo,
                    "summary_model": self.summary_model,
                    "day_key": self.items[index].day_key,
                    "day_index": self.items[index].day_index,
                    "memory_strength": self.items[index].memory_strength,
                    "last_recall_day": self.items[index].last_recall_day,
                    "retention": self.retention(self.items[index]),
                },
            )
            for index in selected
        ]

    def retention(self, item: MemoryBankItem) -> float:
        elapsed = max(0, self._current_day_index - item.last_recall_day)
        return math.exp(-elapsed / max(1, item.memory_strength))

    def snapshot(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "method": self.method_name,
            "reproduction_label": self.reproduction_label,
            "paper_url": self.paper_url,
            "official_repo": self.official_repo,
            "summary_model": self.summary_model,
            "forgetting_curve": "R=exp(-t/S)",
            "random_seed": self.random_seed,
            "max_summary_prompt_chars": self.max_summary_prompt_chars,
            "global_event_summary": self.global_event_summary,
            "global_personality": self.global_personality,
            "items": [asdict(item) for item in self.items],
        }
        target.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target

    def telemetry(self) -> dict[str, Any]:
        return {
            "reproduction_label": self.reproduction_label,
            "paper_url": self.paper_url,
            "official_repo": self.official_repo,
            "summary_model": self.summary_model,
            "forgetting_curve": "R=exp(-t/S)",
            "forgetting_enabled": self.forgetting,
            "memory_items": len(self.items),
            "raw_turn_items": sum(item.memory_type == "memory_bank_turn" for item in self.items),
            "daily_summary_items": sum(item.memory_type == "memory_bank_day_summary" for item in self.items),
            "retrieval_calls": self._retrieval_calls,
            "summary_calls": self._summary_calls,
            "personality_summary_calls": self._personality_calls,
            "global_summary_calls": self._global_summary_calls,
            "global_personality_calls": self._global_personality_calls,
            "write_api_calls": 0,
            "retrieval_api_calls": 0,
        }

    def _summarize(self, kind: str, prompt: str) -> str:
        if self.summarizer is None:
            return self._fallback_summary(prompt)
        prompts = self._split_summary_prompt(prompt)
        values = [self._summarize_once(kind, item) for item in prompts]
        value = " ".join(item for item in values if item).strip()
        return value or self._fallback_summary(prompt)

    def _summarize_once(self, kind: str, prompt: str) -> str:
        value = self.summarizer(kind, prompt).strip()
        if kind == "daily_event":
            self._summary_calls += 1
        elif kind == "daily_personality":
            self._personality_calls += 1
        elif kind == "global_event_summary":
            self._global_summary_calls += 1
        elif kind == "global_personality":
            self._global_personality_calls += 1
        return value

    def _split_summary_prompt(self, prompt: str) -> list[str]:
        if len(prompt.encode("utf-8")) <= self.max_summary_prompt_chars:
            return [prompt]
        marker = "\nDialogue content:\n"
        if marker not in prompt:
            return [prompt[: self.max_summary_prompt_chars]]
        prefix, remainder = prompt.split(marker, 1)
        suffix_marker = "\nSummarization:"
        suffix = ""
        if suffix_marker in remainder:
            body, suffix_tail = remainder.rsplit(suffix_marker, 1)
            suffix = suffix_marker + suffix_tail
        else:
            body = remainder
        budget = max(1_000, self.max_summary_prompt_chars - len((prefix + marker + suffix).encode("utf-8")))
        segments: list[str] = []
        current: list[str] = []
        current_size = 0
        for line in body.splitlines():
            line_size = len((line + "\n").encode("utf-8"))
            if current and current_size + line_size > budget:
                segments.append(prefix + marker + "\n".join(current) + suffix)
                current = []
                current_size = 0
            current.append(line)
            current_size += line_size
        if current:
            segments.append(prefix + marker + "\n".join(current) + suffix)
        return segments or [prompt[: self.max_summary_prompt_chars]]

    @staticmethod
    def _render_event(day_key: str, event: ConversationEvent) -> str:
        return f"[{day_key}] [{event.speaker}]: {event.text.strip()}"

    @staticmethod
    def _render_day_dialogue(day_key: str, events: Sequence[ConversationEvent]) -> str:
        return "\n".join(
            f"[{day_key}] {event.speaker}: {event.text.strip()}" for event in events
        )

    @staticmethod
    def _daily_event_prompt(dialogue: str) -> str:
        return (
            "Please summarize the following dialogue as concisely as possible, "
            "extracting the main themes and key information. If there are multiple "
            "key events, you may summarize them separately.\nDialogue content:\n"
            + dialogue
            + "\nSummarization:"
        )

    @staticmethod
    def _daily_personality_prompt(dialogue: str) -> str:
        return (
            "Based on the following dialogue, please summarize the user's personality "
            "traits and emotions, and devise response strategies based on your speculation.\n"
            "Dialogue content:\n" + dialogue +
            "\nThe user's personality traits, emotions, and response strategy are:"
        )

    @staticmethod
    def _global_event_prompt(summaries: Sequence[tuple[str, str]]) -> str:
        body = "\n".join(f"At {day}, the events are {summary}" for day, summary in summaries)
        return (
            "Please provide a highly concise summary of the following events, capturing "
            "the essential key information as succinctly as possible.\nSummarize the event:\n"
            + body + "\nSummarization:"
        )

    @staticmethod
    def _global_personality_prompt(summaries: Sequence[tuple[str, str]]) -> str:
        body = "\n".join(f"At {day}, the analysis shows {summary}" for day, summary in summaries)
        return (
            "The following are the user's exhibited personality traits and emotions "
            "throughout multiple dialogues, along with appropriate response strategies "
            "for the current situation:\n" + body +
            "\nPlease provide a highly concise and general summary of the user's personality "
            "and the most appropriate response strategy, summarized as:"
        )

    @staticmethod
    def _summary_for_day(item: MemoryBankItem) -> str:
        return item.text.split("Event summary:", 1)[-1].strip()

    @staticmethod
    def _fallback_summary(prompt: str) -> str:
        content = prompt.split("\n", 1)[-1].strip()
        words = content.split()
        return " ".join(words[:80])

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            dimension = self._vectors.shape[1] if self._vectors.ndim == 2 else 0
            return np.empty((0, dimension), dtype=np.float32)
        return np.asarray(
            self.encoder.encode(
                list(texts), batch_size=64, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=False,
            ), dtype=np.float32,
        )
