from __future__ import annotations

import glob
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field


# Official task_eval/evaluation.py category semantics.
CATEGORY_NAMES = {
    '1': 'multi_hop',
    '2': 'temporal',
    '3': 'open_domain',
    '4': 'single_hop',
    '5': 'adversarial',
}


class LoCoMoQuestion(BaseModel):
    model_config = ConfigDict(extra="allow")

    question: str
    answer: str | int | float | list[Any] | None = None
    evidence: list[str] = Field(default_factory=list)
    category: int | str


class LoCoMoTurn(BaseModel):
    model_config = ConfigDict(extra="allow")

    speaker: str
    dia_id: str
    text: str


class LoCoMoSession(BaseModel):
    session_id: str
    date_time: str | None
    turns: list[LoCoMoTurn]


class LoCoMoSample(BaseModel):
    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: list[LoCoMoSession]
    questions: list[LoCoMoQuestion]
    raw: dict[str, Any] = Field(exclude=True)


def locate_locomo(configured_path: str | Path | None, search_paths: Iterable[str]) -> Path:
    env_path = os.getenv("MINDSET_LOCOMO_PATH", "").strip()
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path).expanduser())
    if configured_path:
        candidates.append(Path(configured_path).expanduser())
    for pattern in search_paths:
        candidates.extend(Path(item) for item in sorted(glob.glob(pattern)))
    candidates.extend(
        [
            Path.home() / "Downloads" / "locomo10.json",
            Path.home() / "Downloads" / "LoCoMo" / "locomo10.json",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    rendered = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not locate LoCoMo JSON. Checked: {rendered}")


def load_locomo(path: str | Path) -> list[LoCoMoSample]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("LoCoMo root must be a list")
    samples: list[LoCoMoSample] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"LoCoMo item {index} is not an object")
        conversation = item.get("conversation")
        if not isinstance(conversation, dict):
            raise ValueError(f"LoCoMo item {index} has no conversation object")
        sessions: list[LoCoMoSession] = []
        session_numbers = sorted(
            {
                int(match.group(1))
                for key in conversation
                if (match := re.fullmatch(r"session_(\d+)", key))
            }
        )
        for number in session_numbers:
            raw_turns = conversation[f"session_{number}"]
            if not isinstance(raw_turns, list):
                raise ValueError(f"sample {index} session_{number} must be a list")
            turns = [LoCoMoTurn.model_validate(turn) for turn in raw_turns]
            sessions.append(
                LoCoMoSession(
                    session_id=f"D{number}",
                    date_time=conversation.get(f"session_{number}_date_time"),
                    turns=turns,
                )
            )
        questions = [LoCoMoQuestion.model_validate(question) for question in item.get("qa", [])]
        samples.append(
            LoCoMoSample(
                sample_id=str(item.get("sample_id", index)),
                speaker_a=str(conversation.get("speaker_a", "")),
                speaker_b=str(conversation.get("speaker_b", "")),
                sessions=sessions,
                questions=questions,
                raw=item,
            )
        )
    return samples


def audit_locomo(samples: list[LoCoMoSample], source_path: str | Path) -> dict[str, Any]:
    categories: Counter[str] = Counter()
    turn_ids: set[tuple[str, str]] = set()
    duplicate_turn_ids: list[str] = []
    orphan_timestamps: list[str] = []
    multimodal_turns = 0
    missing_evidence: list[dict[str, str]] = []
    question_count = 0
    missing_answer_count = 0
    session_count = 0
    turn_count = 0
    for sample in samples:
        conversation = sample.raw["conversation"]
        listed_sessions = {session.session_id for session in sample.sessions}
        for key in conversation:
            match = re.fullmatch(r"session_(\d+)_date_time", key)
            if match and f"D{match.group(1)}" not in listed_sessions:
                orphan_timestamps.append(f"{sample.sample_id}:{key}")
        valid_ids: set[str] = set()
        for session in sample.sessions:
            session_count += 1
            for turn in session.turns:
                turn_count += 1
                scoped = (sample.sample_id, turn.dia_id)
                if scoped in turn_ids:
                    duplicate_turn_ids.append(f"{sample.sample_id}:{turn.dia_id}")
                turn_ids.add(scoped)
                valid_ids.add(turn.dia_id)
                raw_turn = next(
                    raw
                    for raw in conversation[f"session_{session.session_id[1:]}"]
                    if raw.get("dia_id") == turn.dia_id
                )
                multimodal_turns += int(bool(raw_turn.get("img_url")))
        for q_index, question in enumerate(sample.questions):
            question_count += 1
            missing_answer_count += int(question.answer is None)
            category = str(question.category)
            categories[category] += 1
            for evidence_group in question.evidence:
                # A few source annotations combine IDs with semicolons.
                for evidence_id in (part.strip() for part in evidence_group.split(";")):
                    if evidence_id and evidence_id not in valid_ids:
                        missing_evidence.append(
                            {
                                "sample_id": sample.sample_id,
                                "question_index": str(q_index),
                                "evidence_id": evidence_id,
                            }
                        )
    return {
        "status": "available",
        "role": "primary_full_conversational_memory_benchmark",
        "source_path": str(Path(source_path).resolve()),
        "conversation_count": len(samples),
        "session_count": session_count,
        "turn_count": turn_count,
        "question_count": question_count,
        "missing_answer_count": missing_answer_count,
        "category_distribution": {
            key: {"name": CATEGORY_NAMES.get(key, "unknown"), "count": value}
            for key, value in sorted(categories.items())
        },
        "multimodal_source_turn_count": multimodal_turns,
        "text_qa_policy": "All QA retained; images are never generated or fetched.",
        "orphan_session_timestamp_count": len(orphan_timestamps),
        "orphan_session_timestamps": orphan_timestamps,
        "duplicate_turn_id_count": len(duplicate_turn_ids),
        "missing_evidence_reference_count": len(missing_evidence),
        "missing_evidence_references": missing_evidence[:100],
    }
