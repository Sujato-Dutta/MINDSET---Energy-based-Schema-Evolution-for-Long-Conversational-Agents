from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from mindset.utils.hashing import stable_hash


def load_json_records(path: str | Path) -> list[dict[str, Any]]:
    if Path(path).name == 'longmemeval_s_cleaned.json':
        with Path(path).open('r', encoding='utf-8') as stream:
            records = list(_iter_top_level_array(stream))
        if not all(isinstance(item, dict) for item in records):
            raise ValueError('Dataset records must be JSON objects')
        return records
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = next(
            (payload[key] for key in ("data", "questions", "examples", "instances") if key in payload),
            None,
        )
        if records is None:
            raise ValueError("Could not find a record list in compact JSON")
    else:
        raise ValueError("Dataset JSON must contain a list or object")
    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        raise ValueError("Dataset records must be JSON objects")
    return records


def _iter_top_level_array(stream, *, chunk_size: int = 1024 * 1024):
    decoder = json.JSONDecoder()
    buffer = ''
    position = 0
    exhausted = False

    def read_more() -> None:
        nonlocal buffer, exhausted
        chunk = stream.read(chunk_size)
        if chunk:
            buffer += chunk
        else:
            exhausted = True

    read_more()
    while position < len(buffer) and buffer[position].isspace():
        position += 1
    if position >= len(buffer) or buffer[position] != '[':
        raise ValueError('Streaming loader requires a top-level JSON array')
    position += 1
    while True:
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position < len(buffer):
                break
            if exhausted:
                raise ValueError('Unterminated JSON array')
            buffer = ''
            position = 0
            read_more()
        if buffer[position] == ']':
            return
        while True:
            try:
                item, next_position = decoder.raw_decode(buffer, position)
                break
            except json.JSONDecodeError:
                if exhausted:
                    raise
                buffer = buffer[position:]
                position = 0
                read_more()
        yield item
        position = next_position
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position < len(buffer):
                break
            if exhausted:
                raise ValueError('Unterminated JSON array')
            buffer = ''
            position = 0
            read_more()
        if buffer[position] == ',':
            position += 1
            continue
        if buffer[position] == ']':
            return
        raise ValueError('Malformed JSON array separator')


def load_memoryagentbench_questions(path: str | Path) -> tuple[list[dict[str, Any]], int]:
    import pandas as pd
    import pandas as pd

    frame = pd.read_parquet(path)
    required = {'questions', 'answers', 'metadata'}
    if not required.issubset(frame.columns):
        raise ValueError(f'MemoryAgentBench parquet lacks columns: {sorted(required - set(frame.columns))}')
    records: list[dict[str, Any]] = []
    for row_index, row in frame.iterrows():
        metadata = dict(row['metadata'])
        source = str(metadata.get('source') or f'row_{row_index}')
        raw_identifiers = metadata.get('qa_pair_ids')
        identifiers = list(raw_identifiers) if raw_identifiers is not None else []
        questions = list(row['questions'])
        answers = list(row['answers'])
        if not (len(identifiers) == len(questions) == len(answers)):
            raise ValueError(f'MemoryAgentBench row {row_index} has misaligned question arrays')
        for identifier, question, answer in zip(identifiers, questions, answers):
            if hasattr(answer, 'tolist'):
                answer = answer.tolist()
            answer_values = answer if isinstance(answer, list) else [answer]
            normalized_answers = [str(value) for value in answer_values]
            records.append(
                {
                    'qa_pair_id': str(identifier),
                    'question': str(question),
                    'answer': normalized_answers[0]
                    if len(normalized_answers) == 1
                    else normalized_answers,
                    'answers': normalized_answers,
                    'category': source,
                    'source_row': int(row_index),
                }
            )
    return records, len(frame)


def stable_record_id(record: dict[str, Any], index: int) -> str:
    for key in ("question_id", "qa_pair_id", "id", "example_id", "sample_id", "task_id"):
        if record.get(key) is not None:
            return str(record[key])
    return f"derived-{stable_hash({'index': index, 'record': record})[:16]}"


def longmemeval_stratum(record: dict[str, Any]) -> str:
    if str(record.get("question_id", "")).endswith("_abs"):
        return "abstention"
    label = str(record.get("question_type", record.get("category", ""))).lower()
    if "abstention" in label:
        return "abstention"
    if "knowledge-update" in label or "knowledge update" in label:
        return "knowledge_update"
    if "temporal" in label:
        return "temporal"
    if "multi-session" in label or "multi_session" in label:
        return "multi_session"
    return "other"


def focused_longmemeval_manifest(
    records: list[dict[str, Any]], target_size: int = 100
) -> dict[str, Any]:

    return _stratified_manifest(
        records,
        target_size,
        longmemeval_stratum,
        dataset="longmemeval",
        selection="Focused diagnostic subset prioritizing updates, temporal, multi-session, and abstention",
        allowed_strata={"knowledge_update", "temporal", "multi_session", "abstention"},
    )


def memoryagentbench_manifest(
    records: list[dict[str, Any]], target_size: int | None = None
) -> dict[str, Any]:
    selected = [
        record
        for record in records
        if "test" in str(record.get("task_family", record.get("category", ""))).lower()
        and "learn" in str(record.get("task_family", record.get("category", ""))).lower()
    ]
    if not selected:
        selected = records
    size = len(selected) if target_size is None else min(target_size, len(selected))
    return _stratified_manifest(
        selected,
        size,
        lambda record: str(record.get("subtask", record.get("category", "unspecified"))),
        dataset="memoryagentbench",
        selection="Complete Test-Time Learning family when feasible; stable stratified subset otherwise",
    )


def _stratified_manifest(
    records: list[dict[str, Any]],
    target_size: int,
    stratum_fn: Callable[[dict[str, Any]], str],
    *,
    dataset: str,
    selection: str,
    allowed_strata: set[str] | None = None,
) -> dict[str, Any]:
    indexed = [(stable_record_id(record, index), record) for index, record in enumerate(records)]
    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for record_id, record in indexed:
        label = stratum_fn(record)
        if allowed_strata is None or label in allowed_strata:
            grouped[label].append((record_id, record))
    for group in grouped.values():
        group.sort(key=lambda item: stable_hash({"dataset": dataset, "id": item[0]}))
    labels = sorted(grouped)
    selected: list[tuple[str, str]] = []
    selection_limit = min(target_size, sum(map(len, grouped.values())))
    while labels and len(selected) < selection_limit:
        remaining: list[str] = []
        for label in labels:
            if grouped[label] and len(selected) < target_size:
                record_id, _ = grouped[label].pop(0)
                selected.append((record_id, label))
            if grouped[label]:
                remaining.append(label)
        labels = remaining
    return {
        "dataset": dataset,
        "diagnostic_subset": dataset == "longmemeval",
        "selection_criteria": selection,
        "selection_algorithm": "round-robin strata; SHA-256 stable ordering; no RNG",
        "target_size": target_size,
        "selected_count": len(selected),
        "selected": [{"id": record_id, "stratum": label} for record_id, label in selected],
    }


def save_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
