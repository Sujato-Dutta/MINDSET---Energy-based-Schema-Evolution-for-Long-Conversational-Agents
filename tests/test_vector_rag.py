from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mindset.baselines.vector_rag import conversation_chunks
from mindset.datasets.locomo import load_locomo


def test_smoke_manifest_is_frozen_sha256_selection() -> None:
    root = Path(__file__).resolve().parents[1]
    samples = load_locomo(root / "data" / "locomo" / "locomo10.json")
    candidates = [
        {
            "id": f"{sample.sample_id}:q{index:04d}",
            "sample_id": sample.sample_id,
            "question_index": index,
            "category": int(question.category),
        }
        for sample in samples
        for index, question in enumerate(sample.questions)
    ]
    expected = []
    for category in range(1, 6):
        ranked = sorted(
            (row for row in candidates if row["category"] == category),
            key=lambda row: hashlib.sha256(row["id"].encode()).hexdigest(),
        )
        expected.extend(ranked[:4])
    manifest = json.loads(
        (root / "data" / "manifests" / "locomo_smoke20.json").read_text(encoding="utf-8")
    )
    assert manifest["records"] == expected


def test_conversation_chunking_preserves_turn_ids_and_dates() -> None:
    root = Path(__file__).resolve().parents[1]
    sample = load_locomo(root / "data" / "locomo" / "locomo10.json")[0]
    chunks = conversation_chunks(sample)
    assert len(chunks) == sum(len(session.turns) for session in sample.sessions)
    assert chunks[0].turn_id == sample.sessions[0].turns[0].dia_id
    assert sample.sessions[0].date_time in chunks[0].text
