from __future__ import annotations
from dataclasses import asdict, dataclass
from collections import Counter
from pathlib import Path
from typing import Any
from mindset.acl.io import digest_file, read_json, write_json
from mindset.datasets.locomo import load_locomo, CATEGORY_NAMES
from mindset.datasets.events import longmemeval_events
from mindset.methods.base import ConversationEvent
from mindset.utils.hashing import stable_hash

@dataclass
class Case:
    id: str
    cluster: str
    category: str
    question: str
    answer: Any
    events: list[ConversationEvent]
    gold: list[str]
    retrieval_unit: str
    question_date: str = ""

def load_cases(dataset: str, path: str | Path) -> list[Case]:
    cases = []
    if dataset == "locomo":
        for sample in load_locomo(path):
            events = [
                ConversationEvent(t.dia_id, sample.sample_id, s.session_id, s.date_time or "", t.speaker, t.text)
                for s in sample.sessions for t in s.turns
            ]
            for i, q in enumerate(sample.questions):
                category = int(q.category)
                answer = q.answer
                # Adversarial records carry adversarial_answer, which must never be an answer target.
                if category == 5:
                    answer = "No information available."
                if answer is None:
                    raise ValueError(f"Missing answer: {sample.sample_id}:{i}")
                gold = list(dict.fromkeys(x.strip() for group in q.evidence for x in group.split(";") if x.strip()))
                cases.append(Case(f"{sample.sample_id}:q{i:04d}", sample.sample_id,
                                  CATEGORY_NAMES[str(category)], q.question, answer, events, gold, "episode"))
    elif dataset == "longmemeval":
        records = read_json(path)
        if not isinstance(records, list):
            raise ValueError("LongMemEval root must be a list")
        for r in records:
            cases.append(Case(str(r["question_id"]), str(r["question_id"]), str(r["question_type"]),
                              str(r["question"]), r["answer"], longmemeval_events(r),
                              list(dict.fromkeys(map(str, r.get("answer_session_ids", [])))),
                              "session", str(r.get("question_date", ""))))
    else:
        raise ValueError(f"Unsupported benchmark: {dataset}")
    if len({c.id for c in cases}) != len(cases):
        raise ValueError("Duplicate benchmark IDs")
    return cases

def manifest(dataset: str, path: str | Path, expected_count: int | None = None, *, categories=None) -> dict:
    cases = load_cases(dataset, path)
    if categories is not None:
        if dataset!="locomo" or not categories or not set(categories)<={1,2,3,4,5}:
            raise ValueError("Invalid LoCoMo category policy")
        names={CATEGORY_NAMES[str(c)] for c in categories}
        cases=[c for c in cases if c.category in names]
    if expected_count is not None and len(cases) != expected_count:
        raise ValueError(f"{dataset}: expected {expected_count} questions, found {len(cases)}")
    # Histories are shared by all questions in a LoCoMo conversation. Hash
    # once per cluster instead of repeatedly serializing the same episodes.
    histories={}
    for c in cases:
        if c.cluster not in histories:
            histories[c.cluster]=stable_hash([asdict(e) for e in c.events])
    value = {"schema_version": "acl-full-manifest-v1", "dataset": dataset,
            "source_sha256": digest_file(path), "count": len(cases),
            "policy": "all_source_questions_in_source_order", "records": [
                {"id": c.id, "cluster": c.cluster, "category": c.category,
                 "question_sha256": stable_hash(c.question),
                 "history_sha256": histories[c.cluster],
                 "gold_evidence_count": len(c.gold)} for c in cases]}
    if categories is not None:
        value["included_categories"]=sorted(set(categories))
        value["policy"]="all_questions_in_declared_categories_in_source_order"
    if dataset=="longmemeval":
        value["episode_id_policy"]="original IDs; deterministic occurrence suffix for repeated native sessions"
    return value

def verified_cases(dataset: str, path: str | Path, manifest_path: str | Path) -> list[Case]:
    frozen = read_json(manifest_path)
    actual = manifest(dataset, path, categories=frozen.get("included_categories"))
    if frozen != actual:
        raise ValueError("Dataset changed since freeze; source checksum or manifest mismatch")
    selected={r["id"] for r in frozen["records"]}
    return [c for c in load_cases(dataset,path) if c.id in selected]


def longmemeval_session_integrity(path):
    records=read_json(path); affected=[]; total=0
    for record in records:
        events=longmemeval_events(record)
        if len(events)!=len({e.event_id for e in events}):
            raise ValueError("Normalized LongMemEval episode IDs are not unique")
        total+=len(events)
        duplicates={s:n for s,n in Counter(map(str,record["haystack_session_ids"])).items() if n>1}
        if duplicates:
            affected.append({"question_id":str(record["question_id"]),
                             "repeated_native_sessions":duplicates,"episodes":len(events)})
    return {"source_sha256":digest_file(path),"questions":len(records),"episodes":total,
            "duplicate_session_histories":affected,
            "policy":"Preserve all occurrences; suffix duplicate episode IDs only; native gold session IDs unchanged"}

