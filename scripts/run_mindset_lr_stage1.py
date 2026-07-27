from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from sentence_transformers import CrossEncoder, SentenceTransformer

from mindset.baselines.vector_rag import VectorRAGIndex
from mindset.datasets.locomo import load_locomo
from mindset.evaluation.metrics import retrieval_metrics
from mindset.methods.base import ConversationEvent, MemoryHit
from mindset.methods.mindset import MindsetConfig, MindsetMemory
from mindset.methods.rerank_cache import CachedReranker
from mindset.retrieval.local_rerank import (
    LocalCandidate,
    RetrievalWeights,
    rank_candidates,
    select_candidates,
)


STAGES = (
    "current_mindset_energy",
    "cross_encoder_only",
    "hybrid",
    "coverage",
    "query_adaptive",
)
THRESHOLDS = {
    "recall@8": 0.558,
    "mrr": 0.473,
    "ndcg@8": 0.458,
    "precision@8": 0.104,
}


@dataclass
class QueryRow:
    identifier: str
    query: str
    gold: set[str]
    memory: MindsetMemory
    candidates: list[LocalCandidate]
    legacy_hits: list[MemoryHit]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-API 60/60 local-reranking development gate"
    )
    parser.add_argument("--dataset", default="data/locomo/locomo10.json")
    parser.add_argument("--manifest", default="data/manifests/locomo_kill120.json")
    parser.add_argument("--split", default="data/manifests/locomo_lr_split120.json")
    parser.add_argument(
        "--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument(
        "--reranker-model", default="cross-encoder/ms-marco-MiniLM-L6-v2"
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--candidate-k", type=int, default=48)
    parser.add_argument(
        "--output", default="outputs/gates/mindset_lr_stage1_v1.json"
    )
    parser.add_argument(
        "--audit", default="outputs/gates/mindset_lr_stage1_rankings_v1.jsonl"
    )
    args = parser.parse_args()
    if args.top_k != 8 or args.candidate_k != 48:
        raise ValueError("The frozen gate requires top_k=8 and candidate_k=48")

    manifest = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))
    split = json.loads((ROOT / args.split).read_text(encoding="utf-8"))
    validate_split(manifest, split)
    samples = {item.sample_id: item for item in load_locomo(ROOT / args.dataset)}
    encoder = SentenceTransformer(args.embedding_model, device="cpu")
    reranker = CachedReranker(
        CrossEncoder(args.reranker_model, device="cpu"),
        args.reranker_model,
        ROOT / "outputs/caches/rerank_scores.sqlite",
    )
    selected_sample_ids = sorted({str(row["sample_id"]) for row in manifest["records"]})
    memories: dict[str, MindsetMemory] = {}
    for sample_id in selected_sample_ids:
        sample = samples[sample_id]
        events = sample_events(sample)
        embedding_index = VectorRAGIndex.build(
            sample,
            encoder,
            args.embedding_model,
            ROOT / "outputs/caches/embeddings",
        )
        precomputed = {
            event.event_id: vector
            for event, vector in zip(events, embedding_index.embeddings, strict=True)
        }
        memory = MindsetMemory(
            MindsetConfig(retrieval_mode="query_adaptive"),
            encoder=encoder,
            local_reranker=reranker,
            precomputed_embeddings=precomputed,
        )
        memory.ingest(events)
        memories[sample_id] = memory

    rows: list[QueryRow] = []
    for position, record in enumerate(manifest["records"], start=1):
        identifier = str(record["id"])
        sample_id = str(record["sample_id"])
        question = samples[sample_id].questions[int(record["question_index"])]
        gold = {
            value.strip()
            for group in question.evidence
            for value in group.split(";")
            if value.strip()
        }
        memory = memories[sample_id]
        rows.append(
            QueryRow(
                identifier=identifier,
                query=question.question,
                gold=gold,
                memory=memory,
                candidates=memory.retrieve_candidate_pool(question.question),
                legacy_hits=memory._retrieve_legacy(question.question, args.candidate_k),
            )
        )
        print(f"LOCAL CANDIDATES {position}/{len(manifest['records'])} {identifier}", flush=True)

    row_by_id = {row.identifier: row for row in rows}
    tune = [row_by_id[item] for item in split["tune_ids"]]
    evaluation = [row_by_id[item] for item in split["evaluation_ids"]]
    initial = RetrievalWeights()
    tuned_weights, weight_trace = coordinate_tune(tune, initial)
    coverage_parameters = tune_coverage(tune, tuned_weights)
    tune_metrics = stage_metrics(
        tune, tuned_weights, coverage_parameters, args.top_k, args.candidate_k
    )
    selected_stage = max(
        STAGES,
        key=lambda stage: (
            tune_metrics[stage]["retrieval_score"],
            -STAGES.index(stage),
        ),
    )
    evaluation_metrics = stage_metrics(
        evaluation, tuned_weights, coverage_parameters, args.top_k, args.candidate_k
    )
    held_out = evaluation_metrics[selected_stage]
    threshold_checks = {
        name: float(held_out[name]) >= required
        for name, required in THRESHOLDS.items()
    }
    passed = all(threshold_checks.values())
    payload = {
        "protocol": "mindset-local-rerank-stage1-v1",
        "network_calls_made": False,
        "answer_model_calls_made": False,
        "judge_calls_made": False,
        "model_ids": {
            "embedding": args.embedding_model,
            "cross_encoder": args.reranker_model,
        },
        "split_manifest": args.split,
        "tune_answerable_count": sum(bool(row.gold) for row in tune),
        "evaluation_answerable_count": sum(bool(row.gold) for row in evaluation),
        "initial_weights": initial.as_dict(),
        "frozen_weights": tuned_weights.as_dict(),
        "coverage_parameters": coverage_parameters,
        "weight_tuning_trace": weight_trace,
        "selection_metric": "0.35*recall@8 + 0.25*mrr + 0.25*ndcg@8 + 0.15*precision@8",
        "selected_stage_on_tune": selected_stage,
        "tune_metrics": tune_metrics,
        "held_out_evaluation_metrics": evaluation_metrics,
        "thresholds": THRESHOLDS,
        "selected_stage_threshold_checks": threshold_checks,
        "decision": "pass" if passed else "fail",
        "stage2_answer_calls_authorized": bool(passed),
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_audit(
        ROOT / args.audit,
        rows,
        set(split["tune_ids"]),
        tuned_weights,
        coverage_parameters,
        selected_stage,
        args.top_k,
        args.candidate_k,
    )
    print(json.dumps(payload, indent=2))


def sample_events(sample: Any) -> list[ConversationEvent]:
    return [
        ConversationEvent(
            f"{sample.sample_id}:{turn.dia_id}",
            sample.sample_id,
            session.session_id,
            session.date_time or session.session_id,
            turn.speaker,
            turn.text,
        )
        for session in sample.sessions
        for turn in session.turns
    ]


def validate_split(manifest: dict[str, Any], split: dict[str, Any]) -> None:
    records = {str(row["id"]): row for row in manifest["records"]}
    tune, evaluation = list(split["tune_ids"]), list(split["evaluation_ids"])
    if len(tune) != 60 or len(evaluation) != 60 or set(tune) & set(evaluation):
        raise ValueError("split must contain two disjoint sets of 60")
    if set(tune) | set(evaluation) != set(records):
        raise ValueError("split does not exactly partition the parent manifest")
    for identifiers in (tune, evaluation):
        counts = Counter(int(records[item]["category"]) for item in identifiers)
        if counts != Counter({category: 12 for category in range(1, 6)}):
            raise ValueError("each split must contain 12 questions per category")


def coordinate_tune(
    rows: Sequence[QueryRow], initial: RetrievalWeights
) -> tuple[RetrievalWeights, list[dict[str, Any]]]:
    names = tuple(initial.as_dict())
    current = initial
    current_score = evaluate_stage(rows, "hybrid", current, {}, 8, 48)[
        "retrieval_score"
    ]
    trace = [{"step": "initial", "score": current_score, "weights": current.as_dict()}]
    for step in (0.05, 0.025, 0.0125):
        for iteration in range(12):
            proposals: list[tuple[float, RetrievalWeights]] = []
            values = current.as_dict()
            for donor in names:
                if values[donor] + 1e-12 < step:
                    continue
                for receiver in names:
                    if donor == receiver:
                        continue
                    candidate = dict(values)
                    candidate[donor] -= step
                    candidate[receiver] += step
                    weights = RetrievalWeights(**candidate)
                    score = evaluate_stage(rows, "hybrid", weights, {}, 8, 48)[
                        "retrieval_score"
                    ]
                    proposals.append((score, weights))
            score, weights = max(
                proposals,
                key=lambda item: (
                    item[0],
                    tuple(-item[1].as_dict()[name] for name in names),
                ),
            )
            if score <= current_score + 1e-12:
                break
            current, current_score = weights, score
            trace.append(
                {
                    "step": step,
                    "iteration": iteration + 1,
                    "score": score,
                    "weights": weights.as_dict(),
                }
            )
    return current, trace


def tune_coverage(
    rows: Sequence[QueryRow], weights: RetrievalWeights
) -> dict[str, float]:
    candidates = []
    for redundancy, entity, temporal in product(
        (0.10, 0.15, 0.20), (0.04, 0.08, 0.12), (0.04, 0.08, 0.12)
    ):
        parameters = {
            "redundancy_penalty": redundancy,
            "new_entity_bonus": entity,
            "new_time_bonus": temporal,
        }
        report = evaluate_stage(rows, "coverage", weights, parameters, 8, 48)
        candidates.append((report["retrieval_score"], parameters))
    return max(
        candidates,
        key=lambda item: (
            item[0],
            -item[1]["redundancy_penalty"],
            -item[1]["new_entity_bonus"],
            -item[1]["new_time_bonus"],
        ),
    )[1]


def stage_metrics(
    rows: Sequence[QueryRow],
    weights: RetrievalWeights,
    coverage: dict[str, float],
    top_k: int,
    candidate_k: int,
) -> dict[str, dict[str, float]]:
    return {
        stage: evaluate_stage(rows, stage, weights, coverage, top_k, candidate_k)
        for stage in STAGES
    }


def evaluate_stage(
    rows: Sequence[QueryRow],
    stage: str,
    weights: RetrievalWeights,
    coverage: dict[str, float],
    top_k: int,
    candidate_k: int,
) -> dict[str, float]:
    totals: dict[str, list[float]] = {
        "candidate_pool_recall@48": [],
        "recall@8": [],
        "precision@8": [],
        "mrr": [],
        "ndcg@8": [],
    }
    for row in rows:
        if not row.gold:
            continue
        retrieved, pool = stage_ids(
            row, stage, weights, coverage, top_k, candidate_k
        )
        report = retrieval_metrics(retrieved, row.gold, top_k)
        totals["candidate_pool_recall@48"].append(
            len(set(pool) & row.gold) / len(row.gold)
        )
        for name in ("recall@8", "precision@8", "mrr", "ndcg@8"):
            totals[name].append(float(report[name]))
    result = {
        name: sum(values) / max(1, len(values)) for name, values in totals.items()
    }
    result["retrieval_score"] = (
        0.35 * result["recall@8"]
        + 0.25 * result["mrr"]
        + 0.25 * result["ndcg@8"]
        + 0.15 * result["precision@8"]
    )
    return result


def stage_ids(
    row: QueryRow,
    stage: str,
    weights: RetrievalWeights,
    coverage: dict[str, float],
    top_k: int,
    candidate_k: int,
) -> tuple[list[str], list[str]]:
    if stage == "current_mindset_energy":
        identifiers = [hit.memory_id.split(":", 1)[-1] for hit in row.legacy_hits]
        return identifiers[:top_k], identifiers[:candidate_k]
    mode = {
        "cross_encoder_only": "cross_encoder_only",
        "hybrid": "hybrid",
        "coverage": "coverage",
        "query_adaptive": "query_adaptive",
    }[stage]
    selected = select_candidates(
        row.query,
        row.candidates,
        mode=mode,
        weights=weights,
        top_k=top_k,
        redundancy_penalty=coverage.get("redundancy_penalty", 0.15),
        new_entity_bonus=coverage.get("new_entity_bonus", 0.08),
        new_time_bonus=coverage.get("new_time_bonus", 0.08),
    )
    ranked = rank_candidates(row.candidates, mode=mode, weights=weights)
    return (
        [item.episode_id.split(":", 1)[-1] for item in selected],
        [item.episode_id.split(":", 1)[-1] for item in ranked[:candidate_k]],
    )


def write_audit(
    path: Path,
    rows: Sequence[QueryRow],
    tune_ids: set[str],
    weights: RetrievalWeights,
    coverage: dict[str, float],
    selected_stage: str,
    top_k: int,
    candidate_k: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            retrieved, pool = stage_ids(
                row, selected_stage, weights, coverage, top_k, candidate_k
            )
            handle.write(
                json.dumps(
                    {
                        "id": row.identifier,
                        "split": "tune" if row.identifier in tune_ids else "evaluation",
                        "question": row.query,
                        "gold_evidence_ids": sorted(row.gold),
                        "selected_stage": selected_stage,
                        "retrieved_ids": retrieved,
                        "candidate_pool_ids": pool,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
