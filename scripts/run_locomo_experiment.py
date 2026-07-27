from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from sentence_transformers import CrossEncoder, SentenceTransformer

from mindset.api.backend import APIBackend
from mindset.api.models import CompletionRequest
from mindset.baselines.hybrid_rag import HybridRAGIndex
from mindset.baselines.lightmem import LightMemACL2026Reproduction
from mindset.baselines.long_context import build_long_context
from mindset.baselines.memory_bank import MemoryBankReproduction
from mindset.baselines.vector_rag import VectorRAGIndex
from mindset.datasets.locomo import LoCoMoSample, load_locomo
from mindset.evaluation.locomo import evaluate_locomo, evidence_recall
from mindset.evaluation.metrics import retrieval_metrics as ranked_retrieval_metrics
from mindset.experiments.batching import (
    SYSTEM_PROMPT,
    AnswerQuestion,
    answer_batches,
    build_answer_batches,
)
from mindset.experiments.evidence import EvidenceBudget, bound_evidence
from mindset.api.tokenizer import TokenCounter
from mindset.experiments.reporting import (
    distribution_summary,
    file_sha256,
    runtime_environment,
    source_tree_sha256,
    token_and_latency_summary,
)
from mindset.methods.base import ConversationEvent, MemoryHit
from mindset.methods.mindset import MindsetConfig, MindsetMemory
from mindset.methods.nemori import NemoriFaithfulReproduction
from mindset.methods.rerank_cache import CachedReranker
from mindset.utils.hashing import stable_hash


METHODS = (
    "long_context",
    "vector_rag",
    "hybrid_rag",
    "nemori",
    "lightmem",
    "mindset",
    "memory_bank",
)
ANSWER_MODELS = ("gpt_oss_120b", "qwen_27b", "llama_scout_17b", "glm_47", "gemma_4_31b")
ABLATIONS = (
    "full", "nearest_schema_merge", "no_contextual_term", "no_temporal_term",
    "no_hysteresis", "no_fragmentation_penalty", "no_provenance_expansion",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one immutable LoCoMo method/model cell")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--ablation", choices=ABLATIONS, default="full")
    parser.add_argument("--model", choices=ANSWER_MODELS, default="gpt_oss_120b")
    parser.add_argument("--dataset", default="data/locomo/locomo10.json")
    parser.add_argument("--manifest", default="data/manifests/locomo_primary700.json")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    parser.add_argument("--hybrid-candidate-k", type=int, default=24)
    parser.add_argument(
        "--mindset-retrieval-mode",
        choices=("legacy", "cross_encoder_only", "hybrid", "coverage", "query_adaptive"),
        default="cross_encoder_only",
    )
    parser.add_argument("--candidate-k", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-prompt-bytes", type=int, default=160000)
    parser.add_argument("--bounded-evidence", action="store_true")
    parser.add_argument("--evidence-candidate-k", type=int, default=20)
    parser.add_argument("--evidence-chunk-size-tokens", type=int, default=768)
    parser.add_argument("--evidence-chunk-overlap-tokens", type=int, default=96)
    parser.add_argument("--evidence-max-chunks", type=int, default=8)
    parser.add_argument("--evidence-token-budget", type=int, default=8000)
    parser.add_argument("--evidence-max-chunks-per-memory", type=int, default=3)
    parser.add_argument("--long-context-max-history-bytes", type=int, default=20000)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--memory-bank-summary-prompt-chars", type=int, default=16000)
    parser.add_argument("--reserve-fraction", type=float, default=0.10)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--repetition-id", default="primary")
    args = parser.parse_args()
    if args.method != "mindset" and args.ablation != "full":
        raise ValueError("Ablations apply only to MINDSET")

    started = dt.datetime.now(dt.timezone.utc)
    suffix = f"_{args.ablation}" if args.method == "mindset" else ""
    run_id = f"locomo_{args.method}{suffix}_{args.model}_{started.strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = ROOT / "outputs/runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    backend = APIBackend.from_file(ROOT / args.models_config)
    manifest = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))
    records = manifest.get("records", [])
    samples = {sample.sample_id: sample for sample in load_locomo(ROOT / args.dataset)}
    selected_samples = {str(record["sample_id"]): samples[str(record["sample_id"])] for record in records}
    events = {sample_id: sample_events(sample) for sample_id, sample in selected_samples.items()}

    preflight = {
        "decision": "pass",
        "ingestion": "local_no_api",
        "network_calls_made": False,
    }
    write_json(run_dir / "preflight.json", preflight)
    if args.preflight_only:
        write_json(run_dir / "run_config.json", run_config(args, run_id, manifest, started, "preflight_passed"))
        print(json.dumps({"run_dir": str(run_dir), "preflight": preflight}, indent=2))
        return

    encoder = None
    evidence_tokenizer = None
    if args.method in {"vector_rag", "hybrid_rag", "nemori", "lightmem", "mindset", "memory_bank"}:
        encoder = SentenceTransformer(args.embedding_model, device="cpu")
        if args.bounded_evidence:
            evidence_tokenizer = TokenCounter(args.model)
    reranker = None
    if args.method in {"lightmem", "mindset"}:
        reranker = CachedReranker(
            CrossEncoder(args.reranker_model, device="cpu"),
            args.reranker_model,
            ROOT / "outputs/caches/rerank_scores.sqlite",
        )
    methods: dict[str, Any] = {}
    ingestion_seconds: dict[str, float] = {}
    memory_construction_records: list[dict[str, object]] = []
    for sample_id, sample in selected_samples.items():
        ingestion_started = perf_counter()
        method = construct_method(
            args, sample, events[sample_id], encoder, reranker, backend,
            sample_id, memory_construction_records,
        )
        ingestion_seconds[sample_id] = perf_counter() - ingestion_started
        methods[sample_id] = method
        snapshot = getattr(method, "snapshot", None)
        if snapshot:
            snapshot(run_dir / "memory" / f"{sample_id}.json")

    print(f"LOCAL_CONSTRUCTION_COMPLETE method={args.method} conversations={len(methods)}", flush=True)
    answer_questions = []
    retrieval_rows: dict[str, list[MemoryHit]] = {}
    candidate_rows: dict[str, list[MemoryHit]] = {}
    retrieval_seconds: dict[str, float] = {}
    evidence_packet_bytes: dict[str, int] = {}
    for record_number, record in enumerate(records, start=1):
        if record_number == 1 or record_number % 25 == 0:
            print(f"RETRIEVAL_START {record_number}/{len(records)}", flush=True)
        sample_id = str(record["sample_id"])
        question = samples[sample_id].questions[int(record["question_index"])]
        requested_k = (
            args.evidence_candidate_k
            if args.bounded_evidence and args.method != "long_context"
            else (max(args.top_k, args.candidate_k) if args.retrieval_only else args.top_k)
        )
        retrieval_started = perf_counter()
        candidates = retrieve(
            args,
            methods[sample_id],
            sample_id,
            question.question,
            top_k=requested_k,
        )
        retrieval_seconds[str(record["id"])] = perf_counter() - retrieval_started
        hits = candidates[: args.top_k]
        if args.bounded_evidence and args.method != "long_context":
            hits = bound_evidence(
                candidates,
                question.question,
                encoder,
                budget=EvidenceBudget(
                    candidate_k=args.evidence_candidate_k,
                    chunk_size_tokens=args.evidence_chunk_size_tokens,
                    chunk_overlap_tokens=args.evidence_chunk_overlap_tokens,
                    max_chunks=args.evidence_max_chunks,
                    max_tokens=args.evidence_token_budget,
                    max_chunks_per_memory=args.evidence_max_chunks_per_memory,
                ),
                tokenizer=evidence_tokenizer,
            )
        identifier = str(record["id"])
        if record_number == 1 or record_number % 25 == 0:
            print(f"RETRIEVAL_DONE {record_number}/{len(records)}", flush=True)
        evidence_packet_bytes[identifier] = sum(
            len(hit.text.encode("utf-8")) for hit in hits
        )
        retrieval_rows[identifier] = hits
        candidate_rows[identifier] = candidates
        answer_questions.append(AnswerQuestion(identifier, question.question, tuple(hits)))
    if args.retrieval_only:
        recalls = []
        candidate_recalls = []
        packet_bytes = []
        answerable_metrics: dict[str, list[float]] = {}
        retrieval_records = []
        for record in records:
            identifier = str(record["id"])
            sample_id = str(record["sample_id"])
            question = samples[sample_id].questions[int(record["question_index"])]
            retrieved_ids = [source.split(":", 1)[-1] for hit in retrieval_rows[identifier] for source in hit.source_event_ids]
            candidate_ids = [
                source.split(":", 1)[-1]
                for hit in candidate_rows[identifier]
                for source in hit.source_event_ids
            ]
            gold_ids = {
                part.strip()
                for group in question.evidence
                for part in group.split(";")
                if part.strip()
            }
            recall = evidence_recall(retrieved_ids, question.evidence)
            candidate_recall = evidence_recall(candidate_ids, question.evidence)
            recalls.append(recall)
            candidate_recalls.append(candidate_recall)
            packet_bytes.append(evidence_packet_bytes[identifier])
            ranked = ranked_retrieval_metrics(retrieved_ids, gold_ids, args.top_k)
            if gold_ids:
                for name, value in ranked.items():
                    answerable_metrics.setdefault(name, []).append(value)
            retrieval_records.append({
                "id": identifier,
                "sample_id": sample_id,
                "question": question.question,
                "gold_evidence_ids": question.evidence,
                "retrieved_ids": retrieved_ids,
                "evidence_recall": recall,
                "candidate_pool_ids": candidate_ids,
                "candidate_pool_recall": candidate_recall,
                "candidate_hits": [
                    {
                        "memory_id": hit.memory_id,
                        "text": hit.text,
                        "score": hit.score,
                        "source_event_ids": hit.source_event_ids,
                        "memory_type": hit.memory_type,
                        "metadata": hit.metadata,
                    }
                    for hit in candidate_rows[identifier]
                ],
                "ranked_metrics": ranked,
                "hits": [
                    {
                        "memory_id": hit.memory_id,
                        "text": hit.text,
                        "score": hit.score,
                        "source_event_ids": hit.source_event_ids,
                        "memory_type": hit.memory_type,
                        "metadata": hit.metadata,
                    }
                    for hit in retrieval_rows[identifier]
                ],
            })
        metrics = {
            "status": "retrieval_smoke_complete",
            "question_count": len(records),
            "mean_evidence_recall_at_k": sum(recalls) / max(1, len(recalls)),
            "mean_candidate_pool_recall": sum(candidate_recalls) / max(1, len(candidate_recalls)),
            "mean_evidence_packet_bytes": sum(packet_bytes) / max(1, len(packet_bytes)),
            "answerable_retrieval": {
                name: sum(values) / len(values)
                for name, values in sorted(answerable_metrics.items())
            },
            "method_telemetry": {
                sample_id: method.telemetry()
                for sample_id, method in methods.items()
                if hasattr(method, "telemetry")
            },
            "memory_invariants": {
                sample_id: method.validate_invariants()
                for sample_id, method in methods.items()
                if hasattr(method, "validate_invariants")
            },
            "network_calls_made_for_answering": False,
            "efficiency": {
                "ingestion_seconds_by_case": ingestion_seconds,
                "ingestion_seconds": distribution_summary(list(ingestion_seconds.values())),
                "retrieval_seconds": distribution_summary(list(retrieval_seconds.values())),
                "evidence_packet_bytes": distribution_summary(packet_bytes),
                "memory_construction_api_calls": int(
                    token_and_latency_summary(memory_construction_records, len(records))["provider_calls"]
                ),
                "memory_construction": token_and_latency_summary(
                    memory_construction_records, max(1, len(selected_samples))
                ),
                "retrieval_api_calls": 0,
                "answer_api_calls": 0,
            },
        }
        write_json(run_dir / "metrics.json", metrics)
        write_jsonl(run_dir / "retrieval.jsonl", retrieval_records)
        write_json(run_dir / "run_config.json", run_config(args, run_id, manifest, started, "retrieval_smoke_complete"))
        for method in methods.values():
            close = getattr(method, "close", None)
            if close:
                close()
        print(json.dumps({"run_dir": str(run_dir), "metrics": metrics}, indent=2))
        return
    answer_questions.sort(key=lambda item: item.question_id.split(":", 1)[0])
    batches = build_answer_batches(
        answer_questions, max_questions=args.batch_size, max_prompt_bytes=args.max_prompt_bytes
    )
    answer_telemetry: list[dict[str, object]] = []
    answers = answer_batches(
        backend,
        batches,
        model_name=args.model,
        method=args.method,
        code_version=f"locomo-paper-run-v8-six-method:{args.repetition_id}",
        max_output_tokens=args.max_output_tokens,
        max_retries=args.max_retries,
        reserve_fraction=args.reserve_fraction,
        telemetry=answer_telemetry,
        request_metadata={
            "run_id": run_id,
            "dataset": "locomo",
            "repetition_id": args.repetition_id,
            "ablation": args.ablation,
        },
    )

    predictions = []
    answerable_metrics: dict[str, list[float]] = {}
    for record in records:
        identifier = str(record["id"])
        sample_id = str(record["sample_id"])
        question = samples[sample_id].questions[int(record["question_index"])]
        hits = retrieval_rows[identifier]
        retrieved_ids = [source.split(":", 1)[-1] for hit in hits for source in hit.source_event_ids]
        gold_ids = {
            part.strip()
            for group in question.evidence
            for part in group.split(";")
            if part.strip()
        }
        ranked = ranked_retrieval_metrics(retrieved_ids, gold_ids, args.top_k)
        if gold_ids:
            for name, value in ranked.items():
                answerable_metrics.setdefault(name, []).append(value)
        predictions.append({
            "id": identifier,
            "sample_id": sample_id,
            "question_index": int(record["question_index"]),
            "category": int(question.category),
            "question": question.question,
            "answer": question.answer,
            "prediction": answers[identifier],
            "gold_evidence_ids": question.evidence,
            "retrieved_ids": retrieved_ids,
            "evidence_recall": evidence_recall(retrieved_ids, question.evidence),
            "ranked_retrieval": ranked,
            "retrieval_seconds": retrieval_seconds[identifier],
            "evidence_packet_bytes": evidence_packet_bytes[identifier],
        })
    metrics = evaluate_locomo(predictions)
    metrics["mean_evidence_recall_at_k"] = sum(row["evidence_recall"] for row in predictions) / max(1, len(predictions))
    metrics["status"] = "complete"
    metrics["answerable_retrieval"] = {
        name: sum(values) / len(values)
        for name, values in sorted(answerable_metrics.items())
    }
    metrics["method_telemetry"] = {
        sample_id: method.telemetry()
        for sample_id, method in methods.items()
        if hasattr(method, "telemetry")
    }
    metrics["memory_invariants"] = {
        sample_id: method.validate_invariants()
        for sample_id, method in methods.items()
        if hasattr(method, "validate_invariants")
    }
    metrics["efficiency"] = {
        "ingestion_seconds_by_case": ingestion_seconds,
        "ingestion_seconds": distribution_summary(list(ingestion_seconds.values())),
        "retrieval_seconds": distribution_summary(list(retrieval_seconds.values())),
        "evidence_packet_bytes": distribution_summary(
            list(evidence_packet_bytes.values())
        ),
        "memory_construction_api_calls": int(
            token_and_latency_summary(memory_construction_records, len(predictions))["provider_calls"]
        ),
        "memory_construction": token_and_latency_summary(
            memory_construction_records, max(1, len(selected_samples))
        ),
        "retrieval_api_calls": 0,
        **token_and_latency_summary(answer_telemetry, len(predictions)),
    }
    write_jsonl(run_dir / "predictions.jsonl", predictions)
    write_jsonl(run_dir / "answer_batches.jsonl", answer_telemetry)
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "run_config.json", run_config(args, run_id, manifest, started, "complete"))
    for method in methods.values():
        close = getattr(method, "close", None)
        if close:
            close()
    print(json.dumps({"run_dir": str(run_dir), "metrics": metrics}, indent=2))


def sample_events(sample: LoCoMoSample) -> list[ConversationEvent]:
    return [
        ConversationEvent(
            f"{sample.sample_id}:{turn.dia_id}", sample.sample_id, session.session_id,
            session.date_time or session.session_id, turn.speaker, turn.text,
            metadata={"memory_bank_date": session.date_time or session.session_id},
        )
        for session in sample.sessions for turn in session.turns
    ]


def construct_method(args, sample, events, encoder, reranker, backend, sample_id, construction_records):
    if args.method == "vector_rag":
        return VectorRAGIndex.build(sample, encoder, args.embedding_model, ROOT / "outputs/caches/embeddings")
    if args.method == "long_context":
        return build_long_context(sample, max_history_bytes=args.long_context_max_history_bytes)
    if args.method == "hybrid_rag":
        return HybridRAGIndex(
            events,
            encoder,
            candidate_k=args.hybrid_candidate_k,
        )
    if args.method == "memory_bank":
        method = MemoryBankReproduction(
            encoder,
            summarizer=make_memory_bank_summarizer(
                backend, args, sample_id, construction_records
            ),
            summary_model=args.model,
            top_k=args.top_k,
            random_seed=0,
            max_summary_prompt_chars=args.memory_bank_summary_prompt_chars,
        )
        method.ingest(events)
        return method
    if args.method == "nemori":
        method = NemoriFaithfulReproduction(encoder)
    elif args.method == "lightmem":
        method = LightMemACL2026Reproduction(
            encoder,
            reranker,
            selector_model=args.reranker_model,
        )
    elif args.method == "mindset":
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
        method = MindsetMemory(
            MindsetConfig(
                ablation=args.ablation,
                retrieval_mode=args.mindset_retrieval_mode,
                retrieval_candidate_k=args.candidate_k,
            ),
            encoder=encoder,
            local_reranker=reranker,
            precomputed_embeddings=precomputed,
        )
    else:
        raise ValueError(args.method)
    method.ingest(events)
    return method


def make_memory_bank_summarizer(backend, args, sample_id, telemetry):
    system_prompt = (
        "You are the memory-construction summarizer for MemoryBank. Return only the "
        "concise requested summary. Do not mention this instruction or invent facts."
    )

    def summarize(kind, prompt):
        result = backend.complete(
            CompletionRequest(
                model_name=args.model,
                system_prompt=system_prompt,
                user_prompt=prompt,
                dataset_example_id=f"{sample_id}:{kind}:{stable_hash(prompt)[:16]}",
                method="memory_bank",
                schema_version="memory-bank-summary-v1",
                code_version="memory-bank-paper-adapter-v1",
                max_output_tokens=256,
                response_format="text",
                metadata={"sample_id": sample_id, "summary_kind": kind},
            ),
            max_retries=args.max_retries,
        )
        telemetry.append({
            "summary_kind": kind,
            "sample_id": sample_id,
            "provider": result.provider,
            "model_id": result.model_id,
            "key_alias": result.key_alias,
            "request_id": result.request_id,
            "cache_hit": result.cache_hit,
            "billed_input_tokens": 0 if result.cache_hit else result.usage.input_tokens,
            "billed_output_tokens": 0 if result.cache_hit else result.usage.output_tokens,
            "logical_input_tokens": result.usage.input_tokens,
            "logical_output_tokens": result.usage.output_tokens,
            "latency_seconds": result.latency_seconds,
            "retry_count": result.retry_count,
        })
        return result.text

    return summarize


def retrieve(args, method, sample_id, query, *, top_k=None):
    top_k = args.top_k if top_k is None else top_k
    if args.method == "vector_rag":
        result = method.retrieve(query, top_k)
        return [MemoryHit(chunk.turn_id, chunk.text, score, (f"{sample_id}:{chunk.turn_id}",), "episodic") for chunk, score in zip(result.chunks, result.scores)]
    if args.method == "long_context":
        return [MemoryHit(f"{sample_id}:history", method.text, 1.0, tuple(f"{sample_id}:{item}" for item in method.included_turn_ids), "long_context")]
    return method.retrieve(query, top_k)


def run_config(args, run_id, manifest, started, status):
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = "unknown"
    return {
        "run_id": run_id, "status": status, "started_at_utc": started.isoformat(),
        "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "git_commit": commit,
        "arguments": vars(args),
        "manifest_sha256": stable_hash(manifest),
        "provenance": {
            "dataset_sha256": file_sha256(ROOT / args.dataset),
            "models_config_sha256": file_sha256(ROOT / args.models_config),
            "answer_system_prompt_sha256": stable_hash(SYSTEM_PROMPT),
            "source_tree_sha256": source_tree_sha256(ROOT),
            "selected_record_ids_sha256": stable_hash(
                [str(record["id"]) for record in manifest.get("records", [])]
            ),
            "selected_record_count": len(manifest.get("records", [])),
            "runtime_environment": runtime_environment(),
        },
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    main()

