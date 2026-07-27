from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from sentence_transformers import CrossEncoder, SentenceTransformer

from mindset.api.backend import APIBackend
from mindset.api.models import CompletionRequest
from mindset.baselines.event_rag import EventRAGIndex, render_event
from mindset.baselines.hybrid_rag import HybridRAGIndex
from mindset.baselines.lightmem import LightMemACL2026Reproduction
from mindset.baselines.memory_bank import MemoryBankReproduction
from mindset.datasets.events import longmemeval_events, memoryagentbench_context_events
from mindset.datasets.events import subtlememory_events
from mindset.datasets.subsets import load_json_records, load_memoryagentbench_questions, stable_record_id
from mindset.evaluation.longmemeval import aggregate_judge_labels, retrieval_scores
from mindset.evaluation.memoryagentbench import evaluate_memoryagentbench_ttl
from mindset.evaluation.proxy_judge import judge_with_proxy_consensus
from mindset.experiments.batching import (
    SYSTEM_PROMPT,
    AnswerQuestion,
    answer_batches,
    assert_budget_preflight,
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
from mindset.methods.base import MemoryHit
from mindset.methods.mindset import MindsetConfig, MindsetMemory
from mindset.methods.nemori import NemoriFaithfulReproduction
from mindset.methods.rerank_cache import CachedReranker
from mindset.utils.hashing import stable_hash


ANSWER_MODELS = ("gpt_oss_120b", "qwen_27b", "llama_scout_17b")


def main():
    parser = argparse.ArgumentParser(description="Run LongMemEval or MemoryAgentBench with the shared method matrix")
    parser.add_argument("--dataset", choices=("longmemeval", "memoryagentbench"), required=True)
    parser.add_argument(
        "--method",
        choices=(
            "long_context",
            "vector_rag",
            "hybrid_rag",
            "nemori",
            "lightmem",
            "mindset",
            "memory_bank",
        ),
        required=True,
    )
    parser.add_argument("--ablation", default="full")
    parser.add_argument("--model", choices=ANSWER_MODELS, default="gpt_oss_120b")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--manifest")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    parser.add_argument("--hybrid-candidate-k", type=int, default=24)
    parser.add_argument(
        "--mindset-retrieval-mode",
        choices=("legacy", "cross_encoder_only", "hybrid", "coverage", "query_adaptive"),
        default="cross_encoder_only",
    )
    parser.add_argument("--candidate-k", type=int, default=40)
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
    parser.add_argument("--repetition-id", default="primary")
    parser.add_argument("--proxy-judge", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--retrieval-only", action="store_true")
    for action in parser._actions:
        if action.dest == 'dataset':
            action.choices = tuple(action.choices) + ('subtlememory',)
            break
    args = parser.parse_args()
    if args.method != "mindset" and args.ablation != "full":
        raise ValueError("Ablations apply only to MINDSET")

    if args.dataset == 'subtlememory':
        args.manifest = 'data/manifests/subtlememory_final150.json'
    manifest_path = ROOT / (args.manifest or (
        "data/manifests/longmemeval_primary100.json" if args.dataset == "longmemeval"
        else "data/manifests/memoryagentbench_ttl.json"
    ))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected_ids = {str(row["id"]) for row in manifest.get("selected", [])}
    cases, questions = load_cases(args.dataset, selected_ids)
    backend = APIBackend.from_file(ROOT / args.models_config)
    preflight = {"decision": "pass", "ingestion": "local_no_api", "network_calls_made": False}
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"_{args.ablation}" if args.method == "mindset" else ""
    run_dir = ROOT / "outputs/runs" / (
        f"{args.dataset}_{args.method}{suffix}_{args.model}_{args.repetition_id}_{stamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "preflight.json", preflight)
    if args.preflight_only:
        write_json(
            run_dir / "run_config.json",
            run_configuration(args, manifest, manifest_path, run_dir.name, "preflight_passed"),
        )
        print(json.dumps({"run_dir": str(run_dir), "preflight": preflight}, indent=2))
        return

    encoder = None
    evidence_tokenizer = None
    if args.method in {"vector_rag", "hybrid_rag", "nemori", "lightmem", "mindset", "memory_bank"}:
        encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        if args.bounded_evidence:
            evidence_tokenizer = TokenCounter(args.model)
    reranker = None
    if args.method in {"lightmem", "mindset"}:
        reranker = CachedReranker(
            CrossEncoder(args.reranker_model, device="cpu"),
            args.reranker_model,
            ROOT / "outputs/caches/rerank_scores.sqlite",
        )
    methods = {}
    ingestion_seconds = {}
    memory_construction_records: list[dict[str, object]] = []
    for case_id, events in cases.items():
        ingestion_started = perf_counter()
        methods[case_id] = build_method(
            args, events, encoder, reranker, backend, case_id, memory_construction_records
        )
        ingestion_seconds[case_id] = perf_counter() - ingestion_started

    answer_inputs = []
    rows = []
    for question in sorted(questions, key=lambda item: (item["case_id"], item["id"])):
        case_id = question["case_id"]
        retrieval_started = perf_counter()
        candidate_k = (
            args.evidence_candidate_k
            if args.bounded_evidence and args.method != "long_context"
            else args.top_k
        )
        hits = retrieve(args, methods[case_id], case_id, question["question"], candidate_k)
        if args.bounded_evidence and args.method != "long_context":
            hits = bound_evidence(
                hits,
                question["question"],
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
        retrieval_seconds = perf_counter() - retrieval_started
        packet_bytes = sum(len(hit.text.encode("utf-8")) for hit in hits)
        answer_inputs.append(AnswerQuestion(question["id"], question["question"], tuple(hits)))
        rows.append({
            **question,
            "hits": hits,
            "retrieval_seconds": retrieval_seconds,
            "evidence_packet_bytes": packet_bytes,
        })
    if args.retrieval_only:
        metrics = retrieval_metrics(args.dataset, rows)
        metrics["status"] = "retrieval_smoke_complete"
        metrics["efficiency"] = local_efficiency(
            methods, ingestion_seconds, rows, answer_records=[],
            construction_records=memory_construction_records,
        )
        write_json(run_dir / "metrics.json", metrics)
        write_jsonl(run_dir / "retrieval.jsonl", retrieval_audit_rows(rows))
        write_json(
            run_dir / "run_config.json",
            run_configuration(
                args, manifest, manifest_path, run_dir.name, "retrieval_smoke_complete"
            ),
        )
        close_methods(methods)
        print(json.dumps({"run_dir": str(run_dir), "metrics": metrics}, indent=2))
        return

    batches = build_answer_batches(answer_inputs, max_questions=args.batch_size, max_prompt_bytes=args.max_prompt_bytes)
    answer_telemetry: list[dict[str, object]] = []
    answers = answer_batches(
        backend,
        batches,
        model_name=args.model,
        method=args.method,
        code_version=f"aux-paper-run-v5-six-method:{args.repetition_id}",
        max_output_tokens=args.max_output_tokens,
        max_retries=args.max_retries,
        reserve_fraction=args.reserve_fraction,
        telemetry=answer_telemetry,
        request_metadata={
            "run_id": run_dir.name,
            "dataset": args.dataset,
            "repetition_id": args.repetition_id,
            "ablation": args.ablation,
        },
    )
    predictions = []
    for row in rows:
        predictions.append({
            **{k: v for k, v in row.items() if k != "hits"},
            "prediction": answers[row["id"]],
            "retrieved_ids": [
                source for hit in row["hits"] for source in hit.source_event_ids
            ],
        })
    judge_telemetry: list[dict[str, object]] = []
    if args.dataset in ("longmemeval", "subtlememory"):
        metrics = retrieval_metrics(args.dataset, rows)
        if args.proxy_judge:
            judge_reservation = sum(
                (len((row["question"] + str(row["answer"]) + answers[row["id"]]).encode("utf-8")) + 256)
                * 3
                for row in rows
            )
            assert_budget_preflight(
                backend, model_name="gemini_judge", required_tokens=judge_reservation
            )
            for prediction in predictions:
                decision = judge_with_proxy_consensus(
                    backend, question_id=prediction["id"], question=prediction["question"],
                    reference=str(prediction["answer"]), prediction=prediction["prediction"],
                    telemetry=judge_telemetry,
                    request_metadata={
                        "run_id": run_dir.name,
                        "dataset": args.dataset,
                        "method": args.method,
                        "repetition_id": args.repetition_id,
                    },
                )
                prediction["judge_label"] = decision.verdict in {"correct", "partially_correct"}
                prediction["proxy_judge"] = decision.__dict__
            metrics["proxy_judge"] = aggregate_judge_labels(predictions)
            metrics["proxy_judge"]["protocol"] = "external_gemini_proxy_v2_not_official_judge"
    else:
        metrics = evaluate_memoryagentbench_ttl(predictions, ROOT / "data/memory_agent_bench/entity2id.json")
    metrics["status"] = "complete"
    metrics["efficiency"] = local_efficiency(
        methods, ingestion_seconds, rows, answer_records=answer_telemetry,
        construction_records=memory_construction_records,
    )
    if judge_telemetry:
        metrics["efficiency"]["judge"] = token_and_latency_summary(
            judge_telemetry, len(predictions)
        )
    metrics["method_telemetry"] = {
        case_id: method.telemetry()
        for case_id, method in methods.items()
        if hasattr(method, "telemetry")
    }
    metrics["memory_invariants"] = {
        case_id: method.validate_invariants()
        for case_id, method in methods.items()
        if hasattr(method, "validate_invariants")
    }
    write_jsonl(run_dir / "predictions.jsonl", predictions)
    write_jsonl(run_dir / "retrieval.jsonl", retrieval_audit_rows(rows))
    write_jsonl(run_dir / "answer_batches.jsonl", answer_telemetry)
    if judge_telemetry:
        write_jsonl(run_dir / "judge_calls.jsonl", judge_telemetry)
    write_json(run_dir / "metrics.json", metrics)
    write_json(
        run_dir / "run_config.json",
        run_configuration(args, manifest, manifest_path, run_dir.name, "complete"),
    )
    close_methods(methods)
    print(json.dumps({"run_dir": str(run_dir), "metrics": metrics}, indent=2))


def load_cases(dataset, selected_ids):
    cases = {}
    questions = []
    if dataset == 'subtlememory':
        records = load_json_records(ROOT / 'data/subtlememory/benchmark_instances_final150.json')
        for record in records:
            identifier = str(record['instance_id'])
            if identifier not in selected_ids:
                continue
            qas = list(record.get('qas', []))
            if not qas:
                raise ValueError(f'SubtleMemory instance has no QA: {identifier}')
            qa = qas[0]
            cases[identifier] = subtlememory_events(record)
            questions.append({
                'id': identifier,
                'case_id': identifier,
                'question': str(qa['query']),
                'answer': list(qa.get('correct_answers', [])),
                'question_type': str(record.get('relation_type', 'unknown')),
                'relation_type': str(record.get('relation_type', 'unknown')),
                'relation_subtype': str(record.get('relation_subtype', 'unknown')),
                'answer_session_ids': [],
            })
        return cases, questions
    if dataset == "longmemeval":
        records = load_json_records(ROOT / "data/longmemeval/longmemeval_s_cleaned.json")
        for index, record in enumerate(records):
            identifier = stable_record_id(record, index)
            if identifier not in selected_ids:
                continue
            cases[identifier] = longmemeval_events(record)
            questions.append({
                "id": identifier, "case_id": identifier, "question": str(record["question"]),
                "answer": record.get("answer"), "question_id": identifier,
                "question_type": str(record.get("question_type", "unknown")),
                "answer_session_ids": list(record.get("answer_session_ids", [])),
            })
    else:
        path = ROOT / "data/memory_agent_bench/Test_Time_Learning-00000-of-00001.parquet"
        records, _ = load_memoryagentbench_questions(path)
        import pandas as pd
        frame = pd.read_parquet(path)
        selected = [record for record in records if record["qa_pair_id"] in selected_ids]
        for row_index in sorted({int(record["source_row"]) for record in selected}):
            case_id = f"mab-row-{row_index}"
            cases[case_id] = memoryagentbench_context_events(str(frame.iloc[row_index]["context"]), source_row=row_index)
        for record in selected:
            questions.append({
                "id": record["qa_pair_id"], "case_id": f"mab-row-{record['source_row']}",
                "question": record["question"], "answer": record["answer"],
                "answers": record["answers"], "category": record["category"],
            })
    return cases, questions


def build_method(args, events, encoder, reranker, backend, case_id, construction_records):
    if args.method == "vector_rag":
        return EventRAGIndex(events, encoder)
    if args.method == "long_context":
        return list(events)
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
                backend, args, case_id, construction_records
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
        method = MindsetMemory(
            MindsetConfig(
                ablation=args.ablation,
                retrieval_mode=args.mindset_retrieval_mode,
                retrieval_candidate_k=args.candidate_k,
            ),
            encoder=encoder,
            local_reranker=reranker,
        )
    else:
        raise ValueError(args.method)
    method.ingest(events)
    return method


def retrieve(args, method, case_id, query, top_k=None):
    if args.method == "long_context":
        selected, size = [], 0
        for event in reversed(method):
            text = render_event(event)
            if selected and size + len(text.encode("utf-8")) > args.long_context_max_history_bytes:
                break
            selected.append(event)
            size += len(text.encode("utf-8"))
        selected.reverse()
        return [MemoryHit(f"{case_id}:history", "\n".join(render_event(e) for e in selected), 1.0, tuple(e.event_id for e in selected), "long_context")]
    return method.retrieve(query, top_k if top_k is not None else args.top_k)


def retrieval_metrics(dataset, rows):
    if dataset == 'subtlememory':
        relation_counts = defaultdict(int)
        for row in rows:
            relation_counts[str(row.get('relation_type', 'unknown'))] += 1
        return {
            'question_count': len(rows),
            'relation_type_distribution': dict(sorted(relation_counts.items())),
        }
    if dataset != "longmemeval":
        return {"question_count": len(rows)}
    values = defaultdict(list)
    for row in rows:
        ranked = []
        for hit in row["hits"]:
            session = hit.metadata.get("benchmark_session_id")
            if session and session not in ranked:
                ranked.append(session)
            for source in hit.source_event_ids:
                parts = source.split(":")
                if len(parts) >= 3 and parts[-2] not in ranked:
                    ranked.append(parts[-2])
        for name, value in retrieval_scores(ranked, row["answer_session_ids"], 5).items():
            values[name].append(value)
    return {name: sum(items) / len(items) for name, items in values.items()} | {"question_count": len(rows)}


def local_efficiency(methods, ingestion_seconds, rows, *, answer_records, construction_records):
    construction = token_and_latency_summary(
        construction_records, max(1, len(methods))
    )
    return {
        "ingestion_seconds_by_case": ingestion_seconds,
        "ingestion_seconds": distribution_summary(list(ingestion_seconds.values())),
        "retrieval_seconds": distribution_summary(
            [float(row["retrieval_seconds"]) for row in rows]
        ),
        "evidence_packet_bytes": distribution_summary(
            [float(row["evidence_packet_bytes"]) for row in rows]
        ),
        "memory_construction_api_calls": int(construction["provider_calls"]),
        "retrieval_api_calls": 0,
        "memory_construction": construction,
        **token_and_latency_summary(answer_records, len(rows)),
    }


def make_memory_bank_summarizer(backend, args, case_id, telemetry):
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
                dataset_example_id=f"{case_id}:{kind}:{stable_hash(prompt)[:16]}",
                method="memory_bank",
                schema_version="memory-bank-summary-v1",
                code_version="memory-bank-paper-adapter-v1",
                max_output_tokens=256,
                response_format="text",
                metadata={"case_id": case_id, "summary_kind": kind},
            ),
            max_retries=args.max_retries,
        )
        telemetry.append({
            "summary_kind": kind,
            "case_id": case_id,
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


def retrieval_audit_rows(rows):
    return [
        {
            **{key: value for key, value in row.items() if key != "hits"},
            "hits": [
                {
                    "memory_id": hit.memory_id,
                    "text": hit.text,
                    "score": hit.score,
                    "source_event_ids": hit.source_event_ids,
                    "memory_type": hit.memory_type,
                    "metadata": hit.metadata,
                }
                for hit in row["hits"]
            ],
        }
        for row in rows
    ]


def run_configuration(args, manifest, manifest_path, run_id, status):
    source_paths = (
        [ROOT / "data/longmemeval/longmemeval_s_cleaned.json"]
        if args.dataset == "longmemeval"
        else [
            ROOT / "data/memory_agent_bench/Test_Time_Learning-00000-of-00001.parquet",
            ROOT / "data/memory_agent_bench/entity2id.json",
        ]
    )
    if args.dataset == 'subtlememory':
        source_paths = [ROOT / 'data/subtlememory/benchmark_instances_final150.json']
    selected = manifest.get("selected", manifest.get("records", []))
    return {
        "run_id": run_id,
        "status": status,
        "dataset": args.dataset,
        "method": args.method,
        "ablation": args.ablation,
        "model": args.model,
        "manifest": str(manifest_path),
        "repetition_id": args.repetition_id,
        "arguments": vars(args),
        "provenance": {
            "manifest_sha256": file_sha256(manifest_path),
            "source_files_sha256": {
                str(path.relative_to(ROOT)): file_sha256(path) for path in source_paths
            },
            "models_config_sha256": file_sha256(ROOT / args.models_config),
            "answer_system_prompt_sha256": stable_hash(SYSTEM_PROMPT),
            "source_tree_sha256": source_tree_sha256(ROOT),
            "selected_record_ids_sha256": stable_hash(
                [str(record.get("id")) for record in selected]
            ),
            "selected_record_count": len(selected),
            "runtime_environment": runtime_environment(),
        },
    }


def close_methods(methods):
    for method in methods.values():
        close = getattr(method, "close", None)
        if close:
            close()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
