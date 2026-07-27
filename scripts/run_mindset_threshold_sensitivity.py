from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from sentence_transformers import CrossEncoder, SentenceTransformer

from mindset.baselines.vector_rag import VectorRAGIndex
from mindset.datasets.locomo import load_locomo
from mindset.evaluation.metrics import retrieval_metrics
from mindset.methods.base import ConversationEvent
from mindset.methods.mindset import MindsetConfig, MindsetMemory
from mindset.methods.rerank_cache import CachedReranker


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-factor MINDSET transition-parameter sensitivity on LoCoMo-700"
    )
    parser.add_argument("--dataset", default="data/locomo/locomo10.json")
    parser.add_argument("--manifest", default="data/manifests/locomo_primary700.json")
    parser.add_argument(
        "--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument(
        "--reranker-model", default="cross-encoder/ms-marco-MiniLM-L6-v2"
    )
    parser.add_argument("--output", default="outputs/analysis/mindset_sensitivity_v1.json")
    parser.add_argument("--markdown", default="outputs/analysis/mindset_sensitivity_v1.md")
    args = parser.parse_args()

    manifest = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))
    records = manifest["records"]
    if len(records) != 700:
        raise ValueError("Paper sensitivity requires the frozen 700-question manifest")

    samples = {sample.sample_id: sample for sample in load_locomo(ROOT / args.dataset)}
    selected_ids = sorted({str(row["sample_id"]) for row in records})
    encoder = SentenceTransformer(args.embedding_model, device="cpu")
    reranker = CachedReranker(
        CrossEncoder(args.reranker_model, device="cpu"),
        args.reranker_model,
        ROOT / "outputs/caches/rerank_scores.sqlite",
    )
    events_by_sample: dict[str, list[ConversationEvent]] = {}
    embeddings_by_sample: dict[str, dict[str, Any]] = {}
    for sample_id in selected_ids:
        sample = samples[sample_id]
        events = sample_events(sample)
        index = VectorRAGIndex.build(
            sample,
            encoder,
            args.embedding_model,
            ROOT / "outputs/caches/embeddings",
        )
        events_by_sample[sample_id] = events
        embeddings_by_sample[sample_id] = {
            event.event_id: vector
            for event, vector in zip(events, index.embeddings, strict=True)
        }

    base = MindsetConfig(retrieval_mode="cross_encoder_only", retrieval_candidate_k=40)
    variants = [
        ("contradiction weight", 1.00, replace(base, contradiction_weight=1.00)),
        ("contradiction weight", 1.25, base),
        ("contradiction weight", 1.50, replace(base, contradiction_weight=1.50)),
        ("fragmentation weight", 0.15, replace(base, fragmentation_weight=0.15)),
        ("fragmentation weight", 0.22, base),
        ("fragmentation weight", 0.30, replace(base, fragmentation_weight=0.30)),
        ("switch margin", 0.12, replace(base, switch_margin=0.12)),
        ("switch margin", 0.18, base),
        ("switch margin", 0.24, replace(base, switch_margin=0.24)),
    ]
    cache: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for position, (parameter, value, config) in enumerate(variants, start=1):
        key = json.dumps(asdict(config), sort_keys=True)
        if key not in cache:
            print(f"SENSITIVITY {position}/{len(variants)} {parameter}={value:.2f}", flush=True)
            cache[key] = evaluate_config(
                config,
                records,
                samples,
                events_by_sample,
                embeddings_by_sample,
                encoder,
                reranker,
            )
        rows.append(
            {
                "parameter": parameter,
                "value": value,
                "chosen": config == base,
                **cache[key],
            }
        )

    payload = {
        "protocol": "one-factor-at-a-time",
        "dataset": args.dataset,
        "manifest": args.manifest,
        "selection_policy": "Evaluate only the frozen 700-question LoCoMo paper manifest.",
        "answer_model_calls": 0,
        "judge_calls": 0,
        "base_config": asdict(base),
        "rows": rows,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = ROOT / args.markdown
    markdown.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"output": str(output), "markdown": str(markdown)}, indent=2))


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


def evaluate_config(
    config: MindsetConfig,
    records: list[dict[str, Any]],
    samples: dict[str, Any],
    events_by_sample: dict[str, list[ConversationEvent]],
    embeddings_by_sample: dict[str, dict[str, Any]],
    encoder: Any,
    reranker: Any,
) -> dict[str, Any]:
    memories: dict[str, MindsetMemory] = {}
    for sample_id, events in events_by_sample.items():
        memory = MindsetMemory(
            config,
            encoder=encoder,
            local_reranker=reranker,
            precomputed_embeddings=embeddings_by_sample[sample_id],
        )
        memory.ingest(events)
        memories[sample_id] = memory

    metric_values = {
        metric: [] for metric in ("recall@8", "precision@8", "mrr", "ndcg@8")
    }
    for record in records:
        sample_id = str(record["sample_id"])
        question = samples[sample_id].questions[int(record["question_index"])]
        gold = {
            value.strip()
            for group in question.evidence
            for value in group.split(";")
            if value.strip()
        }
        if not gold:
            continue
        hits = memories[sample_id].retrieve(question.question, top_k=8)
        retrieved = [hit.memory_id.split(":", 1)[-1] for hit in hits]
        report = retrieval_metrics(retrieved, gold, 8)
        for metric in metric_values:
            metric_values[metric].append(float(report[metric]))

    operation_counts: Counter[str] = Counter()
    for memory in memories.values():
        operation_counts.update(row.operation for row in memory.operation_log.all())
    structure = {
        "schema_count": sum(len(memory.schemas) for memory in memories.values()),
        "active_schema_count": sum(
            schema.status == "active"
            for memory in memories.values()
            for schema in memory.schemas.values()
        ),
        "historical_schema_count": sum(
            schema.status == "historical"
            for memory in memories.values()
            for schema in memory.schemas.values()
        ),
        "operation_counts": dict(sorted(operation_counts.items())),
    }
    return {
        "metrics": {
            "paper700": {
                metric: sum(values) / len(values)
                for metric, values in metric_values.items()
            }
        },
        "structure": structure,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# MINDSET transition-parameter sensitivity",
        "",
        "One factor is varied at a time. Metrics are deterministic local retrieval results on the 696 evidence-annotated questions within the frozen 700-question LoCoMo paper manifest; no answer-model or judge calls are used.",
        "",
        "| Parameter | Value | Recall@8 | Precision@8 | MRR | nDCG@8 | Schemas | New / Split / Supersede |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        metrics = row["metrics"]["paper700"]
        operations = row["structure"]["operation_counts"]
        value = f"**{row['value']:.2f}**" if row["chosen"] else f"{row['value']:.2f}"
        lines.append(
            f"| {row['parameter']} | {value} | {metrics['recall@8']:.4f} | "
            f"{metrics['precision@8']:.4f} | {metrics['mrr']:.4f} | "
            f"{metrics['ndcg@8']:.4f} | {row['structure']['schema_count']} | "
            f"{operations.get('new', 0)} / {operations.get('split', 0)} / "
            f"{operations.get('supersede', 0)} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()




