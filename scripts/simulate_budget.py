from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindset.api.backend import APIBackend


METHOD_INPUT_TOKENS_PER_BATCH = {
    "long_context": 6_000,
    "vector_rag": 12_000,
    "hybrid_rag": 12_000,
    "nemori": 16_000,
    "lightmem": 12_000,
    "mindset": 12_000,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Conservative no-network six-method paper-run budget simulation"
    )
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--model", default="gpt_oss_120b")
    parser.add_argument("--locomo-questions", type=int, default=700)
    parser.add_argument("--longmemeval-questions", type=int, default=100)
    parser.add_argument("--memoryagentbench-questions", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--answer-output-cap", type=int, default=4096)
    parser.add_argument("--primary-ablation-count", type=int, default=6)
    parser.add_argument("--targeted-appendix-questions", type=int, default=0)
    parser.add_argument("--repeatability-methods", type=int, default=2)
    parser.add_argument("--repeatability-runs", type=int, default=3)
    parser.add_argument("--repeatability-questions", type=int, default=100)
    parser.add_argument("--gemini-judge-methods", type=int, default=6)
    parser.add_argument("--gemini-keys", type=int, default=3)
    parser.add_argument("--gemini-requests-per-key-day", type=int, default=500)
    parser.add_argument("--cerebras-keys", type=int, default=10)
    parser.add_argument("--cerebras-tokens-per-key-day", type=int, default=1_000_000)
    parser.add_argument("--cross-model-locomo-questions", type=int, default=120)
    parser.add_argument("--cross-model-longmemeval-questions", type=int, default=60)
    parser.add_argument("--cross-model-memoryagentbench-questions", type=int, default=40)
    parser.add_argument("--cross-model-count", type=int, default=2)
    parser.add_argument("--groq-keys", type=int, default=3)
    parser.add_argument("--groq-tokens-per-key-day", type=int, default=500_000)
    parser.add_argument("--reserve-fraction", type=float, default=0.10)
    parser.add_argument("--output", default="outputs/preflight/six_method_budget.json")
    args = parser.parse_args()
    dataset_questions = {
        "locomo": args.locomo_questions,
        "longmemeval": args.longmemeval_questions,
        "memoryagentbench": args.memoryagentbench_questions,
    }
    if min(dataset_questions.values()) <= 0 or args.batch_size <= 0 or args.answer_output_cap <= 0:
        raise ValueError("dataset question counts, batch-size, and answer-output-cap must be positive")

    batches_by_dataset = {
        dataset: math.ceil(count / args.batch_size)
        for dataset, count in dataset_questions.items()
    }
    by_method = {}
    for method, input_tokens in METHOD_INPUT_TOKENS_PER_BATCH.items():
        method_batch_size = 4 if method == "nemori" else args.batch_size
        answer_batches = sum(
            math.ceil(count / method_batch_size) for count in dataset_questions.values()
        )
        answering = answer_batches * (input_tokens + args.answer_output_cap)
        by_method[method] = {
            "answer_batches": answer_batches,
            "batch_size": method_batch_size,
            "estimated_answer_tokens": answering,
            "ingestion_api_tokens": 0,
            "ingestion": "local_no_api",
            "estimated_total_tokens": answering,
        }
    primary_required = sum(row["estimated_total_tokens"] for row in by_method.values())
    locomo_batches = batches_by_dataset["locomo"]
    mindset_batch_tokens = METHOD_INPUT_TOKENS_PER_BATCH["mindset"] + args.answer_output_cap
    ablation_required = args.primary_ablation_count * locomo_batches * mindset_batch_tokens
    targeted_required = math.ceil(
        args.targeted_appendix_questions / args.batch_size
    ) * mindset_batch_tokens
    repeatability_required = (
        args.repeatability_methods
        * args.repeatability_runs
        * math.ceil(args.repeatability_questions / args.batch_size)
        * mindset_batch_tokens
    )
    required = primary_required + ablation_required + targeted_required + repeatability_required
    cross_questions = {
        "locomo": args.cross_model_locomo_questions,
        "longmemeval": args.cross_model_longmemeval_questions,
        "memoryagentbench": args.cross_model_memoryagentbench_questions,
    }
    cross_batches = {
        "vector_rag": sum(math.ceil(value / args.batch_size) for value in cross_questions.values()),
        "hybrid_rag": sum(math.ceil(value / args.batch_size) for value in cross_questions.values()),
        "nemori": sum(math.ceil(value / 4) for value in cross_questions.values()),
        "lightmem": sum(math.ceil(value / args.batch_size) for value in cross_questions.values()),
        "mindset": sum(math.ceil(value / args.batch_size) for value in cross_questions.values()),
    }
    cross_per_model = sum(
        cross_batches[method] * (METHOD_INPUT_TOKENS_PER_BATCH[method] + args.answer_output_cap)
        for method in cross_batches
    )
    cross_daily_capacity = args.groq_keys * args.groq_tokens_per_key_day
    judge_requests = (
        args.gemini_judge_methods * args.longmemeval_questions
        + args.cross_model_count * len(cross_batches) * args.cross_model_longmemeval_questions
    )
    judge_daily_capacity = args.gemini_keys * args.gemini_requests_per_key_day
    answer_daily_capacity = args.cerebras_keys * args.cerebras_tokens_per_key_day

    backend = APIBackend.from_file(ROOT / args.models_config)
    model = backend.config.models[args.model]
    rows = {row["model_name"]: row for row in backend.ledger.report().get("models", [])}
    row = rows.get(args.model, {})
    used = int(row.get("input_tokens", 0)) + int(row.get("output_tokens", 0))
    reserved = int(row.get("reserved_tokens", 0))
    safety_reserve = int(model.hard_token_budget * args.reserve_fraction)
    usable = model.hard_token_budget - used - reserved - safety_reserve

    report = {
        "simulation_version": "final-paper-protocol-budget-v4",
        "network_calls_made": False,
        "assumptions": {
            "model": args.model,
            "questions_by_dataset": dataset_questions,
            "total_questions": sum(dataset_questions.values()),
            "answer_batches_by_dataset": batches_by_dataset,
            "batch_size": args.batch_size,
            "answer_output_cap": args.answer_output_cap,
            "input_tokens_per_batch": METHOD_INPUT_TOKENS_PER_BATCH,
            "reserve_fraction": args.reserve_fraction,
            "primary_ablation_count": args.primary_ablation_count,
            "targeted_appendix_questions": args.targeted_appendix_questions,
            "repeatability": {
                "methods": args.repeatability_methods,
                "independent_runs": args.repeatability_runs,
                "questions_per_run": args.repeatability_questions,
            },
        },
        "methods": by_method,
        "components": {
            "six_method_primary": primary_required,
            "six_locomo_ablations": ablation_required,
            "targeted_appendix_ablation": targeted_required,
            "repeatability": repeatability_required,
        },
        "judge": {
            "model": "gemini-3.1-flash-lite",
            "requests": judge_requests,
            "daily_request_capacity": judge_daily_capacity,
            "minimum_days": math.ceil(judge_requests / judge_daily_capacity),
        },
        "cross_model": {
            "models": args.cross_model_count,
            "methods": list(cross_batches),
            "questions_by_dataset_per_model": cross_questions,
            "estimated_tokens_per_model": cross_per_model,
            "daily_capacity_per_model": cross_daily_capacity,
            "minimum_days_per_model": math.ceil(cross_per_model / cross_daily_capacity),
        },
        "tokens": {
            "required": required,
            "daily_capacity": answer_daily_capacity,
            "minimum_full_days": math.ceil(required / answer_daily_capacity),
            "used_before_run": used,
            "reserved_before_run": reserved,
            "safety_reserve": safety_reserve,
            "usable_after_reserve": usable,
        },
        "decision": "pass" if required <= usable else "blocked",
        "blocking_reasons": (
            []
            if required <= usable
            else ["Six-method estimate exceeds the remaining model budget after safety reserve."]
        ),
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
