from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "outputs/plans/memorybank_final1000_v1.json"
STATE_PATH = ROOT / "outputs/orchestration/memorybank_final1000_v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    manifests = [
        ("locomo", "data/manifests/locomo_primary700.json", "scripts/run_locomo_experiment.py"),
        ("memoryagentbench", "data/manifests/memoryagentbench_final150.json", "scripts/run_aux_benchmark.py"),
        ("subtlememory", "data/manifests/subtlememory_final150.json", "scripts/run_aux_benchmark.py"),
    ]
    tasks = []
    for dataset, manifest, runner in manifests:
        payload = {
            "dataset": dataset,
            "method": "memory_bank",
            "model": "gpt_oss_120b",
            "manifest": manifest,
            "repetition_id": "primary",
        }
        tasks.append({
            "task_id": "task-" + stable_hash(payload)[:16],
            **payload,
            "runner": runner,
        })
    config = {
        "protocol": "memorybank-final1000-v1",
        "replacement": "hybrid_rag",
        "final_method_set": [
            "nemori", "lightmem", "vector_rag", "long_context", "memory_bank", "mindset"
        ],
        "method": "memory_bank",
        "paper": {
            "title": "MemoryBank: Enhancing Large Language Models with Long-Term Memory",
            "paper_url": "https://ojs.aaai.org/index.php/AAAI/article/view/29946",
            "official_repo": "https://github.com/zhongwanjun/MemoryBank-SiliconFriend",
            "summary_model_substitution": "gpt_oss_120b replaces the released GPT-3.5 summarizer and is recorded in every run",
        },
        "datasets": {dataset: {"manifest": manifest} for dataset, manifest, _ in manifests},
        "sample_counts": {"locomo": 700, "memoryagentbench": 150, "subtlememory": 150},
        "answer_model": "gpt_oss_120b",
        "evidence_budget": {
            "candidate_k": 20,
            "chunk_size_tokens": 768,
            "chunk_overlap_tokens": 96,
            "max_chunks": 8,
            "max_tokens": 8000,
            "max_chunks_per_memory": 3,
        },
        "memorybank": {
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "retrieval": "dense_vector_matrix_equivalent_to_faiss",
            "retrieval_top_k": 8,
            "daily_event_summary": True,
            "daily_personality_summary": True,
            "global_event_summary": True,
            "global_personality_summary": True,
            "forgetting_curve": "R=exp(-t/S)",
            "initial_memory_strength": 1,
            "recall_update": "S+=1 and reset last_recall_day",
            "random_seed": 0,
        },
        "tasks": tasks,
    }
    config["config_sha256"] = stable_hash(config)
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLAN_PATH.write_text(json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    state = {
        "plan": str(PLAN_PATH.relative_to(ROOT)),
        "plan_config_sha256": config["config_sha256"],
        "manifest_sha256": {
            manifest: sha256(ROOT / manifest) for _, manifest, _ in manifests
        },
        "status": "ready",
        "completed": [],
        "attempts": [],
    }
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"plan": str(PLAN_PATH), "state": str(STATE_PATH), "task_count": len(tasks), "config_sha256": config["config_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
