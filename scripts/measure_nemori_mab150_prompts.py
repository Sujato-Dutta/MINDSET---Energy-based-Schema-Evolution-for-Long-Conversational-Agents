"""Measure Nemori answer-prompt sizes locally without provider calls."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from sentence_transformers import SentenceTransformer

from scripts.run_aux_benchmark import build_method, load_cases, retrieve
from mindset.experiments.batching import SYSTEM_PROMPT, AnswerQuestion, render_batch_prompt


def main() -> None:
    manifest_path = ROOT / "data/manifests/memoryagentbench_final150.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected_ids = {str(row["id"]) for row in manifest.get("selected", [])}
    cases, questions = load_cases("memoryagentbench", selected_ids)
    args = SimpleNamespace(
        method="nemori",
        top_k=8,
        long_context_max_history_bytes=20000,
        hybrid_candidate_k=24,
        reranker_model="cross-encoder/ms-marco-MiniLM-L6-v2",
        ablation="full",
    )
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    methods = {case_id: build_method(args, events, encoder, None) for case_id, events in cases.items()}
    rows = []
    for question in sorted(questions, key=lambda item: (item["case_id"], item["id"])):
        hits = retrieve(args, methods[question["case_id"]], question["case_id"], question["question"])
        answer_question = AnswerQuestion(
            str(question["id"]),
            str(question["question"]),
            tuple(hits),
        )
        prompt = render_batch_prompt([answer_question])
        prompt_bytes = len(prompt.encode("utf-8"))
        system_bytes = len(SYSTEM_PROMPT.encode("utf-8"))
        rows.append({
            "question_id": str(question["id"]),
            "case_id": str(question["case_id"]),
            "prompt_bytes": prompt_bytes,
            "system_plus_prompt_bytes": system_bytes + prompt_bytes,
            "estimated_attempt_bytes": system_bytes + prompt_bytes + 4096,
            "estimated_three_attempt_bytes": (system_bytes + prompt_bytes + 4096) * 3,
            "hit_count": len(hits),
            "evidence_bytes": sum(len(hit.text.encode("utf-8")) for hit in hits),
        })
    rows.sort(key=lambda row: (-row["prompt_bytes"], row["question_id"]))
    output = {
        "dataset": "memoryagentbench",
        "manifest": str(manifest_path.relative_to(ROOT)),
        "method": "nemori",
        "top_k": 8,
        "question_count": len(rows),
        "prompt_guard_bytes_current": 1000000,
        "provider_estimator": "UTF-8 byte count plus max_output_tokens; retry multiplier shown separately",
        "max_prompt_bytes": rows[0]["prompt_bytes"] if rows else 0,
        "max_prompt_question_id": rows[0]["question_id"] if rows else None,
        "rows": rows,
    }
    output_path = ROOT / "outputs/diagnostics/nemori_mab150_prompt_sizes.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + chr(10), encoding="utf-8")
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    print(json.dumps({key: output[key] for key in (
        "question_count", "max_prompt_bytes", "max_prompt_question_id", "prompt_guard_bytes_current"
    )}, indent=2))


if __name__ == "__main__":
    main()
