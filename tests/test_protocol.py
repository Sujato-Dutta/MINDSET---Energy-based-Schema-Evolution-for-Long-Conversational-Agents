from __future__ import annotations

import json
from pathlib import Path

import pytest

from mindset.experiments.protocol import compile_protocol


def test_protocol_expands_mindset_ablations_only() -> None:
    tasks = compile_protocol(
        {
            "dataset": "locomo",
            "methods": ["vector_rag", "mindset"],
            "model": "m",
            "required_ablations": ["no_temporal_term", "nearest_schema_merge"],
        }
    )
    assert len(tasks) == 4
    assert sum(task.method == "vector_rag" for task in tasks) == 1
    assert len({task.task_id for task in tasks}) == len(tasks)


def test_protocol_supports_energy_ablations_and_repeatability_cells() -> None:
    tasks = compile_protocol(
        {
            "datasets": {
                "locomo": {
                    "manifest": "primary.json",
                    "cross_model_manifest": "locomo-cross.json",
                },
                "other": {
                    "manifest": "other.json",
                    "cross_model_manifest": "other-cross.json",
                },
            },
            "required_methods": ["hybrid_rag", "lightmem", "mindset"],
            "primary_model": "m",
            "primary_ablations": ["no_hysteresis"],
            "targeted_appendix_ablations": {"no_provenance_expansion": "kill.json"},
            "local_only_diagnostics": ["no_fragmentation_penalty"],
            "repeatability": {
                "methods": ["hybrid_rag", "lightmem", "mindset"],
                "manifest": "repeat.json",
                "independent_runs": 3,
            },
            "cross_model": {
                "models": ["qwen", "llama"],
                "methods": ["hybrid_rag", "lightmem", "mindset"],
            },
        }
    )
    assert len(tasks) == 6 + 1 + 1 + 1 + 9 + 12
    assert sum(task.scope == "primary_ablation" for task in tasks) == 1
    assert sum(task.scope == "targeted_appendix" for task in tasks) == 1
    assert sum(task.scope == "local_only_diagnostic" for task in tasks) == 1
    repeats = [task for task in tasks if task.scope == "repeatability"]
    assert {task.repetition_id for task in repeats} == {"repeat-1", "repeat-2", "repeat-3"}
    assert all(task.manifest == "repeat.json" for task in repeats)
    cross_model = [task for task in tasks if task.scope == "cross_model"]
    assert len(cross_model) == 12
    assert {task.model for task in cross_model} == {"qwen", "llama"}


def test_evaluation_judge_cannot_be_an_answer_model() -> None:
    with pytest.raises(ValueError, match="evaluation-only judge"):
        compile_protocol(
            {
                "dataset": "locomo",
                "methods": ["mindset"],
                "model": "gemini_judge",
                "judge": {"model": "gemini_judge"},
            }
        )


def test_main_protocol_keeps_gemini_out_of_task_execution() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/main.yaml").read_text(encoding="utf-8"))
    tasks = compile_protocol(config)
    assert config["primary_model"] == "gpt_oss_120b"
    assert set(config["cross_model"]["models"]) == {"qwen_27b", "llama_scout_17b"}
    assert config["judge"]["role"] == "evaluation_only"
    assert all(task.model != config["judge"]["model"] for task in tasks)


def test_energy_answer_manifest_is_balanced_and_nested() -> None:
    root = Path(__file__).resolve().parents[1]
    parent = json.loads(
        (root / "data/manifests/locomo_kill120.json").read_text(encoding="utf-8")
    )
    answer = json.loads(
        (root / "data/manifests/locomo_energy_answer40.json").read_text(
            encoding="utf-8"
        )
    )
    parent_ids = {record["id"] for record in parent["records"]}
    answer_ids = [record["id"] for record in answer["records"]]
    assert answer["count"] == len(answer_ids) == 40
    assert len(set(answer_ids)) == 40
    assert set(answer_ids) <= parent_ids
    assert {
        category: sum(int(record["category"]) == category for record in answer["records"])
        for category in range(1, 6)
    } == {category: 8 for category in range(1, 6)}


def test_local_rerank_split_is_disjoint_balanced_and_complete() -> None:
    root = Path(__file__).resolve().parents[1]
    parent = json.loads(
        (root / "data/manifests/locomo_kill120.json").read_text(encoding="utf-8")
    )
    split = json.loads(
        (root / "data/manifests/locomo_lr_split120.json").read_text(encoding="utf-8")
    )
    records = {record["id"]: record for record in parent["records"]}
    tune, evaluation = set(split["tune_ids"]), set(split["evaluation_ids"])
    assert len(tune) == len(evaluation) == 60
    assert not tune & evaluation
    assert tune | evaluation == set(records)
    for identifiers in (tune, evaluation):
        assert {
            category: sum(
                int(records[identifier]["category"]) == category
                for identifier in identifiers
            )
            for category in range(1, 6)
        } == {category: 12 for category in range(1, 6)}
