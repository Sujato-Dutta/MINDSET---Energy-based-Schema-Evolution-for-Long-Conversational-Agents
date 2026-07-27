from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mindset.utils.hashing import stable_hash


@dataclass(frozen=True)
class ProtocolTask:
    task_id: str
    dataset: str
    method: str
    model: str
    ablation: str = "none"
    manifest: str | None = None
    repetition_id: str = "primary"
    scope: str = "primary"


def compile_protocol(config: dict[str, Any]) -> list[ProtocolTask]:
    datasets = config.get("datasets")
    if isinstance(datasets, dict):
        dataset_names = list(datasets)
    elif config.get("dataset"):
        dataset_names = [str(config["dataset"])]
    else:
        raise ValueError("Protocol must define dataset or datasets")
    methods = list(config.get("required_methods", config.get("methods", [])))
    if not methods:
        raise ValueError("Protocol has no methods")
    models = config.get("models") or [config.get("primary_model", config.get("model"))]
    models = [str(model) for model in models if model]
    if not models:
        raise ValueError("Protocol has no model")
    judge_model = str(config.get("judge", {}).get("model", ""))
    cross_models = {
        str(model) for model in config.get("cross_model", {}).get("models", [])
    }
    if judge_model and judge_model in {*models, *cross_models}:
        raise ValueError("The evaluation-only judge cannot be an answer model")
    tasks: list[ProtocolTask] = []

    def add_task(
        dataset: str,
        method: str,
        model: str,
        *,
        ablation: str = "none",
        manifest: str | None = None,
        repetition_id: str = "primary",
        scope: str = "primary",
    ) -> None:
        payload = {
            "dataset": dataset,
            "method": method,
            "model": model,
            "ablation": ablation,
            "manifest": manifest,
            "repetition_id": repetition_id,
            "scope": scope,
        }
        tasks.append(ProtocolTask("task-" + stable_hash(payload)[:16], **payload))

    required_ablations = set(config.get("required_ablations", []))
    for dataset in dataset_names:
        dataset_config = datasets.get(dataset, {}) if isinstance(datasets, dict) else {}
        manifest = dataset_config.get("manifest") if isinstance(dataset_config, dict) else config.get("manifest")
        for method in methods:
            variants = ["none"]
            if method == "mindset" and required_ablations:
                variants.extend(sorted(required_ablations))
            for model in models:
                for ablation in variants:
                    add_task(
                        dataset,
                        method,
                        model,
                        ablation=ablation,
                        manifest=manifest,
                        scope="kill" if required_ablations else "primary",
                    )

    primary_model = models[0]
    if "locomo" in dataset_names:
        locomo_config = datasets.get("locomo", {}) if isinstance(datasets, dict) else {}
        locomo_manifest = (
            locomo_config.get("manifest") if isinstance(locomo_config, dict) else config.get("manifest")
        )
        for ablation in config.get("primary_ablations", []):
            add_task(
                "locomo",
                "mindset",
                primary_model,
                ablation=str(ablation),
                manifest=locomo_manifest,
                scope="primary_ablation",
            )
        for ablation, manifest in config.get("targeted_appendix_ablations", {}).items():
            add_task(
                "locomo",
                "mindset",
                primary_model,
                ablation=str(ablation),
                manifest=str(manifest),
                scope="targeted_appendix",
            )
        for ablation in config.get("local_only_diagnostics", []):
            add_task(
                "locomo",
                "mindset",
                primary_model,
                ablation=str(ablation),
                manifest=locomo_manifest,
                scope="local_only_diagnostic",
            )

    repeatability = config.get("repeatability", {})
    if repeatability:
        for run in range(1, int(repeatability.get("independent_runs", 0)) + 1):
            for method in repeatability.get("methods", []):
                add_task(
                    "locomo",
                    str(method),
                    primary_model,
                    manifest=str(repeatability["manifest"]),
                    repetition_id=f"repeat-{run}",
                    scope="repeatability",
                )

    cross_model = config.get("cross_model", {})
    if cross_model and isinstance(datasets, dict):
        for dataset, dataset_config in datasets.items():
            manifest = dataset_config.get("cross_model_manifest")
            if not manifest:
                raise ValueError(f"Dataset {dataset} has no cross_model_manifest")
            for model in cross_model.get("models", []):
                for method in cross_model.get("methods", []):
                    add_task(
                        str(dataset),
                        str(method),
                        str(model),
                        manifest=str(manifest),
                        repetition_id="cross-model",
                        scope="cross_model",
                    )
    return tasks


def freeze_run_plan(config_path: str | Path, output_path: str | Path) -> Path:
    source = Path(config_path)
    config = json.loads(source.read_text(encoding="utf-8"))
    tasks = compile_protocol(config)
    payload = {
        "plan_version": "paper-run-plan-v1",
        "config_path": str(source.resolve()),
        "config_sha256": stable_hash(config),
        "task_count": len(tasks),
        "tasks": [task.__dict__ for task in tasks],
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output.exists() and output.read_text(encoding="utf-8") != encoded:
        raise FileExistsError(f"Frozen run plan differs: {output}")
    output.write_text(encoded, encoding="utf-8")
    return output
