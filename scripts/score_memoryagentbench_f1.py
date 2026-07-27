"""Compute supplementary deterministic F1 metrics for MemoryAgentBench-150."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from mindset.evaluation.memoryagentbench import parse_icl_prediction, recommendation_recall
from mindset.evaluation.metrics import token_f1

METHODS = ('long_context', 'vector_rag', 'hybrid_rag', 'nemori', 'lightmem', 'mindset')
RUNS_ROOT = ROOT / 'outputs/runs'
ENTITY_PATH = ROOT / 'data/memory_agent_bench/entity2id.json'
OUTPUT_PATH = ROOT / 'outputs/metrics/memoryagentbench_answer_f1_v1.json'


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def latest_run(method: str) -> Path | None:
    candidates = []
    for path in RUNS_ROOT.glob('memoryagentbench_*_primary_*'):
        if not (path / 'predictions.jsonl').exists() or not (path / 'run_config.json').exists():
            continue
        config = read_json(path / 'run_config.json')
        if config.get('method') == method and config.get('status') != 'preflight_passed':
            candidates.append(path)
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def load_predictions(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def recommendation_f1_at_5(prediction: str, answers: list[str]) -> float:
    scored = recommendation_recall(prediction, answers, ENTITY_PATH)
    predicted = set(scored['parsed_prediction'][:5])
    truth = set(scored['ground_truth_movies'])
    common = len(predicted & truth)
    precision = common / len(predicted) if predicted else 0.0
    recall = common / len(truth) if truth else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for method in METHODS:
        run_dir = latest_run(method)
        if run_dir is None:
            results[method] = {'status': 'pending'}
            continue
        icl_scores: list[float] = []
        recsys_scores: list[float] = []
        for row in load_predictions(run_dir / 'predictions.jsonl'):
            answers = row.get('answers', row['answer'])
            if str(row['category']).startswith('recsys'):
                recsys_scores.append(recommendation_f1_at_5(str(row['prediction']), answers))
            else:
                icl_scores.append(token_f1(parse_icl_prediction(str(row['prediction'])), answers))
        results[method] = {
            'status': 'complete',
            'run_dir': str(run_dir.relative_to(ROOT)),
            'icl_count': len(icl_scores),
            'icl_token_f1': sum(icl_scores) / len(icl_scores) if icl_scores else None,
            'recsys_count': len(recsys_scores),
            'recsys_f1_at_5': sum(recsys_scores) / len(recsys_scores) if recsys_scores else None,
        }
    output = {
        'protocol': 'memoryagentbench-final150-v1',
        'metric_status': 'supplementary_not_native_benchmark_metric',
        'icl_metric': 'normalized token F1 against the best valid answer',
        'recsys_metric': 'set-based F1 at five recommendations after native entity canonicalization',
        'methods': results,
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
