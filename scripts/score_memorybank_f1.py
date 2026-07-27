"""Score the frozen faithful MemoryBank MAB-150 run with supplementary F1."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindset.evaluation.memoryagentbench import parse_icl_prediction, recommendation_recall
from mindset.evaluation.metrics import token_f1

RUN = ROOT / "outputs/runs/memoryagentbench_memory_bank_gpt_oss_120b_primary_20260718T162102Z"
ENTITY_PATH = ROOT / "data/memory_agent_bench/entity2id.json"


def recommendation_f1(prediction: str, answers: list[str]) -> float:
    scored = recommendation_recall(prediction, answers, ENTITY_PATH)
    predicted = set(scored["parsed_prediction"][:5])
    truth = set(scored["ground_truth_movies"])
    common = len(predicted & truth)
    precision = common / len(predicted) if predicted else 0.0
    recall = common / len(truth) if truth else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


rows = [
    json.loads(line)
    for line in (RUN / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
icl = []
recsys = []
for row in rows:
    answers = row.get("answers", row["answer"])
    if str(row["category"]).startswith("recsys"):
        recsys.append(recommendation_f1(str(row["prediction"]), answers))
    else:
        icl.append(token_f1(parse_icl_prediction(str(row["prediction"])), answers))

print(json.dumps({
    "run_dir": str(RUN.relative_to(ROOT)),
    "icl_count": len(icl),
    "icl_token_f1": sum(icl) / len(icl),
    "recsys_count": len(recsys),
    "recsys_f1_at_5": sum(recsys) / len(recsys),
}, indent=2))
