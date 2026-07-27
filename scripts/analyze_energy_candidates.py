from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any, Callable


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay saved MINDSET candidate rankings")
    parser.add_argument("run_dir")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in (Path(args.run_dir) / "retrieval.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    strategies: dict[str, Callable[[dict[str, Any]], float]] = {
        "saved_mmr_order": lambda hit: -float(hit["_position"]),
        "fused_relevance": lambda hit: float(hit["score"]),
        "dense": lambda hit: float(hit["metadata"]["dense_score"]),
        "bm25": lambda hit: float(hit["metadata"]["bm25_score"]),
        "direct_rank": lambda hit: reciprocal(hit["metadata"].get("direct_rank")),
        "schema_rank": lambda hit: reciprocal(hit["metadata"].get("schema_rank")),
        "continuity_rank": lambda hit: reciprocal(hit["metadata"].get("continuity_rank")),
    }
    reports = {
        name: evaluate(rows, scorer, args.top_k) for name, scorer in strategies.items()
    }
    best_grid: tuple[tuple[float, float, float, float], dict[str, float]] | None = None
    for dense_weight, schema_weight, direct_weight, continuity_weight in itertools.product(
        (0.0, 0.25, 0.5, 0.75, 1.0),
        (0.0, 0.1, 0.2, 0.4, 0.8),
        (0.0, 0.1, 0.2, 0.4),
        (0.0, 0.1, 0.2, 0.4, 0.8),
    ):
        lexical_weight = 1.0 - dense_weight

        def score(
            hit,
            dw=dense_weight,
            lw=lexical_weight,
            sw=schema_weight,
            rw=direct_weight,
            cw=continuity_weight,
        ):
            metadata = hit["metadata"]
            return (
                dw * float(metadata["dense_score"])
                + lw * float(metadata["bm25_score"])
                + sw * reciprocal(metadata.get("schema_rank"))
                + rw * reciprocal(metadata.get("direct_rank"))
                + cw * reciprocal(metadata.get("continuity_rank"))
            )

        report = evaluate(rows, score, args.top_k)
        candidate = ((dense_weight, schema_weight, direct_weight, continuity_weight), report)
        if best_grid is None or ranking_key(candidate) > ranking_key(best_grid):
            best_grid = candidate
    payload = {
        "question_count": len(rows),
        "top_k": args.top_k,
        "strategies": reports,
        "best_grid": {
            "dense_weight": best_grid[0][0],
            "bm25_weight": 1.0 - best_grid[0][0],
            "schema_reciprocal_rank_weight": best_grid[0][1],
            "direct_reciprocal_rank_weight": best_grid[0][2],
            "continuity_reciprocal_rank_weight": best_grid[0][3],
            "metrics": best_grid[1],
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def evaluate(rows, scorer, top_k):
    recalls, reciprocal_ranks, ndcgs = [], [], []
    for row in rows:
        candidates = [dict(hit, _position=index) for index, hit in enumerate(row["candidate_hits"])]
        ranked = sorted(candidates, key=lambda hit: (-scorer(hit), hit["memory_id"]))[:top_k]
        retrieved = [source.split(":", 1)[-1] for hit in ranked for source in hit["source_event_ids"]]
        gold = {
            value.strip()
            for group in row["gold_evidence_ids"]
            for value in group.split(";")
            if value.strip()
        }
        if not gold:
            continue
        hits = [int(identifier in gold) for identifier in retrieved]
        recalls.append(sum(hits) / len(gold))
        reciprocal_ranks.append(next((1.0 / (index + 1) for index, hit in enumerate(hits) if hit), 0.0))
        dcg = sum(hit / math.log2(index + 2) for index, hit in enumerate(hits))
        ideal = sum(1.0 / math.log2(index + 2) for index in range(min(len(gold), top_k)))
        ndcgs.append(dcg / ideal if ideal else 1.0)
    return {
        "answerable_recall": sum(recalls) / max(1, len(recalls)),
        "mrr": sum(reciprocal_ranks) / max(1, len(reciprocal_ranks)),
        "ndcg": sum(ndcgs) / max(1, len(ndcgs)),
    }


def reciprocal(rank: object) -> float:
    return 0.0 if rank in (None, 0) else 1.0 / float(rank)


def ranking_key(item):
    weights, report = item
    return (report["answerable_recall"], report["ndcg"], report["mrr"], tuple(-x for x in weights))


if __name__ == "__main__":
    main()
