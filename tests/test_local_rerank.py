from __future__ import annotations

import numpy as np

from mindset.retrieval.local_rerank import (
    LocalCandidate,
    RetrievalWeights,
    query_kind,
    select_candidates,
)


def candidate(
    identifier: str,
    cross: float,
    vector: tuple[float, float],
    *,
    entity: str,
    time: str,
    timestamp: float,
    status: str = "active",
) -> LocalCandidate:
    return LocalCandidate(
        episode_index=int(identifier[1:]),
        episode_id=identifier,
        text=identifier,
        vector=np.asarray(vector, dtype=np.float32),
        timestamp_value=timestamp,
        ordinal=int(identifier[1:]),
        entities=frozenset({entity}),
        time_bucket=time,
        statuses=(status,),
        components={
            "cross_encoder": cross,
            "dense": cross,
            "bm25": cross,
            "schema": cross,
            "entity": cross,
            "temporal": cross,
            "validity": cross,
        },
    )


def test_weights_sum_to_one() -> None:
    assert sum(RetrievalWeights().as_dict().values()) == 1.0


def test_coverage_rewards_new_entity_and_time() -> None:
    rows = [
        candidate("e1", 1.0, (1.0, 0.0), entity="sam", time="s1", timestamp=1),
        candidate("e2", 0.99, (1.0, 0.0), entity="sam", time="s1", timestamp=2),
        candidate("e3", 0.90, (0.0, 1.0), entity="lee", time="s2", timestamp=3),
    ]
    selected = select_candidates(
        "What did Sam and Lee do?",
        rows,
        mode="coverage",
        weights=RetrievalWeights(),
        top_k=2,
    )
    assert [item.episode_id for item in selected] == ["e1", "e3"]


def test_temporal_order_is_chronological_after_selection() -> None:
    rows = [
        candidate("e1", 0.8, (1.0, 0.0), entity="sam", time="s1", timestamp=3),
        candidate("e2", 0.9, (0.0, 1.0), entity="lee", time="s2", timestamp=1),
    ]
    selected = select_candidates(
        "What happened before and after the trip?",
        rows,
        mode="query_adaptive",
        weights=RetrievalWeights(),
        top_k=2,
    )
    assert [item.episode_id for item in selected] == ["e2", "e1"]
    assert query_kind("What is current?") == "current"
