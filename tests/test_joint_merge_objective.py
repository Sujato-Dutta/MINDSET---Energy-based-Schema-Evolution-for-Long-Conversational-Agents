"""Algebraic overlap check, not a claim the entire controller is k-means.

For an ordinary merge, D+H equals incremental total spherical distortion.
Exception, version, contextual branch, feasibility and retrieval differ.
"""
from dataclasses import replace
import numpy as np
import pytest
from mindset.memory.energy import unit_vector
from mindset.methods.joint import JointMindsetMemory
from mindset.methods.compact import CompactConfig
from test_compact import Encoder, event


@pytest.mark.parametrize("size", [1, 2, 8, 64, 512])
def test_implemented_merge_is_spherical_distortion_increment(size):
    memory = JointMindsetMemory(CompactConfig(complexity_penalty=.1, switch_margin=.05), encoder=Encoder())
    memory.ingest([event(1, "An unstructured recollection.")])
    original = memory.episodes["e1"]; schema = next(iter(memory.schemas.values()))
    rng = np.random.default_rng(819 + size)
    for _ in range(20):
        vectors = rng.normal(size=(size, 9)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        total = vectors.sum(axis=0)
        incoming_vector = unit_vector(rng.normal(size=9))
        memory._sums[schema.schema_id] = total; schema.centroid = unit_vector(total)
        episode = replace(original, vector=incoming_vector, episode_id="incoming", timestamp="2025-01-02")
        merge = next(p for p in memory._plans(schema, episode) if p["mutation"] == "merge")
        assert merge["terms"].contradiction == merge["terms"].inconsistency == 0.
        before = .5 * (size - np.linalg.norm(total))
        after = .5 * (size + 1 - np.linalg.norm(total + incoming_vector))
        assert merge["terms"].total == pytest.approx(after-before, abs=2e-5)


def test_zero_resultant_has_same_increment_not_division_by_zero():
    memory = JointMindsetMemory(CompactConfig(), encoder=Encoder())
    memory.ingest([event(1, "An unstructured recollection.")])
    episode = memory.episodes["e1"]; schema = next(iter(memory.schemas.values()))
    total = episode.vector.copy(); memory._sums[schema.schema_id] = total
    incoming = replace(episode, vector=-total, episode_id="incoming", timestamp="2025-01-02")
    merge = next(p for p in memory._plans(schema, incoming) if p["mutation"] == "merge")
    assert merge["terms"].total == pytest.approx((1 + np.linalg.norm(total))/2, abs=2e-5)
