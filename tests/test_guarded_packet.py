import pytest
from mindset.methods.guarded_packet import GuardedPacketMindsetMemory
from mindset.methods.compact import CompactConfig
from test_compact import Encoder, Reranker, event


def memory(margin):
    m = GuardedPacketMindsetMemory(CompactConfig(complexity_penalty=.1, switch_margin=margin),
                                   encoder=Encoder(), local_reranker=Reranker())
    m.ingest([event(1, 'I prefer tea at home.'), event(2, 'I prefer tea at home.'),
              event(3, 'I prefer coffee at work.'), event(4, 'I live in Paris.')])
    return m


def test_guard_rejects_excess_surrogate_regret_without_changing_memory():
    m = memory(.01)
    hits = m.select_scored('Describe Sam habits', [0, 1, 2, 3], [1., .95, .9, .05], 2)
    assert [h.memory_id for h in hits] == ['e1', 'e2']
    guard = m.last_readout_certificate['relevance_guard']
    assert not guard['proposal_accepted'] and guard['normalized_ce_regret'] > .01
    assert m.validate_invariants()['grounded_episode_count'] == 4


def test_guard_allows_low_regret_diversification_and_preserves_peak():
    m = memory(.1)
    hits = m.select_scored('Describe Sam habits', [0, 1, 2, 3], [1., .95, .9, .05], 2)
    assert [h.memory_id for h in hits] == ['e1', 'e3']
    guard = m.last_readout_certificate['relevance_guard']
    assert guard['proposal_accepted'] and guard['highest_scored_source_preserved']
    assert guard['normalized_ce_regret'] <= .1


def test_packet_off_reproduces_baseline_and_zero_budget_remains_empty():
    m = memory(.05); m.packet_energy = False
    assert [h.memory_id for h in m.select_scored('Sam', [0, 1, 2, 3], [1., .95, .9, .05], 2)] == ['e1', 'e2']
    assert m.select_scored('Sam', [0, 1, 2, 3], [1., .95, .9, .05], 0) == []
