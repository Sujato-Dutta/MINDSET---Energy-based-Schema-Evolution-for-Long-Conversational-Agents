import numpy as np
import pytest
from mindset.methods.packet_energy import PacketEnergyMindsetMemory, greedy_packet
from mindset.methods.compact import CompactConfig
from test_compact import Encoder, Reranker, event


def test_greedy_certificate_sums_terms_and_does_not_claim_global_optimum():
    chosen, cert = greedy_packet([1, 2, 3], np.array([1., .99, .9]),
        np.array([[1., 1., 0.], [1., 1., 0.], [0., 0., 1.]]), [1, 1, 2], 3, .1, 2)
    assert chosen == [0, 2]
    assert cert['total'] == pytest.approx(cert['relevance_loss'] + cert['repeated_support_cost'] + cert['uncovered_entity_fraction'])
    assert cert['global_optimality_gap'] is None


def test_schema_diversity_has_a_source_level_counterfactual_effect():
    m = PacketEnergyMindsetMemory(CompactConfig(complexity_penalty=.1, switch_margin=.05), encoder=Encoder(), local_reranker=Reranker())
    m.ingest([event(1, 'I prefer tea at home.'), event(2, 'I prefer tea at home.'), event(3, 'I prefer coffee at work.'), event(4, 'I live in Paris.')])
    hits = m.select_scored('Describe Sam habits', [0, 1, 2, 3], [1., .95, .9, .05], 2)
    assert [h.memory_id for h in hits] == ['e1', 'e3']
    m.schema_energy = False
    assert [h.memory_id for h in m.select_scored('Describe Sam habits', [0, 1, 2, 3], [1., .95, .9, .05], 2)] == ['e1', 'e2']
    m.packet_energy = False
    assert [h.memory_id for h in m.select_scored('Describe Sam habits', [0, 1, 2, 3], [1., .95, .9, .05], 2)] == ['e1', 'e2']
    assert m.validate_invariants()['grounded_episode_count'] == 4


def test_energy_off_recovers_pure_relevance_and_unknown_entities_do_not_get_invented():
    m = PacketEnergyMindsetMemory(encoder=Encoder(), local_reranker=Reranker(), packet_energy=False)
    m.ingest([event(1, 'I prefer tea.'), event(2, 'I prefer coffee.')])
    assert [h.memory_id for h in m.select_scored('Alex', [0, 1], [0., 1.], 1)] == ['e2']
    with pytest.raises(ValueError): greedy_packet([0], [1.], [[float('nan')]], [0], 0, .1, 1)
