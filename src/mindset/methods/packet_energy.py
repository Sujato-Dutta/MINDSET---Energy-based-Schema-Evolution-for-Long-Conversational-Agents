"""Experimental query-packet energy over energy-constructed schema provenance.

No fitted read mixture. The existing independent complexity regularizer is
transferred to repeated-schema support. This transfer is an empirical design
choice, not an independently proved/calibrated optimum or physical energy.
"""
from __future__ import annotations
import re
import numpy as np
from mindset.methods.aligned_readout import AlignedMindsetMemory


def greedy_packet(indices, relevance, redundancy, entity_bits, target_bits, penalty, top_k):
    """Greedy marginal minimization; NEVER claim a global optimum certificate.

E(A) = sum(1-u_e) + lambda sum_{i<j} R_ij + uncovered named participants / m.
R_ij = max(0, cos(v_i,v_j)) when sources share a state schema, zero otherwise.
The query-constant uncovered term is zero when no participant is explicit.
Maximal feasible packet cardinality is bounded by top_k, matching the adapter.
"""
    n = len(indices); relevance = np.asarray(relevance, dtype=float); redundancy = np.asarray(redundancy, dtype=float)
    if relevance.shape != (n,) or redundancy.shape != (n, n) or len(entity_bits) != n:
        raise ValueError('Malformed packet energy arrays')
    if not np.isfinite(relevance).all() or not np.isfinite(redundancy).all() or penalty < 0 or top_k < 0:
        raise ValueError('Invalid packet energy values')
    chosen, steps, covered = [], [], 0; participants = target_bits.bit_count()
    for _ in range(min(top_k, n)):
        options = []
        for j in range(n):
            if j in chosen: continue
            gain = ((entity_bits[j] & target_bits) & ~covered).bit_count() / participants if participants else 0.
            repeated = float(sum(redundancy[j, k] for k in chosen))
            marginal = 1. - float(relevance[j]) + penalty * repeated - gain
            options.append((marginal, -float(relevance[j]), indices[j], j, gain, repeated))
        marginal, _, _, j, gain, repeated = min(options)
        chosen.append(j); covered |= entity_bits[j] & target_bits
        steps.append({'source_index': indices[j], 'marginal_energy': marginal,
                      'normalized_relevance': float(relevance[j]), 'new_entity_fraction': gain,
                      'repeated_schema_similarity': repeated})
    relevance_loss = float(sum(1. - relevance[j] for j in chosen))
    repeated_loss = float(penalty * sum(redundancy[j, k] for p, j in enumerate(chosen) for k in chosen[:p]))
    missing = (participants - covered.bit_count()) / participants if participants else 0.
    return chosen, {'relevance_loss': relevance_loss, 'repeated_support_cost': repeated_loss,
        'uncovered_entity_fraction': missing, 'total': relevance_loss + repeated_loss + missing,
        'steps': steps, 'solver': 'greedy_marginal_approximation', 'global_optimality_gap': None}


class PacketEnergyMindsetMemory(AlignedMindsetMemory):
    retrieval_version = 'schema-packet-energy-v2-development'

    def __init__(self, *args, packet_energy=True, entity_energy=True, schema_energy=True, **kwargs):
        self.packet_energy = packet_energy; self.entity_energy = entity_energy; self.schema_energy = schema_energy
        super().__init__(*args, **kwargs)

    def _rebuild_retrieval_index(self):
        super()._rebuild_retrieval_index()
        # Explicit names only, not inferred user/assistant identities. Historical
        # and contextual versions stay distinct provenance groups.
        names = [s for s in self._speakers if s.casefold() not in {'user', 'assistant', 'system', 'tool', 'unknown'}]
        self._participant_patterns = [(1 << i, re.compile(r'(?<!\w)' + re.escape(s) + r'(?!\w)', re.I)) for i, s in enumerate(names)]
        self._source_entity_bits = []
        self._source_schema_groups = []
        for eid in self._episode_order:
            e = self.episodes[eid]; text = e.speaker + ': ' + e.text
            self._source_entity_bits.append(sum(bit for bit, pattern in self._participant_patterns if pattern.search(text)))
            self._source_schema_groups.append(frozenset(state[0] for state, _ in self._compiled_states[eid]))

    def select_scored(self, query, candidates, scores, top_k=8, question_date=''):
        eligible_hits = super().select_scored(query, candidates, scores, len(candidates), question_date)
        certificate = self.last_readout_certificate
        if not self.packet_energy or top_k <= 0:
            hits = eligible_hits[:max(0, top_k)]
            certificate.update(packet_energy_enabled=False, selected_ids=[h.memory_id for h in hits])
            return hits
        ids = [self._index_by_id[h.memory_id] for h in eligible_hits]
        raw = np.asarray([h.score for h in eligible_hits], dtype=float)
        if not len(ids):
            certificate.update(packet_energy_enabled=True, selected_ids=[], packet_energy=None)
            return []
        span = float(raw.max() - raw.min())
        relevance = (raw - raw.min()) / span if span > 0 else np.ones_like(raw)
        vectors = self._episode_vectors[ids]
        similarity = np.clip(vectors @ vectors.T, 0., 1.)
        groups = [self._source_schema_groups[i] for i in ids]
        same = np.asarray([[bool(a & b) for b in groups] for a in groups], dtype=float)
        redundancy = similarity * same if self.schema_energy else np.zeros_like(similarity)
        target = sum(bit for bit, pattern in self._participant_patterns if pattern.search(query)) if self.entity_energy else 0
        penalty = self.config.complexity_penalty
        selected, energy = greedy_packet(ids, relevance, redundancy,
            [self._source_entity_bits[i] for i in ids], target, penalty, top_k)
        # Return selected sources in the same CE order as the common factual
        # adapter. The optimizer chooses the set; no special answer prompt.
        selected = sorted(selected, key=lambda j: (-float(raw[j]), self._episode_order[ids[j]]))
        hits = [eligible_hits[j] for j in selected]
        certificate.update(packet_energy_enabled=True, packet_energy=energy, selected_ids=[h.memory_id for h in hits],
            entity_energy=self.entity_energy, schema_energy=self.schema_energy,
            regularizer=penalty, coefficient_policy='frozen_write_regularizer_transferred_not_benchmark_fitted',
            readout_source='state-derived_schema_groups_and_raw_episode_vectors')
        return hits

    def telemetry(self):
        return {**super().telemetry(), 'packet_energy_enabled': self.packet_energy,
                'entity_energy': self.entity_energy, 'schema_energy': self.schema_energy,
                'packet_solver': 'greedy_marginal_not_global_minimum', 'new_fitted_read_scalars': 0}
