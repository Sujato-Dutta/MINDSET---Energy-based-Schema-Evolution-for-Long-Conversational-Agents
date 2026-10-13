"""Surrogate-relevance safety guard for experimental schema-packet energy."""
from __future__ import annotations
import numpy as np
from mindset.methods.packet_energy import PacketEnergyMindsetMemory
from mindset.methods.aligned_readout import AlignedMindsetMemory


class GuardedPacketMindsetMemory(PacketEnergyMindsetMemory):
    retrieval_version = 'guarded-schema-packet-energy-v2-development'

    def select_scored(self, query, candidates, scores, top_k=8, question_date=''):
        proposal = super().select_scored(query, candidates, scores, top_k, question_date)
        proposal_certificate = self.last_readout_certificate
        if not self.packet_energy or not proposal:
            return proposal
        # Baseline respects exactly the same state-accessibility constraints.
        eligible = AlignedMindsetMemory.select_scored(self, query, candidates, scores, len(candidates), question_date)
        baseline = eligible[:top_k]
        by_id = {h.memory_id: float(h.score) for h in eligible}
        low, high = min(by_id.values()), max(by_id.values())
        span = high - low
        relevance = {eid: (score - low) / span if span > 0 else 1. for eid, score in by_id.items()}
        deficit = max(0., sum(relevance[h.memory_id] for h in baseline) - sum(relevance[h.memory_id] for h in proposal))
        peak_preserved = baseline[0].memory_id in {h.memory_id for h in proposal}
        # A frozen scalar transfer, NOT a read-specific calibration or bound on
        # true answer error. It bounds only this normalized CE surrogate loss.
        margin = self.config.switch_margin
        accepted = peak_preserved and deficit <= margin + 1e-12
        result = proposal if accepted else baseline
        self.last_readout_certificate = {**proposal_certificate,
            'selected_ids': [h.memory_id for h in result],
            'proposal_packet_energy': proposal_certificate.get('packet_energy'),
            'packet_energy': proposal_certificate.get('packet_energy') if accepted else None,
            'relevance_guard': {'proposal_accepted': accepted, 'normalized_ce_regret': deficit,
                'maximum_surrogate_regret': margin, 'highest_scored_source_preserved': peak_preserved,
                'coefficient_policy': 'frozen_write_margin_transfer_not_benchmark_tuned',
                'guarantee_scope': 'surrogate CE regret only; no native accuracy or global-optimality guarantee'},
            'selection_policy': 'guarded_packet_energy' if accepted else 'state_compatible_relevance_fallback'}
        return result

    def telemetry(self):
        return {**super().telemetry(), 'relevance_guard': 'retain_top1_and_normalized_ce_regret_le_frozen_margin'}
