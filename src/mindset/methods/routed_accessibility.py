"""Unmasked state-routing access path; fixed equal-channel RRF, no fitted mix."""
from collections import defaultdict
from mindset.methods.accessibility import AccessibilityMemory
from mindset.memory.energy import unit_vector
from mindset.retrieval.hybrid import ranked_indices, reciprocal_rank_fusion


class RoutedAccessibilityMemory(AccessibilityMemory):
    retrieval_version = 'accessibility-root-routed-rrf-v2-pilot'

    def candidate_pool(self, query):
        q = unit_vector(self._encode([query])[0])
        dense = ranked_indices(self._episode_vectors @ q)
        lexical = ranked_indices(self._bm25.scores(query))
        direct = reciprocal_rank_fusion([dense, lexical])
        flat = sorted(direct, key=lambda i: (-direct[i], i))
        memberships = defaultdict(set)
        rankings = [dense, lexical]
        if self.config.ablation != 'no_provenance':
            roots = [self._schema_ids[i] for i in ranked_indices(
                self._schema_matrix @ q, self.config.nearest_schema_k)]
            seen = set(roots)
            for sid in roots:
                schema = self.schemas[sid]
                for link in (schema.predecessor_schema_id, *schema.successor_schema_ids,
                             *sorted(self._split_links[sid])):
                    if link and link not in seen:
                        roots.append(link); seen.add(link)
            for sid in roots:
                for eid in self._schema_provenance[sid]:
                    memberships[self._index_by_id[eid]].add(sid)
            provenance = [i for i in flat if memberships[i]]
            if provenance:
                rankings.append(provenance)
        fused = reciprocal_rank_fusion(rankings)
        candidates = sorted(fused, key=lambda i: (-fused[i], i))[:self.config.retrieval_candidate_k]
        self.last_candidate_pool = tuple(candidates)
        return candidates, memberships

    def telemetry(self):
        return {**super().telemetry(),
            'candidate_access': 'equal RRF dense + BM25 + root-closure provenance; no protected direct prefix',
            'new_learned_mixtures': 0,
            'novelty_status': 'RRF is standard; only a matched controller ablation can establish energy usefulness'}
