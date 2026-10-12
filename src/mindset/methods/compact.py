"""Compact candidate controller: two decision scalars, no fitted feature mixtures.

This is an empirical method candidate, not a claim of new physical theory.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from collections import defaultdict
from time import perf_counter
import json
import numpy as np
from mindset.methods.mindset import MindsetMemory, MindsetConfig, _render_episode_compact
from mindset.memory.energy import (TransitionSignals, EnergyTerms, unit_vector, cosine,
                                   contradiction_probability, timestamp_value)
from mindset.methods.base import MemoryHit
from mindset.retrieval.hybrid import ranked_indices, reciprocal_rank_fusion

@dataclass(frozen=True)
class CompactConfig:
    complexity_penalty: float = .25
    switch_margin: float = .10
    ablation: str = "full"
    nearest_schema_k: int = 6
    direct_pool_k: int = 24
    schema_pool_k: int = 24
    retrieval_candidate_k: int = 48
    rerank_batch_size: int = 8

    def __post_init__(self):
        if not 0 <= self.complexity_penalty <= 1 or not 0 <= self.switch_margin <= 1:
            raise ValueError("Compact decision scalars must be in [0,1]")
        if self.ablation not in {"full", "no_hysteresis", "no_complexity", "no_context", "no_provenance"}:
            raise ValueError("Unknown compact ablation")
        if min(self.nearest_schema_k,self.direct_pool_k,self.schema_pool_k,
               self.retrieval_candidate_k,self.rerank_batch_size) <= 0:
            raise ValueError("Operational budgets must be positive")

def select_transition(energies, allowed, margin):
    """Minimize feasible energy with a supersession-only switching cost.

    Complexity prices every additional schema; margin prices only changing the
    currently valid state. They are not one interchangeable threshold. Exact
    ties preserve the existing state, including at the switching boundary.
    """
    if "reinforce" not in allowed or margin < 0:
        raise ValueError("A feasible preserve action and nonnegative margin are required")
    priority={"reinforce":0,"supersede":1,"split":2,"new":3}
    def choose(cost):
        return min(allowed,key=lambda a:(energies[a].total+cost*(a=="supersede"),priority[a]))
    raw=choose(0.)
    selected=choose(margin)
    return selected, bool(margin and (raw=="supersede" or selected=="supersede"))

class CompactMindsetMemory(MindsetMemory):
    method_version="v2"
    retrieval_version="bounded-provenance-rrf-ce-v2"

    def __init__(self,config=None,**kwargs):
        super().__init__(MindsetConfig(),**kwargs)
        self.config=config or CompactConfig()
        self._sums={}
        self._latest={}
        self._by_topic=defaultdict(list)
        self._pending=defaultdict(lambda:defaultdict(set))
        self.last_features=None
        self._retrieval_profile=defaultdict(float)
        self._feasibility_rejections=0
        self._decision_certificates={}

    def _create_schema(self,episode,**kwargs):
        schema=super()._create_schema(episode,**kwargs)
        self._sums[schema.schema_id]=episode.vector.copy()
        self._latest[schema.schema_id]=timestamp_value(episode.timestamp)
        self._by_topic[schema.topic].append(schema.schema_id)
        return schema

    def _signals(self,schema,episode):
        old=set(schema.context_counts); new=set(episode.features.contexts)
        context=float(old==new) if bool(old)==bool(new) else .5
        role=float(set(schema.role_counts)==set(episode.features.roles))
        contradiction=contradiction_probability(schema,episode.features)
        value_key=tuple(sorted(episode.features.attributes))
        recurrent=bool(self._pending[schema.schema_id][value_key]-{episode.session_id})
        incoming=timestamp_value(episode.timestamp)
        later=bool(incoming and self._latest[schema.schema_id] and incoming>=self._latest[schema.schema_id])
        temporal=float(contradiction>0 and context==1 and later
                       and (episode.features.explicit_change or recurrent))
        contextual=float(bool(old and new and old!=new))
        if self.config.ablation=="no_context": contextual=0.
        entities=set(schema.entity_counts); incoming_entities=set(episode.features.entities)
        entity=len(entities&incoming_entities)/max(1,len(entities|incoming_entities))
        # Both vectors are normalized once at ingestion, not for every pair.
        dense=max(0.,float(np.clip(np.dot(schema.centroid,episode.vector),-1,1)))
        return TransitionSignals(dense,entity,role,context,1.,dense,contradiction,
                                 temporal,contextual,float(recurrent),float(episode.features.explicit_change))

    def _energies(self,schema,episode,signals):
        old_sum=self._sums[schema.schema_id]; n=len(schema.support_episode_ids)
        center=unit_vector(old_sum+episode.vector)
        distortion=float(np.clip((1-np.dot(episode.vector,center))/2,0,1))
        # Exact increase in old support's average cosine loss, not a past-QA-loss claim.
        damage=float(np.clip((np.linalg.norm(old_sum)-np.dot(old_sum,center))/(2*n),0,1))
        penalty=0. if self.config.ablation=="no_complexity" else self.config.complexity_penalty
        loss={
            "reinforce":(distortion,signals.contradiction_probability,damage,0.,signals.contextual_potential),
            "supersede":(0.,0.,0.,1.,signals.contextual_potential),
            "split":(0.,0.,0.,1.,1-signals.contextual_potential),
            "new":(0.,0.,0.,1.,0.)}
        return {a:EnergyTerms(d,c,h,g,s,d+c+h+s+penalty*g) for a,(d,c,h,g,s) in loss.items()}

    def _ingest_episode(self,episode):
        topic=f"{episode.speaker.casefold()}::{episode.features.primary_slot}"
        pool=[self.schemas[i] for i in self._by_topic[topic] if self.schemas[i].status=="active"]
        if not pool:
            schema=self._create_schema(episode,action="new")
            self.last_features=None
            self._record_transition(episode,"new",None,(schema.schema_id,),{},None)
            return
        # Exact owner/slot blocking replaces the weighted compatibility heuristic.
        matrix=np.asarray([s.centroid for s in pool],dtype=np.float32)
        scores=np.einsum("ij,j->i",matrix,episode.vector)
        ranked=sorted(range(len(pool)),key=lambda i:(-float(scores[i]),pool[i].schema_id))
        evaluated=[(pool[i],self._signals(pool[i],episode)) for i in ranked[:self.config.nearest_schema_k]]
        target,signals=max(evaluated,key=lambda row:(row[1].context_match,row[1].dense_similarity,row[0].schema_id))
        self.last_features=signals
        energies=self._energies(target,episode,signals)
        allowed={"reinforce","new"}
        same_structured_slot=episode.features.primary_slot!="discourse"
        if same_structured_slot: allowed.discard("new")
        if signals.temporal_potential: allowed.add("supersede")
        if signals.contextual_potential: allowed.add("split")
        margin=0. if self.config.ablation=="no_hysteresis" else self.config.switch_margin
        if self.transition_controller is None:
            action,hysteresis=select_transition(energies,allowed,margin)
        elif hasattr(self.transition_controller,"choose"):
            action,hysteresis=self.transition_controller.choose(signals,self.config,allowed),False
        else:
            action,hysteresis=self.transition_controller(signals,self.config)
            if action not in allowed:
                action="reinforce"; self._feasibility_rejections+=1
        exception=False
        if action=="reinforce":
            if signals.contradiction_probability>0 or signals.contextual_potential>0:
                target.add_exception(episode); exception=True
                self._pending[target.schema_id][tuple(sorted(episode.features.attributes))].add(episode.session_id)
            else:
                target.support_episode_ids.append(episode.episode_id)
                target._add_features(episode.features)
                self._sums[target.schema_id]+=episode.vector
                total=self._sums[target.schema_id]; n=len(target.support_episode_ids)
                target.centroid=unit_vector(total)
                target.mean_distortion=float(np.clip((1-np.linalg.norm(total)/n)/2,0,1))
                self._latest[target.schema_id]=max(self._latest[target.schema_id],timestamp_value(episode.timestamp))
            resulting=(target.schema_id,)
        elif action=="supersede":
            target.status="historical"; target.valid_until=episode.timestamp
            successor=self._create_schema(episode,action=action,predecessor_schema_id=target.schema_id)
            target.successor_schema_ids.append(successor.schema_id)
            resulting=(target.schema_id,successor.schema_id)
        else:
            created=self._create_schema(episode,action=action)
            resulting=(target.schema_id,created.schema_id)
        self._record_transition(episode,action,target.schema_id,resulting,energies,signals,
                                hysteresis_applied=hysteresis,stored_as_exception=exception)
        adjusted={a:energies[a].total+margin*(a=="supersede") for a in allowed}
        optimum=min(adjusted.values())
        self._decision_certificates[episode.episode_id]={
            "feasible_actions":sorted(allowed),"switching_cost":margin,
            "adjusted_energies":adjusted,"selected_adjusted_energy":adjusted[action],
            "optimality_gap":max(0.,adjusted[action]-optimum),
            "policy":"energy" if self.transition_controller is None else "comparison_controller"}

    def _rebuild_retrieval_index(self):
        super()._rebuild_retrieval_index()
        self._schema_ids=sorted(self.schemas)
        self._schema_matrix=np.asarray([self.schemas[i].centroid for i in self._schema_ids],dtype=np.float32)
        self._sufficient_sums=np.asarray([self._sums[i] for i in self._schema_ids],dtype=np.float32)
        self._documents=[_render_episode_compact(self.episodes[i]) for i in self._episode_order]
        # True extractive medoids, not an incoming-versus-previous-medoid approximation.
        for schema in self.schemas.values():
            ids=sorted(schema.support_episode_ids)
            scores=np.asarray([self.episodes[i].vector for i in ids])@schema.centroid
            schema.medoid_episode_id=ids[int(np.argmax(scores))]

    def retrieve(self,query,top_k=10):
        if top_k<=0 or not self._episode_order: return []
        started=perf_counter(); q=unit_vector(self._encode([query])[0])
        dense=self._episode_vectors@q; lexical=self._bm25.scores(query)
        budget=self.config.direct_pool_k
        rankings=[ranked_indices(dense,budget),ranked_indices(lexical,budget)]
        fused=reciprocal_rank_fusion(rankings)  # Standard equal-channel RRF, fixed constant 60.
        direct=sorted(fused,key=lambda i:(-fused[i],i))[:budget]
        candidates=list(direct); memberships=defaultdict(set)
        if self.config.ablation!="no_provenance":
            schema_scores=self._schema_matrix@q
            selected=ranked_indices(schema_scores,self.config.nearest_schema_k)
            expanded=set()
            for k in selected:
                sid=self._schema_ids[k]
                for eid in self._schema_provenance[sid]:
                    i=self._index_by_id[eid]; expanded.add(i); memberships[i].add(sid)
            expanded=sorted(expanded,key=lambda i:(-float(dense[i]),i))[:self.config.schema_pool_k]
            candidates.extend(expanded)
        candidates=list(dict.fromkeys(candidates))[:self.config.retrieval_candidate_k]
        built=perf_counter()
        if self.local_reranker is None:
            raise RuntimeError("Compact paper retrieval requires the pinned local cross-encoder")
        scores=np.asarray(self.local_reranker.predict([(query,self._documents[i]) for i in candidates],
                        batch_size=self.config.rerank_batch_size,show_progress_bar=False),dtype=np.float32)
        if len(scores)!=len(candidates) or not np.isfinite(scores).all():
            raise ValueError("Malformed local reranker scores")
        order=sorted(range(len(candidates)),key=lambda j:(-float(scores[j]),self._episode_order[candidates[j]]))[:top_k]
        hits=[MemoryHit(self._episode_order[candidates[j]],self._documents[candidates[j]],float(scores[j]),
                       (self._episode_order[candidates[j]],),"compact_provenance_episode",
                       {"schema_ids":sorted(memberships[candidates[j]]),"retrieval_version":self.retrieval_version})
              for j in order]
        elapsed=perf_counter()-started
        self._query_seconds+=elapsed; self._query_count+=1
        self._retrieval_profile["candidate_build_seconds"]+=built-started
        self._retrieval_profile["cross_encoder_seconds"]+=perf_counter()-built
        self._retrieval_profile["query_episode_pairs"]+=len(candidates)
        return hits

    def snapshot(self,path):
        p=super().snapshot(path)
        payload=json.loads(p.read_text())
        payload["sufficient_statistics"]={"schema_order":self._schema_ids,
            "vector_sum_array":"_sufficient_sums","array_file":p.with_suffix(".npz").name,
            "latest_timestamps":self._latest,
            "pending_confirmations":{sid:[{"attributes":list(key),"sessions":sorted(sessions)}
                for key,sessions in values.items() if sessions] for sid,values in self._pending.items()}}
        for record in payload["transitions"]:
            record["decision_certificate"]=self._decision_certificates.get(record["episode_id"])
        np.savez_compressed(p.with_suffix(".npz"),episode_vectors=self._episode_vectors,
                           _sufficient_sums=self._sufficient_sums)
        p.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        return p

    def telemetry(self):
        return {**super().telemetry(),"decision_scalar_count":2,
                "resolved_config":asdict(self.config),"retrieval_profile":dict(self._retrieval_profile),
                "feasibility_rejections":self._feasibility_rejections,
                "coefficient_policy":"two_independent_development_calibrated_scalars; no_feature_weight_mixtures",
                "max_energy_optimality_gap":max((r["optimality_gap"] for r in self._decision_certificates.values()
                    if r["policy"]=="energy"),default=0.),
                "historical_damage_definition":"old_support_cosine_loss_increase",
                "claim_status":"candidate_not_yet_benchmark_validated"}
