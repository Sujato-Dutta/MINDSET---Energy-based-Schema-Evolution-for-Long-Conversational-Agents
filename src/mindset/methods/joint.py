"""Isolated MINDSET v2 prototype: joint, actual-post-state energy decisions.

Not the paper-frozen implementation. Regex relations are conservative typed
observations, not a claim of complete semantic parsing. Unknown context remains
unknown. All evidence remains immutable; no LLM is used to construct memory.
"""
from __future__ import annotations
from dataclasses import replace
from collections import defaultdict
from datetime import datetime, timezone
from time import perf_counter
import math
import re
import numpy as np
from dateutil import parser
from mindset.methods.compact import CompactMindsetMemory
from mindset.methods.mindset import _render_episode_compact
from mindset.memory.energy import EnergyTerms, unit_vector, extract_episode_features
from mindset.methods.base import MemoryHit
from mindset.retrieval.hybrid import ranked_indices, reciprocal_rank_fusion


def utc_time(value):
    """Stable missing-date default; never use the machine's date or time zone."""
    try:
        dt=parser.parse(value, fuzzy=True, default=datetime(1970,1,1))
        return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp()
    except (ValueError, TypeError, OverflowError, OSError):
        return 0.


def typed_features(speaker, text):
    features=extract_episode_features(speaker,text)
    # 'I am thrilled' and 'I am a pottery fan' are not values of one predicate.
    attributes=[a for a in features.attributes if a[0]!="attribute"]
    for relation,pattern in (
        ("occupation",r"(?i)\bI(?: am|'m| work as)\s+(?:a|an)\s+(teacher|doctor|nurse|engineer|designer|student|chef|lawyer|artist)\b"),
        ("location",r"(?i)\bI(?: now| currently)?\s+(?:reside|am based)\s+in\s+([^,.!?;]+)"),
        ("employer",r"(?i)\bI(?: have)?\s+(?:joined|work for)\s+([^,.!?;]+)"),
    ):
        for match in re.finditer(pattern,text):
            value=" ".join(match.group(1).casefold().split())
            if value and (relation,value) not in attributes: attributes.append((relation,value))
    return replace(features,attributes=tuple(attributes))


class JointMindsetMemory(CompactMindsetMemory):
    """Joint minimum over all feasible (schema, actual mutation) candidates.

    A deferred observation is still logged as reinforce, but its certificate
    identifies an exception mutation explicitly. The creation price is a
    standard logarithmic description-length regularizer, not new physics.
    """
    retrieval_version="joint-accessibility-backfill-v2-prototype"

    def __init__(self,*args,joint_targets=True,**kwargs):
        super().__init__(*args,**kwargs)
        if self.transition_controller is not None: raise ValueError("Prototype does not silently ignore comparison controllers")
        self.joint_targets=joint_targets
        self._split_links=defaultdict(set)
        self.last_candidate_pool=()
        self.last_joint_candidates=()

    def _create_schema(self,episode,**kwargs):
        schema=super()._create_schema(episode,**kwargs)
        self._latest[schema.schema_id]=utc_time(episode.timestamp)
        return schema

    def _signals(self,schema,episode):
        s=super()._signals(schema,episode)
        old=set(schema.context_counts); new=set(episode.features.contexts)
        match=float(old==new) if old and new else .5
        different=bool(old and new and old!=new)
        later=bool(utc_time(episode.timestamp) and self._latest[schema.schema_id]
                   and utc_time(episode.timestamp)>=self._latest[schema.schema_id])
        # Unknown contexts do not become positive evidence of sameness. Explicit
        # self-reports or independent-session confirmations can resolve an update.
        temporal=float(s.contradiction_probability>0 and not different and later
                       and (episode.features.explicit_change or s.recurrence))
        return replace(s,context_match=match,temporal_potential=temporal,
                       contextual_potential=0. if self.config.ablation=="no_context" else float(different))

    def _plans(self,schema,episode):
        s=self._signals(schema,episode)
        total=self._sums[schema.schema_id]; center=unit_vector(total+episode.vector)
        distortion=float(np.clip((1-episode.vector@center)/2,0,1))
        # SUM of old support loss increases: exactly the state being evaluated.
        damage=max(0.,float((np.linalg.norm(total)-total@center)/2))
        complexity=0. if self.config.ablation=="no_complexity" else self.config.complexity_penalty*math.log1p(len(self.episodes))
        margin=0. if self.config.ablation=="no_hysteresis" else self.config.switch_margin
        contradiction=s.contradiction_probability; contextual=s.contextual_potential
        plans=[]
        def add(action,mutation,d,c,h,g,b,switch=0.):
            energy=EnergyTerms(d,c,h,g,b,d+c+h+complexity*g+b)
            plans.append({"schema":schema,"signals":s,"action":action,"mutation":mutation,
                          "terms":energy,"switch":switch,"cost":energy.total+switch})
        if not contradiction and not contextual:
            add("reinforce","merge",distortion,0.,damage,0.,0.)
        else:
            # No centroid mutation: historical damage must be ZERO, not the
            # counterfactual damage of an unexecuted merge.
            routing=float(np.clip((1-episode.vector@schema.centroid)/2,0,1))
            add("reinforce","exception",routing,contradiction,0.,0.,contextual)
        if s.temporal_potential: add("supersede","version",0.,0.,0.,1.,contextual,margin)
        if contextual: add("split","branch",0.,0.,0.,1.,0.)
        if episode.features.primary_slot=="discourse": add("new","create",0.,0.,0.,1.,0.)
        return plans

    def _ingest_episode(self,episode):
        episode=replace(episode,features=typed_features(episode.speaker,episode.text))
        self.episodes[episode.episode_id]=episode
        topic=f"{episode.speaker.casefold()}::{episode.features.primary_slot}"
        pool=[self.schemas[i] for i in self._by_topic[topic] if self.schemas[i].status=="active"]
        if not pool:
            schema=self._create_schema(episode,action="new")
            self._record_transition(episode,"new",None,(schema.schema_id,),{},None)
            return
        matrix=np.asarray([s.centroid for s in pool],dtype=np.float32)
        dense=matrix@episode.vector
        nearest=sorted(range(len(pool)),key=lambda i:(-float(dense[i]),pool[i].schema_id))[:self.config.nearest_schema_k]
        plans=[p for i in nearest for p in self._plans(pool[i],episode)]
        priority={"reinforce":0,"supersede":1,"split":2,"new":3}
        # No compatibility argmax preselection. The target is part of the minimizer.
        feasible=plans
        if not self.joint_targets:
            sid=max((self.schemas[p["schema"].schema_id] for p in plans),
                    key=lambda s:(self._signals(s,episode).context_match,float(s.centroid@episode.vector),s.schema_id)).schema_id
            feasible=[p for p in plans if p["schema"].schema_id==sid]
        chosen=min(feasible,key=lambda p:(p["cost"],priority[p["action"]],p["schema"].schema_id))
        target=chosen["schema"]; signals=chosen["signals"]; action=chosen["action"]
        exception=chosen["mutation"]=="exception"
        self.last_features=signals
        if action=="reinforce":
            if exception:
                target.add_exception(episode)
                self._pending[target.schema_id][tuple(sorted(episode.features.attributes))].add(episode.session_id)
            else:
                target.support_episode_ids.append(episode.episode_id); target._add_features(episode.features)
                self._sums[target.schema_id]+=episode.vector
                total=self._sums[target.schema_id]; target.centroid=unit_vector(total)
                target.mean_distortion=float(np.clip((1-np.linalg.norm(total)/len(target.support_episode_ids))/2,0,1))
                self._latest[target.schema_id]=max(self._latest[target.schema_id],utc_time(episode.timestamp))
            resulting=(target.schema_id,)
        elif action=="supersede":
            target.status="historical"; target.valid_until=episode.timestamp
            successor=self._create_schema(episode,action=action,predecessor_schema_id=target.schema_id)
            target.successor_schema_ids.append(successor.schema_id)
            resulting=(target.schema_id,successor.schema_id)
        else:
            created=self._create_schema(episode,action=action)
            if action=="split":
                self._split_links[target.schema_id].add(created.schema_id)
                self._split_links[created.schema_id].add(target.schema_id)
            resulting=(target.schema_id,created.schema_id)
        energies={p["action"]:p["terms"] for p in plans if p["schema"].schema_id==target.schema_id}
        raw=min(plans,key=lambda p:(p["terms"].total,priority[p["action"]],p["schema"].schema_id))
        self._record_transition(episode,action,target.schema_id,resulting,energies,signals,
                                hysteresis_applied=raw is not chosen,stored_as_exception=exception)
        audit=[{"schema_id":p["schema"].schema_id,"action":p["action"],"mutation":p["mutation"],
                "terms":p["terms"].as_dict(),"adjusted_energy":p["cost"]} for p in plans]
        self.last_joint_candidates=tuple(audit)
        self._decision_certificates[episode.episode_id]={"feasible_candidates":audit,
            "selected_schema_id":target.schema_id,"selected_mutation":chosen["mutation"],
            "selected_adjusted_energy":chosen["cost"],"optimality_gap":max(0.,chosen["cost"]-min(p["cost"] for p in plans)),
            "policy":"joint_energy" if self.joint_targets else "preselected_target_control"}

    def candidate_pool(self,query):
        q=unit_vector(self._encode([query])[0])
        dense=self._episode_vectors@q; lexical=self._bm25.scores(query)
        fused=reciprocal_rank_fusion([ranked_indices(dense),ranked_indices(lexical)])
        flat=sorted(fused,key=lambda i:(-fused[i],i))
        # Backfilled strong episodic channel: provenance replaces only the tail,
        # never the initial 24 direct candidates. Ablation uses all 48 flat hits.
        budget=self.config.retrieval_candidate_k; candidates=flat[:budget]
        memberships=defaultdict(set)
        if self.config.ablation!="no_provenance":
            roots=[self._schema_ids[i] for i in ranked_indices(self._schema_matrix@q,self.config.nearest_schema_k)]
            versions=list(roots); seen=set(versions)
            for sid in versions:
                schema=self.schemas[sid]
                for link in (schema.predecessor_schema_id,*schema.successor_schema_ids,*sorted(self._split_links[sid])):
                    if link and link not in seen: versions.append(link); seen.add(link)
            for sid in versions:
                for eid in self._schema_provenance[sid]: memberships[self._index_by_id[eid]].add(sid)
            # Keep the global hybrid order; use provenance only for additional
            # candidates, rather than filling the packet with medoid summaries.
            extras=[i for i in flat if memberships[i] and i not in candidates]
            reserve=min(self.config.nearest_schema_k,len(extras),max(0,budget-self.config.direct_pool_k))
            if reserve: candidates=candidates[:-reserve]+extras[:reserve]
        self.last_candidate_pool=tuple(candidates)
        return candidates,memberships

    def retrieve(self,query,top_k=8):
        if top_k<=0 or not self._episode_order: return []
        start=perf_counter(); candidates,memberships=self.candidate_pool(query); built=perf_counter()
        if self.local_reranker is None: raise RuntimeError("Pinned local cross-encoder required")
        scores=np.asarray(self.local_reranker.predict([(query,self._documents[i]) for i in candidates],
            batch_size=self.config.rerank_batch_size,show_progress_bar=False),dtype=np.float32)
        if len(scores)!=len(candidates) or not np.isfinite(scores).all(): raise ValueError("Malformed CE scores")
        order=sorted(range(len(candidates)),key=lambda j:(-float(scores[j]),self._episode_order[candidates[j]]))[:top_k]
        hits=[MemoryHit(self._episode_order[candidates[j]],self._documents[candidates[j]],float(scores[j]),
            (self._episode_order[candidates[j]],),"joint_raw_provenance_episode",
            {"schema_ids":sorted(memberships[candidates[j]]),"retrieval_version":self.retrieval_version}) for j in order]
        end=perf_counter(); self._query_seconds+=end-start; self._query_count+=1
        self._retrieval_profile["candidate_build_seconds"]+=built-start
        self._retrieval_profile["cross_encoder_seconds"]+=end-built
        self._retrieval_profile["query_episode_pairs"]+=len(candidates)
        return hits

    def telemetry(self):
        return {**super().telemetry(),"controller":"joint_actual_state_energy_v2_prototype",
                "fragmentation_price":"lambda * log(1 + observed_episode_count)",
                "parameter_status":"must_calibrate_independently_before_prototype_benchmark",
                "context_unknown_policy":"unknown_not_same","historical_damage_definition":"sum_old_support_cosine_loss_increase",
                "max_energy_optimality_gap":max((r["optimality_gap"] for r in self._decision_certificates.values()),default=0.)}
