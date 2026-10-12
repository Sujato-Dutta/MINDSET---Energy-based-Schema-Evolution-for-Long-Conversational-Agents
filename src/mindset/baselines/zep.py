"""Paper-based reproduction: Rasmussen et al., arXiv:2501.13956v1 (2025).

No Zep service or Graphiti package dependency. Model, storage, embedding and
community-update substitutions are enumerated in docs/baseline_fidelity.md.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json
from collections import Counter, defaultdict
import numpy as np
from mindset.methods.base import ConversationEvent, MemoryHit
from mindset.baselines.event_rag import render_event
from mindset.retrieval.hybrid import BM25Index, ranked_indices, reciprocal_rank_fusion
from mindset.memory.energy import timestamp_value
from mindset.utils.hashing import stable_hash
from mindset.acl.io import write_json

EXTRACT = """Build the temporal entity/fact graph for the CURRENT episode only, using the previous
four episodes to resolve references. The speaker is an entity. Ground all facts in the current
episode; do not invent information. Resolve relative dates against its timestamp.
Return JSON: {"entities":[{"name":"...","summary":"..."}],
"facts":[{"subject":"entity name","object":"entity name","predicate":"...",
"fact":"...","valid_at":"ISO timestamp or null","invalid_at":"ISO timestamp or null"}]}."""
REFLECT = """Check the candidate extraction against CURRENT episode and previous four episodes.
Correct unsupported facts, missing entities, unresolved references and dates.
Return the same JSON schema. Extract only facts supported by CURRENT."""
RESOLVE = """Resolve candidate entities/facts against the provided existing graph candidates.
Return JSON {"entity_matches":{"incoming name":"existing entity ID"},
"duplicate_facts":{"incoming fact index":"existing fact ID"},
"invalidations":[{"existing_fact_id":"...","incoming_fact_index":0}]}.
Omit unmatched items. Only invalidate genuinely contradictory, temporally overlapping old
facts; compatible facts, distinct roles, and context-specific facts must coexist."""

class ZepPaperReproduction:
    method_name = "zep"
    reproduction_label = "2025_paper_reproduction_local_model_and_storage_substitutions"
    paper_url = "https://arxiv.org/abs/2501.13956v1"
    def __init__(self, encoder, reranker, construction_client):
        self.encoder = encoder; self.reranker = reranker; self.client = construction_client
        self.events = []; self.entities = {}; self.facts = {}; self.communities = []
        self.calls = []; self._documents = []; self._vectors = np.empty((0, 0))
    def _call(self, system, data):
        result = self.client.complete([{"role":"system","content":system},
                                       {"role":"user","content":json.dumps(data,ensure_ascii=False)}],
                                      max_tokens=2048)
        self.calls.append(asdict(result))
        text=result.text.strip()
        if text.startswith("```"):
            text=text.split("\n",1)[1].rsplit("```",1)[0]
        return json.loads(text)
    def _encode(self, texts):
        if not texts: return np.empty((0,0),dtype=np.float32)
        return np.asarray(self.encoder.encode(texts,batch_size=64,normalize_embeddings=True,
                                              convert_to_numpy=True,show_progress_bar=False))
    def ingest(self, events):
        for event in events:
            context={"current":asdict(event),"previous_four":[asdict(e) for e in self.events[-4:]]}
            proposed=self._call(EXTRACT,context)
            proposed=self._call(REFLECT,{**context,"candidate":proposed})
            names=[e["name"] for e in proposed["entities"]]
            if event.speaker not in names:
                proposed["entities"].append({"name":event.speaker,"summary":""})
            # Hybrid candidate resolution, bounded per incoming entity.
            existing=list(self.entities.values())
            candidates=[]
            if existing:
                incoming=self._encode([e["name"] for e in proposed["entities"]])
                vectors=self._encode([e["name"] for e in existing])
                lexical=BM25Index([e["name"]+" "+e["summary"] for e in existing])
                chosen=set()
                for name,v in zip([e["name"] for e in proposed["entities"]],incoming):
                    chosen.update(ranked_indices(vectors@v,8)); chosen.update(ranked_indices(lexical.scores(name),8))
                candidates=[existing[i] for i in sorted(chosen)]
            pair_facts=[f for f in self.facts.values() if f["subject"] in {e["id"] for e in candidates}
                        or f["object"] in {e["id"] for e in candidates}]
            # Bound input growth; same-pair exact filtering is applied after resolution.
            resolution=self._call(RESOLVE,{"current":context,"incoming":proposed,
                                          "entities":candidates,"facts":pair_facts[-128:]})
            mapping={}
            for ent in proposed["entities"]:
                match=resolution.get("entity_matches",{}).get(ent["name"])
                if match and match not in {e["id"] for e in candidates}:
                    raise ValueError("Entity resolution cited a non-candidate")
                identifier=match or "zep-entity-"+stable_hash((event.conversation_id,ent["name"]))[:16]
                old=self.entities.get(identifier)
                mapping[ent["name"]]=identifier
                sources=list(dict.fromkeys((old["source_event_ids"] if old else [])+[event.event_id]))
                self.entities[identifier]={"id":identifier,"name":ent["name"],"summary":ent["summary"],
                                           "source_event_ids":sources}
            new_ids={}
            for i, f in enumerate(proposed["facts"]):
                if f["subject"] not in mapping or f["object"] not in mapping:
                    raise ValueError("Fact endpoints not present in entity extraction")
                sid,oid=mapping[f["subject"]],mapping[f["object"]]
                dup=resolution.get("duplicate_facts",{}).get(str(i))
                if dup:
                    if dup not in self.facts or (self.facts[dup]["subject"],self.facts[dup]["object"])!=(sid,oid):
                        raise ValueError("Invalid same-pair fact deduplication")
                    self.facts[dup]["source_event_ids"]=list(dict.fromkeys(
                        self.facts[dup]["source_event_ids"]+[event.event_id]))
                    new_ids[i]=dup; continue
                identifier="zep-fact-"+stable_hash((event.event_id,i,f["fact"]))[:16]
                self.facts[identifier]={"id":identifier,"subject":sid,"object":oid,"predicate":f["predicate"],
                    "text":f["fact"],"valid_at":f.get("valid_at") or event.timestamp,
                    "invalid_at":f.get("invalid_at"),"created_at":len(self.events),"expired_at":None,
                    "source_event_ids":[event.event_id]}
                new_ids[i]=identifier
            for change in resolution.get("invalidations",[]):
                oldid=change["existing_fact_id"]; i=int(change["incoming_fact_index"])
                if oldid not in self.facts or i not in new_ids: raise ValueError("Unknown invalidation ID")
                old,new=self.facts[oldid],self.facts[new_ids[i]]
                if oldid==new["id"]: continue
                start=timestamp_value(new["valid_at"])
                if start>=timestamp_value(old["valid_at"]) and (
                    old["invalid_at"] is None or start<timestamp_value(old["invalid_at"])):
                    old["invalid_at"]=new["valid_at"]; old["expired_at"]=len(self.events)
            self.events.append(event)
        self._communities()
        self._build_index()
    def _communities(self):
        # Frozen offline label-propagation refresh (paper section 2.3).
        adjacency=defaultdict(set)
        for f in self.facts.values():
            adjacency[f["subject"]].add(f["object"]); adjacency[f["object"]].add(f["subject"])
        labels={node:node for node in self.entities}
        for _ in range(20):
            changed=False
            for node in sorted(labels):
                counts=Counter(labels[n] for n in adjacency[node])
                if counts:
                    value=min(counts,key=lambda x:(-counts[x],x))
                    changed |= labels[node]!=value; labels[node]=value
            if not changed: break
        groups=defaultdict(list)
        for node,label in labels.items(): groups[label].append(node)
        for label,members in sorted(groups.items()):
            partial=[]
            for start in range(0,len(members),32):
                data=[self.entities[n] for n in members[start:start+32]]
                partial.append(self._call('Return JSON {"name":"keywords","summary":"grounded summary"} '
                                         'summarizing this entity community.',data))
            summary=partial[0] if len(partial)==1 else self._call(
                'Merge these summaries into JSON {"name":"keywords","summary":"grounded summary"}.',partial)
            self.communities.append({"id":"zep-community-"+stable_hash(members)[:16],**summary,
                "members":members,"source_event_ids":list(dict.fromkeys(
                    e for n in members for e in self.entities[n]["source_event_ids"]))})
    def _build_index(self):
        docs=[]
        for f in self.facts.values():
            docs.append({"id":f["id"],"text":f'{f["text"]} (valid: {f["valid_at"]} to {f["invalid_at"] or "present"})',
                         "search":f["text"],"source_event_ids":f["source_event_ids"],"kind":"fact",
                         "endpoints":[f["subject"],f["object"]]})
        for e in self.entities.values():
            docs.append({"id":e["id"],"text":e["name"]+": "+e["summary"],"search":e["name"],
                         "source_event_ids":e["source_event_ids"],"kind":"entity","endpoints":[e["id"]]})
        for c in self.communities:
            docs.append({"id":c["id"],"text":c["summary"],"search":c["name"],
                         "source_event_ids":c["source_event_ids"],"kind":"community","endpoints":c["members"]})
        self._documents=docs; self._vectors=self._encode([d["search"] for d in docs])
        self._bm25=BM25Index([d["search"] for d in docs])
    def retrieve(self, query, top_k=10):
        if not self._documents: return []
        v=self._encode([query])[0]
        dense=ranked_indices(self._vectors@v,top_k); lexical=ranked_indices(self._bm25.scores(query),top_k)
        seeds=set(n for i in dense[:4]+lexical[:4] for n in self._documents[i]["endpoints"])
        bfs=[i for i,d in enumerate(self._documents) if seeds.intersection(d["endpoints"])][:top_k]
        fused=reciprocal_rank_fusion([dense,lexical,bfs])
        pool=sorted(fused,key=lambda i:(-fused[i],self._documents[i]["id"]))[:2*top_k]
        scores=self.reranker.predict([(query,self._documents[i]["text"]) for i in pool],
                                     batch_size=32,show_progress_bar=False)
        ordered=sorted(zip(pool,scores),key=lambda x:(-float(x[1]),self._documents[x[0]]["id"]))[:top_k]
        return [MemoryHit(self._documents[i]["id"],self._documents[i]["text"],float(s),
                          tuple(self._documents[i]["source_event_ids"]),"zep_"+self._documents[i]["kind"])
                for i,s in ordered]
    def snapshot(self,path):
        write_json(path,{"paper":self.paper_url,"reproduction_label":self.reproduction_label,
            "episodes":[asdict(e) for e in self.events],"entities":self.entities,"facts":self.facts,
            "communities":self.communities,"embedding_vectors":self._vectors.tolist(),
            "construction_calls":self.calls})
        return Path(path)
    def telemetry(self):
        return {"reproduction_label":self.reproduction_label,
                "write_local_llm_calls":sum(not c["cache_hit"] for c in self.calls),
                "logical_construction_calls":len(self.calls),
                "construction_input_tokens":sum(c["input_tokens"] for c in self.calls if not c["cache_hit"]),
                "construction_output_tokens":sum(c["output_tokens"] for c in self.calls if not c["cache_hit"]),
                "entities":len(self.entities),"facts":len(self.facts),"communities":len(self.communities)}

