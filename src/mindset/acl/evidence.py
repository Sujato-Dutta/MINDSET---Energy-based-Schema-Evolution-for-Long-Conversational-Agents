"""Shared raw-source adapter that preserves each method's native ranking."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from mindset.baselines.event_rag import render_event
from mindset.acl.io import write_json

class EvidenceCache:
    """Fresh, history-local static index; never cache query results or scores."""
    def __init__(self,events,encoder,tokenizer,cfg):
        self.events={e.event_id:e for e in events}
        self.encoder=encoder; self.tokenizer=tokenizer; self.cfg=cfg
        self.chunks={}; self.vectors={}
        self.tokenization_calls=0; self.chunk_embedding_calls=0

    def event_chunks(self,event):
        if event.event_id not in self.chunks:
            ids=self.tokenizer.encode(render_event(event),add_special_tokens=False)
            self.tokenization_calls+=1
            length=self.cfg["retrieval"]["chunk_tokens"]; overlap=self.cfg["retrieval"]["overlap_tokens"]
            if not 0<=overlap<length: raise ValueError("Invalid chunk overlap")
            chunks=[]
            for start in range(0,len(ids),length-overlap):
                chunks.append({"event_id":event.event_id,"session_id":event.session_id,
                    "timestamp":event.timestamp,"speaker":event.speaker,
                    "text":self.tokenizer.decode(ids[start:start+length]),
                    "token_start":start,"token_end":min(start+length,len(ids))})
                if start+length>=len(ids): break
            self.chunks[event.event_id]=chunks
        return self.chunks[event.event_id]

    def snapshot(self,path):
        path.mkdir(parents=True,exist_ok=True)
        write_json(path/"evidence_index.json",{"chunks":self.chunks,"vector_order":sorted(self.vectors),
            "tokenization_calls":self.tokenization_calls,"chunk_embedding_calls":self.chunk_embedding_calls})
        if self.vectors:
            np.savez_compressed(path/"evidence_index.npz",**{f"event_{i}":self.vectors[k]
                for i,k in enumerate(sorted(self.vectors))})

    def packet(self,hits,query):
        selected=[]; seen=set(); qvec=None; opts=self.cfg["retrieval"]
        def query_vector():
            nonlocal qvec
            if qvec is None:
                qvec=self.encoder.encode([query],normalize_embeddings=True,convert_to_numpy=True,
                                          show_progress_bar=False)[0]
            return qvec
        for hit in hits[:opts["candidate_k"]]:
            sources=[self.events[i] for i in hit.source_event_ids if i in self.events and i not in seen]
            if len(sources)>1:
                vecs=self.encoder.encode([render_event(e) for e in sources],batch_size=64,
                    normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
                sources=[x[0] for x in sorted(zip(sources,vecs@query_vector()),
                                             key=lambda x:(-float(x[1]),x[0].event_id))[:3]]
            for event in sources:
                seen.add(event.event_id); chunks=self.event_chunks(event)
                if not chunks: continue
                if len(chunks)>1:
                    if event.event_id not in self.vectors:
                        self.vectors[event.event_id]=self.encoder.encode([c["text"] for c in chunks],
                            batch_size=64,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
                        self.chunk_embedding_calls+=1
                    scores=self.vectors[event.event_id]@query_vector()
                    chunks=[chunks[int(i)] for i in np.argsort(-scores,kind="stable")]
                # One best passage per raw event before spending evidence on duplicates.
                candidate=chunks[0]
                rendered="\n\n".join(f"[{i}] {c['text']}" for i,c in enumerate([*selected,candidate],1))
                if len(self.tokenizer.encode(rendered,add_special_tokens=False))>opts["evidence_tokens"]: continue
                selected.append(candidate)
                if len(selected)>=opts["top_k"]:
                    return selected,rendered
        return selected,"\n\n".join(f"[{i}] {c['text']}" for i,c in enumerate(selected,1))
