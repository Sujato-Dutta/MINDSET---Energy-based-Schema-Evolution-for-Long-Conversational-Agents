from __future__ import annotations
import json
import numpy as np
from mindset.baselines.zep import ZepPaperReproduction,EXTRACT,REFLECT,RESOLVE
from mindset.acl.clients import Completion
from mindset.methods.base import ConversationEvent

class Encoder:
    def encode(self,texts,**kwargs):
        result=[]
        for text in texts:
            v=np.asarray([float(word in text.lower()) for word in ("sam","paris","rome","live")])+0.01
            result.append(v/np.linalg.norm(v))
        return np.asarray(result,dtype=np.float32)

class Reranker:
    def predict(self,pairs,**kwargs): return np.arange(len(pairs),dtype=float)[::-1]

class Client:
    def complete(self,messages,**kwargs):
        system=messages[0]["content"]; data=json.loads(messages[1]["content"])
        if system in {EXTRACT,REFLECT}:
            current=data["current"]; city="Rome" if "Rome" in current["text"] else "Paris"
            obj={"entities":[{"name":"Sam","summary":"A person"},{"name":city,"summary":"A city"}],
                 "facts":[{"subject":"Sam","object":city,"predicate":"lives_in",
                           "fact":f"Sam lives in {city}","valid_at":current["timestamp"],"invalid_at":None}]}
        elif system==RESOLVE:
            matches={e["name"]:e["id"] for e in data["entities"] if e["name"]=="Sam"}
            invalid=[{"existing_fact_id":f["id"],"incoming_fact_index":0} for f in data["facts"]
                     if "Paris" in f["text"] and "Rome" in data["current"]["current"]["text"]]
            obj={"entity_matches":matches,"duplicate_facts":{},"invalidations":invalid}
        else: obj={"name":"Sam cities","summary":"Sam has lived in Paris and Rome."}
        return Completion(json.dumps(obj),20,10,.01,"fake")

def test_zep_preserves_raw_provenance_and_temporal_history(tmp_path):
    z=ZepPaperReproduction(Encoder(),Reranker(),Client())
    z.ingest([ConversationEvent("e1","c","s1","2025-01-01","Sam","I live in Paris."),
              ConversationEvent("e2","c","s2","2025-02-01","Sam","I now live in Rome.")])
    assert len(z.events)==2 and len(z.facts)==2
    paris=next(f for f in z.facts.values() if "Paris" in f["text"])
    assert paris["invalid_at"]=="2025-02-01" and paris["expired_at"]==1
    assert paris["source_event_ids"]==["e1"]
    hits=z.retrieve("Where did Sam live?",4)
    assert hits and all(set(h.source_event_ids)<={"e1","e2"} for h in hits)
    assert z.communities and z.telemetry()["write_local_llm_calls"]>0
    z.snapshot(tmp_path/"graph.json")
    saved=json.loads((tmp_path/"graph.json").read_text())
    assert len(saved["episodes"])==2 and saved["paper"].endswith("v1")

