from copy import deepcopy
from collections import Counter
from types import SimpleNamespace as N
import numpy as np
import pytest
from mindset.acl.iteration import (stratified_cases,round_robin_unique,candidate_variants,
    variant_hits,logvolume_order,verify_iteration,summarize_iteration,VARIANTS,BASELINES)
from mindset.acl.io import write_json,digest_file,read_json,append_row
from mindset.methods.compact import CompactMindsetMemory,CompactConfig
from mindset.methods.base import ConversationEvent


def test_stratified_selection_is_order_independent_balanced_and_excludes_probes():
    cases=[N(id=f"{c}-{t}-{i}",cluster=c,category=t,gold=["e"]) for c in range(10)
           for t in ("a","b","c","d") for i in range(6)]
    selected=stratified_cases(cases,"locomo",50,0,{"0-a-0"})
    assert len(selected)==50 and "0-a-0" not in {c.id for c in selected}
    assert set(Counter(c.cluster for c in selected).values())=={5}
    for cluster in range(10): assert len({c.category for c in selected if c.cluster==cluster})==4
    assert [c.id for c in selected]==[c.id for c in stratified_cases(list(reversed(cases)),"locomo",50,0,{"0-a-0"})]
    assert [c.id for c in selected]!=[c.id for c in stratified_cases(cases,"locomo",50,1,{"0-a-0"})]
    lme=stratified_cases(cases,"longmemeval",50)
    assert max(Counter(c.category for c in lme).values())-min(Counter(c.category for c in lme).values())<=1
    with pytest.raises(ValueError): stratified_cases(cases,"locomo",99999)


def test_selection_never_reads_answers_or_gold_contents():
    case=N(id="only",cluster="c",category="a",gold=["unknown"])
    assert stratified_cases([case],"locomo",1)[0] is case
    case.gold=["different"]
    assert stratified_cases([case],"locomo",1)[0] is case


def test_unique_budget_backfills_overlapping_paths_fairly():
    assert round_robin_unique([[1,2,3],[1,4,5]],[1],5)==[1,2,4,3,5]
    assert round_robin_unique([[],[1,2,3]],[1],3)==[1,2,3]
    assert round_robin_unique([[1]],[1,1,2],1)==[1]


class Encoder:
    def encode(self,texts,**kwargs):
        rows=np.asarray([[1.,float("tea" in t.lower()),float("work" in t.lower()),
                          float("coffee" in t.lower())] for t in texts],dtype=np.float32)
        return rows/np.linalg.norm(rows,axis=1,keepdims=True)


class Reranker:
    def __init__(self): self.calls=[]
    def predict(self,pairs,**kwargs):
        self.calls.append(list(pairs))
        return np.asarray([float(sum(w.lower() in d.lower() for w in q.split())) for q,d in pairs])


def example_memory():
    memory=CompactMindsetMemory(CompactConfig(),encoder=Encoder(),local_reranker=Reranker())
    events=[ConversationEvent(f"e{i:03d}","c",f"s{i}",f"2025-01-{1+i%27:02d}","Sam",
                "I prefer coffee at work." if i%2 else "I prefer tea at home.") for i in range(60)]
    memory.ingest(events)
    return memory


def test_control_reproduces_current_pool_and_candidate_budgets():
    memory=example_memory(); query="Sam tea home"
    native=memory.retrieve(query,48)
    pools=candidate_variants(memory,query)
    assert set(memory._episode_order[i] for i in pools["current"])=={h.memory_id for h in native}
    assert len(pools["balanced_provenance"])==len(pools["flat_hybrid48"])==48
    assert all(len(p)==len(set(p))<=48 for p in pools.values())
    results=variant_hits(memory,query,pools)
    assert [h.memory_id for h in results["current"]]==[h.memory_id for h in native]
    assert all(h.source_event_ids==(h.memory_id,) for hits in results.values() for h in hits)
    assert len(memory.local_reranker.calls)==2  # one native + one shared union
    assert len(memory.local_reranker.calls[-1])==len({i for pool in pools.values() for i in pool})


def test_version_paths_preserve_historical_provenance():
    memory=CompactMindsetMemory(encoder=Encoder(),local_reranker=Reranker())
    memory.ingest([ConversationEvent("old","c","s1","2025-01-01","Sam","I live in Paris."),
                   ConversationEvent("new","c","s2","2025-01-02","Sam","I moved to Rome.")])
    pools=candidate_variants(memory,"Sam city")
    assert {memory._episode_order[i] for i in pools["balanced_provenance"]}=={"old","new"}


def test_logvolume_keeps_best_first_then_prefers_independent_evidence():
    order=logvolume_order([4.,3.9,3.8],[[1.,0.],[1.,0.],[0.,1.]],["a","b","c"],2)
    assert order==[0,2,1]
    assert logvolume_order([],[],[])==[]
    with pytest.raises(ValueError,match="Malformed"): logvolume_order([float("nan")],[[1.]],["a"])


def test_iteration_verifies_frozen_ids_counts_and_sources(tmp_path,monkeypatch):
    from mindset.acl import iteration
    monkeypatch.setattr(iteration,"ROOT",tmp_path)
    write_json(tmp_path/"calibration.json",{"frozen":True})
    cfg={"calibration":{"artifact":"calibration.json"},"benchmarks":{}}
    value={"count":100,"calibration_sha256":digest_file(tmp_path/"calibration.json"),"records":[],"sources":{}}
    for d in ("locomo","longmemeval"):
        rows=[{"id":str(i),"cluster":str(i),"history_sha256":str(i)} for i in range(50)]
        write_json(tmp_path/f"{d}.json",{"records":rows}); write_json(tmp_path/f"{d}-source.json",{})
        cfg["benchmarks"][d]={"manifest":f"{d}.json","path":f"{d}-source.json"}
        value["sources"][d]={"manifest_path":f"{d}.json","manifest_sha256":digest_file(tmp_path/f"{d}.json"),
            "source_path":f"{d}-source.json","source_sha256":digest_file(tmp_path/f"{d}-source.json")}
        value["records"].extend({"dataset":d,**r} for r in rows)
    verify_iteration(value,cfg)
    bad=deepcopy(value); bad["records"][0]["history_sha256"]="tampered"
    with pytest.raises(ValueError,match="audit mismatch"): verify_iteration(bad,cfg)
    bad=deepcopy(value); bad["records"][0]=deepcopy(bad["records"][1])
    with pytest.raises(ValueError,match="Duplicate"): verify_iteration(bad,cfg)


def test_empty_iteration_exports_pending_status_without_fake_scores(tmp_path):
    assert summarize_iteration(tmp_path,{"count":100})==[]
    status=read_json(tmp_path/"status.json")
    assert status["completed_question_method_pairs"]==0
    assert status["status"]=="partial_development"
    assert not (tmp_path/"retrieval.csv").exists()
    assert not status["full_runs_authorized"]


def test_even_perfect_development_scores_never_authorize_full_runs(tmp_path):
    manifest={"count":100}
    for d in ("locomo","longmemeval"):
        for method in (*VARIANTS,*BASELINES):
            for i in range(50):
                append_row(tmp_path/"retrieval.jsonl",{"id":f"{d}/{method}/{i}","dataset":d,"method":method,
                    "question_id":str(i),"candidate_count":48,"candidate_pool_recall@48":1.,
                    "metrics":{"recall@8":1.,"precision@8":1.,"mrr@8":1.,"ndcg@8":1.}})
    summarize_iteration(tmp_path,manifest)
    status=read_json(tmp_path/"status.json")
    assert status["completed_question_method_pairs"]==700 and status["completed_unique_questions"]==100
    assert not status["full_runs_authorized"] and status["qa_calls"]==status["judge_calls"]==0
