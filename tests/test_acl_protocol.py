from __future__ import annotations
import json
import copy
import numpy as np
import pytest
from dataclasses import asdict
from pathlib import Path
from mindset.acl.data import manifest,verified_cases,Case
from mindset.acl.analysis import paired_stats
from mindset.acl.controllers import FEATURES,ACTIONS,fit_controller,LearnedController,rule_controller,nearest_controller
from mindset.acl.io import write_json,read_json,append_row,unique_rows
from mindset.acl import pipeline
from mindset.methods.base import ConversationEvent,MemoryHit
from mindset.memory.energy import TransitionSignals
from mindset.methods.mindset import MindsetConfig,MindsetMemory

def test_cluster_test_two_sided_minimum_and_weighted_identity():
    # Unequal cluster sizes: estimate must be pooled, not unweighted cluster means.
    a=[1]*12; b=[0]*12; clusters=["c0"]*3+[f"c{i}" for i in range(1,10)]
    r=paired_stats(a,b,clusters,resamples=100)
    assert r["delta"]==np.mean(a)-np.mean(b)==1
    assert r["p"]==2/1024
    assert r["units"]==10
    varied=paired_stats([1,1,0],[0,0,1],["a","a","b"],resamples=100)
    assert varied["delta"]==pytest.approx(1/3)

def test_manifest_is_full_ordered_and_detects_source_mutation(tmp_path):
    source=tmp_path/"locomo.json"
    write_json(source,[{"sample_id":"c1","conversation":{"speaker_a":"Sam","speaker_b":"Pat",
        "session_1_date_time":"2025-01-01","session_1":[{"speaker":"Sam","dia_id":"D1:1","text":"Paris"}]},
        "qa":[{"category":4,"question":"Where?","answer":"Paris","evidence":["D1:1"]},
              {"category":5,"question":"Unknown?","adversarial_answer":"invented","evidence":[]}]}])
    m=manifest("locomo",source,2); saved=tmp_path/"manifest.json"; write_json(saved,m)
    cases=verified_cases("locomo",source,saved)
    assert len(cases)==2 and cases[1].answer=="No information available."
    assert cases[0].id=="c1:q0000"
    source.write_text(source.read_text()+" ")
    with pytest.raises(ValueError,match="Dataset changed"): verified_cases("locomo",source,saved)

def test_retrieval_deduplicates_native_units_and_omits_unannotated():
    case=Case("q","c","x","?","",[],["s1","s2"],"session")
    selected=[{"session_id":"s1"},{"session_id":"s1"},{"session_id":"s2"}]
    r=pipeline.rank_metrics(selected,case,8)
    assert r["recall@8"]==1 and r["precision@8"]==.25
    case.gold=[]
    assert pipeline.rank_metrics(selected,case,8) is None

def test_classifier_rejects_test_leakage_and_energy_pseudolabels(tmp_path):
    records=[]
    for action_i,action in enumerate(ACTIONS):
        for i in range(3):
            features={f:0.1 for f in FEATURES}; features["compatibility"]=action_i/4
            records.append({"id":f"dev-{action}-{i}","features":features,"action":action})
    source=tmp_path/"labels.json"; target=tmp_path/"weights.json"
    write_json(source,{"label_source":"energy_teacher","records":records})
    with pytest.raises(ValueError,match="independently"): fit_controller(source,target,set())
    write_json(source,{"label_source":"synthetic_ground_truth","records":records})
    with pytest.raises(ValueError,match="overlaps"): fit_controller(source,target,{records[0]["id"]})
    fit_controller(source,target,set())
    learned=LearnedController(target)
    sig=TransitionSignals(**{f:.1 for f in FEATURES})
    assert learned(sig,MindsetConfig())[0] in ACTIONS
    assert read_json(target)["training_ids"]==[r["id"] for r in records]

def test_controller_injection_changes_only_transition_policy():
    cfg=MindsetConfig(retrieval_mode="legacy")
    method=MindsetMemory(cfg,transition_controller=lambda signals,config:("new",False))
    method.ingest([ConversationEvent("e1","c","s","2025-01-01","Sam","I live in Paris."),
                   ConversationEvent("e2","c","s","2025-01-02","Sam","I live in Paris.")])
    assert [r.operation for r in method.operation_log.all()]==["new","new"]
    assert len(method.schemas)==2 and len(method.episodes)==2

def test_append_checkpoint_prevents_duplicate_ids(tmp_path):
    path=tmp_path/"records.jsonl"
    append_row(path,{"id":"a","prediction":"answer"})
    assert unique_rows(path)["a"]["prediction"]=="answer"
    append_row(path,{"id":"a","prediction":"same"})
    with pytest.raises(ValueError,match="Duplicate"): unique_rows(path)

def test_bm25_postings_preserve_reference_scores():
    from mindset.retrieval.hybrid import BM25Index
    from mindset.memory.energy import tokenize
    index=BM25Index(["tea garden tea","garden cat","pune quantum"])
    result=index.scores("tea garden")
    expected=np.zeros(3,dtype=np.float32)
    for token in tokenize("tea garden"):
        for i,counts in enumerate(index.term_counts):
            f=counts.get(token,0)
            if f:
                den=f+index.k1*(1-index.b+index.b*index.lengths[i]/max(index.average_length,1))
                expected[i]+=index.idf[token]*f*(index.k1+1)/den
    np.testing.assert_array_equal(result,expected)

def test_resume_answer_and_context_failure_have_no_committed_prediction(tmp_path,monkeypatch):
    cfg={"benchmarks":{"locomo":{"manifest":"manifest.json"}},"generation":{"max_output_tokens":4,
         "common_context_tokens":10},"seed":0}
    lock={"test":{"checkpoint":"test/model","revision":"a"*40}}
    monkeypatch.setattr(pipeline,"ROOT",tmp_path)
    monkeypatch.setattr(pipeline,"config",lambda:copy.deepcopy(cfg))
    monkeypatch.setattr(pipeline,"record_environment",lambda target:None)
    write_json(tmp_path/"configs/checkpoints.lock.json",lock)
    write_json(tmp_path/"configs/models.yaml",{"models":{"test":{}}})
    write_json(tmp_path/"manifest.json",{"records":[{"id":"q"}]})
    prep=tmp_path/"prepared"; run=tmp_path/"answers"
    write_json(prep/"run_config.json",{"protocol":cfg,"checkpoints":lock,"dataset":"locomo","method":"vector_rag"})
    append_row(prep/"evidence.jsonl",{"id":"q","question":"?","question_date":"","evidence":"too long","selected":[]})
    monkeypatch.setattr(pipeline,"load_tokenizer",lambda x:None)
    monkeypatch.setattr(pipeline,"server_client",lambda *x:None)
    monkeypatch.setattr(pipeline,"prompt_tokens",lambda *x,**kw:100)
    with pytest.raises(ValueError,match="exceeds common context"): pipeline.answer(prep,"test",run)
    assert not (run/"predictions.jsonl").exists()

def test_freeze_reuses_published_checkpoint_lock_without_resolving_heads(monkeypatch):
    monkeypatch.setattr(pipeline,"manifest",lambda *args,**kwargs:{"count":0})
    monkeypatch.setattr(pipeline,"write_json",lambda *args,**kwargs:None)
    def unexpected_network(*args,**kwargs):
        raise AssertionError("An existing lock must not resolve moving remote heads")
    monkeypatch.setattr(pipeline,"request_json",unexpected_network)
    monkeypatch.setattr(pipeline.urllib.request,"urlopen",unexpected_network)
    pipeline.freeze()


def test_full_pipeline_resumes_and_exports_saved_metrics(tmp_path,monkeypatch):
    from mindset.acl.clients import Completion
    from mindset.api.models import CompletionResult,TokenUsage
    from mindset.api.backend import APIBackend
    cfg=copy.deepcopy(pipeline.config()); cfg["status"]="frozen"
    cfg["benchmarks"]={"locomo":{"path":"locomo.json","manifest":"manifest.json","expected_count":2}}
    lock={"qwen35_9b":{"checkpoint":"test/model","revision":"a"*40}}
    source=tmp_path/"locomo.json"
    write_json(source,[{"sample_id":"c1","conversation":{"speaker_a":"Sam","speaker_b":"Pat",
        "session_1_date_time":"2025-01-01","session_1":[
            {"speaker":"Sam","dia_id":"D1:1","text":"I live in Paris."},
            {"speaker":"Pat","dia_id":"D1:2","text":"I prefer tea."}]},
        "qa":[{"category":4,"question":"Where does Sam live?","answer":"Paris","evidence":["D1:1"]},
              {"category":4,"question":"What does Pat prefer?","answer":"tea","evidence":["D1:2"]}]}])
    write_json(tmp_path/"manifest.json",manifest("locomo",source,2))
    write_json(tmp_path/"configs/checkpoints.lock.json",lock)
    write_json(tmp_path/"configs/models.yaml",{"models":{"qwen35_9b":{}}})
    write_json(tmp_path/"configs/judge.yaml",{})
    monkeypatch.setattr(pipeline,"ROOT",tmp_path)
    monkeypatch.setattr(pipeline,"config",lambda:copy.deepcopy(cfg))
    monkeypatch.setattr(pipeline,"record_environment",lambda p:None)
    class Encoder:
        def encode(self,texts,**kwargs):
            result=[]
            for text in texts:
                v=np.asarray([int(w in text.lower()) for w in ["paris","tea"]],dtype=float)+.01
                result.append(v/np.linalg.norm(v))
            return np.asarray(result)
    class Tokenizer:
        def __init__(self): self.words=[]
        def encode(self,text,**kwargs):
            ids=[]
            for w in text.split():
                if w not in self.words: self.words.append(w)
                ids.append(self.words.index(w))
            return ids
        def decode(self,ids): return " ".join(self.words[i] for i in ids)
        def apply_chat_template(self,msgs,**kwargs):
            return self.encode(" ".join(m["content"] for m in msgs))
    tokenizer=Tokenizer(); stacks=[]
    monkeypatch.setattr(pipeline,"local_stack",lambda *a,**k:(stacks.append(1) or Encoder(),None))
    monkeypatch.setattr(pipeline,"load_tokenizer",lambda lock:tokenizer)
    prep=tmp_path/"prepared"; run=tmp_path/"run"
    pipeline.prepare("locomo","vector_rag",prep)
    assert len(unique_rows(prep/"evidence.jsonl"))==2
    pipeline.prepare("locomo","vector_rag",prep)
    assert len(stacks)==1 and len(unique_rows(prep/"ingestion.jsonl"))==1
    class Client:
        calls=0
        def complete(self,msgs,*args,**kwargs):
            self.calls+=1
            return Completion("Paris" if "Where does Sam" in msgs[1]["content"] else "tea",20,1,.01,"test")
    client=Client(); monkeypatch.setattr(pipeline,"server_client",lambda *a,**k:client)
    pipeline.answer(prep,"qwen35_9b",run); pipeline.answer(prep,"qwen35_9b",run)
    assert client.calls==2
    # Native LoCoMo evaluation is complete without any supplementary API judge.
    pipeline.summarize([run],tmp_path/"native_report")
    import csv
    with (tmp_path/"native_report/results.csv").open() as f:
        native_row=next(csv.DictReader(f))
    assert native_row["native_f1"]=="1.0" and native_row["judge_count"]=="0"
    assert native_row["judge_accuracy"]==""
    class Judge:
        def complete(self,*args,**kwargs):
            return CompletionResult(text='{"label":"CORRECT"}',provider="test",model_id="gemini-3.1-flash-lite",
                key_alias="key_01",usage=TokenUsage(input_tokens=10,output_tokens=1),
                latency_seconds=.01,cache_key="test",provider_model_version="fixed-test")
    monkeypatch.setattr(APIBackend,"from_file",lambda *args:Judge())
    pipeline.judge(run)
    assert len(unique_rows(run/"judgements.jsonl"))==2
    pipeline.summarize([run],tmp_path/"report")
    import csv
    with (tmp_path/"report/results.csv").open() as f: row=next(csv.DictReader(f))
    assert row["judge_accuracy"]=="1.0" and row["token_f1"]=="1.0"
    assert row["native_f1"]=="1.0"
    assert (tmp_path/"report/efficiency.csv").exists()


