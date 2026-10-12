from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace as N
import sys
import numpy as np
import pytest
from mindset.acl import calibration,pipeline
from mindset.acl.calibration import development_corpus,check_disjoint,validate_calibration
from mindset.acl.io import write_json,digest_file,read_json
from mindset.acl.evidence import EvidenceCache
from mindset.acl.validation import diagnostic_cases
from mindset.acl.profiling import mean_ci,profile_controller
from mindset.acl.analysis import efficiency
from mindset.acl.runtime import ResourceMonitor
from mindset.methods.base import ConversationEvent,MemoryHit
from mindset.acl.evaluation_protocol import published_locomo_prompt,parse_published_label


def test_development_is_independent_split_and_not_teacher_labels(monkeypatch):
    monkeypatch.setattr(calibration,"cases_for",lambda *args:[])
    corpus=development_corpus(per_family=1)
    check_disjoint(corpus,{"benchmarks":{}})
    assert len(corpus["histories"])==18
    bad=deepcopy(corpus); bad["label_source"]="energy_teacher"
    with pytest.raises(ValueError,match="independent"): check_disjoint(bad,{"benchmarks":{}})
    bad=deepcopy(corpus); bad["histories"][1]["split"]="validation"
    # The remaining histories still have all split labels: force a duplicate
    # content history rather than relying only on split names.
    bad["histories"][1]=deepcopy(bad["histories"][0])
    with pytest.raises(ValueError,match="Duplicate"): check_disjoint(bad,{"benchmarks":{}})


def test_development_rejects_benchmark_history_even_with_new_id(monkeypatch):
    corpus=development_corpus(per_family=1)
    events=[ConversationEvent(**e) for e in corpus["histories"][0]["events"]]
    monkeypatch.setattr(calibration,"cases_for",lambda *args:[N(id="test-id",cluster="test-cluster",events=events)])
    with pytest.raises(ValueError,match="overlaps"): check_disjoint(corpus,{"benchmarks":{"locomo":{}}})


def test_calibration_validates_source_data_encoder_and_settings(tmp_path):
    root=tmp_path
    for p in calibration.CONTROLLER_SOURCES:
        write_json(root/p,{"fixture":p})
    write_json(root/"data/dev.json",{"fixture":"dev"})
    encoder={"checkpoint":"fixture","revision":"a"*40}
    write_json(root/"configs/checkpoints.lock.json",{"encoder":encoder})
    artifact={"controller_source_sha256":digest_file(root/"src/mindset/methods/compact.py"),
        "controller_dependency_sha256":{p:digest_file(root/p) for p in calibration.CONTROLLER_SOURCES},
        "dataset_path":"data/dev.json","dataset_sha256":digest_file(root/"data/dev.json"),
        "encoder":encoder,"selected_parameters":{"complexity_penalty":.1,"switch_margin":.2}}
    write_json(root/"artifact.json",artifact)
    cfg={"mindset":{"profile":"compact_v2","complexity_penalty":.1,"switch_margin":.2},
         "calibration":{"artifact":"artifact.json"}}
    validate_calibration(cfg,root)
    changed=deepcopy(cfg); changed["mindset"]["switch_margin"]=.1
    with pytest.raises(ValueError,match="parameters"): validate_calibration(changed,root)
    write_json(root/"src/mindset/memory/energy.py",{"fixture":"changed feature grammar"})
    with pytest.raises(ValueError,match="dependencies"): validate_calibration(cfg,root)


def test_diagnostic_selects_all_clusters_without_looking_at_method_scores():
    cases=[N(id=f"{c}-{i}",cluster=c,category=t,gold=["e"]) for c in ("a","b")
           for i,t in enumerate(("temporal","factual","temporal","factual"))]
    result=diagnostic_cases(cases,"locomo",2)
    assert len(result)==4 and {c.cluster for c in result}=={"a","b"}
    assert len({c.category for c in result if c.cluster=="a"})==2
    assert [c.id for c in result]==[c.id for c in diagnostic_cases(list(reversed(cases)),"locomo",2)]


def test_evidence_cache_preserves_native_ranking_and_static_chunk_reuse():
    class Tokenizer:
        def encode(self,text,**kwargs): return list(text.encode())
        def decode(self,ids): return bytes(ids).decode()
    class Encoder:
        def encode(self,*args,**kwargs): raise AssertionError("Single-chunk raw hits need no dense rescoring")
    events=[ConversationEvent("first","c","s","2025","Sam","alpha"),
            ConversationEvent("second","c","s","2025","Sam","beta")]
    cfg={"retrieval":{"candidate_k":40,"top_k":8,"chunk_tokens":1000,"overlap_tokens":0,"evidence_tokens":8000}}
    cache=EvidenceCache(events,Encoder(),Tokenizer(),cfg)
    hits=[MemoryHit("second","",1.,("second",),"episodic"),MemoryHit("first","",.1,("first",),"episodic")]
    a,_=cache.packet(hits,"alpha"); b,_=cache.packet(hits,"beta")
    assert [c["event_id"] for c in a]==["second","first"]==[c["event_id"] for c in b]
    assert cache.tokenization_calls==2 and cache.chunk_embedding_calls==0


def test_published_semantic_reference_is_strict_json_not_official_acl_baseline():
    prompt=published_locomo_prompt("Where?","Paris","Paris")
    assert "Where?" in prompt and "Paris" in prompt
    assert parse_published_label('{"label":"CORRECT"}') is True
    assert parse_published_label('{"label":"WRONG"}') is False
    for bad in ("yes",'{"label":"maybe"}','{"label":true}'):
        with pytest.raises((ValueError,TypeError)): parse_published_label(bad)


def test_cpu_monitor_never_initializes_gpu_nvml(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    monkeypatch.setitem(sys.modules,"pynvml",N(nvmlInit=lambda:pytest.fail("Do not inspect GPU on CPU allocation")))
    with ResourceMonitor() as monitor:
        assert monitor.gpus==[]
    assert monitor.peak_vram is None


def test_timing_commands_refuse_existing_results(tmp_path):
    out=tmp_path/"existing"; write_json(out/"saved.json",{"status":"existing"})
    with pytest.raises(ValueError,match="fresh output"): profile_controller(out,2,3)
    with pytest.raises(ValueError,match="fresh output"): efficiency("locomo",["mindset"],out,3,2)


def test_student_t_intervals_are_repetition_level_and_finite():
    result=mean_ci([1.,2.,3.])
    assert result["mean"]==2 and result["ci95_low"]<2<result["ci95_high"]
    with pytest.raises(ValueError): mean_ci([1.])
    with pytest.raises(ValueError): mean_ci([1.,np.nan])


def test_method_is_named_mindset_only_and_revision_is_metadata():
    from mindset.methods.compact import CompactMindsetMemory
    assert CompactMindsetMemory.method_name=="mindset"
    assert CompactMindsetMemory.method_version=="v2"
    assert "mindset_v2" not in pipeline.config()["methods"]
    from mindset.methods import MindsetMemory,load_calibrated_config
    cfg=load_calibrated_config(); method=MindsetMemory()
    assert method.config==cfg
    assert cfg.complexity_penalty==.1 and cfg.switch_margin==.2
    assert method.method_name=="mindset" and method.method_version=="v2"


def test_only_registered_ablations_can_change_prepared_protocol():
    cfg=pipeline.config(); prepared=deepcopy(cfg)
    prepared["mindset"]["ablation"]="no_context"
    assert pipeline.check_prepared_protocol(cfg,prepared)=="no_context"
    prepared["mindset"]["complexity_penalty"]+=.1
    with pytest.raises(ValueError,match="Protocol changed"): pipeline.check_prepared_protocol(cfg,prepared)
    prepared=deepcopy(cfg); prepared["mindset"]["ablation"]="test_selected_variant"
    with pytest.raises(ValueError,match="Unregistered"): pipeline.check_prepared_protocol(cfg,prepared)


def test_primary_locomo_manifest_is_exact_filtered_source_audit():
    cfg=pipeline.config(); root=pipeline.ROOT
    from mindset.acl.data import verified_cases
    cases=verified_cases("locomo",root/cfg["benchmarks"]["locomo"]["path"],
                         root/cfg["benchmarks"]["locomo"]["manifest"])
    full=read_json(root/"data/manifests/locomo_full.json")
    primary=read_json(root/cfg["benchmarks"]["locomo"]["manifest"])
    assert len(cases)==1540 and len({c.cluster for c in cases})==10
    assert [r for r in full["records"] if r["category"]!="adversarial"]==primary["records"]
    assert primary["source_sha256"]==full["source_sha256"]


def test_native_locomo_needs_no_proxy_judge_but_lme_and_partial_judges_do():
    predictions={"a":{},"b":{}}
    assert pipeline.complete_judge_labels("locomo",predictions,{}) is False
    with pytest.raises(ValueError,match="LongMemEval"):
        pipeline.complete_judge_labels("longmemeval",predictions,{})
    with pytest.raises(ValueError,match="Partial"):
        pipeline.complete_judge_labels("locomo",predictions,{"a":{"label":True}})


def test_native_statistics_distinguish_full_method_and_energy_ablation(tmp_path,monkeypatch):
    # Toy fixtures are unit-test inputs, not reported benchmark scores.
    from mindset.acl import analysis
    from mindset.acl.io import append_row
    cfg={"benchmarks":{"locomo":{"manifest":"manifest.json"}},"seed":0,
         "statistics":{"resamples":100},"mindset":{},"ablations":["no_context"]}
    ids=[f"synthetic-unit-{i}" for i in range(10)]
    write_json(tmp_path/"manifest.json",{"records":[{"id":i} for i in ids]})
    monkeypatch.setattr(analysis,"ROOT",tmp_path); monkeypatch.setattr(analysis,"config",lambda:cfg)
    runs=[]
    for method,ablation in (("mindset","full"),("mindset","no_context"),("vector_rag","full")):
        folder=tmp_path/f"{method}-{ablation}"; runs.append(folder)
        protocol=deepcopy(cfg)
        if ablation!="full": protocol["mindset"]["ablation"]=ablation
        write_json(folder/"run_config.json",{"model":"fixture","prepared_config":{
            "method":method,"controller":"energy","dataset":"locomo","protocol":protocol}})
        value=1. if method=="mindset" and ablation=="full" else .5
        for i in ids:
            append_row(folder/"predictions.jsonl",{"id":i,"cluster":i,"category":"temporal",
                "token_f1":value,"native_f1":value,"retrieval":{"recall@8":value,
                 "precision@8":value/8,"mrr@8":value,"ndcg@8":value}})
    analysis.statistics(runs,tmp_path/"statistics")
    import csv
    with (tmp_path/"statistics/statistics.csv").open() as f: rows=list(csv.DictReader(f))
    assert all(r["metric"]!="judge_accuracy" for r in rows)
    assert any(r["ablation"]=="no_context" and r["hypothesis_family"]=="controller" for r in rows)
    assert all(float(r["delta"])==.5 for r in rows if r["metric"]=="native_f1")


def test_repeated_lme_source_session_has_unique_episodes_and_unchanged_gold_unit():
    from mindset.datasets.events import longmemeval_events
    record={"question_id":"fixture","haystack_sessions":[[{"role":"user","content":"a"}],
        [{"role":"user","content":"b"}]],"haystack_session_ids":["same","same"],
        "haystack_dates":["2025-01-01","2025-01-02"]}
    events=longmemeval_events(record)
    assert len({e.event_id for e in events})==2 and events[0].event_id=="fixture:same:0"
    assert [e.session_id for e in events]==["same","same"]
    assert [e.text for e in events]==["a","b"]
