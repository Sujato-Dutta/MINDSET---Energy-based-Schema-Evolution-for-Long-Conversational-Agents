"""DGX-only forensic diagnosis; never retune or change production MINDSET."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from dataclasses import replace
from pathlib import Path
import os
import sys
import json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))


def artifacts(output,inputs):
    import numpy as np
    from mindset.acl.io import read_json,write_json,digest_file,read_rows
    from mindset.acl.pipeline import write_csv
    summaries=[]; checks=[]
    for folder in sorted(Path(inputs).iterdir()):
        if not (folder/"metrics.json").exists(): continue
        metrics=read_json(folder/"metrics.json"); cfg=read_json(folder/"run_config.json")
        predictions=read_rows(folder/"predictions.jsonl"); args=cfg.get("arguments",{})
        entry={"run":folder.name,"method":args.get("method"),"ablation":args.get("ablation"),
            "questions":len(predictions),"model":args.get("model"),"retrieval_mode":args.get("mindset_retrieval_mode"),
            "input_tokens":metrics.get("efficiency",{}).get("provider_input_tokens"),
            "output_tokens":metrics.get("efficiency",{}).get("provider_output_tokens"),
            "answer_prompt_sha256":cfg.get("provenance",{}).get("answer_system_prompt_sha256"),
            "dataset_sha256":cfg.get("provenance",{}).get("dataset_sha256"),
            "native_f1":metrics.get("native_f1"),"answerable_retrieval":metrics.get("answerable_retrieval"),
            "category_f1":metrics.get("category_f1"),"category_counts":metrics.get("category_counts"),
            "icl_count":metrics.get("icl_count"),"icl_exact_match":metrics.get("icl_exact_match"),
            "native_metrics":{k:v for k,v in metrics.items() if any(x in k.lower() for x in ("recomm","recsys","recall","supplement")) and k!="per_example"},
            "hashes":{p.name:digest_file(p) for p in folder.iterdir() if p.name in ("metrics.json","predictions.jsonl","run_config.json")}}
        if predictions and "sample_id" in predictions[0]:
            valid=[r for r in predictions if r.get("gold_evidence_ids")]
            entry["ranking_question_count"]=len(valid)
            entry["recomputed_ranking"]={k:float(np.mean([r["ranked_retrieval"][k] for r in valid]))
                for k in ("recall@8","precision@8","mrr","ndcg@8")}
            per=metrics.get("per_example",[])
            nonadv=[float(r["score"]) for r in per if int(r["category"])!=5]
            entry["nonadversarial_native_f1"]=float(np.mean(nonadv)) if nonadv else None
            entry["nonadversarial_count"]=len(nonadv)
            telemetry=metrics.get("method_telemetry",{})
            operations=Counter()
            for t in telemetry.values(): operations.update(t.get("operation_counts",{}))
            entry["write_operations"]=dict(operations)
            if "native_f1" in metrics and per:
                recomputed=float(np.mean([r["score"] for r in per]))
                if not np.isclose(recomputed,metrics["native_f1"],rtol=0,atol=1e-12): raise ValueError("Legacy F1 artifact mismatch")
            for k,v in entry["recomputed_ranking"].items():
                if not np.isclose(v,metrics["answerable_retrieval"][k],rtol=0,atol=1e-12):
                    raise ValueError(f"Legacy ranking artifact mismatch: {folder.name}/{k}")
        elif predictions:
            from mindset.evaluation.metrics import exact_match
            icl=[r for r in predictions if str(r.get("category","")).startswith("icl_")]
            import re
            correct=[]
            for r in icl:
                lines=str(r["prediction"]).splitlines()
                parsed=re.sub(r"^answer\s*:\s*","",lines[0].strip() if lines else "",flags=re.IGNORECASE).strip()
                if exact_match(parsed,r.get("answers",r["answer"])): correct.append(r)
            entry["icl_correct_native_first_line_count"]=len(correct)
            entry["prediction_abstentions_by_category"]={cat:{"count":len(rows),"abstentions":sum(
                "no information available" in str(r["prediction"]).casefold() for r in rows)}
                for cat,rows in ((cat,[r for r in predictions if r.get("category")==cat]) for cat in sorted({r.get("category") for r in predictions}))}
        summaries.append(entry)
        checks.append({"run":folder.name,"prediction_ids_unique":len(predictions)==len({r["id"] for r in predictions}),
                       "status":cfg.get("status"),"saved_metrics_present":True})
    write_json(output/"legacy_artifacts.json",{"runs":summaries,"checks":checks,
        "scope":"Frozen saved artifacts; source tree hash alone does not provide a recoverable exact historical checkout"})
    write_csv(output/"legacy_summary.csv",[{"run":r["run"],"questions":r["questions"],
        "native_f1":r["native_f1"],"nonadversarial_f1":r.get("nonadversarial_native_f1"),
        "nonadversarial_count":r.get("nonadversarial_count"),
        "recall@8":(r["answerable_retrieval"] or {}).get("recall@8"),
        "mrr":(r["answerable_retrieval"] or {}).get("mrr"),
        "ndcg@8":(r["answerable_retrieval"] or {}).get("ndcg@8"),
        "icl_count":r.get("icl_count"),"icl_correct":r.get("icl_correct_native_first_line_count"),
        "icl_exact_match":r.get("icl_exact_match"),"recsys_recall@5":r["native_metrics"].get("recsys_recall@5")}
        for r in summaries])
    # Compare the *same* paired IDs, not the old 700 to an unrelated new 50.
    pilot=read_rows(ROOT/"outputs/acl/mindset_v2_development100_r01/retrieval.jsonl")
    paired=[]
    for entry in summaries:
        if not entry["run"].startswith("locomo_"): continue
        old={r["id"]:r for r in read_rows(Path(inputs)/entry["run"]/"predictions.jsonl")}
        for method in ("current","balanced_provenance","flat_hybrid48","lightmem"):
            new=[r for r in pilot if r["dataset"]=="locomo" and r["method"]==method and r["question_id"] in old
                 and old[r["question_id"]].get("gold_evidence_ids")]
            if not new: continue
            row={"legacy_run":entry["run"],"new_arm":method,"common_questions":len(new)}
            for previous,current in (("recall@8","recall@8"),("precision@8","precision@8"),("mrr","mrr@8"),("ndcg@8","ndcg@8")):
                row[f"legacy_{current}"]=float(np.mean([old[r["question_id"]]["ranked_retrieval"][previous] for r in new]))
                row[f"new_{current}"]=float(np.mean([r["metrics"][current] for r in new]))
            paired.append(row)
    if paired: write_csv(output/"matched_saved_ids.csv",paired)
    missing=defaultdict(Counter); category=defaultdict(list)
    lookup={(r["dataset"],r["method"],r["question_id"]):r for r in pilot}
    for r in pilot:
        if r["method"] not in ("current","balanced_provenance","flat_hybrid48","lightmem"): continue
        category[(r["dataset"],r["method"],r["category"])].append(r)
        if r["method"]=="current":
            flat=lookup[(r["dataset"],"flat_hybrid48",r["question_id"])]
            balanced=lookup[(r["dataset"],"balanced_provenance",r["question_id"])]
            missing[r["dataset"]].update({"questions":1,"pool_below_48":int(r["candidate_count"]<48),
                "flat_pool_recall_better":int(flat["candidate_pool_recall@48"]>r["candidate_pool_recall@48"]),
                "flat_top8_recall_better":int(flat["metrics"]["recall@8"]>r["metrics"]["recall@8"]),
                "balanced_top8_recall_better_than_flat":int(balanced["metrics"]["recall@8"]>flat["metrics"]["recall@8"]),
                "balanced_top8_recall_worse_than_flat":int(balanced["metrics"]["recall@8"]<flat["metrics"]["recall@8"])})
    groups=[]
    for (d,m,c),rows in category.items():
        groups.append({"dataset":d,"method":m,"category":c,"questions":len(rows),**{k:float(np.mean([r["metrics"][k] for r in rows]))
             for k in ("recall@8","precision@8","mrr@8","ndcg@8")}})
    write_json(output/"candidate_losses.json",{d:dict(v) for d,v in missing.items()})
    write_csv(output/"category_retrieval.csv",groups)
    print("Legacy artifacts and candidate-loss analysis complete",flush=True)


def replay(output):
    import numpy as np
    from mindset.acl.io import read_json,write_json,append_row,digest_file
    from mindset.acl.pipeline import config,cases_for,local_stack,rank_metrics,write_csv,source_fingerprint
    from mindset.acl.calibration import validate_calibration
    from mindset.acl.clients import load_tokenizer
    from mindset.acl.runtime import CachedEncoder
    from mindset.acl.evidence import EvidenceCache
    from mindset.acl.iteration import candidate_variants,variant_hits
    from mindset.acl.controllers import rule_controller
    from mindset.methods.compact import CompactMindsetMemory,CompactConfig,select_transition
    from mindset.methods.mindset import MindsetMemory,MindsetConfig,_expand_query
    from mindset.methods.rerank_cache import CachedReranker
    cfg=config(); validate_calibration(cfg,ROOT)
    selected={r["id"] for r in read_json(ROOT/"data/manifests/mindset_v2_development100.json")["records"] if r["dataset"]=="locomo"}
    cases=[c for c in cases_for("locomo",cfg) if c.id in selected]
    grouped=defaultdict(list)
    for c in cases: grouped[c.cluster].append(c)
    write_json(output/"replay_audit.json",{"source_fingerprint":source_fingerprint(),"script_sha256":digest_file(__file__),
        "scope":"50 frozen development LoCoMo questions; retained-v1 code reconstruction, NOT exact original source checkout",
        "qa_calls":0,"parameters_refit":False,"efficiency_claims":False,
        "read_legacy_limits":"Retained v1 can exceed 48 scored candidates; paired budget and candidate count are recorded, not assumed equal",
        "question_ids":sorted(selected)})
    encoder,reranker=local_stack(cfg,cache=False); encoder=CachedEncoder(encoder,enabled=True)
    lock=read_json(ROOT/"configs/checkpoints.lock.json")
    reranker=CachedReranker(reranker,json.dumps(lock["reranker"],sort_keys=True),output/"pair_scores.sqlite")
    tokenizer=load_tokenizer(lock[cfg["evidence_tokenizer"]])
    params=dict(cfg["mindset"]); params.pop("profile")

    class Instrumented(CompactMindsetMemory):
        def __init__(self,**kwargs):
            super().__init__(**kwargs); self.diagnosis=Counter(); self.examples=[]
        def _ingest_episode(self,e):
            self.diagnosis["episodes"]+=1
            self.diagnosis["discourse_episodes"]+=int(e.features.primary_slot=="discourse")
            self.diagnosis["episodes_without_context"]+=int(not e.features.contexts)
            self.diagnosis["episodes_without_attributes"]+=int(not e.features.attributes)
            topic=f"{e.speaker.casefold()}::{e.features.primary_slot}"
            pool=[self.schemas[i] for i in self._by_topic[topic] if self.schemas[i].status=="active"]
            rows=[]
            if pool:
                matrix=np.asarray([s.centroid for s in pool],dtype=np.float32)
                scores=np.einsum("ij,j->i",matrix,e.vector)
                order=sorted(range(len(pool)),key=lambda i:(-float(scores[i]),pool[i].schema_id))[:self.config.nearest_schema_k]
                for i in order:
                    s=pool[i]; signals=self._signals(s,e); energies=self._energies(s,e,signals)
                    allowed={"reinforce","new"}
                    if e.features.primary_slot!="discourse": allowed.discard("new")
                    if signals.temporal_potential: allowed.add("supersede")
                    if signals.contextual_potential: allowed.add("split")
                    action,_=select_transition(energies,allowed,self.config.switch_margin)
                    cost=energies[action].total+self.config.switch_margin*(action=="supersede")
                    rows.append((s,signals,energies,allowed,action,cost))
                target=max(rows,key=lambda r:(r[1].context_match,r[1].dense_similarity,r[0].schema_id))
                s,signals,energies,allowed,action,cost=target
                self.diagnosis["noninitial_decisions"]+=1
                self.diagnosis["one_feasible_action"]+=int(len(allowed)==1)
                self.diagnosis["multiple_feasible_actions"]+=int(len(allowed)>1)
                self.diagnosis["hysteresis_changes_local_action"]+=int(select_transition(energies,allowed,0)[0]!=action)
                no_complexity={a:replace(term,total=term.total-self.config.complexity_penalty*term.fragmentation) for a,term in energies.items()}
                self.diagnosis["complexity_changes_local_action"]+=int(select_transition(no_complexity,allowed,self.config.switch_margin)[0]!=action)
                rule,_=rule_controller(signals,self.config)
                if rule not in allowed: rule="reinforce"
                self.diagnosis["simple_rule_matches_local_action"]+=int(rule==action)
                regret=cost-min(r[5] for r in rows)
                self.diagnosis["lower_energy_other_target"]+=int(regret>1e-7)
                self.diagnosis["target_preselection_energy_regret_sum"]+=max(0.,regret)
                self.diagnosis["supersession_feasible"]+=int("supersede" in allowed)
                self.diagnosis["split_feasible"]+=int("split" in allowed)
                before=len(self.operation_log.all())
                super()._ingest_episode(e)
                record=self.operation_log.all()[before]
                if record.target_schema_id!=s.schema_id or record.operation!=action: raise ValueError("Instrumentation altered the controller")
                if record.stored_as_exception:
                    self.diagnosis["exception_stores"]+=1
                    self.diagnosis["exception_scored_nonzero_historical_damage"]+=int(energies["reinforce"].historical_damage>1e-8)
                    if len(self.examples)<5:
                        self.examples.append({"episode_id":e.episode_id,"text":e.text[:240],
                            "projected_merge_damage":energies["reinforce"].historical_damage,
                            "actual_centroid_change":"none; incoming episode stored as exception"})
                return
            super()._ingest_episode(e)

    rows=[]; structure=[]
    for cluster,queries in sorted(grouped.items()):
        print(f"Replay build {cluster}: frozen energy-v2 and retained energy-v1",flush=True)
        v2=Instrumented(config=CompactConfig(**params),encoder=encoder,local_reranker=reranker)
        v2.ingest(queries[0].events)
        v1=MindsetMemory(MindsetConfig(retrieval_mode="cross_encoder_only"),encoder=encoder,local_reranker=reranker)
        v1.ingest(queries[0].events)
        # The v2 read path needs an all-version centroid table on retained v1.
        v1._schema_ids=sorted(v1.schemas)
        v1._schema_matrix=np.asarray([v1.schemas[i].centroid for i in v1._schema_ids],dtype=np.float32)
        from mindset.methods.mindset import _render_episode_compact
        v1._documents=[_render_episode_compact(v1.episodes[i]) for i in v1._episode_order]
        sizes=[len(s.support_episode_ids) for s in v2.schemas.values()]
        structure.append({"cluster":cluster,"v2_invariants":v2.validate_invariants(),"v1_invariants":v1.validate_invariants(),
            "diagnosis":dict(v2.diagnosis),"examples":v2.examples,"v2_operations":v2.telemetry()["operation_counts"],
            "v1_operations":v1.telemetry()["operation_counts"],"v2_schema_count":len(v2.schemas),"v1_schema_count":len(v1.schemas),
            "v2_support_size_percentiles":list(map(float,np.quantile(sizes,[.5,.95,1.])))})
        write_json(output/"energy_diagnosis.json",structure)
        index=EvidenceCache(queries[0].events,encoder,tokenizer,cfg)
        for c in queries:
            arms={}
            for name,memory in (("v2write_v2read",v2),("v1write_v2read",v1)):
                original_config=memory.config
                try:
                    # Apply read-time operational budgets only; never rebuild
                    # or alter the already constructed controller's state.
                    memory.config=CompactConfig(**params)
                    pools={"current":candidate_variants(memory,c.question)["current"]}
                    arms[name]=variant_hits(memory,c.question,pools)["current"]
                finally: memory.config=original_config
            original=v2.config
            try:
                v2.config=MindsetConfig(retrieval_mode="cross_encoder_only")
                arms["v2write_v1read"]=MindsetMemory.retrieve(v2,c.question,48)
            finally: v2.config=original
            arms["v1write_v1read"]=v1.retrieve(c.question,48)
            # Isolate only the removed query expansion at a fixed direct budget.
            from mindset.retrieval.hybrid import ranked_indices,reciprocal_rank_fusion
            from mindset.memory.energy import unit_vector
            q=unit_vector(v2._encode([_expand_query(c.question)])[0])
            fused=reciprocal_rank_fusion([ranked_indices(v2._episode_vectors@q),ranked_indices(v2._bm25.scores(_expand_query(c.question)))])
            pool=sorted(fused,key=lambda i:(-fused[i],i))[:48]
            arms["v2write_flat48_expanded_query"]=variant_hits(v2,c.question,{"expanded":pool})["expanded"]
            for name,hits in arms.items():
                selected,packet=index.packet(hits,c.question)
                row={"id":f"{name}/{c.id}","question_id":c.id,"cluster":cluster,"category":c.category,"arm":name,
                     "metrics":rank_metrics(selected,c,8),"returned_count":len(hits),"selected_ids":[p["event_id"] for p in selected]}
                append_row(output/"replay.jsonl",row); rows.append(row)
            print(f"Replay retrieved {c.id}: five diagnostic arms",flush=True)
        encoder.cache.clear()
    summary=[]
    for arm in sorted({r["arm"] for r in rows}):
        subset=[r for r in rows if r["arm"]==arm]
        summary.append({"arm":arm,"questions":len(subset),**{k:float(np.mean([r["metrics"][k] for r in subset]))
            for k in ("recall@8","precision@8","mrr@8","ndcg@8")}})
    write_csv(output/"replay_summary.csv",summary)
    totals=Counter()
    for s in structure: totals.update(s["diagnosis"])
    write_json(output/"energy_summary.json",dict(totals))


def main():
    if os.name=="nt" or not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("DGX Slurm-only diagnosis; no laptop or login-node computation")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True); parser.add_argument("--inputs",required=True)
    args=parser.parse_args(); out=Path(args.output)
    if out.exists() and any(out.iterdir()): raise ValueError("Use a fresh diagnosis output directory")
    out.mkdir(parents=True,exist_ok=True)
    from mindset.acl.io import write_json
    write_json(out/"status.json",{"status":"running_forensic_diagnosis","qa_calls":0,"production_changed":False})
    try:
        artifacts(out,args.inputs); replay(out)
    except Exception as exc:
        write_json(out/"status.json",{"status":"failed_partial_diagnosis_preserved","error":f"{type(exc).__name__}: {exc}"})
        raise
    write_json(out/"status.json",{"status":"complete_diagnosis","qa_calls":0,"production_changed":False})

if __name__=="__main__": main()
