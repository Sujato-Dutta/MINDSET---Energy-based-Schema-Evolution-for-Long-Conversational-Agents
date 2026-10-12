"""Independent synthetic development probes; never tune on benchmark questions.

Synthetic transition labels follow the declared world/policy specification,
not actions predicted by MINDSET. They are not natural-dialogue performance.
"""
from __future__ import annotations
from dataclasses import asdict
from collections import Counter
import itertools
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
from mindset.acl.io import read_json,write_json,digest_file
from mindset.acl.pipeline import ROOT,config,cases_for,write_csv
from mindset.acl.controllers import FEATURES,fit_controller
from mindset.methods.compact import CompactConfig,CompactMindsetMemory
from mindset.methods.base import ConversationEvent
from mindset.utils.hashing import stable_hash

FAMILIES=("repeat","update","coexist","noise","confirmation","new_topic")
ACTION_ORDER=("reinforce","supersede","split","new")
CONTROLLER_SOURCES=("src/mindset/methods/compact.py","src/mindset/methods/mindset.py",
                    "src/mindset/memory/energy.py","src/mindset/retrieval/hybrid.py")

def content_hash(events):
    return stable_hash([(e.timestamp,e.speaker," ".join(e.text.split())) for e in events])

def development_corpus(seed=0,per_family=8):
    rng=np.random.default_rng(seed); histories=[]
    cities={"train":("Oslo","Lima","Accra"),"validation":("Kyoto","Sofia","Perth"),
            "holdout":("Bern","Seoul","Dakar")}
    for split in cities:
        for family in FAMILIES:
            for i in range(per_family):
                left,right=rng.choice(cities[split],2,replace=False)
                if family=="repeat":
                    texts=[f"I live in {left}."]*3; labels=["new","reinforce","reinforce"]
                elif family=="update":
                    texts=[f"I live in {left}.",f"I moved to {right}.",f"I live in {right}."]
                    labels=["new","supersede","reinforce"]
                elif family=="coexist":
                    texts=["I prefer tea at home.","I prefer coffee at work.","I prefer tea at home."]
                    labels=["new","split","reinforce"]
                elif family=="noise":
                    texts=[f"I live in {left}.",f"I live in {right}.",f"I live in {left}."]
                    labels=["new","reinforce","reinforce"]
                elif family=="confirmation":
                    texts=[f"I live in {left}.",f"I live in {right}.",f"I live in {right}."]
                    labels=["new","reinforce","supersede"]
                else:
                    texts=["The telescope photographed Saturn.","A cake needs flour and butter.",
                           "A cake needs flour and butter."]
                    labels=["new","new","reinforce"]
                identifier=f"synthetic-{split}-{family}-{i:03d}"
                # Distinct speakers/timestamps keep exact histories disjoint across splits.
                speaker=f"Dev{split}{family}{i}"
                events=[ConversationEvent(f"{identifier}-e{j}",identifier,f"{identifier}-s{j}",
                    f"2031-01-{j+1:02d}",speaker,text) for j,text in enumerate(texts)]
                histories.append({"id":identifier,"split":split,"family":family,"events":[asdict(e) for e in events],
                    "actions":labels,"history_content_sha256":content_hash(events)})
    return {"label_source":"synthetic_ground_truth","generator":"compact-development-v2",
            "seed":seed,"per_family":per_family,"scope":"controlled_development_not_qa_benchmark",
            "limitations":"Canonical English slot templates; not proof of natural-language OOD generalization",
            "histories":histories}

def check_disjoint(corpus,cfg):
    if corpus.get("label_source") not in {"human","synthetic_ground_truth"}:
        raise ValueError("Development labels must be independent of MINDSET energy decisions")
    if {h.get("split") for h in corpus["histories"]}!={"train","validation","holdout"}:
        raise ValueError("Development requires separate train, validation and holdout histories")
    ids=[h["id"] for h in corpus["histories"]]
    if len(ids)!=len(set(ids)): raise ValueError("Duplicate development history IDs")
    forbidden_ids=set(); forbidden_histories=set()
    for name in cfg["benchmarks"]:
        seen_clusters=set()
        for case in cases_for(name,cfg):
            forbidden_ids.update((case.id,case.cluster))
            if case.cluster not in seen_clusters:
                forbidden_histories.add(content_hash(case.events)); seen_clusters.add(case.cluster)
    hashes=[]
    for h in corpus["histories"]:
        events=[ConversationEvent(**e) for e in h["events"]]
        if not events or len(events)!=len(h["actions"]) or not set(h["actions"])<=set(ACTION_ORDER):
            raise ValueError("Development actions must align with nonempty histories")
        if len({e.event_id for e in events})!=len(events):
            raise ValueError("Duplicate development episode IDs")
        digest=content_hash(events); hashes.append(digest)
        if h["id"] in forbidden_ids or digest in forbidden_histories:
            raise ValueError("Development data overlaps a benchmark history")
        if digest!=h["history_content_sha256"]: raise ValueError("Development history changed")
    if len(set(hashes))!=len(hashes): raise ValueError("Development split histories overlap")
    return forbidden_ids,forbidden_histories


def validate_calibration(cfg,root=ROOT):
    """Reject stale parameters, corpus bytes or controller source before a run.

    Overrides for a prespecified sensitivity experiment must be applied only
    after this validation of the base protocol. Never silently refit on tests.
    """
    if cfg.get("mindset",{}).get("profile")!="compact_v2": return
    if not cfg.get("calibration"):
        raise ValueError("MINDSET requires an independent calibration artifact")
    artifact=read_json(root/cfg["calibration"]["artifact"])
    sources=artifact.get("controller_dependency_sha256",{})
    if set(sources)!=set(CONTROLLER_SOURCES) or any(digest_file(root/p)!=h for p,h in sources.items()):
        raise ValueError("Controller dependencies changed or are unpinned; recalibrate in a fresh directory")
    if artifact["controller_source_sha256"]!=digest_file(root/"src/mindset/methods/compact.py"):
        raise ValueError("Controller source changed since calibration; recalibrate in a fresh directory")
    if artifact["dataset_sha256"]!=digest_file(root/artifact["dataset_path"]):
        raise ValueError("Development corpus changed since calibration")
    if artifact["encoder"]!=read_json(root/"configs/checkpoints.lock.json")["encoder"]:
        raise ValueError("Development encoder differs from the frozen benchmark encoder")
    if any(cfg["mindset"].get(k)!=v for k,v in artifact["selected_parameters"].items()):
        raise ValueError("Configured parameters differ from independently calibrated settings")

def calibrate(output,per_family=8,dataset_path="data/development/mindset_v2.json",corpus_path=None):
    import os
    os.environ.setdefault("HF_HOME",str(ROOT/".cache/huggingface"))
    from sentence_transformers import SentenceTransformer
    cfg=config(); lock=read_json(ROOT/"configs/checkpoints.lock.json")
    encspec=lock["encoder"]
    encoder=SentenceTransformer(encspec["checkpoint"],revision=encspec["revision"],device="cpu")
    corpus=read_json(corpus_path) if corpus_path else development_corpus(cfg["seed"],per_family)
    forbidden_ids,forbidden_histories=check_disjoint(corpus,cfg)
    dataset=ROOT/dataset_path; write_json(dataset,corpus,immutable=True)
    histories=corpus["histories"]
    events=[ConversationEvent(**e) for h in histories for e in h["events"]]
    from mindset.methods.mindset import _render_event
    vectors=encoder.encode([_render_event(e) for e in events],batch_size=64,
                            normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
    embedding_map={e.event_id:v for e,v in zip(events,vectors)}
    grid=cfg["parameter_sweep"]; results=[]
    for values in itertools.product(*grid.values()):
        params=dict(zip(grid,values)); truth=[]; predicted=[]; schema_counts=[]
        for h in histories:
            if h["split"]!="validation": continue
            m=CompactMindsetMemory(CompactConfig(**params),precomputed_embeddings=embedding_map)
            m.ingest([ConversationEvent(**e) for e in h["events"]])
            truth.extend(h["actions"]); predicted.extend(r.operation for r in m.operation_log.all())
            schema_counts.append(len(m.schemas))
        results.append({**params,"transition_macro_f1":float(f1_score(truth,predicted,labels=ACTION_ORDER,
            average="macro",zero_division=0)),"operation_accuracy":float(np.mean(np.asarray(truth)==predicted)),
            "mean_schemas":float(np.mean(schema_counts)),"split":"validation"})
    chosen=max(results,key=lambda r:(r["transition_macro_f1"],-r["mean_schemas"],
                                      -r["complexity_penalty"],r["switch_margin"]))
    selected={k:chosen[k] for k in grid}
    truth=[]; predicted=[]; family_scores={}
    for h in histories:
        if h["split"]!="holdout": continue
        m=CompactMindsetMemory(CompactConfig(**selected),precomputed_embeddings=embedding_map)
        m.ingest([ConversationEvent(**e) for e in h["events"]])
        labels=[r.operation for r in m.operation_log.all()]
        truth.extend(h["actions"]); predicted.extend(labels)
        family_scores.setdefault(h["family"],[]).extend(int(a==b) for a,b in zip(h["actions"],labels))
    # Oracle forcing supplies labels independent of the fitted energy controller.
    training=[]
    for h in histories:
        if h["split"]!="train": continue
        expected={"action":None}
        oracle=lambda signals,config:(expected["action"],False)
        m=CompactMindsetMemory(CompactConfig(**selected),precomputed_embeddings=embedding_map,
                              transition_controller=oracle)
        for event_data,action in zip(h["events"],h["actions"]):
            expected["action"]=action; event=ConversationEvent(**event_data); m.ingest([event])
            if m.operation_log.all()[-1].operation!=action:
                raise ValueError("Ground-truth probe is incompatible with declared shared feasibility constraints")
            training.append({"id":event.event_id,"source_cluster_id":h["id"],"source_history_sha256":
                h["history_content_sha256"],"action":action,
                "features":asdict(m.last_features) if m.last_features else {k:0. for k in FEATURES}})
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    training_file=out/"independent_training.json"
    write_json(training_file,{"label_source":corpus["label_source"],"records":training},immutable=True)
    fit_controller(training_file,out/"learned_controller.json",forbidden_ids,forbidden_histories)
    # Evaluate alternatives honestly on the same held-out traces; shared
    # feasibility rules may already explain easy synthetic examples.
    from mindset.acl.controllers import rule_controller,nearest_controller,LearnedController
    comparator_scores={}
    for name,policy in {"rule":rule_controller,"nearest":nearest_controller,
                        "learned":LearnedController(out/"learned_controller.json")}.items():
        expected=[]; observed=[]
        for h in histories:
            if h["split"]!="holdout": continue
            m=CompactMindsetMemory(CompactConfig(**selected),precomputed_embeddings=embedding_map,
                                  transition_controller=policy)
            m.ingest([ConversationEvent(**e) for e in h["events"]])
            expected.extend(h["actions"]); observed.extend(r.operation for r in m.operation_log.all())
        comparator_scores[name]={"operation_accuracy":float(np.mean(np.asarray(expected)==observed)),
            "macro_f1":float(f1_score(expected,observed,labels=ACTION_ORDER,average="macro",zero_division=0))}
    ablation_scores={}
    for ablation in ("no_hysteresis","no_complexity","no_context","no_provenance"):
        expected=[]; observed=[]; counts=[]
        for h in histories:
            if h["split"]!="holdout": continue
            m=CompactMindsetMemory(CompactConfig(**selected,ablation=ablation),precomputed_embeddings=embedding_map)
            m.ingest([ConversationEvent(**e) for e in h["events"]])
            expected.extend(h["actions"]); observed.extend(r.operation for r in m.operation_log.all())
            counts.append(len(m.schemas)); m.validate_invariants()
        ablation_scores[ablation]={"operation_accuracy":float(np.mean(np.asarray(expected)==observed)),
            "macro_f1":float(f1_score(expected,observed,labels=ACTION_ORDER,average="macro",zero_division=0)),
            "mean_schemas":float(np.mean(counts)),
            "scope":"transition_only; no_provenance affects retrieval not writes"}
    artifact={"method":"mindset","method_version":"v2","selected_parameters":selected,"encoder":encspec,
        "label_source":corpus["label_source"],
        "controller_dependency_sha256":{p:digest_file(ROOT/p) for p in CONTROLLER_SOURCES},
        "calibration_source_sha256":digest_file(ROOT/"src/mindset/acl/calibration.py"),
        "dataset_path":dataset.relative_to(ROOT).as_posix(),
        "dataset_sha256":digest_file(dataset),"controller_source_sha256":digest_file(ROOT/"src/mindset/methods/compact.py"),
        "selection":"validation_transition_macro_f1_then_fewer_schemas; holdout_not_used_for_selection",
        "synthetic_holdout_macro_f1":float(f1_score(truth,predicted,labels=ACTION_ORDER,average="macro",zero_division=0)),
        "synthetic_holdout_operation_accuracy":float(np.mean(np.asarray(truth)==predicted)),
        "family_operation_accuracy":{k:float(np.mean(v)) for k,v in family_scores.items()},
        "comparator_holdout":comparator_scores,"ablation_holdout":ablation_scores,
        "history_ids_by_split":{split:[h["id"] for h in histories if h["split"]==split]
                                for split in ("train","validation","holdout")},
        "claim_scope":"synthetic development only; natural QA and OOD efficacy untested"}
    write_json(out/"calibration.json",artifact,immutable=True)
    write_csv(out/"parameter_grid.csv",results)
    print("Independent development calibration:",selected,flush=True)
    return artifact
