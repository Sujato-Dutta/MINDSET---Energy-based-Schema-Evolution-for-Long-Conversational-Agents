"""Independent longer synthetic histories for the isolated joint prototype."""
from dataclasses import asdict
from pathlib import Path
import itertools
import numpy as np
from sklearn.metrics import f1_score
from mindset.acl.calibration import development_corpus,check_disjoint,ACTION_ORDER,CONTROLLER_SOURCES,content_hash
from mindset.acl.io import read_json,write_json,digest_file
from mindset.acl.pipeline import ROOT,config,local_stack,write_csv
from mindset.methods.base import ConversationEvent
from mindset.methods.compact import CompactConfig
from mindset.methods.joint import JointMindsetMemory

SOURCES=(*CONTROLLER_SOURCES,"src/mindset/methods/joint.py","src/mindset/acl/joint_calibration.py")

def long_corpus():
    corpus=development_corpus(seed=41,per_family=4)
    for h in corpus["histories"]:
        original=h["events"]; family=h["family"]
        if family=="repeat": indices=[0]*24; actions=["new"]+["reinforce"]*23
        elif family=="update": indices=[0]*8+[1]+[2]*15; actions=["new"]+["reinforce"]*7+["supersede"]+["reinforce"]*15
        elif family=="coexist": indices=[0,1]*12; actions=["new","split"]+["reinforce"]*22
        elif family=="noise": indices=[0]*8+[1]+[2]*15; actions=["new"]+["reinforce"]*23
        elif family=="confirmation": indices=[0]*8+[1]+[2]*15; actions=["new"]+["reinforce"]*8+["supersede"]+["reinforce"]*14
        else: indices=[0]*8+[1]*16; actions=["new"]+["reinforce"]*7+["new"]+["reinforce"]*15
        # Domain-specific aliases held out by split; action labels are from the
        # declarative world trajectories, never from a controller's decisions.
        texts={"train":("The telescope photographed Saturn.","A cake needs flour and butter."),
               "validation":("The aquarium contains tropical fish.","A bicycle has wheels and pedals."),
               "holdout":("The orchestra rehearsed a symphony.","A computer executes software instructions.")}
        events=[]
        for j,i in enumerate(indices):
            value=dict(original[i]); value.update(event_id=f"{h['id']}-long-e{j}",
                session_id=f"{h['id']}-long-s{j}",timestamp=f"2031-02-{j+1:02d}")
            if family=="new_topic": value["text"]=texts[h["split"]][int(i>0)]
            events.append(ConversationEvent(**value))
        h.update(events=[asdict(e) for e in events],actions=actions,history_content_sha256=content_hash(events))
    corpus.update(generator="joint-long-development-v1",seed=41,
                  limitations="24-turn canonical synthetic histories; small domain holdout; not natural-dialogue OOD proof")
    return corpus

def score(corpus,parameters,vectors,split,joint_targets=True):
    truth=[]; observed=[]; schemas=[]; invariants=[]
    for h in corpus["histories"]:
        if h["split"]!=split: continue
        m=JointMindsetMemory(CompactConfig(**parameters),precomputed_embeddings=vectors,joint_targets=joint_targets)
        m.ingest([ConversationEvent(**e) for e in h["events"]])
        truth.extend(h["actions"]); observed.extend(r.operation for r in m.operation_log.all())
        schemas.append(len(m.schemas)); invariants.append(m.validate_invariants())
    return {"macro_f1":float(f1_score(truth,observed,labels=ACTION_ORDER,average="macro",zero_division=0)),
            "operation_accuracy":float(np.mean(np.asarray(truth)==observed)),"mean_schemas":float(np.mean(schemas)),
            "histories":len(schemas),"all_grounded":all(r["grounded_episode_count"]==24 for r in invariants)}

def calibrate_joint(output):
    out=Path(output); out.mkdir(parents=True,exist_ok=True); cfg=config(); corpus=long_corpus()
    check_disjoint(corpus,cfg)
    write_json(out/"synthetic_long.json",corpus,immutable=True)
    encoder,_=local_stack(cfg,cache=False)
    # This is calibration only, not a latency benchmark. Precompute once.
    events=[ConversationEvent(**e) for h in corpus["histories"] for e in h["events"]]
    vectors=encoder.encode([f"[{e.timestamp}] {e.speaker}: {e.text}" for e in events],
        batch_size=64,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
    embeddings={e.event_id:v for e,v in zip(events,vectors,strict=True)}
    results=[]
    for complexity,margin in itertools.product((.10,.25,.40),(.05,.10,.20)):
        params={"complexity_penalty":complexity,"switch_margin":margin}
        results.append({**params,**score(corpus,params,embeddings,"validation")})
    best=min(results,key=lambda r:(-r["macro_f1"],r["mean_schemas"],-r["complexity_penalty"],r["switch_margin"]))
    chosen={k:best[k] for k in ("complexity_penalty","switch_margin")}
    artifact={"method":"mindset","implementation":"v2_joint_prototype","selected_parameters":chosen,
        "parameter_selection":"synthetic validation macro F1, fewer schemas, larger lambda, smaller margin; never benchmark scores",
        "synthetic_dataset_sha256":digest_file(out/"synthetic_long.json"),
        "dependencies":{p:digest_file(ROOT/p) for p in SOURCES},
        "encoder":read_json(ROOT/"configs/checkpoints.lock.json")["encoder"],
        "holdout":score(corpus,chosen,embeddings,"holdout"),
        "preselected_target_holdout":score(corpus,chosen,embeddings,"holdout",joint_targets=False),
        "ablation_holdout":{a:score(corpus,{**chosen,"ablation":a},embeddings,"holdout")
                            for a in ("no_complexity","no_context","no_hysteresis")},
        "claim_scope":"synthetic controller validation only; no QA or natural-language OOD conclusion"}
    write_json(out/"calibration.json",artifact,immutable=True); write_csv(out/"parameter_grid.csv",results)
    print("Joint prototype independent calibration:",chosen,artifact["holdout"],flush=True)
    return artifact

def validate_joint(path):
    path=Path(path); artifact=read_json(path)
    if set(artifact["dependencies"])!=set(SOURCES) or any(digest_file(ROOT/p)!=h for p,h in artifact["dependencies"].items()):
        raise ValueError("Joint prototype/calibration changed; use a fresh calibration")
    if digest_file(path.with_name("synthetic_long.json"))!=artifact["synthetic_dataset_sha256"]:
        raise ValueError("Synthetic development data changed")
    if read_json(ROOT/"configs/checkpoints.lock.json")["encoder"]!=artifact["encoder"]:
        raise ValueError("Calibration encoder changed")
    return artifact
