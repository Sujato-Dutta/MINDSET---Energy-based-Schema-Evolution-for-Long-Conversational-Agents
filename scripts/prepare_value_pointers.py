"""Pinned source-only value inventory and literal functional checks on DGX."""
import os
import sys
import importlib.util
import hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from mindset.acl.io import read_json, write_json, digest_file


def run():
    if os.name=="nt" or not os.environ.get("SLURM_JOB_ID"):raise RuntimeError("DGX Slurm only")
    spec=importlib.util.spec_from_file_location("value_pointing_preflight",ROOT/"src/mindset/memory/value_pointing.py")
    candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
    import spacy
    nlp=spacy.load("en_core_web_sm")
    if spacy.__version__!="3.8.7" or nlp.meta["version"]!="3.8.0":raise ValueError("Parser revision differs")
    out=ROOT/f"outputs/acl/value_pointer_preflight_{os.environ['SLURM_JOB_ID']}"
    out.mkdir(parents=True,exist_ok=False)
    parent=ROOT/"outputs/acl/bound_attribute_contrasts_r01"
    parent_audit=read_json(parent/"audit.json"); manifest=ROOT/"data/manifests/mindset_v2_development100.json"
    if parent_audit["manifest_sha256"]!=digest_file(manifest) or read_json(parent/"status.json")["status"]!="complete_bound_contrasts":
        raise ValueError("Original source scope differs")
    write_json(out/"audit.json",{"manifest_sha256":digest_file(manifest),"source_inputs_sha256":digest_file(parent/"inputs.json"),
        "worker_sha256":digest_file(__file__),"parser_sha256":hashlib.sha256(nlp.to_bytes()).hexdigest(),
        "spacy":spacy.__version__,"parser_meta":nlp.meta,"contract_hash":candidate.CONTRACT_HASH,
        "modules_sha256":{n:digest_file(ROOT/"src/mindset/memory"/n) for n in
            ("value_pointing.py","value_reconstruction.py","source_pointing.py","source_reference_guard.py",
             "source_binding.py","source_binding_candidate.py","report_frames.py")},
        "QA_or_LLM_calls":0,"scope":"source representation only; not semantic accuracy or novel dynamics"},immutable=True)
    functional=[("I volunteer at the shelter.","I am still volunteering at the shelter.","predicate","volunteer","volunteering",True),
        ("I lead four engineers.","I now lead 5 engineers.","number","four","5",False),
        ("I lead five engineers.","I now lead 5 engineers.","number","five","5",True)]
    checks=[]
    for i,(a,b,kind,word_a,word_b,equal) in enumerate(functional):
        reports=[{"source_id":s,"speaker":"user","observation_time":date,"text":text}
            for s,date,text in [("A","2025-01-01",a),("B","2025-06-01",b)]]
        docs={r["source_id"]:nlp(r["text"]) for r in reports};reg=candidate.registry(reports,docs)
        picked={}
        for sid,word in (("A",word_a),("B",word_b)):
            matches=[u for u in reg["units"] if u["source_id"]==sid and u["role"]=="value" and u["value_kind"]==kind and u["text"]==word]
            if len(matches)!=1:raise ValueError(f"Missing unique functional value: {i}, {sid}, {word}")
            picked[sid]=matches[0]["id"]
        assertions={u["source_id"]:u["id"] for u in reg["units"] if u["role"]=="assertion" and u["kind"]=="full_report"}
        p={"kind":"value_change","variable":"functional source value","scope":"same","direction":"a_to_b",
            "a_assertion":assertions["A"],"b_assertion":assertions["B"],"a_value":picked["A"],"b_value":picked["B"]}
        decoded=candidate.decode(p,reg,reports,docs)
        if decoded["canonical_values_equal"]!=equal or decoded["update_proposal"]==equal:
            raise ValueError(f"Functional value contrast failed: {i}, {decoded}")
        checks.append({"id":str(i),"reports":reports,"expected_canonical_equal":equal,"decoded":decoded})
    write_json(out/"functional.json",checks,immutable=True)
    cases=read_json(parent/"inputs.json")
    if len(cases)!=53:raise ValueError("Frozen bounded source cases differ")
    prepared=[]
    for case in cases:
        docs={r["source_id"]:nlp(r["text"]) for r in case["reports"]}
        reg=candidate.registry(case["reports"],docs);candidate.validate_registry(reg,case["reports"])
        prepared.append({**case,"registry":reg})
    write_json(out/"inputs.json",prepared,immutable=True)
    write_json(out/"analysis.json",{"status":"complete_value_preflight","cases":len(prepared),"real_parser_checks":len(checks),
        "model_QA_or_hosted_calls":0,"scope":"literal normalization feasibility; no natural semantic accuracy or energy advantage"},immutable=True)
    print(read_json(out/"analysis.json"),flush=True)


if __name__=="__main__":run()
