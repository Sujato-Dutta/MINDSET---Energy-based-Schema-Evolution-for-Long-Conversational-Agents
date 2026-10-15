"""Source-value validation and matched conditional controllers; no world truth."""
import argparse
import os
import sys
import importlib.util
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from mindset.acl.io import read_json,write_json,unique_rows,digest_file


def load(name,filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/"src/mindset/memory"/filename)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def expectations(case):
    # Evaluation-only source-constructed labels, never model/controller inputs.
    if "expected_kind" not in case:return None
    index=int(case["id"].split(":")[-1])
    values={0:("oslo","kigali"),1:("7","12"),2:("50","32"),3:("zephyr","nimbus"),
            4:("ferry","train"),5:("300","450"),11:("afternoon","morning")}
    action="revise" if index<6 else "split" if index==11 else "defer"
    return {"action":action,"a_value":values.get(index,(None,None))[0],"b_value":values.get(index,(None,None))[1],
        "direction":case["expected_direction"] if index<6 else None}


def run(out,mode):
    if os.name=="nt" or not os.environ.get("SLURM_JOB_ID"):raise RuntimeError("DGX Slurm only")
    import spacy
    nlp=spacy.load("en_core_web_sm")
    if spacy.__version__!="3.8.7" or nlp.meta["version"]!="3.8.0":raise ValueError("Parser identity differs")
    values=load("value_check_instrument","value_pointing.py")
    controller=load("value_readout_controller","value_reconstruction.py")
    out=Path(out);cases=read_json(out/"inputs.json");raw=unique_rows(out/"measurements.jsonl")
    expected=cases if mode=="complete" else [c for c in cases if "expected_kind" in c]
    if len(raw)!=len(expected) or set(raw)!={c["id"] for c in expected}:raise ValueError("Incomplete/extra measurement cohort")
    records=[]
    for case in expected:
        row=raw[case["id"]]
        if any(row[k]!=v for k,v in case.items()):raise ValueError("Prepared case differs from measured inputs")
        decoded=None;decisions=None;error=None
        try:
            if row["incomplete"] or row["decode_error"] is not None:raise ValueError("Capped or invalid output retained")
            docs={r["source_id"]:nlp(r["text"]) for r in case["reports"]}
            rebuilt=values.registry(case["reports"],docs)
            if rebuilt!=case["registry"]:raise ValueError("Checked parser/source registry identity differs")
            decoded=values.decode(row["prediction"],case["registry"],case["reports"],docs)
            decisions={"full":controller.compare(row["prediction"],decoded),
                "no_history":controller.compare(row["prediction"],decoded,history=False),
                "no_current":controller.compare(row["prediction"],decoded,current=False)}
        except (ValueError,KeyError,TypeError) as exc:error=str(exc)
        record={**case,"prediction":row["prediction"],"decoded":decoded,"decisions":decisions,"error":error,
            "evaluation_expectation":expectations(case),"semantic_correctness_verified":False}
        records.append(record)
    controls=[r for r in records if r["evaluation_expectation"] is not None]
    def endpoint_correct(r):
        e=r["evaluation_expectation"]
        if e["a_value"] is None:return True
        return r["decoded"]["complete_value_witnesses"] and all(
            r["decoded"]["selected"][p]["value"]["canonical"]==e[p+"_value"] for p in ("a","b"))
    positives=[r for r in controls if r["evaluation_expectation"]["action"]=="revise"]
    positive_correct=sum(r["error"] is None and endpoint_correct(r) and
        r["prediction"]["direction"]==r["evaluation_expectation"]["direction"] and
        r["decisions"]["full"]["energy_action"]=="revise" for r in positives)
    false_revision=sum(r["error"] is None and r["evaluation_expectation"]["action"]!="revise" and
        r["decisions"]["full"]["energy_action"]=="revise" for r in controls)
    correct_actions=sum(r["error"] is None and endpoint_correct(r) and
        r["decisions"]["full"]["energy_action"]==r["evaluation_expectation"]["action"] for r in controls)
    natural=[r for r in records if r["evaluation_expectation"] is None]
    valid=[r for r in records if r["error"] is None]
    report={"controls":len(controls),"positive_controls":len(positives),"positive_value_direction_revision_correct":positive_correct,
        "negative_false_revisions":false_revision,"control_invalid":sum(r["error"] is not None for r in controls),
        "control_action_and_required_values_correct":correct_actions,
        "control_gate":len(controls)==12 and len(positives)==6 and positive_correct==6 and correct_actions==12 and false_revision==0,
        "natural_pairs":len(natural),"natural_invalid":sum(r["error"] is not None for r in natural),
        "natural_labels":dict(Counter(r["prediction"]["kind"] for r in natural if r["error"] is None)),
        "natural_actions":dict(Counter(r["decisions"]["full"]["energy_action"] for r in natural if r["error"] is None)),
        "natural_holds":dict(Counter(h for r in natural if r["error"] is None for h in r["decoded"]["holds"])),
        "energy_vs_rule_action_disagreements":sum(r["decisions"]["full"]["energy_action"]!=r["decisions"]["full"]["rule_action"] for r in valid),
        "conditional_reconstruction_evaluated_cases":sum(r["decisions"]["full"]["evaluated"] for r in valid),
        "history_ablation_changed_cases":sum(r["decisions"]["full"]["energy_action"]!=r["decisions"]["no_history"]["energy_action"] for r in valid),
        "current_ablation_changed_cases":sum(r["decisions"]["full"]["energy_action"]!=r["decisions"]["no_current"]["energy_action"] for r in valid),
        "scope":"conditional source-view reconstruction; NOT natural truth, benchmark QA, energy superiority or novelty"}
    target=out/("control_validation" if mode=="controls" else "complete_validation")
    write_json(target/"audit.json",{"checker_sha256":digest_file(__file__),"inputs_sha256":digest_file(out/"inputs.json"),
        "outputs_sha256":digest_file(out/"measurements.jsonl"),"value_module_sha256":digest_file(ROOT/"src/mindset/memory/value_pointing.py"),
        "controller_module_sha256":digest_file(ROOT/"src/mindset/memory/value_reconstruction.py"),
        "expectations":"source-constructed controlled values; contextual case permits split, not temporal replacement",
        "no_labels_passed_to_controller_or_inference":True,"mode":mode},immutable=True)
    write_json(target/"records.json",records,immutable=True);write_json(target/"analysis.json",report,immutable=True)
    print(report,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);p.add_argument("--mode",choices=("controls","complete"),required=True)
    a=p.parse_args();run(a.output,a.mode)
