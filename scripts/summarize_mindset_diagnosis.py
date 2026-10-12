"""Summarize measured mechanism, budget and legacy-split diagnostics on DGX."""
import os
import sys
import argparse
from pathlib import Path
if os.name=="nt" or not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("DGX Slurm only")
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from mindset.acl.io import read_json,write_json,digest_file,read_rows


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output",required=True)
    args=parser.parse_args()
    source=ROOT/"outputs/acl/diagnosis_legacy_v2_27053"
    structure=read_json(source/"energy_diagnosis.json"); energy=read_json(source/"energy_summary.json")
    dev_path=ROOT/"diagnostics_inputs/dev_set.json"; primary_path=ROOT/"diagnostics_inputs/locomo_primary700.json"
    dev=read_json(dev_path); primary=read_json(primary_path)
    dev_ids={r["id"] for r in dev["records"]}; primary_ids={r["id"] for r in primary["records"]}
    dev_clusters={r["sample_id"] for r in dev["records"]}; primary_clusters={r["sample_id"] for r in primary["records"]}
    frozen=ROOT/"diagnostics_inputs/legacy_runs"
    full=frozen/"locomo_mindset_full_gpt_oss_120b_20260717T051244Z"
    base={r["id"]:r for r in read_rows(full/"predictions.jsonl")}
    base_scores={r["id"]:r["score"] for r in read_json(full/"metrics.json")["per_example"]}
    answer_changes=[]
    for folder in sorted(frozen.glob("locomo_mindset_*")):
        if folder==full: continue
        compared=read_rows(folder/"predictions.jsonl")
        scores={r["id"]:r["score"] for r in read_json(folder/"metrics.json")["per_example"]}
        same=[r for r in compared if r["id"] in base and r["retrieved_ids"]==base[r["id"]]["retrieved_ids"]
              and r["evidence_packet_bytes"]==base[r["id"]]["evidence_packet_bytes"]]
        answer_changes.append({"run":folder.name,"same_ordered_ids_and_packet_bytes":len(same),
            "prediction_changed_despite_same_ids_and_bytes":sum(r["prediction"]!=base[r["id"]]["prediction"] for r in same),
            "f1_changed_despite_same_ids_and_bytes":sum(abs(scores[r["id"]]-base_scores[r["id"]])>1e-12 for r in same),
            "limitation":"Same own evidence IDs/bytes is not proof of identical full multi-question API prompts; batching/provider variation may contribute"})
    value={"scope":"diagnostic_development_only_not_qa_or_significance_claim","qa_calls":0,
        "production_changed":False,"episodes":energy["episodes"],"noninitial_decisions":energy["noninitial_decisions"],
        "v1_total_schemas":sum(r["v1_schema_count"] for r in structure),
        "v2_total_schemas":sum(r["v2_schema_count"] for r in structure),
        "discourse_fraction":energy["discourse_episodes"]/energy["episodes"],
        "no_context_fraction":energy["episodes_without_context"]/energy["episodes"],
        "target_preselection_suboptimal_fraction":energy["lower_energy_other_target"]/energy["noninitial_decisions"],
        "complexity_changes_local_action_fraction":energy["complexity_changes_local_action"]/energy["noninitial_decisions"],
        "hysteresis_changes_local_action":energy["hysteresis_changes_local_action"],
        "energy_counts":energy,
        "legacy_same_evidence_answer_changes":answer_changes,
        "legacy_development_overlap":{"dev_questions":len(dev_ids),"test_questions":len(primary_ids),
            "shared_question_ids":len(dev_ids&primary_ids),"shared_conversations":sorted(dev_clusters&primary_clusters),
            "development_conversations":len(dev_clusters),"test_conversations":len(primary_clusters),
            "dev_manifest_sha256":digest_file(dev_path),"primary_manifest_sha256":digest_file(primary_path),
            "conclusion":"Question-disjoint is not conversation/history-disjoint; manifest alone cannot establish pre-test tuning chronology"},
        "source_sha256":{p.name:digest_file(p) for p in (source/"energy_diagnosis.json",source/"energy_summary.json",source/"replay_summary.csv")}}
    write_json(args.output,value,immutable=True)
    print(value,flush=True)

if __name__=="__main__": main()
