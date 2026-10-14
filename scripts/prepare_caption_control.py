"""DGX-only fixed-source-ID native-caption control on original LC50."""
import argparse
from dataclasses import asdict
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',required=True); p.add_argument('--output',required=True)
    a=p.parse_args()
    if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): p.error('DGX Slurm only')
    from mindset.acl.io import read_json,write_json,digest_file,unique_rows,append_row
    from mindset.acl.joint_iteration import protocol,groups_for
    from mindset.acl.clients import load_tokenizer
    from mindset.acl.pipeline import record_environment,write_csv
    from mindset.acl.locomo_protocol_diagnostic import CAPTION_ADAPTER,captioned_cases,caption_packet
    from mindset.baselines.event_rag import render_event
    from mindset.acl.development_answers import regression_plan
    from mindset.utils.hashing import stable_hash
    manifest=ROOT/'data/manifests/mindset_v2_development100.json'
    calibration=ROOT/'outputs/acl/mindset_joint_v2_27058/calibration/calibration.json'
    cfg,cohort,_=protocol(manifest,calibration); prepared=Path(a.prepared); out=Path(a.output)
    prior=read_json(prepared/'audit.json')
    if prior['kind']!='retained_v1_regression_diagnosis' or prior.get('source_adapter') is not None:
        raise ValueError('Expected the unchanged text-only LC regression preparation')
    if read_json(prepared/'status.json')['status']!='complete_development': raise ValueError('Partial preparation')
    rows=list(unique_rows(prepared/'retrieval.jsonl').values()); cases={}
    for dataset,cluster,queries in groups_for(cohort,cfg):
        if dataset=='locomo': cases.update({(dataset,c.id):c for c in queries})
    regression_plan(rows,cases,prepared)
    augmented=captioned_cases(cases,cfg,ROOT)
    tokenizer=load_tokenizer(read_json(ROOT/'configs/checkpoints.lock.json')[cfg['evidence_tokenizer']])
    out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (out/'worker.lock').open('a+b') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity={**prior,'source_adapter':CAPTION_ADAPTER,'worker_sha256':digest_file(__file__),
            'adapter_sha256':digest_file(ROOT/'src/mindset/acl/locomo_protocol_diagnostic.py'),
            'parent_prepared_sha256':digest_file(prepared/'retrieval.jsonl'),
            'parent_audit_sha256':digest_file(prepared/'audit.json'),
            'frozen_manifest_sha256':digest_file(manifest),
            'caption_history_sha256':{cluster:stable_hash([asdict(e) for e in next(c for c in augmented.values() if c.cluster==cluster).events])
                                      for cluster in sorted({c.cluster for c in augmented.values()})},
            'factor':'native source captions, fixed selected episode IDs/order; no re-retrieval or coefficient fit',
            'cohort':'original LC50; original manifest and source files untouched','qa_calls':0,'judge_calls':0}
        write_json(out/'audit.json',identity,immutable=True); record_environment(out)
        done=unique_rows(out/'retrieval.jsonl')
        for row in rows:
            if row['id'] in done: continue
            case=cases[('locomo',row['question_id'])]; old_events={e.event_id:e for e in case.events}
            for selected in row['selected']:
                tokens=tokenizer.encode(render_event(old_events[selected['event_id']]),add_special_tokens=False)
                if selected['token_start']!=0 or selected['token_end']!=len(tokens) or tokenizer.decode(tokens)!=selected['text']:
                    raise ValueError('Content-only caption control requires complete original source episodes')
            changed=caption_packet(row,augmented[('locomo',row['question_id'])],tokenizer,
                chunk_limit=cfg['retrieval']['chunk_tokens'],packet_limit=cfg['retrieval']['evidence_tokens'])
            if [r['event_id'] for r in changed['selected']]!=[r['event_id'] for r in row['selected']]:
                raise ValueError('Caption control changed source selection')
            append_row(out/'retrieval.jsonl',changed); done[row['id']]=changed
        if set(done)!={r['id'] for r in rows}: raise ValueError('Caption control changed declared panel')
        regression_plan(list(done.values()),augmented,out)
        report=[]
        for method in prior['methods']:
            sub=[r for r in done.values() if r['method']==method]
            report.append({'method':method,'questions':len(sub),'mean_evidence_tokens':sum(r['evidence_tokens'] for r in sub)/len(sub),
                          'fixed_source_id_order':True,'source_adapter':CAPTION_ADAPTER})
        write_csv(out/'caption_packets.csv',report)
        write_json(out/'status.json',{'status':'complete_development','completed_cells':len(done),'expected_cells':len(rows),
            'unique_questions':50,'qa_calls':0,'judge_calls':0,'full_runs_authorized':False})
        print('CAPTION_CONTROL',report,flush=True)


if __name__=='__main__': main()
