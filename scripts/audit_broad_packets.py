"""DGX-only audit of changed packets/bindings; never exports answer labels."""
from pathlib import Path
import argparse
import os
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mindset.acl.io import unique_rows,read_json,write_json,digest_file


def main(a):
    run=Path(a.run); rows=list(unique_rows(run/'evidence.jsonl').values())
    if read_json(run/'status.json')['completed_cells']!=546: raise ValueError('Incomplete diagnostic')
    keyed={(r['question_id'],r['arm']):r for r in rows}; primary=[r for r in rows if r['arm']=='energy']
    details=[]
    for row in primary:
        control=keyed[(row['question_id'],'state_off')]
        changed=row['prompt_sha256']!=control['prompt_sha256']
        if not changed and not row['report_certificate'].get('bound_reports'): continue
        witnesses=row['report_certificate']['reserved_ids']
        groups=read_json(run/'structure'/f'longmemeval-{row["cluster"]}-mdl.json')['report_groups']
        matched=[g for g in groups if any(r['eid'] in witnesses for r in g['reports'])]
        details.append({'question_id':row['question_id'],'question':row['question'],'changed_prompt':changed,
            'certificate':row['report_certificate'],'query_mode':row['grounding_diagnostic'].get('broad_query_mode'),
            'bound_reports':row['grounding_diagnostic'].get('bound_report_sequence'),
            'matched_source_predicates':[g['probes'] for g in matched],
            'energy_metrics':row['retrieval'],'control_metrics':control['retrieval'],
            'energy_selected':[{'event_id':s['event_id'],'session_id':s['session_id'],'text':s['text']} for s in row['selected']],
            'control_selected':[{'event_id':s['event_id'],'session_id':s['session_id'],'text':s['text']} for s in control['selected']]})
    write_json(a.output,{'run_sha256':digest_file(run/'evidence.jsonl'),'details':details,
        'scope':'source/question/selection audit; no reference answers; manual interpretation not gold annotation'},immutable=True)
    for r in details:
        print({'id':r['question_id'],'question':r['question'],'changed':r['changed_prompt'],
               'predicates':r['matched_source_predicates'],'certificate':r['certificate'],
               'reports':[(v['value'],v['tense']) for v in r['bound_reports']]},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run',required=True); p.add_argument('--output',required=True); a=p.parse_args()
    if os.name=='nt' or not os.environ.get('SLURM_JOB_ID'): p.error('DGX Slurm only')
    main(a)
