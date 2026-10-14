"""Audit arithmetic and shared-reader construction, not empirical evidence."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as N
import pytest
from test_compact import event
from mindset.acl.io import append_row,read_json
from mindset.methods.semantic_witness import ARMS

spec=importlib.util.spec_from_file_location('witness_answer_worker',
    Path(__file__).resolve().parents[1]/'scripts/answer_semantic_witness.py')
worker=importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)


def panel(out,drop=None):
    planned=[]
    for dataset in ('locomo','longmemeval'):
        for arm in ARMS:
            for i in range(2):
                key=f'{dataset}/{arm}/{i}'; planned.append({'id':key,'dataset':dataset,'method':arm,'question_id':str(i)})
                if key==drop: continue
                changed=arm=='semantic_cover' and i==0
                append_row(out/'predictions.jsonl',{'id':key,'dataset':dataset,'method':arm,'question_id':str(i),
                    'cluster':str(i),'native_f1':.4 if changed else .2,'token_f1':.4 if changed else .2,
                    'retrieval':{m:.5 for m in worker.METRICS},'reused_prior_artifact':not changed,
                    'completion_cache_hit':not changed,'incomplete_response':False,'input_tokens':10,'output_tokens':2,
                    'prompt_sha256':'changed' if changed else 'same','question':'synthetic question',
                    'answer':'synthetic reference','prediction':'synthetic output','certificate':{},'semantic_edges':[],
                    'selected':[{'event_id':'e1'}]})
    return planned


def test_complete_paired_denominators_component_contrasts_and_actual_usage(tmp_path):
    plan=panel(tmp_path); worker.summarize(tmp_path,plan)
    result=read_json(tmp_path/'analysis.json')
    assert result['usage']['completed_cells']==24 and result['usage']['fresh_recorded_answer_calls']==2
    assert result['usage']['fresh_recorded_input_tokens']==20
    for dataset in ('locomo','longmemeval'):
        contrast=result['paired'][dataset]['semantic_cover minus no_order']
        assert contrast['mean_difference']==pytest.approx(.1)
        assert (contrast['wins'],contrast['ties'],contrast['losses'])==(1,1,0)
        assert contrast['per_cluster_mean_differences']['0']==pytest.approx(.2)
    assert result['summary'][-1]['native_f1'] is None
    assert all(r['native_judge_accuracy'] is None for r in result['summary'])


def test_partial_panel_never_reports_final_ablations(tmp_path):
    worker.summarize(tmp_path,panel(tmp_path,'longmemeval/no_order/1'))
    assert read_json(tmp_path/'status.json')['remaining_cells']==1
    assert not (tmp_path/'analysis.json').exists()


def test_duplicate_predictions_rejected(tmp_path):
    plan=panel(tmp_path); append_row(tmp_path/'predictions.jsonl',{'id':'locomo/joint/0'})
    with pytest.raises(ValueError,match='Duplicate'): worker.summarize(tmp_path,plan)


def fixture():
    cases={}; rows=[]
    for dataset in ('locomo','longmemeval'):
        e=event(1,'I live in Paris.')
        case=N(id='q',cluster='c',category='temporal' if dataset=='locomo' else 'temporal-reasoning',
               question='When did I move?',question_date='2025-01-02',answer='REFERENCE_DO_NOT_PROMPT',events=[e])
        cases[(dataset,'q')]=case
        for arm in ARMS:
            selected=[{'event_id':'e1','speaker':'Sam','timestamp':'2025-01-01','text':'I live in Paris.'}]
            rows.append({'id':f'{dataset}/{arm}/q','dataset':dataset,'question_id':'q','method':arm,
                         'selected':selected,'evidence':'[1] I live in Paris.','metrics':{}})
    return rows,cases


def test_all_arms_share_reader_and_references_excluded():
    rows,cases=fixture(); plan=worker.build_plan(rows,cases)
    assert len(plan)==12
    for dataset in ('locomo','longmemeval'):
        prompts=[p['messages'] for p in plan if p['dataset']==dataset]
        assert all(p==prompts[0] for p in prompts)
        assert 'REFERENCE_DO_NOT_PROMPT' not in str(prompts)
    lc=next(p for p in plan if p['dataset']=='locomo')
    lme=next(p for p in plan if p['dataset']=='longmemeval')
    assert 'short phrase' in str(lc['messages'])
    assert lc['messages']!=lme['messages']


def test_incomplete_or_unknown_arm_panel_rejected():
    rows,cases=fixture()
    with pytest.raises(ValueError): worker.build_plan(rows[:-1],cases)
