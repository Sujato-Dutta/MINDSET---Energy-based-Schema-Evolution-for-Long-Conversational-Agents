"""Synthetic audit/arithmetic tests, not empirical performance results."""
import importlib.util
from pathlib import Path
import pytest
from mindset.acl.io import append_row,read_json
from mindset.methods.transition_packets import ARMS

spec=importlib.util.spec_from_file_location('transition_answer_worker',
    Path(__file__).resolve().parents[1]/'scripts/answer_transition_packets.py')
worker=importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)


def panel(out,drop=None):
    planned=[]
    for dataset in ('locomo','longmemeval'):
        for arm in ARMS:
            for i in range(20):
                key=f'{dataset}/{arm}/{i}'; planned.append({'id':key})
                if key==drop: continue
                changed=arm=='compact_energy' and i==0
                append_row(out/'predictions.jsonl',{'id':key,'dataset':dataset,'method':arm,'question_id':str(i),
                    'native_f1':.3 if changed else .2,'token_f1':.3 if changed else .2,
                    'retrieval':{m:.5 for m in worker.METRICS},'reused_prior_artifact':not changed,
                    'completion_cache_hit':not changed,'incomplete_response':False,'input_tokens':10,'output_tokens':2,
                    'prompt_sha256':'new' if changed else 'same','question':'synthetic question',
                    'answer':'synthetic reference','prediction':'synthetic output','certificate':{}})
    return planned


def test_complete_pairing_correct_denominator_and_energy_ablation(tmp_path):
    planned=panel(tmp_path); worker.summarize(tmp_path,planned)
    result=read_json(tmp_path/'analysis.json')
    assert result['usage']['completed_cells']==160
    assert result['usage']['fresh_physical_answer_calls']==2
    assert result['usage']['new_recorded_input_tokens']==20
    for dataset in ('locomo','longmemeval'):
        entry=result['paired'][dataset]['compact_energy']
        assert entry['mean_differences']['secondary_token_f1']==pytest.approx(.1/20)
        assert len(entry['changed_cases'])==1
        assert result['paired'][dataset]['energy_vs_no_energy']['token_f1_difference']==pytest.approx(.1/20)


def test_partial_completion_never_claims_final_panel(tmp_path):
    planned=panel(tmp_path,'longmemeval/compact_energy/19'); worker.summarize(tmp_path,planned)
    result=read_json(tmp_path/'status.json')
    assert result['status']=='partial_development_answers' and result['remaining_cells']==1
    assert not (tmp_path/'analysis.json').exists()


def test_duplicate_saved_answers_rejected(tmp_path):
    planned=panel(tmp_path); saved={'id':'locomo/joint/0'}
    append_row(tmp_path/'predictions.jsonl',saved)
    with pytest.raises(ValueError,match='Duplicate'): worker.summarize(tmp_path,planned)
