from pathlib import Path
from copy import deepcopy
import pytest
from mindset.acl.matched_source_answers import summarize,prior_answers,METRICS
from mindset.acl.clients import messages,ANSWER_PROMPT_HASH
from mindset.acl.locomo_protocol_diagnostic import short_phrase_messages,READER_HASH
from mindset.acl.io import append_row,read_json,write_json
from mindset.utils.hashing import stable_hash

ARMS=('joint_ref','expanded_joint_ref','flat_ce_seed','hybrid_seed')
CONTRASTS=(('flat_ce_seed','expanded_joint_ref'),('hybrid_seed','flat_ce_seed'))


def panel(out,drop=None):
    planned=[]
    for d in ('locomo','longmemeval'):
        for q in ('a','b'):
            for arm in ARMS:
                rid=f'{d}/{arm}/{q}'; planned.append({'id':rid,'dataset':d,'method':arm,'question_id':q})
                if rid==drop: continue
                changed=arm=='hybrid_seed' and q=='a'
                append_row(out/'predictions.jsonl',{'id':rid,'dataset':d,'method':arm,'question_id':q,'cluster':q,
                    'native_f1':.4 if changed else .2,'token_f1':.4 if changed else .2,
                    'retrieval':{m:.5 for m in METRICS},'logical_ce_pairs':48 if arm=='hybrid_seed' else 96,
                    'reused_parent_artifact':not changed,'completion_cache_hit':not changed,'incomplete_response':False,
                    'input_tokens':10,'output_tokens':2,'prompt_sha256':'new' if changed else 'same',
                    'question':'synthetic','answer':'synthetic','prediction':'synthetic','selected':[{'event_id':'e1'}]})
    return planned


def test_complete_paired_denominators_native_and_secondary_usage(tmp_path):
    summarize(tmp_path,panel(tmp_path),ARMS,CONTRASTS)
    report=read_json(tmp_path/'analysis.json')
    assert report['usage']['fresh_recorded_calls']==2 and report['usage']['fresh_recorded_input_tokens']==20
    assert report['paired']['locomo']['hybrid_seed minus flat_ce_seed']['mean_difference']==pytest.approx(.1)
    assert report['paired']['longmemeval']['hybrid_seed minus flat_ce_seed']['metric']=='token_f1'
    assert all(r['native_judge_accuracy'] is None for r in report['summary'])


def test_partial_panel_and_unexpected_predictions(tmp_path):
    planned=panel(tmp_path,'longmemeval/hybrid_seed/b'); summarize(tmp_path,planned,ARMS,CONTRASTS)
    assert read_json(tmp_path/'status.json')['remaining_cells']==1
    assert not (tmp_path/'analysis.json').exists()
    append_row(tmp_path/'predictions.jsonl',{'id':'unexpected'})
    with pytest.raises(ValueError): summarize(tmp_path,planned,ARMS,CONTRASTS)


def parent(out):
    identity={k:'frozen' for k in ('manifest_sha256','calibration_sha256','checkpoint','checkpoint_lock_sha256',
                                  'generation','seed','model_config_sha256','client_module_sha256','source_sha256')}
    identity['reader_hashes']={'locomo':READER_HASH,'longmemeval':ANSWER_PROMPT_HASH}
    write_json(out/'audit.json',identity); write_json(out/'status.json',{'status':'complete_development_answers'})
    (out/'worker.lock').touch()
    r={'id':'locomo/joint/q','dataset':'locomo','question_id':'q','question':'Where do I live?',
       'evidence':'[1] I live in Paris.','question_date':'2025-01-02','category':'factual','prediction':'Paris'}
    r['prompt_sha256']=stable_hash(short_phrase_messages(r['question'],r['evidence'],r['question_date'],category=r['category']))
    append_row(out/'predictions.jsonl',r)
    return identity,r


def test_parent_exact_prompt_and_reader_identity(tmp_path):
    identity,r=parent(tmp_path); previous,metadata=prior_answers(tmp_path,identity)
    assert (r['dataset'],r['question_id'],r['prompt_sha256']) in previous
    changed=deepcopy(identity); changed['reader_hashes']['locomo']='changed'
    with pytest.raises(ValueError,match='reader_hashes'): prior_answers(tmp_path,changed)


def test_conflicting_identical_parent_requests_rejected(tmp_path):
    identity,r=parent(tmp_path); r.update(id='locomo/other/q',prediction='Rome')
    append_row(tmp_path/'predictions.jsonl',r)
    with pytest.raises(ValueError,match='Conflicting'): prior_answers(tmp_path,identity)


def test_parent_prompt_tampering_rejected(tmp_path):
    identity,r=parent(tmp_path); r.update(id='locomo/other/q',evidence='changed source')
    append_row(tmp_path/'predictions.jsonl',r)
    with pytest.raises(ValueError,match='reconstructed'): prior_answers(tmp_path,identity)
