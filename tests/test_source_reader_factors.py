"""Synthetic source/reader isolation tests, not empirical metric assertions."""
from dataclasses import replace
from types import SimpleNamespace as N
import pytest
from mindset.acl.source_reader_factors import ARMS,build_plan
from mindset.baselines.event_rag import render_event
from test_compact import event


class Tokenizer:
    def encode(self,text,**kwargs): return list(text)
    def decode(self,tokens): return ''.join(tokens)


def panel():
    cases={}; captions={}; rows=[]
    for i in range(50):
        eid=f'q{i}'; e=event(1,'I live in Paris.')
        case=N(id=eid,cluster='c',question='Where do I live?',question_date='',answer='SECRETREFERENCE',
            category='temporal',events=[e])
        cases[('locomo',eid)]=case
        captions[('locomo',eid)]=N(events=[replace(e,text=e.text+'\nShared image caption: a blue house')])
        text=render_event(e)
        selected=[{'event_id':e.event_id,'session_id':e.session_id,'timestamp':e.timestamp,
            'speaker':e.speaker,'text':text,'token_start':0,'token_end':len(text)}]
        rows.append({'dataset':'locomo','method':'joint','question_id':eid,'selected':selected,
            'evidence':'[1] '+text,'evidence_tokens':len(text)+4})
    return rows,cases,captions,Tokenizer(),{'retrieval':{'chunk_tokens':768,'evidence_tokens':8000}}


def test_complete_four_factor_panel_fixed_ids_no_reference_leak():
    planned=build_plan(*panel())
    assert len(planned)==200 and len({p['id'] for p in planned})==200
    for arm in ARMS: assert sum(p['method']==arm for p in planned)==50
    assert all([s['event_id'] for s in p['selected']]==['e1'] for p in planned)
    assert all('SECRETREFERENCE' not in str(p['messages']) for p in planned)


def test_source_and_reader_factors_are_independent():
    planned=[p for p in build_plan(*panel()) if p['question_id']=='q0']; by={p['method']:p for p in planned}
    assert by['text_generic']['evidence']==by['text_phrase']['evidence']
    assert by['caption_generic']['evidence']==by['caption_phrase']['evidence']
    assert 'Shared image caption' not in by['text_generic']['evidence']
    assert 'Shared image caption: a blue house' in by['caption_generic']['evidence']
    assert 'short phrase' in by['text_phrase']['messages'][0]['content']
    assert 'DATE of CONVERSATION' in by['text_phrase']['messages'][0]['content']
    assert by['text_generic']['messages'][0]['role']=='system'
    assert by['caption_generic']['evidence_tokens']>by['text_generic']['evidence_tokens']


def test_incomplete_or_duplicate_questions_rejected():
    args=panel(); args[0].pop()
    with pytest.raises(ValueError,match='LC50'): build_plan(*args)
    args=panel(); args[0].append(args[0][0])
    with pytest.raises(ValueError,match='Duplicate'): build_plan(*args)


def test_partial_original_span_rejected_not_silently_expanded():
    args=panel(); args[0][0]['selected'][0]['token_start']=1
    with pytest.raises(ValueError,match='complete original'): build_plan(*args)


def test_caption_over_budget_rejected():
    args=panel(); args[-1]['retrieval']['evidence_tokens']=1
    with pytest.raises(ValueError,match='exceeds'): build_plan(*args)
