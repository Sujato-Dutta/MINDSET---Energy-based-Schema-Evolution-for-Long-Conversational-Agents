import pytest
from mindset.methods.base import ConversationEvent
from mindset.acl.locomo_protocol_diagnostic import augment_events,caption_packet,short_phrase_messages,TEMPORAL_HINT
from mindset.acl.development_answers import request_plan
from types import SimpleNamespace as N


class Characters:
    def encode(self,text,**kwargs): return list(map(ord,text))
    def decode(self,tokens): return ''.join(map(chr,tokens))


def source():
    events=[ConversationEvent('e1','c','D1','date','Rae','Look at this!',{'existing':'metadata'}),
            ConversationEvent('e2','c','D1','date','Sam','Nice picture.')]
    raw={'e1':{'speaker':'Rae','text':'Look at this!','blip_caption':'a sunset painting'},
         'e2':{'speaker':'Sam','text':'Nice picture.'}}
    return events,raw


def test_caption_augmentation_is_immutable_native_annotation_only():
    events,raw=source(); new=augment_events(events,raw)
    assert events[0].text=='Look at this!' and events[0].metadata=={'existing':'metadata'}
    assert new[0].text=='Look at this!\nShared image caption: a sunset painting'
    assert new[0].event_id==events[0].event_id and new[0].timestamp==events[0].timestamp
    assert new[0].metadata['native_caption']=='a sunset painting' and new[1] is events[1]


@pytest.mark.parametrize('field,value',[('speaker','Different'),('text','Rewritten source'),('blip_caption',['nontext'])])
def test_changed_or_nontext_native_caption_source_is_rejected(field,value):
    events,raw=source(); raw['e1'][field]=value
    with pytest.raises(ValueError): augment_events(events,raw)


def test_caption_packet_keeps_fixed_sources_order_and_exact_token_spans():
    events,raw=source(); new=augment_events(events,raw)
    selected=[{'event_id':e.event_id,'speaker':e.speaker,'timestamp':e.timestamp,'text':'old','token_start':0,'token_end':1}
              for e in reversed(events)]
    row={'id':'x','selected':selected,'metrics':{'recall@8':1}}
    changed=caption_packet(row,N(events=new),Characters())
    assert [s['event_id'] for s in changed['selected']]==['e2','e1']
    assert changed['metrics']==row['metrics'] and row['selected'][0]['text']=='old'
    for selected in changed['selected']:
        assert selected['token_start']==0 and selected['token_end']==len(selected['text'])
    assert 'a sunset painting' in changed['evidence']
    with pytest.raises(ValueError,match='budget'): caption_packet(row,N(events=new),Characters(),packet_limit=1)
    with pytest.raises(ValueError,match='chunk budget'): caption_packet(row,N(events=new),Characters(),chunk_limit=1)


def test_published_reader_retains_question_and_evidence_without_reference_or_method_hints():
    prompt=short_phrase_messages('When did Rae go?','RAW SOURCE',category='temporal')
    assert prompt[0]['role']=='user' and 'RAW SOURCE' in prompt[0]['content']
    assert 'When did Rae go?'+TEMPORAL_HINT in prompt[0]['content']
    factual=short_phrase_messages('Where?','RAW SOURCE',category='single_hop')
    assert TEMPORAL_HINT not in factual[0]['content']
    assert 'short phrase' in factual[0]['content']


def test_published_request_plan_does_not_use_reference_answers():
    e=N(event_id='e',speaker='Rae',timestamp='date')
    c=N(events=[e],question='Where?',question_date='',answer='SECRET_REFERENCE',cluster='c',category='single_hop')
    row={'dataset':'locomo','method':'flat48','question_id':'q','selected':[{'event_id':'e','speaker':'Rae','timestamp':'date','text':'RAW'}],
         'evidence':'[1] RAW','metrics':{}}
    planned=request_plan([row],{('locomo','q'):c},methods=('flat48',),reader='locomo_short_phrase')
    assert 'SECRET_REFERENCE' not in str(planned[0]['messages']) and planned[0]['answer']=='SECRET_REFERENCE'
    with pytest.raises(ValueError): request_plan([row],{('locomo','q'):c},methods=('flat48',),reader='unknown')
