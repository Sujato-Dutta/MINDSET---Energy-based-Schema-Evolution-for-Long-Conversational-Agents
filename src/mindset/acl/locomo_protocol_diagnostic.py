"""Isolated shared reader/source controls, not new memory-method contributions."""
from dataclasses import replace
from mindset.utils.hashing import stable_hash

REFERENCE_COMMIT='3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376'
REFERENCE_URL=f'https://github.com/snap-research/locomo/blob/{REFERENCE_COMMIT}/task_eval/gpt_utils.py'
QA_TEMPLATE='''
Based on the above context, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.
Question: {} Short answer:
'''
TEMPORAL_HINT=' Use DATE of CONVERSATION to answer with an approximate date.'
READER_HASH=stable_hash({'qa_template':QA_TEMPLATE,'temporal_hint':TEMPORAL_HINT,'reference':REFERENCE_URL})
CAPTION_ADAPTER='locomo_native_captions_v1'


def short_phrase_messages(question,evidence,question_date='',*,category):
    # Native category-2 date instruction; no reference or method-specific text.
    question=question+(TEMPORAL_HINT if category=='temporal' else '')
    date=f'Question date: {question_date}\n' if question_date else ''
    return [{'role':'user','content':date+evidence+'\n\n'+QA_TEMPLATE.format(question)}]


def augment_events(events,raw_turns):
    """Append only the public source's native caption; never infer image facts."""
    result=[]
    for event in events:
        raw=raw_turns[event.event_id]
        if event.text!=raw['text'] or event.speaker!=raw['speaker']:
            raise ValueError('Native caption augmentation source mismatch')
        cap=raw.get('blip_caption')
        if cap is not None and not isinstance(cap,str): raise ValueError('Nontext native image caption')
        if not cap or not cap.strip(): result.append(event); continue
        result.append(replace(event,text=event.text+'\nShared image caption: '+cap,
            metadata={**event.metadata,'native_caption':cap,'source_adapter':CAPTION_ADAPTER}))
    return result


def captioned_cases(cases,cfg,root):
    from mindset.acl.io import read_json
    samples={str(r.get('sample_id',i)):r for i,r in enumerate(read_json(root/cfg['benchmarks']['locomo']['path']))}
    histories={}; result={}
    for key,case in cases.items():
        if key[0]!='locomo': result[key]=case; continue
        if case.cluster not in histories:
            raw=samples[case.cluster]['conversation']
            turns={t['dia_id']:t for k,session in raw.items() if k.startswith('session_') and not k.endswith('_date_time') for t in session}
            histories[case.cluster]=augment_events(case.events,turns)
        result[key]=replace(case,events=histories[case.cluster])
    return result


def caption_packet(row,case,tokenizer,*,chunk_limit=768,packet_limit=8000):
    """Content-only control: fixed episode IDs/order; no new retrieval ranking."""
    from mindset.baselines.event_rag import render_event
    original={e.event_id:e for e in case.events}; selected=[]
    for prior in row['selected']:
        event=original[prior['event_id']]
        if event.speaker!=prior['speaker'] or event.timestamp!=prior['timestamp']: raise ValueError('Source attribution changed')
        tokens=tokenizer.encode(render_event(event),add_special_tokens=False)
        if len(tokens)>chunk_limit: raise ValueError('Content-only caption control needs a whole episode within the shared chunk budget')
        selected.append({**prior,'text':tokenizer.decode(tokens),'token_start':0,'token_end':len(tokens)})
    packet='\n\n'.join(f'[{i}] {s["text"]}' for i,s in enumerate(selected,1))
    count=len(tokenizer.encode(packet,add_special_tokens=False))
    if count>packet_limit: raise ValueError('Caption packet exceeds shared evidence budget; no silent episode dropping')
    return {**row,'selected':selected,'evidence':packet,'evidence_tokens':count,
        'source_adapter':CAPTION_ADAPTER,'source_selection':'exact original selected episode IDs/order; captions are source annotations, not reference answers'}
