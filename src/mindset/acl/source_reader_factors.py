"""Shared LoCoMo evaluation factors, explicitly NOT a method contribution."""
from mindset.acl.clients import messages
from mindset.acl.locomo_protocol_diagnostic import short_phrase_messages,caption_packet
from mindset.utils.hashing import stable_hash
from mindset.baselines.event_rag import render_event

ARMS=('text_generic','text_phrase','caption_generic','caption_phrase')


def build_plan(rows,cases,captioned,tokenizer,cfg):
    wanted={('locomo',c.id) for c in cases.values()}
    source_rows=[r for r in rows if r['dataset']=='locomo' and r['method']=='joint']
    baseline={('locomo',r['question_id']):r for r in source_rows}
    if len(baseline)!=len(source_rows): raise ValueError('Duplicate baseline questions')
    if set(baseline)!=wanted or len(wanted)!=50: raise ValueError('Original complete LC50 required')
    planned=[]
    for key in sorted(wanted):
        case=cases[key]; row=baseline[key]
        events={e.event_id:e for e in case.events}
        for s in row['selected']:
            tokens=tokenizer.encode(render_event(events[s['event_id']]),add_special_tokens=False)
            if s['token_start']!=0 or s['token_end']!=len(tokens) or tokenizer.decode(tokens)!=s['text']:
                raise ValueError('Caption-only factor requires complete original episodes')
        for arm in ARMS:
            native=arm.startswith('caption')
            packet=caption_packet(row,captioned[key],tokenizer,chunk_limit=cfg['retrieval']['chunk_tokens'],
                packet_limit=cfg['retrieval']['evidence_tokens']) if native else row
            if [r['event_id'] for r in packet['selected']]!=[r['event_id'] for r in row['selected']]:
                raise ValueError('Source factor changed retrieval')
            prompt=short_phrase_messages(case.question,packet['evidence'],case.question_date,category=case.category) if arm.endswith('phrase') else messages(
                case.question,packet['evidence'],case.question_date)
            planned.append({'id':f'locomo/{arm}/{case.id}','dataset':'locomo','method':arm,
                'question_id':case.id,'cluster':case.cluster,'category':case.category,'question':case.question,
                'question_date':case.question_date,'answer':case.answer,'messages':prompt,
                'prompt_sha256':stable_hash(prompt),'selected':packet['selected'],'evidence':packet['evidence'],
                'evidence_tokens':len(tokenizer.encode(packet['evidence'],add_special_tokens=False)),
                'source_factor':'native_captions' if native else 'text_only',
                'reader_factor':'published_short_phrase' if arm.endswith('phrase') else 'current_generic',
                'claim_scope':'shared source/reader diagnosis, not method novelty or retrieval improvement'})
    return planned
