"""Source-only scope measurement contract, NOT a novel memory controller.

Literal quotes can be verified mechanically; semantic labels cannot. Labels are
provisional observations, never world truth or automatically accepted updates.
"""
from mindset.utils.hashing import stable_hash
import json

FRAME={
    'type':'object','additionalProperties':False,
    'properties':{
        'source_id':{'type':'string','enum':['A','B']},
        'holder_quote':{'type':'string'},'predicate_quote':{'type':'string'},'value_quote':{'type':'string'},
        'context_quotes':{'type':'array','items':{'type':'string'}},
        'time_quotes':{'type':'array','items':{'type':'string'}},
        'canonical_relation':{'type':'string'},
        'modality':{'type':'string','enum':['asserted','planned','uncertain','quoted','unknown']},
        'value_mode':{'type':'string','enum':['explicit_exact_count','binary_presence','set_member','unknown']}},
    'required':['source_id','holder_quote','predicate_quote','value_quote','context_quotes','time_quotes',
                'canonical_relation','modality','value_mode']}
SCHEMA={'type':'object','additionalProperties':False,
    'properties':{'frames':{'type':'array','items':FRAME,'minItems':2,'maxItems':2},
        'relationship':{'type':'string','enum':['compatible','incompatible','underdetermined']}},
    'required':['frames','relationship']}

SYSTEM="""Measure the scope of two source reports, not the truth of the world.
Return the supplied JSON schema exactly. Produce one frame for A and one for B.
holder_quote: the literal subject/possessor span; I means the given speaker,
while My brother and My sister are different holders. For My budget/My weight,
My is the holder and budget/weight is the property, not a new person.
predicate_quote and value_quote: literal predicate/property and value spans.
context_quotes/time_quotes: literal scope phrases, including dates; do not invent
normalized dates. An empty string/list means the source does not resolve a field.
canonical_relation: a short provisional semantic index label, NOT a source fact.
modality: an assertion, a plan/hope, uncertainty, quotation or unresolved status.
value_mode: exact count only with explicit exactness; binary presence for a stated
presence/absence; set_member for nonexclusive likes/possessions; otherwise unknown.
relationship asks whether BOTH reports can be true respecting their stated holder,
context, time and modality. Different liked objects, contexts, years, holders or
compatible plans are not contradictions. Do not infer exclusive values without
support. Same-scope positive/negative assertions or incompatible exact counts
can conflict. Under unresolved scope return underdetermined, not a forced update.
All quote fields must be exact contiguous substrings of their own source report.
No explanation, answer to a benchmark question, summary, or state update."""


def scope_messages(left,right):
    return [{'role':'system','content':SYSTEM},
        {'role':'user','content':json.dumps({'reports':[{'source_id':'A','speaker':'user','observation_time':'2025-01-01','text':left},
            {'source_id':'B','speaker':'user','observation_time':'2025-01-01','text':right}],
            'output_schema':SCHEMA},ensure_ascii=False)}]


def validate_measurement(payload,left,right):
    """Verify structure/literal provenance only. This does NOT verify meaning."""
    if not isinstance(payload,dict) or set(payload)!={'frames','relationship'}: raise ValueError('Invalid scope payload')
    if payload['relationship'] not in SCHEMA['properties']['relationship']['enum']: raise ValueError('Unknown relation')
    frames=payload['frames']
    if not isinstance(frames,list) or len(frames)!=2: raise ValueError('Two source frames required')
    if any(not isinstance(f,dict) or set(f)!=set(FRAME['required']) for f in frames): raise ValueError('Invalid frame fields')
    if {f['source_id'] for f in frames}!={'A','B'}: raise ValueError('Missing/duplicate source frame')
    sources={'A':left,'B':right}; spans=[]
    for f in frames:
        source=sources[f['source_id']]; verified={}
        for key in ('holder_quote','predicate_quote','value_quote','context_quotes','time_quotes'):
            quotes=f[key] if key.endswith('quotes') else [f[key]]
            if not isinstance(quotes,list) or any(not isinstance(q,str) for q in quotes): raise ValueError('Malformed quotes')
            verified[key]=[]
            for q in quotes:
                if not q: continue
                a=source.find(q)
                if a<0: raise ValueError('Generated or cross-source quote')
                verified[key].append([a,a+len(q)])
        for key in ('modality','value_mode'):
            if f[key] not in FRAME['properties'][key]['enum']: raise ValueError('Invalid frame label')
        if not isinstance(f['canonical_relation'],str): raise ValueError('Invalid provisional relation label')
        spans.append({'source_id':f['source_id'],'verified_quote_spans':verified})
    return {'literal_provenance_verified':True,'semantic_correctness_verified':False,
            'world_truth_or_update_verified':False,'spans':spans}

CONTRACT_HASH=stable_hash({'system':SYSTEM,'schema':SCHEMA,'observation_time':'2025-01-01'})
