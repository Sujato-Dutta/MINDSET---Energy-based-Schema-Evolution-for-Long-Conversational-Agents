"""Source-witness invariants; run through DGX Slurm only."""
from dataclasses import replace
import numpy as np
import pytest
from test_compact import event
from mindset.methods.semantic_witness import source_spans,semantic_pairs,classified_edges,extend_pool,choose_packet


def test_spans_are_literal_reports_not_world_state():
    events=[event(1,'  I lived in Paris. I do not live there now. My plan is to return.')]
    spans=source_spans(events)
    assert len(spans)==3
    assert all(events[0].text[s.start:s.end]==s.text for s in spans)
    assert 'not' in spans[1].text and 'plan' in spans[2].text


@pytest.mark.parametrize('text',[
    'She said "I live in Paris. I moved to Rome."',
    'She said \u201cI live in Paris. I moved to Rome.\u201d',
    '```I live in Paris.```', 'Web search results: I live in Paris.',
    'URL: I live in Paris.', 'Do I live in Paris?', 'Sam lives in Paris.'
])
def test_quoted_code_search_and_nonreport_text_not_attributed(text):
    assert not source_spans([event(1,text)])


def test_neighbors_respect_author_and_observation_deadline():
    events=[event(1,'I live in Paris.'), event(2,'I live in Rome.'),
            replace(event(3,'I live in Oslo.'),speaker='Rae'),event(4,'I live in London.')]
    spans=source_spans(events); vectors=np.ones((4,2))
    pairs=semantic_pairs(None,spans,vectors,np.ones(2),['e1','e4'],events,question_date='2025-01-02')
    assert pairs==[(0,1)]
    events[1]=replace(events[1],timestamp='not a date')
    assert not semantic_pairs(None,spans,vectors,np.ones(2),['e1'],events,question_date='2025-01-02')


def test_bidirectional_labels_no_truth_or_transitivity():
    events=[event(1,'I live in Paris.'),event(2,'I live in Rome.')]; spans=source_spans(events)
    edges=classified_edges([(1,0)],spans,['contradiction','contradiction'],events)
    assert edges[0]['left']=='e1' and edges[0]['right']=='e2'
    assert edges[0]['relation']=='contrast' and 'not truth' in edges[0]['label_scope']
    for labels in (['entailment','neutral'],['too_long','contradiction']):
        assert not classified_edges([(0,1)],spans,labels,events)
    with pytest.raises(ValueError): classified_edges([(0,1)],spans,['contradiction'],events)
    with pytest.raises(ValueError): classified_edges([(0,1)],spans,['truth','truth'],events)


def test_pool_preserves_direct24_and_candidate_ceiling():
    pool=[f'e{i}' for i in range(48)]
    new=extend_pool(pool,[{'left':'x','right':'y'}])
    assert new[:24]==pool[:24] and len(new)==48 and len(set(new))==48
    assert new[-2:]==['x','y']
    with pytest.raises(ValueError): extend_pool(['x','x'],[])


class Pack:
    budget=100
    def __init__(self,texts,k=2): self.texts=texts; self.k=k
    def __call__(self,ids):
        selected=[{'event_id':eid,'text':self.texts[eid]} for eid in list(dict.fromkeys(ids))[:self.k]]
        return selected,'\n\n'.join(f'[{i}] {s["text"]}' for i,s in enumerate(selected,1))
    def token_count(self,text): return len(text.split())


def edge(relation='contrast'):
    return {'left':'a','right':'b','left_text':'I live in Paris.','right_text':'I live in Rome.',
            'relation':relation}


def test_cover_and_order_ablation_share_proposals_and_membership():
    pack=Pack({'x':'other','a':'I live in Paris.','b':'I live in Rome.'})
    order={'a':0,'b':1,'x':2}; ranking=['x','b','a']
    full=choose_packet('semantic_cover',ranking,[edge()],'Where do I live now?',pack,order)
    no_order=choose_packet('no_order',ranking,[edge()],'Where do I live now?',pack,order)
    assert full[2]['proposals']==no_order[2]['proposals']
    assert {s['event_id'] for s in full[0]}=={s['event_id'] for s in no_order[0]}=={'a','b'}
    assert full[2]['global_minimality_or_world_truth_claim'] is False


def test_episode_id_alone_not_a_literal_span_witness():
    pack=Pack({'a':'unrelated chunk','b':'I live in Rome.'})
    result=choose_packet('semantic_cover',['a','b'],[edge()],'Where do I live now?',pack,{'a':0,'b':1})
    cert=result[2]['proposals'][result[2]['selected_proposal']]
    assert cert['witness_ids']==['b'] and cert['cost'][0]==1


def test_temporal_reports_never_collapsed_as_redundant():
    pack=Pack({'a':'I live in Paris.','b':'I live in Rome.','x':'other'})
    result=choose_packet('semantic_cover',['a','b','x'],[edge('equivalent_report')],
                         'When did I visit?',pack,{'a':0,'b':1,'x':2})
    assert result[2]['equivalent_report_edges']==0
    assert {s['event_id'] for s in result[0]}=={'a','b'}


def test_ce_controls_do_not_use_semantic_cover_objective():
    pack=Pack({'a':'I live in Paris.','b':'I live in Rome.','x':'other'})
    result=choose_packet('semantic_pool',['x','b','a'],[edge()],'Where do I live now?',pack,{'a':0,'b':1,'x':2})
    assert [s['event_id'] for s in result[0]]==['x','b'] and result[2]['policy']=='CE-only'
