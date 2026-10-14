from dataclasses import replace
import pytest
from mindset.methods.report_grounding import PredicateEquivalence, ReportCatalog, inject_witnesses, select_with_witnesses
from mindset.memory.report_frames import normalize_number
from test_compact import event


def fixture(values, *, sessions=None, owner='user', predicate='lead'):
    events=[]; rows={}
    for i, value in enumerate(values, 1):
        text=f'I {predicate} {value} workers.'
        e=replace(event(i,text), speaker='user', session_id=(sessions or {}).get(i,f's{i}'))
        start=text.index(str(value))
        frame={'owner':owner, 'holder_terms':[], 'unit_terms':['worker'], 'contexts':[],
               'probe':f'I {predicate} one worker.', 'normalized_value':str(value), 'value':str(value),
               'value_span':[start,start+len(str(value))], 'tense':'nonpast'}
        events.append(e); rows[e.event_id]={'text':text, 'speaker':'user', 'reports':[frame]}
    return events,rows


def query(probe='I lead one worker.', mode='report_compare'):
    return {'report_query':{'mode':mode,'self_reference':True,
            'probes':[{'unit_terms':['worker'], 'probe':probe}]}}


def test_number_normalization():
    assert normalize_number('twenty-one')=='21'
    assert normalize_number('one hundred and five')=='105'
    assert normalize_number('1,500')=='1500'
    assert normalize_number('about two') is None
    assert normalize_number('-3') is None
    assert normalize_number('NaN') is None


def test_equivalence_requires_both_directions_and_exact_identity():
    a,b='I lead one worker.','I supervise one worker.'
    one=PredicateEquivalence([{'premise':a,'hypothesis':b,'label':'entailment'}])
    assert not one.equivalent(a,b) and one.equivalent(a,a)
    two=PredicateEquivalence([{'premise':a,'hypothesis':b,'label':'entailment'},
                             {'premise':b,'hypothesis':a,'label':'entailment'}])
    assert two.equivalent(a,b)


def test_energy_witnesses_are_actual_committed_raw_report_provenance():
    events,rows=fixture([1]*8+[2]*3)
    catalog=ReportCatalog(rows,events,PredicateEquivalence())
    ids,certificate=catalog.witnesses(query())
    assert ids==['e1','e11'] and certificate['committed_report_edge']
    assert certificate['hard_exclusions']==[] and not certificate['world_truth_certified']
    assert any(a['chosen']['action']=='supersede' for a in catalog.audit)
    assert all(a['chosen']['adjusted_energy']==min(p['adjusted_energy'] for p in a['candidates']) for a in catalog.audit)


def test_no_edge_is_not_misreported_as_energy_transition():
    events,rows=fixture([1]*8+[2])
    catalog=ReportCatalog(rows,events,PredicateEquivalence())
    assert catalog.witnesses(query())[0]==[]
    ids,certificate=catalog.witnesses(query(),selector='chronological')
    assert ids==['e1','e9'] and not certificate['committed_report_edge']


def test_unknown_owner_unit_or_operator_does_not_bind():
    events,rows=fixture([1,2],owner='someone_else')
    assert ReportCatalog(rows,events,PredicateEquivalence()).witnesses(query())[0]==[]
    events,rows=fixture([1,2]); catalog=ReportCatalog(rows,events,PredicateEquivalence())
    assert catalog.witnesses(query(mode='ordinary'))[0]==[]
    assert catalog.witnesses(query('I hire one worker.'))[0]==[]


def test_same_session_spam_is_not_independent_confirmation():
    events,rows=fixture([1]*8+[2]*3,sessions={9:'same',10:'same',11:'same'})
    catalog=ReportCatalog(rows,events,PredicateEquivalence())
    assert not catalog.groups[0]['closed']


def test_bad_source_span_rejected():
    events,rows=fixture([1]); rows['e1']['reports'][0]['value_span']=[0,1]
    with pytest.raises(ValueError,match='span'): ReportCatalog(rows,events,PredicateEquivalence())


def test_conflicting_same_episode_reports_not_resolved_by_order():
    events,rows=fixture([1]); frame=rows['e1']['reports'][0]
    rows['e1']['reports'].append({**frame,'normalized_value':'2'})
    assert not ReportCatalog(rows,events,PredicateEquivalence()).groups


def test_selection_and_pool_keep_budgets_and_direct_prefix():
    pool=list(range(48)); injected=inject_witnesses(pool,[91,92])
    assert injected[:24]==pool[:24] and len(injected)==48 and len(set(injected))==48
    selected=select_with_witnesses(injected,list(range(48)),[91,92])
    assert selected[:2]==[91,92] and len(selected)==8
    assert select_with_witnesses(pool,list(range(48)),[],top_k=0)==[]
    assert inject_witnesses([0],[2])==[0]


def test_two_report_control_can_differ_without_changing_source_inputs():
    events,rows=fixture([1]*2+[2]*2)
    mdl=ReportCatalog(rows,events,PredicateEquivalence())
    rule=ReportCatalog(rows,events,PredicateEquivalence(),policy='two_reports')
    assert not mdl.groups[0]['closed'] and rule.groups[0]['closed']
