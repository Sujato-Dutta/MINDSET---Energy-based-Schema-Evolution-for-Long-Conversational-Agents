from mindset.methods.broad_grounding import BroadReportCatalog
from mindset.methods.report_grounding import PredicateEquivalence
from test_report_grounding import fixture,query


def current():
    return query(mode='state_current')


def test_current_query_can_bind_and_returns_real_mode_support_not_new_text():
    events,rows=fixture([1]*8+[2]*3)
    c=BroadReportCatalog(rows,events,PredicateEquivalence())
    ids,cert=c.witnesses(current())
    assert ids==['e11'] and cert['committed_report_edge']
    assert cert['selected_observed_value']=='2' and not cert['world_truth_certified']
    assert cert['hard_exclusions']==[]


def test_current_query_keeps_mdl_vs_latest_difference_visible():
    events,rows=fixture([1]*8+[2])
    c=BroadReportCatalog(rows,events,PredicateEquivalence())
    assert c.witnesses(current())[0]==['e8']
    assert c.witnesses(current(),selector='latest')[0]==['e9']


def test_past_only_reports_not_promoted_into_current_witness():
    events,rows=fixture([1,2])
    for row in rows.values(): row['reports'][0]['tense']='past'
    c=BroadReportCatalog(rows,events,PredicateEquivalence())
    assert c.witnesses(current())[0]==[]
    assert c.witnesses(current(),selector='latest')[0]==[]


def test_historical_queries_do_not_use_current_reports_or_guess_date():
    events,rows=fixture([1,2]); c=BroadReportCatalog(rows,events,PredicateEquivalence())
    assert c.witnesses(query(mode='historical'))[0]==[]


def test_ambiguous_binding_abstains_even_with_source_candidates():
    events,rows=fixture([1,2]); frame=rows['e2']['reports'][0]
    rows['e2']['reports'].append({**frame,'probe':'I manage one worker.'})
    pairs=[{'premise':q,'hypothesis':p,'label':'entailment'} for p in ('I lead one worker.','I manage one worker.')
           for q in ['I supervise one worker.']]
    pairs += [{'premise':r['hypothesis'],'hypothesis':r['premise'],'label':'entailment'} for r in list(pairs)]
    c=BroadReportCatalog(rows,events,PredicateEquivalence(pairs))
    ids,cert=c.witnesses(query(probe='I supervise one worker.',mode='state_current'))
    assert ids==[] and cert['binding']=='no_unique_owner_family_predicate_binding'


def test_off_control_never_reserves_broad_witness():
    events,rows=fixture([1,2]); c=BroadReportCatalog(rows,events,PredicateEquivalence())
    assert c.witnesses(current(),selector='off')[0]==[]
