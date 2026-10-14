from mindset.memory.query_contracts import compatible, report_contract
from mindset.methods.typed_grounding import TypedReportCatalog
from mindset.methods.report_grounding import PredicateEquivalence
from test_report_grounding import fixture


def contract(kind='quantity',head='worker',names=(),target_names=()):
    return {'answer_kind':kind,'argument_heads':[head],'named_scope':list(names),'target_names':list(target_names)}


def frame(family='quantity',head='worker',names=()):
    return {'family':family,'contract':{'family':family,'argument_head':head,'argument_names':list(names)}}


def test_same_predicate_does_not_erase_target():
    assert not compatible(contract('existence','screwdriver'),frame('object','help'),'I have help.')
    assert compatible(contract('existence','screwdriver'),frame('object','screwdriver'),'I have a screwdriver.')


def test_frequency_does_not_admit_person_name_as_answer():
    q=contract('frequency','therapist',target_names=('morgan',))
    assert not compatible(q,frame('object','morgan',('morgan',)),'I see Morgan.')
    assert compatible(q,frame('frequency',None,('morgan',)),'I see Morgan twice a week.')
    assert not compatible(q,frame('frequency',None,('avery',)),'I see Avery twice a week.')


def test_count_does_not_admit_non_numeric_object():
    assert not compatible(contract('quantity','issue'),frame('object','series'),'I read a series.')


def test_named_query_scope_is_required_in_every_source():
    q=contract('property','score',names=('chess',))
    assert not compatible(q,frame('property','score'),'My high score in golf is 20.')
    assert compatible(q,frame('property','score'),'My high score in Chess is 20.')


def typed_fixture(values):
    events,rows=fixture(values)
    for row in rows.values():
        f=row['reports'][0]
        f.update(family='quantity',unit_terms=['quantity','worker'],contract=frame()['contract'])
    q={'text':'How many workers do I lead now?','report_query':{'mode':'state_current','self_reference':True,
       'contract':contract(),'probes':[{'unit_terms':['quantity','worker'],'probe':'I lead one worker.'}]}}
    return events,rows,q


def test_catalog_keeps_valid_raw_witness_and_off_control():
    events,rows,q=typed_fixture([1]*8+[2]*3)
    c=TypedReportCatalog(rows,events,PredicateEquivalence())
    assert c.witnesses(q)[0]==['e11']
    assert c.witnesses(q,selector='off')[0]==[]


def test_missing_contract_is_not_backward_silent_wildcard():
    events,rows,q=typed_fixture([1,2]); del q['report_query']['contract']
    assert TypedReportCatalog(rows,events,PredicateEquivalence()).witnesses(q)[0]==[]


def test_polluted_group_is_not_certified_from_matching_subset():
    events,rows,q=typed_fixture([1,2])
    q['report_query']['contract']['named_scope']=['chess']
    # Neither report satisfies the named question constraint; binding abstains.
    assert TypedReportCatalog(rows,events,PredicateEquivalence()).witnesses(q)[0]==[]


def test_latest_event_not_promoted_as_current_state():
    events,rows,q=typed_fixture([1,2]); q['report_query']['mode']='latest_event'
    assert TypedReportCatalog(rows,events,PredicateEquivalence()).witnesses(q)[0]==[]


def test_frequency_identity_comes_from_argument_not_value():
    f={'family':'frequency','unit_head':'therapist','probe':'old','predicate':'see',
       'argument_text':'Dr Morgan','argument_names':['dr','morgan']}
    result=report_contract(f,[], 'I see Dr Morgan weekly.')
    assert result['unit_terms']==['frequency','named:dr morgan']
    assert result['contract']['argument_names']==['dr','morgan']
