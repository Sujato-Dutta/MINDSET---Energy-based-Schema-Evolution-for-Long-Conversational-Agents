from dataclasses import replace
from mindset.methods.participant_frontier import ParticipantFrontierCatalog
from mindset.methods.report_grounding import PredicateEquivalence
from test_typed_grounding import typed_fixture
from test_compact import event
from pathlib import Path
import importlib.util


def test_nli_queries_use_the_same_explicit_owner_namespace_as_sources():
    spec=importlib.util.spec_from_file_location('nli_worker',Path(__file__).resolve().parents[1]/'scripts/prepare_report_nli.py')
    worker=importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
    assert worker.predicate_owner({'kind':'query','report_query':{'contract':{'owner':'rae'}}},{})=='rae'
    assert worker.predicate_owner({'kind':'query','report_query':{'contract':{'owner':None}}},{}) is None
    assert worker.predicate_owner({'kind':'query','report_query':{}},{})=='user'
    assert worker.predicate_owner({'kind':'episode'},{'owner':'rae'})=='rae'


def named_fixture():
    events,rows,q=typed_fixture([1,2])
    events=[replace(e,speaker='Rae') for e in events]
    for row in rows.values():
        row['speaker']='Rae'; row['reports'][0]['owner']='rae'
    q['report_query']['contract']['owner']='rae'
    return events,rows,q


def test_named_speaker_binds_without_name_in_first_person_body():
    events,rows,q=named_fixture()
    ids,c=ParticipantFrontierCatalog(rows,events,PredicateEquivalence()).frontier(q)
    assert ids==['e2','e1'] and c['binding']=='unique_explicit_speaker_argument_type_binding'


def test_question_owner_cannot_bind_another_speaker():
    events,rows,q=named_fixture(); q['report_query']['contract']['owner']='avery'
    assert ParticipantFrontierCatalog(rows,events,PredicateEquivalence()).frontier(q)[0]==[]


def test_other_speaker_opaque_text_does_not_replace_owner_witness():
    events,rows,q=named_fixture(); e=replace(event(12,'The worker roster changed.'),speaker='Avery'); events.append(e)
    rows[e.event_id]={'text':e.text,'speaker':e.speaker,'reports':[]}
    assert ParticipantFrontierCatalog(rows,events,PredicateEquivalence()).frontier(q)[0]==['e2','e1']


def test_same_speaker_opaque_text_preserved_with_raw_provenance():
    events,rows,q=named_fixture(); e=replace(event(12,'The worker roster changed.'),speaker='Rae'); events.append(e)
    rows[e.event_id]={'text':e.text,'speaker':e.speaker,'reports':[]}
    ids,c=ParticipantFrontierCatalog(rows,events,PredicateEquivalence()).frontier(q)
    assert ids==['e12','e2'] and c['opaque_tail_ids']==['e12'] and not c['world_truth_certified']


def test_unresolved_owner_abstains_and_user_remains_supported():
    events,rows,q=typed_fixture([1,2])
    assert ParticipantFrontierCatalog(rows,events,PredicateEquivalence()).frontier(q)[0]==[]
    q['report_query']['contract']['owner']='user'
    assert ParticipantFrontierCatalog(rows,events,PredicateEquivalence()).frontier(q)[0]==['e2','e1']
