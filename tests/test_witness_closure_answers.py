from copy import deepcopy
from types import SimpleNamespace as N
import pytest
from mindset.acl.witness_closure_answers import transform_plan
from mindset.memory.witness_closure import calendar_constraints, render_packet


def fixture():
    event = N(event_id='e', text='I left yesterday.', timestamp='2025-02-03')
    selected = [dict(event_id='e', text='[2025-02-03] Sam: I left yesterday.')]
    notes = calendar_constraints('e', event.text, event.timestamp)
    planned = [dict(id='locomo/full/q', dataset='locomo', question_id='q', question='When did Sam leave?',
                    question_date='', category='temporal', selected=selected)]
    retrieval = {'locomo/full/q': dict(reconstruction=dict(calendar_constraints=notes),
                                      annotated_evidence=render_packet(selected, notes))}
    return planned, retrieval, {('locomo', 'q'): N(events=[event])}


def test_source_calendar_annotations_are_revalidated_and_prompt_is_rebuilt():
    planned, retrieval, cases = fixture()
    out = transform_plan(planned, retrieval, cases)[0]
    assert '2025-02-02' in out['evidence']
    assert out['evidence'] in out['messages'][0]['content']
    assert out['question'] == 'When did Sam leave?'
    assert out['prompt_sha256']


def test_forged_calendar_value_never_reaches_reader():
    planned, retrieval, cases = fixture()
    retrieval['locomo/full/q']['reconstruction']['calendar_constraints'][0]['start'] = '1999-01-01'
    with pytest.raises(ValueError):
        transform_plan(planned, retrieval, cases)


def test_hidden_annotation_cannot_be_inserted_into_serialized_packet():
    planned, retrieval, cases = fixture()
    retrieval['locomo/full/q']['annotated_evidence'] += '\nInvented fact'
    with pytest.raises(ValueError):
        transform_plan(planned, retrieval, cases)
