import pytest
from dataclasses import replace
from mindset.methods.aligned_readout import AlignedMindsetMemory, self_report_view, observable_period
from mindset.methods.compact import CompactConfig
from test_compact import Encoder, Reranker, event


def memory(texts, speaker='user', **kwargs):
    m = AlignedMindsetMemory(CompactConfig(complexity_penalty=.1, switch_margin=.05),
        encoder=Encoder(), local_reranker=Reranker(), **kwargs)
    events = [replace(event(i, text), speaker=speaker) for i, text in enumerate(texts, 1)]
    m.ingest(events)
    return m


def selected(m, query, date=''):
    return [h.memory_id for h in m.select_scored(query, list(range(len(m._episode_order))),
        list(range(len(m._episode_order), 0, -1)), 8, date)]


def test_native_first_person_current_and_historical_binding():
    m = memory(['I reside in Paris.', 'I have relocated to Rome.'])
    assert selected(m, 'Where do I live now?') == ['e2']
    assert m.last_readout_certificate['identity_binding'] == 'first_person_bound_to_literal_user_role'
    assert selected(m, 'Where did I live on 2025/01/01?') == ['e1']
    assert selected(m, 'Where do I live?', '2025/01/01 (Wed) 12:00') == ['e1']


def test_do_not_guess_named_speaker_or_nested_friend_identity():
    m = memory(['I live in Paris.', 'I moved to Rome.'], speaker='Sam')
    assert selected(m, 'Where do I live now?') == ['e1', 'e2']
    m = memory(['I live in Paris.', 'I moved to Rome.'])
    assert selected(m, 'Where does my friend live now?') == ['e1', 'e2']
    assert selected(m, "What is my daughter's current address?") == ['e1', 'e2']


def test_observed_dynamic_relationship_predicate_is_queryable():
    m = memory(['My partner is Alex.', 'My partner is Jordan now.'])
    # Value includes the cue in this bounded parser; test routing, not a claim
    # that arbitrary language has been semantically normalized perfectly.
    scope = m._scope('Who is my partner?')
    assert scope.owners == ('user',) and 'relationship_partner' in scope.slots
    assert selected(m, 'Who is my partner?') == ['e2']


def test_parsing_view_never_replaces_raw_evidence_or_ids():
    texts = ["I've recently moved to Paris.", 'I am employed by Acme.']
    m = memory(texts)
    assert [m.episodes[f'e{i}'].text for i in (1, 2)] == texts
    assert m._documents[0].endswith(texts[0])
    assert m.validate_invariants()['transition_count'] == 2
    assert m.telemetry()['llm_ingestion_calls'] == 0


@pytest.mark.parametrize('text', ['My friend said I live in Paris.', 'I do not live in Paris.',
                                  'I used to live in Paris.', '"I live in Paris."'])
def test_unknown_negation_quotation_and_historical_assertion_are_not_current(text):
    assert self_report_view(text) == ''


def test_role_aliases_share_one_predicate_and_handle_new_job_names():
    m = memory(['I am an engineer.', 'I now work as a florist.'])
    assert {s.topic for s in m.schemas.values()} == {'user::relationship_occupation'}
    assert selected(m, 'What is my current job?') == ['e2']


def test_compiled_state_assessment_does_not_reparse_dates(monkeypatch):
    m = memory(['I live in Paris.', 'I moved to Rome.'])
    import mindset.methods.aligned_readout as module
    monkeypatch.setattr(module, 'utc_time', lambda _: (_ for _ in ()).throw(AssertionError('date reparsed')))
    assert selected(m, 'Where do I live now?') == ['e2']


def test_read_disabled_preserves_raw_ranking_and_rule_audit_not_false_energy():
    m = memory(['I live in Paris.', 'I moved to Rome.'], state_readout=False)
    assert selected(m, 'Where do I live now?') == ['e1', 'e2']
    m = memory(['I live in Paris.', 'I moved to Rome.'], write_policy='rule')
    assert m._decision_certificates['e2']['policy'] == 'fixed_rule_control'
    m = memory(['I live in Paris.', 'I moved to Rome.'], write_policy='nearest')
    assert not any(s.status == 'historical' for s in m.schemas.values()) and not m._decision_certificates


def test_date_precision_and_multi_date_abstention():
    start, end = observable_period('2025/01/02 (Thu) 15:00')
    assert end - start == 86400
    assert observable_period('2 January 2025') == (start, end)
    assert observable_period('2025-01-01 and 2025-01-02') == (0., 0.)


def test_fast_budget_one_pass_and_raw_sources():
    m = AlignedMindsetMemory(CompactConfig(complexity_penalty=.1, switch_margin=.05,
        retrieval_candidate_k=24, direct_pool_k=12, schema_pool_k=12), encoder=Encoder(), local_reranker=Reranker())
    m.ingest([replace(event(i, f'I prefer tea item {i}.'), speaker='user') for i in range(1, 41)])
    hits = m.retrieve('What do I prefer?', 8)
    assert m.local_reranker.calls == [24] and len(m.last_candidate_pool) == 24
    assert all(h.source_event_ids == (h.memory_id,) for h in hits)
