from dataclasses import replace
import numpy as np
import pytest
from mindset.memory.grounded_state_energy import (
    Source, Binding, GroundedStateEnergy, FEATURES, fit_risk, probe_loss)


def model(**weights):
    return dict(features=list(FEATURES), weights=[weights.get(k, 0.) for k in FEATURES],
                mean=[0.] * len(FEATURES), scale=[1.] * len(FEATURES))


def fact(i, value, context='', **flags):
    text = f'I prefer {value}.'
    s = Source(str(i), text, i)
    return s, Binding(str(i), 'Alice', 'prefer', value, 9, 9 + len(value), context, i, **flags)


def test_literal_source_and_duplicate_protection():
    s, b = fact(0, 'tea'); m = GroundedStateEnergy(model(state_write=-1))
    with pytest.raises(ValueError):
        m.ingest(s, [replace(b, value='coffee')])
    m.ingest(s, [b])
    with pytest.raises(ValueError):
        m.ingest(s, [b])
    assert m.invariant_report()['immutable_sources'] == 1


def test_actual_versions_and_historical_read_are_preserved():
    m = GroundedStateEnergy(model(state_write=-1))
    for i, v in enumerate(('tea', 'coffee', 'tea')):
        s, b = fact(i, v); m.ingest(s, [b])
    assert m.read('Alice', 'prefer').binding.value == 'tea'
    assert m.read('Alice', 'prefer', before=1).binding.value == 'coffee'
    assert m.read('Alice', 'prefer', before=-1) is None
    assert m.invariant_report()['versions'] == 3


def test_backdated_observation_does_not_replace_current():
    m = GroundedStateEnergy(model(state_write=-1))
    s, b = fact(0, 'coffee'); m.ingest(s, [replace(b, order=8)])
    s, b = fact(1, 'tea'); m.ingest(s, [replace(b, order=2)])
    assert m.read('Alice', 'prefer').binding.value == 'coffee'
    assert m.read('Alice', 'prefer', before=3).binding.value == 'tea'


def test_context_coexistence_and_exception_do_not_destroy_sources():
    m = GroundedStateEnergy(model(state_write=-1, modal_write=2))
    for i, v, c, modal in ((0, 'tea', 'home', False), (1, 'coffee', 'work', False),
                          (2, 'juice', 'home', True)):
        s, b = fact(i, v, c, modal=modal); m.ingest(s, [b])
    assert m.read('Alice', 'prefer', 'home').binding.value == 'tea'
    assert m.read('Alice', 'prefer', 'work').binding.value == 'coffee'
    assert len(m.sources) == 3 and len(m.exceptions) == 1


def test_global_binding_can_differ_from_local_and_hard_grounding():
    learned = model(state_write=-1, known_context_match=-2)
    full = GroundedStateEnergy(learned); local = GroundedStateEnergy(learned, 'local')
    hard = GroundedStateEnergy(learned, 'hard_grounding')
    for m in (full, local, hard):
        s, b = fact(0, 'tea', 'home'); m.ingest(s, [b])
        s, b = fact(1, 'coffee'); chosen, _ = m.ingest(s, [b])
        if m is full:
            assert chosen.binding.context == 'home'
        else:
            assert chosen.binding.context == ''


def test_counterfactual_fork_cannot_mutate_live_state():
    m = GroundedStateEnergy(model(state_write=-1))
    s, b = fact(0, 'tea'); m.ingest(s, [b]); clone = m.fork()
    s, b = fact(1, 'coffee'); clone.ingest(s, [b])
    assert m.read('Alice', 'prefer').binding.value == 'tea'
    assert len(m.sources) == 1 and len(clone.sources) == 2


def test_future_read_loss_not_action_agreement_is_the_training_target():
    m = GroundedStateEnergy(model(state_write=-1)); s, b = fact(0, 'tea')
    plans = m.candidates(s, [b])
    probes = [dict(holder='Alice', variable='prefer', value='tea')]
    losses = []
    for p in plans:
        trial = m.fork(); trial.commit(s, p); losses.append(probe_loss(trial, probes))
    assert losses == [0., 1.] and not m.sources
    records = [dict(features=list(p.features), loss=l, split='train') for p, l in zip(plans, losses)]
    fitted = fit_risk(records)
    result, _ = GroundedStateEnergy(fitted).ingest(s, [b])
    assert result.action == 'new'
    with pytest.raises(ValueError):
        fit_risk([{**records[0], 'split': 'holdout'}])


def test_invalid_or_nonfinite_models_fail_closed():
    with pytest.raises(ValueError):
        GroundedStateEnergy({})
    bad = model(); bad['weights'][0] = float('nan')
    with pytest.raises(ValueError):
        GroundedStateEnergy(bad)


def test_local_predictor_is_fitted_without_global_features():
    records = []
    for i in range(20):
        x = [0.] * len(FEATURES)
        x[FEATURES.index('known_context_match')] = float(i % 2)
        records.append(dict(split='train', features=x, loss=float(i % 2)))
    fitted = fit_risk(records, local=True)
    assert fitted['local_features_only']
    assert np.allclose(fitted['weights'], 0.)
    bad = model(); bad['scale'][0] = 0
    with pytest.raises(ValueError):
        GroundedStateEnergy(bad)
