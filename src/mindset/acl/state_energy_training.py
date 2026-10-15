"""Independent source-language streams and post-state read-risk training.

No benchmark is loaded by this module. Synthetic prose and truth are emitted
from the same explicitly declared world generator, then parsed by the exact
same source-only worker used for natural histories. Parser recall is measured,
never repaired using the expected probes.
"""
from dataclasses import asdict
import random
from mindset.memory.grounded_state_energy import (
    Source, Binding, GroundedStateEnergy, FEATURES, fit_risk, probe_loss)


def corpus(seed=7301, per_family=24):
    rng = random.Random(seed); histories = []
    domains = {
        'train': ('prefer', ('tea', 'coffee', 'juice'), ('home', 'work')),
        'validation': ('like', ('jazz', 'folk', 'opera'), ('school', 'church')),
        'holdout': ('own', ('bicycles', 'scooters', 'skates'), ('studio', 'office')),
    }
    for split, (verb, vocabulary, contexts) in domains.items():
        for family in ('repeat', 'update', 'modal', 'context', 'context_ellipsis', 'unresolved_holder'):
            for i in range(per_family):
                a, b = rng.sample(vocabulary, 2); left, right = rng.sample(contexts, 2)
                identifier = f'joint-risk-{split}-{family}-{i:03d}'
                holder = f'{split}speaker{i:03d}'; rows = []; probes = []
                def add(text, truths):
                    order = len(rows)
                    rows.append(dict(event_id=f'{identifier}/{order}', text=text,
                                     speaker=holder, order=order))
                    # The following are evaluator-only read queries. They never
                    # enter the parser, inference model or natural selector.
                    probes.append([dict(holder=holder, variable=verb, context=c,
                                        value=v, before=t) for c, v, t in truths])
                add(f'I {verb} {a} at {left}.', [(left, a, None)])
                if family == 'repeat':
                    add(f'I still {verb} {a} at {left}.', [(left, a, None), (left, a, 0)])
                elif family == 'update':
                    add(f'I now {verb} {b} at {left}.', [(left, b, None), (left, a, 0)])
                elif family == 'modal':
                    add(f'I might {verb} {b} at {left}.', [(left, a, None), (left, a, 0)])
                elif family == 'context':
                    add(f'I {verb} {b} at {right}.', [(left, a, None), (right, b, None)])
                elif family == 'context_ellipsis':
                    # Declared discourse convention: a single already established
                    # setting is carried forward. This assumption is not ground
                    # truth for arbitrary natural dialogue.
                    add(f'I now {verb} {b}.', [(left, b, None), (left, a, 0)])
                else:
                    add(f'They {verb} {b} at {left}.', [(left, a, None)])
                histories.append(dict(id=identifier, split=split, family=family,
                    sources=rows, probes=probes,
                    truth_provenance='declarative synthetic state trajectories; not controller decisions'))
    return dict(seed=seed, generator='source-language-poststate-risk-v1',
        histories=histories, label_source='independent synthetic read outcomes',
        limitations='Short canonical streams, declared context-ellipsis convention, disjoint value/verb/context domains; not natural OOD proof')


def zero_model():
    return dict(features=list(FEATURES), weights=[0.] * len(FEATURES),
                mean=[0.] * len(FEATURES), scale=[1.] * len(FEATURES))


def bound(row, parsed):
    import hashlib
    result = parsed[row['event_id']]
    if result['source_sha256'] != hashlib.sha256(row['text'].encode()).hexdigest():
        raise ValueError('Source parser hash mismatch')
    source = Source(row['event_id'], row['text'], row['order'])
    bindings = [Binding(**b) for b in result['bindings']]
    for b in bindings:
        b.validate(source)
    return source, bindings


def train(corpus, parsed):
    records = []
    for h in corpus['histories']:
        if h['split'] != 'train':
            continue
        memory = GroundedStateEnergy(zero_model())
        for row, probes in zip(h['sources'], h['probes'], strict=True):
            source, bindings = bound(row, parsed)
            plans = memory.candidates(source, bindings)
            alternatives = []
            for j, plan in enumerate(plans):
                trial = memory.fork(); trial.commit(source, plan)
                loss = probe_loss(trial, probes)
                # Every candidate is scored, not just the selected action.
                records.append(dict(id=f'{row["event_id"]}/{j}', split='train',
                    history_id=h['id'], features=list(plan.features), loss=loss,
                    action=plan.action, binding=asdict(plan.binding)))
                alternatives.append((loss, j, plan))
            # Teacher-forced train prefixes only; held-out rollouts below use
            # their OWN prior state. This limitation is explicitly reported.
            best = min(alternatives, key=lambda x: (x[0], x[1]))[2] if alternatives else None
            memory.commit(source, best)
    return fit_risk(records), records


def evaluate(corpus, parsed, model, split, mode):
    errors = []; losses = {}; operations = {}; bound_sources = 0; source_count = 0
    for h in corpus['histories']:
        if h['split'] != split:
            continue
        memory = GroundedStateEnergy(model, mode)
        local_errors = []
        for row, probes in zip(h['sources'], h['probes'], strict=True):
            source, bindings = bound(row, parsed); source_count += 1
            bound_sources += bool(bindings)
            chosen, _ = memory.ingest(source, bindings)
            local_errors.append(probe_loss(memory, probes))
            action = 'unbound' if chosen is None else chosen.action
            operations[action] = operations.get(action, 0) + 1
        memory.invariant_report()
        errors.extend(local_errors)
        losses.setdefault(h['family'], []).extend(local_errors)
    return dict(mode=mode, split=split, mean_read_error=sum(errors) / len(errors),
        family_read_error={k: sum(v) / len(v) for k, v in losses.items()},
        parsed_sources=bound_sources, total_sources=source_count, operations=operations,
        qualification='Independent short synthetic rollouts; not QA accuracy or natural semantic verification')
