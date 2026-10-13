"""Independent constructed world trajectories; no controller pseudo-labels.

This is a controlled diagnostic suite, not an additional benchmark or a claim
of natural OOD generalization. Identical-observation ambiguity is intentional.
"""
from datetime import datetime, timedelta


def worlds():
    result = []
    families = ('change', 'coexist', 'noise', 'confirmation', 'explicit_noise', 'correlated_noise')
    for family in families:
        for shared_words in (0, 4, 16):
            for distractors in (0, 64):
                identifier = f'{family}-shared{shared_words}-distractors{distractors}'
                prefix = ' '.join(f'sector{i}' for i in range(shared_words))
                a = (prefix + ' Cedar').strip(); b = (prefix + ' Willow').strip()
                events = []; timeline = []; state = {}

                def emit(value, context='at home', changes_world=True, cue=False, actor='Sam', slot='location'):
                    i = len(events); stamp = (datetime(2031, 1, 1) + timedelta(days=i)).strftime('%Y-%m-%d')
                    text = f'I {"moved to" if cue else "live in"} {value} {context}.'
                    eid = f'e{i:04d}'
                    # Family names are evaluator-only, never controller metadata.
                    events.append({'event_id': eid, 'conversation_id': 'controlled-world', 'session_id': f's{i:04d}',
                        'timestamp': stamp, 'speaker': actor, 'text': text,
                        'observation': {'speaker': actor, 'text': text, 'slot': slot, 'value': value,
                                        'contexts': [context], 'explicit_change': cue}})
                    if changes_world: state[(actor, slot, context)] = value
                    timeline.append([{'owner': k[0], 'slot': k[1], 'context': k[2], 'value': v}
                                     for k, v in sorted(state.items())])

                for _ in range(8): emit(a)
                for i in range(distractors): emit(f'Archive{i}', actor='Pat', slot='location')
                if family == 'change':
                    emit(b, cue=True); emit(b); emit(b)
                elif family == 'coexist':
                    emit(b, context='at work'); emit(a); emit(b, context='at work')
                elif family in ('noise', 'explicit_noise'):
                    emit(b, changes_world=False, cue=family == 'explicit_noise'); emit(a); emit(a)
                elif family == 'confirmation':
                    emit(b); emit(b); emit(b)
                else:
                    emit(b, changes_world=False); emit(b, changes_world=False); emit(b, changes_world=False)
                result.append({'id': identifier, 'family': family, 'shared_words': shared_words,
                    'distractors': distractors, 'events': events, 'truth_timeline': timeline,
                    'interpretation': 'latent truth used only by evaluator; controller sees asserted observations'})
    return result
