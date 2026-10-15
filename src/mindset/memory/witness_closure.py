"""Budgeted source-witness closure for reconstructive memory (pilot only).

Declared reply dependencies are parser hypotheses, NOT verified coreference.
Calendar expressions are normalized relative to their own source timestamp;
normalization does not certify that a described event occurred. Selection is an
exact finite budgeted closure problem, not a claim of a new generic optimizer.
"""
from datetime import datetime, timedelta, timezone
from fractions import Fraction
import re
from dateutil import parser
from dateutil.relativedelta import relativedelta

WEEKDAYS = ('monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday')
TIME_PATTERN = re.compile(r'\b(?:yesterday|today|tomorrow|last\s+(?:month|year|'
    + '|'.join(WEEKDAYS) + r'))\b', re.I)


def calendar_constraints(eid, text, observation_time):
    if not isinstance(observation_time, str) or not re.search(r'\b\d{4}\b', observation_time):
        return []  # Never turn a missing year into a machine/default year.
    try:
        stamp = parser.parse(observation_time, fuzzy=True, default=datetime(1970, 1, 1))
    except (TypeError, ValueError, OverflowError):
        return []
    day = stamp.date(); output = []
    for match in TIME_PATTERN.finditer(text):
        term = ' '.join(match.group().casefold().split())
        precision = 'day'
        if term in {'yesterday', 'today', 'tomorrow'}:
            start = day + timedelta(days={'yesterday': -1, 'today': 0, 'tomorrow': 1}[term])
            end = start + timedelta(days=1)
        elif term == 'last month':
            end = day.replace(day=1); start = end - relativedelta(months=1); precision = 'month'
        elif term == 'last year':
            end = day.replace(month=1, day=1); start = end.replace(year=end.year - 1); precision = 'year'
        else:
            weekday = WEEKDAYS.index(term.split()[-1])
            delta = (day.weekday() - weekday) % 7 or 7
            start = day - timedelta(days=delta); end = start + timedelta(days=1)
        output.append(dict(event_id=eid, char_start=match.start(), char_end=match.end(),
            quote=match.group(), observation_time=observation_time,
            start=start.isoformat(), end=end.isoformat(), precision=precision,
            scope='calendar interpretation of literal expression, not verified event occurrence'))
    return output


def validate_calendar(note, event):
    expected = calendar_constraints(event.event_id, event.text, event.timestamp)
    if note not in expected:
        raise ValueError('Calendar annotation cannot be reconstructed from its source')
    if event.text[note['char_start']:note['char_end']] != note['quote']:
        raise ValueError('Calendar source span differs')


def render_packet(selected, notes):
    raw = '\n\n'.join(f'[{i}] {s["text"]}' for i, s in enumerate(selected, 1))
    numbering = {s['event_id']: i for i, s in enumerate(selected, 1)}
    lines = []
    for note in notes:
        if note['event_id'] not in numbering:
            raise ValueError('Calendar constraint without selected source')
        value = note['start'] if note['precision'] == 'day' else f'[{note["start"]}, {note["end"]})'
        lines.append(f'Time [{numbering[note["event_id"]]}]: {note["quote"]!r} => {value}.')
    return raw + ('\n\nSource-anchored calendar interpretations (not new event assertions):\n' + '\n'.join(lines) if lines else '')


def dependency_closure(roots, dependencies, available):
    seen = set(); active = set()
    def visit(eid):
        if eid in active:
            raise ValueError('Cyclic declared context dependency')
        if eid in seen:
            return
        if eid not in available:
            raise KeyError('Required source outside bounded candidate catalog')
        active.add(eid)
        for prior in dependencies.get(eid, ()):
            visit(prior)
        active.remove(eid); seen.add(eid)
    for root in roots:
        visit(root)
    return seen


def select_packet(root_ids, passages, dependencies, notes, token_count, token_budget,
                  source_order, *, close=True, calendar=True, max_sources=8):
    """Enumerate <=2^8 root subsets; union dependencies before budget checks.

    Maximize retained reciprocal-rank mass, then minimize sources and tokens.
    This exact optimum is conditional on the declared graph and candidate set.
    Neither references nor gold labels enter this function.
    """
    if len(root_ids) > 8 or len(set(root_ids)) != len(root_ids):
        raise ValueError('Expected at most eight distinct fixed retrieval anchors')
    if token_budget <= 0 or max_sources <= 0:
        raise ValueError('Positive evidence budget required')
    root_rank = {eid: i for i, eid in enumerate(root_ids)}
    candidates = {}; rejected = 0
    for mask in range(1, 1 << len(root_ids)):
        proposed = [eid for i, eid in enumerate(root_ids) if mask & (1 << i)]
        try:
            ids = dependency_closure(proposed, dependencies if close else {}, passages)
        except KeyError:
            rejected += 1; continue
        if len(ids) > max_sources:
            rejected += 1; continue
        # Keep anchor rank across independent bundles; order a bundle's source
        # dependencies before its referring turn, never globally newest-first.
        ordered = []; placed = set()
        def place(eid):
            if eid in placed:
                return
            for parent in sorted(dependencies.get(eid, ()) if close else (), key=lambda x: source_order[x]):
                if parent in ids:
                    place(parent)
            placed.add(eid); ordered.append(eid)
        for eid in root_ids:
            if eid in ids:
                place(eid)
        if set(ordered) != ids:
            raise ValueError('Closure serialization lost a source')
        selected = [passages[eid] for eid in ordered]
        annotation = [n for eid in ordered for n in notes.get(eid, ())] if calendar else []
        evidence = render_packet(selected, annotation)
        tokens = token_count(evidence)
        if tokens > token_budget:
            rejected += 1; continue
        retained = sum((Fraction(1, root_rank[eid] + 1) for eid in ids if eid in root_rank), Fraction())
        key = (-retained, len(ids), tokens, tuple(ordered))
        old = candidates.get(tuple(ordered))
        if old is None or key < old[0]:
            candidates[tuple(ordered)] = (key, selected, evidence, annotation, retained)
    if not candidates:
        if close:
            selected, evidence, audit = select_packet(root_ids, passages, {}, notes, token_count,
                token_budget, source_order, close=False, calendar=calendar, max_sources=max_sources)
            audit.update(closure_supported=False, fallback='no_nonempty_closed_packet_within_budget')
            return selected, evidence, audit
        raise ValueError('No raw source passage fits the matched token budget')
    best = min(candidates.values(), key=lambda r: r[0])
    return best[1], best[2], dict(closure_supported=close,
        selected_ids=[s['event_id'] for s in best[1]], calendar_constraints=best[3],
        retained_rank_mass=str(best[4]), candidate_packets=len(candidates), rejected_subsets=rejected,
        tokens=token_count(best[2]), objective='max reciprocal-rank anchor mass subject to declared source closure and budget',
        optimality_scope='exact over fixed <=8 anchor subsets, not global or semantic sufficiency',
        dependency_semantics='heuristic source-context requirements, not certified coreference or truth')
