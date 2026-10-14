"""Provisional relationships between lossless source reports, not world truth.

This is a measurement prerequisite for transition energy, not a novel controller.
No generated atom replaces a source assertion; no state update is executed here.
"""
import json
from mindset.utils.hashing import stable_hash

RELATIONS = ('equivalent', 'temporal_change', 'context_branch', 'accumulation',
             'modal_coexistence', 'independent', 'same_scope_conflict', 'unknown')
DIRECTIONS = ('a_to_b', 'b_to_a', 'none', 'unknown')
SCHEMA = {'type': 'object', 'additionalProperties': False,
    'properties': {'relationship': {'type': 'string', 'enum': list(RELATIONS)},
                   'direction': {'type': 'string', 'enum': list(DIRECTIONS)}},
    'required': ['relationship', 'direction']}
SYSTEM = """Measure the relationship of two source reports, not world truth.
Treat their complete text as evidence, including negation, modality, holder,
property, recurring context and factual time. Do not generate rewritten facts.
Return exactly the supplied JSON schema:
equivalent: the same holder/property/value assertion, possibly paraphrased;
temporal_change: the same state variable has explicitly different values at
successive factual times, or an explicit report of replacement/change;
context_branch: the same variable has different values in distinct recurring
or coexisting contexts, rather than one value replacing the other;
accumulation: additional nonexclusive members of the same set, such as likes
or possessions, without a stated exclusive replacement;
modal_coexistence: a plan, hope, uncertainty or hypothetical future can coexist
with the reported current state of that same variable;
independent: different holders, objects or properties, not a single variable;
same_scope_conflict: mutually incompatible assertions at the SAME factual time
and context, e.g. an explicit affirmation/denial or different exact counts;
unknown: unresolved referents/scope, ambiguous exclusivity, or multiple possible
relationships that cannot be assigned one label from the source reports.
Check variable identity before interpreting different values as a change.
Do not assume exclusivity: people may own many things, have multiple roles or
like several objects. Different factual times may be mutually compatible yet
describe temporal_change. Different recurring contexts are context_branch.
Observation_time is provenance, not the factual time of every statement.
A later message may describe an earlier state. For temporal_change, direction
follows factual chronology: a_to_b if A is earlier, b_to_a if B is earlier,
unknown if source chronology is unresolved. For all other relations use none.
Never infer a state transition solely because one message was observed later.
These labels are provisional source interpretations, not confirmed world updates.
Do not answer a benchmark question, explain, summarize, or update memory."""


def relation_messages(reports):
    if not isinstance(reports, list) or len(reports) != 2:
        raise ValueError('Two complete source reports required')
    if {r.get('source_id') for r in reports} != {'A', 'B'}:
        raise ValueError('Two unique source IDs required')
    if any(set(r) != {'source_id', 'speaker', 'observation_time', 'text'} or
           not all(isinstance(v, str) for v in r.values()) for r in reports):
        raise ValueError('Invalid source report fields')
    return [{'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': json.dumps(
                {'reports': reports, 'output_schema': SCHEMA}, ensure_ascii=False)}]


def validate_relation(payload, reports):
    relation_messages(reports)
    if not isinstance(payload, dict) or set(payload) != {'relationship', 'direction'}:
        raise ValueError('Invalid source relation payload')
    if payload['relationship'] not in RELATIONS or payload['direction'] not in DIRECTIONS:
        raise ValueError('Unknown relation/direction label')
    if payload['relationship'] == 'temporal_change':
        if payload['direction'] == 'none':
            raise ValueError('Temporal relation requires a direction or explicit uncertainty')
    elif payload['direction'] != 'none':
        raise ValueError('Only temporal relations may carry chronology')
    return {'source_reports_sha256': stable_hash(reports), 'complete_input_sources_retained': True,
            'semantic_correctness_verified': False, 'world_truth_or_update_verified': False}


CONTRACT_HASH = stable_hash({'system': SYSTEM, 'schema': SCHEMA})
