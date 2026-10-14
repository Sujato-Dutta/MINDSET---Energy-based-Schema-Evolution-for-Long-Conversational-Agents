"""Explicit planner output contract; a software fix, not a novel method."""
import json
from mindset.methods.query_state_agent import PLANNER_PROMPT,planner_schema

FIELD_CONTRACT="""Output exactly these six JSON fields, all with string values:
intent: one of direct_factual, current, historical, temporal_change, contextual, multihop.
operation: the NAME of the tool to execute, one of SEARCH, CURRENT, HISTORY,
AT_TIME, CONTEXT, TRACE_CHANGE, PREDECESSOR. Put CURRENT here, not in context.
entity: the observed source speaker's exact inventory entity label.
attribute: the observed attribute's exact inventory key.
context: a literal setting from the inventory, or the empty string. This field
is NOT an operation name or an intent.
time: an explicit question date as YYYY-MM-DD, or the empty string, never null.
For SEARCH, entity/attribute/context/time must all be empty strings.
The output schema supplied below specifies the permitted values; decoder
constraints do not replace these field meanings. Do not output an answer."""


def explicit_messages(query,rows,question_date=''):
    return [{'role':'system','content':PLANNER_PROMPT+'\n\n'+FIELD_CONTRACT},
        {'role':'user','content':json.dumps({'question':query,'question_date':question_date,
            'observed_inventory':rows,'output_schema':planner_schema(rows)},ensure_ascii=False)}]
