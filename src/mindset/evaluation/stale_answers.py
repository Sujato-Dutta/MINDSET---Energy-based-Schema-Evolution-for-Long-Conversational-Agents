"""Conservative exact-value stale-answer proxy, evaluator-only.

Requires a real prediction and a negative official correctness judgement.
Reference must equal the latest observable report; a prediction must equal a
different earlier report. Not a comprehensive semantic stale-error annotation.
Unmatched/negated/paraphrased answers are unclassified, never counted correct.
"""
import re
from mindset.methods.joint import utc_time


def exact_value(text):
    if not isinstance(text,str): return None
    text=re.sub(r'\s+',' ',text.strip().casefold()).rstrip(' .!?')
    return text or None


def stale_proxy(prediction,reference,*,query_mode,bound_reports,official_correct):
    result={'status':'unclassified','stale':None,'definition':'exact_old_value_negative_official_judge_proxy'}
    if query_mode!='state_current': return {**result,'reason':'not_current_state_query'}
    if official_correct is None: return {**result,'reason':'official_judgement_missing'}
    if not bound_reports: return {**result,'reason':'no_unique_source_binding'}
    stamps=[utc_time(r.get('timestamp','')) for r in bound_reports]
    if not all(stamps) or stamps!=sorted(stamps):
        return {**result,'reason':'report_chronology_unknown_or_unordered'}
    reports=[r for r in bound_reports if r.get('tense')=='nonpast']
    if not reports: return {**result,'reason':'no_nonpast_observable_report'}
    latest=exact_value(reports[-1]['value']); gold=exact_value(reference); answer=exact_value(prediction)
    if not latest or gold!=latest: return {**result,'reason':'reference_not_exact_latest_observed_value'}
    earlier={exact_value(r['value']) for r in reports[:-1]}-{latest,None}
    if answer in earlier and official_correct is False:
        return {**result,'status':'classified','stale':True,'reason':'incorrect_exact_earlier_report_value'}
    if answer==latest and official_correct is True:
        return {**result,'status':'classified','stale':False,'reason':'correct_exact_latest_report_value'}
    return {**result,'reason':'answer_not_an_exact_judged_source_value'}
