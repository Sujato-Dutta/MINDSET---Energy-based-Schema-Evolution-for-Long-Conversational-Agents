"""Evaluator-only audit of annotated source coverage versus saved QA outcomes.

No inference or parameter tuning. Full annotated raw-source coverage is not an
answer-availability certificate: annotations/captions/reference scoring can fail.
"""
import os
from pathlib import Path

import numpy as np

from mindset.acl.pipeline import ROOT, record_environment, rank_metrics
from mindset.acl.joint_iteration import protocol, groups_for
from mindset.acl.io import read_json, write_json, unique_rows, digest_file
from mindset.acl.clients import messages, ANSWER_PROMPT_HASH
from mindset.acl.locomo_protocol_diagnostic import short_phrase_messages, READER_HASH
from mindset.baselines.event_rag import render_event
from mindset.datasets.locomo import CATEGORY_NAMES
from mindset.evaluation.locomo import locomo_native_score
from mindset.evaluation.metrics import token_f1
from mindset.utils.hashing import stable_hash

ARMS = ('joint_ref', 'flat_ce_seed')


def complete_coverage(intervals, length):
    if length <= 0:
        return False
    end = 0
    for a, b in sorted(intervals):
        if not 0 <= a < b <= length:
            raise ValueError('Invalid source token interval')
        if a > end:
            return False
        end = max(end, b)
    return end == length


def source_coverage(events, selected, gold, unit, tokenizer):
    if unit not in ('episode', 'session'):
        raise ValueError('Unknown native retrieval unit')
    by_id = {e.event_id: e for e in events}
    if len(by_id) != len(events):
        raise ValueError('Duplicate source episodes')
    native = {e.event_id: e.session_id if unit == 'session' else e.event_id for e in events}
    ids = {}; intervals = {}
    for s in selected:
        if s['event_id'] not in by_id:
            raise ValueError('Unknown selected source')
        eid = s['event_id']; event = by_id[eid]
        ids[eid] = tokenizer.encode(render_event(event), add_special_tokens=False)
        a, b = s['token_start'], s['token_end']
        if not 0 <= a < b <= len(ids[eid]) or tokenizer.decode(ids[eid][a:b]) != s['text']:
            raise ValueError('Selected passage differs from source tokens')
        intervals.setdefault(eid, []).append((a, b))
    annotated = {e.event_id for e in events if native[e.event_id] in set(gold)}
    if set(gold) - set(native.values()):
        raise ValueError('Gold units not present in source history')
    covered = set()
    for eid in annotated:
        if eid not in intervals:
            continue
        if complete_coverage(intervals[eid], len(ids[eid])):
            covered.add(eid)
    units = {native[e] for e in intervals}
    return {'gold_unit_count': len(set(gold)), 'gold_raw_event_count': len(annotated),
        'selected_gold_raw_events': len(annotated & set(intervals)),
        'fully_visible_gold_raw_events': len(covered),
        'all_gold_units_retrieved': set(gold).issubset(units) if gold else None,
        'all_annotated_raw_event_text_visible': covered == annotated if annotated else None,
        'native_unit': unit, 'scope': 'annotation/token coverage only, not actual answer sufficiency'}


def run():
    if os.name == 'nt' or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('DGX Slurm only')
    manifest = ROOT / 'data/manifests/mindset_v2_development100.json'
    calibration = ROOT / 'outputs/acl/mindset_joint_v2_27058/calibration/calibration.json'
    cfg, cohort, _ = protocol(manifest, calibration)
    parent = ROOT / 'outputs/acl/raw_source_seeds_r01/answers'
    audit = read_json(parent / 'audit.json')
    if read_json(parent / 'status.json')['status'] != 'complete_development_answers':
        raise ValueError('Answer parent not terminal')
    checkpoint = read_json(ROOT / 'configs/checkpoints.lock.json')[cfg['evidence_tokenizer']]
    if any(audit.get(k) != v for k, v in {'manifest_sha256': digest_file(manifest),
        'calibration_sha256': digest_file(calibration),
        'client_module_sha256': digest_file(ROOT / 'src/mindset/acl/clients.py'),
        'reader_hashes': {'locomo': READER_HASH, 'longmemeval': ANSWER_PROMPT_HASH},
        'checkpoint_lock_sha256': digest_file(ROOT / 'configs/checkpoints.lock.json')}.items()):
        raise ValueError('Saved answer protocol changed')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint['checkpoint'], revision=checkpoint['revision'],
        cache_dir=str(Path.home() / 'MINDSET_v2/.cache/gpu_prototype/huggingface/hub'), local_files_only=True)
    saved = unique_rows(parent / 'predictions.jsonl')
    rows = []
    for dataset, cluster, cases in groups_for(cohort, cfg):
        for case in cases:
            for arm in ARMS:
                rid = f'{dataset}/{arm}/{case.id}'; row = saved[rid]
                if row['question'] != case.question or row['answer'] != case.answer or row['category'] != case.category:
                    raise ValueError('Saved QA/source identity changed')
                prompt = short_phrase_messages(case.question, row['evidence'], case.question_date,
                    category=case.category) if dataset == 'locomo' else messages(case.question, row['evidence'], case.question_date)
                if stable_hash(prompt) != row['prompt_sha256']:
                    raise ValueError('Saved prompt changed')
                metrics = rank_metrics(row['selected'], case, 8)
                if any(abs(metrics[m] - row['retrieval'][m]) > 1e-12 for m in metrics):
                    raise ValueError('Saved native-unit retrieval metric mismatch')
                score = token_f1(row['prediction'], case.answer)
                if abs(score - row['token_f1']) > 1e-12:
                    raise ValueError('Saved token F1 mismatch')
                if dataset == 'locomo':
                    category = next(int(k) for k, v in CATEGORY_NAMES.items() if v == case.category)
                    score = locomo_native_score(row['prediction'], case.answer, category)
                    if abs(score - row['native_f1']) > 1e-12:
                        raise ValueError('Saved native F1 mismatch')
                coverage = source_coverage(case.events, row['selected'], case.gold, case.retrieval_unit, tokenizer)
                rows.append({'id': rid, 'dataset': dataset, 'method': arm, 'question_id': case.id, 'cluster': cluster,
                    **coverage, 'score': score, 'score_zero': score == 0,
                    'score_type': 'native_locomo_f1' if dataset == 'locomo' else 'secondary_token_f1',
                    'incomplete_response': row['incomplete_response']})
    if len(rows) != 200:
        raise ValueError('Original100 paired audit incomplete')
    groups = {}
    for dataset in ('locomo', 'longmemeval'):
        for arm in ARMS:
            subset = [r for r in rows if r['dataset'] == dataset and r['method'] == arm]
            groups[f'{dataset}/{arm}'] = {'questions': len(subset), 'mean_score': float(np.mean([r['score'] for r in subset])),
                'no_gold_units': sum(r['gold_unit_count'] == 0 for r in subset),
                'all_gold_native_units_retrieved': sum(r['all_gold_units_retrieved'] is True for r in subset),
                'zero_score_with_all_gold_units_retrieved': sum(r['score_zero'] and r['all_gold_units_retrieved'] is True for r in subset),
                'all_annotated_raw_text_visible': sum(r['all_annotated_raw_event_text_visible'] is True for r in subset),
                'zero_score_despite_all_annotated_raw_text_visible': sum(r['score_zero'] and r['all_annotated_raw_event_text_visible'] is True for r in subset)}
    out = ROOT / 'outputs/acl' / f'answer_source_bottlenecks_{os.environ["SLURM_JOB_ID"]}'
    out.mkdir(parents=True, exist_ok=True); record_environment(out)
    report = {'groups': groups, 'cases': rows, 'parent_hashes': {name: digest_file(parent / name)
        for name in ('audit.json', 'predictions.jsonl')}, 'worker_sha256': digest_file(__file__),
        'source_coverage_is_not_qa_oracle_or_reader_error_certificate': True,
        'gold_used_only_by_evaluator_never_method_or_model_inputs': True,
        'generation_calls': 0, 'full_runs_authorized': False}
    write_json(out / 'analysis.json', report, immutable=True)
    print('BOTTLENECKS', groups, flush=True)


if __name__ == '__main__':
    run()
