"""Audited DGX-only iteration: binding, parsing, state routing and fast24.

Frozen write scalars transferred to a changed parser, not re-fitted or claimed
to have been calibrated for that new parser. Development is repeatedly viewed.
"""
from pathlib import Path
import argparse
from collections import defaultdict
from time import perf_counter
import os
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'scripts'))
from test_mindset_causality import flat_pool
from probe_state_readout_cases import cases as generated_cases
from mindset.methods.base import ConversationEvent
from mindset.methods.compact import CompactConfig
from mindset.methods.state_readout import StateAwareMindsetMemory
from mindset.methods.aligned_readout import AlignedMindsetMemory
from mindset.acl.joint_iteration import protocol, groups_for
from mindset.acl.pipeline import local_stack, record_environment, write_csv, rank_metrics, source_fingerprint
from mindset.acl.io import read_json, write_json, digest_file, append_row, unique_rows
from mindset.acl.clients import load_tokenizer, messages, ANSWER_PROMPT_HASH
from mindset.acl.evidence import EvidenceCache
from mindset.acl.runtime import CachedEncoder
from mindset.acl.latency import latency_summary
from mindset.utils.hashing import stable_hash

ARMS = ('full', 'read_off', 'rule', 'nearest_merge', 'flat48', 'fast24', 'previous')
TIMED = ('full', 'read_off', 'fast24', 'previous')
METRICS = ('recall@8', 'precision@8', 'mrr@8', 'ndcg@8')


def build(arm, events, cfg, parameters, encoder, reranker):
    options = {**{k: v for k, v in cfg['mindset'].items() if k != 'profile'}, **parameters}
    if arm == 'fast24': options.update(retrieval_candidate_k=24, direct_pool_k=12, schema_pool_k=12)
    if arm in ('previous', 'flat48'):
        m = StateAwareMindsetMemory(CompactConfig(**options), encoder=encoder, local_reranker=reranker,
                                    state_readout=arm != 'flat48')
    else:
        m = AlignedMindsetMemory(CompactConfig(**options), encoder=encoder, local_reranker=reranker,
            state_readout=arm != 'read_off', write_policy='rule' if arm == 'rule' else 'nearest' if arm == 'nearest_merge' else 'energy')
    m.ingest(events)
    return m


def pool_for(arm, m, question, date):
    if arm == 'flat48': return flat_pool(m, question)
    if isinstance(m, AlignedMindsetMemory): return m.candidate_pool(question, date)[0]
    return m.candidate_pool(question)[0]


def identity(manifest, calibration, cfg, artifact, scope):
    return {'scope': scope, 'source_fingerprint': source_fingerprint(), 'worker_sha256': digest_file(Path(__file__)),
        'aligned_source_sha256': digest_file(ROOT / 'src/mindset/methods/aligned_readout.py'),
        'manifest_sha256': digest_file(manifest), 'calibration_sha256': digest_file(calibration),
        'parameters': artifact['selected_parameters'], 'coefficient_policy': 'frozen scalar transfer; no parser-specific recalibration or benchmark fitting',
        'checkpoint_lock': read_json(ROOT / 'configs/checkpoints.lock.json'), 'arms': list(ARMS),
        'budgets': {a: {'ce_candidates': 24 if a == 'fast24' else 48, 'raw_episodes': 8, 'evidence_tokens': 8000} for a in ARMS},
        'answer_prompt_sha256': ANSWER_PROMPT_HASH, 'new_fitted_read_scalars': 0,
        'labels': 'native retrieval gold after selection only; natural state and native LME judge scores unavailable',
        'controls': 'full/read_off/rule/nearest at48; fast24 intentionally not search-budget matched',
        'lightmem': 'unchanged; not an arm in this within-MINDSET diagnostic', 'full_runs_authorized': False}


def summarize(out):
    rows = list(unique_rows(out / 'evidence.jsonl').values()); report, differences = [], []
    for dataset in ('locomo', 'longmemeval'):
        full = {r['question_id']: r for r in rows if r['dataset'] == dataset and r['arm'] == 'full'}
        for arm in ARMS:
            subset = [r for r in rows if r['dataset'] == dataset and r['arm'] == arm]
            if not subset: continue
            report.append({'dataset': dataset, 'arm': arm, 'questions': len(subset),
                **{k: float(np.mean([r['retrieval'][k] for r in subset])) for k in METRICS},
                'candidate_recall': float(np.mean([r['candidate_recall'] for r in subset])),
                'mean_evidence_tokens': float(np.mean([r['evidence_tokens'] for r in subset])),
                'grounded_queries': sum(r['certificate']['grounded_candidates'] > 0 for r in subset),
                'queries_with_exclusion': sum(bool(r['certificate']['excluded']) for r in subset),
                'first_person_bindings': sum(r['certificate'].get('identity_binding') == 'first_person_bound_to_literal_user_role' for r in subset)})
            pairs = [(full[r['question_id']], r) for r in subset if r['question_id'] in full]
            if arm != 'full' and pairs:
                differences.append({'dataset': dataset, 'control': arm, 'pairs': len(pairs),
                    'identical_complete_answer_prompts': sum(a['prompt_sha256'] == b['prompt_sha256'] for a, b in pairs),
                    **{k + '_delta': float(np.mean([a['retrieval'][k] - b['retrieval'][k] for a, b in pairs])) for k in METRICS}})
    if report: write_csv(out / 'retrieval.csv', report)
    if differences: write_csv(out / 'causal_opportunity.csv', differences)
    write_json(out / 'status.json', {'completed_cells': len(rows), 'expected_cells': 100 * len(ARMS),
        'status': 'complete_development' if len(rows) == 100 * len(ARMS) else 'partial_development',
        'qa_calls': 0, 'judge_calls': 0, 'full_runs_authorized': False,
        'scope': 'repeated-look development; no answer-performance or physical-energy-efficiency claim'})


def quality(manifest, calibration, output):
    cfg, cohort, artifact = protocol(manifest, calibration); out = Path(output); out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (out / 'worker.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        audit = identity(manifest, calibration, cfg, artifact, 'aligned_readout_development100_quality_not_latency')
        previous = ROOT / 'outputs/acl/mindset_state_readout_27180/quality/evidence.jsonl'
        audit.update(previous_evidence_sha256=digest_file(previous), score_reuse='shared CE pairs for quality ONLY')
        write_json(out / 'audit.json', audit, immutable=True); record_environment(out)
        old = {(r['dataset'], r['question_id']): r for r in unique_rows(previous).values() if r['arm'] == 'full'}
        if set(old) != {(r['dataset'], r['id']) for r in cohort['records']}: raise ValueError('Old manifest mismatch')
        done = unique_rows(out / 'evidence.jsonl'); expected = {f"{r['dataset']}/{a}/{r['id']}" for r in cohort['records'] for a in ARMS}
        if set(done) - expected: raise ValueError('Unexpected resumed IDs')
        encoder, reranker = local_stack(cfg, cache=False); encoder = CachedEncoder(encoder, enabled=True)
        tokenizer = load_tokenizer(read_json(ROOT / 'configs/checkpoints.lock.json')[cfg['evidence_tokenizer']])
        for dataset, cluster, queries in groups_for(cohort, cfg):
            pending = [q for q in queries if any(f'{dataset}/{a}/{q.id}' not in done for a in ARMS)]
            if not pending: continue
            memories = {}
            for arm in ARMS:
                print(f'Build {dataset}/{cluster}/{arm}', flush=True)
                m = build(arm, queries[0].events, cfg, artifact['selected_parameters'], encoder, reranker); memories[arm] = m
                write_json(out / 'structure' / f'{dataset}-{cluster}-{arm}.json', {'invariants': m.validate_invariants(), 'telemetry': m.telemetry()})
            base = memories['full']; index = EvidenceCache(queries[0].events, encoder, tokenizer, cfg)
            if any(m._episode_order != base._episode_order or m._documents != base._documents for m in memories.values()):
                raise ValueError('Parsing adapter changed raw evidence')
            for case in pending:
                arms = [a for a in ARMS if f'{dataset}/{a}/{case.id}' not in done]
                pools = {a: pool_for(a, memories[a], case.question, case.question_date) for a in arms}
                union = sorted({i for pool in pools.values() for i in pool})
                scores = np.asarray(reranker.predict([(case.question, base._documents[i]) for i in union], batch_size=8, show_progress_bar=False), dtype=float)
                if len(scores) != len(union) or not np.isfinite(scores).all(): raise ValueError('Invalid CE scores')
                by_index = dict(zip(union, scores, strict=True))
                unit = {e.event_id: e.session_id if case.retrieval_unit == 'session' else e.event_id for e in case.events}
                for arm in arms:
                    m, pool = memories[arm], pools[arm]
                    hits = m.select_scored(case.question, pool, [by_index[i] for i in pool], 8, case.question_date)
                    selected, packet = index.packet(hits, case.question); prompt = messages(case.question, packet, case.question_date)
                    row = {'id': f'{dataset}/{arm}/{case.id}', 'dataset': dataset, 'question_id': case.id, 'cluster': cluster,
                        'category': case.category, 'arm': arm, 'question': case.question, 'question_date': case.question_date,
                        'answer': case.answer, 'selected': selected, 'evidence': packet, 'messages': prompt,
                        'prompt_sha256': stable_hash(prompt), 'evidence_tokens': len(tokenizer.encode(packet, add_special_tokens=False)),
                        'candidate_count': len(pool), 'candidate_recall': len({unit[base._episode_order[i]] for i in pool} & set(case.gold)) / len(set(case.gold)),
                        'certificate': m.last_readout_certificate, 'retrieval': rank_metrics(selected, case, 8), 'status': 'development_not_paper'}
                    if arm == 'previous' and row['prompt_sha256'] != old[(dataset, case.id)]['prompt_sha256']:
                        raise ValueError('Previous arm failed exact frozen-prompt reproduction')
                    append_row(out / 'evidence.jsonl', row); done[row['id']] = row
                summarize(out); print(f'{len(done)}/700 aligned quality cells', flush=True)
            encoder.cache.clear(); del memories, index
        summarize(out)


def controlled(manifest, calibration, output):
    cfg, _, artifact = protocol(manifest, calibration); out = Path(output)
    if out.exists() and any(out.iterdir()): raise ValueError('Fresh controlled output required')
    out.mkdir(parents=True); generated = generated_cases(); role_cases = []
    for case in generated:
        q = case['question'].replace(case['speaker'], 'I').replace('does I', 'do I')
        role_cases.append({**case, 'id': 'role/' + case['id'], 'speaker': 'user', 'question': q})
    all_cases = generated + role_cases
    write_json(out / 'cases.json', all_cases, immutable=True)
    write_json(out / 'audit.json', {**identity(manifest, calibration, cfg, artifact, 'functional_template_tests_not_natural_ood'),
        'score_modes': ['actual_pinned_ce', 'adversarial_older_first'], 'case_count': 96})
    record_environment(out); encoder, reranker = local_stack(cfg, cache=False); encoder = CachedEncoder(encoder, enabled=True); rows = []
    for case in all_cases:
        events = [ConversationEvent(f'e{i}', case['id'], f's{i}', f'2025-01-{i:02d}', case['speaker'], text) for i, text in enumerate(case['texts'], 1)]
        memories = {a: build(a, events, cfg, artifact['selected_parameters'], encoder, reranker) for a in ARMS}
        base = memories['full']; pool = list(range(len(events)))
        actual = reranker.predict([(case['question'], d) for d in base._documents], batch_size=8, show_progress_bar=False)
        for mode, scores in (('actual_pinned_ce', actual), ('adversarial_older_first', np.arange(len(events), 0, -1, dtype=float))):
            for arm, m in memories.items():
                selected = [h.memory_id for h in m.select_scored(case['question'], pool, scores, 8)]
                required = set(case['required'])
                row = {'id': f"{case['id']}/{mode}/{arm}", 'role_bound_case': case['id'].startswith('role/'),
                    'kind': case['kind'], 'arm': arm, 'score_mode': mode, 'required': case['required'], 'selected': selected,
                    'exact_scoped_packet': set(selected) == required, 'required_source_recall': len(required & set(selected)) / len(required),
                    'top1_expected_state': selected[0] in required if len(required) == 1 else None, 'certificate': m.last_readout_certificate}
                append_row(out / 'results.jsonl', row); rows.append(row)
        encoder.cache.clear()
    report = []
    for role in (False, True):
        for mode in ('actual_pinned_ce', 'adversarial_older_first'):
            for arm in ARMS:
                subset = [r for r in rows if r['role_bound_case'] == role and r['score_mode'] == mode and r['arm'] == arm]
                report.append({'role_bound_cases': role, 'score_mode': mode, 'arm': arm, 'cases': len(subset),
                    'exact_packet_rate': float(np.mean([r['exact_scoped_packet'] for r in subset])),
                    'required_source_recall': float(np.mean([r['required_source_recall'] for r in subset]))})
    write_csv(out / 'summary.csv', report)
    write_json(out / 'status.json', {'status': 'complete_functional_probes', 'cells': len(rows), 'qa_calls': 0,
        'natural_state_accuracy': None, 'scope': 'template functionality only; not native QA or natural OOD'})


def latency_worker(manifest, calibration, output, repetition):
    cfg, cohort, artifact = protocol(manifest, calibration); out = Path(output)
    if out.exists() and any(out.iterdir()): raise ValueError('Fresh latency repetition required')
    out.mkdir(parents=True); write_json(out / 'audit.json', identity(manifest, calibration, cfg, artifact, 'fresh_uncached_retrieval_latency'))
    record_environment(out); encoder, reranker = local_stack(cfg, cache=False)
    tokenizer = load_tokenizer(read_json(ROOT / 'configs/checkpoints.lock.json')[cfg['evidence_tokenizer']])
    rng = np.random.default_rng(cfg['seed'] + repetition)
    for dataset, cluster, queries in groups_for(cohort, cfg):
        for arm in rng.permutation(TIMED).tolist():
            start = perf_counter(); m = build(arm, queries[0].events, cfg, artifact['selected_parameters'], encoder, reranker)
            append_row(out / 'ingestion.jsonl', {'id': f'{dataset}/{arm}/{cluster}', 'dataset': dataset, 'arm': arm,
                'cluster': cluster, 'seconds': perf_counter() - start, 'episodes': len(queries[0].events)})
            m.retrieve('Conversation facts and events', 8)
            for case in queries:
                start = perf_counter(); hits = m.retrieve(case.question, 8, case.question_date)
                selected, packet = EvidenceCache(case.events, encoder, tokenizer, cfg).packet(hits, case.question)
                elapsed = perf_counter() - start
                append_row(out / 'queries.jsonl', {'id': f'{dataset}/{arm}/{case.id}', 'dataset': dataset,
                    'question_id': case.id, 'arm': arm, 'repetition': repetition, 'cluster': cluster,
                    'retrieval_seconds': elapsed, 'end_to_end_seconds': None, 'cache_hit': False,
                    'evidence_sha256': stable_hash(packet), 'ce_candidates': len(m.last_candidate_pool)})
            print(f'Latency rep={repetition} {dataset}/{cluster}/{arm}', flush=True)


def latency(manifest, calibration, output):
    out = Path(output)
    if out.exists() and any(out.iterdir()): raise ValueError('Fresh latency directory required')
    rows, environments, reports = [], [], []
    for rep in range(3):
        folder = out / f'rep-{rep}'
        subprocess.run([sys.executable, str(Path(__file__)), 'latency-worker', '--manifest', str(manifest), '--calibration', str(calibration),
            '--output', str(folder), '--repetition', str(rep)], check=True, cwd=ROOT)
        env = read_json(folder / 'environment.json'); environments.append({k: env[k] for k in ('hostname', 'cpu_affinity', 'dependencies', 'allocation')})
        subset = list(unique_rows(folder / 'queries.jsonl').values())
        if len(subset) != 400: raise ValueError('Incomplete latency repetition')
        rows.extend(subset)
    if any(e != environments[0] for e in environments): raise ValueError('Latency environments differ')
    for dataset in ('locomo', 'longmemeval'):
        for arm in TIMED:
            subset = [r for r in rows if r['dataset'] == dataset and r['arm'] == arm]
            reports.append({'dataset': dataset, 'arm': arm, 'repetitions': 3, **latency_summary(subset),
                'scope': 'same-node uncached retrieval+packet, no answers; fast24 has half candidate budget'})
    write_csv(out / 'efficiency.csv', reports); write_json(out / 'hardware.json', environments[0])
    write_json(out / 'status.json', {'status': 'complete_development_latency', 'qa_calls': 0, 'end_to_end_status': 'not_measured_no_answers'})


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('command', choices=('controlled', 'quality', 'latency', 'latency-worker'))
    p.add_argument('--manifest', default=str(ROOT / 'data/manifests/mindset_v2_development100.json'))
    p.add_argument('--calibration', required=True); p.add_argument('--output', required=True); p.add_argument('--repetition', type=int, default=0)
    args = p.parse_args()
    if os.name == 'nt' or not os.environ.get('SLURM_JOB_ID'): p.error('All experiment computation requires DGX Slurm')
    if args.command == 'quality': quality(args.manifest, args.calibration, args.output)
    elif args.command == 'controlled': controlled(args.manifest, args.calibration, args.output)
    elif args.command == 'latency': latency(args.manifest, args.calibration, args.output)
    else: latency_worker(args.manifest, args.calibration, args.output, args.repetition)


if __name__ == '__main__': main()
