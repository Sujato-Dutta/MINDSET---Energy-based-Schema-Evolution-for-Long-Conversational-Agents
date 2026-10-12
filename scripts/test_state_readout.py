"""State-coupling development experiment. All computation requires DGX Slurm.

No tuning/selection on benchmark scores. Controlled checks are functional
mechanism tests, not evidence of natural-language OOD superiority. Fresh
process, uncached latency measurements are separate from shared-score quality.
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
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from test_mindset_causality import RuleMemory, NearestMergeMemory, flat_pool
from mindset.methods.compact import CompactConfig
from mindset.methods.state_readout import StateReadoutMixin, StateAwareMindsetMemory
from mindset.acl.joint_iteration import protocol, groups_for
from mindset.acl.pipeline import local_stack, record_environment, write_csv, rank_metrics, source_fingerprint
from mindset.acl.io import read_json, write_json, digest_file, append_row, unique_rows
from mindset.acl.clients import load_tokenizer, messages, ANSWER_PROMPT_HASH
from mindset.acl.evidence import EvidenceCache
from mindset.acl.runtime import CachedEncoder
from mindset.acl.latency import latency_summary
from mindset.utils.hashing import stable_hash

ARMS = ('full', 'state_off', 'rule', 'nearest_merge', 'flat48')


class StateRule(StateReadoutMixin, RuleMemory):
    pass


class StateNearest(StateReadoutMixin, NearestMergeMemory):
    pass


def build(arm, events, cfg, parameters, encoder, reranker):
    options = {**{k: v for k, v in cfg['mindset'].items() if k != 'profile'}, **parameters}
    cls = StateRule if arm == 'rule' else StateNearest if arm == 'nearest_merge' else StateAwareMindsetMemory
    m = cls(CompactConfig(**options), encoder=encoder, local_reranker=reranker,
            state_readout=arm not in ('state_off', 'flat48'))
    m.ingest(events)
    return m


def identity(manifest, calibration, cfg, artifact, scope):
    return {'scope': scope, 'source_fingerprint': source_fingerprint(),
            'worker_sha256': digest_file(Path(__file__)),
            'controller_control_sha256': digest_file(ROOT / 'scripts/test_mindset_causality.py'),
            'readout_sha256': digest_file(ROOT / 'src/mindset/methods/state_readout.py'),
            'manifest_sha256': digest_file(manifest), 'calibration_sha256': digest_file(calibration),
            'parameters': artifact['selected_parameters'], 'new_fitted_read_scalars': 0,
            'checkpoint_lock': read_json(ROOT / 'configs/checkpoints.lock.json'),
            'arms': list(ARMS), 'answer_prompt_sha256': ANSWER_PROMPT_HASH,
            'budgets': {'logical_candidates_per_arm': 48, 'raw_episodes': 8, 'evidence_tokens': 8000},
            'labels': 'gold used only after selection; natural state accuracy and native LME judge unavailable',
            'parameter_policy': 'unchanged independent synthetic write calibration; no benchmark fitting',
            'lightmem': 'preserved unchanged; not part of within-MINDSET causal controls',
            'full_runs_authorized': False}


def summarize(out):
    rows = list(unique_rows(out / 'evidence.jsonl').values())
    report, differences = [], []
    for dataset in ('locomo', 'longmemeval'):
        full = {r['question_id']: r for r in rows if r['dataset'] == dataset and r['arm'] == 'full'}
        for arm in ARMS:
            subset = [r for r in rows if r['dataset'] == dataset and r['arm'] == arm]
            if not subset: continue
            report.append({'dataset': dataset, 'arm': arm, 'questions': len(subset),
                **{k: float(np.mean([r['retrieval'][k] for r in subset]))
                   for k in ('recall@8', 'precision@8', 'mrr@8', 'ndcg@8')},
                'candidate_recall@48': float(np.mean([r['candidate_recall@48'] for r in subset])),
                'mean_evidence_tokens': float(np.mean([r['evidence_tokens'] for r in subset])),
                'scoped_queries': sum(r['readout_certificate']['grounded_candidates'] > 0 for r in subset),
                'queries_with_exclusion': sum(bool(r['readout_certificate']['excluded']) for r in subset)})
            pairs = [(full[r['question_id']], r) for r in subset if r['question_id'] in full]
            if arm != 'full' and pairs:
                differences.append({'dataset': dataset, 'control': arm, 'pairs': len(pairs),
                    'identical_complete_answer_prompts': sum(a['prompt_sha256'] == b['prompt_sha256'] for a, b in pairs),
                    **{k + '_delta': float(np.mean([a['retrieval'][k] - b['retrieval'][k] for a, b in pairs]))
                       for k in ('recall@8', 'precision@8', 'mrr@8', 'ndcg@8')}})
    if report: write_csv(out / 'retrieval.csv', report)
    if differences: write_csv(out / 'causal_opportunity.csv', differences)
    write_json(out / 'status.json', {'completed_cells': len(rows), 'expected_cells': 100 * len(ARMS),
        'status': 'complete_development' if len(rows) == 100 * len(ARMS) else 'partial_development',
        'qa_calls': 0, 'judge_calls': 0, 'full_runs_authorized': False,
        'scope': 'repeated-look development; descriptive retrieval not confirmatory answer results'})


def quality(manifest, calibration, output, previous):
    cfg, cohort, artifact = protocol(manifest, calibration)
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (out / 'worker.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        audit = identity(manifest, calibration, cfg, artifact, 'state_readout_quality_development100')
        audit.update(previous_evidence_sha256=digest_file(previous),
                     score_reuse='shared CE pair scores for quality only; latency measured separately')
        write_json(out / 'audit.json', audit, immutable=True); record_environment(out)
        old = {(r['dataset'], r['question_id']): r for r in unique_rows(previous).values() if r['arm'] == 'full'}
        if set(old) != {(r['dataset'], r['id']) for r in cohort['records']}:
            raise ValueError('Old full controller must cover the exact development manifest')
        done = unique_rows(out / 'evidence.jsonl')
        expected = {f"{r['dataset']}/{a}/{r['id']}" for r in cohort['records'] for a in ARMS}
        if set(done) - expected: raise ValueError('Unexpected resumed IDs')
        encoder, reranker = local_stack(cfg, cache=False); encoder = CachedEncoder(encoder, enabled=True)
        tokenizer = load_tokenizer(read_json(ROOT / 'configs/checkpoints.lock.json')[cfg['evidence_tokenizer']])
        checks = []
        for dataset, cluster, queries in groups_for(cohort, cfg):
            pending = [q for q in queries if any(f'{dataset}/{a}/{q.id}' not in done for a in ARMS)]
            if not pending: continue
            memories = {}
            for arm in ARMS:
                if arm == 'flat48': memories[arm] = memories['state_off']; continue
                print(f'Build {dataset}/{cluster}/{arm}', flush=True)
                m = build(arm, queries[0].events, cfg, artifact['selected_parameters'], encoder, reranker)
                memories[arm] = m
                write_json(out / 'structure' / f'{dataset}-{cluster}-{arm}.json',
                    {'invariants': m.validate_invariants(), 'telemetry': m.telemetry()})
            base = memories['full']; index = EvidenceCache(queries[0].events, encoder, tokenizer, cfg)
            if any(m._episode_order != base._episode_order or m._documents != base._documents for m in memories.values()):
                raise ValueError('Controller arms changed raw episode identity/order/text')
            for case in pending:
                arms = [a for a in ARMS if f'{dataset}/{a}/{case.id}' not in done]
                pools = {a: flat_pool(memories[a], case.question) if a == 'flat48'
                         else memories[a].candidate_pool(case.question)[0] for a in arms}
                union = sorted({i for pool in pools.values() for i in pool})
                scores = np.asarray(reranker.predict([(case.question, base._documents[i]) for i in union],
                    batch_size=8, show_progress_bar=False), dtype=float)
                if len(scores) != len(union) or not np.isfinite(scores).all(): raise ValueError('Invalid CE scores')
                by_index = dict(zip(union, scores, strict=True))
                unit = {e.event_id: e.session_id if case.retrieval_unit == 'session' else e.event_id for e in case.events}
                for arm in arms:
                    m, pool = memories[arm], pools[arm]
                    hits = m.select_scored(case.question, pool, [by_index[i] for i in pool], 8, case.question_date)
                    selected, packet = index.packet(hits, case.question)
                    prompt = messages(case.question, packet, case.question_date)
                    row = {'id': f'{dataset}/{arm}/{case.id}', 'dataset': dataset, 'question_id': case.id,
                        'cluster': cluster, 'category': case.category, 'arm': arm, 'question': case.question,
                        'question_date': case.question_date, 'answer': case.answer, 'selected': selected,
                        'evidence': packet, 'evidence_tokens': len(tokenizer.encode(packet, add_special_tokens=False)),
                        'messages': prompt, 'prompt_sha256': stable_hash(prompt), 'candidate_count': len(pool),
                        'candidate_recall@48': len({unit[base._episode_order[i]] for i in pool} & set(case.gold)) / len(set(case.gold)),
                        'readout_certificate': m.last_readout_certificate, 'retrieval': rank_metrics(selected, case, 8),
                        'status': 'development_not_paper'}
                    if arm == 'state_off':
                        same = row['prompt_sha256'] == old[(dataset, case.id)]['prompt_sha256']
                        checks.append({'dataset': dataset, 'question_id': case.id, 'exact_old_full_prompt': same})
                        if not same: raise ValueError('State-off failed exact old-controller reproduction')
                    append_row(out / 'evidence.jsonl', row); done[row['id']] = row
                summarize(out); print(f'{len(done)}/500 state-readout cells', flush=True)
            encoder.cache.clear(); del memories, index
        # Recover full reproducibility checks on resumed rows too.
        checks = [{'dataset': r['dataset'], 'question_id': r['question_id'],
                   'exact_old_full_prompt': r['prompt_sha256'] == old[(r['dataset'], r['question_id'])]['prompt_sha256']}
                  for r in done.values() if r['arm'] == 'state_off']
        if not all(r['exact_old_full_prompt'] for r in checks): raise ValueError('Resumed reproduction failed')
        write_csv(out / 'old_controller_reproduction.csv', checks); summarize(out)


def latency_worker(manifest, calibration, output, repetition):
    cfg, cohort, artifact = protocol(manifest, calibration)
    out = Path(output)
    if out.exists() and any(out.iterdir()): raise ValueError('Fresh uncached latency directory required')
    out.mkdir(parents=True)
    write_json(out / 'audit.json', identity(manifest, calibration, cfg, artifact, 'fresh_uncached_retrieval_latency'))
    record_environment(out)
    encoder, reranker = local_stack(cfg, cache=False)
    tokenizer = load_tokenizer(read_json(ROOT / 'configs/checkpoints.lock.json')[cfg['evidence_tokenizer']])
    rng = np.random.default_rng(cfg['seed'] + repetition)
    for dataset, cluster, queries in groups_for(cohort, cfg):
        for arm in rng.permutation(ARMS).tolist():
            started = perf_counter()
            m = build(arm, queries[0].events, cfg, artifact['selected_parameters'], encoder, reranker)
            ingest = perf_counter() - started
            append_row(out / 'ingestion.jsonl', {'id': f'{dataset}/{arm}/{cluster}', 'dataset': dataset,
                'arm': arm, 'cluster': cluster, 'seconds': ingest, 'episodes': len(queries[0].events)})
            # Same discarded warmup for all arms. No score or completion cache.
            m.retrieve('Conversation facts and events', 8)
            for case in queries:
                start = perf_counter()
                if arm == 'flat48':
                    pool = flat_pool(m, case.question)
                    scores = reranker.predict([(case.question, m._documents[i]) for i in pool], batch_size=8, show_progress_bar=False)
                    hits = m.select_scored(case.question, pool, scores, 8, case.question_date)
                else:
                    hits = m.retrieve(case.question, 8, case.question_date)
                selected, packet = EvidenceCache(case.events, encoder, tokenizer, cfg).packet(hits, case.question)
                elapsed = perf_counter() - start
                append_row(out / 'queries.jsonl', {'id': f'{dataset}/{arm}/{case.id}', 'dataset': dataset,
                    'question_id': case.id, 'arm': arm, 'cluster': cluster, 'repetition': repetition,
                    'retrieval_seconds': elapsed, 'end_to_end_seconds': None, 'cache_hit': False,
                    'selected_ids': [s['event_id'] for s in selected], 'evidence_sha256': stable_hash(packet)})
            print(f'Latency rep={repetition} {dataset}/{cluster}/{arm}', flush=True)


def latency(manifest, calibration, output):
    out = Path(output)
    if out.exists() and any(out.iterdir()): raise ValueError('Fresh three-process latency output required')
    rows, environments, report = [], [], []
    for rep in range(3):
        folder = out / f'rep-{rep}'
        subprocess.run([sys.executable, str(Path(__file__)), 'latency-worker', '--manifest', str(manifest),
            '--calibration', str(calibration), '--output', str(folder), '--repetition', str(rep)], check=True, cwd=ROOT)
        env = read_json(folder / 'environment.json')
        environments.append({k: env[k] for k in ('hostname', 'cpu_affinity', 'dependencies', 'allocation')})
        subset = list(unique_rows(folder / 'queries.jsonl').values())
        if len(subset) != 500: raise ValueError('Incomplete latency repetition')
        rows.extend(subset)
    if any(e != environments[0] for e in environments): raise ValueError('Repetitions differ in environment')
    for dataset in ('locomo', 'longmemeval'):
        for arm in ARMS:
            subset = [r for r in rows if r['dataset'] == dataset and r['arm'] == arm]
            report.append({'dataset': dataset, 'arm': arm, 'repetitions': 3, **latency_summary(subset),
                'scope': 'same_node_serialized_fresh_processes_includes_cold_packet_work_no_answers'})
    write_csv(out / 'efficiency.csv', report)
    write_json(out / 'hardware.json', environments[0])
    write_json(out / 'status.json', {'status': 'complete_development_latency', 'qa_calls': 0,
        'end_to_end_status': 'not_measured_without_actual_answers', 'full_runs_authorized': False})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('quality', 'latency', 'latency-worker'))
    p.add_argument('--manifest', default=str(ROOT / 'data/manifests/mindset_v2_development100.json'))
    p.add_argument('--calibration', required=True); p.add_argument('--output', required=True)
    p.add_argument('--previous'); p.add_argument('--repetition', type=int, default=0)
    args = p.parse_args()
    if os.name == 'nt' or not os.environ.get('SLURM_JOB_ID'): p.error('DGX Slurm only')
    if args.command == 'quality':
        if not args.previous: p.error('--previous original causal evidence required')
        quality(args.manifest, args.calibration, args.output, Path(args.previous))
    elif args.command == 'latency': latency(args.manifest, args.calibration, args.output)
    else: latency_worker(args.manifest, args.calibration, args.output, args.repetition)


if __name__ == '__main__': main()
