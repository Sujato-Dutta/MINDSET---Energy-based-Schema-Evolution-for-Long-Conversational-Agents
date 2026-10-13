"""Same-budget development test of state-derived packet energy; no retuning."""
from pathlib import Path
import argparse
import os
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'scripts'))
from test_mindset_causality import flat_pool
from mindset.methods.compact import CompactConfig
from mindset.methods.packet_energy import PacketEnergyMindsetMemory
from mindset.acl.joint_iteration import protocol, groups_for
from mindset.acl.pipeline import local_stack, record_environment, write_csv, rank_metrics, source_fingerprint
from mindset.acl.io import read_json, write_json, digest_file, append_row, unique_rows
from mindset.acl.clients import load_tokenizer, messages, ANSWER_PROMPT_HASH
from mindset.acl.evidence import EvidenceCache
from mindset.acl.runtime import CachedEncoder
from mindset.utils.hashing import stable_hash

ARMS = ('full', 'packet_off', 'rule', 'nearest_merge', 'no_entity', 'no_schema', 'flat48')
METRICS = ('recall@8', 'precision@8', 'mrr@8', 'ndcg@8')


def summarize(out):
    rows = list(unique_rows(out / 'evidence.jsonl').values()); reports, differences = [], []
    for dataset in ('locomo', 'longmemeval'):
        full = {r['question_id']: r for r in rows if r['dataset'] == dataset and r['arm'] == 'full'}
        for arm in ARMS:
            subset = [r for r in rows if r['dataset'] == dataset and r['arm'] == arm]
            if not subset: continue
            reports.append({'dataset': dataset, 'arm': arm, 'questions': len(subset),
                **{k: float(np.mean([r['retrieval'][k] for r in subset])) for k in METRICS},
                'mean_evidence_tokens': float(np.mean([r['evidence_tokens'] for r in subset]))})
            pairs = [(full[r['question_id']], r) for r in subset if r['question_id'] in full]
            if arm != 'full' and pairs:
                differences.append({'dataset': dataset, 'control': arm, 'pairs': len(pairs),
                    'identical_complete_prompts': sum(a['prompt_sha256'] == b['prompt_sha256'] for a, b in pairs),
                    **{k + '_delta': float(np.mean([a['retrieval'][k] - b['retrieval'][k] for a, b in pairs])) for k in METRICS}})
    if reports: write_csv(out / 'retrieval.csv', reports)
    if differences: write_csv(out / 'causal_opportunity.csv', differences)
    write_json(out / 'status.json', {'completed_cells': len(rows), 'expected_cells': 700,
        'status': 'complete_development' if len(rows) == 700 else 'partial_development',
        'qa_calls': 0, 'judge_calls': 0, 'full_runs_authorized': False,
        'scope': 'repeated-look descriptive development; no native answer or novelty superiority claim'})


def run(manifest, calibration, output):
    cfg, cohort, artifact = protocol(manifest, calibration); out = Path(output); out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (out / 'worker.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = ROOT / 'outputs/acl/mindset_aligned_27208/quality/evidence.jsonl'
        old = {(r['dataset'], r['question_id']): r for r in unique_rows(previous).values() if r['arm'] == 'full'}
        if set(old) != {(r['dataset'], r['id']) for r in cohort['records']}: raise ValueError('Previous manifest mismatch')
        write_json(out / 'audit.json', {'scope': 'development100_schema_packet_energy', 'source_fingerprint': source_fingerprint(),
            'worker_sha256': digest_file(Path(__file__)), 'previous_evidence_sha256': digest_file(previous),
            'manifest_sha256': digest_file(manifest), 'calibration_sha256': digest_file(calibration),
            'parameters': artifact['selected_parameters'], 'coefficient_policy': 'frozen lambda reused for read redundancy; no fitting; not separately calibrated',
            'checkpoint_lock': read_json(ROOT / 'configs/checkpoints.lock.json'), 'arms': list(ARMS),
            'answer_prompt_sha256': ANSWER_PROMPT_HASH, 'candidate_budget': 48, 'raw_episode_budget': 8,
            'solver': 'greedy marginal approximation; global optimum not claimed',
            'ce_reuse': 'shared quality scores only; never use these elapsed times as latency',
            'new_fitted_read_scalars': 0, 'lightmem': 'unchanged', 'full_runs_authorized': False}, immutable=True)
        record_environment(out); done = unique_rows(out / 'evidence.jsonl')
        expected = {f"{r['dataset']}/{a}/{r['id']}" for r in cohort['records'] for a in ARMS}
        if set(done) - expected: raise ValueError('Unexpected resumed IDs')
        encoder, reranker = local_stack(cfg, cache=False); encoder = CachedEncoder(encoder, enabled=True)
        tokenizer = load_tokenizer(read_json(ROOT / 'configs/checkpoints.lock.json')[cfg['evidence_tokenizer']])
        params = {**{k: v for k, v in cfg['mindset'].items() if k != 'profile'}, **artifact['selected_parameters']}
        for dataset, cluster, queries in groups_for(cohort, cfg):
            pending = [q for q in queries if any(f'{dataset}/{a}/{q.id}' not in done for a in ARMS)]
            if not pending: continue
            memories = {}
            for arm in ARMS:
                m = PacketEnergyMindsetMemory(CompactConfig(**params), encoder=encoder, local_reranker=reranker,
                    packet_energy=arm not in ('packet_off', 'flat48'), entity_energy=arm != 'no_entity', schema_energy=arm != 'no_schema',
                    state_readout=arm != 'flat48', write_policy='rule' if arm == 'rule' else 'nearest' if arm == 'nearest_merge' else 'energy')
                m.ingest(queries[0].events); memories[arm] = m
                write_json(out / 'structure' / f'{dataset}-{cluster}-{arm}.json', {'invariants': m.validate_invariants(), 'telemetry': m.telemetry()})
            base = memories['full']; index = EvidenceCache(queries[0].events, encoder, tokenizer, cfg)
            if any(m._documents != base._documents or m._episode_order != base._episode_order for m in memories.values()): raise ValueError('Raw evidence changed')
            for case in pending:
                arms = [a for a in ARMS if f'{dataset}/{a}/{case.id}' not in done]
                pools = {a: flat_pool(memories[a], case.question) if a == 'flat48' else memories[a].candidate_pool(case.question, case.question_date)[0] for a in arms}
                union = sorted({i for pool in pools.values() for i in pool})
                values = reranker.predict([(case.question, base._documents[i]) for i in union], batch_size=8, show_progress_bar=False)
                if len(values) != len(union) or not np.isfinite(values).all(): raise ValueError('Invalid CE scores')
                by_index = dict(zip(union, values, strict=True))
                for arm in arms:
                    m, pool = memories[arm], pools[arm]
                    hits = m.select_scored(case.question, pool, [by_index[i] for i in pool], 8, case.question_date)
                    selected, packet = index.packet(hits, case.question); prompt = messages(case.question, packet, case.question_date)
                    row = {'id': f'{dataset}/{arm}/{case.id}', 'dataset': dataset, 'question_id': case.id, 'cluster': cluster,
                        'category': case.category, 'arm': arm, 'question': case.question, 'question_date': case.question_date,
                        'answer': case.answer, 'selected': selected, 'evidence': packet, 'messages': prompt,
                        'prompt_sha256': stable_hash(prompt), 'evidence_tokens': len(tokenizer.encode(packet, add_special_tokens=False)),
                        'candidate_count': len(pool), 'retrieval': rank_metrics(selected, case, 8), 'certificate': m.last_readout_certificate}
                    if arm == 'packet_off' and row['prompt_sha256'] != old[(dataset, case.id)]['prompt_sha256']:
                        raise ValueError('Packet-off failed previous complete-prompt reproduction')
                    append_row(out / 'evidence.jsonl', row); done[row['id']] = row
                summarize(out); print(f'{len(done)}/700 packet-energy cells', flush=True)
            encoder.cache.clear(); del memories, index
        summarize(out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--manifest', default=str(ROOT / 'data/manifests/mindset_v2_development100.json'))
    p.add_argument('--calibration', required=True); p.add_argument('--output', required=True); args = p.parse_args()
    if os.name == 'nt' or not os.environ.get('SLURM_JOB_ID'): p.error('DGX Slurm only')
    run(args.manifest, args.calibration, args.output)
