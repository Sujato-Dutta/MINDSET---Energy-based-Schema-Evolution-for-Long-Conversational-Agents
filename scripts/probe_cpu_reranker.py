"""Same-48-candidate CPU INT8 drift/latency test; no metric-driven fitting."""
from pathlib import Path
from collections import defaultdict
from time import perf_counter
import argparse
import os
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'scripts'))
from test_aligned_readout import build
from mindset.acl.cpu_reranker import cpu_int8_reranker, tensor_bytes
from mindset.acl.joint_iteration import protocol, groups_for
from mindset.acl.pipeline import local_stack, record_environment, write_csv, rank_metrics, source_fingerprint
from mindset.acl.io import read_json, write_json, digest_file, append_row, unique_rows
from mindset.acl.clients import load_tokenizer, messages, ANSWER_PROMPT_HASH
from mindset.acl.evidence import EvidenceCache
from mindset.acl.runtime import CachedEncoder
from mindset.acl.latency import latency_summary
from mindset.utils.hashing import stable_hash


def worker(manifest, calibration, output, repetition, quality):
    cfg, cohort, artifact = protocol(manifest, calibration); out = Path(output)
    if out.exists() and any(out.iterdir()): raise ValueError('Fresh output required')
    out.mkdir(parents=True); record_environment(out)
    write_json(out / 'audit.json', {'scope': 'development_cpu_backend_quality' if quality else 'fresh_uncached_cpu_backend_latency',
        'source_fingerprint': source_fingerprint(), 'worker_sha256': digest_file(Path(__file__)),
        'shared_builder_sha256': digest_file(ROOT / 'scripts/test_aligned_readout.py'),
        'manifest_sha256': digest_file(manifest), 'calibration_sha256': digest_file(calibration),
        'checkpoint_lock': read_json(ROOT / 'configs/checkpoints.lock.json'), 'parameters': artifact['selected_parameters'],
        'answer_prompt_sha256': ANSWER_PROMPT_HASH, 'candidate_budget': 48, 'raw_episode_budget': 8,
        'backend': 'FP32 versus copied Torch Linear-only dynamic qint8; activation/tokenizer/checkpoint unchanged',
        'cache_policy': 'encoder reuse for QUALITY only' if quality else 'all query/score/packet caches disabled',
        'setup': 'conversion cost reported separately; not hidden inside lower query latency',
        'physical_energy': 'not measured; no electricity/joule claims', 'lightmem': 'implementation unchanged',
        'full_runs_authorized': False})
    encoder, fp32 = local_stack(cfg, cache=False)
    started = perf_counter(); int8 = cpu_int8_reranker(fp32); conversion = perf_counter() - started
    import torch
    write_json(out / 'backend.json', {'conversion_seconds': conversion, 'torch_version': torch.__version__,
        'quantized_engine': torch.backends.quantized.engine, 'fp32_logical_tensor_bytes': tensor_bytes(fp32),
        'int8_logical_tensor_bytes': tensor_bytes(int8), 'storage_scope': 'logical learned parameter/buffer/unpacked quantized-weight byte footprint; NOT serialized checkpoint, pipeline storage, runtime packing size or peak RSS'})
    if quality: encoder = CachedEncoder(encoder, enabled=True)
    tokenizer = load_tokenizer(read_json(ROOT / 'configs/checkpoints.lock.json')[cfg['evidence_tokenizer']])
    rng = np.random.default_rng(cfg['seed'] + repetition)
    for dataset, cluster, queries in groups_for(cohort, cfg):
        m = build('full', queries[0].events, cfg, artifact['selected_parameters'], encoder, fp32)
        # Discarded inference for both backends. No warm packet/query-score cache.
        for scorer in (fp32, int8):
            m.local_reranker = scorer; m.retrieve('Conversation facts and events', 8)
        for case in queries:
            observations = {}
            for backend in rng.permutation(('fp32', 'int8')).tolist():
                m.local_reranker = fp32 if backend == 'fp32' else int8
                started = perf_counter(); hits = m.retrieve(case.question, 8, case.question_date)
                selected, packet = EvidenceCache(case.events, encoder, tokenizer, cfg).packet(hits, case.question)
                elapsed = perf_counter() - started; prompt = messages(case.question, packet, case.question_date)
                row = {'id': f'{dataset}/{backend}/{case.id}', 'dataset': dataset, 'question_id': case.id,
                    'cluster': cluster, 'backend': backend, 'candidate_count': len(m.last_candidate_pool),
                    'candidate_ids': [m._episode_order[i] for i in m.last_candidate_pool], 'repetition': repetition,
                    'retrieval_seconds': None if quality else elapsed, 'end_to_end_seconds': None, 'cache_hit': False,
                    'selected': selected, 'messages': prompt, 'prompt_sha256': stable_hash(prompt),
                    'metrics': rank_metrics(selected, case, 8), 'certificate': m.last_readout_certificate}
                observations[backend] = row; append_row(out / 'queries.jsonl', row)
            if observations['fp32']['candidate_ids'] != observations['int8']['candidate_ids']:
                raise ValueError('Backend comparison changed candidate substrate')
            print(f'{dataset}/{case.id} backend pair complete', flush=True)
        if quality: encoder.cache.clear()
    write_json(out / 'status.json', {'status': 'complete_backend_quality' if quality else 'complete_backend_latency',
        'queries': 200, 'qa_calls': 0, 'judge_calls': 0, 'full_runs_authorized': False})


def summarize(output, quality):
    out = Path(output); reports = []
    folders = [out] if quality else [out / f'rep-{r}' for r in range(3)]
    rows = [r for folder in folders for r in unique_rows(folder / 'queries.jsonl').values()]
    if len(rows) != (200 if quality else 600): raise ValueError('Incomplete backend comparison')
    for dataset in ('locomo', 'longmemeval'):
        for backend in ('fp32', 'int8'):
            subset = [r for r in rows if r['dataset'] == dataset and r['backend'] == backend]
            reports.append({'dataset': dataset, 'backend': backend, 'rows': len(subset),
                **{k: float(np.mean([r['metrics'][k] for r in subset])) for k in ('recall@8', 'precision@8', 'mrr@8', 'ndcg@8')},
                **({} if quality else latency_summary(subset))})
    write_csv(out / ('quality.csv' if quality else 'efficiency.csv'), reports)
    differences = []
    for dataset in ('locomo', 'longmemeval'):
        fp = {(r['repetition'], r['question_id']): r for r in rows if r['dataset'] == dataset and r['backend'] == 'fp32'}
        pairs = [(fp[(r['repetition'], r['question_id'])], r) for r in rows if r['dataset'] == dataset and r['backend'] == 'int8']
        differences.append({'dataset': dataset, 'pairs': len(pairs), 'identical_complete_prompts': sum(a['prompt_sha256'] == b['prompt_sha256'] for a, b in pairs),
            **{k + '_int8_minus_fp32': float(np.mean([b['metrics'][k] - a['metrics'][k] for a, b in pairs]))
               for k in ('recall@8', 'precision@8', 'mrr@8', 'ndcg@8')}})
    write_csv(out / 'drift.csv', differences)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('command', choices=('quality', 'latency', 'worker'))
    p.add_argument('--manifest', default=str(ROOT / 'data/manifests/mindset_v2_development100.json'))
    p.add_argument('--calibration', required=True); p.add_argument('--output', required=True); p.add_argument('--repetition', type=int, default=0)
    args = p.parse_args()
    if os.name == 'nt' or not os.environ.get('SLURM_JOB_ID'): p.error('DGX Slurm only')
    out = Path(args.output)
    if args.command == 'quality':
        worker(args.manifest, args.calibration, out, 0, True); summarize(out, True)
    elif args.command == 'worker': worker(args.manifest, args.calibration, out, args.repetition, False)
    else:
        if out.exists() and any(out.iterdir()): raise ValueError('Fresh latency parent required')
        signatures = []
        for rep in range(3):
            folder = out / f'rep-{rep}'
            subprocess.run([sys.executable, str(Path(__file__)), 'worker', '--manifest', str(args.manifest),
                '--calibration', str(args.calibration), '--output', str(folder), '--repetition', str(rep)], check=True, cwd=ROOT)
            env = read_json(folder / 'environment.json'); signatures.append({k: env[k] for k in ('hostname', 'cpu_affinity', 'dependencies', 'allocation')})
        if any(e != signatures[0] for e in signatures): raise ValueError('Backend latency environments differ')
        summarize(out, False); write_json(out / 'hardware.json', signatures[0])
        write_json(out / 'status.json', {'status': 'complete_backend_latency', 'qa_calls': 0, 'end_to_end_status': 'not_measured_no_answers'})


if __name__ == '__main__': main()
