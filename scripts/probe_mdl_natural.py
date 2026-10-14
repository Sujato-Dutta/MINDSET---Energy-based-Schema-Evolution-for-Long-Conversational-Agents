"""Matched zero-API MDL comparison on the exact frozen development-100."""
from pathlib import Path
from collections import defaultdict
import argparse
import os
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mindset.methods.mdl_state import MDLEnergyMemory
from mindset.methods.factorized_state import FactorizedMindsetMemory
from mindset.methods.compact import CompactConfig
from mindset.acl.joint_iteration import protocol, groups_for
from mindset.acl.pipeline import local_stack, record_environment, source_fingerprint, write_csv, rank_metrics
from mindset.acl.io import append_row, read_json, write_json, digest_file, unique_rows
from mindset.acl.runtime import CachedEncoder
from mindset.acl.evidence import EvidenceCache
from mindset.acl.clients import load_tokenizer, messages, ANSWER_PROMPT_HASH
from mindset.retrieval.hybrid import reciprocal_rank_fusion, ranked_indices
from mindset.utils.hashing import stable_hash

ARMS = ('mdl', 'mdl_read_off', 'mdl_no_boundary', 'mdl_no_likelihood', 'mdl_two_reports',
        'energy', 'rule', 'nearest')
METRICS = ('recall@8', 'precision@8', 'mrr@8', 'ndcg@8')


def summarize(out):
    rows = list(unique_rows(out / 'evidence.jsonl').values()); report, paired = [], []
    for dataset in ('locomo', 'longmemeval'):
        primary = {r['question_id']: r for r in rows if r['dataset'] == dataset and r['arm'] == 'mdl'}
        for arm in ARMS:
            subset = [r for r in rows if r['dataset'] == dataset and r['arm'] == arm]
            if not subset: continue
            report.append({'dataset': dataset, 'arm': arm, 'questions': len(subset),
                **{m: float(np.mean([r['retrieval'][m] for r in subset])) for m in METRICS},
                'candidate_recall@48': float(np.mean([r['candidate_recall'] for r in subset])),
                'mean_evidence_tokens': float(np.mean([r['evidence_tokens'] for r in subset])),
                'grounded_queries': sum(r['certificate']['grounded_candidates'] > 0 for r in subset),
                'queries_with_exclusion': sum(bool(r['certificate']['excluded']) for r in subset)})
            if arm != 'mdl':
                matched = [(primary[r['question_id']], r) for r in subset]
                paired.append({'dataset': dataset, 'control': arm, 'pairs': len(matched),
                    'identical_complete_prompts': sum(a['prompt_sha256'] == b['prompt_sha256'] for a, b in matched),
                    **{m + '_delta': float(np.mean([a['retrieval'][m] - b['retrieval'][m] for a, b in matched])) for m in METRICS}})
    if report: write_csv(out / 'retrieval.csv', report)
    if paired: write_csv(out / 'causal_opportunity.csv', paired)
    write_json(out / 'status.json', {'completed_cells': len(rows), 'expected_cells': 100 * len(ARMS),
        'status': 'complete_development' if len(rows) == 100 * len(ARMS) else 'partial_development',
        'qa_calls': 0, 'judge_calls': 0, 'full_runs_authorized': False,
        'scope': 'repeated-look development; retrieval ONLY; no answer-accuracy/statistical superiority claim'})


def run(manifest, calibration, output):
    cfg, cohort, artifact = protocol(manifest, calibration)
    out = Path(output)
    if out.exists() and any(out.iterdir()): raise ValueError('Fresh development output required')
    out.mkdir(parents=True)
    write_json(out / 'audit.json', {'scope': 'mdl_active_facet_development100',
        'source_fingerprint': source_fingerprint(), 'manifest_sha256': digest_file(manifest),
        'calibration_sha256': digest_file(calibration), 'worker_sha256': digest_file(__file__),
        'controller_sha256': digest_file(ROOT / 'src/mindset/methods/mdl_state.py'),
        'comparator_parameters': artifact['selected_parameters'], 'mdl_fitting': 'none; standard fixed coding defaults, not a new calibration',
        'arms': ARMS, 'candidate_budget': 48, 'raw_packet_budget': 8,
        'candidate_substrate': 'same flat BM25/dense union in every arm; isolates state decisions/readout',
        'answer_prompt_sha256': ANSWER_PROMPT_HASH, 'checkpoint_lock': read_json(ROOT / 'configs/checkpoints.lock.json'),
        'score_cache_scope': 'quality only; not efficiency timings', 'qa_calls': 0, 'judge_calls': 0,
        'full_runs_authorized': False}, immutable=True)
    record_environment(out); encoder, reranker = local_stack(cfg, cache=False)
    encoder = CachedEncoder(encoder, enabled=True)
    tokenizer = load_tokenizer(read_json(ROOT / 'configs/checkpoints.lock.json')[cfg['evidence_tokenizer']])
    params = {**{k: v for k, v in cfg['mindset'].items() if k != 'profile'}, **artifact['selected_parameters']}
    completed = 0
    for dataset, cluster, cases in groups_for(cohort, cfg):
        memories = {}
        for arm in ARMS:
            if arm.startswith('mdl'):
                m = MDLEnergyMemory(CompactConfig(**params), encoder=encoder, local_reranker=reranker,
                    state_readout=arm != 'mdl_read_off', boundary_energy=arm != 'mdl_no_boundary',
                    likelihood_energy=arm != 'mdl_no_likelihood', mdl_policy='two_reports' if arm == 'mdl_two_reports' else 'energy')
            else:
                m = FactorizedMindsetMemory(CompactConfig(**params), encoder=encoder, local_reranker=reranker,
                    write_policy='nearest' if arm == 'nearest' else 'rule' if arm == 'rule' else 'energy')
            m.ingest(cases[0].events); memories[arm] = m
            write_json(out / 'structure' / f'{dataset}-{cluster}-{arm}.json', {
                'invariants': m.validate_invariants(), 'telemetry': m.telemetry(),
                'observation_certificates': m.observation_audit(),
                'categorical_posteriors': getattr(m, '_mdl_posteriors', {})})
        base = memories['energy']; index = EvidenceCache(cases[0].events, encoder, tokenizer, cfg)
        if any(m._documents != base._documents or not np.array_equal(m._episode_vectors, base._episode_vectors) for m in memories.values()):
            raise ValueError('Raw evidence/embeddings differ across arms')
        for case in cases:
            qv = encoder.encode([case.question], normalize_embeddings=True, convert_to_numpy=True)[0]
            fused = reciprocal_rank_fusion([ranked_indices(base._episode_vectors @ qv), ranked_indices(base._bm25.scores(case.question))])
            pool = sorted(fused, key=lambda j: (-fused[j], j))[:48]
            scores = reranker.predict([(case.question, base._documents[j]) for j in pool], batch_size=8, show_progress_bar=False)
            if len(scores) != len(pool) or not np.isfinite(scores).all(): raise ValueError('Invalid CE scores')
            unit = {e.event_id: e.session_id if case.retrieval_unit == 'session' else e.event_id for e in case.events}
            for arm, m in memories.items():
                hits = m.select_scored(case.question, pool, scores, 8, case.question_date)
                selected, packet = index.packet(hits, case.question)
                prompt = messages(case.question, packet, case.question_date)
                append_row(out / 'evidence.jsonl', {'id': f'{dataset}/{arm}/{case.id}', 'dataset': dataset, 'cluster': cluster,
                    'question_id': case.id, 'arm': arm, 'question': case.question, 'question_date': case.question_date,
                    'category': case.category, 'answer': case.answer, 'selected': selected, 'evidence': packet, 'messages': prompt,
                    'prompt_sha256': stable_hash(prompt), 'evidence_tokens': len(tokenizer.encode(packet, add_special_tokens=False)),
                    'candidate_ids': [base._episode_order[j] for j in pool],
                    'candidate_recall': len({unit[base._episode_order[j]] for j in pool} & set(case.gold)) / len(set(case.gold)),
                    'retrieval': rank_metrics(selected, case, 8), 'certificate': m.last_readout_certificate})
                completed += 1
            summarize(out); print(f'{completed}/{100 * len(ARMS)} MDL development cells', flush=True)
        encoder.cache.clear(); del memories, index


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default=str(ROOT / 'data/manifests/mindset_v2_development100.json'))
    p.add_argument('--calibration', required=True); p.add_argument('--output', required=True)
    a = p.parse_args()
    if os.name == 'nt' or not os.environ.get('SLURM_JOB_ID'): p.error('DGX Slurm only')
    run(a.manifest, a.calibration, a.output)
