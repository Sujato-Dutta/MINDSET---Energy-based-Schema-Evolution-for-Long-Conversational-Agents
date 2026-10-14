"""Label-blind MDL write-energy comparison on the fixed constructed worlds."""
from pathlib import Path
from collections import defaultdict, Counter
import argparse
import os
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from probe_state_causality import requests, evaluate, build as old_build, TRACKS, MODES
from mindset.acl.state_worlds import worlds
from mindset.acl.joint_calibration import validate_joint
from mindset.acl.pipeline import config, local_stack, record_environment, source_fingerprint, write_csv
from mindset.acl.io import write_json, append_row, unique_rows, digest_file
from mindset.acl.runtime import CachedEncoder
from mindset.methods.mdl_state import ObservedMDLEnergyMemory, TextMDLEnergyMemory
from mindset.methods.compact import CompactConfig
from mindset.methods.base import ConversationEvent
from mindset.retrieval.hybrid import ranked_indices, reciprocal_rank_fusion
from mindset.utils.hashing import stable_hash

ARMS = ('mdl', 'mdl_no_boundary', 'mdl_no_likelihood', 'mdl_two_reports', 'mdl_read_off',
        'energy', 'rule', 'nearest', 'latest_journal', 'replay_energy')


def build(arm, track, history, parameters, encoder, reranker, vectors, scopes):
    if not arm.startswith('mdl'): return old_build(arm, track, history, parameters, encoder, reranker, vectors, scopes)
    cls = TextMDLEnergyMemory if track == 'existing_text_parser' else ObservedMDLEnergyMemory
    return cls(CompactConfig(**parameters), encoder=encoder, local_reranker=reranker, precomputed_embeddings=vectors,
        observations={e['event_id']: e['observation'] for e in history['events']}, scopes=scopes,
        state_readout=arm != 'mdl_read_off', boundary_energy=arm != 'mdl_no_boundary',
        likelihood_energy=arm != 'mdl_no_likelihood', mdl_policy='two_reports' if arm == 'mdl_two_reports' else 'energy')


def summarize(out):
    rows = list(unique_rows(out / 'queries.jsonl').values()); grouped = defaultdict(list)
    for r in rows:
        for family in ('all', r['family']): grouped[(r['track'], r['score_mode'], r['arm'], family)].append(r)
    results = []
    for k, panel in sorted(grouped.items()):
        current = [r for r in panel if r['query_kind'].startswith('current')]
        historical = [r for r in panel if not r['query_kind'].startswith('current')]
        post = [r for r in current if r['post_perturbation']]
        mean = lambda p, metric: float(np.mean([r['metrics'][metric] for r in p])) if p else None
        results.append({'track': k[0], 'score_mode': k[1], 'arm': k[2], 'family': k[3], 'queries': len(panel),
            'current_top1_state_accuracy': mean(current, 'top1_state_correct'),
            'historical_top1_state_accuracy': mean(historical, 'top1_state_correct'),
            'post_perturbation_current_queries': len(post), 'post_perturbation_current_top1_accuracy': mean(post, 'top1_state_correct'),
            'mean_packet_compatible_fraction': mean(panel, 'packet_compatible_fraction'),
            'mean_evidence_recall@8': mean(panel, 'evidence_recall@8'),
            'future_source_top1_rate': mean(panel, 'future_source_top1')})
    if results: write_csv(out / 'state_results.csv', results)
    keyed = {(r['history'], r['prefix'], r['query_kind'], r['track'], r['score_mode'], r['arm']): r for r in rows}
    pairs = defaultdict(list)
    for r in rows:
        if r['arm'] != 'mdl': continue
        for control in ARMS:
            if control == 'mdl': continue
            b = keyed[(r['history'], r['prefix'], r['query_kind'], r['track'], r['score_mode'], control)]
            delta = int(r['metrics']['top1_state_correct']) - int(b['metrics']['top1_state_correct'])
            for family in ('all', r['family']):
                kind = 'current' if r['query_kind'].startswith('current') else 'historical'
                for period in ('all_prefixes', 'post_perturbation') if r['post_perturbation'] else ('all_prefixes',):
                    pairs[(r['track'], r['score_mode'], control, family, kind, period)].append((delta, r['selected'] == b['selected']))
    comparison = [{'track': k[0], 'score_mode': k[1], 'control': k[2], 'family': k[3], 'query_type': k[4],
        'period': k[5], 'paired_queries': len(p), 'mean_top1_state_accuracy_delta': float(np.mean([d for d, _ in p])),
        'wins': sum(d > 0 for d, _ in p), 'ties': sum(d == 0 for d, _ in p), 'losses': sum(d < 0 for d, _ in p),
        'identical_packets': sum(s for _, s in p)} for k, p in sorted(pairs.items())]
    if comparison: write_csv(out / 'paired_state_effects.csv', comparison)
    ops = list(unique_rows(out / 'operations.jsonl').values()); by_op = defaultdict(list)
    for r in ops:
        if r['speaker'] == 'Sam': by_op[(r['track'], r['arm'])].append(r)
    table = [{'track': k[0], 'arm': k[1], 'observations': len(p),
        **{a: sum(r['operation'] == a for r in p) for a in ('new', 'reinforce', 'supersede', 'split')},
        'exception_stores': sum(r['stored_as_exception'] for r in p),
        'multiple_feasible_actions': sum(len(r['feasible_actions']) > 1 for r in p),
        'non_minimizing_choices': sum((r['energy_gap'] or 0) > 1e-9 for r in p)} for k, p in sorted(by_op.items())]
    if table: write_csv(out / 'transition_counts.csv', table)


def run(calibration, output):
    artifact = validate_joint(calibration); out = Path(output)
    if out.exists() and any(out.iterdir()): raise ValueError('Fresh output directory required')
    out.mkdir(parents=True); corpus = worlds(); write_json(out / 'worlds.json', corpus, immutable=True)
    write_json(out / 'audit.json', {'scope': 'constructed MDL write-energy diagnostic; NOT natural QA',
        'worlds_sha256': digest_file(out / 'worlds.json'), 'source_fingerprint': source_fingerprint(),
        'worker_sha256': digest_file(__file__), 'controller_sha256': digest_file(ROOT / 'src/mindset/methods/mdl_state.py'),
        'shared_harness_sha256': digest_file(ROOT / 'scripts/probe_state_causality.py'),
        'comparator_parameters': artifact['selected_parameters'], 'calibration_sha256': digest_file(calibration),
        'mdl_parameters': 'none fitted; Jeffreys 1/2, geometric segment code, fixed-length boundary code',
        'not_a_new_calibration': True, 'mdl_optimum_scope': 'active owner/predicate/context facet, preserve versus every feasible single cut',
        'older_committed_versions': 'retained, not globally resegmented', 'explicit_change_cues': 'ignored by MDL categorical energy',
        'votes': 'last asserted value per session; sessions not guaranteed reliable/independent',
        'arms': ARMS, 'tracks': TRACKS, 'modes': MODES, 'budget': {'candidate_pool': 48, 'raw_evidence': 8},
        'qa_calls': 0, 'judge_calls': 0, 'benchmark_fitting': False, 'efficiency_claim': False, 'full_runs_authorized': False})
    record_environment(out); encoder, reranker = local_stack(config(), cache=False)
    encoder = CachedEncoder(encoder, enabled=True); ce = {}; completed = 0
    for number, h in enumerate(corpus, 1):
        events = [ConversationEvent(**{k: e[k] for k in ('event_id', 'conversation_id', 'session_id', 'timestamp', 'speaker', 'text')}) for e in h['events']]
        vectors = encoder.encode([f'[{e.timestamp}] {e.speaker}: {e.text}' for e in events], normalize_embeddings=True, convert_to_numpy=True)
        vectors = {e.event_id: v for e, v in zip(events, vectors, strict=True)}
        scopes = {r['question']: r['scope'] for i, e in enumerate(events) if e.speaker == 'Sam' for r in requests(h, i)}
        memories = {(t, a): build(a, t, h, artifact['selected_parameters'], encoder, reranker, vectors, scopes) for t in TRACKS for a in ARMS}
        for i, e in enumerate(events):
            for (track, arm), m in memories.items():
                m.ingest([e]); m.validate_invariants(); rec = m.operation_log.all()[-1]
                cert = next(c for c in reversed(m.observation_audit()) if c['source_id'] == e.event_id)
                append_row(out / 'operations.jsonl', {'id': f'{h["id"]}/{track}/{arm}/{e.event_id}',
                    'history': h['id'], 'family': h['family'], 'speaker': e.speaker, 'track': track, 'arm': arm,
                    'operation': rec.operation, 'stored_as_exception': rec.stored_as_exception,
                    'feasible_actions': sorted({p['action'] for p in cert['feasible_candidates']}) or [rec.operation],
                    'energy_gap': cert.get('optimality_gap'), 'certificate': cert})
            if e.speaker != 'Sam': continue
            base = memories[(TRACKS[0], 'energy')]
            if any(m._documents != base._documents for m in memories.values()): raise ValueError('Raw sources differ across arms')
            for req in requests(h, i):
                q = req['question']; vector = encoder.encode([q], normalize_embeddings=True, convert_to_numpy=True)[0]
                fused = reciprocal_rank_fusion([ranked_indices(base._episode_vectors @ vector), ranked_indices(base._bm25.scores(q))])
                pool = sorted(fused, key=lambda j: (-fused[j], j))[:48]
                pairs = [(q, base._documents[j]) for j in pool]; missing = [p for p in dict.fromkeys(pairs) if p not in ce]
                if missing:
                    scores = reranker.predict(missing, batch_size=8, show_progress_bar=False)
                    if len(scores) != len(missing) or not np.isfinite(scores).all(): raise ValueError('Invalid CE outputs')
                    ce.update(zip(missing, map(float, scores), strict=True))
                for mode in MODES:
                    scores = [ce[p] for p in pairs] if mode == 'actual_ce' else [float(-j if mode == 'oldest_first' else j) for j in pool]
                    for (track, arm), m in memories.items():
                        hits = m.select_scored(q, pool, scores, 8); selected = [hit.memory_id for hit in hits]
                        append_row(out / 'queries.jsonl', {'id': f'{h["id"]}/{i}/{req["kind"]}/{track}/{arm}/{mode}',
                            'history': h['id'], 'prefix': i, 'family': h['family'], 'track': track, 'arm': arm,
                            'score_mode': mode, 'query_kind': req['kind'], 'post_perturbation': i >= 8 + h['distractors'],
                            'question': q, 'selected': selected, 'candidate_ids': [base._episode_order[j] for j in pool],
                            'source_packet_sha256': stable_hash([hit.text for hit in hits]),
                            'metrics': evaluate(h, i, req, selected), 'certificate': m.last_readout_certificate})
                        completed += 1
        summarize(out); write_json(out / 'status.json', {'status': 'complete_controlled_probe' if number == len(corpus) else 'partial_controlled_probe',
            'histories': number, 'expected_histories': len(corpus), 'query_cells': completed, 'unique_ce_pairs': len(ce),
            'qa_calls': 0, 'judge_calls': 0, 'natural_answer_accuracy': None, 'full_runs_authorized': False})
        print(f'{number}/{len(corpus)} constructed histories, {completed} query cells', flush=True)
        encoder.cache.clear(); del memories


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--calibration', required=True); p.add_argument('--output', required=True)
    a = p.parse_args()
    if os.name == 'nt' or not os.environ.get('SLURM_JOB_ID'): p.error('DGX Slurm only')
    run(a.calibration, a.output)
