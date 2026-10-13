"""Independent known-state causal probes; zero answer calls, no parameter fit."""
from pathlib import Path
from collections import defaultdict
import argparse
import os
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mindset.acl.state_worlds import worlds
from mindset.acl.joint_calibration import validate_joint
from mindset.acl.pipeline import config, local_stack, record_environment, source_fingerprint, write_csv
from mindset.acl.io import write_json, append_row, unique_rows, digest_file
from mindset.methods.compact import CompactConfig
from mindset.methods.base import ConversationEvent
from mindset.methods.state_readout import QueryScope
from mindset.methods.state_causal_probe import (ObservedStateMemory, ConfirmationReplayMemory,
    LatestJournalMemory, TextStateMemory, TextReplayMemory, TextLatestJournalMemory)
from mindset.methods.joint import utc_time
from mindset.retrieval.hybrid import ranked_indices, reciprocal_rank_fusion
from mindset.utils.hashing import stable_hash

ARMS = ('energy', 'read_off', 'rule', 'nearest', 'latest_journal', 'replay_energy', 'replay_rule', 'replay_no_debt')
TRACKS = ('structured_observations', 'existing_text_parser')
MODES = ('actual_ce', 'oldest_first', 'newest_first')


def requests(history, prefix):
    contexts = sorted({c for e in history['events'][:prefix + 1] if e['speaker'] == 'Sam'
                       for c in e['observation']['contexts']})
    specs = [(f'current/{c}', c, None) for c in contexts]
    specs.append(('initial_history', 'at home', 0))
    first_change = 8 + history['distractors']
    if prefix >= first_change: specs.append(('first_report_history', history['events'][first_change]['observation']['contexts'][0], first_change))
    result = []
    for kind, context, point in specs:
        date = history['events'][point]['timestamp'] if point is not None else ''
        text = f'Where does Sam live now {context}?' if point is None else f'Where did Sam live {context} on {date}?'
        t = utc_time(date) if date else 0.
        scope = QueryScope('current' if point is None else 'as_of', ('sam',), ('location',), (context,), t, t + 86400 if t else 0.)
        result.append({'kind': kind, 'question': text, 'scope': scope, 'point': point})
    return result


def build(arm, track, history, parameters, encoder, reranker, vectors, scopes):
    kw = dict(observations={e['event_id']: e['observation'] for e in history['events']}, scopes=scopes,
              encoder=encoder, local_reranker=reranker, precomputed_embeddings=vectors,
              state_readout=arm != 'read_off', write_policy='nearest' if arm == 'nearest' else 'rule' if arm in ('rule', 'replay_rule') else 'energy')
    text = track == 'existing_text_parser'
    if arm == 'latest_journal': cls = TextLatestJournalMemory if text else LatestJournalMemory
    elif arm.startswith('replay'):
        cls = TextReplayMemory if text else ConfirmationReplayMemory
        kw['contradiction_debt'] = arm != 'replay_no_debt'
    else: cls = TextStateMemory if text else ObservedStateMemory
    return cls(CompactConfig(**parameters), **kw)


def evaluate(history, prefix, req, selected):
    point = prefix if req['point'] is None else req['point']
    scope = req['scope']; context = scope.contexts[0]
    truth = next(s['value'] for s in history['truth_timeline'][point]
                 if s['owner'] == 'Sam' and s['slot'] == 'location' and s['context'] == context)
    by_id = {e['event_id']: e for e in history['events'][:prefix + 1]}
    gold = {e['event_id'] for e in history['events'][:point + 1]
        if e['observation']['speaker'] == 'Sam' and e['observation']['contexts'] == [context]
        and e['observation']['value'] == truth}
    # For current queries, matching source assertions in the observed prefix.
    if req['point'] is None:
        gold = {eid for eid, e in by_id.items() if e['speaker'] == 'Sam'
                and e['observation']['contexts'] == [context] and e['observation']['value'] == truth}
    compatible = [eid in gold for eid in selected]
    return {'expected_value': truth, 'gold_source_ids': sorted(gold),
        'top1_state_correct': bool(compatible and compatible[0]), 'abstained': not selected,
        'evidence_recall@8': len(set(selected) & gold) / len(gold),
        'packet_compatible_fraction': sum(compatible) / len(compatible) if compatible else 0.,
        'wrong_context_top1': bool(selected and by_id[selected[0]]['observation']['contexts'] != [context]),
        'future_source_top1': bool(selected and req['point'] is not None and
            utc_time(by_id[selected[0]]['timestamp']) >= scope.period_end),
        'metric_scope': 'deterministic known-world source selection; NOT LLM answer accuracy'}


def summarize(out):
    rows = list(unique_rows(out / 'queries.jsonl').values()); report = []
    grouped = defaultdict(list)
    for r in rows:
        for family in ('all', r['family']): grouped[(r['track'], r['score_mode'], r['arm'], family)].append(r)
    for track in TRACKS:
        for mode in MODES:
            for arm in ARMS:
                for family in ('all', 'change', 'coexist', 'noise', 'confirmation', 'explicit_noise', 'correlated_noise'):
                    panel = grouped[(track, mode, arm, family)]
                    if not panel: continue
                    current = [r for r in panel if r['query_kind'].startswith('current')]
                    historical = [r for r in panel if not r['query_kind'].startswith('current')]
                    post = [r for r in current if r['post_perturbation']]
                    report.append({'track': track, 'score_mode': mode, 'arm': arm, 'family': family, 'queries': len(panel),
                        'current_top1_state_accuracy': float(np.mean([r['metrics']['top1_state_correct'] for r in current])) if current else None,
                        'historical_top1_state_accuracy': float(np.mean([r['metrics']['top1_state_correct'] for r in historical])) if historical else None,
                        'post_perturbation_current_queries': len(post),
                        'post_perturbation_current_top1_accuracy': float(np.mean([r['metrics']['top1_state_correct'] for r in post])) if post else None,
                        'mean_packet_compatible_fraction': float(np.mean([r['metrics']['packet_compatible_fraction'] for r in panel])),
                        'mean_evidence_recall@8': float(np.mean([r['metrics']['evidence_recall@8'] for r in panel])),
                        'wrong_context_top1_rate': float(np.mean([r['metrics']['wrong_context_top1'] for r in panel])),
                        'future_source_top1_rate': float(np.mean([r['metrics']['future_source_top1'] for r in panel]))})
    if report: write_csv(out / 'state_results.csv', report)
    # Matched causal chain: identical input and scores -> operation/state -> packet.
    keyed = {(r['history'], r['prefix'], r['query_kind'], r['track'], r['score_mode'], r['arm']): r for r in rows}
    paired = defaultdict(list)
    for r in rows:
        if r['arm'] not in ('replay_energy', 'energy'): continue
        for control in ('energy', 'rule', 'nearest', 'latest_journal', 'read_off', 'replay_rule', 'replay_no_debt'):
            if control == r['arm']: continue
            b = keyed[(r['history'], r['prefix'], r['query_kind'], r['track'], r['score_mode'], control)]
            cell = {'identical_packet': r['source_packet_sha256'] == b['source_packet_sha256'],
                    'delta': int(r['metrics']['top1_state_correct']) - int(b['metrics']['top1_state_correct'])}
            for family in ('all', r['family']):
                paired[(r['track'], r['score_mode'], r['arm'], control, family,
                        'current' if r['query_kind'].startswith('current') else 'historical')].append(cell)
    comparisons = [{'track': k[0], 'score_mode': k[1], 'arm': k[2], 'control': k[3], 'family': k[4],
                    'query_type': k[5], 'paired_queries': len(p),
                    'identical_packets': sum(c['identical_packet'] for c in p),
                    'mean_top1_state_accuracy_delta': float(np.mean([c['delta'] for c in p])),
                    'wins': sum(c['delta'] > 0 for c in p), 'ties': sum(c['delta'] == 0 for c in p),
                    'losses': sum(c['delta'] < 0 for c in p)} for k, p in paired.items()]
    if comparisons: write_csv(out / 'paired_state_effects.csv', comparisons)
    ops = list(unique_rows(out / 'operations.jsonl').values()); summaries = []
    for track in TRACKS:
        for arm in ARMS:
            panel = [r for r in ops if r['track'] == track and r['arm'] == arm and r['speaker'] == 'Sam']
            if panel: summaries.append({'track': track, 'arm': arm, 'observations': len(panel),
                **{a: sum(r['operation'] == a for r in panel) for a in ('new', 'reinforce', 'split', 'supersede')},
                'exception_stores': sum(r['stored_as_exception'] for r in panel),
                'supersede_eligible': sum('supersede' in r['feasible_actions'] for r in panel),
                'split_eligible': sum('split' in r['feasible_actions'] for r in panel),
                'multiple_feasible_actions': sum(len(r['feasible_actions']) > 1 for r in panel),
                'non_minimizing_controls': sum((r.get('energy_gap') or 0.) > 0 for r in panel)})
    if summaries: write_csv(out / 'transition_counts.csv', summaries)


def run(calibration, output):
    artifact = validate_joint(calibration); cfg = config(); out = Path(output)
    if out.exists() and any(out.iterdir()): raise ValueError('Fresh controlled experiment output required')
    out.mkdir(parents=True); corpus = worlds()
    write_json(out / 'worlds.json', corpus, immutable=True)
    write_json(out / 'audit.json', {'scope': 'constructed controlled state causality; NOT natural QA/OOD',
        'source_fingerprint': source_fingerprint(), 'worker_sha256': digest_file(__file__),
        'controller_sha256': digest_file(ROOT / 'src/mindset/methods/state_causal_probe.py'),
        'world_generator_sha256': digest_file(ROOT / 'src/mindset/acl/state_worlds.py'),
        'worlds_sha256': digest_file(out / 'worlds.json'), 'calibration_sha256': digest_file(calibration),
        'parameters': artifact['selected_parameters'], 'parameter_fitting': 'none; scalar transfer, not new calibration',
        'tracks': list(TRACKS), 'arms': list(ARMS), 'score_modes': list(MODES),
        'budget': {'shared_flat_ce_candidate_pool': 48, 'raw_episode_packet': 8},
        'labels': 'world states fixed by generator before any controller runs; excluded from constructor inputs',
        'stress_modes': 'oldest/newest ordering does not use latent labels', 'qa_calls': 0,
        'efficiency': 'cached CE/encoder quality ONLY; no latency claim', 'full_runs_authorized': False})
    record_environment(out); encoder, reranker = local_stack(cfg, cache=False)
    from mindset.acl.runtime import CachedEncoder
    encoder = CachedEncoder(encoder, enabled=True); ce_cache = {}; completed = 0
    for h in corpus:
        raw = h['events']
        events = [ConversationEvent(**{k: e[k] for k in ('event_id', 'conversation_id', 'session_id', 'timestamp', 'speaker', 'text')}) for e in raw]
        vectors = encoder.encode([f'[{e.timestamp}] {e.speaker}: {e.text}' for e in events],
                                 normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        by_id = {e.event_id: v for e, v in zip(events, vectors, strict=True)}
        scopes = {r['question']: r['scope'] for i in range(len(raw)) if raw[i]['speaker'] == 'Sam' for r in requests(h, i)}
        memories = {(track, arm): build(arm, track, h, artifact['selected_parameters'], encoder, reranker, by_id, scopes)
                    for track in TRACKS for arm in ARMS}
        for i, event in enumerate(events):
            for (track, arm), m in memories.items():
                m.ingest([event]); m.validate_invariants()
                record = m.operation_log.all()[-1]
                certificates = [c for c in m.observation_audit() if c['source_id'] == event.event_id]
                cert = certificates[-1]
                append_row(out / 'operations.jsonl', {'id': f'{h["id"]}/{track}/{arm}/{event.event_id}',
                    'history': h['id'], 'family': h['family'], 'track': track, 'arm': arm, 'speaker': event.speaker,
                    'operation': record.operation, 'stored_as_exception': record.stored_as_exception,
                    'feasible_actions': sorted({p['action'] for p in cert['feasible_candidates']}) or ['new'],
                    'energy_gap': cert.get('optimality_gap'), 'certificate': cert})
            if event.speaker != 'Sam': continue
            base = memories[(TRACKS[0], 'energy')]
            if any(m._documents != base._documents for m in memories.values()): raise ValueError('Source evidence differs across controls')
            for req in requests(h, i):
                q = req['question']; qv = encoder.encode([q], normalize_embeddings=True,
                    convert_to_numpy=True, show_progress_bar=False)[0]
                fused = reciprocal_rank_fusion([ranked_indices(base._episode_vectors @ qv), ranked_indices(base._bm25.scores(q))])
                pool = sorted(fused, key=lambda k: (-fused[k], k))[:48]
                pairs = [(q, base._documents[k]) for k in pool]
                missing = [p for p in dict.fromkeys(pairs) if p not in ce_cache]
                if missing:
                    values = reranker.predict(missing, batch_size=8, show_progress_bar=False)
                    if len(values) != len(missing) or not np.isfinite(values).all(): raise ValueError('Invalid CE output')
                    ce_cache.update(zip(missing, map(float, values), strict=True))
                for mode in MODES:
                    scores = [ce_cache[p] for p in pairs] if mode == 'actual_ce' else [float(-k if mode == 'oldest_first' else k) for k in pool]
                    for (track, arm), m in memories.items():
                        hits = m.select_scored(q, pool, scores, 8)
                        selected = [hit.memory_id for hit in hits]
                        row = {'id': f'{h["id"]}/{i}/{req["kind"]}/{track}/{arm}/{mode}', 'history': h['id'],
                            'family': h['family'], 'shared_words': h['shared_words'], 'distractors': h['distractors'],
                            'prefix': i, 'track': track, 'arm': arm, 'score_mode': mode,
                            'post_perturbation': i >= 8 + h['distractors'],
                            'query_kind': req['kind'], 'question': q, 'selected': selected,
                            'candidate_ids': [base._episode_order[k] for k in pool],
                            'source_packet_sha256': stable_hash([hit.text for hit in hits]),
                            'metrics': evaluate(h, i, req, selected), 'certificate': m.last_readout_certificate}
                        append_row(out / 'queries.jsonl', row); completed += 1
        summarize(out)
        write_json(out / 'status.json', {'status': 'partial_controlled_probe', 'histories': corpus.index(h) + 1,
            'expected_histories': len(corpus), 'query_cells': completed, 'unique_ce_pairs': len(ce_cache),
            'qa_calls': 0, 'judge_calls': 0, 'natural_answer_accuracy': None})
        print(f'{corpus.index(h) + 1}/{len(corpus)} independent histories, {completed} query cells', flush=True)
        encoder.cache.clear(); del memories
    write_json(out / 'status.json', {'status': 'complete_controlled_probe', 'histories': len(corpus),
        'query_cells': completed, 'unique_ce_pairs': len(ce_cache), 'qa_calls': 0,
        'judge_calls': 0, 'natural_answer_accuracy': None, 'full_runs_authorized': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--calibration', required=True); p.add_argument('--output', required=True)
    args = p.parse_args()
    if os.name == 'nt' or not os.environ.get('SLURM_JOB_ID'): p.error('DGX Slurm only')
    run(args.calibration, args.output)
