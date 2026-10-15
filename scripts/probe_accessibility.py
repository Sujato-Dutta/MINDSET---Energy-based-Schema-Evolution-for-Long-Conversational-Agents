"""Integrated source-routing energy and matched ablations on development100."""
import os
from pathlib import Path
from time import perf_counter
import numpy as np
from mindset.acl.pipeline import ROOT, local_stack, rank_metrics, write_csv, record_environment
from mindset.acl.joint_iteration import protocol, groups_for
from mindset.acl.io import read_json, write_json, unique_rows, append_row, digest_file
from mindset.acl.runtime import CachedEncoder
from mindset.acl.evidence import EvidenceCache
from mindset.methods.compact import CompactConfig
from mindset.methods.accessibility import AccessibilityMemory
from mindset.methods.conflict_frontier import LeanEpisodeIndex
from mindset.methods.base import MemoryHit
from mindset.methods.semantic_witness import source_spans, semantic_pairs, extend_pool

ARMS = ('access_full', 'access_no_history', 'access_no_current', 'access_off', 'flat_expanded')
METRICS = ('recall@8', 'precision@8', 'mrr@8', 'ndcg@8')


def summarize(out, expected):
    rows = list(unique_rows(out / 'retrieval.jsonl').values())
    if {r['id'] for r in rows} - expected:
        raise ValueError('Unexpected cells')
    reports = []
    for dataset in ('locomo', 'longmemeval'):
        base = {r['question_id']: r for r in rows if r['dataset'] == dataset and r['method'] == 'access_off'}
        for arm in ARMS:
            subset = [r for r in rows if r['dataset'] == dataset and r['method'] == arm]
            if not subset:
                continue
            reports.append(dict(dataset=dataset, method=arm, questions=len(subset),
                **{m: float(np.mean([r['metrics'][m] for r in subset])) for m in METRICS},
                mean_evidence_tokens=float(np.mean([r['evidence_tokens'] for r in subset])),
                packet_changes_from_no_access=sum(r['evidence'] != base[r['question_id']]['evidence']
                    for r in subset if r['question_id'] in base),
                mean_logical_ce_pairs=float(np.mean([r['logical_ce_pairs'] for r in subset])),
                scope='repeated-look development; no significance, novelty, or fresh latency claim'))
    if reports:
        write_csv(out / 'comparison.csv', reports)
    write_json(out / 'status.json', dict(status='complete_development' if len(rows) == len(expected) else 'partial_development',
        completed_cells=len(rows), expected_cells=len(expected), qa_calls=0, judge_calls=0, full_runs_authorized=False))


def run(*, output_name='accessibility_r01', memory_class=AccessibilityMemory, replay_reference=True):
    if os.name == 'nt' or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('DGX Slurm only')
    manifest = ROOT / 'data/manifests/mindset_v2_development100.json'
    calibration = ROOT / 'outputs/acl/mindset_joint_v2_27058/calibration/calibration.json'
    cfg, cohort, fitted = protocol(manifest, calibration)
    parent = ROOT / 'outputs/acl/raw_source_seeds_r01'
    if read_json(parent / 'status.json')['status'] != 'complete_development':
        raise ValueError('Reference panel not complete')
    reference = unique_rows(parent / 'retrieval.jsonl')
    out = ROOT / 'outputs/acl' / output_name; out.mkdir(parents=True, exist_ok=True)
    audit = dict(manifest_sha256=digest_file(manifest), calibration_sha256=digest_file(calibration),
        checkpoint_lock_sha256=digest_file(ROOT / 'configs/checkpoints.lock.json'),
        source_adapter='original_frozen_text_only', methods=ARMS, worker_sha256=digest_file(__file__),
        controller_sha256=digest_file(ROOT / 'src/mindset/methods/accessibility.py'),
        candidate_class=memory_class.__name__,
        candidate_module_sha256=digest_file(__import__('inspect').getfile(memory_class)),
        reference_packet_replay_required=replay_reference,
        source_association_sha256=digest_file(ROOT / 'src/mindset/methods/semantic_witness.py'),
        reference_retrieval_sha256=digest_file(parent / 'retrieval.jsonl'),
        limits=dict(schema_roots=6, source_probes=128, final_pool=48, passages=8, ce_pairs=96),
        new_fit=False, parameter_provenance='original lambda inherited; switching margin removed; no new tuning',
        scope='integrated source-accessibility surrogate, not verified semantic updates or novelty',
        full_runs_authorized=False)
    expected = {f"{r['dataset']}/{a}/{r['id']}" for r in cohort['records'] for a in ARMS}
    import fcntl
    with (out / 'worker.lock').open('a+b') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(out / 'audit.json', audit, immutable=True)
        record_environment(out / 'continuations' / os.environ['SLURM_JOB_ID'])
        encoder, ce = local_stack(cfg, cache=False); encoder = CachedEncoder(encoder, enabled=True)
        from transformers import AutoTokenizer
        checkpoint = read_json(ROOT / 'configs/checkpoints.lock.json')[cfg['evidence_tokenizer']]
        tokenizer = AutoTokenizer.from_pretrained(checkpoint['checkpoint'], revision=checkpoint['revision'],
            cache_dir=str(Path.home() / 'MINDSET_v2/.cache/gpu_prototype/huggingface/hub'), local_files_only=True)
        done = unique_rows(out / 'retrieval.jsonl')
        if set(done) - expected:
            raise ValueError('Unexpected resumed cells')
        params = {**{k: v for k, v in cfg['mindset'].items() if k != 'profile'}, **fitted['selected_parameters']}
        for dataset, cluster, cases in groups_for(cohort, cfg):
            events = cases[0].events
            if all(f'{dataset}/{arm}/{case.id}' in done for arm in ARMS for case in cases):
                continue
            index = LeanEpisodeIndex(events, encoder)
            documents = dict(zip(index.ids, index.documents, strict=True))
            spans = source_spans(events)
            vectors = encoder.encode([s.text for s in spans], batch_size=64, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False) if spans else np.empty((0, index.vectors.shape[1]))
            embeddings = dict(zip(index.ids, index.vectors, strict=True))
            for arm in ARMS:
                if arm == 'flat_expanded':
                    for case in cases:
                        rid = f'{dataset}/{arm}/{case.id}'
                        if rid in done:
                            continue
                        ref = reference[f'{dataset}/flat_ce_seed/{case.id}']
                        row = {**ref, 'id': rid, 'method': arm}
                        append_row(out / 'retrieval.jsonl', row); done[rid] = row
                    summarize(out, expected)
                    continue
                if all(f'{dataset}/{arm}/{case.id}' in done for case in cases):
                    continue
                mode = dict(access_full='full', access_no_history='no_history', access_no_current='no_current', access_off='no_access')[arm]
                started = perf_counter()
                memory = memory_class(CompactConfig(**params), encoder=encoder, local_reranker=ce,
                    precomputed_embeddings=embeddings, access_mode=mode)
                memory.ingest(events)
                write_json(out / 'structure' / f'{dataset}-{cluster}-{arm}.json', dict(
                    invariants=memory.validate_invariants(), telemetry=memory.telemetry(),
                    build_wall_seconds=perf_counter() - started, timing_scope='shared cached embeddings, NOT paper efficiency',
                    decisions=memory.access_records))
                print(f'BUILD {dataset}/{cluster}/{arm} episodes={len(events)} interventions={memory.telemetry()["route_interventions"]}', flush=True)
                for case in cases:
                    rid = f'{dataset}/{arm}/{case.id}'
                    if rid in done:
                        continue
                    flat = [index.ids[i] for i in index.candidates(case.question, encoder, 48)]
                    budget = reference[f'{dataset}/joint_ref/{case.id}']['evidence_tokens']
                    packet_cfg = {**cfg, 'retrieval': {**cfg['retrieval'], 'candidate_k': 48, 'evidence_tokens': budget}}
                    packet_index = EvidenceCache(events, encoder, tokenizer, packet_cfg)
                    initial_selected, _ = packet_index.packet(memory.retrieve(case.question, 40), case.question)
                    seeds = [s['event_id'] for s in initial_selected]
                    qvec = encoder.encode([case.question], normalize_embeddings=True, convert_to_numpy=True)[0]
                    pairs = semantic_pairs(index, spans, vectors, qvec, seeds, events, question_date=case.question_date)
                    pool = extend_pool(flat, [{'left': spans[a].eid, 'right': spans[b].eid} for a, b in pairs])
                    values = np.asarray(ce.predict([(case.question, documents[eid]) for eid in pool],
                        batch_size=cfg['mindset']['rerank_batch_size'], show_progress_bar=False))
                    if len(values) != len(pool) or not np.isfinite(values).all():
                        raise ValueError('Bad local scores')
                    scores = dict(zip(pool, map(float, values), strict=True))
                    ranking = sorted(pool, key=lambda eid: (-scores[eid], eid))
                    selected, packet = packet_index.packet([
                        MemoryHit(eid, documents[eid], scores[eid], (eid,), 'access_raw_source', {}) for eid in ranking], case.question)
                    tokens = len(tokenizer.encode(packet, add_special_tokens=False))
                    if tokens > budget or len(selected) > 8 or len(pool) > 48:
                        raise ValueError('Packet budget violated')
                    row = dict(id=rid, dataset=dataset, method=arm, question_id=case.id, cluster=cluster,
                        category=case.category, selected=selected, evidence=packet, evidence_tokens=tokens,
                        metrics=rank_metrics(selected, case, 8), seed_ids=seeds, candidate_ids=pool,
                        logical_ce_pairs=len(memory.last_candidate_pool) + len(pool),
                        scope='integrated controller plus shared source association; exposed development100')
                    if arm == 'access_off' and replay_reference:
                        ref = reference[f'{dataset}/expanded_joint_ref/{case.id}']
                        if row['evidence'] != ref['evidence']:
                            raise ValueError('No-access arm fails exact prior expansion replay')
                    append_row(out / 'retrieval.jsonl', row); done[rid] = row
                    summarize(out, expected)
                    print(f'QUERY {len(done)}/{len(expected)} {rid}', flush=True)
                del memory
            encoder.cache.clear()
        summarize(out, expected)


if __name__ == '__main__':
    run()
