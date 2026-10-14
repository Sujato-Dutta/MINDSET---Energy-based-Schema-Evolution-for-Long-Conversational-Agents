"""Prepare original100 source-only pairs; no generation, QA or new retrieval arm."""
import os
from dataclasses import asdict

import numpy as np

from mindset.acl.pipeline import ROOT, record_environment
from mindset.acl.joint_iteration import protocol, groups_for
from mindset.acl.io import read_json, write_json, unique_rows, append_row, digest_file
from mindset.acl.runtime import CachedEncoder
from mindset.acl.source_relation_candidates import measured_pairs
from mindset.methods.semantic_witness import source_spans, semantic_pairs
from mindset.utils.hashing import stable_hash


def run():
    if os.name == 'nt' or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('DGX Slurm only')
    manifest = ROOT / 'data/manifests/mindset_v2_development100.json'
    calibration = ROOT / 'outputs/acl/mindset_joint_v2_27058/calibration/calibration.json'
    cfg, cohort, _ = protocol(manifest, calibration)
    lock = read_json(ROOT / 'configs/checkpoints.lock.json')
    parent = ROOT / 'outputs/acl/raw_source_seeds_r01'
    parent_audit = read_json(parent / 'audit.json')
    if read_json(parent / 'status.json')['status'] != 'complete_development':
        raise ValueError('Seed parent not terminal')
    dependencies = {'manifest_sha256': digest_file(manifest), 'calibration_sha256': digest_file(calibration),
        'checkpoint_lock_sha256': digest_file(ROOT / 'configs/checkpoints.lock.json'),
        'source_module_sha256': digest_file(ROOT / 'src/mindset/methods/semantic_witness.py'),
        'worker_sha256': digest_file(ROOT / 'scripts/probe_raw_source_seeds.py')}
    if any(parent_audit.get(k) != v for k, v in dependencies.items()):
        raise ValueError('Frozen seed parent identity changed')
    refs = unique_rows(parent / 'retrieval.jsonl')
    expected_parent = {f'{r["dataset"]}/{a}/{r["id"]}' for r in cohort['records']
        for a in ('joint_ref', 'expanded_joint_ref', 'flat_ce_seed', 'hybrid_seed')}
    if set(refs) != expected_parent:
        raise ValueError('Seed parent panel incomplete or unexpected')
    out = ROOT / 'outputs/acl/natural_source_relations_r01'
    out.mkdir(parents=True, exist_ok=True)
    expected = {f'{r["dataset"]}/{r["id"]}' for r in cohort['records']}
    audit = {'manifest_sha256': digest_file(manifest), 'calibration_sha256': digest_file(calibration),
        'checkpoint_lock_sha256': dependencies['checkpoint_lock_sha256'],
        'parent_retrieval_sha256': digest_file(parent / 'retrieval.jsonl'),
        'parent_audit_sha256': digest_file(parent / 'audit.json'), 'encoder': lock['encoder'],
        'source_module_sha256': dependencies['source_module_sha256'],
        'candidate_module_sha256': digest_file(ROOT / 'src/mindset/acl/source_relation_candidates.py'),
        'runtime_module_sha256': digest_file(ROOT / 'src/mindset/acl/runtime.py'),
        'worker_sha256': digest_file(__file__),
        'pair_selection': 'first/closest recorded neighbor per flat-CE seed span, up to four; before labels/answers',
        'reference_budget': 'same sealed original100 flat-CE expansion substrate; not new quality/latency results',
        'question_use': 'source span relevance/admission only; not included in relation-model inputs',
        'no_gold_answers_categories_in_pair_inputs': True,
        'no_schema_construction_or_new_controller': True, 'full_runs_authorized': False}
    import fcntl
    with (out / 'worker.lock').open('a+b') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(out / 'audit.json', audit, immutable=True)
        record_environment(out / 'continuations' / os.environ['SLURM_JOB_ID'])
        if cfg['retrieval']['device'] != 'cpu':
            raise ValueError('CPU preparation required')
        from sentence_transformers import SentenceTransformer
        encoder = CachedEncoder(SentenceTransformer(lock['encoder']['checkpoint'],
            revision=lock['encoder']['revision'], device='cpu'), enabled=True)
        done = unique_rows(out / 'queries.jsonl')
        if set(done) - expected:
            raise ValueError('Unexpected natural query')
        for dataset, cluster, cases in groups_for(cohort, cfg):
            if all(f'{dataset}/{c.id}' in done for c in cases):
                continue
            events = cases[0].events; spans = source_spans(events)
            vectors = encoder.encode([s.text for s in spans], batch_size=64, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False) if spans else np.empty((0, 384))
            span_hash = stable_hash([asdict(s) for s in spans])
            for case in cases:
                rid = f'{dataset}/{case.id}'
                if rid in done:
                    continue
                ref = refs[f'{dataset}/flat_ce_seed/{case.id}']
                qvec = encoder.encode([case.question], normalize_embeddings=True,
                    convert_to_numpy=True, show_progress_bar=False)[0]
                # semantic_pairs does not use its legacy index argument.
                pairs = semantic_pairs(None, spans, vectors, qvec, ref['seed_ids'], events,
                    question_date=case.question_date)
                episode_pairs = [[spans[a].eid, spans[b].eid] for a, b in pairs]
                if episode_pairs != ref['source_pair_ids']:
                    raise ValueError('Frozen source-neighbor replay changed')
                measurements = measured_pairs(pairs, spans, events, question_date=case.question_date)
                row = {'id': rid, 'dataset': dataset, 'question_id': case.id, 'cluster': cluster,
                    'span_index_sha256': span_hash, 'recorded_neighbor_pairs': episode_pairs,
                    'measurement_pairs': measurements, 'expected_max_pairs': 4,
                    'status': 'source_pair_candidates' if measurements else 'no_admissible_source_pair',
                    'scope': 'exposed original100 instrument preparation; no new memory/QA claims'}
                append_row(out / 'queries.jsonl', row); done[rid] = row
                write_json(out / 'status.json', {'status': 'complete_inputs' if set(done) == expected else 'partial_inputs',
                    'completed_queries': len(done), 'expected_queries': 100,
                    'source_measurement_pairs': sum(len(r['measurement_pairs']) for r in done.values()),
                    'no_pair_queries': sum(not r['measurement_pairs'] for r in done.values()),
                    'generation_calls': 0, 'full_runs_authorized': False})
                print(f'NATURAL_INPUT {len(done)}/100 {rid} pairs={len(measurements)}', flush=True)
            encoder.cache.clear()
        if set(done) != expected:
            raise ValueError('Incomplete original100 preparation')
        all_pairs = [p for r in done.values() for p in r['measurement_pairs']]
        unique = {}
        for p in all_pairs:
            if p['pair_key'] in unique and unique[p['pair_key']] != p:
                raise ValueError('Pair identity collision')
            unique[p['pair_key']] = p
        write_json(out / 'unique_pairs.json', sorted(unique.values(), key=lambda p: p['pair_key']), immutable=True)
        write_json(out / 'preparation_summary.json', {'queries': len(done), 'pair_occurrences': len(all_pairs),
            'unique_pairs': len(unique), 'no_pair_queries': sum(not r['measurement_pairs'] for r in done.values()),
            'scope': 'coverage preparation, not correct natural relationship labels or energy evidence',
            'full_runs_authorized': False}, immutable=True)


if __name__ == '__main__':
    run()
