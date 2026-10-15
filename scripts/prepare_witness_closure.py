"""One integrated reconstruction candidate with matched component controls."""
import os
import json
import subprocess
from pathlib import Path
import numpy as np
from mindset.acl.pipeline import ROOT, rank_metrics, record_environment, write_csv
from mindset.acl.joint_iteration import protocol, groups_for
from mindset.acl.io import read_json, write_json, unique_rows, append_row, digest_file
from mindset.baselines.event_rag import render_event
from mindset.memory.witness_closure import calendar_constraints, select_packet

ARMS = ('raw', 'calendar_only', 'closure_only', 'witness_closed', 'window_calendar')
SOURCE_ADAPTER = 'frozen_raw_sources_with_explicit_calendar_constraints_v1'


def summarize(out):
    rows = list(unique_rows(out / 'retrieval.jsonl').values()); reports = []
    for dataset in ('locomo', 'longmemeval'):
        for arm in ARMS:
            subset = [r for r in rows if r['dataset'] == dataset and r['method'] == arm]
            if subset:
                reports.append(dict(dataset=dataset, method=arm, questions=len(subset),
                    **{m: float(np.mean([r['metrics'][m] for r in subset])) for m in ('recall@8', 'precision@8', 'mrr@8', 'ndcg@8')},
                    mean_evidence_tokens=float(np.mean([r['evidence_tokens'] for r in subset])),
                    declared_closed_packets=sum(r['reconstruction']['closure_supported'] for r in subset),
                    scope='repeated-look development, not final paper results'))
    if reports:
        write_csv(out / 'comparison.csv', reports)
    write_json(out / 'status.json', dict(status='complete_development' if len(rows) == 500 else 'partial_development',
        completed_cells=len(rows), expected_cells=500, qa_calls=0, judge_calls=0, full_runs_authorized=False))


def run():
    if os.name == 'nt' or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('DGX Slurm only')
    manifest = ROOT / 'data/manifests/mindset_v2_development100.json'
    calibration = ROOT / 'outputs/acl/mindset_joint_v2_27058/calibration/calibration.json'
    cfg, cohort, _ = protocol(manifest, calibration)
    parent = ROOT / 'outputs/acl/raw_source_seeds_r01'
    if read_json(parent / 'status.json')['status'] != 'complete_development':
        raise ValueError('Parent not terminal')
    reference = unique_rows(parent / 'retrieval.jsonl')
    groups = list(groups_for(cohort, cfg))
    out = ROOT / 'outputs/acl/witness_closure_r01'; out.mkdir(parents=True, exist_ok=True)
    audit = dict(manifest_sha256=digest_file(manifest), calibration_sha256=digest_file(calibration),
        checkpoint_lock_sha256=digest_file(ROOT / 'configs/checkpoints.lock.json'), methods=ARMS,
        source_adapter=SOURCE_ADAPTER, worker_sha256=digest_file(__file__),
        method_sha256=digest_file(ROOT / 'src/mindset/memory/witness_closure.py'),
        syntax_worker_sha256=digest_file(ROOT / 'scripts/parse_witness_context.py'),
        reference_sha256=digest_file(parent / 'retrieval.jsonl'),
        hypothesis='Budgeted context closure improves reconstruction beyond date normalization or naive neighbouring windows',
        limits=dict(roots=8, final_sources=8, max_previous_sources_per_root=8),
        new_fitted_coefficients=0, operational_rank_utility='1 / (1 + frozen anchor rank)',
        root_retrieval='unchanged strong flat_ce_seed panel, logical 96 local CE pairs',
        schema_write_energy_used=False, reference_answers_available_to_selector=False,
        semantic_dependency_correctness='not certified; parser hypotheses explicitly logged',
        novelty='unproven; closure optimization and temporal normalization individually established',
        full_runs_authorized=False)
    import fcntl
    with (out / 'worker.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(out / 'audit.json', audit, immutable=True)
        record_environment(out / 'continuations' / os.environ['SLURM_JOB_ID'])
        catalogs = {}; syntax_rows = []
        for dataset, cluster, cases in groups:
            events = cases[0].events; original = {e.event_id: e for e in events}
            previous = {}; latest = {}; order = {e.event_id: i for i, e in enumerate(events)}
            for e in events:
                previous[e.event_id] = latest.get(e.session_id)
                latest[e.session_id] = e.event_id
            needed = set()
            for case in cases:
                for s in reference[f'{dataset}/flat_ce_seed/{case.id}']['selected']:
                    eid = s['event_id']
                    for _ in range(9):
                        if eid is None:
                            break
                        needed.add(eid); eid = previous[eid]
            for eid in sorted(needed, key=lambda x: order[x]):
                syntax_rows.append(dict(event_id=f'{dataset}/{cluster}/{eid}', text=original[eid].text))
            catalogs[(dataset, cluster)] = original, previous, order, needed
        write_json(out / 'syntax_inputs.json', syntax_rows, immutable=True)
        if not (out / 'syntax.json').exists():
            result = subprocess.run([str(ROOT / '.venv-syntax/bin/python'), str(ROOT / 'scripts/parse_witness_context.py')],
                input=json.dumps(syntax_rows), capture_output=True, text=True, check=True)
            write_json(out / 'syntax.json', json.loads(result.stdout), immutable=True)
        syntax = read_json(out / 'syntax.json')['sources']
        from transformers import AutoTokenizer
        checkpoint = read_json(ROOT / 'configs/checkpoints.lock.json')[cfg['evidence_tokenizer']]
        tokenizer = AutoTokenizer.from_pretrained(checkpoint['checkpoint'], revision=checkpoint['revision'],
            cache_dir=str(Path.home() / 'MINDSET_v2/.cache/gpu_prototype/huggingface/hub'), local_files_only=True)
        counts = {}
        def count(text):
            if text not in counts:
                counts[text] = len(tokenizer.encode(text, add_special_tokens=False))
            return counts[text]
        done = unique_rows(out / 'retrieval.jsonl')
        expected = {f"{r['dataset']}/{a}/{r['id']}" for r in cohort['records'] for a in ARMS}
        if set(done) - expected:
            raise ValueError('Unexpected resumed reconstruction cell')
        for dataset, cluster, cases in groups:
            original, previous, order, needed = catalogs[(dataset, cluster)]
            for case in cases:
                base = reference[f'{dataset}/flat_ce_seed/{case.id}']
                ceiling = reference[f'{dataset}/joint_ref/{case.id}']['evidence_tokens']
                roots = [s['event_id'] for s in base['selected']]
                passages = {s['event_id']: s for s in base['selected']}
                candidates = set()
                for eid in roots:
                    for _ in range(9):
                        if eid is None:
                            break
                        candidates.add(eid); eid = previous[eid]
                deps = {}; window = {}; notes = {}
                for eid in sorted(candidates, key=lambda x: order[x]):
                    event = original[eid]
                    if eid not in passages:
                        tokens = tokenizer.encode(render_event(event), add_special_tokens=False)
                        # Context predecessor's last window; exact source token
                        # offsets are retained, including any truncated header.
                        start = max(0, len(tokens) - cfg['retrieval']['chunk_tokens'])
                        passages[eid] = dict(event_id=eid, session_id=event.session_id,
                            timestamp=event.timestamp, speaker=event.speaker,
                            text=tokenizer.decode(tokens[start:]), token_start=start, token_end=len(tokens))
                    parsed = syntax[f'{dataset}/{cluster}/{eid}']
                    import hashlib
                    if parsed['source_sha256'] != hashlib.sha256(event.text.encode()).hexdigest():
                        raise ValueError('Parser source differs')
                    if parsed['requires_previous_turn']:
                        deps[eid] = [previous[eid] or f'UNAVAILABLE/{eid}']
                    if eid in roots and previous[eid]:
                        window[eid] = [previous[eid]]
                    notes[eid] = [n for n in calendar_constraints(eid, event.text, event.timestamp)
                                  if n['quote'] in passages[eid]['text']]
                for arm in ARMS:
                    rid = f'{dataset}/{arm}/{case.id}'
                    if rid in done:
                        continue
                    if arm == 'raw':
                        selected = base['selected']; packet = base['evidence']
                        proof = dict(closure_supported=False, calendar_constraints=[], fallback=None)
                    else:
                        selected, packet, proof = select_packet(roots, passages,
                            window if arm == 'window_calendar' else deps, notes, count, ceiling, order,
                            close=arm != 'calendar_only', calendar=arm != 'closure_only')
                    raw_packet = '\n\n'.join(f'[{i}] {s["text"]}' for i, s in enumerate(selected, 1))
                    if count(packet) > ceiling or len(selected) > 8:
                        raise ValueError('Matched budget violated')
                    row = dict(id=rid, dataset=dataset, method=arm, question_id=case.id, cluster=cluster,
                        category=case.category, selected=selected, evidence=raw_packet,
                        annotated_evidence=packet, evidence_tokens=count(packet), matched_token_ceiling=ceiling,
                        metrics=rank_metrics(selected, case, 8), logical_ce_pairs=96,
                        reconstruction=proof, root_ids=roots, declared_dependencies=deps,
                        scope='integrated reconstruction pilot; exposed original100; not final paper results')
                    append_row(out / 'retrieval.jsonl', row); done[rid] = row
                summarize(out); counts.clear()
                print(f'RECONSTRUCT {len(done)}/500 {case.id}', flush=True)
        if set(done) != expected:
            raise ValueError('Incomplete reconstruction panel')
        summarize(out)


if __name__ == '__main__':
    run()
