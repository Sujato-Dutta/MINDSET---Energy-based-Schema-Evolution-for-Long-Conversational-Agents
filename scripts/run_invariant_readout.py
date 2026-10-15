"""One fixed natural100 executable-readout pilot; no hosted/judge calls."""
import json
import os
import fcntl
from pathlib import Path
import numpy as np
from mindset.acl.pipeline import ROOT, server_client, request_json, record_environment, write_csv
from mindset.acl.joint_iteration import protocol, groups_for
from mindset.acl.io import read_json, write_json, digest_file, unique_rows, append_row
from mindset.acl.clients import messages as generic_messages, prompt_tokens, ANSWER_PROMPT_HASH
from mindset.acl.locomo_protocol_diagnostic import short_phrase_messages, READER_HASH
from mindset.acl.matched_source_answers import prior_answers
from mindset.baselines.event_rag import render_event
from mindset.memory.invariant_readout import registry, validate_registry, schema, messages, readout, CONTRACT_HASH
from mindset.evaluation.locomo import locomo_native_score
from mindset.evaluation.metrics import token_f1
from mindset.datasets.locomo import CATEGORY_NAMES
from mindset.utils.hashing import stable_hash

ARMS = ('raw_reference', 'same_response_direct', 'first_binding', 'invariant')


def summarize(out):
    rows = list(unique_rows(out / 'predictions.jsonl').values()); summaries = []
    for dataset in ('locomo', 'longmemeval'):
        for arm in ARMS:
            subset = [r for r in rows if r['dataset'] == dataset and r['method'] == arm]
            if not subset:
                continue
            summaries.append(dict(dataset=dataset, method=arm, questions=len(subset), expected=50,
                native_f1=float(np.mean([r['native_f1'] for r in subset])) if dataset == 'locomo' else None,
                secondary_token_f1=float(np.mean([r['token_f1'] for r in subset])),
                native_judge_accuracy=None, judge_status='deferred_by_user',
                **{m: float(np.mean([r['retrieval'][m] for r in subset])) for m in ('recall@8', 'precision@8', 'mrr@8', 'ndcg@8')},
                mean_logical_input_tokens=float(np.mean([r['input_tokens'] for r in subset])),
                mean_logical_output_tokens=float(np.mean([r['output_tokens'] for r in subset])),
                scope='exposed development100, no significance or final-paper inference'))
    if summaries:
        write_csv(out / 'comparison.csv', summaries)
    measurements = list(unique_rows(out / 'measurements.jsonl').values())
    fresh = [r for r in measurements if not r['cache_hit'] and not r['no_request'] and not r.get('inherited_program_measurement', False)]
    status = dict(status='complete_development_answers' if len(rows) == 400 else 'partial_development_answers',
        completed_cells=len(rows), expected_cells=400, program_requests_recorded=len(measurements),
        fresh_local_calls=len(fresh), fresh_local_input_tokens=sum(r['input_tokens'] for r in fresh),
        fresh_local_output_tokens=sum(r['output_tokens'] for r in fresh), hosted_calls=0, judge_calls=0,
        inherited_program_measurements=sum(r.get('inherited_program_measurement', False) for r in measurements),
        full_runs_authorized=False)
    write_json(out / 'status.json', status)
    if len(rows) != 400:
        return
    paired = {}; changes = []
    for dataset in ('locomo', 'longmemeval'):
        by = {a: {r['question_id']: r for r in rows if r['dataset'] == dataset and r['method'] == a} for a in ARMS}
        ids = set(by['raw_reference'])
        if len(ids) != 50 or any(set(v) != ids for v in by.values()):
            raise ValueError('Unpaired terminal panel')
        metric = 'native_f1' if dataset == 'locomo' else 'token_f1'
        paired[dataset] = {}
        for other in ARMS[:-1]:
            deltas = [by['invariant'][q][metric] - by[other][q][metric] for q in sorted(ids)]
            paired[dataset]['invariant minus ' + other] = dict(mean_difference=float(np.mean(deltas)),
                wins=sum(d > 1e-12 for d in deltas), ties=sum(abs(d) <= 1e-12 for d in deltas),
                losses=sum(d < -1e-12 for d in deltas), metric=metric)
        for q in sorted(ids):
            if any(by[a][q]['prediction'] != by['invariant'][q]['prediction'] for a in ARMS[:-1]):
                changes.append(dict(dataset=dataset, question_id=q, question=by['invariant'][q]['question'],
                    evaluator_only_reference=by['invariant'][q]['answer'],
                    predictions={a: by[a][q]['prediction'] for a in ARMS},
                    scores={a: by[a][q][metric] for a in ARMS},
                    execution=by['invariant'][q]['execution']))
    write_json(out / 'analysis.json', dict(summary=summaries, paired=paired, changes=changes,
        program_status_counts={s: sum(r['execution']['status'] == s for r in measurements)
            for s in sorted({r['execution']['status'] for r in measurements})},
        usage=status, novelty_established=False,
        guarantee_scope='agreement over declared operand bindings, not semantic correctness or retrieval completeness',
        caveat='Same-response direct control isolates execution, raw reference also exposes prompt-format effects; cached timings are NOT efficiency.'))


def run():
    if os.name == 'nt' or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('DGX Slurm only')
    manifest = ROOT / 'data/manifests/mindset_v2_development100.json'
    calibration = ROOT / 'outputs/acl/mindset_joint_v2_27058/calibration/calibration.json'
    cfg, cohort, _ = protocol(manifest, calibration)
    if cohort['count'] != 100:
        raise ValueError('Original100 only')
    parent = ROOT / 'outputs/acl/raw_source_seeds_r01'
    reference = unique_rows(parent / 'retrieval.jsonl')
    checkpoint = read_json(ROOT / 'configs/checkpoints.lock.json')[cfg['primary_generator']]
    compatible = dict(manifest_sha256=digest_file(manifest), calibration_sha256=digest_file(calibration),
        checkpoint=checkpoint, checkpoint_lock_sha256=digest_file(ROOT / 'configs/checkpoints.lock.json'),
        generation=cfg['generation'], seed=cfg['seed'], model_config_sha256=digest_file(ROOT / 'configs/models.yaml'),
        client_module_sha256=digest_file(ROOT / 'src/mindset/acl/clients.py'),
        reader_hashes=dict(locomo=READER_HASH, longmemeval=ANSWER_PROMPT_HASH),
        source_sha256={d: digest_file(ROOT / cfg['benchmarks'][d]['path']) for d in ('locomo', 'longmemeval')})
    previous, parent_identity = prior_answers(parent / 'answers', compatible)
    name = os.environ.get('MINDSET_INVARIANT_OUTPUT', 'invariant_readout_r02')
    if not name.startswith('invariant_readout_r') or '/' in name or '\\' in name:
        raise ValueError('Isolated experiment directory required')
    out = ROOT / 'outputs/acl' / name; out.mkdir(parents=True, exist_ok=True)
    audit = dict(**compatible, contract_hash=CONTRACT_HASH,
        module_sha256=digest_file(ROOT / 'src/mindset/memory/invariant_readout.py'),
        worker_sha256=digest_file(__file__), parent=parent_identity, methods=ARMS,
        source_retrieval_sha256=digest_file(parent / 'retrieval.jsonl'),
        fitted_coefficients=0, operational_limits=dict(source_passages=8, operand_groups=8,
            alternatives_per_group=16, execution_worlds=512, output_tokens=512, context_tokens=16384),
        declared_bindings_are_model_interpretations=True, benchmark_answers_used_for_inference=False,
        scope='single fixed exploratory natural100, not final paper or absence-of-prior-work proof',
        hosted_calls=0, full_runs_authorized=False)
    partial_parent = ROOT / 'outputs/acl/invariant_readout_r01'
    inherited = {}
    if name == 'invariant_readout_r02':
        # The original job terminally failed at the LME template boundary. Its
        # completed LC requests may be reused ONLY after exact prompt+registry
        # reconstruction below. No outcome/reference-based selection.
        with (partial_parent / 'worker.lock').open('rb') as handle:
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
            old_audit = read_json(partial_parent / 'audit.json')
            for k, v in compatible.items():
                if stable_hash(old_audit[k]) != stable_hash(v):
                    raise ValueError('Incompatible partial continuation: ' + k)
            inherited = unique_rows(partial_parent / 'measurements.jsonl')
            audit['partial_continuation'] = dict(source=str(partial_parent),
                measurements_sha256=digest_file(partial_parent / 'measurements.jsonl'),
                old_audit_sha256=digest_file(partial_parent / 'audit.json'),
                policy='all completed requests with exact reconstructed prompt/registry; no score selection')
    with (out / 'worker.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(out / 'audit.json', audit, immutable=True)
        record_environment(out / 'continuations' / os.environ['SLURM_JOB_ID'])
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint['checkpoint'], revision=checkpoint['revision'],
            cache_dir=str(Path.home() / 'MINDSET_v2/.cache/gpu_prototype/huggingface/hub'), local_files_only=True)
        url = os.environ.get('MINDSET_LOCAL_BASE_URL', '')
        if url.split('://')[-1].split(':')[0] not in ('localhost', '127.0.0.1'):
            raise ValueError('DGX local server only')
        if {r['id'] for r in request_json(url.rstrip('/') + '/models')['data']} != {checkpoint['checkpoint']}:
            raise ValueError('Wrong model')
        client = server_client(cfg['primary_generator'], cfg, cache_path=out / 'completion_cache.sqlite')
        client.allow_incomplete = True; base_options = dict(client.options)
        measured = unique_rows(out / 'measurements.jsonl'); done = unique_rows(out / 'predictions.jsonl')
        expected = {f"{r['dataset']}/{a}/{r['id']}" for r in cohort['records'] for a in ARMS}
        if set(done) - expected:
            raise ValueError('Unexpected resumed predictions')
        for dataset, cluster, cases in groups_for(cohort, cfg):
            for case in cases:
                ref = reference[f'{dataset}/flat_ce_seed/{case.id}']
                selected = ref['selected']; original = {e.event_id: e for e in case.events}
                evidence = '\n\n'.join(f'[{i}] {s["text"]}' for i, s in enumerate(selected, 1))
                if evidence != ref['evidence'] or len(selected) > 8:
                    raise ValueError('Reference evidence changed')
                for s in selected:
                    event = original[s['event_id']]
                    tokens = tokenizer.encode(render_event(event), add_special_tokens=False)
                    a, b = s['token_start'], s['token_end']
                    if not 0 <= a < b <= len(tokens) or tokenizer.decode(tokens[a:b]) != s['text']:
                        raise ValueError('Non-source evidence')
                ordinary = short_phrase_messages(case.question, evidence, case.question_date, category=case.category) if dataset == 'locomo' else generic_messages(case.question, evidence, case.question_date)
                baseline = previous[(dataset, case.id, stable_hash(ordinary))]
                reg = registry(selected); validate_registry(reg, selected)
                key = f'{dataset}/{case.id}'
                prompt = messages(ordinary, reg); count = prompt_tokens(tokenizer, prompt, thinking=False)
                if key not in measured and key in inherited:
                    old = inherited[key]
                    if old['prompt_sha256'] != stable_hash(prompt) or old['registry'] != reg:
                        raise ValueError('Partial continuation request changed')
                    copied = {**old, 'inherited_program_measurement': True}
                    append_row(out / 'measurements.jsonl', copied); measured[key] = copied
                if key not in measured:
                    no_request = count + 512 > 16384
                    if no_request:
                        raw = ''; payload = None; error = 'context_cap_no_trimming'; incomplete = True
                        input_tokens = output_tokens = 0; cache_hit = False; request_id = None
                    else:
                        client.options = {**base_options, 'response_format': {'type': 'json_schema',
                            'json_schema': {'name': 'source_program', 'strict': True, 'schema': schema(reg)}}}
                        completion = client.complete(prompt, max_tokens=512, seed=cfg['seed'])
                        if completion.input_tokens != count:
                            raise ValueError('Tokenizer count mismatch')
                        raw = completion.text; incomplete = completion.incomplete
                        input_tokens = completion.input_tokens; output_tokens = completion.output_tokens
                        cache_hit = completion.cache_hit; request_id = completion.request_id
                        payload = None; error = None
                        try:
                            payload = json.loads(raw)
                        except (TypeError, ValueError) as exc:
                            error = str(exc)
                    try:
                        if payload is None or incomplete:
                            raise ValueError(error or 'incomplete_program_response')
                        execution = readout(payload, reg)
                    except (TypeError, ValueError, KeyError, ArithmeticError) as exc:
                        error = str(exc)
                        # No additional generation and no evaluator-based repair.
                        # A malformed structured response cannot borrow the old
                        # baseline answer and pretend this is a one-call system.
                        direct = payload.get('answer', '') if isinstance(payload, dict) else ''
                        if not isinstance(direct, str): direct = ''
                        execution = dict(answer=direct, first_binding_answer=direct,
                            replaced=False, status='invalid_program', worlds=0, outcomes=[], guarantee=None)
                    measurement = dict(id=key, dataset=dataset, question_id=case.id,
                        registry=reg, payload=payload, raw_response=raw, error=error,
                        execution=execution, input_tokens=input_tokens, output_tokens=output_tokens,
                        incomplete=incomplete, cache_hit=cache_hit, no_request=no_request,
                        request_id=request_id, prompt_sha256=stable_hash(prompt), source_job_id=os.environ['SLURM_JOB_ID'])
                    append_row(out / 'measurements.jsonl', measurement); measured[key] = measurement
                m = measured[key]
                if m['prompt_sha256'] != stable_hash(prompt) or m['registry'] != reg:
                    raise ValueError('Resumed program differs')
                predictions = dict(raw_reference=baseline['prediction'],
                    same_response_direct=m['payload'].get('answer', '') if isinstance(m['payload'], dict) else '',
                    first_binding=m['execution']['first_binding_answer'], invariant=m['execution']['answer'])
                for arm, prediction in predictions.items():
                    rid = f'{dataset}/{arm}/{case.id}'
                    if rid in done:
                        continue
                    native = None
                    if dataset == 'locomo':
                        category = next(int(k) for k, v in CATEGORY_NAMES.items() if v == case.category)
                        native = locomo_native_score(prediction, case.answer, category)
                    row = dict(id=rid, dataset=dataset, method=arm, question_id=case.id, cluster=cluster,
                        question=case.question, answer=case.answer, category=case.category,
                        prediction=prediction, native_f1=native, token_f1=token_f1(prediction, case.answer),
                        retrieval=ref['metrics'], selected=selected, evidence=evidence,
                        input_tokens=baseline['input_tokens'] if arm == 'raw_reference' else m['input_tokens'],
                        output_tokens=baseline['output_tokens'] if arm == 'raw_reference' else m['output_tokens'],
                        execution=m['execution'], source_job_id=os.environ['SLURM_JOB_ID'],
                        scope='exploratory repeated-look100; native LME judge deferred')
                    append_row(out / 'predictions.jsonl', row); done[rid] = row
                summarize(out)
                print(f'INVARIANT {len(measured)}/100 {key} {m["execution"]["status"]}', flush=True)
        if set(done) != expected:
            raise ValueError('Incomplete final panel')
        summarize(out)


if __name__ == '__main__':
    run()
