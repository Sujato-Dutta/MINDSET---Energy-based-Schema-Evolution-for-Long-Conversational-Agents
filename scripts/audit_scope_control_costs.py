"""Read terminal, cached HTTP timings for the scope instrumentation diagnostic.

These are instrument request costs on one offloaded MIG, not paper retrieval
latency, end-to-end QA latency, or CPU-only deployment measurements.
"""
import json
import math
import os
import sqlite3

from mindset.acl.pipeline import ROOT, record_environment
from mindset.acl.io import read_json, unique_rows, write_json, digest_file
from mindset.memory.scope_instrument_controls import summarize_controls


def percentile(values, quantile):
    if not values or not 0 <= quantile <= 1:
        raise ValueError('Nonempty timing sample and valid quantile required')
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('Invalid timing value')
    values = sorted(values); pos = (len(values) - 1) * quantile
    lo = int(pos); hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def cost_rows(rows, payloads):
    by_id = {}
    for payload in payloads:
        rid = payload.get('request_id')
        if rid in by_id and payload != by_id[rid]:
            raise ValueError('Nonunique request payload')
        by_id[rid] = payload
    output = []
    for row in rows:
        if row['cache_hit'] or row.get('parent_reuse', False):
            continue  # cache-hit zero timings are not fresh model performance
        p = by_id.get(row['request_id'])
        if p is None:
            raise ValueError('Missing cached timing payload')
        if any(p.get(a) != row.get(b) for a, b in (
            ('text', 'raw_response'), ('input_tokens', 'input_tokens'),
            ('output_tokens', 'output_tokens'), ('incomplete', 'incomplete'))):
            raise ValueError('Request payload differs from sealed row')
        latency = p['latency_seconds']
        if not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency < 0:
            raise ValueError('Invalid request timing')
        output.append({'id': row['id'], 'arm': row['arm'], 'request_id': row['request_id'],
            'latency_seconds': latency, 'input_tokens': row['input_tokens'],
            'output_tokens': row['output_tokens'], 'validation_error': row['validation_error']})
    groups = {}
    for arm in sorted({r['arm'] for r in output}):
        subset = [r for r in output if r['arm'] == arm]; times = [r['latency_seconds'] for r in subset]
        groups[arm] = {'fresh_requests': len(subset), 'invalid_or_incomplete_not_dropped': sum(
            r['validation_error'] is not None for r in subset),
            'request_p50_ms': 1000 * percentile(times, .5), 'request_p95_ms': 1000 * percentile(times, .95),
            'sum_request_seconds': sum(times), 'total_input_tokens': sum(r['input_tokens'] for r in subset),
            'total_output_tokens': sum(r['output_tokens'] for r in subset)}
    return {'groups': groups, 'fresh_request_details': output,
        'quantile_convention': 'linear interpolation at (n-1)*q',
        'scope': 'single offloaded MIG, serial instrument HTTP requests including invalid responses; NOT paper retrieval/E2E/CPU-only efficiency',
        'no_steady_state_or_multi_repetition_claim': True, 'full_runs_authorized': False}


def run():
    if os.name == 'nt' or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('DGX Slurm only')
    parent = ROOT / 'outputs/acl/scope_instrument_controls_r01'
    status = read_json(parent / 'status.json')
    if status.get('status') != 'complete_synthetic' or status.get('completed_cases') != 96:
        raise ValueError('Only terminal complete instrument outputs may be audited')
    rows = list(unique_rows(parent / 'measurements.jsonl').values())
    if len(rows) != 96:
        raise ValueError('Incomplete/mismatched diagnostic panel')
    analysis = read_json(parent / 'analysis.json')
    rederived = summarize_controls(rows)['groups']
    if any(analysis['groups'].get(k) != v for k, v in rederived.items()):
        raise ValueError('Recomputed control counts differ from terminal report')
    db = sqlite3.connect(f'file:{(parent / "completion_cache.sqlite").as_posix()}?mode=ro', uri=True)
    try:
        payloads = [json.loads(payload) for _, payload in db.execute('SELECT key,payload FROM local_completions')]
    finally:
        db.close()
    report = cost_rows(rows, payloads)
    report['source_hashes'] = {name: digest_file(parent / name) for name in (
        'audit.json', 'analysis.json', 'measurements.jsonl', 'completion_cache.sqlite')}
    report['worker_sha256'] = digest_file(__file__)
    out = ROOT / 'outputs/acl' / f'scope_control_costs_{os.environ["SLURM_JOB_ID"]}'
    out.mkdir(parents=True, exist_ok=True); record_environment(out)
    write_json(out / 'analysis.json', report, immutable=True)
    print('COSTS', {k: v for k, v in report.items() if k != 'fresh_request_details'}, flush=True)


if __name__ == '__main__':
    run()
