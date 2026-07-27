from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = ROOT / 'data/manifests/memoryagentbench_ttl.json'
SAMPLE_MANIFEST = ROOT / 'data/manifests/memoryagentbench_final150.json'
SAMPLE_IDS = ROOT / 'data/manifests/memoryagentbench_final150_ids.txt'
BASE_PLAN = ROOT / 'outputs/plans/main_final_v2.json'
REDUCED_PLAN = ROOT / 'outputs/plans/memoryagentbench_final150_v1.json'
STATE = ROOT / 'outputs/orchestration/memoryagentbench_final150_v1.json'


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )


def main() -> None:
    source = json.loads(SOURCE_MANIFEST.read_text(encoding='utf-8'))
    if source.get('dataset') != 'memoryagentbench':
        raise ValueError('Unexpected source dataset')
    if int(source.get('selected_count', -1)) != 700:
        raise ValueError('Expected the frozen 700-question source manifest')

    selected = copy.deepcopy(source['selected'][:150])
    ids = [str(row['id']) for row in selected]
    if len(ids) != 150 or len(set(ids)) != 150:
        raise ValueError('The reduced sample must contain 150 unique IDs')
    stratum_counts = {}
    for row in selected:
        stratum_counts[row['stratum']] = stratum_counts.get(row['stratum'], 0) + 1
    if sorted(stratum_counts.values()) != [25] * 6:
        raise ValueError(f'Unexpected reduced stratum counts: {stratum_counts}')

    source_hash = file_sha256(SOURCE_MANIFEST)
    ids_hash = sha256_bytes(('\n'.join(ids) + '\n').encode('utf-8'))
    manifest = {
        'manifest_version': 'memoryagentbench-final150-v1',
        'dataset': 'memoryagentbench',
        'role': 'reduced_final_paper_evaluation',
        'source_manifest': 'data/manifests/memoryagentbench_ttl.json',
        'source_manifest_sha256': source_hash,
        'source_target_size': 700,
        'target_size': 150,
        'selected_count': 150,
        'selection_algorithm': (
            'Deterministic prefix of the frozen 700-question round-robin order; '
            'no RNG, answers, evidence, predictions, or metrics used.'
        ),
        'selection_criteria': (
            'First 150 source entries, yielding 25 IDs in each of the six source strata.'
        ),
        'sample_ids_path': 'data/manifests/memoryagentbench_final150_ids.txt',
        'sample_ids_sha256': ids_hash,
        'stratum_distribution': dict(sorted(stratum_counts.items())),
        'sample_ids': ids,
        'selected': selected,
    }
    write_json(SAMPLE_MANIFEST, manifest)
    SAMPLE_IDS.write_text('\n'.join(ids) + '\n', encoding='utf-8')

    base_plan = json.loads(BASE_PLAN.read_text(encoding='utf-8'))
    tasks = [
        copy.deepcopy(task)
        for task in base_plan['tasks']
        if task['dataset'] == 'memoryagentbench' and task['scope'] == 'primary'
    ]
    if len(tasks) != 6:
        raise ValueError(f'Expected six primary MemoryAgentBench tasks, found {len(tasks)}')
    for task in tasks:
        task['manifest'] = 'data/manifests/memoryagentbench_final150.json'
    plan_material = {
        'parent_plan_config_sha256': base_plan['config_sha256'],
        'sample_manifest_sha256': file_sha256(SAMPLE_MANIFEST),
        'tasks': tasks,
    }
    config_hash = sha256_bytes(
        json.dumps(plan_material, sort_keys=True, separators=(',', ':')).encode('utf-8')
    )
    plan = {
        'config_path': base_plan['config_path'],
        'config_sha256': config_hash,
        'parent_plan': 'outputs/plans/main_final_v2.json',
        'parent_plan_config_sha256': base_plan['config_sha256'],
        'plan_version': 'memoryagentbench-final150-v1',
        'sample_manifest': 'data/manifests/memoryagentbench_final150.json',
        'sample_manifest_sha256': file_sha256(SAMPLE_MANIFEST),
        'task_count': 6,
        'tasks': tasks,
    }
    write_json(REDUCED_PLAN, plan)
    write_json(
        STATE,
        {
            'plan': 'outputs/plans/memoryagentbench_final150_v1.json',
            'plan_config_sha256': config_hash,
            'sample_manifest': 'data/manifests/memoryagentbench_final150.json',
            'sample_manifest_sha256': file_sha256(SAMPLE_MANIFEST),
            'completed': [],
            'attempts': [],
            'status': 'ready',
        },
    )
    print(json.dumps({
        'manifest': str(SAMPLE_MANIFEST),
        'ids_file': str(SAMPLE_IDS),
        'plan': str(REDUCED_PLAN),
        'state': str(STATE),
        'sample_count': len(ids),
        'stratum_distribution': stratum_counts,
        'task_count': len(tasks),
        'config_sha256': config_hash,
    }, indent=2))


if __name__ == '__main__':
    main()

