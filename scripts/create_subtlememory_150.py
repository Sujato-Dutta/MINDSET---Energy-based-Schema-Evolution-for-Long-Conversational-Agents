from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = 'Yummytanmo/SubtleMemory'
SNAPSHOT = ROOT / 'data/subtlememory/benchmark_instances_final150.json'
MANIFEST = ROOT / 'data/manifests/subtlememory_final150.json'
IDS_FILE = ROOT / 'data/manifests/subtlememory_final150_ids.txt'
BASE_PLAN = ROOT / 'outputs/plans/main_final_v2.json'
PLAN = ROOT / 'outputs/plans/subtlememory_final150_v1.json'
STATE = ROOT / 'outputs/orchestration/subtlememory_final150_v1.json'
RELATIONS = ('complementary', 'nuanced', 'contradictory')


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_hash(value: object) -> str:
    return sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    )


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )


def main() -> None:
    dataset = load_dataset(DATASET_NAME)
    all_rows = []
    split_counts = {}
    fingerprints = {}
    for split_name in sorted(dataset):
        split = dataset[split_name]
        split_counts[split_name] = len(split)
        fingerprints[split_name] = getattr(split, '_fingerprint', None)
        for row in split:
            record = dict(row)
            record['source_split'] = split_name
            all_rows.append(record)

    relation_counts = Counter(str(row.get('relation_type')) for row in all_rows)
    if set(relation_counts) != set(RELATIONS):
        raise ValueError(f'Unexpected relation labels: {relation_counts}')
    selected = []
    for relation in RELATIONS:
        candidates = [
            row for row in all_rows if str(row.get('relation_type')) == relation
        ]
        candidates.sort(
           key=lambda row: sha256_bytes(
                (str(row.get('source_split')) + '\n' + str(row.get('instance_id'))).encode('utf-8')
           )
        )
        selected.extend(copy.deepcopy(candidates[:50]))
    selected.sort(key=lambda row: (str(row['source_split']), str(row['instance_id'])))
    ids = [str(row['instance_id']) for row in selected]
    if len(ids) != 150 or len(set(ids)) != 150:
        raise ValueError('SubtleMemory sample must contain 150 unique instances')
    selected_counts = Counter(str(row['relation_type']) for row in selected)
    if selected_counts != Counter({relation: 50 for relation in RELATIONS}):
        raise ValueError(f'Unexpected selected relation counts: {selected_counts}')

    source_rows_hash = canonical_hash(all_rows)
    snapshot = {
        'source_dataset': DATASET_NAME,
        'source_split_counts': split_counts,
        'source_relation_counts': dict(sorted(relation_counts.items())),
        'source_dataset_fingerprints': fingerprints,
        'selected_count': len(selected),
        'selection_algorithm': (
            'SHA-256 stable ordering by source split and instance ID within each relation; '
            'take 50 per relation; no RNG, answers, predictions, or metrics used.'
        ),
        'relation_counts': dict(sorted(selected_counts.items())),
        'sample_ids': ids,
        'records': selected,
    }
    write_json(SNAPSHOT, selected)
    source_hash = file_sha256(SNAPSHOT)
    ids_hash = canonical_hash(ids)
    manifest = {
        'manifest_version': 'subtlememory-final150-v1',
        'dataset': 'subtlememory',
        'role': 'reduced_final_paper_evaluation',
        'source_dataset': DATASET_NAME,
        'source_split_counts': split_counts,
        'source_relation_counts': dict(sorted(relation_counts.items())),
        'source_rows_sha256': source_rows_hash,
        'snapshot_path': 'data/subtlememory/benchmark_instances_final150.json',
        'snapshot_sha256': source_hash,
        'target_size': 150,
        'selected_count': 150,
        'selection_algorithm': snapshot['selection_algorithm'],
        'relation_counts': dict(sorted(selected_counts.items())),
        'sample_ids_path': 'data/manifests/subtlememory_final150_ids.txt',
        'sample_ids_sha256': ids_hash,
        'sample_ids': ids,
        'selected': [
            {
                'id': str(row['instance_id']),
                'relation_type': str(row['relation_type']),
                'relation_subtype': str(row.get('relation_subtype', 'unknown')),
                'source_split': str(row['source_split']),
            }
            for row in selected
        ],
    }
    write_json(MANIFEST, manifest)
    IDS_FILE.write_text('\n'.join(ids) + '\n', encoding='utf-8')

    base_plan = json.loads(BASE_PLAN.read_text(encoding='utf-8'))
    base_tasks = [
        task for task in base_plan['tasks']
        if task['dataset'] == 'memoryagentbench' and task['scope'] == 'primary'
    ]
    if len(base_tasks) != 6:
        raise ValueError(f'Expected six primary method templates, found {len(base_tasks)}')
    tasks = []
    for template in base_tasks:
        task = copy.deepcopy(template)
        task['dataset'] = 'subtlememory'
        task['manifest'] = 'data/manifests/subtlememory_final150.json'
        task['task_id'] = 'task-' + hashlib.sha256(
            ('subtlememory-final150:' + str(task['method']) + ':' + str(task['model']) + ':' + str(task['repetition_id'])).encode()
        ).hexdigest()[:16]
        tasks.append(task)
    plan_material = {
        'parent_plan_config_sha256': base_plan['config_sha256'],
        'sample_manifest_sha256': file_sha256(MANIFEST),
        'tasks': tasks,
    }
    config_hash = canonical_hash(plan_material)
    plan = {
        'config_path': base_plan['config_path'],
        'config_sha256': config_hash,
        'parent_plan': 'outputs/plans/main_final_v2.json',
        'parent_plan_config_sha256': base_plan['config_sha256'],
        'plan_version': 'subtlememory-final150-v1',
        'sample_manifest': 'data/manifests/subtlememory_final150.json',
        'sample_manifest_sha256': file_sha256(MANIFEST),
        'task_count': 6,
        'tasks': tasks,
    }
    write_json(PLAN, plan)
    write_json(
        STATE,
        {
            'plan': 'outputs/plans/subtlememory_final150_v1.json',
            'plan_config_sha256': config_hash,
            'sample_manifest': 'data/manifests/subtlememory_final150.json',
            'sample_manifest_sha256': file_sha256(MANIFEST),
            'completed': [],
            'attempts': [],
            'status': 'ready',
        },
    )
    print(json.dumps({
        'snapshot': str(SNAPSHOT),
        'manifest': str(MANIFEST),
        'ids_file': str(IDS_FILE),
        'plan': str(PLAN),
        'state': str(STATE),
        'source_total': len(all_rows),
        'source_relations': dict(sorted(relation_counts.items())),
        'sample_relations': dict(sorted(selected_counts.items())),
        'sample_count': len(ids),
        'task_count': len(tasks),
        'config_sha256': config_hash,
    }, indent=2))


if __name__ == '__main__':
    main()
