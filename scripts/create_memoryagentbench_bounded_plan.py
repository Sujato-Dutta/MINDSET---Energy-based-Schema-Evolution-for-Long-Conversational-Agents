"""Create a reproducible bounded-evidence MAB-150 rerun plan."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_PLAN = ROOT / 'outputs/plans/memoryagentbench_final150_v1.json'
PLAN = ROOT / 'outputs/plans/memoryagentbench_final150_bounded_v1.json'
STATE = ROOT / 'outputs/orchestration/memoryagentbench_final150_bounded_v1.json'


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def main() -> None:
    base = json.loads(BASE_PLAN.read_text(encoding='utf-8'))
    tasks = [copy.deepcopy(task) for task in base['tasks']]
    settings = {
        'bounded_evidence': True,
        'candidate_episodes': 20,
        'chunk_size_tokens': 768,
        'chunk_overlap_tokens': 96,
        'maximum_selected_chunks': 8,
        'maximum_evidence_tokens': 8000,
        'maximum_chunks_per_original_memory': 3,
        'long_context_history_bytes': 20000,
    }
    material = {
        'base_plan_sha256': sha256_bytes(BASE_PLAN.read_bytes()),
        'sample_manifest_sha256': base['sample_manifest_sha256'],
        'tasks': tasks,
        'evidence_settings': settings,
    }
    config_hash = sha256_bytes(json.dumps(material, sort_keys=True, separators=(',', ':')).encode('utf-8'))
    plan = {
        'config_path': base['config_path'],
        'config_sha256': config_hash,
        'parent_plan': 'outputs/plans/memoryagentbench_final150_v1.json',
        'parent_plan_config_sha256': base['config_sha256'],
        'plan_version': 'memoryagentbench-final150-bounded-v1',
        'sample_manifest': base['sample_manifest'],
        'sample_manifest_sha256': base['sample_manifest_sha256'],
        'task_count': len(tasks),
        'evidence_settings': settings,
        'tasks': tasks,
    }
    write_json(PLAN, plan)
    write_json(STATE, {
        'plan': 'outputs/plans/memoryagentbench_final150_bounded_v1.json',
        'plan_config_sha256': config_hash,
        'sample_manifest': base['sample_manifest'],
        'sample_manifest_sha256': base['sample_manifest_sha256'],
        'completed': [],
        'attempts': [],
        'status': 'ready',
    })
    print(json.dumps({
        'plan': str(PLAN),
        'state': str(STATE),
        'task_count': len(tasks),
        'evidence_settings': settings,
        'config_sha256': config_hash,
    }, indent=2))


if __name__ == '__main__':
    main()
