"""Separately authorized native knowledge-update validation, not dev100 tuning."""
from pathlib import Path
from mindset.acl.pipeline import ROOT, config
from mindset.acl.io import read_json, write_json, digest_file
from mindset.acl.calibration import validate_calibration
from mindset.acl.joint_calibration import validate_joint
from mindset.acl.iteration import verify_iteration
from mindset.acl.joint_iteration import groups_for

SCHEMA='mindset-native-knowledge-update-validation-v1'


def select_records(native, development):
    """Select by published category only; never answers or method results."""
    dev={r['id'] for r in development['records'] if r['dataset']=='longmemeval'}
    selected=[{'dataset':'longmemeval', **r, 'in_development100':r['id'] in dev}
              for r in native['records'] if r['category']=='knowledge-update']
    if not selected or len({r['id'] for r in selected})!=len(selected): raise ValueError('Bad native category IDs')
    return selected


def protocol(manifest_path,calibration_path):
    cfg=config(); validate_calibration(cfg,ROOT); calibration=validate_joint(calibration_path)
    value=read_json(manifest_path); source=ROOT/cfg['benchmarks']['longmemeval']['manifest']
    development=ROOT/'data/manifests/mindset_v2_development100.json'
    if value['schema_version']!=SCHEMA or value['selection']!='all_native_knowledge_update_in_source_order':
        raise ValueError('Wrong knowledge-update protocol')
    verify_iteration(read_json(development),cfg)
    if value['native_manifest_sha256']!=digest_file(source) or value['development_manifest_sha256']!=digest_file(development):
        raise ValueError('Manifest changed')
    if value['source_sha256']!=digest_file(ROOT/cfg['benchmarks']['longmemeval']['path']): raise ValueError('Source changed')
    if value['calibration_sha256']!=digest_file(calibration_path): raise ValueError('Calibration changed')
    if value['records']!=select_records(read_json(source),read_json(development)): raise ValueError('Category subset changed')
    if value['count']!=len(value['records']): raise ValueError('Count mismatch')
    for path,sha in value['frozen_controller_files'].items():
        if digest_file(ROOT/path)!=sha: raise ValueError('Frozen controller changed: '+path)
    return cfg,value,calibration


def freeze(manifest_path,calibration_path):
    cfg=config(); validate_calibration(cfg,ROOT); calibration=validate_joint(calibration_path)
    native=ROOT/cfg['benchmarks']['longmemeval']['manifest']
    development=ROOT/'data/manifests/mindset_v2_development100.json'
    verify_iteration(read_json(development),cfg)
    records=select_records(read_json(native),read_json(development))
    # This exposure audit supplements, rather than replaces, disclosure of
    # legacy-paper benchmark use. It does not certify untouched test data.
    target={r['id'] for r in records}; exposed={}; files={}
    for pattern in ('**/evidence.jsonl','**/retrieval.jsonl','**/queries.jsonl'):
        for path in sorted((ROOT/'outputs/acl').glob(pattern)):
            found=set()
            with path.open(encoding='utf8') as stream:
                import json
                for line in stream:
                    if not line.strip(): continue
                    row=json.loads(line)
                    if row.get('dataset')=='longmemeval' and row.get('question_id') in target:
                        found.add(row['question_id'])
            if found:
                relative=path.relative_to(ROOT).as_posix(); files[relative]=digest_file(path)
                for qid in found: exposed.setdefault(qid,[]).append(relative)
    controllers=('src/mindset/methods/mdl_state.py','src/mindset/methods/report_grounding.py',
                 'src/mindset/memory/report_frames.py','configs/acl.yaml','configs/checkpoints.lock.json')
    value={'schema_version':SCHEMA,'count':len(records),'dataset':'longmemeval',
        'selection':'all_native_knowledge_update_in_source_order','role':'frozen_no_retuning_diagnostic',
        'native_manifest_sha256':digest_file(native),'development_manifest_sha256':digest_file(development),
        'source_sha256':digest_file(ROOT/cfg['benchmarks']['longmemeval']['path']),
        'calibration_sha256':digest_file(calibration_path),'parameters':calibration['selected_parameters'],
        'records':records,'overlap_development100_ids':sorted(r['id'] for r in records if r['in_development100']),
        'known_v2_scored_exposure':exposed,'exposure_artifact_hashes':files,
        'historical_benchmark_exposure':True,'pristine_holdout_claim':False,
        'report_panels':['all_native_KU','development100_overlap','not_in_development100','not_in_known_v2_scored_artifacts'],
        'frozen_controller_files':{p:digest_file(ROOT/p) for p in controllers},'parameter_fit':False,
        'new_scalar_search':False,'full_runs_authorized':False}
    write_json(manifest_path,value,immutable=True)
    return value
