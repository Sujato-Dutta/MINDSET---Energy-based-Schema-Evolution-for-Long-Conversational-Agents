"""DGX-only input-only NLI predicate-equivalence cache; no QA/API calls."""
from pathlib import Path
from collections import defaultdict
import argparse
import json
import hashlib
import os
import re


def predicate_owner(row,frame):
    if row['kind']=='episode': return frame['owner']
    # Old projections have no owner field and keep the exact user default.
    # An explicitly unresolved new owner must not inherit that default.
    return row['report_query'].get('contract',{}).get('owner','user')


def main(a):
    import torch
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
    projection = Path(a.projections)
    status = json.loads((projection/'status.json').read_text())
    if status['status'] != 'complete_projection': raise ValueError('Incomplete projection')
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    repo = 'cross-encoder/nli-MiniLM2-L6-H768'
    frozen = json.loads(Path(a.frozen_audit).read_text()) if a.frozen_audit else None
    if frozen and frozen['repository'] != repo: raise ValueError('Frozen model differs')
    revision = frozen['revision'] if frozen else HfApi().model_info(repo).sha
    if not re.fullmatch('[0-9a-f]{40}', revision): raise ValueError('Unpinned revision')
    model_path = Path(snapshot_download(repo, revision=revision, cache_dir=a.cache,
        local_files_only=bool(frozen),
        allow_patterns=['*.json', 'vocab.txt', 'vocab.json', 'merges.txt', '*.model', 'model.safetensors']))
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True,
        trust_remote_code=False, use_safetensors=True).cpu().eval()
    if model.config.num_labels != 3: raise ValueError('Unexpected label count')
    labels = ['contradiction', 'entailment', 'neutral']  # Official publisher mapping.
    if frozen and frozen['model_files'] != {p.name:sha(p) for p in model_path.iterdir() if p.is_file()}:
        raise ValueError('Frozen model files changed')
    audit = {'repository': repo, 'revision': revision, 'label_mapping': labels,
        'model_files': {p.name: sha(p) for p in model_path.iterdir() if p.is_file()},
        'projection_sha256': sha(projection/'reports.jsonl'), 'worker_sha256': sha(__file__),
        'scope': 'value-free predicate paraphrases ONLY; no answer/gold/truth inputs',
        'threshold': 'argmax entailment both directions; no tuned probability cutoff', 'api_calls': 0}
    (out/'audit.json').write_text(json.dumps(audit, indent=2))
    def infer(pairs):
        result = []
        with torch.inference_mode():
            for start in range(0, len(pairs), 16):
                batch = pairs[start:start+16]
                encoded = tokenizer([p[0] for p in batch], [p[1] for p in batch],
                    padding=True, truncation=True, max_length=128, return_tensors='pt')
                logits = model(**encoded).logits
                if not torch.isfinite(logits).all(): raise ValueError('Invalid NLI logits')
                result.extend({'premise': p[0], 'hypothesis': p[1], 'label': labels[int(v.argmax())],
                    'logits': v.tolist()} for p, v in zip(batch, logits, strict=True))
        return result
    sanity = infer([('A person is eating food.', 'A person is eating something.'),
                    ('A person is eating food.', 'Nobody is eating.')])
    if [r['label'] for r in sanity] != ['entailment', 'contradiction']: raise ValueError(('Label sanity failed', sanity))
    (out/'sanity.json').write_text(json.dumps(sanity, indent=2))
    typed = defaultdict(set)
    input_rows=[]
    with (projection/'reports.jsonl').open(encoding='utf8') as f:
        for line in f:
            row = json.loads(line)
            input_rows.append(row)
            frames = row['reports'] if row['kind'] == 'episode' else row['report_query']['probes']
            for frame in frames:
                # Canonical probes only. No values, source gold, answers or categories.
                typed[(row['dataset'], row['cluster'], tuple(frame['unit_terms']))].add(frame['probe'])
    if a.neighbors:
        import numpy as np
        from sentence_transformers import SentenceTransformer
        lock=json.loads((Path(__file__).resolve().parents[1]/'configs/checkpoints.lock.json').read_text())['encoder']
        encoder=SentenceTransformer(lock['checkpoint'],revision=lock['revision'],device='cpu')
        all_probes=sorted({f['probe'] for row in input_rows for f in
            (row['reports'] if row['kind']=='episode' else row['report_query']['probes'])})
        vectors=encoder.encode(all_probes,batch_size=64,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
        by_probe=dict(zip(all_probes,vectors,strict=True)); past=defaultdict(set); selected=set()
        for row in input_rows:
            for frame in row['reports'] if row['kind']=='episode' else row['report_query']['probes']:
                key=(row['dataset'],row['cluster'],tuple(frame['unit_terms']),
                     predicate_owner(row,frame),
                     tuple(frame.get('holder_terms',[])),tuple(frame.get('contexts',[])))
                probe=frame['probe']; eligible=sorted(past[key]-{probe})
                if eligible:
                    ranked=sorted(eligible,key=lambda p:(-float(by_probe[probe]@by_probe[p]),p))[:a.neighbors]
                    for candidate in ranked: selected.update(((probe,candidate),(candidate,probe)))
                if row['kind']=='episode': past[key].add(probe)
        pairs=sorted(selected)
        audit.update(pair_candidate_budget=a.neighbors,embedding_checkpoint=lock,
            pair_policy='source-to-previous-source only; queries-to-observed-source; fixed dense topK and bidirectional NLI',
            no_future_source_candidate_binding=True)
        (out/'audit.json').write_text(json.dumps(audit,indent=2))
        del encoder
    else:
        pairs = sorted({(x,y) for probes in typed.values() for x in probes for y in probes if x != y})
    with (out/'pairs.jsonl').open('w', encoding='utf8') as stream:
        for start in range(0, len(pairs), 256):
            for row in infer(pairs[start:start+256]): stream.write(json.dumps(row)+'\n')
            stream.flush(); print(f'{min(start+256,len(pairs))}/{len(pairs)} NLI pairs', flush=True)
    (out/'status.json').write_text(json.dumps({'status':'complete_nli_cache', 'pairs':len(pairs), 'api_calls':0}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--projections', required=True); p.add_argument('--output', required=True); p.add_argument('--cache', required=True)
    p.add_argument('--frozen-audit')
    p.add_argument('--neighbors',type=int,default=0)
    a=p.parse_args()
    if a.neighbors<0: p.error('Invalid candidate budget')
    if os.name == 'nt' or not os.environ.get('SLURM_JOB_ID'): p.error('DGX Slurm only')
    main(a)
