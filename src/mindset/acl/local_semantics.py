"""Offline source-span NLI, fixed checkpoint, cache usable for quality only."""
from pathlib import Path
import json
import sqlite3
from mindset.acl.io import digest_file
from mindset.utils.hashing import stable_hash


class SourceNLI:
    def __init__(self,root,cache):
        import torch
        from transformers import AutoTokenizer,AutoModelForSequenceClassification
        root=Path(root); frozen=json.loads((root/'outputs/acl/report_nli_27346/audit.json').read_text())
        path=root/'.cache/grounding'/('models--'+frozen['repository'].replace('/','--'))/'snapshots'/frozen['revision']
        if {p.name:digest_file(p) for p in path.iterdir() if p.is_file()}!=frozen['model_files']: raise ValueError('NLI files changed')
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
        self.model=AutoModelForSequenceClassification.from_pretrained(path,local_files_only=True,
            trust_remote_code=False,use_safetensors=True).cpu().eval()
        self.labels=frozen['label_mapping']
        if self.labels!=['contradiction','entailment','neutral'] or self.model.config.num_labels!=3:
            raise ValueError('Unexpected NLI label contract')
        self.revision=frozen['revision']; self.identity={
            'repository':frozen['repository'],'revision':self.revision,'model_files':frozen['model_files'],
            'decision':'bidirectional argmax; no probability threshold or posterior-truth claim',
            'max_pair_tokens':512,'long_pair_policy':'unknown/too_long, never silent truncation'}
        self.db=sqlite3.connect(cache); self.db.execute('CREATE TABLE IF NOT EXISTS nli (key TEXT PRIMARY KEY, payload TEXT)')
        self.actual_pairs=0
        sanity=self.predict([('A person is eating food.','A person is eating something.'),
            ('A person is eating food.','Nobody is eating.')])
        if sanity!=['entailment','contradiction']: raise ValueError('NLI label sanity failed')
    def predict(self,pairs):
        import torch
        output=[None]*len(pairs); pending=[]
        for i,pair in enumerate(pairs):
            key=stable_hash({'pair':pair,'revision':self.revision,'max_pair_tokens':512})
            cached=self.db.execute('SELECT payload FROM nli WHERE key=?',(key,)).fetchone()
            if cached: output[i]=json.loads(cached[0])['label']; continue
            tokens=self.tokenizer(*pair,truncation=False,add_special_tokens=True)['input_ids']
            if len(tokens)>512: output[i]='too_long'; continue
            pending.append((i,pair,key))
        with torch.inference_mode():
            for start in range(0,len(pending),16):
                batch=pending[start:start+16]; encoded=self.tokenizer([r[1][0] for r in batch],[r[1][1] for r in batch],
                    padding=True,truncation=False,return_tensors='pt')
                logits=self.model(**encoded).logits
                if not torch.isfinite(logits).all(): raise ValueError('Invalid source NLI logits')
                for (i,pair,key),values in zip(batch,logits,strict=True):
                    label=self.labels[int(values.argmax())]; output[i]=label
                    self.db.execute('INSERT OR IGNORE INTO nli VALUES (?,?)',(key,json.dumps({'label':label,'logits':values.tolist()})))
                self.db.commit(); self.actual_pairs+=len(batch)
        return output
