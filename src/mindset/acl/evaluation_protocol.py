"""External published semantic-judge reference; not the ACL LightMem baseline.

The source belongs to the different ICLR LightMem repository. Using its prompt
does not reproduce the ACL paper's protocol or its reported scores. Native
LoCoMo F1 is the primary conventional metric; this judge is supplementary.
"""
from __future__ import annotations
import ast
import json
import re
import urllib.request
from pathlib import Path
from mindset.acl.io import digest_file,read_json,write_json
ROOT=Path(__file__).resolve().parents[3]
COMMIT="8449d574df6bae1bdf3314a1564da65e2f37e046"
URL=f"https://raw.githubusercontent.com/zjunlp/LightMem/{COMMIT}/experiments/locomo/llm_judge.py"

def pin_evaluation():
    lock=ROOT/"configs/evaluation.lock.json"; path=ROOT/"external/reference/lightmem_locomo_judge.py"
    if lock.exists():
        value=read_json(lock)
        if value["locomo"]["commit"]!=COMMIT or digest_file(path)!=value["locomo"]["sha256"]:
            raise ValueError("Published evaluator differs from frozen evaluation lock")
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen(URL,timeout=60) as r: data=r.read()
    path.write_bytes(data)
    license_url=f"https://raw.githubusercontent.com/zjunlp/LightMem/{COMMIT}/LICENSE"
    with urllib.request.urlopen(license_url,timeout=60) as r: data=r.read()
    (path.parent/"LICENSE.LightMem").write_bytes(data)
    write_json(lock,{"locomo":{"repository":"zjunlp/LightMem","commit":COMMIT,
        "url":URL,"sha256":digest_file(path),"categories":[1,2,3,4],"prompt":"ACCURACY_PROMPT",
        "judge_model_substitution":"gemini-3.1-flash-lite",
        "comparability":"published_prompt_and_category_protocol; not identical published model setup"}},immutable=True)

def published_locomo_prompt(question,reference,prediction):
    lock=read_json(ROOT/"configs/evaluation.lock.json")["locomo"]
    path=ROOT/"external/reference/lightmem_locomo_judge.py"
    if digest_file(path)!=lock["sha256"]: raise ValueError("LoCoMo judge source hash mismatch")
    tree=ast.parse(path.read_text(encoding="utf-8"))
    node=next(n for n in tree.body if isinstance(n,ast.Assign)
              and any(isinstance(t,ast.Name) and t.id=="ACCURACY_PROMPT" for t in n.targets))
    return ast.literal_eval(node.value).format(question=question,gold_answer=reference,generated_answer=prediction)

def parse_published_label(text):
    text=text.strip()
    if text.startswith(chr(96)*3):
        match=re.fullmatch(chr(96)*3+r"(?:json)?\s*(.*?)\s*"+chr(96)*3,text,flags=re.DOTALL)
        if not match: raise ValueError("Malformed fenced JSON")
        text=match.group(1)
    value=json.loads(text)
    if not isinstance(value,dict) or value.get("label") not in {"CORRECT","WRONG"}:
        raise ValueError("Published LoCoMo judge must return JSON label CORRECT or WRONG")
    return value["label"]=="CORRECT"
