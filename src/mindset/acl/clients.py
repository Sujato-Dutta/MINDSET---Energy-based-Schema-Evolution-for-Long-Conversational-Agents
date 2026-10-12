from __future__ import annotations
import json
import time
import sqlite3
from pathlib import Path
import urllib.error
import urllib.request
from dataclasses import dataclass
from mindset.acl.io import append_row
from mindset.utils.hashing import stable_hash

@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    request_id: str | None
    cache_hit: bool = False

class LocalClient:
    """No remote fallback. Calls an explicitly configured local vLLM server."""
    def __init__(self, base_url: str, model: str, *, timeout: int = 600, options: dict | None = None,
                 cache_path=None,revision=""):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model; self.timeout = timeout; self.options = options or {}
        self.revision=revision
        self.cache=None
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True,exist_ok=True)
            self.cache=sqlite3.connect(cache_path,timeout=30)
            self.cache.execute("PRAGMA journal_mode=WAL")
            self.cache.execute("CREATE TABLE IF NOT EXISTS local_completions (key TEXT PRIMARY KEY, payload TEXT)")
    def complete(self, messages: list[dict], max_tokens: int = 512, seed: int = 0) -> Completion:
        body = {"model": self.model, "messages": messages, "temperature": 0, "seed": seed,
                "max_tokens": max_tokens, **self.options}
        key=stable_hash({"request":body,"checkpoint_revision":self.revision})
        if self.cache:
            cached=self.cache.execute("SELECT payload FROM local_completions WHERE key=?",(key,)).fetchone()
            if cached:
                payload=json.loads(cached[0]); payload["cache_hit"]=True; payload["latency_seconds"]=0
                return Completion(**payload)
        started = time.perf_counter()
        # Limited transient retries, without silently accepting malformed/empty output.
        for attempt in range(3):
            try:
                req = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    data = json.load(response)
                break
            except urllib.error.HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504} or attempt == 2:
                    raise
                time.sleep(min(10, 2 ** attempt))
        text = data["choices"][0]["message"].get("content", "")
        if not text or data["choices"][0].get("finish_reason") == "length":
            raise ValueError("Empty or truncated answer; no prediction committed")
        usage = data["usage"]
        result=Completion(text.strip(), usage["prompt_tokens"], usage["completion_tokens"],
                          time.perf_counter() - started, data.get("id"))
        if self.cache:
            from dataclasses import asdict
            self.cache.execute("INSERT OR IGNORE INTO local_completions VALUES (?,?)",
                               (key,json.dumps(asdict(result))))
            self.cache.commit()
        return result

def load_tokenizer(checkpoint: dict):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(checkpoint["checkpoint"], revision=checkpoint["revision"])

ANSWER_PROMPT = """Answer the question using only the provided timestamped conversation evidence.
Give a concise, complete answer. Respect changes over time and the question's requested date.
If the evidence does not contain the answer, respond: No information available.
Do not explain the retrieval process."""
ANSWER_PROMPT_HASH = stable_hash(ANSWER_PROMPT)

def messages(question: str, evidence: str, question_date: str = "") -> list[dict]:
    date = f"Question date: {question_date}\n" if question_date else ""
    return [{"role": "system", "content": ANSWER_PROMPT},
            {"role": "user", "content": f"{date}Question: {question}\n\nEvidence:\n{evidence}"}]

def prompt_tokens(tokenizer, prompt_messages: list[dict], *, thinking: bool | None = None) -> int:
    kwargs = {"enable_thinking": thinking} if thinking is not None else {}
    return len(tokenizer.apply_chat_template(prompt_messages, tokenize=True,
                                             add_generation_prompt=True, **kwargs))

