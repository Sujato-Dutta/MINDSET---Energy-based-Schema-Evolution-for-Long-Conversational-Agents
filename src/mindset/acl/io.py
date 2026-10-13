from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from typing import Any

def digest_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_json(path: str | Path, value: Any, *, immutable: bool = False) -> None:
    p = Path(path)
    encoded = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if immutable and p.exists():
        if p.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"Frozen artifact differs: {p}; use a new experiment directory")
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + ".tmp")
    temp.write_text(encoded, encoding="utf-8")
    os.replace(temp, p)

def read_rows(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    # JSONL is framed by physical newlines, not every Unicode character that
    # str.splitlines() treats as a separator. U+2028/U+2029/U+0085 can legally
    # occur inside JSON strings and must remain part of the raw source text.
    with p.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]

def append_row(path: str | Path, row: dict) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())

def unique_rows(path: str | Path) -> dict[str, dict]:
    rows = read_rows(path)
    mapped = {r["id"]: r for r in rows}
    if len(mapped) != len(rows):
        raise ValueError(f"Duplicate question IDs in {path}")
    return mapped

