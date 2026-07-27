from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from mindset.utils.hashing import stable_hash


def immutable_artifact_path(
    root: str | Path,
    *,
    method: str,
    dataset_hash: str,
    upstream_commit: str,
    config: dict[str, Any],
    suffix: str = ".json",
) -> Path:
    fingerprint = stable_hash(
        {
            "method": method,
            "dataset_hash": dataset_hash,
            "upstream_commit": upstream_commit,
            "config": config,
        }
    )
    return Path(root) / method / f"{fingerprint}{suffix}"


def write_json_once(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if target.exists():
        if target.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"Immutable artifact already exists with different contents: {target}")
        return target
    temporary = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, target)
    return target
