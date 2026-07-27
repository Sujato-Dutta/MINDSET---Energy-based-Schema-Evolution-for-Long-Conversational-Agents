from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import sys
from pathlib import Path
from typing import Iterable, Sequence


def distribution_summary(values: Sequence[float]) -> dict[str, float | int]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            "count": 0,
            "total": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "minimum": 0.0,
            "maximum": 0.0,
        }
    return {
        "count": len(ordered),
        "total": sum(ordered),
        "mean": sum(ordered) / len(ordered),
        "median": percentile(ordered, 0.50),
        "p95": percentile(ordered, 0.95),
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def percentile(ordered: Sequence[float], quantile: float) -> float:
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def token_and_latency_summary(
    records: Sequence[dict[str, object]], question_count: int
) -> dict[str, object]:
    input_tokens = sum(int(record.get("billed_input_tokens", 0)) for record in records)
    output_tokens = sum(int(record.get("billed_output_tokens", 0)) for record in records)
    calls = sum(not bool(record.get("cache_hit", False)) for record in records)
    cache_hits = sum(bool(record.get("cache_hit", False)) for record in records)
    divisor = max(1, question_count)
    return {
        "provider_input_tokens": input_tokens,
        "provider_output_tokens": output_tokens,
        "provider_total_tokens": input_tokens + output_tokens,
        "provider_tokens_per_question": (input_tokens + output_tokens) / divisor,
        "provider_input_tokens_per_question": input_tokens / divisor,
        "provider_output_tokens_per_question": output_tokens / divisor,
        "provider_calls": calls,
        "provider_calls_per_question": calls / divisor,
        "cache_hits": cache_hits,
        "batch_attempts": len(records),
        "answer_batch_latency_seconds": distribution_summary(
            [float(record.get("latency_seconds", 0.0)) for record in records]
        ),
    }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_sha256(root: str | Path) -> str:
    base = Path(root)
    selected: list[Path] = []
    for relative in ("src", "scripts", "configs"):
        directory = base / relative
        if directory.exists():
            selected.extend(path for path in directory.rglob("*") if path.is_file())
    requirements = base / "requirements.txt"
    if requirements.exists():
        selected.append(requirements)
    digest = hashlib.sha256()
    for path in sorted(selected, key=lambda item: item.relative_to(base).as_posix()):
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        relative = path.relative_to(base).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def runtime_environment() -> dict[str, object]:
    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "sentence-transformers", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "packages": packages,
    }


def sum_packet_bytes(packets: Iterable[Iterable[str]]) -> list[int]:
    return [
        sum(len(text.encode("utf-8")) for text in packet)
        for packet in packets
    ]
