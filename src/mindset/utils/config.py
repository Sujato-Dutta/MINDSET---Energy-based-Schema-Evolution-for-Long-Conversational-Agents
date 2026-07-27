from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load JSON or YAML without requiring YAML for JSON-compatible configs."""
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                f"{source} is YAML rather than JSON-compatible YAML; install PyYAML"
            ) from exc
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration root must be a mapping: {source}")
    return value


def resolve_optional_path(value: str | None, env_name: str) -> Path | None:
    override = os.getenv(env_name, "").strip()
    selected = override or (value or "").strip()
    return Path(selected).expanduser() if selected else None


def load_dotenv(path: str | Path = '.env') -> int:
    """Load simple KEY=VALUE entries without logging values.

    Existing process variables win. Names are normalized to uppercase so the
    documented numbered aliases behave consistently across operating systems.
    """
    source = Path(path)
    if not source.is_file():
        return 0
    loaded = 0
    for raw_line in source.read_text(encoding='utf-8-sig').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        name, value = line.split('=', 1)
        name = name.strip().upper()
        value = value.strip()
        if value[:1] == value[-1:] and value[:1] in {'\'', '"'}:
            value = value[1:-1]
        if name and name not in os.environ:
            os.environ[name] = value
            loaded += 1
    return loaded
