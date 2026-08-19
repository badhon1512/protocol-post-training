"""Small JSON/JSONL helpers shared by training and dataset-building scripts."""

import json
from pathlib import Path


def load_records(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []

    try:
        value = json.loads(text)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            return [value]
    except json.JSONDecodeError:
        pass

    records = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"JSONL record at {path}:{line_number} is not an object")
        records.append(value)
    return records
