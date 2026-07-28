from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


SCHEMA_DIRECTORY = Path(__file__).resolve().parents[2] / "schemas"


def validate_document(document: Any, schema_name: str) -> str | None:
    schema_path = SCHEMA_DIRECTORY / f"{schema_name}.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors = sorted(
        Draft202012Validator(schema).iter_errors(document),
        key=lambda error: tuple(str(item) for item in error.absolute_path),
    )
    if not errors:
        return None
    error = errors[0]
    location = ".".join(str(item) for item in error.absolute_path) or "<root>"
    return f"{schema_name} schema validation failed at {location}: {error.message}"
