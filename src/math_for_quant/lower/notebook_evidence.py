from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_oracle_and_fixture(oracle_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    fixture_path = Path(str(oracle["fixture"]["path"]))
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    return oracle, fixture


def assert_expected(
    observed: dict[str, float | int], oracle: dict[str, Any]
) -> None:
    tolerance = float(oracle["absolute_tolerance"])
    for key, expected in oracle["expected"].items():
        value = observed[key]
        if abs(float(value) - float(expected)) > tolerance:
            raise SystemExit(f"{key} mismatch: observed={value} expected={expected}")
