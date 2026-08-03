from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from collections.abc import Callable


def load_oracle_and_fixture(oracle_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    fixture_path = Path(str(oracle["fixture"]["path"]))
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    return oracle, fixture


def assert_expected(
    observed: dict[str, float | int | str], oracle: dict[str, Any]
) -> None:
    tolerance = float(oracle["absolute_tolerance"])
    for key, expected in oracle["expected"].items():
        value = observed[key]
        if isinstance(expected, str):
            if value != expected:
                raise SystemExit(f"{key} mismatch: observed={value} expected={expected}")
            continue
        observed_number = float(value)
        expected_number = float(expected)
        if not math.isfinite(observed_number) or not math.isfinite(expected_number):
            raise SystemExit(
                f"{key} nonfinite: observed={value} expected={expected}"
            )
        if abs(observed_number - expected_number) > tolerance:
            raise SystemExit(f"{key} mismatch: observed={value} expected={expected}")


def expect_value_error(callable_: Callable[[], object], diagnostic: str) -> int:
    try:
        callable_()
    except ValueError as error:
        if diagnostic not in str(error):
            raise AssertionError(f"wrong diagnostic: {error!s}") from error
        return 1
    raise AssertionError(f"expected rejection containing {diagnostic!r}")
