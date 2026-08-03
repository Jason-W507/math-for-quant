from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import NamedTuple

import numpy as np


class ConvexBoundResult(NamedTuple):
    weighted_return: float
    lower: float
    upper: float
    counterexample: float


def evaluate_contract(
    fixture: dict[str, object], oracle: dict[str, object]
) -> ConvexBoundResult:
    returns = np.array(fixture["returns"], dtype=np.float64)
    weights = np.array(fixture["weights"], dtype=np.float64)
    counterexample_weights = np.array(
        fixture["counterexample_weights"], dtype=np.float64
    )
    tolerance = float(oracle["absolute_tolerance"])
    if returns.size != weights.size:
        raise ValueError("assumption gate failed: returns and weights must have equal length")
    if not np.all(np.isfinite(returns)) or not np.all(np.isfinite(weights)):
        raise ValueError("assumption gate failed: returns and weights must be finite")
    if returns.size != counterexample_weights.size:
        raise ValueError(
            "counterexample gate failed: returns and weights must have equal length"
        )
    if not np.all(np.isfinite(counterexample_weights)):
        raise ValueError("counterexample gate failed: weights must be finite")
    if abs(float(np.sum(weights)) - 1.0) > tolerance:
        raise ValueError("assumption gate failed: weights must sum to one")
    if np.any(weights < 0.0):
        raise ValueError("assumption gate failed: weights must be nonnegative")
    if abs(float(np.sum(counterexample_weights)) - 1.0) > tolerance:
        raise ValueError("counterexample gate failed: weights must sum to one")
    if not np.any(counterexample_weights < 0.0):
        raise ValueError(
            "counterexample gate failed: deleted nonnegativity assumption is absent"
        )

    result = ConvexBoundResult(
        weighted_return=float(weights @ returns),
        lower=float(np.min(returns)),
        upper=float(np.max(returns)),
        counterexample=float(counterexample_weights @ returns),
    )
    expected = (
        float(oracle["expected_weighted_return"]),
        float(oracle["expected_lower"]),
        float(oracle["expected_upper"]),
        float(oracle["expected_counterexample"]),
    )
    if any(abs(left - right) > tolerance for left, right in zip(result, expected)):
        raise ValueError(f"oracle mismatch: observed={tuple(result)} expected={expected}")
    if not result.lower <= result.weighted_return <= result.upper:
        raise ValueError("convex bound unexpectedly failed")
    if result.counterexample <= result.upper:
        raise ValueError("deleted-assumption counterexample was not observed")
    return result


def run_contract(fixture_path: Path, oracle_path: Path) -> ConvexBoundResult:
    fixture_bytes = fixture_path.read_bytes()
    fixture = json.loads(fixture_bytes)
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    expected_digest = str(oracle["fixture"]["sha256"])
    observed_digest = hashlib.sha256(fixture_bytes).hexdigest()
    if observed_digest != expected_digest:
        raise ValueError(
            "fixture hash mismatch: "
            f"observed={observed_digest} expected={expected_digest}"
        )
    return evaluate_contract(fixture, oracle)


def main(input_path: Path | None = None) -> int:
    oracle_path = Path("evidence/ch01/oracle.json")
    if input_path is None:
        result = run_contract(Path("data/fixtures/ch01.json"), oracle_path)
    else:
        document = json.loads(input_path.read_text(encoding="utf-8"))
        input_fields = {"returns", "weights", "counterexample_weights"}
        if input_fields.intersection(document):
            fixture = json.loads(
                Path(document["fixture"]["path"]).read_text(encoding="utf-8")
            )
            fixture.update(
                {field: document[field] for field in input_fields if field in document}
            )
            result = evaluate_contract(fixture, document)
        else:
            result = run_contract(Path(document["fixture"]["path"]), input_path)
    print(
        "oracle=passed "
        f"weighted_return={result.weighted_return:.6f} lower={result.lower:.6f} "
        f"upper={result.upper:.6f} counterexample={result.counterexample:.6f}"
    )
    return 0


if __name__ == "__main__":
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    try:
        raise SystemExit(main(input_path))
    except ValueError as error:
        raise SystemExit(str(error)) from error
