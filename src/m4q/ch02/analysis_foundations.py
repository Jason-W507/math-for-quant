from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

from m4q.evidence import load_oracle_bundle


def main(oracle_path: Path = Path("evidence/ch02/oracle.json")) -> int:
    oracle = load_oracle_bundle(oracle_path)
    try:
        tolerance = float(oracle["absolute_tolerance"])
        value = float(oracle["initial_value"])
        intercept = float(oracle["intercept"])
        factor = float(oracle["contraction_factor"])
        iterations = int(oracle["iterations"])
        fixed_point = float(oracle["expected_fixed_point"])
        expected_iterate = float(oracle["expected_iterate"])
        expected_error = float(oracle["expected_error"])
        expected_bound = float(oracle["expected_bound"])
        expected_witness_errors = [
            float(item) for item in oracle["expected_witness_errors"]
        ]
    except (KeyError, TypeError, ValueError, OverflowError):
        raise SystemExit("numeric gate failed: oracle scalars must be finite")
    numeric_scalars = [
        tolerance,
        value,
        intercept,
        factor,
        fixed_point,
        expected_iterate,
        expected_error,
        expected_bound,
        *expected_witness_errors,
    ]
    if not all(np.isfinite(item) for item in numeric_scalars):
        raise SystemExit("numeric gate failed: oracle scalars must be finite")
    if iterations != 5:
        raise SystemExit("ledger gate failed: iterations must equal 5")
    if not np.isfinite(factor) or not 0.0 <= factor < 1.0:
        raise SystemExit("contraction gate failed: factor must satisfy 0 <= q < 1")
    first_step = intercept + factor * value
    for _ in range(iterations):
        value = intercept + factor * value

    if abs((intercept + factor * fixed_point) - fixed_point) > tolerance:
        raise SystemExit(
            "fixed-point gate failed: declared point does not satisfy T(x)=x"
        )
    error = abs(fixed_point - value)
    bound = factor**iterations / (1.0 - factor) * abs(
        first_step - float(oracle["initial_value"])
    )
    witness_indices = oracle["witness_indices"]
    if any(
        isinstance(index, bool) or not isinstance(index, int) or index <= 0
        for index in witness_indices
    ):
        raise SystemExit("witness gate failed: indices must be positive integers")
    if witness_indices != [10, 100]:
        raise SystemExit("ledger gate failed: witness indices must equal [10, 100]")
    if len(expected_witness_errors) != len(witness_indices):
        raise SystemExit(
            "witness gate failed: expected error count must match witness index count"
        )
    if len(witness_indices) < 2:
        raise SystemExit("witness gate failed: at least two witness indices are required")
    witness_errors = []
    for index in witness_indices:
        witness = np.exp(-np.log(2.0) / int(index))
        witness_errors.append(float(witness ** int(index)))

    observed = [value, error, bound, *witness_errors]
    expected = [
        expected_iterate,
        expected_error,
        expected_bound,
        *expected_witness_errors,
    ]
    if any(abs(left - right) > tolerance for left, right in zip(observed, expected)):
        raise SystemExit(f"oracle mismatch: observed={observed} expected={expected}")

    print(
        "oracle=passed "
        f"contraction_x5={value:.6f} error={error:.6f} bound={bound:.6f} "
        f"witness10={witness_errors[0]:.6f} witness100={witness_errors[1]:.6f}"
    )
    return 0


if __name__ == "__main__":
    oracle_path = Path("evidence/ch02/oracle.json")
    if Path(sys.argv[0]).stem == "ch02_analysis_foundations" and len(sys.argv) > 1:
        oracle_path = Path(sys.argv[1])
    main(oracle_path)
