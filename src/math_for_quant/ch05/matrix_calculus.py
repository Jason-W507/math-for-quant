from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Callable


Vector = list[float]


from math_for_quant.evidence import load_oracle_bundle

def central_gradient(function: Callable[[Vector], float], point: Vector, step: float) -> Vector:
    result = []
    for index in range(len(point)):
        left = point.copy()
        right = point.copy()
        left[index] -= step
        right[index] += step
        result.append((function(right) - function(left)) / (2.0 * step))
    return result


def main(oracle_path: Path = Path("evidence/ch05/oracle.json")) -> int:
    oracle = load_oracle_bundle(oracle_path)
    required_fields = {
        "point",
        "matrix",
        "linear",
        "expected_quadratic_value",
        "expected_quadratic_gradient",
        "expected_symmetric_hessian",
        "expected_chain_gradient",
        "finite_difference_step",
        "finite_difference_scan_steps",
        "tiny_step_error_floor",
        "maximum_gradient_error",
        "expected_abs_left_slope",
        "expected_abs_right_slope",
        "portfolio_covariance",
        "portfolio_expected_returns",
        "portfolio_risk_tradeoff",
        "expected_portfolio_weights",
        "expected_portfolio_objective",
        "expected",
        "absolute_tolerance",
    }
    if not required_fields.issubset(oracle):
        raise SystemExit(
            "numeric gate failed: oracle scalars must be finite numbers"
        )
    if (
        not isinstance(oracle["point"], list)
        or len(oracle["point"]) != 2
        or not isinstance(oracle["matrix"], list)
        or len(oracle["matrix"]) != 2
        or any(not isinstance(row, list) or len(row) != 2 for row in oracle["matrix"])
        or not isinstance(oracle["linear"], list)
        or len(oracle["linear"]) != 2
        or not isinstance(oracle["portfolio_covariance"], list)
        or len(oracle["portfolio_covariance"]) != 2
        or any(
            not isinstance(row, list) or len(row) != 2
            for row in oracle["portfolio_covariance"]
        )
        or not isinstance(oracle["portfolio_expected_returns"], list)
        or len(oracle["portfolio_expected_returns"]) != 2
    ):
        raise SystemExit("ledger gate failed: derivative inputs must be 2D")
    if (
        not isinstance(oracle["expected_quadratic_gradient"], list)
        or len(oracle["expected_quadratic_gradient"]) != 2
        or not isinstance(oracle["expected_chain_gradient"], list)
        or len(oracle["expected_chain_gradient"]) != 2
        or not isinstance(oracle["expected_symmetric_hessian"], list)
        or len(oracle["expected_symmetric_hessian"]) != 2
        or any(
            not isinstance(row, list) or len(row) != 2
            for row in oracle["expected_symmetric_hessian"]
        )
        or not isinstance(oracle["expected_portfolio_weights"], list)
        or len(oracle["expected_portfolio_weights"]) != 2
    ):
        raise SystemExit(
            "ledger gate failed: derivative labels must match fixed dimensions"
        )
    if (
        not isinstance(oracle["finite_difference_scan_steps"], list)
        or oracle["finite_difference_scan_steps"]
        != [1e-2, 1e-5, 1e-8, 1e-12]
    ):
        raise SystemExit("ledger gate failed: step scan must match fixed ledger")

    numeric_scalars = [
        *oracle["point"],
        *(value for row in oracle["matrix"] for value in row),
        *oracle["linear"],
        oracle["expected_quadratic_value"],
        *oracle["expected_quadratic_gradient"],
        *(value for row in oracle["expected_symmetric_hessian"] for value in row),
        *oracle["expected_chain_gradient"],
        oracle["finite_difference_step"],
        *oracle["finite_difference_scan_steps"],
        oracle["tiny_step_error_floor"],
        oracle["maximum_gradient_error"],
        oracle["expected_abs_left_slope"],
        oracle["expected_abs_right_slope"],
        *(value for row in oracle["portfolio_covariance"] for value in row),
        *oracle["portfolio_expected_returns"],
        oracle["portfolio_risk_tradeoff"],
        *oracle["expected_portfolio_weights"],
        oracle["expected_portfolio_objective"],
        oracle["expected"],
        oracle["absolute_tolerance"],
    ]
    try:
        parsed_scalars = [float(value) for value in numeric_scalars]
    except (TypeError, ValueError, OverflowError):
        raise SystemExit(
            "numeric gate failed: oracle scalars must be finite numbers"
        ) from None
    if not all(math.isfinite(value) for value in parsed_scalars):
        raise SystemExit(
            "numeric gate failed: oracle scalars must be finite numbers"
        )
    if (
        oracle["point"] != [0.5, -1.0]
        or oracle["matrix"] != [[4.0, 2.0], [0.0, 3.0]]
        or oracle["linear"] != [-1.0, 2.0]
    ):
        raise SystemExit(
            "ledger gate failed: derivative inputs must match analytic ledger"
        )
    if float(oracle["finite_difference_step"]) != 1e-5:
        raise SystemExit(
            "ledger gate failed: finite-difference step must equal 1e-5"
        )
    if (
        float(oracle["absolute_tolerance"]) != 1e-12
        or float(oracle["maximum_gradient_error"]) != 1e-9
    ):
        raise SystemExit(
            "ledger gate failed: numeric tolerances must match fixed ledger"
        )
    if float(oracle["expected"]) != -1.0:
        raise SystemExit("ledger gate failed: published expected must equal -1")
    if (
        oracle["portfolio_covariance"] != [[2.0, 0.5], [0.5, 1.0]]
        or oracle["portfolio_expected_returns"] != [0.08, 0.04]
        or float(oracle["portfolio_risk_tradeoff"]) != 5.0
        or float(oracle["tiny_step_error_floor"]) != 1e-6
    ):
        raise SystemExit(
            "ledger gate failed: portfolio and scan inputs must match analytic ledger"
        )

    point = [float(item) for item in oracle["point"]]
    matrix = [[float(item) for item in row] for row in oracle["matrix"]]
    linear = [float(item) for item in oracle["linear"]]

    def quadratic(vector: Vector) -> float:
        product = [sum(row[j] * vector[j] for j in range(2)) for row in matrix]
        return 0.5 * sum(vector[i] * product[i] for i in range(2)) + sum(
            linear[i] * vector[i] for i in range(2)
        )

    symmetric = [
        [(matrix[i][j] + matrix[j][i]) / 2.0 for j in range(2)]
        for i in range(2)
    ]
    analytic_quadratic = [
        sum(symmetric[i][j] * point[j] for j in range(2)) + linear[i]
        for i in range(2)
    ]

    def composite(vector: Vector) -> float:
        first = vector[0] * vector[1]
        second = vector[0] + vector[1] ** 2
        return first**2 + 3.0 * second

    first = point[0] * point[1]
    outer_gradient = [2.0 * first, 3.0]
    jacobian = [[point[1], point[0]], [1.0, 2.0 * point[1]]]
    analytic_chain = [
        sum(jacobian[row][column] * outer_gradient[row] for row in range(2))
        for column in range(2)
    ]

    step = float(oracle["finite_difference_step"])
    numeric_quadratic = central_gradient(quadratic, point, step)
    numeric_chain = central_gradient(composite, point, step)
    max_error = max(
        *(abs(left - right) for left, right in zip(analytic_quadratic, numeric_quadratic)),
        *(abs(left - right) for left, right in zip(analytic_chain, numeric_chain)),
    )
    abs_left = (abs(-step) - abs(0.0)) / (-step)
    abs_right = (abs(step) - abs(0.0)) / step

    scan_errors = [
        max(
            abs(left - right)
            for left, right in zip(
                analytic_quadratic,
                central_gradient(quadratic, point, float(scan_step)),
            )
        )
        for scan_step in oracle["finite_difference_scan_steps"]
    ]
    tiny_step_failure = (
        scan_errors[-1] > float(oracle["tiny_step_error_floor"])
        and scan_errors[-1] > 1000.0 * scan_errors[1]
    )
    if not tiny_step_failure:
        raise SystemExit(
            f"finite-difference failure witness missing: errors={scan_errors}"
        )

    covariance = [
        [float(value) for value in row]
        for row in oracle["portfolio_covariance"]
    ]
    expected_returns = [
        float(value) for value in oracle["portfolio_expected_returns"]
    ]
    risk_tradeoff = float(oracle["portfolio_risk_tradeoff"])
    determinant = (
        covariance[0][0] * covariance[1][1]
        - covariance[0][1] * covariance[1][0]
    )
    inverse = [
        [covariance[1][1] / determinant, -covariance[0][1] / determinant],
        [-covariance[1][0] / determinant, covariance[0][0] / determinant],
    ]
    target = [risk_tradeoff * value for value in expected_returns]
    portfolio_weights = [
        sum(inverse[i][j] * target[j] for j in range(2))
        for i in range(2)
    ]
    covariance_product = [
        sum(covariance[i][j] * portfolio_weights[j] for j in range(2))
        for i in range(2)
    ]
    portfolio_objective = 0.5 * sum(
        portfolio_weights[i] * covariance_product[i] for i in range(2)
    ) - sum(target[i] * portfolio_weights[i] for i in range(2))

    expected = [
        float(oracle["expected_quadratic_value"]),
        *(float(item) for item in oracle["expected_quadratic_gradient"]),
        *(float(item) for item in oracle["expected_chain_gradient"]),
        float(oracle["expected_abs_left_slope"]),
        float(oracle["expected_abs_right_slope"]),
        *(float(item) for item in oracle["expected_portfolio_weights"]),
        float(oracle["expected_portfolio_objective"]),
    ]
    observed = [
        quadratic(point),
        *analytic_quadratic,
        *analytic_chain,
        abs_left,
        abs_right,
        *portfolio_weights,
        portfolio_objective,
    ]
    tolerance = float(oracle["absolute_tolerance"])
    if any(abs(left - right) > tolerance for left, right in zip(observed, expected)):
        raise SystemExit(f"oracle mismatch: observed={observed} expected={expected}")
    if symmetric != oracle["expected_symmetric_hessian"]:
        raise SystemExit(f"hessian mismatch: observed={symmetric}")
    if max_error > float(oracle["maximum_gradient_error"]):
        raise SystemExit(f"finite-difference mismatch: max_error={max_error}")

    print(
        "oracle=passed "
        f"value={observed[0]:.6f} "
        f"gradient=({analytic_quadratic[0]:.6f},{analytic_quadratic[1]:.6f}) "
        f"chain=({analytic_chain[0]:.6f},{analytic_chain[1]:.6f}) "
        f"max_error={max_error:.3e} left={abs_left:.1f} right={abs_right:.1f} "
        f"portfolio=({portfolio_weights[0]:.6f},{portfolio_weights[1]:.6f}) "
        f"objective={portfolio_objective:.6f} tiny_step_failure=passed"
    )
    return 0


if __name__ == "__main__":
    oracle_path = Path("evidence/ch05/oracle.json")
    if Path(sys.argv[0]).stem == "ch05_matrix_calculus" and len(sys.argv) > 1:
        oracle_path = Path(sys.argv[1])
    main(oracle_path)
