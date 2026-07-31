from __future__ import annotations

from decimal import Decimal, getcontext
import json
import math
from pathlib import Path
import sys

import numpy as np


FIXED_NUMPY_VERSIONS = ("2.3.5", "2.5.1")
FIXED_LABELS = [
    "IEEE 754",
    "cancellation",
    "summation",
    "linear solve",
    "least squares",
    "rank deficiency",
]
FIXED_PROVENANCE = (
    "80-digit Decimal arithmetic supplies cancellation and summation oracles; "
    "closed-form eigenvalues supply the two-by-two condition oracle; analytic "
    "solutions and fixed singular-value identities independently constrain the "
    "least-squares, scaling and rank-deficiency experiments"
)
FIXED_MARKERS = [
    "机器精度为 $2.220\\times10^{-16}$，单位舍入误差为 $1.110\\times10^{-16}$",
    "直接相减得到 $0$",
    "稳定改写与高精度值均为 $5.000\\times10^{-9}$",
    "朴素、成对、补偿与 Decimal 求和分别得到 $0$、$8$、$10$、$10$",
    "条件数约为 $4.000\\times10^8$",
    "前向误差为 $1.000\\times10^{-2}$",
    "相对残差不超过 $10^{-15}$",
    "正规方程把条件数从 $2.449\\times10^6$ 放大到约 $6.000\\times10^{12}$",
    "稳定 log-sum-exp 为 $1000.693147$",
    "列缩放把条件数从 $1.000\\times10^9$ 降到 $1.407\\times10^1$",
]
FIXED_ARRAYS = {
    "summation_values": [
        "1e16", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1",
        "-1e16",
    ],
    "logsumexp_values": [1000.0, 1000.0],
    "scaling_matrix": [[1e-8, 1.0], [2e-8, 3.0]],
    "rank_deficient_matrix": [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]],
}
FIXED_SCALARS = {
    "cancellation_input": "1e16",
    "matrix_delta": "1e-8",
    "rhs_perturbation": "1e-10",
    "least_squares_epsilon": 1e-6,
    "least_squares_rhs_perturbation": 1e-10,
    "relative_tolerance": 1e-6,
    "absolute_tolerance": 0.0,
    "misconfigured_absolute_tolerance": 1e-6,
    "decimal_tolerance": "1e-23",
    "linear_relative_tolerance": 1e-7,
    "linear_forward_tolerance": 2e-9,
    "minimum_amplification": 2e8,
    "maximum_residual": 1e-15,
    "numeric_relative_tolerance": 1e-9,
    "numeric_absolute_tolerance": 1e-12,
    "least_squares_relative_tolerance": 1e-3,
    "least_squares_absolute_tolerance": 1e-12,
}
FIXED_INTEGERS = {"decimal_precision": 80}
EXPECTED_FIELDS = {
    "cancellation_exact",
    "cancellation_naive",
    "summation_naive",
    "summation_pairwise",
    "summation_compensated",
    "summation_exact",
    "least_squares_condition",
    "normal_equation_condition",
    "qr_relative_error",
    "svd_relative_error",
    "normal_equation_relative_error",
    "normal_equation_residual",
    "qr_residual",
    "svd_residual",
    "least_squares_rank",
    "stable_logsumexp",
    "unscaled_condition",
    "scaled_condition",
    "rank_deficient_rank",
    "rank_singular_values",
    "tolerance_flags",
}
REQUIRED_FIELDS = {
    "numpy_versions",
    "numerical_labels",
    "provenance",
    "published_markers",
    "expected",
    *FIXED_ARRAYS.keys(),
    *FIXED_SCALARS.keys(),
    *FIXED_INTEGERS.keys(),
}


from math_for_quant.evidence import load_oracle_bundle

def reject_nonfinite(value: object) -> None:
    if isinstance(value, bool):
        raise SystemExit("oracle numeric inputs must not contain booleans")
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise SystemExit("oracle numeric inputs must be finite")
    elif isinstance(value, list):
        for item in value:
            reject_nonfinite(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            if key == "tolerance_flags":
                if not isinstance(item, list) or any(
                    not isinstance(flag, bool) for flag in item
                ):
                    raise SystemExit("tolerance flags must be booleans")
                continue
            reject_nonfinite(item)


def validate_oracle(oracle: dict[str, object]) -> None:
    missing = sorted(REQUIRED_FIELDS - oracle.keys())
    if missing:
        raise SystemExit(f"oracle missing required fields: {', '.join(missing)}")
    expected = oracle["expected"]
    if not isinstance(expected, dict):
        raise SystemExit("oracle expected ledger must be an object")
    missing_expected = sorted(EXPECTED_FIELDS - expected.keys())
    if missing_expected:
        raise SystemExit(
            "expected ledger missing required fields: " + ", ".join(missing_expected)
        )
    reject_nonfinite(oracle)
    if oracle["numpy_versions"] != list(FIXED_NUMPY_VERSIONS) or np.__version__ not in FIXED_NUMPY_VERSIONS:
        raise SystemExit(f"NumPy version must be one of {FIXED_NUMPY_VERSIONS}")
    if oracle["numerical_labels"] != FIXED_LABELS:
        raise SystemExit("numerical labels must match the published design")
    if oracle["provenance"] != FIXED_PROVENANCE:
        raise SystemExit("oracle provenance must match the published design")
    if oracle["published_markers"] != FIXED_MARKERS:
        raise SystemExit("published markers must match the chapter evidence")
    if any(oracle[name] != value for name, value in FIXED_ARRAYS.items()):
        raise SystemExit("canonical array design must not change")
    if (
        float(oracle["relative_tolerance"]) == 0.0
        and float(oracle["absolute_tolerance"]) > 0.0
    ):
        raise SystemExit(
            "invalid tolerance policy: equal relative errors disagree across scales"
        )
    tolerance_names = {
        "relative_tolerance",
        "absolute_tolerance",
        "misconfigured_absolute_tolerance",
        "decimal_tolerance",
        "linear_relative_tolerance",
        "linear_forward_tolerance",
        "minimum_amplification",
        "maximum_residual",
        "numeric_relative_tolerance",
        "numeric_absolute_tolerance",
    }
    if any(oracle[name] != FIXED_SCALARS[name] for name in tolerance_names):
        raise SystemExit("oracle tolerances must match the published design")
    scalar_names = FIXED_SCALARS.keys() - tolerance_names
    if any(oracle[name] != FIXED_SCALARS[name] for name in scalar_names):
        raise SystemExit("canonical scalar design must not change")
    for name, expected_integer in FIXED_INTEGERS.items():
        value = oracle[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise SystemExit(f"oracle {name} must be an integer")
        if value != expected_integer:
            raise SystemExit(f"oracle {name} must match the published design")


def neumaier_sum(values: list[float]) -> float:
    total = 0.0
    correction = 0.0
    for value in values:
        updated = total + value
        if abs(total) >= abs(value):
            correction += (total - updated) + value
        else:
            correction += (value - updated) + total
        total = updated
    return total + correction


def pairwise_sum(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    midpoint = len(values) // 2
    return pairwise_sum(values[:midpoint]) + pairwise_sum(values[midpoint:])


def scale_aware_close(a: float, b: float, atol: float, rtol: float) -> bool:
    if not all(math.isfinite(value) for value in (a, b, atol, rtol)):
        raise ValueError("comparison inputs must be finite")
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


def close(actual: float, expected: object, oracle: dict[str, object]) -> bool:
    return math.isclose(
        actual,
        float(expected),
        rel_tol=float(oracle["numeric_relative_tolerance"]),
        abs_tol=float(oracle["numeric_absolute_tolerance"]),
    )


def least_squares_close(
    actual: float, expected: object, oracle: dict[str, object]
) -> bool:
    return math.isclose(
        actual,
        float(expected),
        rel_tol=float(oracle["least_squares_relative_tolerance"]),
        abs_tol=float(oracle["least_squares_absolute_tolerance"]),
    )


def main(oracle_path: Path = Path("evidence/ch14/oracle.json")) -> int:
    oracle = load_oracle_bundle(oracle_path)
    validate_oracle(oracle)
    expected = oracle["expected"]
    assert isinstance(expected, dict)
    getcontext().prec = int(oracle["decimal_precision"])

    machine_epsilon = float(np.finfo(float).eps)
    unit_roundoff = machine_epsilon / 2.0
    smallest_subnormal = float(np.nextafter(0.0, 1.0))

    cancellation_input = Decimal(str(oracle["cancellation_input"]))
    exact_cancellation = (cancellation_input + 1).sqrt() - cancellation_input.sqrt()
    float_input = float(cancellation_input)
    naive_cancellation = math.sqrt(float_input + 1.0) - math.sqrt(float_input)
    stable_cancellation = 1.0 / (
        math.sqrt(float_input + 1.0) + math.sqrt(float_input)
    )
    exact_tolerance = Decimal(str(oracle["decimal_tolerance"]))
    if not (
        abs(exact_cancellation - Decimal(str(expected["cancellation_exact"])))
        <= exact_tolerance
        and naive_cancellation == float(expected["cancellation_naive"])
        and abs(Decimal(str(stable_cancellation)) - exact_cancellation)
        <= exact_tolerance
    ):
        raise SystemExit("cancellation ledger failed")

    values = [float(item) for item in oracle["summation_values"]]
    naive_sum = 0.0
    for value in values:
        naive_sum += value
    pairwise = pairwise_sum(values)
    compensated_sum = neumaier_sum(values)
    decimal_sum = sum(Decimal(str(item)) for item in oracle["summation_values"])
    if not (
        naive_sum == float(expected["summation_naive"])
        and pairwise == float(expected["summation_pairwise"])
        and compensated_sum == float(expected["summation_compensated"])
        and decimal_sum == Decimal(str(expected["summation_exact"]))
    ):
        raise SystemExit("summation ledger failed")

    delta = float(oracle["matrix_delta"])
    perturbation = float(oracle["rhs_perturbation"])
    matrix = np.array([[1.0, 1.0], [1.0, 1.0 + delta]])
    exact_solution = np.array([1.0, 1.0])
    rhs = matrix @ exact_solution
    perturbed_rhs = rhs.copy()
    perturbed_rhs[1] += perturbation
    numerical_solution = np.linalg.solve(matrix, perturbed_rhs)
    condition_number = float(np.linalg.cond(matrix))
    forward_error = float(
        np.linalg.norm(numerical_solution - exact_solution)
        / np.linalg.norm(exact_solution)
    )
    relative_rhs_error = float(
        np.linalg.norm(perturbed_rhs - rhs) / np.linalg.norm(rhs)
    )
    amplification = forward_error / relative_rhs_error
    relative_residual = float(
        np.linalg.norm(matrix @ numerical_solution - perturbed_rhs)
        / np.linalg.norm(perturbed_rhs)
    )
    decimal_delta = Decimal(str(oracle["matrix_delta"]))
    trace = Decimal(2) + decimal_delta
    discriminant = (Decimal(4) + decimal_delta**2).sqrt()
    analytic_condition = (trace + discriminant) / (trace - discriminant)
    analytic_forward_error = Decimal(str(oracle["rhs_perturbation"])) / decimal_delta
    if not (
        abs(condition_number - float(analytic_condition)) / float(analytic_condition)
        <= float(oracle["linear_relative_tolerance"])
        and abs(forward_error - float(analytic_forward_error))
        <= float(oracle["linear_forward_tolerance"])
        and amplification >= float(oracle["minimum_amplification"])
        and relative_residual <= float(oracle["maximum_residual"])
    ):
        raise SystemExit("linear-system ledger failed")

    epsilon = float(oracle["least_squares_epsilon"])
    least_matrix = np.array(
        [[1.0, 1.0], [1.0, 1.0 + epsilon], [1.0, 1.0 - epsilon]]
    )
    least_rhs = least_matrix @ exact_solution + np.array(
        [0.0, float(oracle["least_squares_rhs_perturbation"]),
         -float(oracle["least_squares_rhs_perturbation"])]
    )
    normal_solution = np.linalg.solve(
        least_matrix.T @ least_matrix, least_matrix.T @ least_rhs
    )
    q_matrix, r_matrix = np.linalg.qr(least_matrix, mode="reduced")
    qr_solution = np.linalg.solve(r_matrix, q_matrix.T @ least_rhs)
    u_matrix, singular_values_ls, vt_matrix = np.linalg.svd(
        least_matrix, full_matrices=False
    )
    svd_solution = vt_matrix.T @ ((u_matrix.T @ least_rhs) / singular_values_ls)
    least_condition = float(np.linalg.cond(least_matrix))
    normal_condition = float(np.linalg.cond(least_matrix.T @ least_matrix))
    normal_error = float(
        np.linalg.norm(normal_solution - exact_solution) / np.linalg.norm(exact_solution)
    )
    qr_error = float(
        np.linalg.norm(qr_solution - exact_solution) / np.linalg.norm(exact_solution)
    )
    svd_error = float(
        np.linalg.norm(svd_solution - exact_solution) / np.linalg.norm(exact_solution)
    )
    rhs_norm = float(np.linalg.norm(least_rhs))
    normal_residual = float(np.linalg.norm(least_matrix @ normal_solution - least_rhs) / rhs_norm)
    qr_residual = float(np.linalg.norm(least_matrix @ qr_solution - least_rhs) / rhs_norm)
    svd_residual = float(np.linalg.norm(least_matrix @ svd_solution - least_rhs) / rhs_norm)
    least_rank = int(np.linalg.matrix_rank(least_matrix))
    if not all(
        (
            least_squares_close(least_condition, expected["least_squares_condition"], oracle),
            least_squares_close(normal_condition, expected["normal_equation_condition"], oracle),
            least_squares_close(normal_error, expected["normal_equation_relative_error"], oracle),
            least_squares_close(qr_error, expected["qr_relative_error"], oracle),
            least_squares_close(svd_error, expected["svd_relative_error"], oracle),
            least_squares_close(normal_residual, expected["normal_equation_residual"], oracle),
            least_squares_close(qr_residual, expected["qr_residual"], oracle),
            least_squares_close(svd_residual, expected["svd_residual"], oracle),
            least_rank == expected["least_squares_rank"],
            normal_condition > least_condition**1.9,
            normal_error > qr_error,
        )
    ):
        raise SystemExit("least-squares ledger failed")

    lse_values = np.asarray(oracle["logsumexp_values"], dtype=float)
    with np.errstate(over="ignore"):
        naive_logsumexp = float(np.log(np.exp(lse_values).sum()))
    maximum = float(lse_values.max())
    stable_logsumexp = maximum + math.log(float(np.exp(lse_values - maximum).sum()))
    if not math.isinf(naive_logsumexp) or not close(
        stable_logsumexp, expected["stable_logsumexp"], oracle
    ):
        raise SystemExit("log-sum-exp ledger failed")

    scaling_matrix = np.asarray(oracle["scaling_matrix"], dtype=float)
    column_scale = np.diag(1.0 / np.linalg.norm(scaling_matrix, axis=0))
    unscaled_condition = float(np.linalg.cond(scaling_matrix))
    scaled_condition = float(np.linalg.cond(scaling_matrix @ column_scale))
    if not (
        close(unscaled_condition, expected["unscaled_condition"], oracle)
        and close(scaled_condition, expected["scaled_condition"], oracle)
        and scaled_condition < unscaled_condition
    ):
        raise SystemExit("scaling ledger failed")

    rank_matrix = np.asarray(oracle["rank_deficient_matrix"], dtype=float)
    singular_values = np.linalg.svd(rank_matrix, compute_uv=False)
    numerical_rank = int(np.linalg.matrix_rank(rank_matrix))
    expected_singular_values = np.asarray(expected["rank_singular_values"], dtype=float)
    if not (
        numerical_rank == expected["rank_deficient_rank"]
        and np.allclose(
            singular_values,
            expected_singular_values,
            rtol=float(oracle["numeric_relative_tolerance"]),
            atol=float(oracle["numeric_absolute_tolerance"]),
        )
    ):
        raise SystemExit("rank-deficiency ledger failed")

    rtol = float(oracle["relative_tolerance"])
    atol = float(oracle["absolute_tolerance"])
    small_close = scale_aware_close(1e-8, 1.0000001e-8, atol, rtol)
    large_close = scale_aware_close(1e8, 1.0000001e8, atol, rtol)
    bad_atol = float(oracle["misconfigured_absolute_tolerance"])
    bad_small = abs(1e-8 - 2e-8) <= bad_atol
    bad_large = abs(1e8 - 1.0000001e8) <= bad_atol
    if [small_close, large_close, bad_small, bad_large] != expected["tolerance_flags"]:
        raise SystemExit("tolerance ledger failed")

    print(
        "oracle=passed "
        f"ieee=({machine_epsilon:.3e},{unit_roundoff:.3e},{smallest_subnormal:.3e}) "
        f"cancel=({naive_cancellation:.3e},{stable_cancellation:.3e},{float(exact_cancellation):.3e}) "
        f"sums=({naive_sum:.1f},{pairwise:.1f},{compensated_sum:.1f},{float(decimal_sum):.1f}) "
        f"linear=({condition_number:.3e},{forward_error:.3e},{amplification:.3e},residual<=1e-15) "
        f"leastsq=({least_condition:.3e},{normal_condition:.3e},{normal_error:.3e},"
        f"{qr_error:.3e},{svd_error:.3e},{normal_residual:.3e},{qr_residual:.3e},"
        f"{svd_residual:.3e},rank={least_rank}) "
        f"logsumexp=({naive_logsumexp:.0f},{stable_logsumexp:.6f}) "
        f"scaling=({unscaled_condition:.3e},{scaled_condition:.3e}) "
        f"rank=({numerical_rank},{singular_values[0]:.3e},{singular_values[1]:.3e}) "
        f"tolerance=({int(small_close)},{int(large_close)},{int(bad_small)},{int(bad_large)})"
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        try:
            if len(sys.argv) != 4:
                raise ValueError("comparison requires two values")
            result = scale_aware_close(
                float(sys.argv[2]), float(sys.argv[3]), 0.0, 1e-6
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
        print(f"close={int(result)}")
    else:
        oracle_path = Path("evidence/ch14/oracle.json")
        if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
            oracle_path = Path(sys.argv[1])
        main(oracle_path)
