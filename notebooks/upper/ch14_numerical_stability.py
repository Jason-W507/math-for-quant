# %% [markdown]
# # 浮点误差、稳定算法与病态线性系统
#
# Decimal 和解析二阶公式提供独立 oracle；float64 实验只做被测实现。

# %%
from __future__ import annotations

from decimal import Decimal, getcontext
import json
import math
from pathlib import Path
import sys

import numpy as np


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


def scale_aware_close(a: float, b: float, atol: float, rtol: float) -> bool:
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


def main(oracle_path: Path = Path("evidence/ch14/oracle.json")) -> int:
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    getcontext().prec = int(oracle["decimal_precision"])

    cancellation_input = Decimal(oracle["cancellation_input"])
    exact_cancellation = (cancellation_input + 1).sqrt() - cancellation_input.sqrt()
    float_input = float(cancellation_input)
    naive_cancellation = math.sqrt(float_input + 1.0) - math.sqrt(float_input)
    stable_cancellation = 1.0 / (
        math.sqrt(float_input + 1.0) + math.sqrt(float_input)
    )

    values = [float(item) for item in oracle["summation_values"]]
    naive_sum = 0.0
    for value in values:
        naive_sum += value
    compensated_sum = neumaier_sum(values)
    decimal_sum = sum(Decimal(item) for item in oracle["summation_values"])

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

    decimal_delta = Decimal(oracle["matrix_delta"])
    trace = Decimal(2) + decimal_delta
    discriminant = (Decimal(4) + decimal_delta**2).sqrt()
    analytic_condition = (trace + discriminant) / (trace - discriminant)
    analytic_forward_error = Decimal(oracle["rhs_perturbation"]) / decimal_delta

    rtol = float(oracle["relative_tolerance"])
    atol = float(oracle["absolute_tolerance"])
    small_close = scale_aware_close(1e-8, 1.0000001e-8, atol, rtol)
    large_close = scale_aware_close(1e8, 1.0000001e8, atol, rtol)
    bad_atol = float(oracle["misconfigured_absolute_tolerance"])
    bad_small = abs(1e-8 - 2e-8) <= bad_atol
    bad_large = abs(1e8 - 1.0000001e8) <= bad_atol
    if small_close != large_close:
        raise SystemExit(
            "invalid tolerance policy: equal relative errors disagree across scales"
        )

    expected = oracle["expected"]
    exact_tolerance = Decimal(oracle["decimal_tolerance"])
    checks = [
        abs(exact_cancellation - Decimal(expected["cancellation_exact"]))
        <= exact_tolerance,
        naive_cancellation == float(expected["cancellation_naive"]),
        abs(Decimal(str(stable_cancellation)) - exact_cancellation)
        <= exact_tolerance,
        naive_sum == float(expected["summation_naive"]),
        compensated_sum == float(expected["summation_compensated"]),
        decimal_sum == Decimal(expected["summation_exact"]),
        abs(condition_number - float(analytic_condition))
        / float(analytic_condition)
        <= float(oracle["linear_relative_tolerance"]),
        abs(forward_error - float(analytic_forward_error))
        <= float(oracle["linear_forward_tolerance"]),
        amplification >= float(oracle["minimum_amplification"]),
        relative_residual <= float(oracle["maximum_residual"]),
        [small_close, large_close, bad_small, bad_large]
        == expected["tolerance_flags"],
    ]
    if not all(checks):
        raise SystemExit("numerical-stability oracle or tolerance policy failed")

    print(
        "oracle=passed "
        f"cancel=({naive_cancellation:.3e},{stable_cancellation:.3e},"
        f"{float(exact_cancellation):.3e}) "
        f"sums=({naive_sum:.1f},{compensated_sum:.1f},{float(decimal_sum):.1f}) "
        f"linear=({condition_number:.3e},{forward_error:.3e},"
        f"{amplification:.3e},{relative_residual:.3e}) "
        f"tolerance=({int(small_close)},{int(large_close)},"
        f"{int(bad_small)},{int(bad_large)})"
    )
    return 0


oracle_path = Path("evidence/ch14/oracle.json")
if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
    oracle_path = Path(sys.argv[1])
main(oracle_path)
