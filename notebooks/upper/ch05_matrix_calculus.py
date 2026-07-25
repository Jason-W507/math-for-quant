# %% [markdown]
# # 二次型、Jacobian 链式法则与不可微反例
#
# 解析梯度先由微分形式手算；中心差分只负责独立交叉核验。

# %%
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable


Vector = list[float]


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
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
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

    expected = [
        float(oracle["expected_quadratic_value"]),
        *(float(item) for item in oracle["expected_quadratic_gradient"]),
        *(float(item) for item in oracle["expected_chain_gradient"]),
        float(oracle["expected_abs_left_slope"]),
        float(oracle["expected_abs_right_slope"]),
    ]
    observed = [
        quadratic(point),
        *analytic_quadratic,
        *analytic_chain,
        abs_left,
        abs_right,
    ]
    tolerance = float(oracle["absolute_tolerance"])
    if any(abs(left - right) > tolerance for left, right in zip(observed, expected)):
        raise SystemExit(f"oracle mismatch: observed={observed} expected={expected}")
    if symmetric != oracle["expected_symmetric_hessian"]:
        raise SystemExit(f"hessian mismatch: observed={symmetric}")
    if max_error > float(oracle["maximum_gradient_error"]):
        raise SystemExit(f"finite-difference mismatch: max_error={max_error}")

    print(
        "oracle=passed value=-1.000000 gradient=(0.000000,-0.500000) "
        "chain=(4.000000,-6.500000) "
        f"max_error={max_error:.3e} left={abs_left:.1f} right={abs_right:.1f}"
    )
    return 0


main()
