# %% [markdown]
# # 凸优化、KKT 与对偶
#
# 手算 KKT 与对偶解是独立 oracle；投影梯度只负责数值交叉验证。
# 非凸驻点和约束资格失败问题用于展示 KKT 边界。

# %%
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def project_feasible(point: np.ndarray) -> np.ndarray:
    nonnegative = np.maximum(point, 0.0)
    if float(nonnegative.sum()) >= 1.0:
        return nonnegative
    # In two dimensions the Euclidean projection onto x+y=1, x,y>=0
    # has this closed form; the clipping keeps the endpoints valid.
    first = float(np.clip((point[0] - point[1] + 1.0) / 2.0, 0.0, 1.0))
    return np.array([first, 1.0 - first])


def main(oracle_path: Path = Path("evidence/ch13/oracle.json")) -> int:
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    analytic = np.asarray(oracle["analytic_solution"], dtype=float)
    multiplier = float(oracle["analytic_multiplier"])

    point = np.asarray(oracle["optimizer_start"], dtype=float)
    step = float(oracle["optimizer_step"])
    for _ in range(int(oracle["optimizer_iterations"])):
        point = project_feasible(point - step * point)

    primal_value = float(0.5 * analytic @ analytic)
    dual_value = float(multiplier - multiplier**2)
    duality_gap = primal_value - dual_value
    constraint = 1.0 - float(analytic.sum())
    stationarity = float(
        np.max(np.abs(analytic - np.array([multiplier, multiplier])))
    )
    primal_residual = max(0.0, constraint, -analytic[0], -analytic[1])
    dual_residual = max(0.0, -multiplier)
    complementarity = abs(multiplier * constraint)

    nonconvex_stationary_value = float((0.0**2 - 1.0) ** 2)
    nonconvex_global_value = float((1.0**2 - 1.0) ** 2)
    # min x subject to x^2 <= 0 has the sole feasible optimum x=0.
    # Compute the best nonnegative multiplier from the actual gradients.
    cq_point = 0.0
    objective_gradient = 1.0
    constraint_gradient = 2.0 * cq_point
    cq_multiplier = (
        max(0.0, -objective_gradient / constraint_gradient)
        if constraint_gradient != 0.0
        else 0.0
    )
    cq_stationarity_residual = abs(
        objective_gradient + cq_multiplier * constraint_gradient
    )

    tolerance = float(oracle["absolute_tolerance"])
    checks = [
        np.max(np.abs(point - analytic)) <= float(oracle["optimizer_tolerance"]),
        abs(primal_value - float(oracle["expected_primal_value"])) <= tolerance,
        abs(dual_value - float(oracle["expected_dual_value"])) <= tolerance,
        abs(duality_gap - float(oracle["expected_duality_gap"])) <= tolerance,
        stationarity <= tolerance,
        primal_residual <= tolerance,
        dual_residual <= tolerance,
        complementarity <= tolerance,
        nonconvex_stationary_value > nonconvex_global_value,
        cq_stationarity_residual == float(oracle["expected_cq_residual"]),
    ]
    if not all(checks):
        raise SystemExit("optimization oracle or declared tolerance failed")

    print(
        "oracle=passed "
        f"analytic=({analytic[0]:.6f},{analytic[1]:.6f},{multiplier:.6f}) "
        f"numeric=({point[0]:.6f},{point[1]:.6f}) "
        f"values=({primal_value:.6f},{dual_value:.6f},{duality_gap:.3e}) "
        f"kkt=({stationarity:.3e},{primal_residual:.3e},"
        f"{dual_residual:.3e},{complementarity:.3e}) "
        f"nonconvex=({nonconvex_stationary_value:.6f},"
        f"{nonconvex_global_value:.6f}) "
        f"cq_residual={cq_stationarity_residual:.6f}"
    )
    return 0


main()
