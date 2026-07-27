# %% [markdown]
# # 凸优化、KKT 与对偶
#
# 手算 KKT、对偶、灵敏度和最小方差组合是独立 oracle；投影梯度只负责
# 数值交叉验证。非凸驻点和约束资格失败问题用于展示 KKT 边界。

# %%
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np


FIXED_NUMPY_VERSION = "2.5.1"
FIXED_LABELS = ["KKT", "duality", "sensitivity", "minimum variance"]
FIXED_PROVENANCE = (
    "KKT stationarity, complementary slackness, the scalar dual and right-hand-"
    "side sensitivity are solved by hand for a two-variable quadratic program; "
    "the minimum-variance portfolio follows from the equality-constrained KKT "
    "system; projected gradient is only an independent numerical cross-check"
)
FIXED_MARKERS = [
    "手算最优解为 $(0.5,0.5)$",
    "最优乘子 $\\lambda^*=0.5$",
    "原问题与对偶目标均为 $0.25$",
    "数值解为 $(0.500000,0.500000)$",
    "右端项灵敏度为 $0.600000$",
    "最小方差权重为 $(0.711864,0.288136)$",
    "驻点 $x=0$ 的目标值为 1",
    "KKT 驻点残差恒为 1",
]
FIXED_ARRAYS = {
    "analytic_solution": [0.5, 0.5],
    "optimizer_start": [2.0, 0.0],
    "portfolio_covariance": [[0.04, 0.006], [0.006, 0.09]],
    "portfolio_covariance_perturbed": [[0.04, 0.02], [0.02, 0.09]],
    "portfolio_budget_vector": [1.0, 1.0],
}
FIXED_TOLERANCES = {
    "absolute_tolerance": 1e-12,
    "optimizer_tolerance": 1e-10,
    "sensitivity_tolerance": 1e-9,
}
FIXED_INTEGERS = {"optimizer_iterations": 500}
FIXED_SCALARS = {
    "analytic_multiplier": 0.5,
    "expected_primal_value": 0.25,
    "expected_dual_value": 0.25,
    "expected_duality_gap": 0.0,
    "optimizer_step": 0.2,
    "expected_cq_residual": 1.0,
    "sensitivity_rhs": 1.2,
    "finite_difference_step": 1e-5,
    "expected": 0.25,
}
REQUIRED_FIELDS = {
    "absolute_tolerance", "analytic_multiplier", "analytic_solution",
    "expected", "expected_cq_residual", "expected_dual_value", "expected_duality_gap",
    "expected_minimum_variance_weights", "expected_portfolio_condition_number",
    "expected_portfolio_variance", "expected_primal_value",
    "expected_perturbed_condition_number", "expected_perturbed_variance",
    "expected_perturbed_weights", "expected_weight_amplification",
    "expected_sensitivity_derivative", "expected_sensitivity_multiplier",
    "expected_sensitivity_value", "finite_difference_step", "numpy_version",
    "optimizer_iterations", "optimizer_start", "optimizer_step",
    "optimizer_tolerance", "portfolio_budget_vector", "portfolio_covariance",
    "portfolio_covariance_perturbed",
    "provenance", "published_markers", "sensitivity_rhs",
    "sensitivity_tolerance", "optimization_labels",
}


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
        for item in value.values():
            reject_nonfinite(item)


def validate_oracle(oracle: dict[str, object]) -> None:
    missing = sorted(REQUIRED_FIELDS - oracle.keys())
    if missing:
        raise SystemExit(f"oracle missing required fields: {', '.join(missing)}")
    reject_nonfinite(oracle)
    if (
        oracle["numpy_version"] != FIXED_NUMPY_VERSION
        or np.__version__ != FIXED_NUMPY_VERSION
    ):
        raise SystemExit(f"NumPy version must equal {FIXED_NUMPY_VERSION}")
    if oracle["optimization_labels"] != FIXED_LABELS:
        raise SystemExit("optimization labels must match the published design")
    if oracle["provenance"] != FIXED_PROVENANCE:
        raise SystemExit("oracle provenance must match the published design")
    if oracle["published_markers"] != FIXED_MARKERS:
        raise SystemExit("published markers must match the chapter evidence")
    if any(oracle[name] != value for name, value in FIXED_ARRAYS.items()):
        raise SystemExit("canonical array design must not change")
    if any(oracle[name] != value for name, value in FIXED_TOLERANCES.items()):
        raise SystemExit("oracle tolerances must match the published design")
    if any(oracle[name] != value for name, value in FIXED_SCALARS.items()):
        raise SystemExit("canonical scalar design must not change")
    for name, expected in FIXED_INTEGERS.items():
        value = oracle[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise SystemExit(f"oracle {name} must be an integer")
        if value != expected:
            raise SystemExit(f"oracle {name} must match the published design")


def project_feasible(point: np.ndarray) -> np.ndarray:
    nonnegative = np.maximum(point, 0.0)
    if float(nonnegative.sum()) >= 1.0:
        return nonnegative
    first = float(np.clip((point[0] - point[1] + 1.0) / 2.0, 0.0, 1.0))
    return np.array([first, 1.0 - first])


def kkt_certificate(
    candidate: np.ndarray, multiplier: float
) -> tuple[float, float, float, float, float, float, float]:
    primal_value = float(0.5 * candidate @ candidate)
    dual_value = float(multiplier - multiplier**2)
    duality_gap = primal_value - dual_value
    constraint = 1.0 - float(candidate.sum())
    stationarity = float(
        np.max(np.abs(candidate - np.array([multiplier, multiplier])))
    )
    primal_residual = max(0.0, constraint, -candidate[0], -candidate[1])
    dual_residual = max(0.0, -multiplier)
    complementarity = abs(multiplier * constraint)
    return (
        primal_value,
        dual_value,
        duality_gap,
        stationarity,
        primal_residual,
        dual_residual,
        complementarity,
    )


def minimum_variance_portfolio(
    covariance: np.ndarray, budget: np.ndarray
) -> tuple[np.ndarray, float, float]:
    inverse_budget = np.linalg.solve(covariance, budget)
    weights = inverse_budget / float(budget @ inverse_budget)
    variance = float(weights @ covariance @ weights)
    condition = float(np.linalg.cond(covariance))
    return weights, variance, condition


def main(oracle_path: Path = Path("evidence/ch13/oracle.json")) -> int:
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    validate_oracle(oracle)
    analytic = np.asarray(oracle["analytic_solution"], dtype=float)
    multiplier = float(oracle["analytic_multiplier"])

    point = np.asarray(oracle["optimizer_start"], dtype=float)
    step = float(oracle["optimizer_step"])
    for _ in range(int(oracle["optimizer_iterations"])):
        point = project_feasible(point - step * point)

    (
        primal_value,
        dual_value,
        duality_gap,
        stationarity,
        primal_residual,
        dual_residual,
        complementarity,
    ) = kkt_certificate(point, multiplier)

    rhs = float(oracle["sensitivity_rhs"])
    sensitivity_solution = np.array([rhs / 2.0, rhs / 2.0])
    sensitivity_value = float(0.5 * sensitivity_solution @ sensitivity_solution)
    sensitivity_multiplier = rhs / 2.0
    delta = float(oracle["finite_difference_step"])
    value_plus = (rhs + delta) ** 2 / 4.0
    value_minus = (rhs - delta) ** 2 / 4.0
    sensitivity_derivative = (value_plus - value_minus) / (2.0 * delta)
    sensitivity_tolerance = float(oracle["sensitivity_tolerance"])
    if (
        abs(
            sensitivity_multiplier
            - float(oracle["expected_sensitivity_multiplier"])
        )
        > sensitivity_tolerance
        or abs(
            sensitivity_value - float(oracle["expected_sensitivity_value"])
        )
        > sensitivity_tolerance
        or abs(
            sensitivity_derivative
            - float(oracle["expected_sensitivity_derivative"])
        )
        > sensitivity_tolerance
    ):
        raise SystemExit("sensitivity ledger failed")

    covariance = np.asarray(oracle["portfolio_covariance"], dtype=float)
    perturbed_covariance = np.asarray(
        oracle["portfolio_covariance_perturbed"], dtype=float
    )
    budget = np.asarray(oracle["portfolio_budget_vector"], dtype=float)
    portfolio_weights, portfolio_variance, portfolio_condition = (
        minimum_variance_portfolio(covariance, budget)
    )
    perturbed_weights, perturbed_variance, perturbed_condition = (
        minimum_variance_portfolio(perturbed_covariance, budget)
    )
    covariance_change = abs(perturbed_covariance[0, 1] - covariance[0, 1])
    weight_amplification = float(
        np.linalg.norm(perturbed_weights - portfolio_weights, ord=1)
        / covariance_change
    )
    tolerance = float(oracle["absolute_tolerance"])
    if (
        not np.allclose(
            portfolio_weights,
            oracle["expected_minimum_variance_weights"],
            atol=tolerance,
            rtol=0.0,
        )
        or abs(
            portfolio_variance - float(oracle["expected_portfolio_variance"])
        )
        > tolerance
        or abs(
            portfolio_condition
            - float(oracle["expected_portfolio_condition_number"])
        )
        > tolerance
    ):
        raise SystemExit("portfolio ledger failed")
    if (
        not np.allclose(
            perturbed_weights,
            oracle["expected_perturbed_weights"],
            atol=tolerance,
            rtol=0.0,
        )
        or abs(
            perturbed_variance - float(oracle["expected_perturbed_variance"])
        )
        > tolerance
        or abs(
            perturbed_condition
            - float(oracle["expected_perturbed_condition_number"])
        )
        > tolerance
        or abs(
            weight_amplification
            - float(oracle["expected_weight_amplification"])
        )
        > tolerance
    ):
        raise SystemExit("portfolio perturbation ledger failed")

    nonconvex_stationary_value = float((0.0**2 - 1.0) ** 2)
    nonconvex_global_value = float((1.0**2 - 1.0) ** 2)
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
        failed = [str(index) for index, passed in enumerate(checks) if not passed]
        raise SystemExit(
            "optimization oracle or declared tolerance failed: "
            + ",".join(failed)
        )

    print(
        "oracle=passed "
        f"analytic=({analytic[0]:.6f},{analytic[1]:.6f},{multiplier:.6f}) "
        f"numeric=({point[0]:.6f},{point[1]:.6f}) "
        f"values=({primal_value:.6f},{dual_value:.6f},{duality_gap:.3e}) "
        f"kkt=({stationarity:.3e},{primal_residual:.3e},"
        f"{dual_residual:.3e},{complementarity:.3e}) "
        f"sensitivity=({sensitivity_multiplier:.6f},"
        f"{sensitivity_value:.6f},{sensitivity_derivative:.6f}) "
        f"portfolio=({portfolio_weights[0]:.6f},{portfolio_weights[1]:.6f},"
        f"{portfolio_variance:.6f},{portfolio_condition:.6f}) "
        f"portfolio_perturbed=({perturbed_weights[0]:.6f},"
        f"{perturbed_weights[1]:.6f},{perturbed_variance:.6f},"
        f"{perturbed_condition:.6f},{weight_amplification:.6f}) "
        f"nonconvex=({nonconvex_stationary_value:.6f},"
        f"{nonconvex_global_value:.6f}) "
        f"cq_residual={cq_stationarity_residual:.6f}"
    )
    return 0


def audit_candidate(values: list[str]) -> int:
    oracle = json.loads(Path("evidence/ch13/oracle.json").read_text(encoding="utf-8"))
    validate_oracle(oracle)
    if len(values) != 2:
        raise SystemExit("--audit-candidate requires exactly two coordinates")
    candidate = np.asarray([float(value) for value in values], dtype=float)
    certificate = kkt_certificate(candidate, float(oracle["analytic_multiplier"]))
    tolerance = float(oracle["absolute_tolerance"])
    if any(abs(value) > tolerance for value in certificate[2:]):
        raise SystemExit("candidate KKT certificate failed")
    print("candidate KKT certificate passed")
    return 0


if len(sys.argv) > 1 and sys.argv[1] == "--audit-candidate":
    audit_candidate(sys.argv[2:])
else:
    main(
        Path(sys.argv[1])
        if len(sys.argv) > 1 and not sys.argv[1].startswith("-")
        else Path("evidence/ch13/oracle.json")
    )
