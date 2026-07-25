# %% [markdown]
# # 概率空间、分布变换与重尾反例
#
# 所有目标值先由有限求和或解析积分给出；程序只负责复算和交叉核验。

# %%
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np


def main(oracle_path: Path = Path("evidence/ch07/oracle.json")) -> int:
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    tolerance = float(oracle["absolute_tolerance"])

    values = np.asarray(oracle["finite_values"], dtype=float)
    probabilities = np.asarray(oracle["finite_probabilities"], dtype=float)
    if (
        values.ndim != 1
        or probabilities.ndim != 1
        or values.size == 0
        or values.shape != probabilities.shape
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(probabilities))
    ):
        raise SystemExit("finite distribution requires matching finite one-dimensional arrays")
    if np.any(probabilities < 0.0):
        raise SystemExit("finite probability mass must be nonnegative")
    if abs(float(probabilities.sum()) - 1.0) > tolerance:
        raise SystemExit("finite probability mass does not sum to one")
    mean = float(probabilities @ values)
    variance = float(probabilities @ ((values - mean) ** 2))

    joint = np.asarray(oracle["joint_distribution"], dtype=float)
    if joint.ndim != 2 or joint.size == 0 or not np.all(np.isfinite(joint)):
        raise SystemExit("joint distribution must be a finite nonempty matrix")
    if np.any(joint < 0.0):
        raise SystemExit("joint probability mass must be nonnegative")
    joint_mass = float(joint.sum())
    if abs(joint_mass - 1.0) > tolerance:
        raise SystemExit("joint probability mass does not sum to one")
    row_marginal = joint.sum(axis=1)
    column_marginal = joint.sum(axis=0)
    expected_marginals = np.asarray(oracle["expected_marginals"], dtype=float)
    if (
        expected_marginals.shape != row_marginal.shape
        or expected_marginals.shape != column_marginal.shape
    ):
        raise SystemExit("expected marginal shape is incompatible with joint distribution")

    p = float(oracle["bernoulli_p"])
    bernoulli_mean = p
    bernoulli_variance = p * (1.0 - p)
    bernoulli_mgf = (1.0 - p) + p * math.exp(math.log(2.0))

    poisson_lambda = float(oracle["poisson_lambda"])
    poisson_argument = float(oracle["poisson_pgf_argument"])
    poisson_pgf = math.exp(poisson_lambda * (poisson_argument - 1.0))

    normal_argument = float(oracle["normal_mgf_argument"])
    normal_mgf = math.exp(0.5 * normal_argument**2)

    cutoffs = np.asarray(oracle["pareto_cutoffs"], dtype=float)
    pareto_truncated = np.log(cutoffs)

    scalar_checks = [
        (mean, oracle["expected_mean"]),
        (variance, oracle["expected_variance"]),
        (joint_mass, 1.0),
        (bernoulli_mean, oracle["expected_bernoulli_mean"]),
        (bernoulli_variance, oracle["expected_bernoulli_variance"]),
        (bernoulli_mgf, oracle["expected_bernoulli_mgf_log2"]),
        (poisson_pgf, oracle["expected_poisson_pgf"]),
        (normal_mgf, oracle["expected_normal_mgf"]),
    ]
    if any(abs(float(observed) - float(expected)) > tolerance for observed, expected in scalar_checks):
        raise SystemExit("analytic distribution oracle mismatch")
    if not np.allclose(row_marginal, oracle["expected_marginals"], atol=tolerance, rtol=0.0):
        raise SystemExit("row marginal mismatch")
    if not np.allclose(column_marginal, oracle["expected_marginals"], atol=tolerance, rtol=0.0):
        raise SystemExit("column marginal mismatch")
    if not np.allclose(
        pareto_truncated,
        oracle["expected_pareto_truncated_means"],
        atol=tolerance,
        rtol=0.0,
    ):
        raise SystemExit("Pareto truncated-mean oracle mismatch")

    print(
        "oracle=passed "
        f"mean={mean:.6f} variance={variance:.6f} "
        f"joint_mass={joint_mass:.6f} "
        f"marginals=({row_marginal[0]:.6f},{column_marginal[0]:.6f}) "
        f"bernoulli=({bernoulli_mean:.6f},{bernoulli_variance:.6f},{bernoulli_mgf:.6f}) "
        f"poisson_pgf={poisson_pgf:.6f} normal_mgf={normal_mgf:.6f} "
        f"pareto_truncated=({pareto_truncated[0]:.6f},{pareto_truncated[1]:.6f})"
    )
    return 0


oracle_path = Path("evidence/ch07/oracle.json")
if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
    oracle_path = Path(sys.argv[1])
main(oracle_path)
