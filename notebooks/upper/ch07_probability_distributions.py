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


FIXED_VALUES = [-1.0, 0.0, 2.0]
FIXED_PROBABILITIES = [0.25, 0.5, 0.25]
FIXED_JOINT = [[0.375, 0.125], [0.125, 0.375]]
FIXED_CDF_POINTS = [-1.0, 0.0, 1.0, 2.0]
FIXED_CUTOFFS = [10.0, 1000.0]
FIXED_TOLERANCE = 1e-10


def finite_array(value: object, name: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SystemExit(f"oracle {name} must be a numeric array") from exc
    if array.shape != shape:
        raise SystemExit(f"oracle {name} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise SystemExit("oracle numeric inputs must be finite")
    return array


def finite_scalar(value: object, name: str) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SystemExit(f"oracle {name} must be numeric") from exc
    if not math.isfinite(scalar):
        raise SystemExit("oracle numeric inputs must be finite")
    return scalar


def main(oracle_path: Path = Path("evidence/ch07/oracle.json")) -> int:
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    required = {
        "finite_values",
        "finite_probabilities",
        "expected_mean",
        "expected_variance",
        "cdf_points",
        "expected_cdf",
        "expected_squared_values",
        "expected_squared_probabilities",
        "joint_distribution",
        "joint_support",
        "expected_marginals",
        "expected_dependence_gap",
        "expected_covariance",
        "expected_correlation",
        "bernoulli_p",
        "expected_bernoulli_mean",
        "expected_bernoulli_variance",
        "expected_bernoulli_mgf_log2",
        "poisson_lambda",
        "poisson_pgf_argument",
        "expected_poisson_pgf",
        "normal_mgf_argument",
        "expected_normal_mgf",
        "expected_normal_cdf_at_one",
        "uniform_bounds",
        "expected_uniform_mean",
        "expected_uniform_variance",
        "exponential_rate",
        "expected_exponential_mean",
        "expected_exponential_variance",
        "expected_exponential_survival_at_one",
        "lognormal_parameters",
        "expected_lognormal_mean",
        "expected_lognormal_variance",
        "pareto_cutoffs",
        "expected_pareto_truncated_means",
        "expected",
        "absolute_tolerance",
    }
    missing = sorted(required - oracle.keys())
    if missing:
        raise SystemExit(f"oracle missing required fields: {', '.join(missing)}")

    values = finite_array(oracle["finite_values"], "finite_values", (3,))
    probabilities = finite_array(
        oracle["finite_probabilities"], "finite_probabilities", (3,)
    )
    joint = finite_array(oracle["joint_distribution"], "joint_distribution", (2, 2))
    joint_support = finite_array(oracle["joint_support"], "joint_support", (2,))
    cdf_points = finite_array(oracle["cdf_points"], "cdf_points", (4,))
    cutoffs = finite_array(oracle["pareto_cutoffs"], "pareto_cutoffs", (2,))
    uniform_bounds = finite_array(oracle["uniform_bounds"], "uniform_bounds", (2,))
    lognormal_parameters = finite_array(
        oracle["lognormal_parameters"], "lognormal_parameters", (2,)
    )
    tolerance = finite_scalar(oracle["absolute_tolerance"], "absolute_tolerance")
    published_expected = finite_scalar(oracle["expected"], "expected")
    if np.any(probabilities < 0.0) or np.any(joint < 0.0):
        raise SystemExit("probability mass must be nonnegative")
    if (
        values.tolist() != FIXED_VALUES
        or probabilities.tolist() != FIXED_PROBABILITIES
        or joint.tolist() != FIXED_JOINT
        or joint_support.tolist() != [-1.0, 1.0]
        or cdf_points.tolist() != FIXED_CDF_POINTS
        or cutoffs.tolist() != FIXED_CUTOFFS
        or uniform_bounds.tolist() != [-1.0, 3.0]
        or lognormal_parameters.tolist() != [0.0, 0.5]
        or finite_scalar(oracle["bernoulli_p"], "bernoulli_p") != 0.25
        or finite_scalar(oracle["poisson_lambda"], "poisson_lambda") != 2.0
        or finite_scalar(oracle["poisson_pgf_argument"], "poisson_pgf_argument") != 0.5
        or finite_scalar(oracle["normal_mgf_argument"], "normal_mgf_argument") != 0.5
        or finite_scalar(oracle["exponential_rate"], "exponential_rate") != 2.0
    ):
        raise SystemExit("oracle must use the fixed distribution ledger")
    if tolerance != FIXED_TOLERANCE:
        raise SystemExit("oracle numeric tolerance is fixed")
    if published_expected != 0.25:
        raise SystemExit("published expected must equal 1/4")
    if abs(float(probabilities.sum()) - 1.0) > tolerance:
        raise SystemExit("finite probability mass does not sum to one")
    if abs(float(joint.sum()) - 1.0) > tolerance:
        raise SystemExit("joint probability mass does not sum to one")

    expected_mean = finite_scalar(oracle["expected_mean"], "expected_mean")
    expected_variance = finite_scalar(oracle["expected_variance"], "expected_variance")
    expected_cdf = finite_array(oracle["expected_cdf"], "expected_cdf", (4,))
    expected_squared_values = finite_array(
        oracle["expected_squared_values"], "expected_squared_values", (3,)
    )
    expected_squared_probabilities = finite_array(
        oracle["expected_squared_probabilities"],
        "expected_squared_probabilities",
        (3,),
    )
    expected_marginals = finite_array(
        oracle["expected_marginals"], "expected_marginals", (2,)
    )
    expected_pareto = finite_array(
        oracle["expected_pareto_truncated_means"],
        "expected_pareto_truncated_means",
        (2,),
    )

    scalar_names = [
        "expected_dependence_gap",
        "expected_covariance",
        "expected_correlation",
        "expected_bernoulli_mean",
        "expected_bernoulli_variance",
        "expected_bernoulli_mgf_log2",
        "expected_poisson_pgf",
        "expected_normal_mgf",
        "expected_normal_cdf_at_one",
        "expected_uniform_mean",
        "expected_uniform_variance",
        "expected_exponential_mean",
        "expected_exponential_variance",
        "expected_exponential_survival_at_one",
        "expected_lognormal_mean",
        "expected_lognormal_variance",
    ]
    expected_scalars = {name: finite_scalar(oracle[name], name) for name in scalar_names}

    mean = float(probabilities @ values)
    variance = float(probabilities @ ((values - mean) ** 2))
    cdf = np.asarray(
        [float(probabilities[values <= point].sum()) for point in cdf_points]
    )
    squared_values, inverse = np.unique(values**2, return_inverse=True)
    squared_probabilities = np.zeros(squared_values.shape)
    np.add.at(squared_probabilities, inverse, probabilities)

    row_marginal = joint.sum(axis=1)
    column_marginal = joint.sum(axis=0)
    independent_table = np.outer(row_marginal, column_marginal)
    dependence_gap = float(np.max(np.abs(joint - independent_table)))
    x_mean = float(row_marginal @ joint_support)
    y_mean = float(column_marginal @ joint_support)
    cross_moment = float(joint_support @ joint @ joint_support)
    covariance = cross_moment - x_mean * y_mean
    x_variance = float(row_marginal @ ((joint_support - x_mean) ** 2))
    y_variance = float(column_marginal @ ((joint_support - y_mean) ** 2))
    correlation = covariance / math.sqrt(x_variance * y_variance)

    p = 0.25
    bernoulli_mean = p
    bernoulli_variance = p * (1.0 - p)
    bernoulli_mgf = (1.0 - p) + p * 2.0
    poisson_pgf = math.exp(2.0 * (0.5 - 1.0))
    normal_mgf = math.exp(0.5 * 0.5**2)
    normal_cdf_at_one = 0.5 * (1.0 + math.erf(1.0 / math.sqrt(2.0)))
    lower, upper = uniform_bounds
    uniform_mean = (lower + upper) / 2.0
    uniform_variance = (upper - lower) ** 2 / 12.0
    exponential_mean = 0.5
    exponential_variance = 0.25
    exponential_survival = math.exp(-2.0)
    lognormal_mu, lognormal_sigma = lognormal_parameters
    lognormal_mean = math.exp(lognormal_mu + 0.5 * lognormal_sigma**2)
    lognormal_variance = (
        math.exp(lognormal_sigma**2) - 1.0
    ) * math.exp(2.0 * lognormal_mu + lognormal_sigma**2)
    pareto_truncated = np.log(cutoffs)

    observed_scalars = {
        "expected_dependence_gap": dependence_gap,
        "expected_covariance": covariance,
        "expected_correlation": correlation,
        "expected_bernoulli_mean": bernoulli_mean,
        "expected_bernoulli_variance": bernoulli_variance,
        "expected_bernoulli_mgf_log2": bernoulli_mgf,
        "expected_poisson_pgf": poisson_pgf,
        "expected_normal_mgf": normal_mgf,
        "expected_normal_cdf_at_one": normal_cdf_at_one,
        "expected_uniform_mean": uniform_mean,
        "expected_uniform_variance": uniform_variance,
        "expected_exponential_mean": exponential_mean,
        "expected_exponential_variance": exponential_variance,
        "expected_exponential_survival_at_one": exponential_survival,
        "expected_lognormal_mean": lognormal_mean,
        "expected_lognormal_variance": lognormal_variance,
    }
    if abs(mean - expected_mean) > tolerance or abs(variance - expected_variance) > tolerance:
        raise SystemExit("finite-distribution moment ledger mismatch")
    array_checks = [
        (cdf, expected_cdf),
        (squared_values, expected_squared_values),
        (squared_probabilities, expected_squared_probabilities),
        (row_marginal, expected_marginals),
        (column_marginal, expected_marginals),
        (pareto_truncated, expected_pareto),
    ]
    if any(float(np.max(np.abs(a - b))) > tolerance for a, b in array_checks):
        raise SystemExit("distribution transform or marginal ledger mismatch")
    if any(
        abs(observed_scalars[name] - expected_scalars[name]) > tolerance
        for name in scalar_names
    ):
        raise SystemExit("analytic distribution oracle mismatch")

    print(
        "oracle=passed "
        f"mean={mean:.6f} variance={variance:.6f} "
        f"cdf=({cdf[0]:.2f},{cdf[1]:.2f},{cdf[2]:.2f},{cdf[3]:.2f}) "
        f"square=({squared_probabilities[0]:.2f},{squared_probabilities[1]:.2f},{squared_probabilities[2]:.2f}) "
        f"dependence={dependence_gap:.3f} covariance={covariance:.3f} "
        f"bernoulli=({bernoulli_mean:.3f},{bernoulli_variance:.4f},{bernoulli_mgf:.3f}) "
        f"poisson={poisson_pgf:.6f} normal=({normal_mgf:.6f},{normal_cdf_at_one:.6f}) "
        f"uniform=({uniform_mean:.3f},{uniform_variance:.6f}) "
        f"exponential=({exponential_mean:.3f},{exponential_survival:.6f}) "
        f"lognormal=({lognormal_mean:.6f},{lognormal_variance:.6f}) "
        f"pareto=({pareto_truncated[0]:.6f},{pareto_truncated[1]:.6f})"
    )
    return 0


oracle_path = Path("evidence/ch07/oracle.json")
if Path(sys.argv[0]).stem == "ch07_probability_distributions" and len(sys.argv) > 1:
    oracle_path = Path(sys.argv[1])
main(oracle_path)
