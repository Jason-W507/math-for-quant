# %% [markdown]
# # Monte Carlo、bootstrap 与方差缩减
#
# 解析积分、解析方差和可枚举小样本提供独立 oracle；伪随机模拟只是被测实现。

# %%
from __future__ import annotations

from fractions import Fraction
from itertools import product
import json
import math
from pathlib import Path
import sys

import numpy as np


def bootstrap_oracle(values: list[int]) -> tuple[Fraction, Fraction, int]:
    sample_size = len(values)
    means = [
        sum(Fraction(value) for value in resample) / sample_size
        for resample in product(values, repeat=sample_size)
    ]
    mean = sum(means, Fraction(0)) / len(means)
    variance = sum((value - mean) ** 2 for value in means) / len(means)
    return mean, variance, len(means)


def main(oracle_path: Path) -> int:
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    if not oracle["bootstrap_observations_iid"]:
        raise SystemExit(
            "iid bootstrap invalid: observations are dependent; "
            "use a dependence-aware resampling scheme"
        )
    multiplier = float(oracle["confidence_multiplier"])
    if multiplier < 3.0:
        raise SystemExit("invalid Monte Carlo tolerance: use at least three standard errors")

    expected = oracle["expected"]
    absolute_tolerance = float(oracle["absolute_tolerance"])
    analytic_integral = Fraction(1, 3)
    analytic_integrand_variance = Fraction(4, 45)
    analytic_control_beta = Fraction(1, 1)
    analytic_control_variance = Fraction(1, 180)
    analytic_control_vrf = Fraction(16, 1)
    rng = np.random.default_rng(int(oracle["seed"]))
    sample = rng.random(int(oracle["sample_size"]))
    values = sample**2
    estimate = float(values.mean())
    standard_error = math.sqrt(float(analytic_integrand_variance) / values.size)
    monte_carlo_passed = (
        abs(estimate - float(analytic_integral)) <= multiplier * standard_error
    )

    adjusted = values - (sample - float(oracle["control_known_mean"]))
    adjusted_standard_error = math.sqrt(
        float(analytic_control_variance) / adjusted.size
    )
    adjusted_mean_passed = (
        abs(float(adjusted.mean()) - float(analytic_integral))
        <= multiplier * adjusted_standard_error
    )
    if not adjusted_mean_passed:
        raise SystemExit("control variate invalid: adjusted mean changed target")
    sample_vrf = float(values.var(ddof=1) / adjusted.var(ddof=1))

    rate_rng = np.random.default_rng(int(oracle["rate_seed"]))
    repetitions = int(oracle["rate_repetitions"])
    small_n = int(oracle["rate_small_n"])
    large_n = int(oracle["rate_large_n"])
    small_means = (rate_rng.random((repetitions, small_n)) ** 2).mean(axis=1)
    large_means = (rate_rng.random((repetitions, large_n)) ** 2).mean(axis=1)
    target = float(analytic_integral)
    small_rmse = float(np.sqrt(np.mean((small_means - target) ** 2)))
    large_rmse = float(np.sqrt(np.mean((large_means - target) ** 2)))
    rate_ratio = small_rmse / large_rmse

    bootstrap_mean, bootstrap_variance, bootstrap_states = bootstrap_oracle(
        [int(value) for value in oracle["bootstrap_sample"]]
    )
    exact_bootstrap_mean = Fraction(7, 3)
    exact_bootstrap_variance = Fraction(14, 27)

    checks = [
        monte_carlo_passed,
        abs(float(expected["integral"]) - float(analytic_integral))
        <= absolute_tolerance,
        abs(float(expected["integrand_variance"]) - float(analytic_integrand_variance))
        <= absolute_tolerance,
        float(oracle["rate_ratio_min"]) <= rate_ratio <= float(oracle["rate_ratio_max"]),
        abs(float(expected["control_beta"]) - float(analytic_control_beta))
        <= absolute_tolerance,
        abs(float(expected["control_variance_reduction"]) - float(analytic_control_vrf))
        <= absolute_tolerance,
        adjusted_mean_passed,
        sample_vrf >= float(oracle["minimum_sample_variance_reduction"]),
        bootstrap_mean == exact_bootstrap_mean,
        bootstrap_variance == exact_bootstrap_variance,
        bootstrap_states == int(expected["bootstrap_states"]),
        abs(float(expected["bootstrap_mean"]) - float(exact_bootstrap_mean))
        <= absolute_tolerance,
        abs(float(expected["bootstrap_variance"]) - float(exact_bootstrap_variance))
        <= absolute_tolerance,
    ]
    if not all(checks):
        raise SystemExit("Monte Carlo, variance-reduction, or bootstrap oracle failed")

    print(
        "oracle=passed "
        "mc=(error<=4se,rate=passed) "
        "control=(beta=1.0,mean_error<=4se,theory_vrf=16.0,sample_vrf>=10) "
        f"bootstrap=(mean={float(bootstrap_mean):.6f},"
        f"var={float(bootstrap_variance):.6f},states={bootstrap_states})"
    )
    return 0


oracle_path = Path("evidence/ch15/oracle.json")
if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
    oracle_path = Path(sys.argv[1])
main(oracle_path)
