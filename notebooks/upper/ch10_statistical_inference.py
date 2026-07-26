# %% [markdown]
# # 估计、异方差标准误与多重检验
#
# 解析矩、固定设计真方差和独立检验族的精确 FWER 均由 oracle 给出；
# 固定种子模拟只检验声明的名义性质。

# %%
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def main(oracle_path: Path = Path("evidence/ch10/oracle.json")) -> int:
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    rng = np.random.default_rng(int(oracle["seed"]))

    p = float(oracle["bernoulli_p"])
    n = int(oracle["sample_size"])
    repetitions = int(oracle["estimator_repetitions"])
    means = rng.binomial(n, p, size=repetitions) / n
    theory_variance = p * (1.0 - p) / n
    empirical_mean = float(means.mean())
    empirical_variance = float(means.var(ddof=1))
    theory_se = math.sqrt(theory_variance)
    clt_coverage = float(np.mean(np.abs((means - p) / theory_se) <= 1.96))

    regression_n = int(oracle["regression_sample_size"])
    regression_repetitions = int(oracle["regression_repetitions"])
    x = np.linspace(-1.0, 1.0, regression_n)
    centered_x = x - x.mean()
    sxx = float(centered_x @ centered_x)
    sigma = 0.5 + 2.0 * np.abs(centered_x)
    errors = rng.normal(size=(regression_repetitions, regression_n)) * sigma
    true_slope = float(oracle["true_slope"])
    slopes = true_slope + (errors @ centered_x) / sxx
    intercept_errors = errors.mean(axis=1)
    residuals = (
        errors
        - intercept_errors[:, None]
        - (slopes - true_slope)[:, None] * centered_x
    )
    true_slope_se = math.sqrt(float((centered_x**2) @ (sigma**2))) / sxx
    empirical_slope_se = float(slopes.std(ddof=1))
    naive_se = np.sqrt(np.sum(residuals**2, axis=1) / (regression_n - 2) / sxx)
    robust_se = np.sqrt(
        (regression_n / (regression_n - 2))
        * np.sum(residuals**2 * centered_x**2, axis=1)
        / sxx**2
    )
    naive_coverage = float(
        np.mean(np.abs(slopes - true_slope) <= 1.96 * naive_se)
    )
    robust_coverage = float(
        np.mean(np.abs(slopes - true_slope) <= 1.96 * robust_se)
    )

    alpha = float(oracle["nominal_alpha"])
    hypothesis_count = int(oracle["hypothesis_count"])
    testing_repetitions = int(oracle["multiple_testing_repetitions"])
    z_scores = rng.normal(size=(testing_repetitions, hypothesis_count))
    naive_fwer = float(
        np.mean(
            np.any(
                np.abs(z_scores) > float(oracle["normal_two_sided_critical"]),
                axis=1,
            )
        )
    )
    bonferroni_fwer = float(
        np.mean(
            np.any(
                np.abs(z_scores)
                > float(oracle["bonferroni_two_sided_critical"]),
                axis=1,
            )
        )
    )
    exact_naive_fwer = 1.0 - (1.0 - alpha) ** hypothesis_count
    exact_bonferroni_fwer = 1.0 - (1.0 - alpha / hypothesis_count) ** hypothesis_count

    simulation_tolerance = float(oracle["simulation_tolerance"])
    checks = [
        abs(theory_variance - float(oracle["expected_variance"]))
        <= float(oracle["absolute_tolerance"]),
        abs(empirical_mean - float(oracle["expected_simulated_mean"]))
        <= simulation_tolerance,
        abs(empirical_variance - float(oracle["expected_simulated_variance"]))
        <= simulation_tolerance,
        abs(clt_coverage - float(oracle["expected_clt_coverage"]))
        <= simulation_tolerance,
        abs(float(slopes.mean()) - float(oracle["expected_simulated_slope"]))
        <= simulation_tolerance,
        abs(true_slope_se - float(oracle["expected_true_slope_se"]))
        <= simulation_tolerance,
        abs(empirical_slope_se - float(oracle["expected_empirical_slope_se"]))
        <= simulation_tolerance,
        abs(naive_coverage - float(oracle["expected_naive_coverage"]))
        <= simulation_tolerance,
        abs(robust_coverage - float(oracle["expected_robust_coverage"]))
        <= simulation_tolerance,
        abs(exact_naive_fwer - float(oracle["expected_exact_naive_fwer"]))
        <= float(oracle["absolute_tolerance"]),
        abs(
            exact_bonferroni_fwer
            - float(oracle["expected_exact_bonferroni_fwer"])
        )
        <= float(oracle["absolute_tolerance"]),
        abs(naive_fwer - float(oracle["expected_simulated_naive_fwer"]))
        <= simulation_tolerance,
        abs(
            bonferroni_fwer
            - float(oracle["expected_simulated_bonferroni_fwer"])
        )
        <= simulation_tolerance,
        abs(empirical_mean - p) <= float(oracle["mean_tolerance"]),
        abs(empirical_variance - theory_variance)
        <= float(oracle["variance_tolerance"]),
        abs(clt_coverage - (1.0 - alpha)) <= float(oracle["coverage_tolerance"]),
        abs(float(slopes.mean()) - true_slope) <= float(oracle["slope_tolerance"]),
        abs(empirical_slope_se - true_slope_se)
        <= float(oracle["standard_error_tolerance"]),
        naive_coverage < 0.90,
        abs(robust_coverage - (1.0 - alpha))
        <= float(oracle["coverage_tolerance"]),
        abs(naive_fwer - exact_naive_fwer) <= float(oracle["fwer_tolerance"]),
        abs(bonferroni_fwer - exact_bonferroni_fwer)
        <= float(oracle["fwer_tolerance"]),
    ]
    if not all(checks):
        raise SystemExit("statistical-inference oracle or declared tolerance failed")

    print(
        "oracle=passed "
        f"mean={empirical_mean:.6f} theory_var={theory_variance:.6f} "
        f"empirical_var={empirical_variance:.6f} clt_coverage={clt_coverage:.6f} "
        f"slope={slopes.mean():.6f} true_se={true_slope_se:.6f} "
        f"empirical_se={empirical_slope_se:.6f} "
        f"naive_coverage={naive_coverage:.6f} "
        f"robust_coverage={robust_coverage:.6f} "
        f"naive_fwer={naive_fwer:.6f} "
        f"bonferroni_fwer={bonferroni_fwer:.6f}"
    )
    return 0


main()
