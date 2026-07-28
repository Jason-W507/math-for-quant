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
import sys

import numpy as np


FIXED_ABSOLUTE_TOLERANCE = 1e-12
FIXED_NUMPY_VERSIONS = ("2.3.5", "2.5.1")
FIXED_PROVENANCE = (
    "analytic Bernoulli estimator moments, a fixed-design heteroskedastic "
    "linear model with known Gaussian error variances, standard-normal "
    "critical values, and exact independent-test family-wise error probabilities"
)
FIXED_ORDERED_P_VALUES = [0.001, 0.009, 0.021, 0.04, 0.2]
FIXED_INFERENCE_LABELS = [
    "bias_variance_mse",
    "unbiased_not_consistent",
    "ols_projection",
    "robust_se_not_identification",
    "fwer_fdr_not_tradability",
]
FIXED_PUBLISHED_MARKERS = [
    "MSE 是 $0.20$，小于无偏估计的 $0.25$",
    "无偏也不保证一致",
    "plug-in 一致不等于有限样本无偏",
    "理论方差为 $0.000960$",
    "异方差下朴素区间覆盖率为 $0.868900$",
    "稳健区间覆盖率为 $0.938900$",
    "AR(1) 相关系数 $0.6$",
    "HAC 与簇稳健标准误分别为 $0.559017$ 与 $0.707107$",
    "遗漏后的斜率是 $2$，偏差是 $1$",
    "Bonferroni、Holm、BH 分别拒绝 $(2,2,4)$ 项",
    "未经校正的 FWER 为 $0.639975$",
    "Bonferroni 后为 $0.047400$",
    "两个独立零效应正态估计中选择较大者，其期望为 $1/\\sqrt\\pi=0.564190$",
]
FIXED_SCALAR_DESIGN = {
    "bernoulli_p": 0.4,
    "expected": 0.4,
    "expected_mean": 0.4,
    "normal_mean_theta": 1.0,
    "unbiased_estimator_variance": 0.25,
    "shrinkage_factor": 0.8,
    "ovb_beta_x": 1.0,
    "ovb_beta_z": 2.0,
    "ovb_cov_xz": 0.5,
    "ovb_var_x": 1.0,
    "ar1_correlation": 0.6,
    "cluster_correlation": 0.4,
    "delta_method_probability": 0.4,
    "true_intercept": 0.0,
    "true_slope": 1.5,
    "nominal_alpha": 0.05,
    "normal_two_sided_critical": 1.959963984540054,
    "bonferroni_two_sided_critical": 3.023341439739154,
}
REQUIRED_FIELDS = {
    "absolute_tolerance", "ar1_correlation", "bernoulli_p",
    "bonferroni_two_sided_critical", "cluster_correlation", "cluster_size",
    "cluster_labels", "coverage_tolerance", "delta_method_probability", "delta_method_sample_size",
    "dependent_score_fixture",
    "dependent_error_sample_size", "estimator_repetitions", "expected",
    "expected_ar1_mean_variance", "expected_bh_rejections",
    "expected_bonferroni_rejections", "expected_clt_coverage",
    "expected_cluster_mean_variance", "expected_delta_variance",
    "expected_empirical_slope_se", "expected_exact_bonferroni_fwer",
    "expected_exact_naive_fwer", "expected_holm_rejections",
    "expected_cluster_standard_error", "expected_hac_standard_error",
    "expected_iid_mean_variance", "expected_mean", "expected_naive_coverage",
    "expected_omitted_bias", "expected_omitted_slope",
    "expected_optional_stopping_false_positive", "expected_plugin_bias",
    "expected_selected_null_effect",
    "expected_robust_coverage", "expected_shrinkage_bias",
    "expected_shrinkage_mse", "expected_shrinkage_variance",
    "expected_simulated_bonferroni_fwer", "expected_simulated_mean",
    "expected_simulated_naive_fwer", "expected_simulated_slope",
    "expected_simulated_variance", "expected_true_slope_se",
    "expected_unbiased_inconsistent_bias",
    "expected_unbiased_inconsistent_variance", "expected_variance",
    "fwer_tolerance", "hac_bandwidth", "hypothesis_count", "inference_labels", "mean_tolerance",
    "multiple_testing_repetitions", "nominal_alpha", "normal_mean_theta",
    "normal_two_sided_critical", "numpy_versions", "ordered_p_values",
    "ovb_beta_x", "ovb_beta_z", "ovb_cov_xz", "ovb_var_x", "provenance",
    "published_markers", "regression_repetitions", "regression_sample_size",
    "selection_candidate_count",
    "sample_size", "seed", "shrinkage_factor", "simulation_tolerance",
    "slope_tolerance", "standard_error_tolerance", "true_intercept",
    "true_slope", "unbiased_estimator_variance", "variance_tolerance",
}


def reject_nonfinite_numbers(value: object) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise SystemExit("oracle numeric inputs must be finite")
        return
    if isinstance(value, list):
        for item in value:
            reject_nonfinite_numbers(item)
    if isinstance(value, dict):
        for item in value.values():
            reject_nonfinite_numbers(item)


def require_integer(value: object, name: str, expected: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SystemExit(f"oracle {name} must be an integer")
    if value != expected:
        raise SystemExit(f"oracle {name} must equal {expected}")
    return value


def main(oracle_path: Path = Path("evidence/ch10/oracle.json")) -> int:
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_FIELDS - oracle.keys())
    if missing:
        raise SystemExit(f"oracle missing required fields: {', '.join(missing)}")
    reject_nonfinite_numbers(oracle)
    non_numeric_fields = {
        "provenance",
        "numpy_versions",
        "ordered_p_values",
        "inference_labels",
        "published_markers",
        "dependent_score_fixture",
        "cluster_labels",
    }
    for name in sorted(REQUIRED_FIELDS - non_numeric_fields):
        if isinstance(oracle[name], bool) or not isinstance(
            oracle[name], (int, float)
        ):
            raise SystemExit(f"oracle {name} must be numeric")
    if oracle["numpy_versions"] != list(FIXED_NUMPY_VERSIONS) or np.__version__ not in FIXED_NUMPY_VERSIONS:
        raise SystemExit(f"NumPy version must be one of {FIXED_NUMPY_VERSIONS}")
    if oracle["provenance"] != FIXED_PROVENANCE:
        raise SystemExit("oracle provenance must match the published design")
    if float(oracle["absolute_tolerance"]) != FIXED_ABSOLUTE_TOLERANCE:
        raise SystemExit("absolute tolerance must equal 1e-12")
    if (
        float(oracle["simulation_tolerance"]) != 1e-10
        or float(oracle["mean_tolerance"]) != 0.002
        or float(oracle["variance_tolerance"]) != 0.0001
        or float(oracle["coverage_tolerance"]) != 0.02
        or float(oracle["slope_tolerance"]) != 0.015
        or float(oracle["standard_error_tolerance"]) != 0.02
        or float(oracle["fwer_tolerance"]) != 0.015
    ):
        raise SystemExit("simulation gates must match the published design")
    seed = require_integer(oracle["seed"], "seed", 20260726)
    sample_size = require_integer(oracle["sample_size"], "sample_size", 250)
    estimator_repetitions = require_integer(
        oracle["estimator_repetitions"], "estimator_repetitions", 20000
    )
    require_integer(oracle["regression_sample_size"], "regression_sample_size", 80)
    require_integer(oracle["regression_repetitions"], "regression_repetitions", 10000)
    require_integer(oracle["hypothesis_count"], "hypothesis_count", 20)
    require_integer(
        oracle["multiple_testing_repetitions"],
        "multiple_testing_repetitions",
        40000,
    )
    require_integer(
        oracle["dependent_error_sample_size"], "dependent_error_sample_size", 100
    )
    require_integer(oracle["cluster_size"], "cluster_size", 5)
    require_integer(oracle["hac_bandwidth"], "hac_bandwidth", 1)
    require_integer(
        oracle["selection_candidate_count"], "selection_candidate_count", 2
    )
    require_integer(oracle["delta_method_sample_size"], "delta_method_sample_size", 250)
    require_integer(
        oracle["expected_bonferroni_rejections"],
        "expected_bonferroni_rejections",
        2,
    )
    require_integer(oracle["expected_holm_rejections"], "expected_holm_rejections", 2)
    require_integer(oracle["expected_bh_rejections"], "expected_bh_rejections", 4)
    if oracle["ordered_p_values"] != FIXED_ORDERED_P_VALUES:
        raise SystemExit("ordered p-values must match the published ledger")
    if oracle["dependent_score_fixture"] != [1.0, 1.0, -1.0, -1.0]:
        raise SystemExit("dependent score fixture must match the published ledger")
    if oracle["cluster_labels"] != [0, 0, 1, 1]:
        raise SystemExit("cluster labels must match the published ledger")
    if oracle["inference_labels"] != FIXED_INFERENCE_LABELS:
        raise SystemExit("inference labels must match the published chapter")
    if oracle["published_markers"] != FIXED_PUBLISHED_MARKERS:
        raise SystemExit("published markers must match the chapter evidence")
    if any(float(oracle[name]) != expected for name, expected in FIXED_SCALAR_DESIGN.items()):
        raise SystemExit("canonical inference design must not change")
    rng = np.random.default_rng(seed)

    theta = float(oracle["normal_mean_theta"])
    unbiased_variance = float(oracle["unbiased_estimator_variance"])
    shrinkage_factor = float(oracle["shrinkage_factor"])
    shrinkage_bias = (shrinkage_factor - 1.0) * theta
    shrinkage_variance = shrinkage_factor**2 * unbiased_variance
    shrinkage_mse = shrinkage_bias**2 + shrinkage_variance
    absolute_tolerance = float(oracle["absolute_tolerance"])
    if abs(shrinkage_bias - float(oracle["expected_shrinkage_bias"])) > absolute_tolerance:
        raise SystemExit("shrinkage bias oracle failed")
    if abs(shrinkage_variance - float(oracle["expected_shrinkage_variance"])) > absolute_tolerance:
        raise SystemExit("shrinkage variance oracle failed")
    if abs(shrinkage_mse - float(oracle["expected_shrinkage_mse"])) > absolute_tolerance:
        raise SystemExit("shrinkage MSE oracle failed")
    if not shrinkage_mse < unbiased_variance:
        raise SystemExit("shrinkage MSE must improve on the unbiased estimator")

    delta_probability = float(oracle["delta_method_probability"])
    delta_n = require_integer(
        oracle["delta_method_sample_size"], "delta_method_sample_size", 250
    )
    plugin_bias = delta_probability * (1.0 - delta_probability) / delta_n
    delta_variance = (
        (2.0 * delta_probability) ** 2
        * delta_probability
        * (1.0 - delta_probability)
        / delta_n
    )
    if (
        abs(plugin_bias - float(oracle["expected_plugin_bias"]))
        > absolute_tolerance
        or abs(delta_variance - float(oracle["expected_delta_variance"]))
        > absolute_tolerance
    ):
        raise SystemExit("plug-in and Delta-method ledger failed")

    inconsistent_bias = 0.0
    inconsistent_variance = 1.0
    if (
        abs(
            inconsistent_bias
            - float(oracle["expected_unbiased_inconsistent_bias"])
        )
        > absolute_tolerance
        or abs(
            inconsistent_variance
            - float(oracle["expected_unbiased_inconsistent_variance"])
        )
        > absolute_tolerance
    ):
        raise SystemExit("unbiased inconsistent estimator ledger failed")

    beta_x = float(oracle["ovb_beta_x"])
    beta_z = float(oracle["ovb_beta_z"])
    omitted_bias = (
        beta_z * float(oracle["ovb_cov_xz"]) / float(oracle["ovb_var_x"])
    )
    omitted_slope = beta_x + omitted_bias
    if (
        abs(omitted_slope - float(oracle["expected_omitted_slope"]))
        > absolute_tolerance
        or abs(omitted_bias - float(oracle["expected_omitted_bias"]))
        > absolute_tolerance
    ):
        raise SystemExit("omitted-variable bias ledger failed")

    dependent_n = int(oracle["dependent_error_sample_size"])
    rho = float(oracle["ar1_correlation"])
    iid_mean_variance = 1.0 / dependent_n
    ar1_mean_variance = (
        dependent_n
        + 2.0
        * sum((dependent_n - lag) * rho**lag for lag in range(1, dependent_n))
    ) / dependent_n**2
    cluster_size = int(oracle["cluster_size"])
    cluster_mean_variance = (
        1.0 + (cluster_size - 1.0) * float(oracle["cluster_correlation"])
    ) / dependent_n
    if (
        abs(iid_mean_variance - float(oracle["expected_iid_mean_variance"]))
        > absolute_tolerance
        or abs(ar1_mean_variance - float(oracle["expected_ar1_mean_variance"]))
        > absolute_tolerance
        or abs(
            cluster_mean_variance
            - float(oracle["expected_cluster_mean_variance"])
        )
        > absolute_tolerance
    ):
        raise SystemExit("dependent-error variance ledger failed")

    scores = np.asarray(oracle["dependent_score_fixture"], dtype=float)
    score_count = scores.size
    gamma_zero = float(scores @ scores) / score_count
    gamma_one = float(scores[1:] @ scores[:-1]) / score_count
    bartlett_weight = 1.0 - 1.0 / (int(oracle["hac_bandwidth"]) + 1.0)
    hac_long_run_variance = gamma_zero + 2.0 * bartlett_weight * gamma_one
    hac_standard_error = math.sqrt(hac_long_run_variance / score_count)
    cluster_sums = [
        float(scores[np.asarray(oracle["cluster_labels"]) == label].sum())
        for label in sorted(set(oracle["cluster_labels"]))
    ]
    cluster_standard_error = math.sqrt(
        sum(value**2 for value in cluster_sums) / score_count**2
    )
    if (
        abs(hac_standard_error - float(oracle["expected_hac_standard_error"]))
        > absolute_tolerance
    ):
        raise SystemExit("HAC standard error ledger failed")
    if (
        abs(
            cluster_standard_error
            - float(oracle["expected_cluster_standard_error"])
        )
        > absolute_tolerance
    ):
        raise SystemExit("cluster standard error ledger failed")

    ordered_p_values = [float(value) for value in oracle["ordered_p_values"]]
    ordered_alpha = float(oracle["nominal_alpha"])
    ordered_count = len(ordered_p_values)
    bonferroni_rejections = sum(
        value <= ordered_alpha / ordered_count for value in ordered_p_values
    )
    holm_rejections = 0
    for index, value in enumerate(ordered_p_values):
        if value <= ordered_alpha / (ordered_count - index):
            holm_rejections += 1
        else:
            break
    bh_rejections = max(
        (
            index
            for index, value in enumerate(ordered_p_values, start=1)
            if value <= index * ordered_alpha / ordered_count + 1e-15
        ),
        default=0,
    )
    optional_stopping_false_positive = 1.0 - (1.0 - ordered_alpha) ** 10
    selected_null_effect = 1.0 / math.sqrt(math.pi)
    if (
        bonferroni_rejections
        != int(oracle["expected_bonferroni_rejections"])
        or holm_rejections != int(oracle["expected_holm_rejections"])
        or bh_rejections != int(oracle["expected_bh_rejections"])
        or abs(
            optional_stopping_false_positive
            - float(oracle["expected_optional_stopping_false_positive"])
        )
        > absolute_tolerance
    ):
        raise SystemExit("ordered p-value ledger failed")
    if (
        abs(
            selected_null_effect - float(oracle["expected_selected_null_effect"])
        )
        > absolute_tolerance
    ):
        raise SystemExit("post-selection effect inflation ledger failed")

    p = float(oracle["bernoulli_p"])
    n = sample_size
    repetitions = estimator_repetitions
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
        f"bonferroni_fwer={bonferroni_fwer:.6f} "
        f"mse=({unbiased_variance:.6f},{shrinkage_mse:.6f}) "
        f"inconsistent_var={inconsistent_variance:.6f} "
        f"plugin=({plugin_bias:.6f},{delta_variance:.6f}) "
        f"ovb=({omitted_slope:.6f},{omitted_bias:.6f}) "
        f"dependent=({iid_mean_variance:.6f},{ar1_mean_variance:.6f},"
        f"{cluster_mean_variance:.6f}) "
        f"robust_se=({hac_standard_error:.6f},{cluster_standard_error:.6f}) "
        f"ordered=({bonferroni_rejections},{holm_rejections},{bh_rejections}) "
        f"optional={optional_stopping_false_positive:.6f} "
        f"selected={selected_null_effect:.6f}"
    )
    return 0


main(
    Path(sys.argv[1])
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-")
    else Path("evidence/ch10/oracle.json")
)
